"""
vit_moe_vmoe.py
---------------
V-MoE-inspired sparse MoE feed-forward block for ViT.

Key changes vs. the original cap_moe / vit_moe implementation:

1. Buffer-token routing  (V-MoE §3.2)
   Each expert has a hard capacity C = (tokens_per_batch / num_experts) * capacity_factor.
   A small number of *buffer tokens* (learned, prepended) absorb overflow so that
   real image tokens are never dropped.  Buffer token outputs are discarded.

2. Straight-through top-k gating with noise (V-MoE §3.1 / Switch Transformer)
   During training we add unit-normal noise to the router logits before top-k
   selection, which encourages exploration and load balance.

3. Auxiliary load-balance loss (V-MoE / Switch eq. 4)
   L_aux = num_experts * sum_i( f_i * p_i )
   where f_i = fraction of tokens dispatched to expert i,
         p_i = mean router probability for expert i.
   Coefficient recommended in V-MoE: 0.01 (vs 0.1 in the original cap_moe).

4. Overflow tracking
   self.overflow_fraction is updated each forward pass so the training loop
   can log it to wandb.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VMoEFeedForward(nn.Module):
    """
    Sparse MoE feed-forward layer following V-MoE design.

    Args:
        hidden_size:       Model dimension (e.g. 768 for ViT-B).
        intermediate_size: Inner FF dimension (e.g. 3072 for ViT-B).
        num_experts:       Total number of experts (paper uses 8 or 16).
        top_k:             Experts activated per token (paper uses 1 or 2).
        capacity_factor:   Multiplier on the "fair share" capacity per expert.
                           1.0 = no slack, 1.25 gives 25 % headroom.
        num_buffer_tokens: Extra learnable tokens prepended to each batch to
                           absorb routing overflow (paper §3.2).  Set to 0 to
                           disable buffer tokens and use hard dropping instead.
        noise_std:         Std of Gaussian noise added to router logits during
                           training (0.0 to disable).
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int = 8,
        top_k: int = 1,
        capacity_factor: float = 1.25,
        num_buffer_tokens: int = 8,
        noise_std: float = 1.0,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.num_buffer_tokens = num_buffer_tokens
        self.noise_std = noise_std

        # Per-expert feed-forward networks
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, intermediate_size, bias=True),
                nn.GELU(),
                nn.Linear(intermediate_size, hidden_size, bias=True),
            )
            for _ in range(num_experts)
        ])

        # Router
        self.router = nn.Linear(hidden_size, num_experts, bias=False)

        # Buffer tokens: shape (num_buffer_tokens, hidden_size)
        if num_buffer_tokens > 0:
            self.buffer_tokens = nn.Parameter(
                torch.zeros(num_buffer_tokens, hidden_size)
            )
            nn.init.normal_(self.buffer_tokens, std=0.02)
        else:
            self.buffer_tokens = None

        # Diagnostics (updated every forward, not a parameter)
        self.load_balance_loss = torch.tensor(0.0)
        self.overflow_fraction = 0.0

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, hidden_size)
        returns: same shape
        """
        B, S, H = x.shape

        # ---- 1. Prepend buffer tokens ----
        if self.buffer_tokens is not None and self.num_buffer_tokens > 0:
            buf = self.buffer_tokens.unsqueeze(0).expand(B, -1, -1)  # (B, nb, H)
            x_aug = torch.cat([buf, x], dim=1)                        # (B, nb+S, H)
        else:
            x_aug = x

        B2, S2, H2 = x_aug.shape  # S2 = S + num_buffer_tokens

        # ---- 2. Router logits + noise ----
        logits = self.router(x_aug)                           # (B, S2, E)
        if self.training and self.noise_std > 0:
            logits = logits + torch.randn_like(logits) * self.noise_std

        # ---- 3. Top-k gating ----
        topk_vals, topk_idx = torch.topk(logits, self.top_k, dim=-1)  # (B, S2, k)
        # Soft weights via softmax over selected experts only
        gate_weights = F.softmax(topk_vals, dim=-1)                    # (B, S2, k)

        # ---- 4. Load-balance auxiliary loss (V-MoE / Switch) ----
        router_probs = F.softmax(logits, dim=-1)                       # (B, S2, E)
        # f_i: fraction of tokens routed to expert i (via argmax / one-hot)
        one_hot = F.one_hot(topk_idx[..., 0], self.num_experts).float()  # (B, S2, E)
        f_i = one_hot.mean(dim=[0, 1])                                   # (E,)
        p_i = router_probs.mean(dim=[0, 1])                              # (E,)
        aux = self.num_experts * (f_i * p_i).sum()
        self.load_balance_loss = aux

        # ---- 5. Dispatch tokens to experts with capacity limit ----
        capacity = max(
            1,
            int(self.capacity_factor * self.top_k * S2 * B / self.num_experts)
        )

        output = torch.zeros_like(x_aug)   # accumulate expert outputs here
        total_tokens = B * S2 * self.top_k
        dropped = 0

        for expert_idx in range(self.num_experts):
            # Find all (batch, seq, k) positions that chose this expert
            mask = (topk_idx == expert_idx)             # (B, S2, k)  bool
            b_idx, s_idx, k_idx = mask.nonzero(as_tuple=True)

            if b_idx.numel() == 0:
                continue

            # Apply capacity limit
            if b_idx.numel() > capacity:
                dropped += b_idx.numel() - capacity
                b_idx = b_idx[:capacity]
                s_idx = s_idx[:capacity]
                k_idx = k_idx[:capacity]

            tokens = x_aug[b_idx, s_idx]                # (n, H)
            weights = gate_weights[b_idx, s_idx, k_idx].unsqueeze(-1)  # (n, 1)

            expert_out = self.experts[expert_idx](tokens) * weights    # (n, H)

            # Scatter-add back
            output.index_put_((b_idx, s_idx), expert_out, accumulate=True)

        self.overflow_fraction = dropped / max(total_tokens, 1)

        # ---- 6. Strip buffer tokens, return only real token outputs ----
        if self.buffer_tokens is not None and self.num_buffer_tokens > 0:
            output = output[:, self.num_buffer_tokens:, :]   # (B, S, H)

        return output
