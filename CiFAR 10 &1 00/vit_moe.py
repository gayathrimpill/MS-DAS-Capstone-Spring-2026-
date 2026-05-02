import torch
import torch.nn as nn
import torch.nn.functional as F


class MoEFeedForward(nn.Module):

    def __init__(
        self,
        hidden_size,
        intermediate_size,
        num_experts=4,
        top_k=2
    ):
        super().__init__()

        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)

        # Experts
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, intermediate_size),
                nn.GELU(),
                nn.Linear(intermediate_size, hidden_size)
            )
            for _ in range(num_experts)
        ])

        # Router
        self.router = nn.Linear(hidden_size, num_experts)

        # load balancing loss value
        self.load_balance_loss = 0


    def forward(self, x):

        batch, tokens, hidden = x.shape

        router_logits = self.router(x)
        router_probs = F.softmax(router_logits, dim=-1)

        # compute load balancing stats
        expert_usage = router_probs.mean(dim=(0,1))

        self.load_balance_loss = (self.num_experts * torch.sum(expert_usage ** 2) / self.num_experts)

        topk_vals, topk_idx = torch.topk(
            router_probs,
            self.top_k,
            dim=-1
        )

        output = torch.zeros_like(x)

        for k in range(self.top_k):

            expert_idx = topk_idx[..., k]
            weight = topk_vals[..., k].unsqueeze(-1)

            for e, expert in enumerate(self.experts):

                mask = (expert_idx == e).unsqueeze(-1)

                if mask.any():
                    expert_out = expert(x)
                    output += weight * mask * expert_out

        return output