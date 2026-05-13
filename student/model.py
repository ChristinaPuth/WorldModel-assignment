# """Student world model - GRU with residual prediction."""

# from __future__ import annotations

# import torch
# from torch import nn


# class StudentWorldModel(nn.Module):
#     def __init__(
#         self,
#         obs_dim: int = 4,
#         act_dim: int = 1,
#         hidden_dim: int = 256,
#         num_layers: int = 3,
#         use_gru: bool = True,
#         delta_limit: float = 3.0,
#     ):
#         super().__init__()
#         self.obs_dim = obs_dim
#         self.hidden_dim = hidden_dim
#         self.use_gru = True
#         self.delta_limit = float(delta_limit)
#         self.num_layers = int(num_layers)

#         self.input_encoder = nn.Sequential(
#             nn.Linear(obs_dim + act_dim, hidden_dim),
#             nn.LayerNorm(hidden_dim),
#             nn.SiLU(),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.SiLU(),
#         )

#         self.gru = nn.GRU(
#             input_size=hidden_dim,
#             hidden_size=hidden_dim,
#             num_layers=self.num_layers,
#             batch_first=True,
#             dropout=0.05,
#         )

#         self.head = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim // 2),
#             nn.SiLU(),
#             nn.Linear(hidden_dim // 2, obs_dim),
#         )

#     def initial_hidden(self, batch_size: int, device: torch.device):
#         return torch.zeros(
#             self.num_layers, batch_size, self.hidden_dim, device=device
#         )

#     def forward(self, obs_norm: torch.Tensor, act_norm: torch.Tensor, hidden=None):
#         B = obs_norm.shape[0]
#         if hidden is None:
#             hidden = self.initial_hidden(B, obs_norm.device)

#         feat = self.input_encoder(torch.cat([obs_norm, act_norm], dim=-1))
#         feat = feat.unsqueeze(1)
#         out, hidden = self.gru(feat, hidden)
#         out = out.squeeze(1)

#         raw_delta = self.head(out)
#         delta = self.delta_limit * torch.tanh(raw_delta / self.delta_limit)

#         return delta, hidden

"""Student world model.

Plain residual-delta MLP world model.

This is closer to the version that gave the best dev result so far.
It avoids GRU hidden-state mismatch and avoids unstable local-linear initialization.
"""

from __future__ import annotations

import torch
from torch import nn


class StudentWorldModel(nn.Module):
    def __init__(
        self,
        obs_dim: int = 4,
        act_dim: int = 1,
        hidden_dim: int = 256,
        num_layers: int = 4,
        use_gru: bool = False,
        delta_limit: float = 5.0,
    ):
        super().__init__()

        self.use_gru = False
        self.delta_limit = float(delta_limit)

        in_dim = obs_dim + act_dim
        layers: list[nn.Module] = []

        for _ in range(int(num_layers)):
            layers += [
                nn.Linear(in_dim, hidden_dim),
                nn.SiLU(),
            ]
            in_dim = hidden_dim

        self.encoder = nn.Sequential(*layers)
        self.head = nn.Linear(hidden_dim, obs_dim)

    def initial_hidden(self, batch_size: int, device: torch.device):
        return None

    def forward(self, obs_norm: torch.Tensor, act_norm: torch.Tensor, hidden=None):
        x = torch.cat([obs_norm, act_norm], dim=-1)

        feat = self.encoder(x)
        raw_delta = self.head(feat)

        # Conservative but not too restrictive.
        delta = self.delta_limit * torch.tanh(raw_delta / self.delta_limit)

        return delta, None