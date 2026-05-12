# """Student world model.

# Students may replace this residual MLP with a GRU or another dynamics model,
# but the public interface must stay the same.
# """

# from __future__ import annotations

# import torch
# from torch import nn


# class StudentWorldModel(nn.Module):
#     def __init__(
#         self,
#         obs_dim: int = 4,
#         act_dim: int = 1,
#         hidden_dim: int = 128,
#         num_layers: int = 2,
#         use_gru: bool = False,
#         delta_limit: float = 3.0,
#     ):
#         super().__init__()
#         self.use_gru = bool(use_gru)
#         self.delta_limit = float(delta_limit)
#         in_dim = obs_dim + act_dim
#         layers: list[nn.Module] = []
#         for _ in range(int(num_layers)):
#             layers += [nn.Linear(in_dim, hidden_dim), nn.SiLU()]
#             in_dim = hidden_dim
#         self.encoder = nn.Sequential(*layers)
#         self.gru = nn.GRUCell(hidden_dim, hidden_dim) if self.use_gru else None
#         self.head = nn.Linear(hidden_dim, obs_dim)

#     def initial_hidden(self, batch_size: int, device: torch.device):
#         if not self.use_gru:
#             return None
#         return torch.zeros(batch_size, self.gru.hidden_size, device=device)

#     def forward(self, obs_norm: torch.Tensor, act_norm: torch.Tensor, hidden=None):
#         feat = self.encoder(torch.cat([obs_norm, act_norm], dim=-1))
#         if self.gru is not None:
#             if hidden is None:
#                 hidden = self.initial_hidden(obs_norm.shape[0], obs_norm.device)
#             hidden = self.gru(feat, hidden)
#             feat = hidden
#         raw_delta = self.head(feat)
#         delta = self.delta_limit * torch.tanh(raw_delta / self.delta_limit)
#         return delta, hidden


"""Student world model - GRU with residual prediction."""

from __future__ import annotations

import torch
from torch import nn


class StudentWorldModel(nn.Module):
    def __init__(
        self,
        obs_dim: int = 4,
        act_dim: int = 1,
        hidden_dim: int = 256,
        num_layers: int = 3,
        use_gru: bool = True,
        delta_limit: float = 3.0,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.use_gru = True  # 强制使用 GRU
        self.delta_limit = float(delta_limit)
        self.num_layers = int(num_layers)

        # 输入编码器：把 obs+act 映射到 hidden_dim
        self.input_encoder = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

        # 多层 GRU
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
        )

        # 输出头：预测 delta
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, obs_dim),
        )

    def initial_hidden(self, batch_size: int, device: torch.device):
        return torch.zeros(
            self.num_layers, batch_size, self.hidden_dim, device=device
        )

    def forward(self, obs_norm: torch.Tensor, act_norm: torch.Tensor, hidden=None):
        """
        obs_norm: [B, obs_dim]
        act_norm: [B, act_dim]
        hidden:   [num_layers, B, hidden_dim] or None
        """
        B = obs_norm.shape[0]
        if hidden is None:
            hidden = self.initial_hidden(B, obs_norm.device)

        # 编码输入
        feat = self.input_encoder(torch.cat([obs_norm, act_norm], dim=-1))

        # GRU 需要 [B, 1, hidden_dim] 的序列输入
        feat = feat.unsqueeze(1)
        out, hidden = self.gru(feat, hidden)
        out = out.squeeze(1)  # [B, hidden_dim]

        # 预测 delta 并 tanh 限制范围
        raw_delta = self.head(out)
        delta = self.delta_limit * torch.tanh(raw_delta / self.delta_limit)

        return delta, hidden