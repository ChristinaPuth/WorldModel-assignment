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
#############最近的best
# """Student world model - GRU with residual prediction."""

# from __future__ import annotations

# import torch
# from torch import nn


# class StudentWorldModel(nn.Module):
#     def __init__(
#         self,
#         obs_dim: int = 4,
#         act_dim: int = 1,
#         hidden_dim: int = 512,
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
#             nn.LayerNorm(hidden_dim),
#             nn.SiLU(),
#         )

#         self.gru = nn.GRU(
#             input_size=hidden_dim,
#             hidden_size=hidden_dim,
#             num_layers=self.num_layers,
#             batch_first=True,
#             dropout=0.15,
#         )

#         self.gru_norm = nn.LayerNorm(hidden_dim)

#         self.head_fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
#         self.head_norm = nn.LayerNorm(hidden_dim // 2)
#         self.head_fc2 = nn.Linear(hidden_dim // 2, obs_dim)
#         self.skip_proj = nn.Linear(hidden_dim, obs_dim)

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
#         out = self.gru_norm(out.squeeze(1))

#         h = torch.nn.functional.silu(self.head_norm(self.head_fc1(out)))
#         raw_delta = self.head_fc2(h) + self.skip_proj(out)
#         delta = self.delta_limit * torch.tanh(raw_delta / self.delta_limit)

#         return delta, hidden




"""Student world model - Transformer with residual prediction."""

from __future__ import annotations

import torch
from torch import nn


MAX_HISTORY = 64  # 最多保留64步历史，防止内存爆炸


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
        self.delta_limit = float(delta_limit)
        self.num_layers = int(num_layers)

        # 输入编码
        self.input_proj = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=hidden_dim * 2,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.num_layers,
        )

        # 输出头
        self.head_fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.head_norm = nn.LayerNorm(hidden_dim // 2)
        self.head_fc2 = nn.Linear(hidden_dim // 2, obs_dim)
        self.skip_proj = nn.Linear(hidden_dim, obs_dim)

    def initial_hidden(self, batch_size: int, device: torch.device):
        # Transformer用list存历史，初始为空
        return None

    def forward(self, obs_norm: torch.Tensor, act_norm: torch.Tensor, hidden=None):
        # 编码当前输入
        feat = self.input_proj(
            torch.cat([obs_norm, act_norm], dim=-1)
        )  # [B, D]

        # 更新历史序列
        if hidden is None:
            seq = feat.unsqueeze(1)  # [B, 1, D]
        else:
            # hidden是之前的序列 [B, T, D]
            seq = torch.cat([hidden, feat.unsqueeze(1)], dim=1)  # [B, T+1, D]

        # 限制最大历史长度防止内存爆
        if seq.shape[1] > MAX_HISTORY:
            seq = seq[:, -MAX_HISTORY:, :]

        # Transformer处理序列，取最后一步
        out = self.transformer(seq)  # [B, T, D]
        last = out[:, -1, :]  # [B, D]

        # 残差预测head
        h = torch.nn.functional.silu(
            self.head_norm(self.head_fc1(last))
        )
        raw_delta = self.head_fc2(h) + self.skip_proj(last)
        delta = self.delta_limit * torch.tanh(raw_delta / self.delta_limit)

        # 返回新的hidden（保存整个序列供下次使用）
        # 用detach防止梯度通过历史序列反传导致内存爆炸
        new_hidden = seq.detach()

        return delta, new_hidden