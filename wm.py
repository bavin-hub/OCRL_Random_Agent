import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple


class CrossAttentionBlock(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.q_proj = nn.Linear(state_dim, embed_dim)
        self.k_proj = nn.Linear(action_dim, embed_dim)
        self.v_proj = nn.Linear(action_dim, embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, st: torch.Tensor, at: torch.Tensor) -> torch.Tensor:
        # st: (B, state_dim), at: (B, action_dim)
        q = self.q_proj(st).unsqueeze(1)  # (B, 1, E)
        k = self.k_proj(at).unsqueeze(1)
        v = self.v_proj(at).unsqueeze(1)
        attn_out, _ = self.attn(q, k, v, need_weights=False)
        x = (attn_out + q).squeeze(1)
        return self.norm(x)


class FeedForwardBlock(nn.Module):
    def __init__(self, embed_dim: int, mlp_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.net(x))


class TransWM(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        embed_dim: int,
        mlp_dim: int,
        num_heads: int = 4,
        num_layers: int = 1,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        device: str = "cpu",
        std_range: Tuple[float, float] = (0.01, 0.06),
        dropout: float = 0.0,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        assert num_layers >= 1

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.lr = lr
        self.state_dim = state_dim

        self.attn_blocks = nn.ModuleList()
        self.ffn_blocks = nn.ModuleList()
        for i in range(num_layers):
            q_in = state_dim if i == 0 else embed_dim
            self.attn_blocks.append(
                CrossAttentionBlock(q_in, action_dim, embed_dim, num_heads, dropout)
            )
            self.ffn_blocks.append(FeedForwardBlock(embed_dim, mlp_dim, dropout))

        self.mean_mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, state_dim),
        )
        self.logstd_mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, state_dim),
        )
        self.contact_mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, 30),
        )

        self.logstd_range = (float(np.log(std_range[0])), float(np.log(std_range[1])))

        self.optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        self.state_discrepancy = nn.GaussianNLLLoss()

        self.to(self.device)

    def encode(self, st: torch.Tensor, at: torch.Tensor) -> torch.Tensor:
        h = st
        for attn, ffn in zip(self.attn_blocks, self.ffn_blocks):
            h = attn(h, at)
            h = ffn(h)
        return h

    def forward(
        self,
        st: torch.Tensor,
        at: torch.Tensor,
        predict: bool = False,
        sample: bool = False,
        x_prev: torch.Tensor = None,
    ):
        if not predict:
            return self.encode(st, at)

        h = self.encode(st, at)
        mu_t_next = self.mean_mlp(h)
        contact_pred = self.contact_mlp(h)
        if x_prev is not None:
            mu_t_next = mu_t_next + x_prev

        logstd_t = self.logstd_mlp(h).clamp(*self.logstd_range)
        std = logstd_t.exp()

        if sample:
            return mu_t_next, contact_pred, std

        eps = torch.randn_like(mu_t_next, device=st.device)
        st_next = mu_t_next + eps * std
        return st_next, contact_pred, std

    def gnll_loss(self, state_mean: torch.Tensor, state_std: torch.Tensor, state_target: torch.Tensor):
        return self.state_discrepancy(state_mean, state_target, state_std**2)

    def nll_loss(self, dist, target):
        return -dist.log_prob(target).sum(dim=-1).mean()

    def mse_loss(self, st_pred: torch.Tensor, st_true: torch.Tensor):
        return torch.sum(torch.square(st_pred - st_true), dim=1).mean(dim=0)


class MlpWM(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        contact_dim: int = 30,
        hidden_dim: int = 256,
        num_layers: int = 2,
        mlp_head_dim: int = 128,
        with_uncertainty: bool = False,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        device: str = "cpu",
        std_range: Tuple[float, float] = (0.03, 0.5),
    ):
        super().__init__()
        assert num_layers >= 1

        d = str(device)
        if d == "cuda" and not torch.cuda.is_available():
            d = "cpu"
        elif d == "mps" and not (
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        ):
            d = "cpu"
        self.device = torch.device(d)
        print(f"Device set to {self.device}")
        self.state_dim = state_dim
        self.with_uncertainty = with_uncertainty

        layers = [nn.Linear(state_dim + action_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        self.backbone = nn.Sequential(*layers)

        self.mean_head = nn.Sequential(
            nn.Linear(hidden_dim, mlp_head_dim),
            nn.ReLU(),
            nn.Linear(mlp_head_dim, state_dim),
        )
        self.contact_head = nn.Sequential(
            nn.Linear(hidden_dim, mlp_head_dim),
            nn.ReLU(),
            nn.Linear(mlp_head_dim, contact_dim),
        )

        if with_uncertainty:
            self.logstd_head = nn.Sequential(
                nn.Linear(hidden_dim, mlp_head_dim),
                nn.ReLU(),
                nn.Linear(mlp_head_dim, state_dim),
            )
            self.logstd_range = (float(np.log(std_range[0])), float(np.log(std_range[1])))
            self.gnll = nn.GaussianNLLLoss()

        self.optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        self.to(self.device)

    def forward(
        self,
        s_t: torch.Tensor,
        a_t: torch.Tensor,
        x_prev: torch.Tensor = None,
    ):
        h = self.backbone(torch.cat([s_t, a_t], dim=-1))
        mu = self.mean_head(h)
        contact = self.contact_head(h)
        if x_prev is not None:
            mu = mu + x_prev

        if self.with_uncertainty:
            logstd = self.logstd_head(h).clamp(*self.logstd_range)
            std = logstd.exp()
            return mu, contact, std
        return mu, contact

    def mse_loss(self, st_pred: torch.Tensor, st_true: torch.Tensor):
        return torch.sum((st_pred - st_true) ** 2, dim=-1).mean()

    def gnll_loss(self, state_mean: torch.Tensor, state_std: torch.Tensor, state_target: torch.Tensor):
        return self.gnll(state_mean, state_target, state_std ** 2)


if __name__ == "__main__":
    B, S, A = 8, 12, 4
    m = TransWM(
        state_dim=S,
        action_dim=A,
        embed_dim=64,
        mlp_dim=128,
        num_heads=4,
        num_layers=1,
        device="cpu",
    )
    st, at = torch.randn(B, S), torch.randn(B, A)
    out, contact, std = m(st, at, predict=True, sample=False, x_prev=st)
    assert out.shape == (B, S) and std.shape == (B, S)
    print("TransWM smoke OK:", out.shape, std.shape)

    mlp = MlpWM(state_dim=S, action_dim=A, hidden_dim=64, num_layers=2, mlp_head_dim=32, device="cpu")
    mu, contact = mlp(st, at, x_prev=st)
    assert mu.shape == (B, S)
    print("MlpWM smoke OK:", mu.shape, "params:", sum(p.numel() for p in mlp.parameters()))
