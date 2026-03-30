import torch
import torch.nn.functional as F
from torch import nn
import torch.optim as optim


class VisionRssmWorldModel(nn.Module):
    """One-step RGB-D world model with stochastic latent dynamics."""

    def __init__(
        self,
        action_dim: int,
        image_height: int,
        image_width: int,
        cnn_embed_dim: int,
        action_embed_dim: int,
        latent_dim: int,
        hidden_dim: int,
        lr: float,
        weight_decay: float,
        device: str,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.image_height = image_height
        self.image_width = image_width
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        # Shared RGB-D encoder.
        self.encoder = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        # flattens the 4x4x256 to a single embedding vector of size cnn_embed_dim
        self.encoder_proj = nn.Linear(256 * 4 * 4, cnn_embed_dim)

        self.action_proj = nn.Sequential(
            nn.Linear(action_dim, action_embed_dim),
            nn.SiLU(),
        )
        #frame, action, hidden_state
        prior_in = cnn_embed_dim + action_embed_dim + hidden_dim
        #frame, action, next_frame, hidden_state
        post_in = cnn_embed_dim + action_embed_dim + cnn_embed_dim + hidden_dim

        #Ga
        self.prior_mu = nn.Linear(prior_in, latent_dim)
        self.prior_logvar = nn.Linear(prior_in, latent_dim)
        self.post_mu = nn.Linear(post_in, latent_dim)
        self.post_logvar = nn.Linear(post_in, latent_dim)

        self.gru = nn.GRUCell(input_size=latent_dim + action_embed_dim, hidden_size=hidden_dim)

        dec_in = cnn_embed_dim + latent_dim + hidden_dim
        self.decoder_fc = nn.Sequential(
            nn.Linear(dec_in, 256 * 8 * 8),
            nn.SiLU(),
        )
        self.decoder_body = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        self.rgb_head = nn.Conv2d(32, 3, kernel_size=3, padding=1)
        self.depth_head = nn.Conv2d(32, 1, kernel_size=3, padding=1)

        self.optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        self.to(device if torch.cuda.is_available() else "cpu")

    def _encode_obs(self, rgb: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        x = torch.cat([rgb, depth], dim=1)
        h = self.encoder(x).flatten(start_dim=1)
        return self.encoder_proj(h)

    @staticmethod
    def _reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    @staticmethod
    def _kl_gaussian(mu_q, logvar_q, mu_p, logvar_p):
        var_q = torch.exp(logvar_q)
        var_p = torch.exp(logvar_p)
        kl = 0.5 * (
            logvar_p
            - logvar_q
            + (var_q + (mu_q - mu_p).pow(2)) / torch.clamp(var_p, min=1e-6)
            - 1.0
        )
        return kl.sum(dim=-1).mean()

    def forward(
        self,
        rgb_t: torch.Tensor,
        depth_t: torch.Tensor,
        action_t: torch.Tensor,
        rgb_t1: torch.Tensor | None = None,
        depth_t1: torch.Tensor | None = None,
        hidden_state: torch.Tensor | None = None,
    ):
        bsz = rgb_t.shape[0]
        if hidden_state is None:
            hidden_state = torch.zeros(bsz, self.hidden_dim, device=rgb_t.device)

        enc_t = self._encode_obs(rgb_t, depth_t)
        action_emb = self.action_proj(action_t)

        prior_in = torch.cat([enc_t, action_emb, hidden_state], dim=-1)
        prior_mu = self.prior_mu(prior_in)
        prior_logvar = self.prior_logvar(prior_in).clamp(-8.0, 6.0)

        post_mu, post_logvar = prior_mu, prior_logvar
        use_posterior = rgb_t1 is not None and depth_t1 is not None
        if use_posterior:
            enc_t1 = self._encode_obs(rgb_t1, depth_t1)
            post_in = torch.cat([enc_t, action_emb, enc_t1, hidden_state], dim=-1)
            post_mu = self.post_mu(post_in)
            post_logvar = self.post_logvar(post_in).clamp(-8.0, 6.0)

        z_t = self._reparameterize(post_mu, post_logvar) if use_posterior else self._reparameterize(prior_mu, prior_logvar)
        hidden_next = self.gru(torch.cat([z_t, action_emb], dim=-1), hidden_state)

        dec_in = torch.cat([enc_t, z_t, hidden_next], dim=-1)
        dec_seed = self.decoder_fc(dec_in).reshape(bsz, 256, 8, 8)
        dec_feat = self.decoder_body(dec_seed)

        pred_rgb_t1 = torch.sigmoid(self.rgb_head(dec_feat))
        pred_depth_t1 = self.depth_head(dec_feat)

        if pred_rgb_t1.shape[-2:] != (self.image_height, self.image_width):
            pred_rgb_t1 = F.interpolate(
                pred_rgb_t1, size=(self.image_height, self.image_width), mode="bilinear", align_corners=False
            )
            pred_depth_t1 = F.interpolate(
                pred_depth_t1, size=(self.image_height, self.image_width), mode="bilinear", align_corners=False
            )

        return {
            "pred_rgb_t1": pred_rgb_t1,
            "pred_depth_t1": pred_depth_t1,
            "prior_mu": prior_mu,
            "prior_logvar": prior_logvar,
            "post_mu": post_mu,
            "post_logvar": post_logvar,
            "hidden_next": hidden_next,
            "used_posterior": use_posterior,
        }

    def compute_loss(
        self,
        outputs: dict,
        rgb_t1: torch.Tensor,
        depth_t1: torch.Tensor,
        kl_weight: float = 1.0,
        rgb_weight: float = 1.0,
        depth_weight: float = 1.0,
    ):
        rgb_loss = F.l1_loss(outputs["pred_rgb_t1"], rgb_t1)
        depth_loss = F.l1_loss(outputs["pred_depth_t1"], depth_t1)
        kl_loss = self._kl_gaussian(
            outputs["post_mu"],
            outputs["post_logvar"],
            outputs["prior_mu"],
            outputs["prior_logvar"],
        )
        total = rgb_weight * rgb_loss + depth_weight * depth_loss + kl_weight * kl_loss
        return {
            "total_loss": total,
            "rgb_loss": rgb_loss.detach(),
            "depth_loss": depth_loss.detach(),
            "kl_loss": kl_loss.detach(),
        }
