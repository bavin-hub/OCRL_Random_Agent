import torch
import torch.nn.functional as F
from torch import nn
import torch.optim as optim

try:
    import lpips as _lpips_module
    _LPIPS_AVAILABLE = True
except ImportError:
    _LPIPS_AVAILABLE = False


class ResBlock(nn.Module):
    """Residual block: two 3x3 convs with GroupNorm and a skip connection."""

    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(min(32, channels), channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(min(32, channels), channels),
        )
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class VisionRssmWorldModel(nn.Module):
    """Vision world model: RGB-D(t) + action(t) -> predicted RGB-D(t+1).

    RSSM with discrete categorical latents (DreamerV3-style), 6-stage
    ResNet encoder/decoder for 256x256, U-Net skip connections, GRU
    dynamics, and LPIPS perceptual loss.
    """

    _ENC_CHANNELS = (4, 64, 96, 128, 192, 256, 384)

    def __init__(
        self,
        action_dim: int,
        image_height: int,
        image_width: int,
        cnn_embed_dim: int,
        action_embed_dim: int,
        num_categories: int,
        num_classes: int,
        hidden_dim: int,
        lr: float,
        weight_decay: float,
        device: str,
        prop_dim: int = 0,
        prop_embed_dim: int = 64,
    ):
        super().__init__()
        self.image_height   = image_height
        self.image_width    = image_width
        self.num_categories = num_categories
        self.num_classes    = num_classes
        self.latent_size    = num_categories * num_classes
        self.hidden_dim     = hidden_dim
        self.prop_dim       = prop_dim

        ch = self._ENC_CHANNELS

        # Encoder: 6 conv+resblock stages, each halving spatial dims (256 -> 4)
        self.enc_stages = nn.ModuleList()
        for i in range(6):
            self.enc_stages.append(nn.Sequential(
                nn.Conv2d(ch[i], ch[i + 1], kernel_size=4, stride=2, padding=1, bias=False),
                nn.GroupNorm(min(32, ch[i + 1]), ch[i + 1]),
                nn.SiLU(),
                ResBlock(ch[i + 1]),
            ))

        bottleneck_spatial = image_height // 64
        bottleneck_flat = ch[6] * bottleneck_spatial * bottleneck_spatial

        self.encoder_proj = nn.Sequential(
            nn.Linear(bottleneck_flat, cnn_embed_dim),
            nn.LayerNorm(cnn_embed_dim),
            nn.SiLU(),
        )

        self.action_proj = nn.Sequential(
            nn.Linear(action_dim, action_embed_dim),
            nn.LayerNorm(action_embed_dim),
            nn.SiLU(),
        )

        _prop_emb_dim = 0
        if prop_dim > 0:
            self.prop_proj = nn.Sequential(
                nn.Linear(prop_dim, prop_embed_dim),
                nn.LayerNorm(prop_embed_dim),
                nn.SiLU(),
            )
            _prop_emb_dim = prop_embed_dim
        else:
            self.prop_proj = None

        # Prior predicts z from action + prop + hidden only (no future frame).
        # Posterior also sees enc(frame_t) and enc(frame_t+1).
        prior_in = action_embed_dim + _prop_emb_dim + hidden_dim
        post_in  = cnn_embed_dim + action_embed_dim + _prop_emb_dim + cnn_embed_dim + hidden_dim

        self.prior_head = nn.Sequential(
            nn.Linear(prior_in, prior_in * 2),
            nn.LayerNorm(prior_in * 2),
            nn.SiLU(),
            nn.Linear(prior_in * 2, self.latent_size),
        )
        self.post_head = nn.Sequential(
            nn.Linear(post_in, post_in),
            nn.LayerNorm(post_in),
            nn.SiLU(),
            nn.Linear(post_in, self.latent_size),
        )

        self.gru = nn.GRUCell(input_size=self.latent_size + action_embed_dim, hidden_size=hidden_dim)

        # Decoder: mirrors encoder with upsampling. Stages 1-5 concat skip
        # features from the encoder of frame_t (U-Net), stage 6 has no skip.
        dec_in = self.latent_size + hidden_dim
        self.decoder_fc = nn.Sequential(
            nn.Linear(dec_in, dec_in),
            nn.LayerNorm(dec_in),
            nn.SiLU(),
            nn.Linear(dec_in, bottleneck_flat),
            nn.SiLU(),
        )

        self.dec_ups = nn.ModuleList()
        self.dec_convs = nn.ModuleList()
        dec_ch_in   = [ch[6], ch[5], ch[4], ch[3], ch[2], ch[1]]
        dec_skip_ch = [ch[5], ch[4], ch[3], ch[2], ch[1], 0]
        dec_ch_out  = [ch[5], ch[4], ch[3], ch[2], ch[1], 32]

        for i in range(6):
            self.dec_ups.append(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False))
            in_c = dec_ch_in[i] + dec_skip_ch[i]
            out_c = dec_ch_out[i]
            if dec_skip_ch[i] > 0:
                self.dec_convs.append(nn.Sequential(
                    nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
                    nn.GroupNorm(min(32, out_c), out_c),
                    nn.SiLU(),
                    ResBlock(out_c),
                ))
            else:
                self.dec_convs.append(nn.Sequential(
                    nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
                    nn.GroupNorm(min(32, out_c), out_c),
                    nn.SiLU(),
                ))

        self.rgb_head   = nn.Conv2d(32, 3, kernel_size=3, padding=1)
        self.depth_head = nn.Conv2d(32, 1, kernel_size=3, padding=1)

        # Frozen VGG net for perceptual loss
        self._lpips_net = None
        if _LPIPS_AVAILABLE:
            self._lpips_net = _lpips_module.LPIPS(net='vgg')
            self._lpips_net.eval()
            for p in self._lpips_net.parameters():
                p.requires_grad = False

        self.optimizer = optim.Adam(
            (p for p in self.parameters() if p.requires_grad),
            lr=lr, weight_decay=weight_decay,
        )
        self.to(device if torch.cuda.is_available() else "cpu")

    def _encode_obs(self, rgb: torch.Tensor, depth: torch.Tensor):
        """Run RGB-D through encoder. Returns (embedding, skip_features)."""
        x = torch.cat([rgb, depth], dim=1)
        skips = []
        for i, stage in enumerate(self.enc_stages):
            x = stage(x)
            if i < 5:
                skips.append(x)
        emb = self.encoder_proj(x.flatten(start_dim=1))
        return emb, tuple(skips)

    def _decode(self, z: torch.Tensor, h: torch.Tensor, skips: tuple):
        """Reconstruct RGB-D from latent z + hidden h, using encoder skips."""
        bsz = z.shape[0]
        bottleneck_spatial = self.image_height // 64
        x = self.decoder_fc(torch.cat([z, h], dim=-1))
        x = x.reshape(bsz, self._ENC_CHANNELS[6], bottleneck_spatial, bottleneck_spatial)

        rev_skips = list(reversed(skips)) + [None]
        for i in range(6):
            x = self.dec_ups[i](x)
            if rev_skips[i] is not None:
                x = torch.cat([x, rev_skips[i]], dim=1)
            x = self.dec_convs[i](x)

        return torch.sigmoid(self.rgb_head(x)), self.depth_head(x)

    def _straight_through(self, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Discrete latent sampling with straight-through gradients."""
        B = logits.shape[0]
        logits = logits.reshape(B, self.num_categories, self.num_classes)
        probs  = F.softmax(logits, dim=-1)
        one_hot = F.one_hot(probs.argmax(dim=-1), self.num_classes).float()
        z = (one_hot - probs.detach() + probs).reshape(B, -1)
        return z, probs

    @staticmethod
    def _kl_categorical(probs_q, probs_p, free_bits: float):
        kl_per_cat = (probs_q * (torch.log(probs_q + 1e-8) - torch.log(probs_p + 1e-8))).sum(dim=-1)
        return torch.clamp(kl_per_cat, min=free_bits).sum(dim=-1).mean()

    @staticmethod
    def _symlog(x: torch.Tensor) -> torch.Tensor:
        return torch.sign(x) * torch.log(torch.abs(x) + 1.0)

    def _symlog_mse(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(self._symlog(pred), self._symlog(target))

    def _lpips_loss(self, pred_rgb: torch.Tensor, target_rgb: torch.Tensor) -> torch.Tensor:
        """Perceptual distance in VGG feature space. Inputs should be [0,1]."""
        if self._lpips_net is None:
            return torch.tensor(0.0, device=pred_rgb.device)
        return self._lpips_net(pred_rgb * 2.0 - 1.0, target_rgb * 2.0 - 1.0).mean()

    def forward(
        self,
        rgb_t: torch.Tensor,
        depth_t: torch.Tensor,
        action_t: torch.Tensor,
        rgb_t1: torch.Tensor | None = None,
        depth_t1: torch.Tensor | None = None,
        hidden_state: torch.Tensor | None = None,
        prop_t: torch.Tensor | None = None,
    ):
        bsz = rgb_t.shape[0]
        if hidden_state is None:
            hidden_state = torch.zeros(bsz, self.hidden_dim, device=rgb_t.device)

        enc_t, skips_t = self._encode_obs(rgb_t, depth_t)
        action_emb = self.action_proj(action_t)

        extras = []
        if self.prop_proj is not None and prop_t is not None:
            extras = [self.prop_proj(prop_t)]

        # Prior: predict z without seeing the future frame
        prior_logits = self.prior_head(torch.cat([action_emb, *extras, hidden_state], dim=-1))
        prior_z, prior_probs = self._straight_through(prior_logits)

        # Posterior: also conditions on the actual next frame (training only)
        post_z, post_probs = prior_z, prior_probs
        if rgb_t1 is not None and depth_t1 is not None:
            enc_t1, _ = self._encode_obs(rgb_t1, depth_t1)
            post_logits = self.post_head(torch.cat([enc_t, action_emb, *extras, enc_t1, hidden_state], dim=-1))
            post_z, post_probs = self._straight_through(post_logits)

        hidden_next = self.gru(torch.cat([post_z, action_emb], dim=-1), hidden_state)
        pred_rgb_t1, pred_depth_t1 = self._decode(post_z, hidden_next, skips_t)

        return {
            "pred_rgb_t1":   pred_rgb_t1,
            "pred_depth_t1": pred_depth_t1,
            "prior_probs":   prior_probs,
            "post_probs":    post_probs,
            "hidden_next":   hidden_next,
        }

    @torch.no_grad()
    def rollout(
        self,
        rgb_0: torch.Tensor,
        depth_0: torch.Tensor,
        actions: torch.Tensor,
        hidden_state: torch.Tensor | None = None,
        prop_seq: torch.Tensor | None = None,
        use_predicted_skips: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Imagine T steps forward from a single starting frame using the prior.

        Returns (pred_rgbs, pred_depths, final_hidden).
        """
        B, T, _ = actions.shape
        device = rgb_0.device
        if hidden_state is None:
            hidden_state = torch.zeros(B, self.hidden_dim, device=device)

        rgb_cur, depth_cur = rgb_0, depth_0
        pred_rgbs, pred_depths = [], []

        for t in range(T):
            prop_t = prop_seq[:, t] if prop_seq is not None else None
            out = self.forward(
                rgb_t=rgb_cur,
                depth_t=depth_cur,
                action_t=actions[:, t],
                hidden_state=hidden_state,
                prop_t=prop_t,
            )
            pred_rgbs.append(out["pred_rgb_t1"])
            pred_depths.append(out["pred_depth_t1"])
            hidden_state = out["hidden_next"]

            if use_predicted_skips:
                rgb_cur   = out["pred_rgb_t1"]
                depth_cur = out["pred_depth_t1"]

        return (
            torch.stack(pred_rgbs,   dim=1),
            torch.stack(pred_depths, dim=1),
            hidden_state,
        )

    def compute_loss(
        self,
        outputs: dict,
        rgb_t1: torch.Tensor,
        depth_t1: torch.Tensor,
        free_bits: float = 1.0,
        kl_balance: float = 0.8,
        rgb_weight: float = 1.0,
        depth_weight: float = 1.0,
        lpips_weight: float = 0.5,
    ):
        rgb_loss   = self._symlog_mse(outputs["pred_rgb_t1"],   rgb_t1)
        depth_loss = self._symlog_mse(outputs["pred_depth_t1"], depth_t1)

        perceptual_loss = torch.tensor(0.0, device=rgb_t1.device)
        if lpips_weight > 0:
            perceptual_loss = self._lpips_loss(outputs["pred_rgb_t1"], rgb_t1)

        post_probs  = outputs["post_probs"]
        prior_probs = outputs["prior_probs"]

        # KL balancing: mostly train the prior to match posterior (dynamics loss),
        # with a smaller term training the posterior (representation loss).
        dynamics_loss = self._kl_categorical(post_probs.detach(), prior_probs,          free_bits)
        repr_loss     = self._kl_categorical(post_probs,          prior_probs.detach(), free_bits)
        kl_loss       = kl_balance * dynamics_loss + (1.0 - kl_balance) * repr_loss

        total = (rgb_weight * rgb_loss
                 + depth_weight * depth_loss
                 + lpips_weight * perceptual_loss
                 + kl_loss)

        return {
            "total_loss":      total,
            "rgb_loss":        rgb_loss.detach(),
            "depth_loss":      depth_loss.detach(),
            "kl_loss":         kl_loss.detach(),
            "perceptual_loss": perceptual_loss.detach(),
        }
