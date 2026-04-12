import math
import torch
import torch.nn.functional as F
from torch import nn
import torch.optim as optim

class ResBlock(nn.Module):

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
        # original input + residual connection (delta(x))
        return self.act(x + self.net(x))


class VisionWM(nn.Module):
    ENC_RGB   = (3, 64, 96, 128, 192, 256, 384)
    ENC_DEPTH = (1, 64, 96, 128, 192, 256, 384)

    def __init__(
        self,
        action_dim: int,
        image_height: int,
        image_width: int,
        cnn_embed_dim: int,
        action_embed_dim: int,
        hidden_dim: int,
        latent_dim: int,
        lr: float,
        weight_decay: float,
        device: str,
        prop_dim: int = 0,
        prop_embed_dim: int = 64,
    ):
        super().__init__()
        self.image_height = image_height
        self.image_width = image_width
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        self.n_stages = int(math.log2(image_height // 4))
        enc_rgb = self.ENC_RGB[:self.n_stages + 1]
        enc_depth = self.ENC_DEPTH[:self.n_stages + 1]

        self.rgb_enc_stages = self._build_encoder(enc_rgb, self.n_stages)
        self.depth_enc_stages = self._build_encoder(enc_depth, self.n_stages)

        bottleneck = 4
        rgb_flat = enc_rgb[-1] * bottleneck * bottleneck
        depth_flat = enc_depth[-1] * bottleneck * bottleneck

        self.rgb_enc_proj = nn.Sequential(
            nn.Linear(rgb_flat, cnn_embed_dim),
            nn.LayerNorm(cnn_embed_dim),
            nn.SiLU(),
        )
        self.depth_enc_proj = nn.Sequential(
            nn.Linear(depth_flat, cnn_embed_dim),
            nn.LayerNorm(cnn_embed_dim),
            nn.SiLU(),
        )
        self.action_proj = nn.Sequential(
            nn.Linear(action_dim, action_embed_dim),
            nn.LayerNorm(action_embed_dim),
            nn.SiLU(),
        )

        prop_emb = 0
        if prop_dim > 0:
            self.prop_proj = nn.Sequential(
                nn.Linear(prop_dim, prop_embed_dim),
                nn.LayerNorm(prop_embed_dim),
                nn.SiLU(),
            )
            prop_emb = prop_embed_dim
        else:
            self.prop_proj = None

        # GRU 
        self.gru = nn.GRU(
            input_size=2 * cnn_embed_dim + action_embed_dim + prop_emb,
            hidden_size=hidden_dim, num_layers=1, batch_first=True,
        )

        # Latent bottleneck
        self.latent_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim), nn.LayerNorm(latent_dim), nn.SiLU(),
        )
        self.lat_rgb_proj = nn.Linear(latent_dim, latent_dim // 2)
        self.lat_depth_proj = nn.Linear(latent_dim, latent_dim // 2)

        dec_rgb = list(reversed(enc_rgb[1:])) + [32] # added 32 just for last stage for smooth transition
        dec_depth = list(reversed(enc_depth[1:])) + [32]

        self.rgb_dec_fc = nn.Sequential(nn.Linear(latent_dim // 2, rgb_flat), nn.SiLU())
        self.rgb_dec_ups, self.rgb_dec_convs = self._build_decoder(dec_rgb, self.n_stages)
        self.rgb_head = nn.Conv2d(dec_rgb[-1], 3, 3, padding=1)

        self.depth_dec_fc = nn.Sequential(nn.Linear(latent_dim // 2, depth_flat), nn.SiLU())
        self.depth_dec_ups, self.depth_dec_convs = self._build_decoder(dec_depth, self.n_stages)
        self.depth_head = nn.Conv2d(dec_depth[-1], 1, 3, padding=1)

        self.optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        self.to(device if torch.cuda.is_available() else "cpu")

    @staticmethod
    def _build_encoder(channels, n_stages):
        stages = nn.ModuleList()
        for i in range(n_stages):
            stages.append(nn.Sequential(
                nn.Conv2d(channels[i], channels[i + 1], 4, stride=2, padding=1, bias=False),
                nn.GroupNorm(min(32, channels[i + 1]), channels[i + 1]),
                nn.SiLU(),
                ResBlock(channels[i + 1]),
            ))
        return stages

    @staticmethod
    def _build_decoder(channels, n_stages):
        ups = nn.ModuleList()
        convs = nn.ModuleList()
        for i in range(n_stages):
            ups.append(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False))
            convs.append(nn.Sequential(
                nn.Conv2d(channels[i], channels[i + 1], 3, padding=1, bias=False),
                nn.GroupNorm(min(32, channels[i + 1]), channels[i + 1]),
                nn.SiLU(),
                ResBlock(channels[i + 1]),
            ))
        return ups, convs

    # ENCODER
    def _run_encoder(self, x, stages, proj):
        for stage in stages:
            x = stage(x)
        return proj(x.flatten(1))

    def _encode_sequence(self, rgb, depth, action, prop):
        #rgb = (B, T, H, W, 3)
        #depth = (B, T, H, W, 1)
        #action = (B, T, action_dim)
        #prop = (B, T, prop_dim)

        B, T = rgb.shape[:2]
        flat = lambda x: x.reshape(B * T, *x.shape[2:])
        unflat = lambda x: x.reshape(B, T, -1)

        rgb_enc = unflat(self._run_encoder(flat(rgb), self.rgb_enc_stages, self.rgb_enc_proj))
        depth_enc = unflat(self._run_encoder(flat(depth), self.depth_enc_stages, self.depth_enc_proj))
        act_enc = unflat(self.action_proj(flat(action)))

        parts = [rgb_enc, depth_enc, act_enc]
        if self.prop_proj is not None and prop is not None:
            parts.append(unflat(self.prop_proj(flat(prop))))

        return torch.cat(parts, dim=-1)

    # DECODER

    def _run_decoder(self, lat, fc, top_ch, ups, convs, head):
        x = fc(lat).reshape(lat.shape[0], top_ch, 4, 4)
        for up, conv in zip(ups, convs):
            x = conv(up(x))
        return head(x)

    # LOSS HELPERS

    @staticmethod
    def _symlog(x):
        return x.sign() * (x.abs() + 1.0).log()

    def _symlog_mse(self, pred, target):
        return F.mse_loss(self._symlog(pred), self._symlog(target))

    # FORWARD

    def forward(self, rgb_t, depth_t, action_t, hidden_state=None, prop_t=None):
        
        #(B,C,H,W) -> (B,T,C,H,W)
        single = rgb_t.dim() == 4
        if single:
            rgb_t = rgb_t.unsqueeze(1)
            depth_t = depth_t.unsqueeze(1)
            action_t = action_t.unsqueeze(1)
            if prop_t is not None:
                prop_t = prop_t.unsqueeze(1)

        B, T = rgb_t.shape[:2]

        if hidden_state is None:
            h0 = torch.zeros(1, B, self.hidden_dim, device=rgb_t.device)
        elif hidden_state.dim() == 2:
            h0 = hidden_state.unsqueeze(0)
        else:
            h0 = hidden_state

        fused = self._encode_sequence(rgb_t, depth_t, action_t, prop_t)
        gru_out, h_n = self.gru(fused, h0)

        flat = gru_out.reshape(B * T, -1)
        lat = self.latent_head(flat)

        enc_rgb = self.ENC_RGB[:self.n_stages + 1]
        enc_depth = self.ENC_DEPTH[:self.n_stages + 1]

        pred_rgb = self._run_decoder(
            self.lat_rgb_proj(lat), self.rgb_dec_fc, enc_rgb[-1],
            self.rgb_dec_ups, self.rgb_dec_convs, lambda x: torch.sigmoid(self.rgb_head(x)),
        ).reshape(B, T, 3, self.image_height, self.image_width)

        pred_depth = self._run_decoder(
            self.lat_depth_proj(lat), self.depth_dec_fc, enc_depth[-1],
            self.depth_dec_ups, self.depth_dec_convs, self.depth_head,
        ).reshape(B, T, 1, self.image_height, self.image_width)

        if single:
            pred_rgb = pred_rgb.squeeze(1)
            pred_depth = pred_depth.squeeze(1)

        return {"pred_rgb_t1": pred_rgb, "pred_depth_t1": pred_depth, "hidden_next": h_n}

    # ROLLOUT

    @torch.no_grad()
    def rollout(self, rgb_0, depth_0, actions, hidden_state=None, prop_seq=None, use_predicted_skips=False):
        B, T, _ = actions.shape
        if hidden_state is None:
            hidden_state = torch.zeros(1, B, self.hidden_dim, device=rgb_0.device)
        elif hidden_state.dim() == 2:
            hidden_state = hidden_state.unsqueeze(0)

        rgb_cur, depth_cur = rgb_0, depth_0
        pred_rgbs, pred_depths = [], []

        for t in range(T):
            prop_t = prop_seq[:, t] if prop_seq is not None else None
            out = self.forward(rgb_cur, depth_cur, actions[:, t], hidden_state, prop_t)
            pred_rgbs.append(out["pred_rgb_t1"])
            pred_depths.append(out["pred_depth_t1"])
            hidden_state = out["hidden_next"]
            if use_predicted_skips:
                rgb_cur = out["pred_rgb_t1"]
                depth_cur = out["pred_depth_t1"]

        return torch.stack(pred_rgbs, 1), torch.stack(pred_depths, 1), hidden_state

    # LOSS

    def compute_loss(self, outputs, rgb_t1, depth_t1, rgb_weight=1.0, depth_weight=1.0, **_):
        rgb_loss = self._symlog_mse(outputs["pred_rgb_t1"], rgb_t1)
        depth_loss = self._symlog_mse(outputs["pred_depth_t1"], depth_t1)
        total = rgb_weight * rgb_loss + depth_weight * depth_loss
        return {
            "total_loss": total,
            "rgb_loss": rgb_loss.detach(),
            "depth_loss": depth_loss.detach(),
            "kl_loss": torch.tensor(0.0),
        }
