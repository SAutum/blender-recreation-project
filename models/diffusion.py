from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_STATE_DIM = 9


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int = 128) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=t.device, dtype=torch.float32)
            / max(half - 1, 1)
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if emb.shape[1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[1]))
        return emb


class ImageEncoder(nn.Module):
    """Legacy early-fusion encoder used by v1-v4 checkpoints."""

    def __init__(self, out_dim: int = 256, in_channels: int = 3) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.net = nn.Sequential(
            nn.Conv2d(self.in_channels, 32, 5, stride=2, padding=2),
            nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, out_dim),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.net(image)


class SpatialPairImageEncoder(nn.Module):
    """v5 paired-view encoder with shared weights and spatial fusion.

    Input is always two RGB views concatenated channel-wise (6 channels).
    The same CNN encodes both branches independently. Fusion happens while the
    feature maps are still spatial, using left, right, and signed right-left
    features. Only after fusion do we compress to a 256-D conditioning vector.

    Mono and stereo runs therefore use exactly the same architecture:
      mono   = (left, left)
      stereo = (left, right)
    """

    def __init__(self, out_dim: int = 256) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2),
            nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(128, 128, 3, stride=2, padding=1),
            nn.SiLU(),
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(128 * 3, 256, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.SiLU(),
            # Preserve a coarse 4x4 spatial layout instead of collapsing directly
            # to 1x1 as the legacy encoder does.
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, out_dim),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.shape[1] != 6:
            raise ValueError(
                f"SpatialPairImageEncoder expects 6 channels (two RGB views), got {image.shape[1]}"
            )
        left = image[:, :3]
        right = image[:, 3:6]
        left_f = self.shared(left)
        right_f = self.shared(right)
        fused = torch.cat([left_f, right_f, right_f - left_f], dim=1)
        return self.projection(self.fusion(fused))


class ConditionalStateDenoiser(nn.Module):
    def __init__(
        self,
        state_dim: int = DEFAULT_STATE_DIM,
        cond_dim: int = 256,
        time_dim: int = 128,
        image_channels: int = 3,
        encoder_type: str = "legacy",
    ) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.image_channels = int(image_channels)
        self.encoder_type = str(encoder_type)

        if self.encoder_type == "legacy":
            self.image_encoder = ImageEncoder(cond_dim, in_channels=self.image_channels)
        elif self.encoder_type == "spatial_pair":
            if self.image_channels != 6:
                raise ValueError(
                    "encoder_type='spatial_pair' requires image_channels=6 (two RGB branches)"
                )
            self.image_encoder = SpatialPairImageEncoder(cond_dim)
        else:
            raise ValueError(
                f"Unknown encoder_type={self.encoder_type!r}; expected legacy or spatial_pair"
            )

        self.time_embedding = SinusoidalTimeEmbedding(time_dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.state_dim + cond_dim + time_dim, 512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Linear(512, 256),
            nn.SiLU(),
            nn.Linear(256, self.state_dim),
        )

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        return self.image_encoder(image)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        image: torch.Tensor | None = None,
        image_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if image_features is None:
            if image is None:
                raise ValueError("Either image or image_features must be supplied")
            image_features = self.image_encoder(image)
        t_emb = self.time_embedding(t)
        return self.mlp(torch.cat([x_t, image_features, t_emb], dim=1))


class GaussianDiffusion:
    def __init__(
        self,
        steps: int = 100,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        device: torch.device | str = "cpu",
    ) -> None:
        self.steps = steps
        self.device = torch.device(device)
        self.betas = torch.linspace(beta_start, beta_end, steps, device=self.device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        ab = self.alpha_bars[t].unsqueeze(1)
        return torch.sqrt(ab) * x0 + torch.sqrt(1.0 - ab) * noise

    def training_loss(self, model: nn.Module, image: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
        batch = x0.shape[0]
        t = torch.randint(0, self.steps, (batch,), device=x0.device)
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)
        pred_noise = model(x_t, t, image=image)
        return F.mse_loss(pred_noise, noise)

    @torch.no_grad()
    def sample(self, model: ConditionalStateDenoiser, image: torch.Tensor, n: int = 1) -> torch.Tensor:
        model.eval()
        if image.shape[0] != 1:
            raise ValueError("sample() currently expects one conditioning sample at a time")
        image_features = model.encode_image(image).repeat(n, 1)
        x = torch.randn(n, model.state_dim, device=image.device)

        for step in reversed(range(self.steps)):
            t = torch.full((n,), step, device=image.device, dtype=torch.long)
            pred_noise = model(x, t, image_features=image_features)
            alpha = self.alphas[step]
            alpha_bar = self.alpha_bars[step]
            beta = self.betas[step]
            mean = (x - (beta / torch.sqrt(1.0 - alpha_bar)) * pred_noise) / torch.sqrt(alpha)
            if step > 0:
                x = mean + torch.sqrt(beta) * torch.randn_like(x)
            else:
                x = mean

        return torch.clamp(x, -1.25, 1.25)
