import torch.nn as nn
import torch


# ── Architecture spec ────────────────────────────────────────────────
# Each entry is an output channel count.
# Rule: first time a channel count appears  → downsample (encoder) / upsample (decoder)
#        repeated channel count              → residual refine block
#
# Examples:
#   [32, 64, 64, 128, 128]           → 3 downsamples, latent 4x4
#   [32, 32, 64, 64, 128, 128]       → 3 downsamples, latent 4x4, refine at each scale
#   [64, 128]                         → 2 downsamples, latent 8x8 (less compression)
#   [32, 64, 64, 64, 128, 128, 128]  → 3 downsamples, deeper refine blocks

CHANNEL_MAP = [64, 64, 128, 128]


# ── Building blocks ──────────────────────────────────────────────────

class ResBlock(nn.Module):
    """Conv → GroupNorm → SiLU → Conv → GroupNorm + skip → SiLU."""
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
        )
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(x + self.block(x))


class DownBlock(nn.Module):
    """Strided conv to halve spatial dims and change channels."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    """Transposed conv to double spatial dims and change channels."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 3, stride=2, padding=1, output_padding=1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.block(x)


# ── Encoder ──────────────────────────────────────────────────────────

class Encoder(nn.Module):
    def __init__(self, latent_dim, channel_map=None):
        super().__init__()

        if channel_map is None:
            channel_map = CHANNEL_MAP
        layers = []
        in_ch = 3
        seen = set()

        for ch in channel_map:
            if ch not in seen:
                layers.append(DownBlock(in_ch, ch))
                seen.add(ch)
            else:
                layers.append(ResBlock(ch))
            in_ch = ch

        self.layers = nn.Sequential(*layers)
        self.conv_mu = nn.Conv2d(in_ch, latent_dim, 1)
        self.conv_log_var = nn.Conv2d(in_ch, latent_dim, 1)

    def forward(self, x):
        x = self.layers(x)
        return self.conv_mu(x), self.conv_log_var(x)


# ── Decoder ──────────────────────────────────────────────────────────

class Decoder(nn.Module):
    def __init__(self, latent_dim, channel_map=None):
        super().__init__()

        # Build encoder ops, then reverse them for the decoder
        if channel_map is None:
            channel_map = CHANNEL_MAP
        in_ch = 3
        seen = set()
        encoder_ops = []

        for ch in channel_map:
            if ch not in seen:
                encoder_ops.append(("down", in_ch, ch))
                seen.add(ch)
            else:
                encoder_ops.append(("res", ch, ch))
            in_ch = ch

        # Decoder input projection: latent_dim → last encoder channel
        self.conv_in = nn.Conv2d(latent_dim, channel_map[-1], 1)

        # Mirror encoder ops in reverse
        layers = []
        for op, a, b in reversed(encoder_ops):
            if op == "res":
                layers.append(ResBlock(a))
            else:
                # down(a -> b) becomes up(b -> a), but skip the 3-channel input
                out_ch = a if a != 3 else b
                layers.append(UpBlock(b, out_ch))

        # Final projection to RGB
        layers.append(nn.Conv2d(channel_map[0], 3, 3, padding=1))
        layers.append(nn.Sigmoid())

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(self.conv_in(x))


# ── VAE ──────────────────────────────────────────────────────────────

class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dim, channel_map=None):
        super().__init__()
        if channel_map is None:
            channel_map = CHANNEL_MAP
        self.encoder = Encoder(latent_dim, channel_map)
        self.decoder = Decoder(latent_dim, channel_map)

    @staticmethod
    def reparameterize(mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        x_hat = self.decoder(z)
        return x_hat, mu, log_var