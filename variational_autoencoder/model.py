import torch.nn as nn
import torch

class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim

        self.conv_stack = nn.Sequential(
            # 32 x 32 x 3 → 32 x 32 x 32 (refine)
            nn.Conv2d(3, 32, 3, stride=1, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),

            # 32 x 32 x 32 → 16 x 16 x 64 (downsample)
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),

            # 16 x 16 x 64 → 16 x 16 x 64 (refine)
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),

            # 16 x 16 x 64 → 8 x 8 x 128 (downsample)
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU(),

            # 8 x 8 x 128 → 8 x 8 x 128 (refine)
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU(),

            # 8 x 8 x 128 → 4 x 4 x 128 (downsample)
            nn.Conv2d(128, 128, 3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
        )

        self.conv_mu = nn.Conv2d(128, latent_dim, 1)
        self.conv_log_var = nn.Conv2d(128, latent_dim, 1)


    def forward(self, x):
        x = self.conv_stack(x)
        mu = self.conv_mu(x)
        log_var = self.conv_log_var(x)

        return mu, log_var


class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()

        self.conv_in = nn.Conv2d(latent_dim, 128, 1)

        self.conv_stack = nn.Sequential(
            # 4 x 4 x 128 → 8 x 8 x 128 (upsample)
            nn.ConvTranspose2d(128, 128, 3, stride=2, padding=1, output_padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU(),

            # 8 x 8 x 128 → 8 x 8 x 128 (refine)
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU(),

            # 8 x 8 x 128 → 16 x 16 x 64 (upsample)
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),

            # 16 x 16 x 64 → 16 x 16 x 64 (refine)
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),

            # 16 x 16 x 64 → 32 x 32 x 32 (upsample)
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),

            # 32 x 32 x 32 → 32 x 32 x 32 (refine)
            nn.Conv2d(32, 32, 3, stride=1, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),

            # 32 x 32 x 32 → 32 x 32 x 3 (output)
            nn.Conv2d(32, 3, 3, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.conv_in(x)
        x = self.conv_stack(x)
        return x


class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

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