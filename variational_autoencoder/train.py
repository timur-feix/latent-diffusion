from dataset_utils import STL10
from variational_autoencoder.model import VariationalAutoencoder as VAEModel
from functional_utils.save_checkpoint import save_checkpoint

import torch.nn.functional as functional
import torch

from tqdm import tqdm
from pathlib import Path

TRAIN_LOADER, VALIDATION_LOADER, TEST_LOADER = STL10.dataloaders()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def vae_loss(x, x_hat, mu, log_var, kl_weight=1e-6, perc_weight=5):
    recon = functional.mse_loss(x_hat, x, reduction="sum") / x.shape[0]
    kl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / x.shape[0]

    return 0.1 * recon + kl_weight * kl, recon, kl


def print_output(losses, sched_lrs):
    for epoch in range(EPOCHS):
        print(f"epoch {epoch + 1} | loss {losses[epoch]:.4f} | lr {sched_lrs[epoch]:.6f}")


MODEL = VAEModel(latent_dim=8).to(DEVICE)
OPTIM = torch.optim.Adam(MODEL.parameters(), lr=1e-3)
SCHED = torch.optim.lr_scheduler.CosineAnnealingLR(OPTIM, T_max=40)

EPOCHS = 40


def train():
    best_loss = float("inf")
    print("DEBUG: THIS IS THE RIGHT ONE with STL10 96x96 and 40 epochs")
    losses = []
    sched_lrs = []

    with tqdm(total=EPOCHS * len(TRAIN_LOADER), desc="training vae") as train_progress:
        for epoch in range(EPOCHS):
            MODEL.train()
            total_loss = 0

            for batch, _ in TRAIN_LOADER:
                batch = batch.to(DEVICE)
                x_hat, mu, log_var = MODEL(batch)

                loss, recon, kl = vae_loss(batch, x_hat, mu, log_var)

                OPTIM.zero_grad()
                loss.backward()
                OPTIM.step()

                total_loss += loss.item()

                train_progress.update()

            SCHED.step()
            avg_loss = total_loss / len(TRAIN_LOADER)

            checkpoint_filepath = Path("checkpoints")
            checkpoint_filepath.mkdir(parents=True, exist_ok=True)

            save_checkpoint(epoch + 1,
                            MODEL,
                            OPTIM,
                            SCHED,
                            avg_loss,
                            checkpoint_filepath / "last.pt")

            if avg_loss < best_loss:
                save_checkpoint(epoch + 1,
                                MODEL,
                                OPTIM,
                                SCHED,
                                avg_loss,
                                checkpoint_filepath / "best.pt")
                best_loss = avg_loss

            losses.append(avg_loss)
            sched_lrs.append(SCHED.get_last_lr()[0])

    print_output(losses, sched_lrs)
    return MODEL