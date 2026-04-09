from variational_autoencoder.train import train, DEVICE, TEST_LOADER
import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

from variational_autoencoder.model import VariationalAutoencoder as VAEModel

def show_reconstructions(model, loader, device, n=8):
    model.eval()
    batch, _ = next(iter(loader))
    batch = batch[:n].to(device)

    with torch.no_grad():
        x_hat, _, _ = model(batch)

    # stack originals on top, reconstructions on bottom
    comparison = torch.cat([batch, x_hat], dim=0)
    grid = make_grid(comparison, nrow=n).cpu()

    plt.figure(figsize=(n * 1.5, 3))
    plt.imshow(grid.permute(1, 2, 0))  # channels last for matplotlib
    plt.axis('off')
    plt.title('Top: Original  |  Bottom: Reconstruction')
    plt.tight_layout()
    plt.show()

def run_main():
    model = train()
    # model = VAEModel(latent_dim=4).to(DEVICE)
    # ckpt = torch.load("checkpoints/last.pt", map_location=DEVICE)
    # model.load_state_dict(ckpt["model"])
    show_reconstructions(model, TEST_LOADER, DEVICE)