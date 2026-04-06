import torchvision.models as models
import torch.nn.functional as functional
import torch.nn as nn
import torch

class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg19(pretrained=True).features

        # grab layers at different depths
        self.slice1 = vgg[:4]  # after relu1_2 — edges, colors
        self.slice2 = vgg[:9]  # after relu2_2 — textures
        self.slice3 = vgg[:18]  # after relu3_4 — shapes

        # freeze everything — we never train VGG
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x, x_hat):
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x.device)

        x = (x - mean) / std
        x_hat = (x_hat - mean) / std

        loss = 0
        for _slice in [self.slice1, self.slice2, self.slice3]:
            loss += functional.mse_loss(_slice(x), _slice(x_hat))

        return loss