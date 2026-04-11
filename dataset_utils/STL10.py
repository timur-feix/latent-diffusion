import torch
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import Resize, ToTensor, Compose
from torchvision.datasets import STL10

torch.manual_seed(1)
transform = Compose([
    Resize((96, 96)),
    ToTensor()
])

train_dataset = STL10(
    root="data",
    split="unlabeled",
    download=True,
    transform=transform
)

validation_dataset = STL10(
    root="data",
    split="train",
    download=True,
    transform=transform
)

test_dataset = STL10(
    root="data",
    split="test",
    download=True,
    transform=transform
)



def dataloaders(batch_size=64):
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True)

    validation_dataloader = DataLoader(validation_dataset,
                                       batch_size=batch_size,
                                       shuffle=False)

    test_dataloader = DataLoader(test_dataset,
                                 batch_size=batch_size,
                                 shuffle=True)

    return train_dataloader, validation_dataloader, test_dataloader