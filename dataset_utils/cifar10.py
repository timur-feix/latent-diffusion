import torch
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import Resize, ToTensor, Compose
from torchvision.datasets import CIFAR10

torch.manual_seed(1)
transform = Compose([
    Resize((32, 32)),
    ToTensor()
])

train_dataset = CIFAR10(
    root="data",
    train=True,
    download=True,
    transform=transform
)

non_train_dataset = CIFAR10(
    root="data",
    train=False,
    download=True,
    transform=transform
)

# want to have a 'validation' and 'test' dataset as opposed to just 'test'

validation_dataset, test_dataset = random_split(non_train_dataset,
                                                [len(non_train_dataset) // 2]*2)


def dataloaders(batch_size=64):
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True)

    validation_dataloader = DataLoader(validation_dataset,
                                       batch_size=batch_size,
                                       shuffle=False)

    test_dataloader = DataLoader(test_dataset,
                                 batch_size=batch_size,
                                 shuffle=False)

    return train_dataloader, validation_dataloader, test_dataloader