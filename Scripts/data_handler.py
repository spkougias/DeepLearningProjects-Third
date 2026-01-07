import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
from config import Config

def get_data_loaders():
    # Tried to use data augmentation but it did not lead to any benefit
    train_transform = transforms.Compose([
        #transforms.RandomHorizontalFlip(p=0.5),
        #transforms.RandomRotation(degrees=10),
        #transforms.RandomAffine(0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
    ])

    #Clean transform for val/test
    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Load training with augmentation
    train_dataset_aug = datasets.CIFAR10(
        root='./data', 
        train=True, 
        transform=train_transform, 
        download=True
    )

    # Load training for validation
    train_dataset_clean = datasets.CIFAR10(
        root='./data', 
        train=True, 
        transform=test_transform, 
        download=True
    )

    # Load test
    test_dataset = datasets.CIFAR10(
        root='./data', 
        train=False, 
        transform=test_transform, 
        download=True
    )

    num_train = len(train_dataset_aug)
    indices = list(range(num_train))
    split = int(np.floor(Config.VALIDATION_SPLIT * num_train))

    np.random.shuffle(indices)

    train_idx, val_idx = indices[split:], indices[:split]

    # Loaders
    train_loader = DataLoader(
        train_dataset_aug, 
        batch_size=Config.BATCH_SIZE, 
        sampler=SubsetRandomSampler(train_idx)
    )
    
    val_loader = DataLoader(
        train_dataset_clean, 
        batch_size=Config.BATCH_SIZE, 
        sampler=SubsetRandomSampler(val_idx)
    )

    test_loader = DataLoader(
        test_dataset, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=False
    )

    return train_loader, val_loader, test_loader