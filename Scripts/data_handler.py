import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from config import Config

def get_dataloaders():
    """
    Downloads CIFAR-10 and prepares DataLoaders.
    INCLUDES DATA AUGMENTATION for the Training set.
    """
    print(f"Preparing Data on {Config.DEVICE}...")
    means = (0.4914, 0.4822, 0.4465)
    stds = (0.2023, 0.1994, 0.2010)
    # --- TRAINING TRANSFORM (With Augmentation) ---
    # RandomCrop: Jitters the image slightly so the model sees different framing
    # HorizontalFlip: Doubles the effective dataset size
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(means,stds)
    ])

    # --- TESTING TRANSFORM (No Augmentation) ---
    # We want to evaluate on the "real" images, not the jittered ones
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), 
                             (0.2023, 0.1994, 0.2010)),
    ])

    # Download Training Data
    trainset = torchvision.datasets.CIFAR10(
        root=Config.DATA_PATH, train=True, download=True, transform=train_transform)
    
    trainloader = DataLoader(
        trainset, batch_size=Config.BATCH_SIZE, shuffle=True, 
        num_workers=Config.NUM_WORKERS, pin_memory=True)

    # Download Test Data
    testset = torchvision.datasets.CIFAR10(
        root=Config.DATA_PATH, train=False, download=True, transform=test_transform)
    
    testloader = DataLoader(
        testset, batch_size=Config.BATCH_SIZE, shuffle=False, 
        num_workers=Config.NUM_WORKERS, pin_memory=True)

    print(f"Data Loaded: {len(trainset)} Training Images (Augmented), {len(testset)} Test Images")
    return trainloader, testloader