import torch
import torch.nn as nn
import torch.optim as optim
from config import Config
from data_handler import get_dataloaders
from mae_model import MaskedAutoencoder
from pca_compare2 import run_pca_comparison, plot_comparison
import os

def patchify_images(imgs):
    """ Turn images into patches for Loss Calculation """
    p = Config.PATCH_SIZE
    x = imgs.unfold(2, p, p).unfold(3, p, p)
    x = x.permute(0, 2, 3, 1, 4, 5).contiguous()
    x = x.view(imgs.shape[0], -1, p*p*3)
    return x

def unpatchify(x):
    """ Reconstruct images from patches for Visualization """
    p = Config.PATCH_SIZE
    h = w = Config.IMAGE_SIZE // p
    x = x.reshape(x.shape[0], h, w, p, p, 3)
    x = torch.einsum('nhwpqc->nchpwq', x)
    imgs = x.reshape(x.shape[0], 3, h * p, h * p)
    return imgs

def train():
    # 1. Setup
    trainloader, testloader = get_dataloaders()
    model = MaskedAutoencoder().to(Config.DEVICE)
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY)
    
    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.EPOCHS)
    
    print("Model initialized. Starting training...")

    # 2. Training Loop
    for epoch in range(Config.EPOCHS):
        model.train()
        total_loss = 0
        
        for batch_idx, (imgs, _) in enumerate(trainloader):
            imgs = imgs.to(Config.DEVICE)
            
            # Forward pass
            pred, mask = model(imgs, mask_ratio=Config.MASK_RATIO)
            
            # Prepare target
            target = patchify_images(imgs)
            
            # MSE Loss on Masked Patches only
            loss = (pred - target) ** 2
            loss = loss.mean(dim=-1)
            loss = (loss * mask).sum() / mask.sum() 
            
            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        # Step the scheduler at the end of the epoch
        scheduler.step()
        
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch+1}/{Config.EPOCHS} | Loss: {total_loss/len(trainloader):.4f} | LR: {current_lr:.6f}")

    # --- SAVE THE MODEL ---
    save_path = "mae_cifar10_final.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Training Complete. Model saved to {save_path}")

    # 3. Evaluation & Comparison
    print("Running Evaluation...")
    model.eval()
    
    # Get a single batch for visualization
    imgs, _ = next(iter(testloader))
    imgs = imgs.to(Config.DEVICE)
    
    with torch.no_grad():
        # MAE Reconstruction
        pred_patches, _ = model(imgs, mask_ratio=Config.MASK_RATIO)
        mae_recon = unpatchify(pred_patches)
        
        # PCA Reconstruction
        # Note: This will now run safely with n_components = 64
        pca_recon = run_pca_comparison(imgs)
        
        # Plot
        plot_comparison(imgs, mae_recon, pca_recon)

if __name__ == "__main__":
    train()