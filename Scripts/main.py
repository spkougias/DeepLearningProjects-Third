#Keep part of the image
import torch
import torch.nn as nn
import torch.optim as optim
from config import Config
from data_handler import get_dataloaders
from mae_model import MaskedAutoencoder
from pca_compare import run_pca_comparison, plot_comparison

def patchify_images(imgs):
    """ Turn images into patches for Loss Calculation """
    p = Config.PATCH_SIZE
    # Create patches
    x = imgs.unfold(2, p, p).unfold(3, p, p)
    x = x.permute(0, 2, 3, 1, 4, 5).contiguous()
    x = x.view(imgs.shape[0], -1, p*p*3)
    return x

def unpatchify(x):
    """ Correctly reconstructs (B, 3, 32, 32) from patches. """
    p = Config.PATCH_SIZE
    h = w = Config.IMAGE_SIZE // p
    c = 3
    # Reshape to (Batch, GridH, GridW, Channel, PatchH, PatchW)
    x = x.reshape(x.shape[0], h, w, c, p, p)
    # Permute to (Batch, Channel, GridH, PatchH, GridW, PatchW)
    x = x.permute(0, 3, 1, 4, 2, 5)
    # Fuse spatial dimensions
    x = x.reshape(x.shape[0], c, h * p, w * p)
    return x

def train():
    # 1. Setup
    trainloader, testloader = get_dataloaders()
    model = MaskedAutoencoder().to(Config.DEVICE)
    
    optimizer = optim.AdamW(model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.EPOCHS)
    
    print(f"Model initialized. Starting training in {Config.DEVICE}")

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
            
            # Loss Calculation (MSE on masked patches only)
            loss = (pred - target) ** 2
            loss = loss.mean(dim=-1)
            loss = (loss * mask).sum() / mask.sum() 
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch+1}/{Config.EPOCHS} | Loss: {total_loss/len(trainloader):.4f} | LR: {current_lr:.6f}")

    # 3. Evaluation & Comparison
    print("Training Complete. Running Evaluation...")
    model.eval()
    
    # Get a single batch for visualization
    imgs, _ = next(iter(testloader))
    imgs = imgs.to(Config.DEVICE)
    
    with torch.no_grad():
        # --- MAE Composite Reconstruction Logic ---
        # 1. Get Model Predictions and the Mask (0=Visible, 1=Removed)
        # We use a high masking ratio (75%) for the demo to make it hard
        pred_patches, mask = model(imgs, mask_ratio=0.75)
        
        # 2. Get the actual ground truth patches
        target_patches = patchify_images(imgs)
        
        # 3. Create Composite: (Prediction * Mask) + (Target * (1-Mask))
        # This fills the "holes" (Mask=1) with prediction, and keeps original (Mask=0)
        mask = mask.unsqueeze(-1) # Match shape
        composite_patches = (pred_patches * mask) + (target_patches * (1 - mask))
        
        # 4. Reconstruct to image
        mae_recon = unpatchify(composite_patches)
        
        # --- PCA Reconstruction ---
        pca_recon = run_pca_comparison(imgs)
        
        # --- Plotting ---
        # We pass the composite 'mae_recon' to the plotter
        plot_comparison(imgs, mae_recon, pca_recon)

if __name__ == "__main__":
    train()