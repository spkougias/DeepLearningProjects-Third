import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import torch

def run_pca_comparison(original_images):
    """
    Takes a batch of tensors (B, 3, 32, 32), flattens them,
    runs PCA, and reconstructs them to compare with MAE.
    """
    # Convert Torch -> Numpy
    imgs_np = original_images.cpu().numpy()
    B, C, H, W = imgs_np.shape
    
    # Flatten: [Batch, 3072]
    flat_imgs = imgs_np.reshape(B, -1)
    
    # --- FIX FOR CRASH ---
    # PCA cannot find more components than there are samples.
    # We want 120, but if batch_size is 64, we must limit to 64.
    n_components = min(120, B)
    
    pca = PCA(n_components=n_components) 
    compressed = pca.fit_transform(flat_imgs)
    reconstructed_flat = pca.inverse_transform(compressed)
    
    # Reshape back
    reconstructed_imgs = reconstructed_flat.reshape(B, C, H, W)
    return reconstructed_imgs

def plot_comparison(original, mae_recon, pca_recon, index=0):
    """
    Plots the three stages side-by-side with ACCURATE COLORS.
    """
    fig, axes = plt.subplots(1, 3, figsize=(12, 5))
    
    # Helper to un-normalize correctly
    def prepare(img):
        # Transpose from (C, H, W) -> (H, W, C)
        if len(img.shape) == 3: 
            img = img.transpose(1, 2, 0)
        
        # CIFAR-10 Mean and Std
        mean = np.array([0.4914, 0.4822, 0.4465])
        std = np.array([0.2023, 0.1994, 0.2010])
        
        # Reverse the normalization: pixel = (val * std) + mean
        img = img * std + mean
        
        # Clip to ensure valid pixel range [0, 1]
        return np.clip(img, 0, 1)

    # 1. Original Image
    axes[0].imshow(prepare(original[index].cpu().numpy()))
    axes[0].set_title("Original", fontsize=14)
    
    # 2. MAE Reconstruction (Transformer)
    axes[1].imshow(prepare(mae_recon[index].cpu().detach().numpy()))
    axes[1].set_title("MAE (Transformer)\nMasked Autoencoder", fontsize=14)
    
    # 3. PCA Reconstruction
    axes[2].imshow(prepare(pca_recon[index]))
    axes[2].set_title(f"PCA (Classic)\n{min(120, original.shape[0])} Components", fontsize=14)
    
    for ax in axes: 
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('comparison_result.png', dpi=300)
    plt.close()
    print("Comparison saved to comparison_result.png")