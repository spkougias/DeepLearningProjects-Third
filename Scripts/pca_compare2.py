import numpy as np
from sklearn.decomposition import PCA
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
    
    # Use 64 components to match the number of patches (8x8 grid = 64)
    # This makes for a fair "Component vs Component" comparison
    n_components = min(64, B)
    
    pca = PCA(n_components=n_components) 
    compressed = pca.fit_transform(flat_imgs)
    reconstructed_flat = pca.inverse_transform(compressed)
    
    # Reshape back
    reconstructed_imgs = reconstructed_flat.reshape(B, C, H, W)
    return reconstructed_imgs

def plot_comparison(original, mae_recon, pca_recon, index=0):
    """
    Plots the three stages side-by-side.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    
    # Helper to un-normalize correctly
    def prepare(img_tensor):
        # Handle both Tensor and Numpy inputs
        if isinstance(img_tensor, torch.Tensor):
            img = img_tensor.cpu().detach().numpy()
        else:
            img = img_tensor
            
        # Transpose (C, H, W) -> (H, W, C)
        if img.shape[0] == 3: 
            img = img.transpose(1, 2, 0)
        
        # CIFAR-10 Mean and Std
        mean = np.array([0.4914, 0.4822, 0.4465])
        std = np.array([0.2023, 0.1994, 0.2010])
        
        # Reverse normalization
        img = img * std + mean
        return np.clip(img, 0, 1)

    # 1. Original
    axes[0].imshow(prepare(original[index]))
    axes[0].set_title("Original", fontsize=16)
    
    # 2. MAE (Composite)
    axes[1].imshow(prepare(mae_recon[index]))
    axes[1].set_title("MAE (Transformer)\nVisible + In-painted", fontsize=16)
    
    # 3. PCA
    axes[2].imshow(prepare(pca_recon[index]))
    axes[2].set_title("PCA (Classic)\n64 Components", fontsize=16)
    
    for ax in axes: ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('comparison_result.png', dpi=300)
    plt.close()
    print("✅ Result saved to comparison_result.png")