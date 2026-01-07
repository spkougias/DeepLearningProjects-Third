import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import time
import os
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from config import Config
from data_handler import get_data_loaders
from model import LinearAutoencoder, ConvAutoencoder
from skimage.metrics import structural_similarity as ssim #SSIM metric

# TRAIN
def train_model():
    
    train_loader, val_loader, test_loader = get_data_loaders()
    
    #model = LinearAutoencoder(Config.INPUT_DIM, Config.HIDDEN_DIM).to(Config.DEVICE)
    model = ConvAutoencoder().to(Config.DEVICE)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
    


    print(f"Training on: {Config.DEVICE}")
    start_time = time.time() 
    train_loss_history = []
    val_loss_history = []
    
    for epoch in range(Config.NUM_EPOCHS):

        model.train()
        running_loss = 0.0
        for data in train_loader:
            img, _ = data
            img = img.to(Config.DEVICE)
            
            # Add noise
            if Config.NOISE_FACTOR > 0:
                noise = torch.randn_like(img) * Config.NOISE_FACTOR
                model_input = img + noise
                model_input = torch.clamp(model_input, 0., 1.)
            else:
                model_input = img
            output = model(model_input)
            loss = criterion(output, img)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        avg_train_loss = running_loss / len(train_loader)
        train_loss_history.append(avg_train_loss)
        
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for data in val_loader:
                img, _ = data
                img = img.to(Config.DEVICE)
                # Add Noise
                if Config.NOISE_FACTOR > 0:
                    noise = torch.randn_like(img) * Config.NOISE_FACTOR
                    model_input = img + noise
                    model_input = torch.clamp(model_input, 0., 1.)
                else:
                    model_input = img
                
                output = model(model_input)
                loss = criterion(output, img)
                running_val_loss += loss.item()
        
        avg_val_loss = running_val_loss / len(val_loader)
        val_loss_history.append(avg_val_loss)

        scheduler.step(avg_val_loss)
        
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch [{epoch+1}/{Config.NUM_EPOCHS}] "f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | LR: {current_lr:.6f}")
    end_time = time.time()
    training_duration = end_time - start_time
    print(f"Training time {training_duration:.2f} seconds.")
    
    return model, train_loss_history, val_loss_history, train_loader, test_loader

# RUN PCA
def run_pca_comparison(train_loader, test_loader):

    start_pca = time.time()
    
    train_data_list = []
    for imgs, _ in train_loader:
        train_data_list.append(imgs.view(imgs.size(0), -1).numpy())
    
    X_train = np.concatenate(train_data_list, axis=0)

    pca = PCA(n_components=Config.HIDDEN_DIM)
    pca.fit(X_train)
    
    print(f"PCA time {time.time() - start_pca:.2f}s. Variance: {np.sum(pca.explained_variance_ratio_):.4f}")
    return pca

# RESULTS
def results(model, train_history, val_history, test_loader, pca):

    if not os.path.exists('results'):
        os.makedirs('results')

    plt.figure(figsize=(10, 5))
    plt.plot(train_history, label='Training Loss')
    plt.plot(val_history, label='Validation Loss')
    plt.title('Training vs Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True)
    
    save_path_loss = 'results/loss_curve.png'
    plt.savefig(save_path_loss)
    print(f"Loss saved to {save_path_loss}")
    plt.show()
    # Final metrics
    criterion = nn.MSELoss()
    total_test_loss = 0.0
    total_ssim = 0.0 
    n_samples = 0
    
    model.eval()
    
    start_test_time = time.time()
    
    with torch.no_grad():
        for data in test_loader:
            images, _ = data
            images = images.to(Config.DEVICE)
            #SSIM
            batch_size = images.size(0)
            n_samples += batch_size

            #Noise
            if Config.NOISE_FACTOR > 0:
                noise = torch.randn_like(images) * Config.NOISE_FACTOR
                model_input = images + noise
                model_input = torch.clamp(model_input, 0., 1.)
            else:
                model_input = images
            
            
            outputs = model(model_input)

            loss = criterion(outputs, images)
            total_test_loss += loss.item()
            
            # --------------FOR SSIM CALCULATION--------------
            clean_imgs_np = images.cpu().permute(0, 2, 3, 1).numpy()
            output_imgs_np = outputs.cpu().permute(0, 2, 3, 1).numpy()
            
            batch_ssim = 0.0
            for i in range(batch_size):
                # data_range=1.0 since images are 0-1. channel_axis=2 for (H, W, C)
                try:
                    score = ssim(clean_imgs_np[i], output_imgs_np[i], data_range=1.0, channel_axis=2)
                except TypeError:
                    # Fallback for older scikit-image versions
                    score = ssim(clean_imgs_np[i], output_imgs_np[i], data_range=1.0, multichannel=True)
                batch_ssim += score
            
            total_ssim += batch_ssim
            #--------------------------------------------------------
            
    avg_test_loss = total_test_loss / len(test_loader)
    # --- SSIM AVG ---
    avg_ssim = total_ssim / n_samples
    # ----------------------------------------
    
    total_test_time = time.time() - start_test_time
    
    pca_mse_total = 0.0
    n_batches = 0
    
    for data in test_loader:
        images, _ = data

        if Config.NOISE_FACTOR > 0:
            noise = torch.randn_like(images) * Config.NOISE_FACTOR
            noisy_images = images + noise
            noisy_images = torch.clamp(noisy_images, 0., 1.)
            flat_input = noisy_images.view(images.size(0), -1).numpy()
        else:
            flat_input = images.view(images.size(0), -1).numpy()

        flat_target = images.view(images.size(0), -1).numpy()
        
        pca_encoded = pca.transform(flat_input)
        pca_decoded = pca.inverse_transform(pca_encoded)
        
        batch_mse = mean_squared_error(flat_target, pca_decoded)
        pca_mse_total += batch_mse
        n_batches += 1

    avg_pca_mse = pca_mse_total / n_batches
    
    print(f"Final Test Loss (Autoencoder): {avg_test_loss:.6f}")
    print(f"Final Test SSIM (Autoencoder): {avg_ssim:.4f}")
    print(f"Final Test Loss (PCA): {avg_pca_mse:.6f}")
    print(f"Evaluation Time: {total_test_time:.4f}s")

    test_images, _ = next(iter(test_loader))
    test_images = test_images.to(Config.DEVICE)
    

    noisy_test_images = test_images.clone()
    if Config.NOISE_FACTOR > 0:
        noise = torch.randn_like(test_images) * Config.NOISE_FACTOR
        noisy_test_images = test_images + noise
        noisy_test_images = torch.clamp(noisy_test_images, 0., 1.)
    
    with torch.no_grad():
        ae_outputs = model(noisy_test_images)

    
    
    flat_noisy = noisy_test_images.cpu().view(noisy_test_images.size(0), -1).numpy()
    pca_encoded = pca.transform(flat_noisy) # Feed NOISY images to PCA
    pca_decoded = pca.inverse_transform(pca_encoded)
    pca_outputs = torch.tensor(pca_decoded).view(-1, 3, 32, 32)
    
    test_images = test_images.cpu()
    noisy_test_images = noisy_test_images.cpu()
    ae_outputs = ae_outputs.cpu()
    
    n_samples = 5
    # Increased rows to 4 to include Noisy Pictures
    # One for-loop for noise on for no noise
    fig, axes = plt.subplots(4, n_samples, figsize=(12, 10))
    '''
    for i in range(n_samples):
        # Original
        axes[0, i].imshow(test_images[i].permute(1, 2, 0))
        axes[0, i].axis('off')
        if i == 2: axes[0, i].set_title("Original Images", fontsize=12, fontweight='bold')

        # Autoencoder
        axes[1, i].imshow(ae_outputs[i].permute(1, 2, 0))
        axes[1, i].axis('off')
        if i == 2: axes[1, i].set_title(f"AE Denoised (MSE: {avg_test_loss:.4f}, SSIM: {avg_ssim:.3f})", fontsize=12, fontweight='bold')
        # PCA
        pca_img = pca_outputs[i].permute(1, 2, 0).numpy()
        pca_img = np.clip(pca_img, 0, 1)
        
        axes[2, i].imshow(pca_img)
        axes[2, i].axis('off')
        if i == 2: axes[2, i].set_title(f"PCA (MSE: {avg_pca_mse:.4f})", fontsize=12, fontweight='bold')
    '''
    for i in range(n_samples):
        # Row 1: Noisy Input
        axes[0, i].imshow(noisy_test_images[i].permute(1, 2, 0))
        axes[0, i].axis('off')
        if i == 2: axes[0, i].set_title(f"Noisy Input (Factor {Config.NOISE_FACTOR})", fontsize=12, fontweight='bold')

        # Row 2: Autoencoder Reconstruction
        axes[1, i].imshow(ae_outputs[i].permute(1, 2, 0))
        axes[1, i].axis('off')
        if i == 2: axes[1, i].set_title(f"AE Denoised (MSE: {avg_test_loss:.4f}, SSIM: {avg_ssim:.3f})", fontsize=12, fontweight='bold')
        
        # Row 3: PCA Reconstruction
        pca_img = np.clip(pca_outputs[i].permute(1, 2, 0).numpy(), 0, 1)
        axes[2, i].imshow(pca_img)
        axes[2, i].axis('off')
        if i == 2: axes[2, i].set_title(f"PCA Denoised (MSE: {avg_pca_mse:.4f})", fontsize=12, fontweight='bold')

        # Row 4: Original Clean Image
        axes[3, i].imshow(test_images[i].permute(1, 2, 0))
        axes[3, i].axis('off')
        if i == 2: axes[3, i].set_title("Original Clean Target", fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    save_path_recon = 'results/reconstruction_comparison.png'
    plt.savefig(save_path_recon)
    print(f"Comparison saved to {save_path_recon}")
    plt.show()
    
if __name__ == "__main__":
    
    trained_model, t_hist, v_hist, train_set, test_set = train_model()
    
    trained_pca = run_pca_comparison(train_set, test_set)
    
    results(trained_model, t_hist, v_hist, test_set, trained_pca)