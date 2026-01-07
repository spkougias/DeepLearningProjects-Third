import torch

class Config:
    # Hyperparameters
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-3
    
    NOISE_FACTOR = 0.1
    
    NUM_EPOCHS = 200
    VALIDATION_SPLIT = 0.1
    
    INPUT_DIM = 3072
    HIDDEN_DIM = 256
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Device: {Config.DEVICE}")