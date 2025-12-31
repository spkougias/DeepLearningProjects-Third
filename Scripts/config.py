# config.py
# Centralized configuration to control the experiment

class Config:
    # --- Data Settings ---
    DATA_PATH = './data'
    BATCH_SIZE = 64
    NUM_WORKERS = 2  # Adjust based on your CPU
    
    # --- Model Architecture (The Transformer) ---
    IMAGE_SIZE = 32
    PATCH_SIZE = 4
    # Calculation: (32/4)^2 = 64 patches
    NUM_PATCHES = (IMAGE_SIZE // PATCH_SIZE) ** 2
    INPUT_CHANNELS = 3
    
    # Embed Dimension: Size of the vector we project patches into
    EMBED_DIM = 192  
    
    # Encoder Settings
    ENC_LAYERS = 6
    ENC_HEADS = 8
    ENC_DIM_FEEDFORWARD = 512
    
    # Decoder Settings (Usually lighter than Encoder)
    DEC_LAYERS = 4
    DEC_HEADS = 8
    DEC_DIM_FEEDFORWARD = 512
    
    # --- Training Settings ---
    EPOCHS = 50       # Transformers take longer to converge than CNNs
    LR = 1e-3         # Learning rate (AdamW usually likes 1e-3 to 1e-4)
    WEIGHT_DECAY = 0.05
    MASK_RATIO = 0.25 # Hide 75% of the image
    
    # Device selection (Auto-detect GPU)
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    def __str__(self):
        return str(self.__class__.__dict__)