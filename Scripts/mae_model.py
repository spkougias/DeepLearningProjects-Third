import torch
import torch.nn as nn
import numpy as np
from config import Config

# ---------------------------------------------------------
# 1. Patch Embedding Layer
#    Takes an image, chops it up, and projects it to vectors
# ---------------------------------------------------------
class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        # Calculate size of a flattened patch (4*4*3 = 48)
        self.patch_dim = in_channels * patch_size * patch_size

        # We use a Convolution to do the splitting and projecting in one step.
        # Kernel Size = Stride = Patch Size ensures non-overlapping patches.
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x shape: [Batch, 3, 32, 32]
        x = self.proj(x) 
        # x new shape: [Batch, Embed_Dim, 8, 8] (if patch=4)
        
        # Flatten the spatial dimensions (8x8 -> 64)
        x = x.flatten(2) 
        # x new shape: [Batch, Embed_Dim, 64]
        
        # Swap dimensions to fit Transformer standard [Batch, Seq_Len, Dim]
        x = x.transpose(1, 2)
        # Final shape: [Batch, 64, Embed_Dim]
        return x

# ---------------------------------------------------------
# 2. The MAE Transformer
# ---------------------------------------------------------
class MaskedAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        # --- ENCODER setup ---
        self.patch_embed = PatchEmbedding(
            Config.IMAGE_SIZE, Config.PATCH_SIZE, Config.INPUT_CHANNELS, Config.EMBED_DIM
        )
        
        # Positional Embedding: Learnable vectors added to patches to give them "location"
        self.cls_token = nn.Parameter(torch.zeros(1, 1, Config.EMBED_DIM))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + Config.NUM_PATCHES, Config.EMBED_DIM))
        
        # PyTorch's built-in Transformer Encoder Layer
        enc_layer = nn.TransformerEncoderLayer(
            d_model=Config.EMBED_DIM, 
            nhead=Config.ENC_HEADS, 
            dim_feedforward=Config.ENC_DIM_FEEDFORWARD, 
            activation="gelu", 
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=Config.ENC_LAYERS)
        self.enc_norm = nn.LayerNorm(Config.EMBED_DIM)

        # --- DECODER setup ---
        # We need to project encoder output to decoder dimension (if they differ)
        # Here we assume they are the same for simplicity, but strictly they can differ.
        self.decoder_embed = nn.Linear(Config.EMBED_DIM, Config.EMBED_DIM, bias=True)
        
        self.mask_token = nn.Parameter(torch.zeros(1, 1, Config.EMBED_DIM))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, 1 + Config.NUM_PATCHES, Config.EMBED_DIM))
        
        dec_layer = nn.TransformerEncoderLayer(
            d_model=Config.EMBED_DIM, 
            nhead=Config.DEC_HEADS, 
            dim_feedforward=Config.DEC_DIM_FEEDFORWARD, 
            activation="gelu", 
            batch_first=True,
            norm_first=True
        )
        self.decoder = nn.TransformerEncoder(dec_layer, num_layers=Config.DEC_LAYERS)
        self.dec_norm = nn.LayerNorm(Config.EMBED_DIM)
        
        # Prediction Head: Projects back to pixel space (Patch_Dim = 48)
        self.pred_head = nn.Linear(Config.EMBED_DIM, self.patch_embed.patch_dim, bias=True)

        # Initialize weights (Xavier/GLOROT)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # --- MASKING LOGIC (The Hard Part) ---
    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        x: [Batch, L, D], sequence
        """
        N, L, D = x.shape  # Batch, Length (64), Dim
        len_keep = int(L * (1 - mask_ratio))
        
        # Create random noise to sort by
        noise = torch.rand(N, L, device=x.device)
        
        # sort noise to get indices
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        
        # Use torch.gather to pick the patches we want to keep
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        
        # Generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # Unshuffle to get the mask in original order
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        # 1. Embed patches
        x = self.patch_embed(x)
        
        # 2. Add positional embedding (exclude CLS for now)
        x = x + self.pos_embed[:, 1:, :]

        # 3. Masking
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # 4. Append CLS token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # 5. Apply Transformer Encoder
        x = self.encoder(x)
        x = self.enc_norm(x)
        
        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # 1. Embed decoder input
        x = self.decoder_embed(x)

        # 2. Append Mask Tokens to placeholders
        # Since we removed patches, we need to put "Mask Tokens" back in the gaps
        # so the decoder can predict them.
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        
        # We separate CLS token from patches
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        
        # Unshuffle to put patches back in original positions
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2])) 
        
        # Re-append CLS token
        x = torch.cat([x[:, :1, :], x_], dim=1)

        # 3. Add Decoder Positional Embedding
        x = x + self.decoder_pos_embed

        # 4. Apply Transformer Decoder
        x = self.decoder(x)
        x = self.dec_norm(x)

        # 5. Predict pixels
        x = self.pred_head(x)
        
        # Remove CLS token from output (we only care about pixels)
        x = x[:, 1:, :]
        return x

    def forward(self, imgs, mask_ratio=0.75):
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)
        return pred, mask