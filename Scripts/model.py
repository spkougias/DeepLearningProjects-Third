import torch
import torch.nn as nn

class LinearAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(LinearAutoencoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Flatten(),
            
            # Layer 1
            nn.Linear(input_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            
            # Layer 2
            nn.Linear(2048, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        

        self.decoder = nn.Sequential(

            nn.Linear(hidden_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),

            nn.Linear(2048, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded.view(-1, 3, 32, 32)
 

class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        
        self.encoder = nn.Sequential(

            nn.Conv2d(3, 16, 3, padding=1, stride=2), 
            nn.BatchNorm2d(16),
            nn.ReLU(),
            

            nn.Conv2d(16, 32, 3, padding=1, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        

        self.decoder = nn.Sequential(

            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            
            nn.ConvTranspose2d(16, 3, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)

        return self.decoder(encoded)
