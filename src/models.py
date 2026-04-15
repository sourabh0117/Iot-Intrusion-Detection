# src/models.py
import torch
import torch.nn as nn

# Import the configuration settings
from src import config

class Autoencoder(nn.Module):
    """
    A PyTorch Autoencoder model with a distinct encoder and decoder.
    The architecture is defined by parameters in the config file.
    """
    def __init__(self):
        super(Autoencoder, self).__init__()
        
        # --- Encoder ---
        # The encoder compresses the input data into a lower-dimensional representation.
        # nn.Sequential is a container for modules that will be added to it in order.
        self.encoder = nn.Sequential(
            nn.Linear(config.INPUT_DIM, 32),
            nn.ReLU(),  # ReLU activation introduces non-linearity
            nn.Linear(32, config.ENCODING_DIM),
            nn.ReLU()
        )
        
        # --- Decoder ---
        # The decoder attempts to reconstruct the original input from the compressed representation.
        self.decoder = nn.Sequential(
            nn.Linear(config.ENCODING_DIM, 32),
            nn.ReLU(),
            nn.Linear(32, config.INPUT_DIM),
            nn.Sigmoid()  # Sigmoid activation scales the output to be between 0 and 1,
                          # matching our MinMaxScaler-transformed input data.
        )

    def forward(self, x):
        """
        Defines the forward pass of the model.
        
        Args:
            x (torch.Tensor): The input tensor.
            
        Returns:
            torch.Tensor: The reconstructed output tensor.
        """
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded