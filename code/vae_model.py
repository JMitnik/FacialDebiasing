"""
Here the structure of the network is made in pytorch
"""

import torch.nn as nn
import torch.nn.functional as functional

class Encoder(nn.Module):
    """
    Encodes the data using a CNN

    Input => 64x64 image
    Output => 
    """

    def __init__(self, hidden_dim=500, z_dim=20):
        super().__init__()

        self.layers = nn.Sequential(   
            nn.Conv2d(1, 64, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 128, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(128),

            nn.Conv2d(128, 256, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(256),

            nn.Conv2d(256, 512, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(512),

            nn.Flatten(),
            nn.Linear(512, 1000),
            nn.ReLU(),

            
        )

    def forward(self, input):
        """
        Perform forward pass of encoder.
        """
        
        
        return


class Decoder(nn.Module):
    """
    Encodes the data using a CNN

    Input => 
    Output => 64x64 image
    """

    def __init__(self, hidden_dim=500, z_dim=20):
        super().__init__()

    def forward(self, input):
        """
        Perform forward pass of encoder.
        """

        return


class Db_vae(nn.Module):

    def __init__(self, hidden_dim=500, z_dim=20):
        super().__init__()
        self.encoder = Encoder(hidden_dim, z_dim)

    def forward(self, input):
        """
        Given input, perform an encoding and decoding step and return the
        negative average elbo for the given batch.
        """
        self.encoder(input)
        return

    def sample(self, n_samples, z_samples=[]):
        """
        Sample n_samples from the model. Return both the sampled images
        (from bernoulli) and the means for these bernoullis (as these are
        used to plot the data manifold).
        """

        return 
