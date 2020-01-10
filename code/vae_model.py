"""
Here the structure of the network is made in pytorch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """
    Encodes the data using a CNN

    Input => 64x64 image
    Output => mean vector z_dim
              log_std vector z_dim
              predicted value
    """

    def __init__(self, z_dim=20):
        super().__init__()

        self.z_dim = z_dim

        self.layers = nn.Sequential(   
            nn.Conv2d(3, 64, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 128, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(128),

            nn.Conv2d(128, 256, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(256),

            nn.Conv2d(256, 512, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.Flatten(),

            nn.Linear(512*4*4, 1000),
            nn.ReLU(),

            nn.Linear(1000, z_dim*2+1)
        )

    def forward(self, input):
        """
        Perform forward pass of encoder.
        """
        
        out = self.layers(input)

        print(out.size())
        
        # return classification, mean and log_std
        return out[:, 0], out[:, 1:self.z_dim+1], out[:,self.z_dim+1:]


class UnFlatten(nn.Module):
    def __init__(self, channel_size, image_size):
        super(UnFlatten, self).__init__()
        self.channel_size = channel_size
        self.image_size = image_size

    def forward(self, input):
        return input.view(-1, self.channel_size, self.image_size, self.image_size)

class Decoder(nn.Module):
    """
    Encodes the data using a CNN

    Input => sample vector z_dim
    Output => 64x64 image

    4 6 13 29 61
    """

    def __init__(self, z_dim=20):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(z_dim, 1000),
            nn.ReLU(),
            nn.Linear(1000, 512*4*4),
            UnFlatten(512, 4),

            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),

            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(128),

            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, output_padding = 1),
            nn.ReLU(),
            nn.BatchNorm2d(64),

            nn.ConvTranspose2d(64, 3, kernel_size=5, stride=2, output_padding = 1),
            nn.Sigmoid()
        )

    def forward(self, input):
        """
        Perform forward pass of encoder.
        """

        out = self.layers(input)


        return out


class Db_vae(nn.Module):

    def __init__(self, hidden_dim=500, z_dim=20, device="cpu"):
        super().__init__()

        self.device = device
        self.z_dim = z_dim

        self.encoder = Encoder(z_dim)
        self.decoder = Decoder(z_dim)

    def forward(self, input):
        """
        Given input, perform an encoding and decoding step and return the
        negative average elbo for the given batch.
        """
        pred, mean, log_std = self.encoder(input)

        std = torch.exp(log_std).to(self.device)

        epsilon = torch.randn(self.z_dim).to(self.device)

        z = mean + epsilon * std

        res = self.decoder(z)

        print(torch.min(res), torch.max(res))
        
        loss_recon = F.binary_cross_entropy(res, input, reduction='sum')

        D_kl = 0.5 * torch.sum(std**2 + mean**2 - 2*log_std - 1)

        print("loss_recon:", loss_recon)
        print("KL:", D_kl)
        return pred, loss_recon + D_kl

    def sample(self, n_samples, z_samples=[]):
        """
        Sample n_samples from the model. Return both the sampled images
        (from bernoulli) and the means for these bernoullis (as these are
        used to plot the data manifold).
        """

        return 
