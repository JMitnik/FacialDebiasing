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
        return out[:, 0], out[:, 1:self.z_dim+1], torch.exp(out[:,self.z_dim+1:])


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

    def __init__(self, z_dim=20, hist_size=10000, device="cpu"):
        super().__init__()

        self.device = device
        self.z_dim = z_dim

        self.encoder = Encoder(z_dim)
        self.decoder = Decoder(z_dim)

        self.histo = torch.zeros((z_dim, hist_size))

        self.target_dist = torch.distributions.normal.Normal(0, 1)

        self.c1 = 1
        self.c2 = 1
        self.c3 = 1

    def forward(self, images, labels):
        """
        Given images, perform an encoding and decoding step and return the
        negative average elbo for the given batch.
        """
        pred, mean, std = self.encoder(images)

        # TODO:
        # Acc of the classfication should be added properly - Requires some 
        # extra target input
        loss_class = F.cross_entropy_loss(pred, labels, reduction='sum')

        # Slice the face images from the batch
        face_images = images[labels == 1]
        face_mean = mean[labels == 1]
        face_std = std[labels == 1]

        # Get single samples from the distributions with reparametrisation trick
        dist = torch.distributions.normal.Normal(face_mean, face_std)
        z = dist.rsample().to(self.device)

        res = self.decoder(z)

        # TODO:
        # Change losses of VAE part only towards those of the actual faces.
        # Also shouldnt feed those to the decoder, waste of time
        
        # calculate VAE losses
        loss_recon = F.l1_loss(res, face_images, reduction='sum')
        loss_kl = F.kl_div(z, self.target_dist, reduction='sum')

        # calculate total loss
        loss_total = self.c1 * loss_class + self.c2 * loss_recon + self.c3 * loss_kl

        # return predictions and the loss
        return pred, loss_total

    def histo_forward(self, input, build_histo=True):
        """
            Creates histos or samples Qs from it
            NOTE:
            Make sure you only put faces into this
            functions
        """
        _, mean, log_std = self.encoder(input)
        
        # TODO: 
        # Sample from distributions
        # Update self.histo accordingly if build_histo = True
        # else:
        # return the values from the histograms given the means

        return


    def sample(self, n_samples, z_samples=[]):
        """
        Sample n_samples from the model. Return both the sampled images
        (from bernoulli) and the means for these bernoullis (as these are
        used to plot the data manifold).
        """

        return 
