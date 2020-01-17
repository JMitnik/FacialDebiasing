"""
Here the structure of the network is made in pytorch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


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
            nn.LeakyReLU(),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 128, kernel_size=5, stride=2),
            nn.LeakyReLU(),
            nn.BatchNorm2d(128),

            nn.Conv2d(128, 256, kernel_size=5, stride=2),
            nn.LeakyReLU(),
            nn.BatchNorm2d(256),

            nn.Conv2d(256, 512, kernel_size=5, stride=2),
            nn.LeakyReLU(),
            nn.BatchNorm2d(512),
            nn.Flatten(),

            nn.Linear(512, 1000),
            nn.LeakyReLU(),

            nn.Linear(1000, z_dim*2+1)
        )


    def forward(self, input):
        """
        Perform forward pass of encoder.
        """
        
        out = self.layers(input)
        
        # return classification, mean and log_std
        return out[:, 0], out[:, 1:self.z_dim+1], F.softplus(out[:,self.z_dim+1:])


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
            nn.LeakyReLU(),
            nn.Linear(1000, 512*1*1),
            UnFlatten(512, 1),

            nn.ConvTranspose2d(512, 256, kernel_size=5, stride=2),
            nn.LeakyReLU(),
            nn.BatchNorm2d(256),

            nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2),
            nn.LeakyReLU(),
            nn.BatchNorm2d(128),

            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, output_padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64),

            nn.ConvTranspose2d(64, 3, kernel_size=5, stride=2, output_padding=1),
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

        self.target_dist = torch.distributions.normal.Normal(0, 1)

        self.c1 = 1
        self.c2 = 1
        self.c3 = 1

        self.num_bins = 1000
        self.min_val = -15
        self.max_val = 15
        self.hist = torch.ones((z_dim, self.num_bins))
        self.means = torch.Tensor().to(self.device)
        
        self.alpha = 0.01


    def forward(self, images, labels):
        """
        Given images, perform an encoding and decoding step and return the
        negative average elbo for the given batch.
        """
        pred, mean, std = self.encoder(images)
        # TODO:
        # Acc of the classfication should be added properly - Requires some 
        # extra target input

        loss_class = F.binary_cross_entropy_with_logits(pred, labels.float(), reduction='sum')

        slice_indices = labels == 1

        if labels[slice_indices].size(0) > 0:
            # Slice the face images from the batch
            face_images = images[slice_indices]
            face_mean = mean[slice_indices]
            face_std = std[slice_indices]
        
            # Get single samples from the distributions with reparametrisation trick
            dist = torch.distributions.normal.Normal(face_mean, face_std)
            z = dist.rsample().to(self.device)

            res = self.decoder(z)

            # TODO:
            # Change losses of VAE part only towards those of the actual faces.
            # Also shouldnt feed those to the decoder, waste of time
            
            # calculate VAE losses
            # loss_recon = F.l1_loss(res, face_images, reduction='sum')
            loss_recon = torch.abs(face_images - res).sum()


            loss_kl = torch.distributions.kl.kl_divergence(dist, self.target_dist)
            loss_kl = loss_kl.sum()

            # calculate total loss
            loss_total = self.c1 * loss_class + self.c2 * loss_recon + self.c3 * loss_kl

        else:
            # OPTIONAL: 
            # multiply by c1
            loss_total = loss_class * self.c1

        return pred, loss_total

    def build_histo(self, input):
        """
            Creates histos or samples Qs from it
            NOTE:
            Make sure you only put faces into this
            functions
        """

        samples_per_dist = 10000
        
        _, mean, log_std = self.encoder(input)

        self.means = torch.cat((self.means, mean))

        dist = torch.distributions.normal.Normal(mean, log_std)
        z = dist.rsample((samples_per_dist,)).to(self.device)
        # NOTE those samples are added to the first axis!

        self.hist += torch.stack([torch.histc(z[:, :, i], 
                                  min=self.min_mu, 
                                  max=self.max_mu, 
                                  bins=self.num_bins) for i in range(self.z_dim)])
        
        return

    def get_histo(self):
        """
            Returns the probabilities given the means given the histo values
        """

        # print(self.means.shape)
        weights = torch.Tensor().to(DEVICE)
        for mu in self.means:
            # Gets probability for each 
            newhist = torch.stack([torch.histc(i, 
                                  min=self.min_val, 
                                  max=self.max_val, 
                                  bins=self.num_bins) for i in mu])
            newhist *= self.hist
            newhist = newhist.sum(dim=1)
            weights = torch.cat((weights, ((1 / (newhist + self.alpha)).prod())))
            
        # Reset values
        self.hist = torch.ones((self.z_dim, self.num_bins))
        self.means = torch.Tensor().to(self.device)
        return weights

    def recon_images(self, images):
        with torch.no_grad():
            pred, mean, std = self.encoder(images)
            
            # Get single samples from the distributions with reparametrisation trick
            dist = torch.distributions.normal.Normal(mean, std)
            z = dist.rsample().to(self.device)

            recon_images = self.decoder(z)

        # return predictions and the loss
        return recon_images

    def sample(self, n_samples, z_samples=[]):
        """
        Sample n_samples from the model. Return both the sampled images
        (from bernoulli) and the means for these bernoullis (as these are
        used to plot the data manifold).
        """

        return 
