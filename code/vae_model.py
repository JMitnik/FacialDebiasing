"""
Here the structure of the network is made in pytorch
"""

from typing import List, Union, Optional
import torch
import os
from logger import logger
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.stats import norm

class Encoder(nn.Module):
    """
    Encodes the data using a CNN

    Input => 64x64 image
    Output => mean vector z_dim
              log_std vector z_dim
              predicted value
    """

    def __init__(self, z_dim=20, custom_layers: Optional[nn.Sequential] = None):
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

    def __init__(self, z_dim=20, custom_layers: Optional[nn.Sequential] = None):
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

    def __init__(
        self,
        z_dim=20,
        hist_size=1000,
        alpha=0.01,
        num_bins=10,
        device="cpu",
        custom_encoding_layers: Optional[nn.Sequential] = None,
        custom_decoding_layers: Optional[nn.Sequential] = None
    ):
        super().__init__()

        self.device = device
        self.z_dim = z_dim

        self.encoder = Encoder(z_dim, custom_encoding_layers)
        self.decoder = Decoder(z_dim, custom_decoding_layers)

        self.target_dist = torch.distributions.normal.Normal(0, 1)

        self.c1 = 1
        self.c2 = 1
        self.c3 = 0.1

        self.num_bins = num_bins
        # self.num_bins = 500
        self.min_val = -15
        self.max_val = 15
        self.xlin = np.linspace(self.min_val, self.max_val, self.num_bins).reshape(1,1,self.num_bins)
        self.hist = np.zeros((z_dim, self.num_bins))
        # self.hist = torch.ones((z_dim, self.num_bins)).to(self.device)
        self.means = torch.Tensor().to(self.device)
        self.std = torch.Tensor().to(self.device)

        self.alpha = alpha

    @staticmethod
    def init(path_to_model: str, device, z_dim):
        full_path_to_model = f"results/{path_to_model}/model.pt"
        if not os.path.exists(full_path_to_model):
            logger.error(
                f"Can't find model at {full_path_to_model}",
                next_step="Evaluation will stop",
                tip="Double check your path to model"
            )
            raise Exception

        model: Db_vae = Db_vae(z_dim=z_dim, device=device)
        model.load_state_dict(torch.load(full_path_to_model, map_location=device))

        return model


    def forward(self, images: torch.Tensor, labels: torch.Tensor):
        """
        Given images, perform an encoding and decoding step and return the
        negative average elbo for the given batch.
        """
        pred, mean, std = self.encoder(images)

        loss_class = F.binary_cross_entropy_with_logits(pred, labels.float(), reduction='none')

        # We only want to calculate the loss towards actual faces
        faceslicer = labels == 1
        facemean = mean[faceslicer]
        facestd = std[faceslicer]

        # Get single samples from the distributions with reparametrisation trick
        dist = torch.distributions.normal.Normal(facemean, facestd)
        z = dist.rsample().to(self.device)

        res = self.decoder(z)


        # calculate VAE losses
        loss_recon = (images[faceslicer] - res)**2
        loss_recon = loss_recon.view(loss_recon.shape[0],-1).mean(1)

        loss_kl = torch.distributions.kl.kl_divergence(dist, self.target_dist)
        loss_kl = loss_kl.view(loss_kl.shape[0],-1).mean(1)

        loss_vae = self.c2 * loss_recon + self.c3 * loss_kl
        loss_total = self.c1 * loss_class

        # Only add loss to positions of faces, rest is zero
        zeros = torch.zeros(faceslicer.shape[0]).to(self.device)
        zeros[faceslicer] = loss_vae

        loss_total = loss_total + zeros

        return pred, loss_total

    def forward_eval(self, images):
        """
        Given images, perform an encoding and decoding step and return the
        negative average elbo for the given batch.
        """
        with torch.no_grad():
            pred, _,_ = self.encoder(images)

        return pred


    def interpolate(self, images, amount):
        with torch.no_grad():
            _, mean, std = self.encoder(images)

            mean_1, std_1 = mean[0,:], std[0,:]
            mean_2, std_2 = mean[1,:], std[1,:]

            all_mean  = torch.tensor([]).to(self.device)
            all_std = torch.tensor([]).to(self.device)

            diff_mean = mean_1 - mean_2
            diff_std = std_1 = std_2

            steps_mean = diff_mean / (amount-1)
            steps_std = diff_std / (amount-1)

            for i in range(amount):
                all_mean = torch.cat((all_mean, mean_1 - steps_mean*i))
                all_std = torch.cat((all_std, std_1 - steps_std*i))

            all_mean = all_mean.view(amount, -1)
            all_std = all_std.view(amount, -1)
            dist = torch.distributions.normal.Normal(all_mean, all_std)
            z = dist.rsample().to(self.device)

            recon_images = self.decoder(z)

        return recon_images

    def build_means(self, input):
        _, mean, log_std = self.encoder(input)

        self.means = torch.cat((self.means, mean))

        return


    def build_histo(self, input):
        """
            Creates histos or samples Qs from it
            NOTE:
            Make sure you only put faces into this
            functions
        """

        # samples_per_dist = 1000

        _, mean, std = self.encoder(input)

        self.means = torch.cat((self.means, mean))
        self.std = torch.cat((self.std, std))
        
        values = norm.pdf(self.xlin, mean.unsqueeze(-1).cpu(), std.unsqueeze(-1).cpu()).sum(0)
        self.hist += values
        # dist = torch.distributions.normal.Normal(mean, std)
        # z = dist.rsample((samples_per_dist,)).to(self.device)
        # NOTE those samples are added to the first axis!

        # self.hist += torch.stack([torch.histc(z[:, :, i],
        #                           min=self.min_val,
        #                           max=self.max_val,
        #                           bins=self.num_bins) for i in range(self.z_dim)])

        return

    def get_histo_base(self):
        probs = torch.zeros_like(self.means[:,0]).to(self.device)

        for i in range(self.z_dim):
            dist = self.means[:,i].cpu().numpy()

            hist, bins = np.histogram(dist, density=True, bins=self.num_bins)

            bins[0] = -float('inf')
            bins[-1] = float('inf')
            bin_idx = np.digitize(dist, bins)

            hist = hist + self.alpha
            hist /= np.sum(hist)

            p = 1.0/(hist[bin_idx-1])
            p /= np.sum(p)

            probs = torch.max(probs, torch.Tensor(p).to(self.device))

        probs /= probs.sum()

        return probs

    def get_histo_max5(self):
        probs = torch.zeros_like(self.means, dtype=float).to(self.device)

        for i in range(self.z_dim):
            dist = self.means[:,i].cpu().numpy()

            hist, bins = np.histogram(dist, density=True, bins=self.num_bins)

            bins[0] = -float('inf')
            bins[-1] = float('inf')
            bin_idx = np.digitize(dist, bins)

            hist = hist + self.alpha
            hist /= np.sum(hist)

            p = 1.0/(hist[bin_idx-1])
            p /= np.sum(p)

            probs[:,i] = torch.Tensor(p).to(self.device)

        probs = probs.sort(1, descending=True)[0][:,:5]
        probs = probs.prod(1)

        print(probs)
        return probs

    def get_histo_our(self):
        """
            Returns the probabilities given the means given the histo values
        """
        results = np.empty(self.means.shape[0])
        hist_batch_size = 4000
        # Iterate in large batches over dataset to prevent memory lockup
        for i in range(0, self.means.shape[0], hist_batch_size):
            i_end = i  + hist_batch_size
            if i_end > self.means.shape[0]:
                i_end = self.means.shape[0]
            mean = self.means[i:i_end, :]
            std = self.std[i:i_end, :]


            lins = norm.pdf(self.xlin, mean.unsqueeze(-1).cpu(), std.unsqueeze(-1).cpu())
            Q = lins * self.hist
            Q = Q.sum(-1)
            W = 1 / (Q + self.alpha)
            # Performing the max value technique, TODO: analyse top 5
            results[i:i_end] = W.max(-1)

        # # Reset values
        self.hist.fill(0)
        self.means = torch.Tensor().to(self.device)
        self.std = torch.Tensor().to(self.device)
        return torch.tensor(results).to(self.device)

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
