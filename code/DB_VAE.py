#
# Here the structure of the network is made in pytorch
#

import torch.nn as nn

class Encoder(nn.Module):

    def __init__(self, hidden_dim=500, z_dim=20):
        super().__init__()

    def forward(self, input):
        """
        Perform forward pass of encoder.
        """

        return


class Decoder(nn.Module):

    def __init__(self, hidden_dim=500, z_dim=20):
        super().__init__()

    def forward(self, input):
        """
        Perform forward pass of encoder.
        """

        return


class DB_VAE(nn.Module):

    def __init__(self, hidden_dim=500, z_dim=20):
        super().__init__()

    def forward(self, input):
        """
        Given input, perform an encoding and decoding step and return the
        negative average elbo for the given batch.
        """
      return

    def sample(self, n_samples, z_samples=[]):
        """
        Sample n_samples from the model. Return both the sampled images
        (from bernoulli) and the means for these bernoullis (as these are
        used to plot the data manifold).
        """

        return 
