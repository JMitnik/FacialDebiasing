"""
In this file the training of the network is done
"""

import torch
import vae_model
import argparse
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEVICE = 'cpu'
print("device:", DEVICE)
ARGS = None

def train_epoch(model, data, optimizer):
    """
    train the model for one epoch
    """

    # traindata, _ = data

    model.train()

    train_error = 100

    return train_error

def eval_epoch(model, data):
    """
    Calculates the validation error of the model
    """

    # _, valdata = data

    model.eval()
    
    val_error = 200

    return val_error

def main():
    # import data
    data = torch.ones((1,64,64))

    # create model
    model = vae_model.Db_vae(z_dim=ARGS.zdim).to(DEVICE)

    # create optimizer
    optimizer = torch.optim.Adam(model.parameters())

    for epoch in range(ARGS.epochs):
        model.forward(data)
        train_error = train_epoch(model, data, optimizer)
        val_error = eval_epoch(model, data)

        print("epoch {}/{}, train_error={}, test_error={}".format(epoch, 
                                    ARGS.epochs, train_error, val_error))

    return 

if __name__ == "__main__":
    print("start training")

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=40, type=int,
                        help='max number of epochs')
    parser.add_argument('--zdim', default=20, type=int,
                        help='dimensionality of latent space')
    parser.add_argument('--alpha', default=0, type=int,
                        help='importance of debiasing')

    ARGS = parser.parse_args()

    main()