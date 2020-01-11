"""
In this file the training of the network is done
"""

import torch
import torch.functional as F
import vae_model2 as vae_model
import argparse
from setup import config
from torch.utils.data import ConcatDataset, DataLoader
from dataset import train_and_valid_loaders

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# DEVICE = 'cpu'
print("device:", DEVICE)
ARGS = None

def train_epoch(model, data_loader, optimizer):
    """
    train the model for one epoch
    """

    # traindata, _ = data

    model.train()
    avg_loss = 0
    avg_class_loss = 0

    for i, batch in enumerate(data_loader):
        images, labels = batch
        batch_size = labels.size(0)


        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        pred, loss, class_loss = model.forward(images, labels)

        optimizer.zero_grad()
        loss = loss/batch_size

        avg_loss += loss.item()
        avg_class_loss += class_loss/batch_size

        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=5)
        optimizer.step()
    
    print("Train.py => loss:{}, class_loss:{}".format(avg_loss/i, avg_class_loss/i))

    return avg_loss/i

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
    train_loader, valid_loader = train_and_valid_loaders(batch_size=ARGS.batch_size, train_size=0.8)

    # create model
    model = vae_model.Db_vae(z_dim=ARGS.zdim, device=DEVICE).to(DEVICE)

    # create optimizer
    optimizer = torch.optim.Adam(model.parameters())

    for epoch in range(ARGS.epochs):
        print("Starting epoch:{}/{}".format(epoch, ARGS.epochs))
        train_error = train_epoch(model, train_loader, optimizer)
        val_error = eval_epoch(model, valid_loader)

        # print("epoch {}/{}, train_error={}, validation_error={}".format(epoch, 
                                    # ARGS.epochs, train_error, val_error))

    return 

if __name__ == "__main__":
    print("start training")

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=128, type=int,
                        help='size of batch')
    parser.add_argument('--epochs', default=10, type=int,
                        help='max number of epochs')
    parser.add_argument('--zdim', default=200, type=int,
                        help='dimensionality of latent space')
    parser.add_argument('--alpha', default=0, type=int,
                        help='importance of debiasing')

    ARGS = parser.parse_args()

    main()