"""
In this file the training of the network is done
"""

import torch
import torch.functional as F
import vae_model
import argparse
from setup import config
from torch.utils.data import ConcatDataset, DataLoader
from dataset import train_and_valid_loaders

from torchvision.utils import make_grid
import matplotlib.pyplot as plt

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# DEVICE = 'cpu'
print("device:", DEVICE)
ARGS = None

def calculate_accuracy(labels, pred):
    pred[pred > 0] = 1
    pred[pred <= 0] = 0
    pred = pred.long()

    correct_pred = labels == pred

    return float(correct_pred.sum().item())/labels.size()[0]

def calculate_scores(labels, pred):
    return


def print_reconstruction(model, images, epoch):
    model.eval()
    ######### RECON IMAGES ###########
    n_rows = 4
    n_samples = n_rows**2

    fig=plt.figure(figsize=(16, 8))

    recon_images = model.recon_images(images[:n_samples])

    fig.add_subplot(1, 2, 1)
    grid = make_grid(images[:n_samples].reshape(n_samples,3,64,64), n_rows)
    plt.imshow(grid.permute(1,2,0).cpu())

    ########## REMOVE FRAME ##########
    frame = plt.gca()
    for xlabel_i in frame.axes.get_xticklabels():
        xlabel_i.set_visible(False)
        xlabel_i.set_fontsize(0.0)
    for xlabel_i in frame.axes.get_yticklabels():
        xlabel_i.set_fontsize(0.0)
        xlabel_i.set_visible(False)
    for tick in frame.axes.get_xticklines():
        tick.set_visible(False)
    for tick in frame.axes.get_yticklines():
        tick.set_visible(False)

    fig.add_subplot(1, 2, 2)
    grid = make_grid(recon_images.reshape(n_samples,3,64,64), n_rows)
    plt.imshow(grid.permute(1,2,0).cpu())
    
    ########## REMOVE FRAME ##########
    frame = plt.gca()
    for xlabel_i in frame.axes.get_xticklabels():
        xlabel_i.set_visible(False)
        xlabel_i.set_fontsize(0.0)
    for xlabel_i in frame.axes.get_yticklabels():
        xlabel_i.set_fontsize(0.0)
        xlabel_i.set_visible(False)
    for tick in frame.axes.get_xticklines():
        tick.set_visible(False)
    for tick in frame.axes.get_yticklines():
        tick.set_visible(False)

    fig.savefig('images/training_epoch={}'.format(epoch), bbox_inches='tight')

    return 

def train_epoch(model, data_loader, optimizer):
    """
    train the model for one epoch
    """

    model.train()
    avg_loss = 0
    avg_acc = 0

    for i, batch in enumerate(data_loader):
        images, labels = batch
        batch_size = labels.size(0)


        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        pred, loss = model.forward(images, labels)

        optimizer.zero_grad()
        loss = loss/batch_size

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
        optimizer.step()

        acc = calculate_accuracy(labels, pred)

        avg_loss += loss.item()
        avg_acc += acc

    return avg_loss/(i+1), avg_acc/(i+1)

def eval_epoch(model, data_loader):
    """
    Calculates the validation error of the model
    """

    model.eval()
    
    avg_loss = 0
    avg_acc = 0

    for i, batch in enumerate(data_loader):
        images, labels = batch
        batch_size = labels.size(0)

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        pred, loss = model.forward(images, labels)

        loss = loss/batch_size
        acc = calculate_accuracy(labels, pred)

        avg_loss += loss
        avg_acc += acc

    model.train()
    return avg_loss/(i+1), avg_acc/(i+1)

def main():
    # import data
    train_loader, valid_loader = train_and_valid_loaders(batch_size=ARGS.batch_size, train_size=0.8)

    # create model
    model = vae_model.Db_vae(z_dim=ARGS.zdim, device=DEVICE).to(DEVICE)

    # create optimizer
    optimizer = torch.optim.Adam(model.parameters())

    for epoch in range(ARGS.epochs):
        print("Starting epoch:{}/{}".format(epoch, ARGS.epochs))
        train_error, train_acc = train_epoch(model, train_loader, optimizer)
        val_error, val_acc = eval_epoch(model, valid_loader)

        print("epoch {}/{}, train_error={:.2f}, train_acc={:.2f}, val_error={:.2f}, val_acc={:.2f}".format(epoch, 
                                    ARGS.epochs, train_error, train_acc, val_error, val_acc))

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
    parser.add_argument('--alpha', default=0.0, type=float,
                        help='importance of debiasing')
    parser.add_argument('--dataset_size', default=10000, type=int,
                        help='total size of database')

    ARGS = parser.parse_args()

    main()