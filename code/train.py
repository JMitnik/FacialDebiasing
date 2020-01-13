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

def train_epoch(model, data_loader, optimizer, epoch):
    """
    train the model for one epoch
    """

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
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
        optimizer.step()
    
    accuracy = calculate_accuracy(labels, pred)
    print("Train.py => loss:{}, class_loss:{}, accuracy:{:.2f}".format(avg_loss/i, avg_class_loss/i, accuracy))

    print_reconstruction(model, images, epoch)

    return avg_loss/i

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

    plt.savefig('images/training_epoch={}'.format(epoch), bbox_inches='tight')

    model.train()

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
        train_error = train_epoch(model, train_loader, optimizer, epoch)
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