"""
In this file the training of the network is done
"""
# Import internal
from typing import Tuple

# Import external
import torch
import torch.functional as F
import numpy as np
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

# Import project-based
import vae_model
import utils
import setup
from setup import config

from torch.utils.data import ConcatDataset, DataLoader
from dataset import concat_datasets, make_train_and_valid_loaders, sample_dataset, sample_idxs_from_loader, make_hist_loader
from datasets.generic import DataLoaderTuple

def update_histogram(model, data_loader, epoch):
    all_labels = torch.tensor([], dtype=torch.long).to(config.device)
    all_index = torch.tensor([], dtype=torch.long).to(config.device)

    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            images, labels, index = batch
            batch_size = labels.size(0)

            images = images.to(config.device)
            labels = labels.to(config.device)
            index = index.to(config.device)

            all_labels = torch.cat((all_labels, labels))
            all_index = torch.cat((all_index, index))
            model.build_histo(images)

        base = model.get_histo_base()
        our = model.get_histo()
        difference = base-our

        # print("Base version:")
        # print(base)
        # print('Our version:')
        # print(our)
        # print('diff:')
        # print(difference)

    n_rows = 3
    n_samples = n_rows**2

    highest_base = base.argsort(descending=True)[:n_samples]
    highest_our = our.argsort(descending=True)[:n_samples]

    print("highest => base:{}, our:{}".format(highest_base, highest_our))

    print(all_index[highest_base])
    print(all_labels[highest_base])

    img_base = sample_idxs_from_loader(all_index[highest_base], data_loader, 1)
    img_our = sample_idxs_from_loader(all_index[highest_our], data_loader, 1)

    fig=plt.figure(figsize=(16, 8))

    ax = fig.add_subplot(1, 2, 1)
    grid = make_grid(img_base.reshape(n_samples,3,64,64), n_rows)
    plt.imshow(grid.permute(1,2,0).cpu())
    ax.set_title("base", fontdict={"fontsize":30})

    utils.remove_frame_from_plot(plt)

    ax = fig.add_subplot(1, 2, 2)
    grid = make_grid(img_our.reshape(n_samples,3,64,64), n_rows)
    plt.imshow(grid.permute(1,2,0).cpu())
    ax.set_title("our", fontdict={"fontsize":30})

    utils.remove_frame_from_plot(plt)

    fig.savefig('images/{}/base_vs_our/epoch={}'.format(config.run_folder, epoch), bbox_inches='tight')
    print("DONE WITH UPDATE")

    plt.close()
    return base

def train_epoch(model, data_loaders: DataLoaderTuple, optimizer):
    """
    train the model for one epoch
    """

    face_loader, nonface_loader = data_loaders

    model.train()
    avg_loss = 0
    avg_acc = 0

    # The batches contain Image(rgb x w x h), Labels (1 for 0), original dataset indices
    face_batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    nonface_batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]

    count: int = 0

    # TODO: divide the batch-size of the loader over both face_Batch and nonface_batch, rather than doubling the batch-size
    for i, (face_batch, nonface_batch) in enumerate(zip(face_loader, nonface_loader)):
        images, labels, idxs = utils.concat_batches(face_batch, nonface_batch)
        print("START TRAINING")
        batch_size = labels.size(0)

        images = images.to(config.device)
        labels = labels.to(config.device)
        pred, loss = model.forward(images, labels)

        optimizer.zero_grad()
        loss = loss/batch_size
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)

        optimizer.step()

        acc = utils.calculate_accuracy(labels, pred)
        avg_loss += loss.item()
        avg_acc += acc

        if ARGS and i % config.eval_freq == 0:
            print("batch:{} accuracy:{}".format(i, acc))

        count = i

    return avg_loss/(count+1), avg_acc/(count+1)

def eval_epoch(model, data_loaders: DataLoaderTuple, epoch):
    """
    Calculates the validation error of the model
    """
    face_loader, nonface_loader = data_loaders

    model.eval()
    avg_loss = 0
    avg_acc = 0

    all_labels = torch.tensor([], dtype=torch.long).to(config.device)
    all_preds = torch.tensor([]).to(config.device)
    all_idxs = torch.tensor([], dtype=torch.long).to(config.device)

    count = 0

    with torch.no_grad():
        for i, (face_batch, nonface_batch) in enumerate(zip(face_loader, nonface_loader)):
            images, labels, idxs = utils.concat_batches(face_batch, nonface_batch)
            batch_size = labels.size(0)

            images = images.to(config.device)
            labels = labels.to(config.device)
            idxs = idxs.to(config.device)
            pred, loss = model.forward(images, labels)

            loss = loss/batch_size
            acc = utils.calculate_accuracy(labels, pred)

            avg_loss += loss.item()
            avg_acc += acc

            all_labels = torch.cat((all_labels, labels))
            all_preds = torch.cat((all_preds, pred))
            all_idxs = torch.cat((all_idxs, idxs))

            count = i

    print(f"Length of all evals: {all_labels.shape[0]}")
    # best_faces, worst_faces, best_other, worst_other = utils.get_best_and_worst(all_labels, all_preds)
    # utils.visualize_best_and_worst(data_loader, all_labels, all_idxs, epoch, best_faces, worst_faces, best_other, worst_other)
    return avg_loss/(count+1), avg_acc/(count+1)

def main():
    train_loaders: DataLoaderTuple
    valid_loaders: DataLoaderTuple

    train_loaders, valid_loaders = train_and_valid_loaders(
        batch_size=config.batch_size,
        max_images=config.dataset_size
    )

    # Initialize model
    model = vae_model.Db_vae(z_dim=config.zdim, device=config.device).to(config.device)

    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters())

    for epoch in range(config.epochs):
        # Generic sequential dataloader to sample histogram
        hist_loader = make_hist_loader(train_loaders.faces.dataset, config.batch_size)
        hist = update_histogram(model, hist_loader, epoch)
        train_loaders.faces.sampler.weights = hist

        print("Starting epoch:{}/{}".format(epoch, config.epochs))
        train_error, train_acc = train_epoch(model, train_loaders, optimizer)
        print("training done")
        val_error, val_acc = eval_epoch(model, valid_loaders, epoch)

        print("epoch {}/{}, train_error={:.2f}, train_acc={:.2f}, val_error={:.2f}, val_acc={:.2f}".format(epoch,
                                    config.epochs, train_error, train_acc, val_error, val_acc))

        valid_data = concat_datasets(*valid_loaders, proportion_a=0.5)
        utils.print_reconstruction(model, valid_data, epoch)
    return

if __name__ == "__main__":
    print("Start training")

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
    parser.add_argument('--eval_freq', default=5, type=int,
                        help='total size of database')

    ARGS = parser.parse_args()

    main()
