"""
In this file the training of the network is done
"""

import torch
import torch.functional as F
import numpy as np
from typing import Tuple
from torch.utils.data.sampler import SequentialSampler
import datetime

import vae_model
import argparse
from setup import config
from torch.utils.data import ConcatDataset, DataLoader
from datasets import DataLoaderTuple, concat_datasets, train_and_valid_loaders, sample_dataset, sample_idxs_from_loaders, sample_idxs_from_loader, make_hist_loader

from torchvision.utils import make_grid
import matplotlib.pyplot as plt

import gc
from collections import Counter
import os

FOLDER_NAME = ""

if not os.path.exists("results"):
    os.makedirs("results")

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# DEVICE = 'cpu'
print("device:", DEVICE)
ARGS = None

def calculate_accuracy(labels, pred):
    return float(((pred > 0) == (labels > 0)).sum()) / labels.size()[0]

def remove_frame(plt):
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

def get_best_and_worst(labels, pred):
    n_rows = 4
    n_samples = n_rows**2

    labels = labels.float().cpu()
    print("face procentage:", float(labels.sum().item())/len(labels))

    pred = pred.cpu()

    faces_max = torch.Tensor([x for x in pred])
    faces_min = torch.Tensor([x for x in pred])
    other_max = torch.Tensor([x for x in pred])
    other_min = torch.Tensor([x for x in pred])

    faces_max[labels == 0] = np.inf
    faces_min[labels == 0] = -np.inf
    other_max[labels == 1] = np.inf
    other_min[labels == 1] = -np.inf

    worst_faces = faces_max.argsort()[:n_samples]
    best_faces = faces_min.argsort(descending=True)[:n_samples]

    worst_other = other_min.argsort(descending=True)[:n_samples]
    best_other = other_max.argsort()[:n_samples]

    return best_faces, worst_faces, best_other, worst_other

def visualize_best_and_worst(data_loaders, all_labels, all_indeces, epoch, best_faces, worst_faces, best_other, worst_other, n_rows=4):
    n_samples = n_rows**2

    fig=plt.figure(figsize=(16, 16))

    sub_titles = ["Best faces", "Worst faces", "Best non-faces", "Worst non-faces"]
    for i, indeces in enumerate((best_faces, worst_faces, best_other, worst_other)):
        labels, indeces = all_labels[indeces], all_indeces[indeces]

        images = sample_idxs_from_loaders(indeces, data_loaders, labels[0])

        ax = fig.add_subplot(2, 2, i+1)
        grid = make_grid(images.reshape(n_samples,3,64,64), n_rows)
        plt.imshow(grid.permute(1,2,0).cpu())
        ax.set_title(sub_titles[i], fontdict={"fontsize":30})

        remove_frame(plt)


    fig.savefig('results/{}/best_and_worst/epoch:{}'.format(FOLDER_NAME,epoch), bbox_inches='tight')

    plt.close()

def debug_memory():
    tensors = Counter(
        (str(o.device), o.dtype, tuple(o.shape))
        for o in gc.get_objects() if torch.is_tensor(o)
    )

    for line in tensors.items():
        print('{}\t{}'.format(*line))

def print_reconstruction(model, data, epoch, n_rows=4):
    model.eval()
    n_samples = n_rows**2

    images = sample_dataset(data, n_samples).to(DEVICE)

    recon_images = model.recon_images(images)

    fig=plt.figure(figsize=(16, 8))

    fig.add_subplot(1, 2, 1)
    grid = make_grid(images.reshape(n_samples,3,64,64), n_rows)
    plt.imshow(grid.permute(1,2,0).cpu())

    remove_frame(plt)

    fig.add_subplot(1, 2, 2)
    grid = make_grid(recon_images.reshape(n_samples,3,64,64), n_rows)
    plt.imshow(grid.permute(1,2,0).cpu())

    remove_frame(plt)

    fig.savefig('results/{}/reconstructions/epoch={}'.format(FOLDER_NAME, epoch), bbox_inches='tight')

    plt.close()

def visualize_bias(probs, data_loader, all_labels, all_index, epoch, n_rows=3):
    n_samples = n_rows**2

    highest_probs = probs.argsort(descending=True)[:n_samples]
    lowest_probs = probs.argsort()[:n_samples]

    highest_imgs = sample_idxs_from_loader(all_index[highest_probs], data_loader, 1)
    worst_imgs = sample_idxs_from_loader(all_index[lowest_probs], data_loader, 1)

    img_list = (highest_imgs, worst_imgs)
    titles = ("Highest", "Lowest")
    fig = plt.figure(figsize=(16, 16))

    for i in range(2):
        ax = fig.add_subplot(1, 2, i+1)
        grid = make_grid(img_list[i].reshape(n_samples,3,64,64), n_rows)
        plt.imshow(grid.permute(1,2,0).cpu())
        ax.set_title(titles[i], fontdict={"fontsize":30})

        remove_frame(plt)

    fig.savefig('results/{}/bias_probs/epoch={}'.format(FOLDER_NAME, epoch), bbox_inches='tight')
    plt.close()

def concat_batches(batch_a, batch_b):
    # TODO: Merge by interleaving the batches
    images = torch.cat((batch_a[0], batch_b[0]), 0)
    labels = torch.cat((batch_a[1], batch_b[1]), 0)
    idxs = torch.cat((batch_a[2], batch_b[2]), 0)

    return images, labels, idxs

def update_histogram(model, data_loader, epoch):

    # reset the means and histograms
    model.hist = torch.ones((ARGS.zdim, model.num_bins)).to(DEVICE)
    model.means = torch.Tensor().to(DEVICE)

    all_labels = torch.LongTensor([]).to(DEVICE)
    all_index = torch.LongTensor([]).to(DEVICE)

    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            images, labels, index = batch

            #TEMPORARY TAKE ONLY FACES
            images, labels, index = images.to(DEVICE), labels.to(DEVICE), index.to(DEVICE)
            batch_size = labels.size(0)

            all_labels = torch.cat((all_labels, labels))
            all_index = torch.cat((all_index, index))
            if ARGS.debias_type == "base":
                model.build_means(images)
            elif ARGS.debias_type == "our":
                model.build_histo(images)

        if ARGS.debias_type == "base":
            probs = model.get_histo_base()
        elif ARGS.debias_type == "our":
            probs = model.get_histo_our()

        if probs is None:
            raise Exception("No coorrect debias method given. choos \"base\" or \"our\"")

    visualize_bias(probs, data_loader, all_labels, all_index, epoch)

    return probs

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

    # TODO: divide the batch-size of the loader over both face_Batch and nonface_batch, rather than doubling the batch-size
    for i, (face_batch, nonface_batch) in enumerate(zip(face_loader, nonface_loader)):
        images, labels, idxs = concat_batches(face_batch, nonface_batch)
        batch_size = labels.size(0)

        images, labels = images.to(DEVICE), labels.to(DEVICE)
        pred, loss = model.forward(images, labels)

        optimizer.zero_grad()
        loss = loss/batch_size

        # calculate the gradients and clip them at 5
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)

        optimizer.step()

        acc = calculate_accuracy(labels, pred)
        avg_loss += loss.item()
        avg_acc += acc

        if i % ARGS.eval_freq == 0:
            print("batch:{} accuracy:{}".format(i, acc))

    # debug_memory()

    return avg_loss/(i+1), avg_acc/(i+1)

def eval_epoch(model, data_loaders: DataLoaderTuple, epoch):
    """
    Calculates the validation error of the model
    """
    face_loader, nonface_loader = data_loaders

    model.eval()
    avg_loss = 0
    avg_acc = 0

    all_labels = torch.LongTensor([]).to(DEVICE)
    all_preds = torch.Tensor([]).to(DEVICE)
    all_idxs = torch.LongTensor([]).to(DEVICE)

    with torch.no_grad():
        for i, (face_batch, nonface_batch) in enumerate(zip(face_loader, nonface_loader)):
            images, labels, idxs = concat_batches(face_batch, nonface_batch)
            batch_size = labels.size(0)

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            idxs = idxs.to(DEVICE)
            pred, loss = model.forward(images, labels)

            loss = loss/batch_size
            acc = calculate_accuracy(labels, pred)

            avg_loss += loss.item()
            avg_acc += acc

            all_labels = torch.cat((all_labels, labels))
            all_preds = torch.cat((all_preds, pred))
            all_idxs = torch.cat((all_idxs, idxs))


    best_faces, worst_faces, best_other, worst_other = get_best_and_worst(all_labels, all_preds)
    visualize_best_and_worst(data_loaders, all_labels, all_idxs, epoch, best_faces, worst_faces, best_other, worst_other)

    return avg_loss/(i+1), avg_acc/(i+1)

def main():
    train_loaders: DataLoaderTuple
    valid_loaders: DataLoaderTuple

    train_loaders, valid_loaders = train_and_valid_loaders(
        batch_size=ARGS.batch_size,
        train_size=0.8,
        max_images=ARGS.dataset_size
    )

    # Initialize model
    model = vae_model.Db_vae(z_dim=ARGS.zdim, device=DEVICE).to(DEVICE)

    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters())

    for epoch in range(ARGS.epochs):
        # Generic sequential dataloader to sample histogram

        print("Starting epoch:{}/{}".format(epoch, ARGS.epochs))

        if ARGS.debias_type != 'none':
            hist_loader = make_hist_loader(train_loaders.faces.dataset, ARGS.batch_size)
            hist = update_histogram(model, hist_loader, epoch)

            train_loaders.faces.sampler.weights = hist

        train_loss, train_acc = train_epoch(model, train_loaders, optimizer)
        print("training done")
        val_loss, val_acc = eval_epoch(model, valid_loaders, epoch)

        print("epoch {}/{}, train_loss={:.2f}, train_acc={:.2f}, val_loss={:.2f}, val_acc={:.2f}".format(epoch+1,
                                    ARGS.epochs, train_loss, train_acc, val_loss, val_acc))

        valid_data = concat_datasets(valid_loaders.faces.dataset, valid_loaders.nonfaces.dataset, proportion_a=0.5)
        print(valid_data)
        print_reconstruction(model, valid_data, epoch)

        with open("results/"+FOLDER_NAME + "/training_results.csv", "a") as write_file:
            s = "{},{},{},{},{}\n".format(epoch, train_loss, val_loss, train_acc, val_acc)
            print("S:", s)
            write_file.write(s)

        torch.save(model.state_dict(), "results/"+FOLDER_NAME+"/model.pt".format(epoch))

if __name__ == "__main__":
    print("start training")

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=256, type=int,
                        help='size of batch')
    parser.add_argument('--epochs', default=10, type=int,
                        help='max number of epochs')
    parser.add_argument('--zdim', default=200, type=int,
                        help='dimensionality of latent space')
    parser.add_argument('--alpha', default=0.0, type=float,
                        help='importance of debiasing')
    parser.add_argument('--dataset_size', default=-1, type=int,
                        help='total size of database')
    parser.add_argument('--eval_freq', default=5, type=int,
                        help='total size of database')
    parser.add_argument('--debias_type', default='none', type=str,
                        help='type of debiasing used')


    ARGS = parser.parse_args()

    print("ARGS => batch_size:{}, epochs:{}, z_dim:{}, alpha:{}, dataset_size:{}, eval_freq:{}, debiasing type:{}".format(ARGS.batch_size,
                                ARGS.epochs, ARGS.zdim, ARGS.alpha, ARGS.dataset_size, ARGS.eval_freq, ARGS.debias_type))

    FOLDER_NAME = "{}".format(datetime.datetime.now().strftime("%m/%d/%Y---%H_%M_%S"))

    os.makedirs("results/"+ FOLDER_NAME + '/best_and_worst')
    os.makedirs("results/"+ FOLDER_NAME + '/bias_probs')
    os.makedirs("results/"+ FOLDER_NAME + '/reconstructions')

    with open("results/" + FOLDER_NAME + "/flags.txt", "w") as write_file:
        write_file.write(f"zdim = {ARGS.zdim}\n")
        write_file.write(f"alpha = {ARGS.alpha}\n")
        write_file.write(f"epochs = {ARGS.epochs}\n")
        write_file.write(f"batch size = {ARGS.batch_size}\n")
        write_file.write(f"eval frequency = {ARGS.eval_freq}\n")
        write_file.write(f"dataset size = {ARGS.dataset_size}\n")
        write_file.write(f"debiasing type = {ARGS.debias_type}\n")

    with open("results/" + FOLDER_NAME + "/training_results.csv", "w") as write_file:
        write_file.write("epoch,train_loss,valid_loss,train_acc,valid_acc\n")


    main()
