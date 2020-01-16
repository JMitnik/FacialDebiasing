"""
In this file the training of the network is done
"""

import torch
import torch.functional as F
import numpy as np
import datetime

import vae_model
import argparse
from setup import config
from torch.utils.data import ConcatDataset, DataLoader
from datasets import train_and_valid_loaders, sample_dataset, sample_idxs_from_loader

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

def visualize_best_and_worst(data_loader, all_labels, all_indeces, epoch, best_faces, worst_faces, best_other, worst_other):
    n_rows = 4
    n_samples = n_rows**2
    
    fig=plt.figure(figsize=(16, 16))

    sub_titles = ["Best faces", "Worst faces", "Best non-faces", "Worst non-faces"]
    for i, indeces in enumerate([best_faces, worst_faces, best_other, worst_other]):
        labels = all_labels[indeces]
        indeces = all_indeces[indeces]

        images = sample_idxs_from_loader(indeces, data_loader, labels[0])

        ax = fig.add_subplot(2, 2, i+1)
        grid = make_grid(images.reshape(n_samples,3,64,64), n_rows)
        plt.imshow(grid.permute(1,2,0).cpu())
        ax.set_title(sub_titles[i], fontdict={"fontsize":30})

        remove_frame(plt)


    fig.savefig('results/{}/best_and_worst/epoch:{}'.format(FOLDER_NAME,epoch), bbox_inches='tight')

    plt.close()
    
    return

def debug_memory():
    tensors = Counter(
        (str(o.device), o.dtype, tuple(o.shape))
        for o in gc.get_objects() if torch.is_tensor(o)
    )

    for line in tensors.items():
        print('{}\t{}'.format(*line))

def print_reconstruction(model, data, epoch):
    model.eval()
    ######### RECON IMAGES ###########
    n_rows = 4
    n_samples = n_rows**2

    images = sample_dataset(data, n_samples).to(DEVICE)

    fig=plt.figure(figsize=(16, 8))

    recon_images = model.recon_images(images)

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
    return

def update_histogram(model, data_loader, epoch):
    all_labels = torch.LongTensor([]).to(DEVICE)
    all_index = torch.LongTensor([]).to(DEVICE)

    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            images, labels, index = batch
            
            #TEMPORARY TAKE ONLY FACES
            slicer = labels == 1
            images, labels, index = images[slicer], labels[slicer], index[slicer]


            batch_size = labels.size(0)

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            index = index.to(DEVICE)

            all_labels = torch.cat((all_labels, labels))
            all_index = torch.cat((all_index, index))
            model.build_histo(images)

        base = model.get_histo_base()
        our = model.get_histo()
        difference = base-our

    n_rows = 3
    n_samples = n_rows**2

    best_base = base.argsort(descending=True)[:n_samples]
    best_our = our.argsort(descending=True)[:n_samples]
    worst_base = base.argsort()[:n_samples]
    worst_our = our.argsort()[:n_samples]

    print("best => base:{}, our:{}".format(best_base, best_our))

    print(all_index[best_base])
    print(all_labels[best_base])

    best_img_base = sample_idxs_from_loader(all_index[best_base], data_loader, 1)
    best_img_our = sample_idxs_from_loader(all_index[best_our], data_loader, 1)
    worst_img_base = sample_idxs_from_loader(all_index[worst_base], data_loader, 1)
    worst_img_our = sample_idxs_from_loader(all_index[worst_our], data_loader, 1)

    img_list = (best_img_base, best_img_our, worst_img_base, worst_img_our)
    titles = ("best base", "best our", "worst base", "worst our")
    fig=plt.figure(figsize=(16, 16))

    for i in range(4):
        ax = fig.add_subplot(2, 2, i+1)
        grid = make_grid(img_list[i].reshape(n_samples,3,64,64), n_rows)
        plt.imshow(grid.permute(1,2,0).cpu())
        ax.set_title(titles[i], fontdict={"fontsize":30})

        remove_frame(plt)

    fig.savefig('results/{}/base_vs_our/epoch={}'.format(FOLDER_NAME, epoch), bbox_inches='tight')
    print("DONE WITH UPDATE")

    plt.close()
    return


def train_epoch(model, data_loader, optimizer):
    """
    train the model for one epoch
    """

    model.train()
    avg_loss = 0
    avg_acc = 0

    print("START TRAINING")
    for i, batch in enumerate(data_loader):
        images, labels, index = batch
        batch_size = labels.size(0)

        # print("TRAIN => BATCH SIZE:", batch_size)

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

        # print_best_and_worst(images, labels, pred, 123)

        if i % ARGS.eval_freq == 0:
            print("batch:{} accuracy:{}".format(i, acc))

    # debug_memory()

    return avg_loss/(i+1), avg_acc/(i+1)


def eval_epoch(model, data_loader, epoch):
    """
    Calculates the validation error of the model
    """

    model.eval()
    avg_loss = 0
    avg_acc = 0

    all_labels = torch.LongTensor([]).to(DEVICE)
    all_preds = torch.Tensor([]).to(DEVICE)
    all_indeces = torch.LongTensor([]).to(DEVICE)

    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            images, labels, index = batch
            batch_size = labels.size(0)

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            index = index.to(DEVICE)
            pred, loss = model.forward(images, labels)

            loss = loss/batch_size
            acc = calculate_accuracy(labels, pred)

            avg_loss += loss.item()
            avg_acc += acc

            all_labels = torch.cat((all_labels, labels))
            all_preds = torch.cat((all_preds, pred))
            all_indeces = torch.cat((all_indeces, index))
        

    print("length of all eval:", len(all_labels))
    # best_faces, worst_faces, best_other, worst_other = get_best_and_worst(all_labels, all_preds)

    # visualize_best_and_worst(data_loader, all_labels, all_indeces, epoch, best_faces, worst_faces, best_other, worst_other)
    return avg_loss/(i+1), avg_acc/(i+1)

def main():
    # import data
    train_loader, valid_loader, train_data, valid_data = train_and_valid_loaders(batch_size=ARGS.batch_size,
                                                                                 train_size=0.8, max_images=ARGS.dataset_size)

    # create model
    model = vae_model.Db_vae(z_dim=ARGS.zdim, device=DEVICE).to(DEVICE)

    # create optimizer
    optimizer = torch.optim.Adam(model.parameters())

    for epoch in range(ARGS.epochs):
        print("Starting epoch:{}/{}".format(epoch, ARGS.epochs))
        update_histogram(model, train_loader, epoch)
        train_loss, train_acc = train_epoch(model, train_loader, optimizer)
        print("training done")
        val_loss, val_acc = eval_epoch(model, valid_loader, epoch)

        print("epoch {}/{}, train_loss={:.2f}, train_acc={:.2f}, val_loss={:.2f}, val_acc={:.2f}".format(epoch,
                                    ARGS.epochs, train_loss, train_acc, val_loss, val_acc))

        print_reconstruction(model, valid_data, epoch)

        with open("results/"+FOLDER_NAME + "/training_results.csv", "a") as write_file:
            s = "{},{},{},{},{}\n".format(epoch, train_loss, val_loss, train_acc, val_acc)
            print("S:", s)
            write_file.write(s)

        torch.save(model.state_dict(), "results/"+FOLDER_NAME+"/model.pt".format(epoch))
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
    parser.add_argument('--dataset_size', default=-1, type=int,
                        help='total size of database')
    parser.add_argument('--eval_freq', default=5, type=int,
                        help='total size of database')


    ARGS = parser.parse_args()

    print("ARGS => batch_size:{}, epochs:{}, z_dim:{}, alpha:{}, dataset_size:{}, eval_freq:{}".format(ARGS.batch_size, 
                                ARGS.epochs, ARGS.zdim, ARGS.alpha, ARGS.dataset_size, ARGS.eval_freq))

    FOLDER_NAME = "results_zdim_{}_alpha_{}_batch_size_{}_size_{}_time_{}".format(ARGS.zdim, str(ARGS.alpha)[2:], 
                                ARGS.batch_size, ARGS.dataset_size, datetime.datetime.now())

    os.makedirs("results/"+ FOLDER_NAME + '/best_and_worst')
    os.makedirs("results/"+ FOLDER_NAME + '/base_vs_our')
    os.makedirs("results/"+ FOLDER_NAME + '/reconstructions')

    with open("results/" + FOLDER_NAME + "/training_results.csv", "w") as write_file:
        write_file.write("epoch, train_loss, valid_loss, train_acc, valid_acc\n")
    

    main()
