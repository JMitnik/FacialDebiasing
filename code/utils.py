import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data.sampler import SequentialSampler
from torch.utils.data import ConcatDataset, DataLoader
import torchvision.transforms as transforms
from torchvision.utils import make_grid
import gc
from collections import Counter

from setup import config
from dataset import sample_dataset, sample_idxs_from_loader, sample_idxs_from_loaders

def calculate_accuracy(labels, pred):
    """Calculates accuracy given labels and predictions
    """
    return float(((pred > 0) == (labels > 0)).sum()) / labels.size()[0]

def remove_frame_from_plot(plt):
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

def get_best_and_worst_predictions(labels, pred):
    n_rows = 4
    n_samples = n_rows**2

    print("face percentage:", float(labels.sum().item())/len(labels))
    indices = torch.tensor([i for i in range(len(labels))]).long().to(config.device)

    faceslice = labels == 1
    faces,       other       = pred[faceslice],    pred[~faceslice]
    faces_index, other_index = indices[faceslice], indices[~faceslice]

    worst_faces = faces_index[faces.argsort()[:n_samples]]
    best_faces = faces_index[faces.argsort(descending=True)[:n_samples]]

    worst_other = other_index[other.argsort(descending=True)[:n_samples]]
    best_other = other_index[other.argsort()[:n_samples]]

    return best_faces, worst_faces, best_other, worst_other


def debug_memory():
    tensors = Counter(
        (str(o.device), o.dtype, tuple(o.shape))
        for o in gc.get_objects() if torch.is_tensor(o)
    )

    for line in tensors.items():
        print('{}\t{}'.format(*line))

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

def print_reconstruction(model, data, epoch, n_rows=4):
    model.eval()
    n_samples = n_rows**2

    images = sample_dataset(data, n_samples).to(config.device)

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

    fig.savefig('results/{}/reconstructions/epoch={}'.format(config.run_folder, epoch), bbox_inches='tight')

    plt.close()

def concat_batches(batch_a, batch_b):
    # TODO: Merge by interleaving the batches
    images = torch.cat((batch_a[0], batch_b[0]), 0)
    labels = torch.cat((batch_a[1], batch_b[1]), 0)
    idxs = torch.cat((batch_a[2], batch_b[2]), 0)

    return images, labels, idxs

def visualize_best_and_worst(data_loaders, all_labels, all_indices, epoch, best_faces, worst_faces, best_other, worst_other, n_rows=4):
    n_samples = n_rows**2

    fig=plt.figure(figsize=(16, 16))

    sub_titles = ["Best faces", "Worst faces", "Best non-faces", "Worst non-faces"]
    for i, indices in enumerate((best_faces, worst_faces, best_other, worst_other)):
        labels, indices = all_labels[indices], all_indices[indices]

        images = sample_idxs_from_loaders(indices, data_loaders, labels[0])

        ax = fig.add_subplot(2, 2, i+1)
        grid = make_grid(images.reshape(n_samples,3,64,64), n_rows)
        plt.imshow(grid.permute(1,2,0).cpu())
        ax.set_title(sub_titles[i], fontdict={"fontsize":30})

        remove_frame(plt)


    fig.savefig('results/{}/best_and_worst/epoch:{}'.format(config.run_folder,epoch), bbox_inches='tight')

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

    fig.savefig('results/{}/bias_probs/epoch={}'.format(config.run_folder, epoch), bbox_inches='tight')
    plt.close()
