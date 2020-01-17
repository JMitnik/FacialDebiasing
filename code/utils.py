import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data.sampler import SequentialSampler
from torch.utils.data import ConcatDataset, DataLoader
from torchvision.utils import make_grid
import gc
from collections import Counter

from setup import config
from datasets import DataLoaderTuple, concat_datasets, train_and_valid_loaders, sample_dataset, sample_idxs_from_loader, make_hist_loader

def calculate_accuracy(labels, pred):
    """Calculates accuracy given labels and predictions

    Arguments:
        labels {[type]} -- [description]
        pred {[type]} -- [description]

    Returns:
        [type] -- [description]
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

def get_best_and_worst(labels, pred):
    n_rows = 4
    n_samples = n_rows**2

    labels = labels.float().cpu()
    print("face procentage:", float(labels.sum().item())/len(labels))

    pred = pred.cpu()

    faces_max = torch.tensor([x for x in pred])
    faces_min = torch.tensor([x for x in pred])
    other_max = torch.tensor([x for x in pred])
    other_min = torch.tensor([x for x in pred])

    faces_max[labels == 0] = np.inf
    faces_min[labels == 0] = -np.inf
    other_max[labels == 1] = np.inf
    other_min[labels == 1] = -np.inf

    worst_faces = faces_max.argsort()[:n_samples]
    best_faces = faces_min.argsort(descending=True)[:n_samples]

    worst_other = other_min.argsort(descending=True)[:n_samples]
    best_other = other_max.argsort()[:n_samples]

    return best_faces, worst_faces, best_other, worst_other

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

    images = sample_dataset(data, n_samples).to(config.device)

    fig=plt.figure(figsize=(16, 8))

    recon_images = model.recon_images(images)

    fig.add_subplot(1, 2, 1)
    grid = make_grid(images.reshape(n_samples,3,64,64), n_rows)
    plt.imshow(grid.permute(1,2,0).cpu())

    remove_frame_from_plot(plt)

    fig.add_subplot(1, 2, 2)
    grid = make_grid(recon_images.reshape(n_samples,3,64,64), n_rows)
    plt.imshow(grid.permute(1,2,0).cpu())

    remove_frame_from_plot(plt)

    fig.savefig('images/{}/reconstructions/epoch={}'.format(config.run_folder, epoch), bbox_inches='tight')

    plt.close()


def concat_batches(batch_a, batch_b):
    # TODO: Merge by interleaving the batches
    images = torch.cat((batch_a[0], batch_b[0]), 0)
    labels = torch.cat((batch_a[1], batch_b[1]), 0)
    idxs = torch.cat((batch_a[2], batch_b[2]), 0)

    return images, labels, idxs

def visualize_best_and_worst(data_loader, all_labels, all_idxs, epoch, best_faces, worst_faces, best_other, worst_other):
    n_rows = 4
    n_samples = n_rows**2

    fig = plt.figure(figsize=(16, 16))

    sub_titles = ["Best faces", "Worst faces", "Best non-faces", "Worst non-faces"]
    for i, idxs in enumerate([best_faces, worst_faces, best_other, worst_other]):
        labels = all_labels[idxs]
        idxs = all_idxs[idxs]

        images = sample_idxs_from_loader(idxs, data_loader, labels[0])

        ax = fig.add_subplot(2, 2, i+1)
        grid = make_grid(images.reshape(n_samples,3,64,64), n_rows)
        plt.imshow(grid.permute(1,2,0).cpu())
        ax.set_title(sub_titles[i], fontdict={"fontsize":30})

        remove_frame_from_plot(plt)

    fig.savefig('images/{}/best_and_worst/epoch:{}'.format(config.run_folder, epoch), bbox_inches='tight')
    plt.close()

    return
