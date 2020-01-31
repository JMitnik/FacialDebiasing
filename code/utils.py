from typing import List
from logger import logger
from datasets.data_utils import DatasetOutput
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data.sampler import SequentialSampler
from torch.utils.data import ConcatDataset, DataLoader
import torchvision.transforms as transforms
from torchvision.utils import make_grid
import gc
from collections import Counter
from PIL import Image

from dataset import sample_dataset, sample_idxs_from_loader, sample_idxs_from_loaders

def calculate_accuracy(labels, pred):
    """Calculates accuracy given labels and predictions."""
    return float(((pred > 0) == (labels > 0)).sum()) / labels.size()[0]

def get_best_and_worst_predictions(labels, pred, device):
    """Returns indices of the best and worst predicted faces."""
    n_rows = 4
    n_samples = n_rows**2

    logger.info(f"Face percentage: {float(labels.sum().item())/len(labels)}")
    indices = torch.tensor([i for i in range(len(labels))]).long().to(device)

    faceslice = labels == 1
    faces,       other       = pred[faceslice],    pred[~faceslice]
    faces_index, other_index = indices[faceslice], indices[~faceslice]

    worst_faces = faces_index[faces.argsort()[:n_samples]]
    best_faces = faces_index[faces.argsort(descending=True)[:n_samples]]

    worst_other = other_index[other.argsort(descending=True)[:n_samples]]
    best_other = other_index[other.argsort()[:n_samples]]

    return best_faces, worst_faces, best_other, worst_other

def calculate_places(name_list, setups, w, s):
    """Calculates the places in the final barplot."""
    x_axis = np.arange(len(setups))
    counter = len(name_list)-1

    if (len(name_list) % 2) == 0:
        places = []
        times = 0
        while counter > 0:
            places.append(x_axis-(s/2)-s*times)
            places.append(x_axis+(s/2)+s*times)

            times += 1
            counter -= 2

    else:
        places = [x_axis]
        times = 1
        while counter > 0:
            places.append(x_axis-s*times)
            places.append(x_axis+s*times)

            times += 1
            counter -= 2

    return x_axis, sorted(places, key = lambda sub: (sub[0], sub[0]))


def make_bar_plot(df, name_list, setups, colors=None, training_type=None, y_label="",
                      title="", y_lim=None, y_ticks=None):
    """Writes a bar plot for the final evaluation, based on the dataframe which stems from a results.csv."""
    if training_type == None:
        training_type = name_list
    if colors == None:
        colors = np.random.rand(len(name_list), 3)

    s = 0.8/len(name_list)
    w = s-0.02

    x_axis, places = calculate_places(name_list, setups, w, s)


    _ = plt.figure(figsize=(16, 6))
    ax = plt.subplot(111)
    for i in range(len(name_list)):
        ax.bar(places[i], df.loc[df["name"].str.contains(name_list[i]), setups].mean(), label=training_type[i],
                          yerr=df.loc[df["name"].str.contains(name_list[i]), setups].std(),
                          color=colors[i], width=w, edgecolor="black", linewidth=2,capsize=10)


    plt.ylabel(y_label, fontdict={"fontsize":20})
    plt.xticks(x_axis, setups, fontsize=25)

    if y_lim != None:
        plt.ylim(y_lim[0], y_lim[1])

    if y_ticks != None:
        plt.yticks(y_ticks, fontsize=20)

    plt.title(title)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10), ncol=5, frameon=False, prop={'size': 19})
    plt.show()

def make_box_plot(df, name_list, training_type=None, colors=None, y_label="", title="", y_lim=None):
    """Writes a box plot for the final evaluation, based on the dataframe which stems from a results.csv."""
    if training_type == None:
        training_type = [""] + name_list

    fig = plt.figure(figsize=(16, 6))

    box_plot_data=[df.loc[df["name"].str.contains(name),:]['var'] for name in name_list]
    box = plt.boxplot(box_plot_data, patch_artist=True)


    if y_lim != None:
        plt.ylim(y_lim[0], y_lim[1])

    plt.xticks(range(len(training_type)), training_type, fontsize=12)

    if colors != None:
        for patch, color in zip(box['boxes'], colors):
            patch.set_facecolor(color)

    plt.ylabel(y_label, fontsize=20)
    plt.show()


def remove_frame(plt):
    """Removes frames from a pyplot plot. """
    # TODO: Add annotation
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

def concat_batches(batch_a: DatasetOutput, batch_b: DatasetOutput):
    """Concatenates two batches of data of shape image x label x idx."""
    images: torch.Tensor = torch.cat((batch_a.image, batch_b.image), 0)
    labels: torch.Tensor = torch.cat((batch_a.label, batch_b.label), 0)
    idxs: torch.Tensor = torch.cat((batch_a.idx, batch_b.idx), 0)

    return images, labels, idxs


def read_image(path_to_image):
    """Reads an image into memory and transform to a tensor."""
    img: Image = Image.open(path_to_image)

    transforms = default_transforms()
    img_tensor: torch.Tensor = transforms(img)

    return img_tensor

def read_flags(path_to_model):
    """"""
    path_to_flags = f"results/{path_to_model}/flags.txt"

    with open(path_to_flags, 'r') as f:
        data = f.readlines()

def find_face_in_subimages(model, sub_images: torch.Tensor, device: str):
    """Finds a face in a tensor of subimages using a models' evaluation method."""
    model.eval()

    for images in sub_images:
        if len(images.shape) == 5:
            images = images.squeeze(dim=0)

        # If one image
        if len(images.shape) == 3:
            images = images.view(1, 3, 64, 64)
        images = images.to(device)
        pred = model.forward_eval(images)

        # If face
        if (pred > 0).any():
            return True

    return False


def default_transforms():
    """Transforms a transform object to a 64 by 64 tensor."""
    return transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

def visualize_tensor(img_tensor: torch.Tensor):
    """Visualizes a image tensor."""
    pil_transformer = transforms.ToPILImage()
    pil_transformer(img_tensor).show()
