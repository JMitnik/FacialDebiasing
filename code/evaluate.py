"""
In this file the evaluation of the network is done
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
from dataset import concat_datasets, make_eval_loader, sample_dataset, sample_idxs_from_loader, make_hist_loader
from datasets.generic import DataLoaderTuple

from torchvision.utils import make_grid
import matplotlib.pyplot as plt

import gc
from collections import Counter
import os

FOLDER_NAME = ""

if not os.path.exists("results"):
    os.makedirs("results")

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("device:", DEVICE)

def calculate_accuracy(labels, pred):
    return float(((pred > 0) == (labels > 0)).sum()) / labels.size()[0]

def eval_model(model, data_loader):
    """
    perform evaluation of a single epoch
    """

    model.eval()
    avg_loss = 0

    all_labels = torch.LongTensor([]).to(DEVICE)
    all_preds = torch.Tensor([]).to(DEVICE)
    all_idxs = torch.LongTensor([]).to(DEVICE)

    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            # print(f"batch:{i}")
            images, labels, idxs = batch
            batch_size = labels.size(0)

            # print(f"size:{batch_size}")
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            idxs = idxs.to(DEVICE)
            pred, loss = model.forward(images, labels)

            loss = loss/batch_size


            avg_loss += loss.item()

            all_labels = torch.cat((all_labels, labels))
            all_preds = torch.cat((all_preds, pred))
            all_idxs = torch.cat((all_idxs, idxs))

    print(f"Amount of labels:{len(all_labels)}, Amount of faces:{all_labels.sum()}")
    acc = calculate_accuracy(all_labels, all_preds)


    # best_faces, worst_faces, best_other, worst_other = get_best_and_worst(all_labels, all_preds)
    # visualize_best_and_worst(data_loaders, all_labels, all_idxs, epoch, best_faces, worst_faces, best_other, worst_other)

    return avg_loss/(i+1), acc

def main():

    gender_list = [[], ["Female"], ["Male"], ["Female"], ["Male"]]
    skin_list = [[], ["lighter"], ["lighter"], ["darker"], ["darker"]]
    name_list = ["all", "dark man", "dark female", "light man", "light female"]


    # Load model
    model = vae_model.Db_vae(z_dim=ARGS.zdim, device=DEVICE).to(DEVICE)
    model.load_state_dict(torch.load(f"results/{ARGS.adress}/model.pt"))
    model.eval()

    for i in range(5):
        eval_loader: DataLoader = make_eval_loader(batch_size=ARGS.batch_size, filter_exclude_skin_color=skin_list[i], filter_exclude_gender=gender_list[i])

        loss, acc = eval_model(model, eval_loader)

        print(f"{name_list[i]} => loss:{loss}, acc:{acc}")

    # print_reconstruction(model, valid_data, epoch)

    return

if __name__ == "__main__":
    print("start evaluation")

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=256, type=int,
                        help='dimensionality of latent space')
    parser.add_argument('--zdim', default=20, type=int,
                        help='dimensionality of latent space')
    parser.add_argument("--adress", required=True, type=str)

    ARGS = parser.parse_args()

    main()
