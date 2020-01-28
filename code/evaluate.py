"""
In this file the evaluation of the network is done
"""

import torch

import vae_model
import argparse
from setup import config
from torch.utils.data import DataLoader
from dataset import make_eval_loader

from setup import config
import utils

from torchvision.utils import make_grid
import matplotlib.pyplot as plt

def eval_model(model, data_loader):
    """Perform evaluation of a single epoch."""
    model.eval()

    count = 0
    correct_count = 0

    for i, batch in enumerate(data_loader):
        count += 1
        images_list, _, _ , _= batch

        for images in images_list:
            if len(images.shape) == 5:
                images = images.squeeze(dim=0)

            images = images.to(config.device)
            pred = model.forward_eval(images)

            if (pred > 0).any():
                correct_count += 1
                break

    print(f"Amount of labels:{count}, Correct labels:{correct_count}")

    return correct_count, count

def interpolate_images(model, amount):
    eval_loader: DataLoader = make_eval_loader(
        batch_size=config.batch_size,
        nr_windows=config.eval_nr_windows,
    )

    image_1 = []
    image_2 = []
    for i, batch in enumerate(eval_loader):
        _, _, _, img = batch

        if i == 0:
            image_1 = img.view(3, 64, 64)

        if i == 1:
            image_2 = img.view(3, 64, 64)

        if i > 1:
            break

    images = torch.stack((image_1, image_2)).to(config.device)
    recon_images = model.interpolate(images, amount)

    fig=plt.figure(figsize=(16, 16))

    ax = fig.add_subplot(2, 2, 1)
    grid = make_grid(images[1,:,:,:].view(1,3,64,64), 1)
    plt.imshow(grid.permute(1,2,0).cpu())
    utils.remove_frame(plt)

    ax = fig.add_subplot(2, 2, 2)
    grid = make_grid(images[0,:,:,:].view(1,3,64,64), 1)
    plt.imshow(grid.permute(1,2,0).cpu())
    utils.remove_frame(plt)

    ax = fig.add_subplot(2, 1, 2)
    grid = make_grid(recon_images.reshape(amount,3,64,64), amount)
    plt.imshow(grid.permute(1,2,0).cpu())
    utils.remove_frame(plt)

    plt.show()



def main():
    gender_list = [["Female"], ["Male"], ["Female"], ["Male"]]
    skin_list = [["lighter"], ["lighter"], ["darker"], ["darker"]]
    name_list = ["dark male", "dark female", "light male", "light female"]

    # Load model
    model = vae_model.Db_vae(z_dim=config.z_dim, device=config.device).to(config.device)

    if not config.path_to_model:
        raise Exception('Load up a model using --path_to_model')

    model.load_state_dict(torch.load(f"results/{config.path_to_model}/model.pt", map_location=config.device))
    model.eval()

    # interpolate_images(model, 20)
    # return

    losses = []
    recalls = []

    correct_pos = 0
    total_count = 0

    wf = open(f"results/{config.path_to_model}/{config.eval_name}", 'a+')

    wf.write(f"name,dark male,dark female,light male,light female,var,precision,recall,accuracy\n")
    wf.write(f"{config.path_to_model}")
    for i in range(4):
        eval_loader: DataLoader = make_eval_loader(
            batch_size=config.batch_size,
            filter_exclude_skin_color=skin_list[i],
            filter_exclude_gender=gender_list[i],
            nr_windows=config.eval_nr_windows,
            stride=config.stride
        )

        correct_count, count = eval_model(model, eval_loader)

        recall = correct_count/count * 100
        correct_pos += correct_count
        total_count += count

        print(f"{name_list[i]} => recall:{recall:.3f}")

        wf.write(f",{recall:.3f}")

        recalls.append(recall)

    avg_recall = correct_pos/total_count*100
    print(f"Recall => all:{avg_recall:.3f}, dark male: {recalls[0]:.3f}, dark female: {recalls[1]:.3f}, white male: {recalls[2]:.3f}, white female: {recalls[3]:.3f}")
    print(f"Variance => {(torch.Tensor(recalls)).var().item():.3f}")

    wf.write(f",{(torch.Tensor(recalls)).var().item():.3f}")

#################### NEGATIVE SAMPLING ####################
    eval_loader: DataLoader = make_eval_loader(
        batch_size=config.batch_size,
        nr_windows=config.eval_nr_windows,
        dataset_type=config.eval_dataset,
        max_images=config.max_images
    )

    neg_count, count = eval_model(model, eval_loader)
    correct_neg = count - neg_count
    neg_recall = (correct_neg/count) * 100

    wf.write(f",{correct_pos/(correct_pos + neg_count)*100:.3f},{avg_recall:.3f},{(correct_pos + correct_neg)/(2*1270)*100:.3f}\n")

    wf.close()
    return

if __name__ == "__main__":
    main()
