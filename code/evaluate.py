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

def eval_model(model, data_loader):
    """
    perform evaluation of a single epoch
    """

    model.eval()
    avg_loss = 0

    count = 0

    all_labels = torch.tensor([], dtype=torch.int).to(config.device)
    all_preds = torch.tensor([]).to(config.device)
    all_idxs = torch.tensor([], dtype=torch.int).to(config.device)

    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            images, labels, idxs = batch
            batch_size = labels.size(0)

            images = images.to(config.device)
            labels = labels.to(config.device)
            idxs = idxs.to(config.device)
            pred, loss = model.forward(images, labels)

            loss = loss/batch_size


            avg_loss += loss.item()

            all_labels = torch.cat((all_labels, labels))
            all_preds = torch.cat((all_preds, pred))
            all_idxs = torch.cat((all_idxs, idxs))

            count = i

    print(f"Amount of labels:{len(all_labels)}, Amount of faces:{all_labels.sum()}")
    acc = utils.calculate_accuracy(all_labels, all_preds)

    return avg_loss/(count+1), acc

def main():
    gender_list = [[], ["Female"], ["Male"], ["Female"], ["Male"]]
    skin_list = [[], ["lighter"], ["lighter"], ["darker"], ["darker"]]
    name_list = ["all", "dark man", "dark female", "light man", "light female"]

    # Load model
    model = vae_model.Db_vae(z_dim=ARGS.zdim, device=config.device).to(config.device)
    model.load_state_dict(torch.load(f"results/{ARGS.adress}/model.pt"))
    model.eval()

    for i in range(5):
        eval_loader: DataLoader = make_eval_loader(batch_size=ARGS.batch_size, filter_exclude_skin_color=skin_list[i], filter_exclude_gender=gender_list[i])

        loss, acc = eval_model(model, eval_loader)

        print(f"{name_list[i]} => loss:{loss}, acc:{acc}")

    return

if __name__ == "__main__":
    print("start evaluation")

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=256, type=int,
                        help='dimensionality of latent space')
    parser.add_argument('--zdim', default=20, type=int,
                        help='dimensionality of latent space')
    parser.add_argument("--path_to_model", type=str,
                        help='Path to stored model')

    ARGS = parser.parse_args()

    main()
