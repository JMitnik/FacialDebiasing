"""
In this file the training of the network is done
"""
# Import internal
from typing import Tuple

# Import external
import torch

# Import project-based
import vae_model
import utils
from setup import config, init_trainining_results

from dataset import concat_datasets, make_train_and_valid_loaders, make_hist_loader
from datasets.generic import DataLoaderTuple

from torchvision.utils import make_grid
import matplotlib.pyplot as plt

def update_histogram(model, data_loader, epoch):
    # reset the means and histograms

    print(f"update weight histogram using method: {config.debias_type}")
    model.hist = torch.ones((config.zdim, model.num_bins)).to(config.device)
    model.means = torch.Tensor().to(config.device)

    all_labels = torch.tensor([], dtype=torch.long).to(config.device)
    all_index = torch.tensor([], dtype=torch.long).to(config.device)

    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            images, labels, index = batch

            #TEMPORARY TAKE ONLY FACES
            images, labels, index = images.to(config.device), labels.to(config.device), index.to(config.device)
            batch_size = labels.size(0)

            all_labels = torch.cat((all_labels, labels))
            all_index = torch.cat((all_index, index))

            if config.debias_type == "base":
                model.build_means(images)
            elif config.debias_type == "our":
                model.build_histo(images)

        if config.debias_type == "base":
            probs = model.get_histo_base()
        elif config.debias_type == "our":
            probs = model.get_histo_our()
        else:
            raise Exception("No correct debias method given. choose \"base\" or \"our\"")

    utils.visualize_bias(probs, data_loader, all_labels, all_index, epoch)

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

    count: int = 0

    # TODO: divide the batch-size of the loader over both face_Batch and nonface_batch, rather than doubling the batch-size
    for i, (face_batch, nonface_batch) in enumerate(zip(face_loader, nonface_loader)):
        images, labels, idxs = utils.concat_batches(face_batch, nonface_batch)
        batch_size = labels.size(0)

        images, labels = images.to(config.device), labels.to(config.device)

        pred, loss = model.forward(images, labels)

        optimizer.zero_grad()
        loss = loss.mean()

        # calculate the gradients and clip them at 5
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)

        optimizer.step()

        acc = utils.calculate_accuracy(labels, pred)
        avg_loss += loss.item()
        avg_acc += acc

        if i % config.eval_freq == 0:
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

            loss = loss.mean()
            acc = utils.calculate_accuracy(labels, pred)

            avg_loss += loss.item()
            avg_acc += acc

            all_labels = torch.cat((all_labels, labels))
            all_preds = torch.cat((all_preds, pred))
            all_idxs = torch.cat((all_idxs, idxs))

            count = i

    print(f"Length of all evals: {all_labels.shape[0]}")

    best_faces, worst_faces, best_other, worst_other = utils.get_best_and_worst_predictions(all_labels, all_preds)
    utils.visualize_best_and_worst(data_loaders, all_labels, all_idxs, epoch, best_faces, worst_faces, best_other, worst_other)

    return avg_loss/(count+1), avg_acc/(count+1)

def main():
    train_loaders: DataLoaderTuple
    valid_loaders: DataLoaderTuple

    train_loaders, valid_loaders = make_train_and_valid_loaders(
        batch_size=config.batch_size,
        max_images=config.dataset_size
    )

    # Initialize model
    model = vae_model.Db_vae(z_dim=config.zdim, device=config.device, alpha=config.alpha).to(config.device)

    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters())

    for epoch in range(config.epochs):
        # Generic sequential dataloader to sample histogram
        print("Starting epoch:{}/{}".format(epoch, config.epochs))

        if config.debias_type != 'none':
            hist_loader = make_hist_loader(train_loaders.faces.dataset, config.batch_size)
            hist = update_histogram(model, hist_loader, epoch)
            utils.write_hist(hist, epoch)

            train_loaders.faces.sampler.weights = hist

        train_loss, train_acc = train_epoch(model, train_loaders, optimizer)
        print("Training done")
        val_loss, val_acc = eval_epoch(model, valid_loaders, epoch)

        print("epoch {}/{}, train_loss={:.2f}, train_acc={:.2f}, val_loss={:.2f}, val_acc={:.2f}".format(epoch+1,
                                    config.epochs, train_loss, train_acc, val_loss, val_acc))

        valid_data = concat_datasets(valid_loaders.faces.dataset, valid_loaders.nonfaces.dataset, proportion_a=0.5)
        utils.print_reconstruction(model, valid_data, epoch)

        with open("results/" + config.run_folder + "/training_results.csv", "a") as write_file:
            s = "{},{},{},{},{}\n".format(epoch, train_loss, val_loss, train_acc, val_acc)
            print("S:", s)
            write_file.write(s)

        torch.save(model.state_dict(), "results/"+config.run_folder+"/model.pt".format(epoch))

if __name__ == "__main__":
    init_trainining_results()
    main()
