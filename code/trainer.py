import torch
import torch.nn as nn
from typing import Optional
from datetime import datetime
from logger import logger
from torch.utils.data.dataset import Dataset

from setup import init_trainining_results
from vae_model import Db_vae
from datasets.data_utils import DataLoaderTuple, DatasetOutput
import utils
from dataset import make_hist_loader, make_train_and_valid_loaders, concat_datasets

from torchvision.utils import make_grid
from matplotlib import pyplot as plt

class Trainer:
    def __init__(
        self,
        epochs: int,
        batch_size: int,
        hist_size: int,
        z_dim: int,
        alpha: float,
        num_bins: int,
        max_images: int,
        debias_type: str,
        device: str,
        lr: float = 0.001,
        eval_freq: int = 10,
        optimizer = torch.optim.Adam,
        run_folder: Optional[str] = None,
        custom_encoding_layers: Optional[nn.Sequential] = None,
        custom_decoding_layers: Optional[nn.Sequential] = None,
        **kwargs
    ):
        init_trainining_results()
        self.epochs = epochs
        self.z_dim = z_dim
        self.batch_size = batch_size
        self.debias_type = debias_type
        self.device = device
        self.eval_freq = eval_freq
        self.run_folder = run_folder

        self.model: Db_vae = Db_vae(
            z_dim=z_dim,
            hist_size=hist_size,
            alpha=alpha,
            num_bins=num_bins,
            device=self.device
        ).to(device=self.device)

        self.optimizer = optimizer(params=self.model.parameters(), lr=lr)

        train_loaders: DataLoaderTuple
        valid_loaders: DataLoaderTuple

        train_loaders, valid_loaders = make_train_and_valid_loaders(
            batch_size=batch_size,
            max_images=max_images,
            **kwargs
        )

        self.train_loaders = train_loaders
        self.valid_loaders = valid_loaders

    def train(self, epochs: Optional[int] = None):
        # Optionally use passed epochs
        epochs = self.epochs if epochs is None else epochs

        # Start training and validation cycle
        for epoch in range(epochs):
            epoch_start_t = datetime.now()
            logger.info(f"Starting epoch: {epoch+1}/{epochs}")

            self._update_sampling_histogram(epoch)

            # Training
            train_loss, train_acc = self._train_epoch()
            epoch_train_t = datetime.now() - epoch_start_t
            logger.info(f"epoch {epoch+1}/{epochs}::Training done")
            logger.info(f"epoch {epoch+1}/{epochs} => train_loss={train_loss:.2f}, train_acc={train_acc:.2f}")

            # Validation
            logger.info("Starting validation")
            val_loss, val_acc = self._eval_epoch(epoch)
            epoch_val_t = datetime.now() - epoch_start_t
            logger.info(f"epoch {epoch+1}/{epochs}::Validation done")
            logger.info(f"epoch {epoch+1}/{epochs} => val_loss={val_loss:.2f}, val_acc={val_acc:.2f}")

            # Print reconstruction
            valid_data = concat_datasets(self.valid_loaders.faces.dataset, self.valid_loaders.nonfaces.dataset, proportion_a=0.5)
            utils.print_reconstruction(self.model, valid_data, epoch)

            # Save model and scores
            self._save_epoch(epoch, train_loss, val_loss, train_acc, val_acc)

        logger.success(f"Finished training on {epochs} epochs.")

    def _save_epoch(self, epoch: int, train_loss: float, val_loss: float, train_acc: float, val_acc: float):
        """Writes training and validation scores to a csv, and stores a model to disk."""
        if not self.run_folder:
            logger.warning(f"`--run_folder` could not be found.",
                           f"The program will continue, but won't save anything",
                           f"Double-check if --run_folder is configured."
            )

            return

        # Write epoch metrics
        path_to_results = f"results/{self.run_folder}/training_results.csv"
        with open(path_to_results, "a") as wf:
            wf.write(f"{epoch}, {train_loss}, {val_loss}, {train_acc}, {val_acc}\n")

        # Write model to disk
        path_to_model = f"results/{self.run_folder}/model.pt"
        torch.save(self.model.state_dict(), path_to_model)

        logger.save("Stored model and results")


    def _eval_epoch(self, epoch):
        """Calculates the validation error of the model."""
        face_loader, nonface_loader = self.valid_loaders

        self.model.eval()
        avg_loss = 0
        avg_acc = 0

        all_labels = torch.tensor([], dtype=torch.long).to(self.device)
        all_preds = torch.tensor([]).to(self.device)
        all_idxs = torch.tensor([], dtype=torch.long).to(self.device)

        count = 0

        with torch.no_grad():
            for i, (face_batch, nonface_batch) in enumerate(zip(face_loader, nonface_loader)):
                images, labels, idxs = utils.concat_batches(face_batch, nonface_batch)

                images = images.to(self.device)
                labels = labels.to(self.device)
                idxs = idxs.to(self.device)
                pred, loss = self.model.forward(images, labels)

                loss = loss.mean()
                acc = utils.calculate_accuracy(labels, pred)

                avg_loss += loss.item()
                avg_acc += acc

                all_labels = torch.cat((all_labels, labels))
                all_preds = torch.cat((all_preds, pred))
                all_idxs = torch.cat((all_idxs, idxs))

                count = i

        best_faces, worst_faces, best_other, worst_other = utils.get_best_and_worst_predictions(all_labels, all_preds)
        utils.visualize_best_and_worst(self.valid_loaders, all_labels, all_idxs, epoch, best_faces, worst_faces, best_other, worst_other)

        return avg_loss/(count+1), avg_acc/(count+1)

    def _train_epoch(self):
        """Trains the model for one epoch."""
        face_loader, nonface_loader = self.train_loaders

        self.model.train()

        # The batches contain Image(rgb x w x h), Labels (1 for 0), original dataset indices
        face_batch: DatasetOutput
        nonface_batch: DatasetOutput

        avg_loss: float = 0
        avg_acc: float = 0
        count: int = 0

        for i, (face_batch, nonface_batch) in enumerate(zip(face_loader, nonface_loader)):
            images, labels, _ = utils.concat_batches(face_batch, nonface_batch)
            images, labels = images.to(self.device), labels.to(self.device)

            # Forward pass
            pred, loss = self.model.forward(images, labels)

            # Calculate the gradient, and clip at 5
            self.optimizer.zero_grad()
            loss = loss.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5)
            self.optimizer.step()

            # Calculate metrics
            acc = utils.calculate_accuracy(labels, pred)
            avg_loss += loss.item()
            avg_acc += acc

            if i % self.eval_freq == 0:
                logger.info(f"Training: batch:{i} accuracy:{acc}")

            count = i

        return avg_loss/(count+1), avg_acc/(count+1)

    def _update_sampling_histogram(self, epoch: int):
        """Updates the data loader for faces to be proportional to how challenge each image is, in case
        debias_type not none is.
        """
        hist_loader = make_hist_loader(self.train_loaders.faces.dataset, self.batch_size)

        if self.debias_type != 'none':
            hist = self._update_histogram(hist_loader, epoch)
            utils.write_hist(hist, epoch)
            self.train_loaders.faces.sampler.weights = hist
        else:
            self.train_loaders.faces.sampler.weights = torch.ones(len(self.train_loaders.faces.sampler.weights))


    def _update_histogram(self, data_loader, epoch):
        """Updates the histogram of `self.model`."""
        logger.info(f"Updating weight histogram using method: {self.debias_type}")

        self.model.hist = torch.ones((self.z_dim, self.model.num_bins)).to(self.device)
        self.model.means = torch.Tensor().to(self.device)

        all_labels = torch.tensor([], dtype=torch.long).to(self.device)
        all_index = torch.tensor([], dtype=torch.long).to(self.device)

        with torch.no_grad():
            for _, batch in enumerate(data_loader):
                images, labels, index, _ = batch
                images, labels, index = images.to(self.device), labels.to(self.device), index.to(self.device)

                all_labels = torch.cat((all_labels, labels))
                all_index = torch.cat((all_index, index))

                if self.debias_type == "base" or self.debias_type == "logsum":
                    self.model.build_means(images)

                elif self.debias_type == "our":
                    self.model.build_histo(images)

            if self.debias_type == "base":
                probs = self.model.get_histo_base()
            elif self.debias_type == "logsum":
                probs = self.model.get_histo_logsum()
            elif self.debias_type == "our":
                probs = self.model.get_histo_our()
            else:
                logger.error("No correct debias method given!",
                            next_step="The program will now close",
                            tip="Set --debias_method to 'base' or 'our'.")
                raise Exception()

        utils.visualize_bias(probs, data_loader, all_labels, all_index, epoch)

        return probs

    def sample(self, n_rows=4):
        n_samples = n_rows**2
        sample_images = self.model.sample(n_samples = n_samples)

        plt.figure(figsize=(n_rows*2,n_rows*2))
        grid = make_grid(sample_images.reshape(n_samples,3,64,64), n_rows)
        plt.imshow(grid.permute(1,2,0).cpu())

        utils.remove_frame(plt)
        plt.show()

        return

    def reconstruction_samples(self, n_rows=4):
        valid_data = concat_datasets(self.valid_loaders.faces.dataset, self.valid_loaders.nonfaces.dataset, proportion_a=0.5)
        fig = utils.print_reconstruction(self.model, valid_data, 0, save=False)

        fig.show()

        return

    def best_and_worst(self, n_rows=4):
        """Calculates the validation error of the model."""
        face_loader, nonface_loader = self.valid_loaders

        self.model.eval()
        avg_loss = 0
        avg_acc = 0

        all_labels = torch.tensor([], dtype=torch.long).to(self.device)
        all_preds = torch.tensor([]).to(self.device)
        all_idxs = torch.tensor([], dtype=torch.long).to(self.device)

        count = 0

        with torch.no_grad():
            for i, (face_batch, nonface_batch) in enumerate(zip(face_loader, nonface_loader)):
                images, labels, idxs = utils.concat_batches(face_batch, nonface_batch)

                images = images.to(self.device)
                labels = labels.to(self.device)
                idxs = idxs.to(self.device)
                pred, loss = self.model.forward(images, labels)

                loss = loss.mean()
                acc = utils.calculate_accuracy(labels, pred)

                avg_loss += loss.item()
                avg_acc += acc

                all_labels = torch.cat((all_labels, labels))
                all_preds = torch.cat((all_preds, pred))
                all_idxs = torch.cat((all_idxs, idxs))

                count = i

        best_faces, worst_faces, best_other, worst_other = utils.get_best_and_worst_predictions(all_labels, all_preds)
        fig = utils.visualize_best_and_worst(self.valid_loaders, all_labels, all_idxs, 0, best_faces, worst_faces, best_other, worst_other, save=False)

        fig.show()

        return
