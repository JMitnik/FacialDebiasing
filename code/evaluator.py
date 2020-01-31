import os
from setup import Config
from logger import logger

from typing import Optional, List
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from vae_model import Db_vae
from dataset import make_eval_loader
from dataclasses import asdict

class Evaluator:
    """
    Class that evaluates a model based on a given pre-initialized model or path_to_model
    and displays several performance metrics.
    """
    def __init__(
        self,
        path_to_eval_dataset,
        z_dim: int,
        batch_size: int,
        device: str,
        nr_windows: int,
        stride: float,
        model_name: str,
        path_to_model: Optional[str] = None,
        model: Optional[Db_vae] = None,
        config: Optional[Config] = None,
        **kwargs
    ):
        self.z_dim = z_dim
        self.device = device
        self.batch_size = batch_size
        self.model_name = model_name
        self.nr_windows = nr_windows
        self.stride = stride

        self.config = config

        self.path_to_model = path_to_model
        self.model: Db_vae = self.init_model(path_to_model, model)
        self.path_to_eval_dataset = path_to_eval_dataset

    def init_model(self, path_to_model: Optional[str] = None, model: Optional[Db_vae] = None):
        """Initializes a stored model or one that directly comes from training."""
        if model is not None:
            logger.info("Using model passed")
            return model.to(self.device)

        # If path_to_model, load model from file
        if path_to_model:
            return Db_vae.init(path_to_model, self.device, self.z_dim).to(self.device)

        logger.error(
            "No model or path_to_model given",
            next_step="Evaluation will not run",
            tip="Instantiate with a trained model, or set `path_to_model`."
        )
        raise Exception

    def eval(self, filter_exclude_skin_color: List[str] = [], filter_exclude_gender: List[str] = [],
                   dataset_type: str= "", max_images: int = -1):
        """Evaluates a model based and returns the amount of correctly classified and total classified images."""
        self.model.eval()

        if dataset_type == "":
            eval_loader: DataLoader = make_eval_loader(
                filter_exclude_skin_color=filter_exclude_skin_color,
                filter_exclude_gender=filter_exclude_gender,
                **asdict(self.config)
            )
        else:
            params = {**asdict(self.config), 'max_images': max_images}

            eval_loader: DataLoader = make_eval_loader(
                filter_exclude_skin_color=filter_exclude_skin_color,
                filter_exclude_gender=filter_exclude_gender,
                dataset_type=dataset_type,
                **params
            )

        correct_count, count = self.eval_model(eval_loader)
        return correct_count, count

    def eval_on_setups(self, eval_name: Optional[str] = None):
        """Evaluates a model and writes the results to a given file name."""
        eval_name = self.config.eval_name if eval_name is None else eval_name

        # Define the predefined setups
        gender_list = [["Female"], ["Male"], ["Female"], ["Male"]]
        skin_list = [["lighter"], ["lighter"], ["darker"], ["darker"]]
        name_list = ["dark male", "dark female", "light male", "light female"]

        # Init the metrics
        recalls = []
        correct_pos = 0
        total_count = 0

        # Go through the predefined setup
        for i in range(4):
            logger.info(f"Running setup for {name_list[i]}")

            # Calculate on the current setup
            correct_count, count = self.eval(
                filter_exclude_gender=gender_list[i],
                filter_exclude_skin_color=skin_list[i]
            )

            # Calculate the metrics
            recall = correct_count / count * 100
            correct_pos += correct_count
            total_count += count

            # Log the recall
            logger.info(f"Recall for {name_list[i]} is {recall:.3f}")
            recalls.append(recall)

        # Calculate the average recall
        avg_recall = correct_pos/total_count*100
        variance = (torch.tensor(recalls)).var().item()

        # Calculate the amount of negative performance
        logger.info("Evaluating on negative samples")
        incorrect_neg, neg_count = self.eval(dataset_type='h5_imagenet', max_images=1270)
        correct_neg: int = neg_count - incorrect_neg

        # Calculate the precision and accuracy
        precision = correct_pos/(correct_pos + neg_count)*100
        accuracy = (correct_pos + correct_neg)/(2*1270)*100

        # Logger info
        logger.info(f"Recall => all: {avg_recall:.3f}")
        logger.info(f"Recall => dark male: {recalls[0]:.3f}")
        logger.info(f"Recall => dark female: {recalls[1]:.3f}")
        logger.info(f"Recall => white male: {recalls[2]:.3f}")
        logger.info(f"Recall => white female: {recalls[3]:.3f}")
        logger.info(f"Variance => {variance:.3f}")
        logger.info(f"Precision => {precision:.3f}")
        logger.info(f"Accuracy => {accuracy:.3f}")

        # Write final results
        path_to_eval_results = f"results/{self.path_to_model}/{eval_name}"
        with open(path_to_eval_results, 'a+') as write_file:

            # If file has no header
            if not os.path.exists(path_to_eval_results) or os.path.getsize(path_to_eval_results) == 0:
                write_file.write(f"name,dark male,dark female,light male,light female,var,precision,recall,accuracy\n")

            write_file.write(f"{self.path_to_model}_{self.model_name}")
            write_file.write(f",{recalls[0]:.3f},{recalls[1]:.3f},{recalls[2]:.3f},{recalls[3]:.3f},{variance:.3f},{precision:.3f},{avg_recall:.3f},{accuracy:.3f}\n")

        logger.success("Finished evaluation!")

    def eval_model(self, eval_loader: DataLoader):
        """Perform evaluation of a single epoch."""
        self.model.eval()

        count = 0
        correct_count = 0

        # Iterate over all images and their sub_images
        for _, batch in enumerate(eval_loader):
            count += 1
            _, _, _ , sub_images = batch

            for images in sub_images:
                if len(images.shape) == 5:
                    images = images.squeeze(dim=0)

                images = images.to(self.device)

                pred = self.model.forward_eval(images)

                if (pred > 0).any():
                    correct_count += 1
                    break

        logger.info(f"Amount of labels:{count}, Correct labels:{correct_count}")

        return correct_count, count
