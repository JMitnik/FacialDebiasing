import os
from logger import logger

from typing import Optional, List
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from vae_model import Db_vae
from dataset import make_eval_loader

class Evaluator:
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
        **kwargs
    ):
        self.z_dim = z_dim
        self.device = device
        self.batch_size = batch_size
        self.model_name = model_name
        self.nr_windows = nr_windows
        self.stride = stride

        self.path_to_model = path_to_model
        self.model: Db_vae = self.init_model(path_to_model, model)
        self.path_to_eval_dataset = path_to_eval_dataset

    def init_model(self, path_to_model: Optional[str] = None, model: Optional[Db_vae] = None):
        if model is not None:
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
                   dataset_type: str= ""):
        self.model.eval()

        if dataset_type == "":
            eval_loader: DataLoader = make_eval_loader(
                batch_size=self.batch_size,
                filter_exclude_skin_color=filter_exclude_skin_color,
                filter_exclude_gender=filter_exclude_gender,
                nr_windows=self.nr_windows,
                stride=self.stride,
            )
        else:
            eval_loader: DataLoader = make_eval_loader(
                batch_size=self.batch_size,
                filter_exclude_skin_color=filter_exclude_skin_color,
                filter_exclude_gender=filter_exclude_gender,
                nr_windows=self.nr_windows,
                stride=self.stride,
                dataset_type=dataset_type
            )

        correct_count, count = self.eval_model(eval_loader)
        return correct_count, count

    def eval_on_setups(self, eval_name: str):
        # Define the setups
        gender_list = [["Female"], ["Male"], ["Female"], ["Male"]]
        skin_list = [["lighter"], ["lighter"], ["darker"], ["darker"]]
        name_list = ["dark male", "dark female", "light male", "light female"]

        # Init the metrics
        recalls = []
        correct_pos = 0
        total_count = 0

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

        # Logger info
        logger.info(f"Recall => all:{avg_recall:.3f}, dark male: {recalls[0]:.3f}, dark female: {recalls[1]:.3f}, white male: {recalls[2]:.3f}, white female: {recalls[3]:.3f}")
        logger.info(f"Variance => {(torch.tensor(recalls)).var().item():.3f}")

        incorrect_neg, count = self.eval()
        correct_neg: int = count - incorrect_neg

        with open(f"results/{self.path_to_model}/{eval_name}", 'a+') as write_file:
            write_file.write(f"name,dark male,dark female,light male,light female,var,precision,recall,accuracy\n")
            write_file.write(f"{self.path_to_model}_{self.model_name}")
            write_file.write(f",{recalls[0]:.3f},{recalls[1]:.3f},{recalls[2]:.3f},{recalls[3]:.3f},{(torch.Tensor(recalls)).var().item():.3f},{correct_pos/(correct_pos + neg_count)*100:.3f},{avg_recall:.3f},{(correct_pos + correct_neg)/(2*1270)*100:.3f}\n")

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
