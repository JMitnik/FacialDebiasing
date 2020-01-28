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
        path_to_model: Optional[str] = None,
        model: Optional[Db_vae] = None,
        **kwargs
    ):
        self.z_dim = z_dim
        self.device = device
        self.batch_size = batch_size
        self.nr_windows = nr_windows
        self.stride = stride

        if not model and not path_to_model:
            logger.error(
                "No model or path_to_model given",
                next_step="Evaluation will not run",
                tip="Instantiate with a trained model, or set `path_to_model`."
            )
            raise Exception

        self.path_to_model = path_to_model
        self.model: Db_vae = self._init_model(self.path_to_model) if model is None else model
        self.path_to_eval_dataset = path_to_eval_dataset

    def _init_model(
        self,
        path_to_model
    ):
        full_path_to_model = f"results/{path_to_model}/model.pt"
        if not os.path.exists(full_path_to_model):
            logger.error(
                f"Can't find model at {full_path_to_model}",
                next_step="Evaluation will stop",
                tip="Double check your path to model"
            )
            raise Exception

        model: Db_vae = Db_vae(z_dim=self.z_dim, device=self.device)
        model.load_state_dict(torch.load(full_path_to_model, map_location=self.device))

        return model

    def eval(
        self,
        filter_exclude_skin_color: List[str] = [],
        filter_exclude_gender: List[str] = []
    ):
        self.model.eval()

        eval_loader: DataLoader = make_eval_loader(
            batch_size=self.batch_size,
            filter_exclude_skin_color=filter_exclude_skin_color,
            filter_exclude_gender=filter_exclude_gender,
            nr_windows=self.nr_windows,
            stride=self.stride
        )

        correct_count, count = self.eval_model(eval_loader)
        return correct_count, count

    def eval_on_setups(
        self,
        eval_name: str
    ):
        # Define the setups
        gender_list = [["Female"], ["Male"], ["Female"], ["Male"]]
        skin_list = [["lighter"], ["lighter"], ["darker"], ["darker"]]
        name_list = ["dark male", "dark female", "light male", "light female"]

        # Init the metrics
        recalls = []
        correct_pos = 0
        total_count = 0

        # Write and init the results and the header
        wf = open(f"results/{self.path_to_model}/{eval_name}", 'a+')
        wf.write(f"name,dark male,dark female,light male,light female,var,precision,recall,accuracy\n")
        wf.write(f"{self.path_to_model}")

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
            wf.write(f",{recall:.3f}")
            recalls.append(recall)

        # Calculate the average recall
        avg_recall = correct_pos/total_count*100

        # Logger info
        logger.info(f"Recall => all:{avg_recall:.3f}, dark male: {recalls[0]:.3f}, dark female: {recalls[1]:.3f}, white male: {recalls[2]:.3f}, white female: {recalls[3]:.3f}")
        logger.info(f"Variance => {(torch.tensor(recalls)).var().item():.3f}")
        wf.write(f",{(torch.tensor(recalls)).var().item():.3f}")

        return

    def eval_model(
        self,
        eval_loader: DataLoader
    ):
        """Perform evaluation of a single epoch."""
        self.model.eval()

        count = 0
        correct_count = 0

        for i, batch in enumerate(eval_loader):
            count += 1
            images_list, _, _ , _= batch

            for images in images_list:
                if len(images.shape) == 5:
                    images = images.squeeze(dim=0)

                batch_size = images.size(0)
                images = images.to(self.device)

                pred = self.model.forward_eval(images)

                if (pred > 0).any():
                    correct_count += 1
                    break

        print(f"Amount of labels:{count}, Correct labels:{correct_count}")

        return correct_count, count



