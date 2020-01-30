from utils import visualize_tensor
from vae_model import Db_vae
from pathlib import Path
import numpy as np
from evaluator import Evaluator
from typing import Optional
from trainer import Trainer
from setup import Config, config
import torch
from logger import logger
import utils
import torch.nn as nn
from datasets.data_utils import slide_windows_over_img
from dataclasses import asdict

def make_trainer(config: Config):
     return Trainer(**asdict(config))

# trainer = make_trainer(config)
# trainer.train(10)

def make_evaluator(config: Config):
     """Creates an Evaluator object which is ready to .eval on, or .eval_on_setups in case of the automated experience. """
     return Evaluator(
          nr_windows=config.sub_images_nr_windows,
          path_to_eval_dataset=config.path_to_eval_face_images,
          **config._asdict()
     )

# evaluator = make_evaluator(config)
# evaluator.eval_on_setups('Test')

def classify_image(
     path_to_image: str,
     model: Optional[Db_vae] = None,
     path_to_model: Optional[str] = None,
     z_dim: Optional[int] = None,
     device: Optional[str] = None,
     batch_size: int = 10
):
     if not model and not path_to_model:
          logger.error(
               "No model or path_to_model given",
               next_step="Classification will not be done",
               tip="Instantiate with a trained model, or set `path_to_model`."
          )
          raise Exception

     model = Db_vae.init(path_to_model, device, z_dim) if model is None else model

     # Make sub-images
     img = utils.read_image(path_to_image)
     sub_images: torch.Tensor = slide_windows_over_img(img, 30, 64, 10, 0.2)
     sub_images = torch.split(sub_images, batch_size)

     if utils.find_face_in_subimages(model, sub_images):
          logger.success("This is a face!")
          return True
     else:
          logger.error("This is NOT a face!")
          return False

def classify_random_image(
     model: Optional[Db_vae] = None,
     path_to_model: Optional[str] = None,
     z_dim: Optional[int] = None,
     device: Optional[str] = None,
     batch_size: int = 10
):
     path_to_data = 'data/**/*.jpg'
     images = list(Path().glob(path_to_data))
     idx = np.random.choice(len(images))

     path_to_img = images[idx]
     img = utils.read_image(path_to_img)

     utils.visualize_tensor(img)
     classify_image(path_to_img, path_to_model=path_to_model, z_dim=z_dim, device=device, batch_size=batch_size)

# classify_random_image(path_to_model='dante', z_dim=200, device=config.device)

if __name__ == "__main__":
     config = Config()