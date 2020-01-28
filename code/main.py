from vae_model import Db_vae
from evaluator import Evaluator
from typing import Optional
from trainer import Trainer
from setup import Config, config
import torch
from logger import logger
import utils
import torch.nn as nn
from datasets.generic import slide_windows_over_img

def make_trainer(config: Config):
     return Trainer(**config._asdict())

def make_evaluator(config: Config):
     return Evaluator(
          nr_windows=config.eval_nr_windows,
          path_to_eval_dataset=config.path_to_eval_face_images,
          **config._asdict()
     )

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
     else:
          logger.success("This is NOT a face!")

     return utils.find_face_in_subimages(model, sub_images)
