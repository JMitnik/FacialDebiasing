from evaluator import Evaluator
from trainer import Trainer
from setup import Config, config
import torch
from logger import logger
import torch.nn as nn

def make_trainer(config: Config):
     return Trainer(**config._asdict())

# trainer = make_trainer(config)
# trainer.train()

def make_evaluator(config: Config):
     return Evaluator(
          nr_windows=config.eval_nr_windows,
          path_to_eval_dataset=config.path_to_eval_face_images,
          **config._asdict()
     )

evaluator = make_evaluator(config)
evaluator.eval_on_setups('test')
