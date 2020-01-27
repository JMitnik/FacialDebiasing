from trainer import Trainer
from setup import Config, config
import torch
from logger import logger
import torch.nn as nn

def make_trainer(config: Config):
     return Trainer(**config._asdict())

trainer = make_trainer(config)
trainer.train()

def make_evaluator():
    pass

