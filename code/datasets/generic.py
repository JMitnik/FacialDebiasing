import torch
from torch.utils.data import Dataset as TorchDataset, ConcatDataset, DataLoader, Dataset, Sampler, WeightedRandomSampler, BatchSampler, SequentialSampler
from torch.utils.data.dataset import Subset
from torch.utils.data.sampler import RandomSampler
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from setup import config
import os
import numpy as np
import pandas as pd
from PIL import Image
from typing import Callable, Optional, List, NamedTuple
from enum import Enum

from torch import float64

class GenderEnum(Enum):
    MEN = "men"
    WOMEN = "woman"

class DataLabel(Enum):
    POSITIVE = 1
    NEGATIVE = 0

class CountryEnum(Enum):
    FINLAND = "Finland"
    ICELAND = "Iceland"
    SWEDEN = "Sweden"
    RWANDA = "Rwanda"
    SENEGAL = "Senegal"
    SOUTHAFRICA = "South Africa"

class SkinColorEnum(Enum):
    DARKER = "darker"
    LIGHTER = "lighter"

class DataLoaderTuple(NamedTuple):
    faces: DataLoader
    nonfaces: DataLoader

# Default transform
default_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])
