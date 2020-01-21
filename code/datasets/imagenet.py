import torch
from torch.utils.data import Dataset as TorchDataset, ConcatDataset, DataLoader, Dataset, Sampler, WeightedRandomSampler, BatchSampler, SequentialSampler
from torch.utils.data.dataset import Subset
from torch.utils.data.sampler import RandomSampler
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

from setup import config
import numpy as np
import pandas as pd
from PIL import Image
from typing import Callable
from enum import Enum
from torch import float64

from .generic import default_transform, DataLabel, slide_windows_over_img

class ImagenetDataset(ImageFolder):
    def __init__(
        self,
        path_to_images: str,
        transform: Callable = default_transform,
        nr_windows: int = -1,
        batch_size: int = -1
    ):
        super().__init__(path_to_images, transform)

        self.nr_windows: int = nr_windows
        self.batch_size = batch_size

    def __getitem__(self, idx: int):
        # Override label with negative
        img, _ = super().__getitem__(idx)

        imgs = img

        if self.nr_windows > 0:
            imgs = slide_windows_over_img(img, min_win_size=config.eval_min_size, max_win_size=config.eval_max_size, nr_windows=self.nr_windows)
            imgs = torch.split(imgs, self.batch_size)

        return (imgs, DataLabel.NEGATIVE.value, idx)

    def sample(self, amount: int):
        max_idx: int = len(self)
        idxs: np.array = np.random.choice(np.linspace(0, max_idx - 1), amount)

        return [self.__getitem__(idx) for idx in idxs]
