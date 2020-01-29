from datasets.generic import GenericImageDataset
import torch
from torch.utils.data import Dataset as TorchDataset
import torchvision.transforms as transforms
import h5py
from PIL import Image
from typing import Callable
from .data_utils import default_transform, DataLabel, DatasetOutput

class H5CelebA(GenericImageDataset):
    def __init__(self, h5_dataset: h5py.Dataset, **kwargs):
        super().__init__(**kwargs)

        self.store = h5_dataset
        self.classification_label = 1

    def __len__(self):
        return len(self.store)

    def read_image(self, idx: int):
        return self.pil_transformer(self.store[idx, :, :, ::-1])
