from torch.utils.data import Dataset as TorchDataset
import torchvision.transforms as transforms
import h5py
from PIL import Image
from typing import Callable
from .generic import default_transform, DataLabel

class H5CelebA(TorchDataset):
    def __init__(self, h5_dataset: h5py.Dataset, transform: Callable = default_transform):
        self.dataset: h5py.Dataset = h5_dataset
        self.transform = transform
        self.pil_transformer = transforms.ToPILImage()

    def __getitem__(self, idx):
        img: Image = self.pil_transformer(self.dataset[idx, :, :, ::-1])

        img = self.transform(img)

        label: int = DataLabel.POSITIVE.value

        return img, label, idx

    def __len__(self):
        return len(self.dataset)
