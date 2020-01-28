from torch.utils.data import Dataset as TorchDataset
import torchvision.transforms as transforms
import h5py
from PIL import Image
import torch
from setup import config
from typing import Callable, Optional
from .data_utils import default_transform, DataLabel, slide_windows_over_img, DatasetOutput

class H5Imagenet(TorchDataset):
    def __init__(
        self,
        h5_dataset: h5py.Dataset,
        transform: Callable = default_transform,
        get_sub_images: bool = False,
        nr_windows: int = 10,
        batch_size: int = -1,
        stride: float = 0.2
    ):
        self.dataset: h5py.Dataset = h5_dataset
        self.transform = transform
        self.pil_transformer = transforms.ToPILImage()

        self.nr_windows: int = nr_windows
        self.batch_size: Optional[int] = None if batch_size < 0 else batch_size
        self.stride: float = stride

        self.get_sub_images = get_sub_images

    def __getitem__(self, idx: int):
        # Override label with negative
        img: Image = self.pil_transformer(self.dataset[idx, :, :, ::-1])
        img = self.transform(img)

        if self.get_sub_images:
            sub_images = slide_windows_over_img(img, min_win_size=config.eval_min_size,
                                          max_win_size=config.eval_max_size,
                                          nr_windows=self.nr_windows,
                                          stride=self.stride)
            sub_images = torch.split(sub_images, self.batch_size)
        else:
            sub_images = torch.tensor(0)

        return DatasetOutput(
            image=img,
            label=DataLabel.NEGATIVE.value,
            idx=idx,
            sub_images=sub_images
        )

    def __len__(self):
        return len(self.dataset)
