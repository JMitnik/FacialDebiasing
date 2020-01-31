
from torchvision.transforms import ToPILImage
from datasets.data_utils import DatasetOutput, default_transform
from typing import Callable
from PIL import Image
from .data_utils import slide_windows_over_img, DatasetOutput
import torch
import torch.nn as nn
from torch.utils.data import Dataset

class GenericImageDataset(Dataset):
    """Generic dataset which defines all basic operations for the images."""
    def __init__(
        self,
        path_to_images: str,
        get_sub_images: bool = False,
        sub_images_nr_windows: int = 10,
        sub_images_batch_size: int = 10,
        sub_images_min_size: int = 30,
        sub_images_max_size: int = 64,
        sub_images_stride: float = 0.2,
        classification_label: int = 0,
        transform: Callable = default_transform,
        **kwargs
    ):
        self.path_to_images = path_to_images
        self.transform = transform

        self.classification_label = classification_label

        # Sub images properties
        self.get_sub_images = get_sub_images
        self.sub_images_min_size = sub_images_min_size
        self.sub_images_max_size = sub_images_max_size
        self.sub_images_nr_windows = sub_images_nr_windows
        self.sub_images_batch_size = sub_images_batch_size
        self.sub_images_stride = sub_images_stride

        self.pil_transformer = ToPILImage()

        # Create store for data
        self.store = None

    def __getitem__(self, idx: int):
        # Read the image from the store index, and a dataset-defined `.read_image`
        img: Image = self.read_image(idx)

        # Apply transformation to the image
        tensor_img: torch.Tensor = self.transform(img)

        sub_images: torch.Tensor = torch.tensor(0)

        # Extract sub images if applicable
        if self.get_sub_images:
            sub_images = slide_windows_over_img(
                tensor_img,
                min_win_size=self.sub_images_min_size,
                max_win_size=self.sub_images_max_size,
                nr_windows=self.sub_images_nr_windows,
                stride=self.sub_images_stride
            )

        return DatasetOutput(
            image=tensor_img,
            label=torch.tensor(self.classification_label),
            idx=torch.tensor(idx).long(),
            sub_images=sub_images
        )


    def read_image(self, idx: int):
        """Interface, returns an PIL Image using the index."""
        pass

    def __len__(self):
        return len(self.store)
