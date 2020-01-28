from datasets.data_utils import default_transform
from typing import Callable
from PIL import Image
from data_utils import slide_windows_over_img
import torch
import torch.nn as nn
from torch.utils.data import Dataset

class GenericImageDataset(Dataset):

    def __init__(
        self,
        store,
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

        # Create store for data
        self.store = self.init_store()

    def __getitem__(self, idx: int):
        # Read the image from the dataset collection using the index
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


    def read_image(self, idx: int):
        """Interface, returns an image tensor using the index."""
        pass

    def init_store(self, **kwargs):
        """Method which creates a store to index. Create a store such as a Dataframe."""
        pass
