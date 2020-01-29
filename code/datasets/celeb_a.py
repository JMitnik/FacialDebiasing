from datasets.generic import GenericImageDataset
import torchvision.transforms as transforms
import os
import pandas as pd
from PIL import Image

# Default transform
default_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

class CelebDataset(GenericImageDataset):
    """Dataset for CelebA"""

    def __init__(self, path_to_bbox: str, **kwargs):
        super().__init__(**kwargs)
        self.store: pd.DataFrame = pd.read_table(path_to_bbox, delim_whitespace=True)
        self.classification_label = 1

    def read_images(self, idx: int):
        img: Image = Image.open(os.path.join(
            self.path_to_images,
            self.store.iloc[idx].image_id)
        )

        return img

    def __len__(self):
        return len(self.store)
