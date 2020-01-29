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

        self.store: pd.DataFrame = self.init_store(path_to_bbox)
        self.classification_label = 1

    def read_image(self, idx: int):
        img: Image = Image.open(os.path.join(
            self.path_to_images,
            self.store.iloc[idx].image_id)
        )

        return img

    def __len__(self):
        return len(self.store)

    def init_store(self, path_to_bbox):
        if not os.path.exists(path_to_bbox):
            logger.error(f"Path to bbox does not exist at {path_to_bbox}!")
            raise Exception

        try:
            store = pd.read_table(path_to_bbox, delim_whitespace=True)
            return store
        except:
            logger.error(
                f"Unable to read the bbox file located at {path_to_bbox}"
            )
