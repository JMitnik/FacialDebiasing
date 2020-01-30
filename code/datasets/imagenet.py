from datasets.generic import GenericImageDataset
from pathlib import Path
from logger import logger
from PIL import Image

class ImagenetDataset(GenericImageDataset):
    def __init__(
        self,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.store = self.init_store()
        self.classification_label = 0

    def read_image(self, idx: int):
        img: Image = Image.open(
            self.store[idx]
        ).convert("RGB")

        return img

    def init_store(self):
        store = list(Path().glob(f'{self.path_to_images}/**/*.jpg'))

        if len(store) == 0:
            logger.warning(
                "Data-store is empty.",
                tip=f"Check if your path to data {self.path_to_images} is not empty"
            )

        return list(Path().glob(f'{self.path_to_images}/**/*.jpg'))

    def __len__(self):
        return len(self.store)