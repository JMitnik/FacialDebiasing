from datasets.generic import GenericImageDataset
from pathlib import Path
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
        )

        return img

    def init_store(self):
        return list(Path().glob(f'{self.path_to_images}/**/*.jpg'))
