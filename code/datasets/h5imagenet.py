from datasets.generic import GenericImageDataset
import h5py
from PIL import Image

class H5Imagenet(GenericImageDataset):
    def __init__(
        self,
        h5_dataset: h5py.Dataset,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.store: h5py.Dataset = h5_dataset
        self.classification_label = 0

    def read_image(self, idx: int):
        img: Image = self.pil_transformer(self.store[idx, :, :, ::-1])
        return img

    def __len__(self):
        return len(self.store)
