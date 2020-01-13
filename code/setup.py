from typing import NamedTuple

class Config(NamedTuple):
    # Path to CelebA images
    path_to_celeba_images: str = 'data/test_celeba/images'
    # Path to CelebA bounding-boxes
    path_to_celeba_bbox_file: str = 'data/test_celeba/list_bbox_celeba.txt'
    # Path to ImageNet images
    path_to_imagenet_images: str = 'data/test_imagenet'

config = Config()
