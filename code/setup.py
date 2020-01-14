from typing import NamedTuple

class Config(NamedTuple):
    # Path to CelebA images
    path_to_celeba_images: str = 'data/celeba/images'
    # Path to CelebA bounding-boxes
    path_to_celeba_bbox_file: str = 'data/celeba/list_bbox_celeba.txt'
    # Path to ImageNet images
    path_to_imagenet_images: str = 'data/imagenet'
    # Random seed for reproducability
    random_seed: int = 0

config = Config()
