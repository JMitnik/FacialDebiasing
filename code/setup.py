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
    # Path to evaluation images (Faces)
    path_to_eval_face_images: str = 'data/pbb/imgs'
    # Path to evaluation metadata
    path_to_eval_metadata: str = 'data/pbb/PPB-2017-metadata.csv'
    # Path to evaluation images (Nonfaces such as Imagenet)
    path_to_eval_nonface_images: str = 'data/eval_imagenet'


config = Config()
