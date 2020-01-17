# %%
from setup import config
from datasets import PBBDataset


eval_set = PBBDataset(
    path_to_images=config.path_to_eval_images,
    path_to_metadata=config.path_to_eval_metadata,
    filter_excl_country=['Senegal']
)

eval_set[20]

# %%
