# %%
from setup import config
from torch.utils.data import ConcatDataset, DataLoader
from datasets import train_and_valid_loaders, sample_dataset

sample_train_loader, sample_valid_loader, sample_train_dataset, sample_valid_dataset = train_and_valid_loaders(2, max_images=100)

# %%
