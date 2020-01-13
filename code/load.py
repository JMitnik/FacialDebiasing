# %%
from setup import config
from torch.utils.data import ConcatDataset, DataLoader
from datasets import train_and_valid_loaders

train_loader, valid_loader, train_dataset, valid_dataset = train_and_valid_loaders(2)
sample_train_loader, sample_valid_loader, sample_train_dataset, sample_valid_dataset = train_and_valid_loaders(2, max_train_images=5, max_valid_images=5)

# %%
sample_train_dataset.sample()

# %%
