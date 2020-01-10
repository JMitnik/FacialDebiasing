# %%
from setup import config
from torch.utils.data import ConcatDataset, DataLoader
from dataset import train_and_valid_loaders

train_loader, valid_loader = train_and_valid_loaders(2)


# %%
test = next(enumerate(train_loader))

# %%
