# %%
from torch.utils.data.dataloader import DataLoader
from setup import config
from dataset import make_eval_loader


eval_set: DataLoader = make_eval_loader(
    filter_exclude_country=['Senegal']
)



# %%
