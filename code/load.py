# %%
from torch.utils.data.dataloader import DataLoader
from setup import config
from dataset import make_eval_loader


eval_loader_negative: DataLoader = make_eval_loader(
    use_positive_class=False,
    batch_size=config.batch_size,
    nr_windows=config.eval_nr_windows
)

eval_loader_positive: DataLoader = make_eval_loader(
    use_positive_class=True,
    batch_size=config.batch_size,
    nr_windows=config.eval_nr_windows
)

# %%
next(iter(eval_loader_negative))
next(iter(eval_loader_positive))
