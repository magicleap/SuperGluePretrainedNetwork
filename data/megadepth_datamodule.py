import pytorch_lightning as pl
from torch.utils.data import DataLoader
from typing import Optional

from .megadepth_dataset import MegaDepthPairsDataset


class MegaDepthPairsDataModule(pl.LightningDataModule):
    def __init__(self, root_path, train_list_path, val_list_path, test_list_path,
                 batch_size, num_workers, target_size, color_aug_transform, val_max_pairs_per_scene):
        super(MegaDepthPairsDataModule, self).__init__()
        self.root_path = root_path
        self.train_list_path = train_list_path
        self.val_list_path = val_list_path
        self.test_list_path = test_list_path

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.target_size = target_size

        self.color_aug_transform = color_aug_transform
        self.val_max_pairs_per_scene = val_max_pairs_per_scene

    @staticmethod
    def read_scenes_list(path):
        with open(path) as f:
            scenes_list = f.readlines()
        return [s.rstrip() for s in scenes_list]

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_ds = MegaDepthPairsDataset(
            root_path=self.root_path,
            scenes_list=self.read_scenes_list(self.train_list_path),
            target_size=self.target_size,
            color_aug_transform=self.color_aug_transform
        )
        self.val_ds = MegaDepthPairsDataset(
            root_path=self.root_path,
            scenes_list=self.read_scenes_list(self.val_list_path),
            target_size=self.target_size,
            max_pairs_per_scene=self.val_max_pairs_per_scene
        )
        self.test_ds = MegaDepthPairsDataset(
            root_path=self.root_path,
            scenes_list=self.read_scenes_list(self.test_list_path),
            target_size=self.target_size,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(self.val_ds, shuffle=False, batch_size=1)

    def test_dataloader(self):
        return DataLoader(self.test_ds, shuffle=False, batch_size=1)
