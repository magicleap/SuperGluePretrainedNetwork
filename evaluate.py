import os
import yaml
import pytorch_lightning as pl

from models.matching_module import MatchingModule
from data.megadepth_datamodule import MegaDepthPairsDataModule

experiment_path = '/home/ostap/logs/superglue/lightning/2021-05-19-20-02-48'
with open(os.path.join(experiment_path, 'config.yaml')) as f:
    config = yaml.full_load(f)

data_config = config['data']
dm = MegaDepthPairsDataModule(
    root_path=data_config['root_path'],
    train_list_path=data_config['train_list_path'],
    val_list_path=data_config['val_list_path'],
    test_list_path=data_config['test_list_path'],
    batch_size=data_config['batch_size_per_gpu'],
    num_workers=data_config['dataloader_workers_per_gpu'],
    target_size=data_config['target_size'],
    color_aug_transform=None,
    val_max_pairs_per_scene=50
)

config['superpoint']['weights'] = '/home/ostap/projects/SuperGluePretrainedNetwork/models/weights/superpoint_v1.pth'
config['superglue']['weights'] = '/home/ostap/projects/SuperGluePretrainedNetwork/models/weights/superglue_outdoor.pth'
config['superpoint']['keypoint_threshold'] = 0.005

model = MatchingModule.load_from_checkpoint(
    '/home/ostap/logs/superglue/lightning/2021-05-21-23-38-16/superglue-step=569999.ckpt',
    train_config={**config['train'], **config['inference']},
    superpoint_config=config['superpoint'],
    superglue_config=config['superglue'],
)

trainer = pl.Trainer(
    gpus=[0, 1, 2],
    max_epochs=1,
    accelerator='ddp',
    num_sanity_val_steps=0
)

trainer.validate(model, datamodule=dm)
