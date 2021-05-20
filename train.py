import yaml
import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor
from pytorch_lightning import loggers as pl_loggers
from datetime import datetime

from data.megadepth_datamodule import MegaDepthPairsDataModule
from models.matching_module import MatchingModule


def main():
    pl.seed_everything(0)

    with open('config/config.yaml') as f:
        config = yaml.full_load(f)

    experiment_name = str(datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    log_path = os.path.join(config['logging']['root_path'], config['logging']['name'], experiment_name)
    if not os.path.exists(log_path):
        os.makedirs(log_path, exist_ok=True)
    with open(os.path.join(log_path, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)

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
        val_max_pairs_per_scene=data_config['val_max_pairs_per_scene']
    )

    model = MatchingModule(
        train_config={**config['train'], **config['inference']},
        superpoint_config=config['superpoint'],
        superglue_config=config['superglue'],
    )
    # configure callbacks and logger
    checkpoint_callback = ModelCheckpoint(
        dirpath=log_path,
        filename='superglue-{step:d}',
        every_n_val_epochs=1,
        save_top_k=-1
    )
    lr_monitor = LearningRateMonitor()
    tb_logger = pl_loggers.TensorBoardLogger(
        config['logging']['root_path'],
        name=config['logging']['name'],
        version=experiment_name
    )

    trainer = pl.Trainer(
        gpus=config['gpus'],
        max_epochs=1,
        accelerator="ddp",
        val_check_interval=config['logging']['val_frequency'],
        log_every_n_steps=config['logging']['train_logs_steps'],
        num_sanity_val_steps=0,
        callbacks=[checkpoint_callback, lr_monitor],
        logger=[tb_logger]
    )
    trainer.fit(model, datamodule=dm)


if __name__ == '__main__':
    main()
