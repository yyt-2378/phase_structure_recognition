import argparse
import yaml
import os
import numpy as np
from pathlib import Path

# preprocess_model
from preprocess_model.image_preprocess_model import DCVAESR
from preprocess_model.preprocess_data import DCVAESRDataLoader
from preprocess_model.preprocess_experiment import Prexpriment

# SR model
from preprocess_model.configs.option import args
import SR_model.utility as utility

# pytorch_lightning
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

import torch
import torch.nn as nn


if __name__ == '__main__':
    sr_model_args = args
    with open(args.filename, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    vae_model_args = config['model_params']
    eff_batch_size = args.batch_size

    # fix the seed for reproducibility
    seed_everything(args.seed, True)

    # define tb_logger
    tb_logger = TensorBoardLogger(save_dir=config['logging_params']['save_dir'], name=config['model_params']['name'], )
    # define model
    preprocess_model = DCVAESR(sr_model_args, vae_model_args)
    # 打印整个模型的结构
    print(preprocess_model)

    # 计算并打印模型的参数量
    total_params = sum(p.numel() for p in preprocess_model.parameters())
    print(f"Total Parameters: {total_params}")
    experiment = Prexpriment(preprocess_model, config['exp_params'])

    # todo: define dataset and dataloader
    data = DCVAESRDataLoader(**config["data_params"], pin_memory=len(config['trainer_params']['gpus']) != 0)
    data.setup()

    # todo: define train one epoch
    runner = Trainer(logger=tb_logger,
                     callbacks=[
                         LearningRateMonitor(),
                         ModelCheckpoint(save_top_k=2,
                                         dirpath=os.path.join(tb_logger.log_dir, "checkpoints"),
                                         monitor="val_loss",
                                         save_last=True),
                     ],
                     # strategy=DDPPlugin(find_unused_parameters=False),
                     **config['trainer_params'])

    Path(f"{tb_logger.log_dir}/Samples").mkdir(exist_ok=True, parents=True)
    Path(f"{tb_logger.log_dir}/VAE_Reconstructions").mkdir(exist_ok=True, parents=True)
    Path(f"{tb_logger.log_dir}/SR_Reconstructions").mkdir(exist_ok=True, parents=True)

    print(f"======= Training {config['model_params']['name']} =======")
    runner.fit(experiment, datamodule=data)
