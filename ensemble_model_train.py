# fusion_ensemble_model
from fusion_ensemble_model.ensemble_model import DCVAESRUnet
from fusion_ensemble_model.ensemble_dataset import DCVAESRUnetDataset

# SR model
from SR_model.option import args as sr_args
import SR_model.utility as utility

# vae module
from VAE_module.vae_models.vanilla_vae import VanillaVAE

import argparse
import yaml
import numpy as np

import torch
import torch.nn as nn


if __name__ == '__main__':
    sr_model_args = sr_args
    parser = argparse.ArgumentParser(description='Generic runner for VAE vae_models')
    parser.add_argument('--config', '-c',
                        dest="filename",
                        metavar='FILE',
                        help='path to the config file',
                        default='D:\\project\\deep_learning_recovery\\VAE_module\\configs\\vae.yaml')

    args = parser.parse_args()
    with open(args.filename, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    vae_model_args = config['model_params']
    device = torch.device(args.device)

    eff_batch_size = args.batch_size
    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    checkpoint = utility.checkpoint(args)
    # define model
    pretrained_model = DCVAESRUnet(sr_model_args, vae_model_args)

    # todo: define dataset
    train_dataset = DCVAESRUnetDataset()

    # todo: define optimizer

    # todo: define train one epoch

    # 打印整个模型的结构
    print(pretrained_model)

    # 计算并打印模型的参数量
    total_params = sum(p.numel() for p in pretrained_model.parameters())
    print(f"Total Parameters: {total_params}")
