# vae module
from VAE_module.vae_models.vanilla_vae import VanillaVAE

import torch
import torch.nn as nn


def loss_function_total(*args, **kwargs) -> dict:
    vae_output_noise, sr_output, output = args[0], args[2], args[3]
    vae_loss = VanillaVAE.loss_function(*vae_output_noise, M_N=kwargs['kld_weight'])
    sr_loss = nn.L1Loss(sr_output, vae_output_noise[2])
    segm_loss_function = nn.CrossEntropyLoss().cuda()
    segm_loss = segm_loss_function(output, vae_output_noise[2])
    total_loss = vae_loss['loss'] + sr_loss + segm_loss
    return {'loss': total_loss, 'VAE_Loss': vae_loss['loss'], 'sr_loss': sr_loss.detach(), 'segm_loss': segm_loss.detach()}

