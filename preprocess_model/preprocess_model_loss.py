import torch
import torch.nn as nn
from torch.nn import functional as F


def loss_function_total(*args, **kwargs) -> dict:
    vae_output_noise, sr_output, hr_label = args[0], args[2], args[3]
    vae_loss_input = vae_output_noise
    vae_loss = vae_loss_function(*vae_loss_input, M_N=kwargs['M_N'])
    l1_loss = nn.L1Loss()
    sr_loss = l1_loss(sr_output, hr_label)
    total_loss = vae_loss['loss'] + sr_loss
    return {'loss': total_loss, 'VAE_Loss': vae_loss['loss'], 'sr_loss': sr_loss.detach()}


def vae_loss_function(*args, **kwargs) -> dict:
    """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
    """
    recons = args[0]
    input = args[1]
    label = args[2]
    mu = args[3]
    log_var = args[4]
    noise = input - label

    kld_weight = kwargs['M_N']  # Account for the minibatch samples from the dataset
    # recons_loss = F.mse_loss(recons, input)
    recons_loss = F.mse_loss(recons, noise)

    kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

    loss = recons_loss + kld_weight * kld_loss
    return {'loss': loss, 'Reconstruction_Loss': recons_loss.detach(), 'KLD': -kld_loss.detach()}