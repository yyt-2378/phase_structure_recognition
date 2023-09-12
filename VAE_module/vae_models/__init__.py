from .base import *
from .vanilla_vae import *
from .gamma_vae import *
from .vampvae import *
from .joint_vae import *
from .info_vae import *
# from .twostage_vae import *
from .vq_vae import *


# Aliases
VAE = VanillaVAE
GaussianVAE = VanillaVAE

vae_models = {'VQVAE': VQVAE,
              'InfoVAE': InfoVAE,
              'VampVAE': VampVAE,
              'GammaVAE': GammaVAE,
              'JointVAE': JointVAE,
              'VanillaVAE': VanillaVAE,
              }
