import numpy as np
import math


def compute_sparsity(zs, norm):
    '''
    Hoyer metric
    norm: normalise input along dimension
    '''
    latent_dim = zs.size(-1)
    if norm:
        zs = zs / zs.std(0)
    l1_l2 = (zs.abs().sum(-1) / zs.pow(2).sum(-1).sqrt()).mean()
    return (math.sqrt(latent_dim) - l1_l2) / (math.sqrt(latent_dim) - 1)
