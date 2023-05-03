import torch
import torch.nn as nn
import torch.nn.functional as F

from config import global_config
CFG = global_config.cfg

# 嵌入器
def embed(prob, image, noise):
    output = -0.5 * F.tanh((prob - 2 * noise) * CFG.TRAIN.TANH_LAMBDA) + 0.5 * F.tanh((prob - (2.0 - 2 * noise)) * CFG.TRAIN.TANH_LAMBDA)
    return output + image
