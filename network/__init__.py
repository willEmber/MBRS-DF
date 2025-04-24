import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks import *
import numpy as np
from utils.settings import JsonConfig
import kornia.losses

# 导入噪声层模块中的所有类，确保它们可以被eval函数识别
from .noise_layers.identity import Identity
from .noise_layers.crop import Crop, Cropout, Dropout
from .noise_layers.gaussian_noise import GN
from .noise_layers.middle_filter import MF
from .noise_layers.gaussian_filter import GF
from .noise_layers.salt_pepper_noise import SP
from .noise_layers.jpeg import Jpeg, JpegSS, JpegMask, JpegTest
from .noise_layers.combined import Combined
from .noise_layers.deepfake_sim import DeepfakeProxy, RegionalDestruction, FrequencyModulation
