import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

class DeepfakeProxy(nn.Module):
    """
    使用一个轻量级自编码器模型模拟Deepfake操作
    这个模型会破坏图像中的细节信息，特别是可能包含水印的区域
    """
    def __init__(self, bottleneck_size=16):
        super(DeepfakeProxy, self).__init__()
        # 编码器部分 - 降低分辨率，破坏细节
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, bottleneck_size, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        
        # 解码器部分 - 重建图像，但会丢失水印信息
        self.decoder = nn.Sequential(
            nn.Conv2d(bottleneck_size, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )
        
    def forward(self, image_and_cover):
        image, cover_image = image_and_cover
        # 通过轻量级自编码器，模拟Deepfake重建过程
        encoded = self.encoder(image)
        decoded = self.decoder(encoded)
        return decoded


class RegionalDestruction(nn.Module):
    """
    在图像中随机选择一个区域，应用强烈的破坏性操作
    这模拟了Deepfake对特定区域（如人脸）的强烈修改
    """
    def __init__(self, region_size_ratio=0.4, destruction_type='zero'):
        super(RegionalDestruction, self).__init__()
        self.region_size_ratio = region_size_ratio
        self.destruction_type = destruction_type
        
    def forward(self, image_and_cover):
        image, cover_image = image_and_cover
        batch_size, channels, height, width = image.shape
        
        # 创建一个与原图像相同的输出张量
        output = image.clone()
        
        # 对批次中的每张图像单独处理
        for i in range(batch_size):
            # 计算要破坏的区域大小
            region_size_h = int(height * self.region_size_ratio)
            region_size_w = int(width * self.region_size_ratio)
            
            # 随机选择区域的左上角坐标
            top = random.randint(0, height - region_size_h)
            left = random.randint(0, width - region_size_w)
            
            # 根据不同的破坏类型应用不同的操作
            if self.destruction_type == 'zero':
                # 将选定区域置零
                output[i, :, top:top+region_size_h, left:left+region_size_w] = 0.0
            elif self.destruction_type == 'noise':
                # 在选定区域应用随机噪声
                noise = torch.randn(channels, region_size_h, region_size_w).to(image.device)
                noise = (noise * 2) - 1  # 缩放到[-1,1]范围
                output[i, :, top:top+region_size_h, left:left+region_size_w] = noise
            elif self.destruction_type == 'blur':
                # 在选定区域应用强烈的模糊
                region = image[i, :, top:top+region_size_h, left:left+region_size_w]
                blurred = F.avg_pool2d(region, kernel_size=5, stride=1, padding=2)
                output[i, :, top:top+region_size_h, left:left+region_size_w] = blurred
                
        return output


class FrequencyModulation(nn.Module):
    """
    通过DCT变换修改图像的频率域特性，模拟Deepfake操作可能引入的频率特征变化
    """
    def __init__(self, strength=0.8):
        super(FrequencyModulation, self).__init__()
        self.strength = strength
        
    def forward(self, image_and_cover):
        image, cover_image = image_and_cover
        batch_size, channels, height, width = image.shape
        
        # 创建输出张量
        output = torch.zeros_like(image)
        
        for i in range(batch_size):
            for c in range(channels):
                # 将图像从[-1,1]转换为[0,1]范围
                img = (image[i, c] + 1) / 2
                
                # 应用DCT变换
                img_dct = torch.fft.rfft2(img)
                
                # 修改高频成分（模拟Deepfake通常会影响高频信息）
                mask = torch.ones_like(img_dct)
                h, w = mask.shape
                mask[h//4:, :] = self.strength
                mask[:, w//4:] = self.strength
                
                img_dct = img_dct * mask
                
                # 应用逆DCT变换
                img_restored = torch.fft.irfft2(img_dct, s=(height, width))
                
                # 将图像转回[-1,1]范围
                img_restored = (img_restored * 2) - 1
                
                # 限制到[-1,1]范围
                output[i, c] = torch.clamp(img_restored, -1, 1)
                
        return output