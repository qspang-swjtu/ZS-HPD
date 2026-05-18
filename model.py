import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import fft
import cmath
import random
import time
import kornia
import numpy as np
import torchvision.transforms as transforms
from test_1 import reg_loss
from pytorch_msssim import ssim, ms_ssim
from scipy import ndimage  # 用于计算梯度
from test import content_aware_downsample_larger_region


def mse(gt: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    loss = torch.nn.MSELoss()
    return loss(gt, pred)
    
    
    
def create_high_pass_filter(shape, radius_ratio=0.25, device='cpu'):
    H, W = shape
    center_h, center_w = H // 2, W // 2
    radius = int(min(H, W) * radius_ratio)
    
    y_coords, x_coords = torch.meshgrid(torch.arange(H, device=device), 
                                        torch.arange(W, device=device), 
                                        indexing='ij')
    
    dist_from_center = torch.sqrt((y_coords - center_h)**2 + (x_coords - center_w)**2)
    
    mask = (dist_from_center > radius).float()
    return mask


def spectral_flatness_loss(power_spectrum):
    # power_spectrum shape: [B, C, H, W] (real-valued)
    geometric_mean = torch.exp(torch.mean(torch.log(power_spectrum + 1e-8), dim=(-2, -1), keepdim=True))
    arithmetic_mean = torch.mean(power_spectrum, dim=(-2, -1), keepdim=True)
    sfm = geometric_mean / (arithmetic_mean + 1e-8)
    return -torch.mean(torch.log(sfm + 1e-8)) # 最小化负对数，即最大化平坦度
# 理想低通滤波
def create_idae_filter(img, rate_cut_off):
    row, col = img.shape[-2], img.shape[-1]
    crow, ccol = row // 2, col // 2
    u = torch.arange(row, device=img.device) - crow
    v = torch.arange(col, device=img.device) - ccol
    # 
    U, V = torch.meshgrid(u, v, indexing='ij')
    # 每个位置
    D = torch.sqrt(U**2 + V**2)
    D_0 = rate_cut_off * torch.sqrt(torch.tensor((row / 2)**2 + (col / 2)**2))
    return (D <= D_0)


def zsn2n_cross_loss(model, d1, d2, noisy_img, lamb1):
    """ 下采样图像间的交叉预测损失 """
    N = d1.shape[2] * d1.shape[3]

    pred_noise_d1 = model(d1)
    pred_noise_d2 = model(d2)
    
    pred_d1 = d1 - pred_noise_d1
    pred_d2 = d2 - pred_noise_d2

    loss12 = 1 / 2 * (mse(pred_d1 ,d2) + mse(pred_d2 ,d1)) 

    
    loss_fft =1/2* (dual_band_freq_loss(pred_d1 , d2,lamb1 ) + dual_band_freq_loss( pred_d2, d1, lamb1 ))

    
    
    return  loss_fft

def up_loss(model,up_D1,up_D2 ,lamb1):
    N = up_D1.shape[-2] * up_D2.shape[-1]

    pred_noise_D1 = model(up_D1)

    denoise_D1 = up_D1 - pred_noise_D1

    loss_fft =  dual_band_freq_loss( denoise_D1, up_D2, lamb1)
    return loss_fft



def create_freq_mask(shape, radius_ratio, device='cpu', temperature=10.0):
    B, C, H, W = shape
    center_h, center_w = H // 2, W // 2
    
    # 计算半径（保持浮点数）
    min_dim = min(H, W)
    radius = min_dim * radius_ratio
    
    # 创建坐标网格
    y_coords, x_coords = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing='ij'
    )
    
    # 计算到中心的距离
    dist_from_center = torch.sqrt(
        (y_coords - center_h)**2 + 
        (x_coords - center_w)**2
    )
    
    # 使用sigmoid创建软掩码（可微分）
    # temperature控制边缘锐度
    low_pass_mask = torch.sigmoid(temperature * (radius - dist_from_center))
    high_pass_mask = 1.0 - low_pass_mask
    
    # 添加批次和通道维度
    low_pass_mask = low_pass_mask.unsqueeze(0).unsqueeze(0)
    high_pass_mask = high_pass_mask.unsqueeze(0).unsqueeze(0)
    
    return low_pass_mask, high_pass_mask

def dual_band_freq_loss(pred, target, epoch, lambda_low=0.5, lambda_high=1):
    
    N = pred.shape[2] * pred.shape[3]
    # 1. 进入频域并中心化
    fft_pred = torch.fft.fftshift(torch.fft.fft2(pred), dim=(-2, -1))
    fft_target = torch.fft.fftshift(torch.fft.fft2(target), dim=(-2, -1))

    # 2. 创建高/低通掩码
    low_mask, high_mask = create_freq_mask(pred.shape, 0.2, pred.device)
    
    # 3. 分离高低频分量

    fft_pred_low = fft_pred * low_mask
    fft_pred_high = fft_pred * high_mask
    
    fft_target_low = fft_target * low_mask
    fft_target_high = fft_target * high_mask

    # 4. 分别计算损失并加权
    loss_low = torch.norm(fft_pred_low - fft_target_low ,p=2) / N

    loss_high = torch.norm(fft_pred_high- fft_target_high,p=2) / N 
    # 总损失
    total_loss = loss_high * lambda_high + lambda_low * loss_low

    
    return total_loss 

    


def adjust(fft_noisy):
    amplitude = torch.abs(fft_noisy)
    phase = torch.angle(fft_noisy)
    energy = torch.var(amplitude , dim = (-2 , -1) )
    scaled_energy = (energy - energy.mean()) / (energy.std() + 1e-6)
    # weight = (energy - energy.min(dim = 1 ,keepdim = True)[0] ) / (energy.max(dim = 1 ,keepdim = True)[0] -energy.min(dim = 1 ,keepdim = True)[0])
    weight = torch.sigmoid(scaled_energy)
    adjust_amplitude = amplitude * weight.unsqueeze(-1).unsqueeze(-1)
    adjust_noisy = torch.polar(adjust_amplitude , phase)
    return adjust_noisy

def adjust_loss(fft_noisy):
    adjust_noisy = adjust(fft_noisy)
    energy = torch.mean(torch.abs(adjust_noisy)**2)
    return energy

        
class network(nn.Module):

    def __init__(self, n_chan, chan_embed=48 , p=0.2):
        super().__init__()

        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv1 = nn.Conv2d(n_chan, chan_embed, 3, padding=1)
        self.conv2 = nn.Conv2d(chan_embed, chan_embed, 3, padding=1 )
        
        self.conv3 = nn.Conv2d(chan_embed, chan_embed, 3, padding=1 )
        self.conv4 = nn.Conv2d(chan_embed, chan_embed, 3, padding=1 )
        
        self.conv5 = nn.Conv2d(chan_embed, chan_embed, 3, padding=1 )
        self.conv6 = nn.Conv2d(chan_embed, chan_embed, 3, padding=1 )

        self.conv7 = nn.Conv2d(chan_embed, chan_embed, 3, padding=1 )
        self.conv9 = nn.Conv2d(chan_embed, n_chan, 1)


        
    def forward(self, x):
        res = x
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        # x = self.afeb_block(x)
        x = self.act(self.conv3(x))
        
        x = self.act(self.conv4(x))
        

        x = self.conv9(x)
        
        return x + res
    # def _initialize_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             init.orthogonal_(m.weight)
    #             if m.bias is not None:
    #                 init.constant_(m.bias, 0)
    #         elif isinstance(m, nn.BatchNorm2d):
    #             init.constant_(m.weight, 1)
    #             init.constant_(m.bias, 0)
        

