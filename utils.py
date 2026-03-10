import os
import time
import torch
from os import path as osp
from math import exp
import math
import torch.nn.functional as F


def add_noise(x, noise_type, noise_level):

    if noise_type == 'gaussian':
        noisy = x + torch.normal(
            0, noise_level / 255, x.shape, device=x.device)
        noisy = torch.clamp(noisy, 0, 1)
    elif noise_type == 'poisson':
        noisy = torch.poisson(noise_level * x) / noise_level
    elif noise_type == 'saltpepper':
        prob = torch.rand_like(x)
        noisy = x.clone()
        noisy[prob < noise_level] = 0
        noisy[prob > 1 - noise_level] = 1
    elif noise_type == 'bernoulli':
        prob = torch.rand_like(x)
        mask = (prob > noise_level).float()
        noisy = x * mask
    elif noise_type == 'impulse':
        prob = torch.rand_like(x)
        noise = torch.rand_like(x)
        noisy = x.clone()
        noisy[prob < noise_level] = noise[prob < noise_level]
    elif noise_type == 'salt and gauss':
        noisy = x + torch.normal(
            0, noise_level / 255, x.shape, device=x.device)
        prob = torch.rand_like(x)
        noisy = noisy.clone()
        noisy[prob < 0.2] = 0
        noisy[prob > 1 - 0.2] = 1
        noisy = torch.clamp(noisy, 0, 1)
    else:
        raise ValueError("Invalid noise type. Choose 'gaussian' or 'poisson'.")

    return noisy


def dict2str(opt, indent_level=1):
    """dict to string for printing options.

    Args:
        opt (dict): Option dict.
        indent_level (int): Indent level. Default: 1.

    Return:
        (str): Option string for printing.
    """
    msg = '\n'
    for k, v in opt.items():
        if isinstance(v, dict):
            msg += ' ' * (indent_level * 2) + k + ':['
            msg += dict2str(v, indent_level + 1)
            msg += ' ' * (indent_level * 2) + ']\n'
        else:
            msg += ' ' * (indent_level * 2) + k + ': ' + str(v) + '\n'
    return msg


def get_time_str():
    return time.strftime('%Y%m%d_%H%M%S', time.localtime())


def mkdir_and_rename(path):
    """mkdirs. If path exists, rename it with timestamp and create a new one.

    Args:
        path (str): Folder path.
    """
    if osp.exists(path):
        new_name = path + '_archived_' + get_time_str()
        print(f'Path already exists. Rename it to {new_name}', flush=True)
        os.rename(path, new_name)
    os.makedirs(path, exist_ok=True)


def gaussian(window_size, sigma):
    gauss = torch.Tensor([
        exp(-(x - window_size // 2)**2 / (2 * sigma**2))
        for x in range(window_size)
    ])
    return gauss / gauss.sum()


def create_window(window_size, channel=1, sigma=1.5):
    _1D_window = gaussian(window_size, sigma).unsqueeze(1)
    _2D_window = _1D_window.mm(
        _1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    return _2D_window.expand(channel, 1, window_size, window_size).contiguous()


def ssim_torch(img1, img2, window_size=11, window=None, size_average=True):
    """
        img1, img2 : torch.Tensor (B, C, H, W)
        window_size : int
            
        window : torch.Tensor
            
        ssim_score : torch.Tensor
            SSIM 
    """
    # 参数校验
    if img1.dim() != 4:
        raise ValueError("Input images must be 4D tensors (B, C, H, W)")

    # 像素范围设定
    max_val = 1.0 if img1.max() <= 1 else 255.0

    # 创建窗口
    if window is None:
        window = create_window(window_size, img1.shape[1],
                               sigma=1.5).to(img1.device)

    # 计算均值
    mu1 = F.conv2d(img1,
                   window,
                   padding=window_size // 2,
                   groups=img1.shape[1])
    mu2 = F.conv2d(img2,
                   window,
                   padding=window_size // 2,
                   groups=img1.shape[1])

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    # 计算方差
    sigma1_sq = F.conv2d(
        img1 * img1, window, padding=window_size // 2,
        groups=img1.shape[1]) - mu1_sq
    sigma2_sq = F.conv2d(
        img2 * img2, window, padding=window_size // 2,
        groups=img1.shape[1]) - mu2_sq
    sigma12 = F.conv2d(
        img1 * img2, window, padding=window_size // 2,
        groups=img1.shape[1]) - mu1_mu2

    # SSIM 公式
    C1 = (0.01 * max_val)**2
    C2 = (0.03 * max_val)**2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean() if size_average else ssim_map.mean(1).mean(1).mean(
        1)
