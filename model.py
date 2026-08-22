import torch
import torch.nn as nn


def zsn2n_cross_loss(model, d1, d2):
    """ 下采样图像间的交叉预测损失 """
    pred_noise_d1 = model(d1)
    pred_noise_d2 = model(d2)
    
    pred_d1 = d1 - pred_noise_d1
    pred_d2 = d2 - pred_noise_d2

    return 1 / 2 * (dual_band_freq_loss(pred_d1, d2) +
                    dual_band_freq_loss(pred_d2, d1))

def up_loss(model, up_D1, up_D2):
    pred_noise_D1 = model(up_D1)

    denoise_D1 = up_D1 - pred_noise_D1

    return dual_band_freq_loss(denoise_D1, up_D2)



def create_freq_mask(shape, radius_ratio, device='cpu', temperature=10.0):
    _, _, H, W = shape
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

def dual_band_freq_loss(pred, target, lambda_low=0.5, lambda_high=1):
    
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

    

class network(nn.Module):

    def __init__(self, n_chan, chan_embed=48):
        super().__init__()

        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv1 = nn.Conv2d(n_chan, chan_embed, 3, padding=1)
        self.conv2 = nn.Conv2d(chan_embed, chan_embed, 3, padding=1 )
        
        self.conv3 = nn.Conv2d(chan_embed, chan_embed, 3, padding=1 )
        self.conv4 = nn.Conv2d(chan_embed, chan_embed, 3, padding=1 )
        self.conv5 = nn.Conv2d(chan_embed, chan_embed, 3, padding=1 )
        self.conv6 = nn.Conv2d(chan_embed, chan_embed, 3, padding=1 )
        self.conv7 = nn.Conv2d(chan_embed, chan_embed, 3, padding=1 )
        self.final = nn.Conv2d(chan_embed, n_chan, 1)


        
    def forward(self, x):
        res = x
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.act(self.conv3(x))
        
        x = self.act(self.conv4(x))
        x = self.act(self.conv5(x))
        x = self.act(self.conv6(x))
        x = self.act(self.conv7(x))
        x = self.final(x)
        
        return x + res

