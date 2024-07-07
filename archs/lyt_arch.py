# -*- coding: utf-8 -*-
import math

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from basicsr.utils.registry import ARCH_REGISTRY
from einops import rearrange
from timm.models.layers import trunc_normal_, DropPath

import torchvision
from torchstat import stat


class MLP(nn.Module):
    def __init__(self, n_feats):
        super().__init__()

        self.norm = LayerNorm(n_feats, data_format='channels_first')
        i_feats = 2 * n_feats
        self.scale = nn.Parameter(torch.zeros((1, n_feats, 1, 1)), requires_grad=True)
        self.fc11= nn.Conv2d(n_feats, n_feats,kernel_size=(1, 3), stride=(1, 1), padding=(0, 1),
                               )
        self.fc12 = nn.Conv2d(n_feats,i_feats, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0),
                                )
        #self.fc1 = nn.Conv2d(n_feats, i_feats, 1, 1, 0)
        self.act = nn.GELU()

        self.fc2 = nn.Conv2d(i_feats, n_feats, 1, 1, 0)

       # self.fc3 = nn.Conv2d(n_feats, n_feats, 1, 1, 0)



    def forward(self, x):
        #shortcut = x.clone()
        shortcut = nn.Identity()(x)
        x = self.norm(x)
        x = self.fc11(x)
        x = self.fc12(x)
        x = self.act(x)
        x = self.fc2(x)
        #x = self.fc3(x)
        x = x * self.scale + shortcut
        return x


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class LKA(nn.Module):
    def __init__(self, n_feats):
        super().__init__()
        self.conv0 = nn.Conv2d(n_feats, n_feats, 5, padding=2, groups=n_feats)
        self.conv_spatial = nn.Conv2d(n_feats, n_feats, 7, stride=1, padding=9, groups=n_feats, dilation=3)
        self.conv1 = nn.Conv2d(n_feats, n_feats // 2, 1)
        self.conv2 = nn.Conv2d(n_feats, n_feats // 2, 1)
        self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)
        self.conv = nn.Conv2d(n_feats // 2, n_feats, 1)


    def forward(self, x):
        attn1 = self.conv0(x)
        attn2 = self.conv_spatial(attn1)

        attn1 = self.conv1(attn1)
        attn2 = self.conv2(attn2)

        attn = torch.cat([attn1, attn2], dim=1)  
        avg_attn = torch.mean(attn, dim=1, keepdim=True)
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)
        agg = torch.cat([avg_attn, max_attn], dim=1) 
        sig = self.conv_squeeze(agg).sigmoid()
        attn = attn1 * sig[:, 0, :, :].unsqueeze(1) + attn2 * sig[:, 1, :, :].unsqueeze(1) 
       # attn = self.conv(attn)
        return attn2


class Attention(nn.Module):
    def __init__(self, n_feats):
        super().__init__()

        self.norm = LayerNorm(n_feats, data_format='channels_first')
        self.scale = nn.Parameter(torch.zeros((1, n_feats, 1, 1)), requires_grad=True)

        self.proj_1 = nn.Conv2d(n_feats, n_feats, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = LKA(n_feats)
        self.proj_2 = nn.Conv2d(n_feats, n_feats, 1)

    def forward(self, x):
        shorcut = x.clone()
        x0 = self.proj_1(self.norm(x))
        #x1 = self.gap(x)
       # x2 = self.scale * (x0 - x1) + x0
        x = self.activation(x0)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x * self.scale + shorcut
        return x


class MAB(nn.Module):
    def __init__(
            self, n_feats):
        super().__init__()
        self.n_feats = n_feats
        self.LKA = Attention(self.n_feats)
        self.LFE = MLP(self.n_feats)

    def forward(self, x):
        # large kernel attention
        x = self.LKA(x)
        # local feature extraction
        x = self.LFE(x)

        return x


class ResGroup(nn.Module):
    def __init__(self, n_resblocks, n_feats):
        super(ResGroup, self).__init__()

      
        self.body = nn.ModuleList([
            MAB(n_feats) \
            for i in range(n_resblocks)])

    def forward(self, x):
        res = x.clone()
        for i, block in enumerate(self.body):
            res = block(res)
        x = res + x

        return x

class MeanShift(nn.Conv2d):
    def __init__(
            self, rgb_range,
            rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = True 


class Upsample(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):  
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n 
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. ' 'Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)


@ARCH_REGISTRY.register()
class Lyt3(nn.Module):
    def __init__(self, n_resblocks=12, n_resgroups=1, n_colors=3, n_feats=60, scale=4):
        super(Lyt3, self).__init__()
        # res_scale = res_scale
        self.n_resgroups = n_resgroups
        self.scale = scale
        self.sub_mean = MeanShift(1.0) 
        self.head = nn.Conv2d(n_colors, n_feats, 3, 1, 1)  

        # define body module
        self.body = nn.ModuleList([
            ResGroup(
                n_resblocks, n_feats)
            for i in range(n_resgroups)])

     
        # define tail module
        self.tail = nn.Sequential( 
            nn.Conv2d(n_feats, n_colors * (scale ** 2), 3, 1, 1),  
            nn.PixelShuffle(scale)
        )
        # self.tail = nn.Sequential(nn.Conv2d(n_feats, 64, 3, 1, 1), nn.LeakyReLU(inplace=True))
        # self.upsample = Upsample(scale, 64)
        # self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)
        self.add_mean = MeanShift(1.0, sign=1)

    def forward(self, x):
        x = self.sub_mean(x)
        y = F.interpolate(x, size=[x.shape[2] * self.scale, x.shape[3] * self.scale], mode="bilinear",
                          align_corners=False) 
        x = self.head(x) 
        res = x
        for i in self.body:
            res = i(res)
        res = res + x

        x = self.tail(res) 
        # x = self.conv_last(self.upsample(self.tail(res)))
        x = x + y  
        x = self.add_mean(x) 
        return x


