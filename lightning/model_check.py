import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import torch.nn.functional as F
from torch.nn import Parameter as P
#from util import util
from torchvision import models
#import scipy.io as sio
import numpy as np
#import scipy.ndimage
import torch.nn.utils.spectral_norm as SpectralNorm
#from spectral_norm import SpectralNorm

from torch.autograd import Function
from math import sqrt
import random
import os
import math

import pytorch_lightning as pl

from functions import *


class VGGFeat(pl.LightningModule):
    """
    Input: (B, C, H, W), RGB, [-1, 1]
    """
    def __init__(self, weight_path='/content/DFDNet/weights/vgg19.pth'):
        super().__init__()
        self.model = models.vgg19(pretrained=False)
        self.build_vgg_layers()

        self.model.load_state_dict(torch.load(weight_path))

        self.register_parameter("RGB_mean", nn.Parameter(torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)))
        self.register_parameter("RGB_std", nn.Parameter(torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)))

        # self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def build_vgg_layers(self):
        vgg_pretrained_features = self.model.features
        self.features = []
        # feature_layers = [0, 3, 8, 17, 26, 35]
        feature_layers = [0, 8, 17, 26, 35]
        for i in range(len(feature_layers)-1):
            module_layers = torch.nn.Sequential()
            for j in range(feature_layers[i], feature_layers[i+1]):
                module_layers.add_module(str(j), vgg_pretrained_features[j])
            self.features.append(module_layers)
        self.features = torch.nn.ModuleList(self.features)

    def preprocess(self, x):
        x = (x + 1) / 2
        x = (x - self.RGB_mean) / self.RGB_std
        if x.shape[3] < 224:
            x = torch.nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        return x

    def forward(self, x):
        x = self.preprocess(x)
        features = []
        for m in self.features:
            # print(m)
            x = m(x)
            features.append(x)
        return features


class StyledUpBlock(pl.LightningModule):
    def __init__(self, in_channel, out_channel, kernel_size=3, padding=1,upsample=False):
        super().__init__()
        if upsample:
            self.conv1 = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                Blur(out_channel),
                # EqualConv2d(in_channel, out_channel, kernel_size, padding=padding),
                SpectralNorm(nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding)),
                nn.LeakyReLU(0.2),
            )
        else:
            self.conv1 = nn.Sequential(
                Blur(in_channel),
                # EqualConv2d(in_channel, out_channel, kernel_size, padding=padding)
                SpectralNorm(nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding)),
                nn.LeakyReLU(0.2),
            )
        self.convup = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                # EqualConv2d(out_channel, out_channel, kernel_size, padding=padding),
                SpectralNorm(nn.Conv2d(out_channel, out_channel, kernel_size, padding=padding)),
                nn.LeakyReLU(0.2),
                # Blur(out_channel),
            )
        # self.noise1 = equal_lr(NoiseInjection(out_channel))
        # self.adain1 = AdaptiveInstanceNorm(out_channel)
        self.lrelu1 = nn.LeakyReLU(0.2)

        # self.conv2 = EqualConv2d(out_channel, out_channel, kernel_size, padding=padding)
        # self.noise2 = equal_lr(NoiseInjection(out_channel))
        # self.adain2 = AdaptiveInstanceNorm(out_channel)
        # self.lrelu2 = nn.LeakyReLU(0.2)

        self.ScaleModel1 = nn.Sequential(
            # Blur(in_channel),
            SpectralNorm(nn.Conv2d(in_channel,out_channel,3, 1, 1)),
            # nn.Conv2d(in_channel,out_channel,3, 1, 1),
            nn.LeakyReLU(0.2, True),
            SpectralNorm(nn.Conv2d(out_channel, out_channel, 3, 1, 1))
            # nn.Conv2d(out_channel, out_channel, 3, 1, 1)
        )
        self.ShiftModel1 = nn.Sequential(
            # Blur(in_channel),
            SpectralNorm(nn.Conv2d(in_channel,out_channel,3, 1, 1)),
            # nn.Conv2d(in_channel,out_channel,3, 1, 1),
            nn.LeakyReLU(0.2, True),
            SpectralNorm(nn.Conv2d(out_channel, out_channel, 3, 1, 1)),
            nn.Sigmoid(),
            # nn.Conv2d(out_channel, out_channel, 3, 1, 1)
        )

    def forward(self, input, style):
        out = self.conv1(input)
#         out = self.noise1(out, noise)
        out = self.lrelu1(out)

        Shift1 = self.ShiftModel1(style)
        Scale1 = self.ScaleModel1(style)
        out = out * Scale1 + Shift1
        # out = self.adain1(out, style)
        outup = self.convup(out)

        return outup





class StyledUpBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, padding=1,upsample=False):
        super().__init__()
        if upsample:
            self.conv1 = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                Blur(out_channel),
                # EqualConv2d(in_channel, out_channel, kernel_size, padding=padding),
                SpectralNorm(nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding)),
                nn.LeakyReLU(0.2),
            )
        else:
            self.conv1 = nn.Sequential(
                Blur(in_channel),
                # EqualConv2d(in_channel, out_channel, kernel_size, padding=padding)
                SpectralNorm(nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding)),
                nn.LeakyReLU(0.2),
            )
        self.convup = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                # EqualConv2d(out_channel, out_channel, kernel_size, padding=padding),
                SpectralNorm(nn.Conv2d(out_channel, out_channel, kernel_size, padding=padding)),
                nn.LeakyReLU(0.2),
                # Blur(out_channel),
            )
        # self.noise1 = equal_lr(NoiseInjection(out_channel))
        # self.adain1 = AdaptiveInstanceNorm(out_channel)
        self.lrelu1 = nn.LeakyReLU(0.2)

        # self.conv2 = EqualConv2d(out_channel, out_channel, kernel_size, padding=padding)
        # self.noise2 = equal_lr(NoiseInjection(out_channel))
        # self.adain2 = AdaptiveInstanceNorm(out_channel)
        # self.lrelu2 = nn.LeakyReLU(0.2)

        self.ScaleModel1 = nn.Sequential(
            # Blur(in_channel),
            SpectralNorm(nn.Conv2d(in_channel,out_channel,3, 1, 1)),
            # nn.Conv2d(in_channel,out_channel,3, 1, 1),
            nn.LeakyReLU(0.2, True),
            SpectralNorm(nn.Conv2d(out_channel, out_channel, 3, 1, 1))
            # nn.Conv2d(out_channel, out_channel, 3, 1, 1)
        )
        self.ShiftModel1 = nn.Sequential(
            # Blur(in_channel),
            SpectralNorm(nn.Conv2d(in_channel,out_channel,3, 1, 1)),
            # nn.Conv2d(in_channel,out_channel,3, 1, 1),
            nn.LeakyReLU(0.2, True),
            SpectralNorm(nn.Conv2d(out_channel, out_channel, 3, 1, 1)),
            nn.Sigmoid(),
            # nn.Conv2d(out_channel, out_channel, 3, 1, 1)
        )

    def forward(self, input, style):
        out = self.conv1(input)
#         out = self.noise1(out, noise)
        out = self.lrelu1(out)

        Shift1 = self.ShiftModel1(style)
        Scale1 = self.ScaleModel1(style)
        out = out * Scale1 + Shift1
        # out = self.adain1(out, style)
        outup = self.convup(out)

        return outup

class UNetDictFace(pl.LightningModule):
    def __init__(self, ngf=64, dictionary_path='/content/DFDNet/DictionaryCenter512'):
        super().__init__()

        self.part_sizes = np.array([80,80,50,110]) # size for 512
        self.feature_sizes = np.array([256,128,64,32])
        self.channel_sizes = np.array([128,256,512,512])
        Parts = ['left_eye','right_eye','nose','mouth']
        self.Dict_256 = {}
        self.Dict_128 = {}
        self.Dict_64 = {}
        self.Dict_32 = {}
        for j,i in enumerate(Parts):
            f_256 = torch.from_numpy(np.load(os.path.join(dictionary_path, '{}_256_center.npy'.format(i)), allow_pickle=True))

            f_256_reshape = f_256.reshape(f_256.size(0),self.channel_sizes[0],self.part_sizes[j]//2,self.part_sizes[j]//2)
            max_256 = torch.max(torch.sqrt(compute_sum(torch.pow(f_256_reshape, 2), axis=[1, 2, 3], keepdim=True)),torch.FloatTensor([1e-4]))
            self.Dict_256[i] = f_256_reshape #/ max_256

            f_128 = torch.from_numpy(np.load(os.path.join(dictionary_path, '{}_128_center.npy'.format(i)), allow_pickle=True))

            f_128_reshape = f_128.reshape(f_128.size(0),self.channel_sizes[1],self.part_sizes[j]//4,self.part_sizes[j]//4)
            max_128 = torch.max(torch.sqrt(compute_sum(torch.pow(f_128_reshape, 2), axis=[1, 2, 3], keepdim=True)),torch.FloatTensor([1e-4]))
            self.Dict_128[i] = f_128_reshape #/ max_128

            f_64 = torch.from_numpy(np.load(os.path.join(dictionary_path, '{}_64_center.npy'.format(i)), allow_pickle=True))

            f_64_reshape = f_64.reshape(f_64.size(0),self.channel_sizes[2],self.part_sizes[j]//8,self.part_sizes[j]//8)
            max_64 = torch.max(torch.sqrt(compute_sum(torch.pow(f_64_reshape, 2), axis=[1, 2, 3], keepdim=True)),torch.FloatTensor([1e-4]))
            self.Dict_64[i] = f_64_reshape #/ max_64

            f_32 = torch.from_numpy(np.load(os.path.join(dictionary_path, '{}_32_center.npy'.format(i)), allow_pickle=True))

            f_32_reshape = f_32.reshape(f_32.size(0),self.channel_sizes[3],self.part_sizes[j]//16,self.part_sizes[j]//16)
            max_32 = torch.max(torch.sqrt(compute_sum(torch.pow(f_32_reshape, 2), axis=[1, 2, 3], keepdim=True)),torch.FloatTensor([1e-4]))
            self.Dict_32[i] = f_32_reshape #/ max_32

        self.le_256 = AttentionBlock(128)
        self.le_128 = AttentionBlock(256)
        self.le_64 = AttentionBlock(512)
        self.le_32 = AttentionBlock(512)

        self.re_256 = AttentionBlock(128)
        self.re_128 = AttentionBlock(256)
        self.re_64 = AttentionBlock(512)
        self.re_32 = AttentionBlock(512)

        self.no_256 = AttentionBlock(128)
        self.no_128 = AttentionBlock(256)
        self.no_64 = AttentionBlock(512)
        self.no_32 = AttentionBlock(512)

        self.mo_256 = AttentionBlock(128)
        self.mo_128 = AttentionBlock(256)
        self.mo_64 = AttentionBlock(512)
        self.mo_32 = AttentionBlock(512)

        #norm
        self.VggExtract = VGGFeat()

        ######################
        self.MSDilate = MSDilateBlock(ngf*8, dilation = [4,3,2,1])  #

        self.up0 = StyledUpBlock(ngf*8,ngf*8)
        self.up1 = StyledUpBlock(ngf*8, ngf*4) #
        self.up2 = StyledUpBlock(ngf*4, ngf*2) #
        self.up3 = StyledUpBlock(ngf*2, ngf) #
        self.up4 = nn.Sequential( # 128
            # nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            SpectralNorm(nn.Conv2d(ngf, ngf, 3, 1, 1)),
            # nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            UpResBlock(ngf),
            UpResBlock(ngf),
            # SpectralNorm(nn.Conv2d(ngf, 3, kernel_size=3, stride=1, padding=1)),
            nn.Conv2d(ngf, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )
        self.to_rgb0 = ToRGB(ngf*8)
        self.to_rgb1 = ToRGB(ngf*4)
        self.to_rgb2 = ToRGB(ngf*2)
        self.to_rgb3 = ToRGB(ngf*1)

        # for param in self.BlurInputConv.parameters():
        #     param.requires_grad = False

    def forward(self, input, part_locations):
        VggFeatures = self.VggExtract(input)
        # for b in range(input.size(0)):
        b = 0
        delete_flag = 0
        UpdateVggFeatures = []
        for i, f_size in enumerate(self.feature_sizes):
            cur_feature = VggFeatures[i]

            update_feature = cur_feature.clone() #* 0
            cur_part_sizes = self.part_sizes // (512/f_size)
            dicts_feature = getattr(self, 'Dict_'+str(f_size))

            LE_Dict_feature = dicts_feature['left_eye'].to(input)
            RE_Dict_feature = dicts_feature['right_eye'].to(input)
            NO_Dict_feature = dicts_feature['nose'].to(input)
            MO_Dict_feature = dicts_feature['mouth'].to(input)

            le_location = (part_locations[0][b] // (512/f_size)).int()
            re_location = (part_locations[1][b] // (512/f_size)).int()
            no_location = (part_locations[2][b] // (512/f_size)).int()
            mo_location = (part_locations[3][b] // (512/f_size)).int()

            LE_feature = cur_feature[:,:,le_location[1]:le_location[3],le_location[0]:le_location[2]].clone()
            RE_feature = cur_feature[:,:,re_location[1]:re_location[3],re_location[0]:re_location[2]].clone()
            NO_feature = cur_feature[:,:,no_location[1]:no_location[3],no_location[0]:no_location[2]].clone()
            MO_feature = cur_feature[:,:,mo_location[1]:mo_location[3],mo_location[0]:mo_location[2]].clone()

            #resize
            # check
            # good: torch.Size([1, 128, 45, 46])
            # bad:  torch.Size([1, 128, 36, 0])

            if LE_feature.shape[2] == 0 or LE_feature.shape[3] == 0:
              delete_flag = 1
            if RE_feature.shape[2] == 0 or RE_feature.shape[3] == 0:
              delete_flag = 1
            if NO_feature.shape[2] == 0 or NO_feature.shape[3] == 0:
              delete_flag = 1
            if MO_feature.shape[2] == 0 or MO_feature.shape[3] == 0:
              delete_flag = 1

        return delete_flag




# smaller blocks
def AttentionBlock(in_channel):
    return nn.Sequential(
        SpectralNorm(nn.Conv2d(in_channel, in_channel, 3, 1, 1)),
        nn.LeakyReLU(0.2),
        SpectralNorm(nn.Conv2d(in_channel, in_channel, 3, 1, 1))
    )


class MSDilateBlock(pl.LightningModule):
    def __init__(self, in_channels,conv_layer=nn.Conv2d, norm_layer=nn.BatchNorm2d, kernel_size=3, dilation=[1,1,1,1], bias=True):
        super(MSDilateBlock, self).__init__()
        self.conv1 =  convU(in_channels, in_channels,conv_layer, norm_layer, kernel_size,dilation=dilation[0], bias=bias)
        self.conv2 =  convU(in_channels, in_channels,conv_layer, norm_layer, kernel_size,dilation=dilation[1], bias=bias)
        self.conv3 =  convU(in_channels, in_channels,conv_layer, norm_layer, kernel_size,dilation=dilation[2], bias=bias)
        self.conv4 =  convU(in_channels, in_channels,conv_layer, norm_layer, kernel_size,dilation=dilation[3], bias=bias)
        self.convi =  SpectralNorm(conv_layer(in_channels*4, in_channels, kernel_size=kernel_size, stride=1, padding=(kernel_size-1)//2, bias=bias))
    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(x)
        conv3 = self.conv3(x)
        conv4 = self.conv4(x)
        cat  = torch.cat([conv1, conv2, conv3, conv4], 1)
        out = self.convi(cat) + x
        return out


class VggClassNet(pl.LightningModule):
    def __init__(self, select_layer = ['0','5','10','19']):
        super(VggClassNet, self).__init__()
        self.select = select_layer
        self.vgg = models.vgg19(pretrained=True).features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        features = []
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            if name in self.select:
                features.append(x)
        return features




class UpResBlock(pl.LightningModule):
    def __init__(self, dim, conv_layer = nn.Conv2d, norm_layer = nn.BatchNorm2d):
        super(UpResBlock, self).__init__()
        self.Model = nn.Sequential(
            # SpectralNorm(conv_layer(dim, dim, 3, 1, 1)),
            conv_layer(dim, dim, 3, 1, 1),
            # norm_layer(dim),
            nn.LeakyReLU(0.2,True),
            # SpectralNorm(conv_layer(dim, dim, 3, 1, 1)),
            conv_layer(dim, dim, 3, 1, 1),
        )
    def forward(self, x):
        out = x + self.Model(x)
        return out
