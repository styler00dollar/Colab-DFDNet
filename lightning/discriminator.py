"""
discriminators.py (12-2-20)
https://github.com/victorca25/BasicSR/blob/master/codes/models/modules/architectures/discriminators.py
"""
import math
import torch
import torch.nn as nn
import torchvision
#from . import block as B
from block import conv_block
from torch.nn.utils import spectral_norm as SN
import pytorch_lightning as pl



####################
# Discriminator
####################


# VGG style Discriminator
class Discriminator_VGG(pl.LightningModule):
    def __init__(self, size, in_nc, base_nf, norm_type='batch', act_type='leakyrelu', mode='CNA', convtype='Conv2D', arch='ESRGAN'):
        super(Discriminator_VGG, self).__init__()

        conv_blocks = []
        conv_blocks.append(conv_block(  in_nc, base_nf, kernel_size=3, stride=1, norm_type=None, \
            act_type=act_type, mode=mode))
        conv_blocks.append(conv_block(base_nf, base_nf, kernel_size=4, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode))

        cur_size = size // 2
        cur_nc = base_nf
        while cur_size > 4:
            out_nc = cur_nc * 2 if cur_nc < 512 else cur_nc
            conv_blocks.append(conv_block(cur_nc, out_nc, kernel_size=3, stride=1, norm_type=norm_type, \
                act_type=act_type, mode=mode))
            conv_blocks.append(conv_block(out_nc, out_nc, kernel_size=4, stride=2, norm_type=norm_type, \
                act_type=act_type, mode=mode))
            cur_nc = out_nc
            cur_size //= 2

        self.features = sequential(*conv_blocks)

        # classifier
        if arch=='PPON':
            self.classifier = nn.Sequential(
                nn.Linear(cur_nc * cur_size * cur_size, 128), nn.LeakyReLU(0.2, True), nn.Linear(128, 1))
        else: #arch='ESRGAN':
            self.classifier = nn.Sequential(
                nn.Linear(cur_nc * cur_size * cur_size, 100), nn.LeakyReLU(0.2, True), nn.Linear(100, 1))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# VGG style Discriminator with input size 96*96
class Discriminator_VGG_96(pl.LightningModule):
    def __init__(self, in_nc, base_nf, norm_type='batch', act_type='leakyrelu', mode='CNA', convtype='Conv2D', arch='ESRGAN'):
        super(Discriminator_VGG_96, self).__init__()
        # features
        # hxw, c
        # 96, 64
        conv0 = conv_block(in_nc, base_nf, kernel_size=3, norm_type=None, act_type=act_type, \
            mode=mode)
        conv1 = conv_block(base_nf, base_nf, kernel_size=4, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        # 48, 64
        conv2 = conv_block(base_nf, base_nf*2, kernel_size=3, stride=1, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        conv3 = conv_block(base_nf*2, base_nf*2, kernel_size=4, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        # 24, 128
        conv4 = conv_block(base_nf*2, base_nf*4, kernel_size=3, stride=1, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        conv5 = conv_block(base_nf*4, base_nf*4, kernel_size=4, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        # 12, 256
        conv6 = conv_block(base_nf*4, base_nf*8, kernel_size=3, stride=1, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        conv7 = conv_block(base_nf*8, base_nf*8, kernel_size=4, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        # 6, 512
        conv8 = conv_block(base_nf*8, base_nf*8, kernel_size=3, stride=1, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        conv9 = conv_block(base_nf*8, base_nf*8, kernel_size=4, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        # 3, 512
        self.features = sequential(conv0, conv1, conv2, conv3, conv4, conv5, conv6, conv7, conv8,\
            conv9)

        # classifier
        if arch=='PPON':
            self.classifier = nn.Sequential(
                nn.Linear(512 * 3 * 3, 128), nn.LeakyReLU(0.2, True), nn.Linear(128, 1))
        else: #arch='ESRGAN':
            self.classifier = nn.Sequential(
                nn.Linear(512 * 3 * 3, 100), nn.LeakyReLU(0.2, True), nn.Linear(100, 1))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# VGG style Discriminator with input size 128*128, Spectral Normalization
class Discriminator_VGG_128_SN(pl.LightningModule):
    def __init__(self):
        super(Discriminator_VGG_128_SN, self).__init__()
        # features
        # hxw, c
        # 128, 64
        self.lrelu = nn.LeakyReLU(0.2, True)

        self.conv0 = spectral_norm(nn.Conv2d(3, 64, 3, 1, 1))
        self.conv1 = spectral_norm(nn.Conv2d(64, 64, 4, 2, 1))
        # 64, 64
        self.conv2 = spectral_norm(nn.Conv2d(64, 128, 3, 1, 1))
        self.conv3 = spectral_norm(nn.Conv2d(128, 128, 4, 2, 1))
        # 32, 128
        self.conv4 = spectral_norm(nn.Conv2d(128, 256, 3, 1, 1))
        self.conv5 = spectral_norm(nn.Conv2d(256, 256, 4, 2, 1))
        # 16, 256
        self.conv6 = spectral_norm(nn.Conv2d(256, 512, 3, 1, 1))
        self.conv7 = spectral_norm(nn.Conv2d(512, 512, 4, 2, 1))
        # 8, 512
        self.conv8 = spectral_norm(nn.Conv2d(512, 512, 3, 1, 1))
        self.conv9 = spectral_norm(nn.Conv2d(512, 512, 4, 2, 1))
        # 4, 512

        # classifier
        self.linear0 = spectral_norm(nn.Linear(512 * 4 * 4, 100))
        self.linear1 = spectral_norm(nn.Linear(100, 1))

    def forward(self, x):
        x = self.lrelu(self.conv0(x))
        x = self.lrelu(self.conv1(x))
        x = self.lrelu(self.conv2(x))
        x = self.lrelu(self.conv3(x))
        x = self.lrelu(self.conv4(x))
        x = self.lrelu(self.conv5(x))
        x = self.lrelu(self.conv6(x))
        x = self.lrelu(self.conv7(x))
        x = self.lrelu(self.conv8(x))
        x = self.lrelu(self.conv9(x))
        x = x.view(x.size(0), -1)
        x = self.lrelu(self.linear0(x))
        x = self.linear1(x)
        return x

# VGG style Discriminator with input size 128*128
class Discriminator_VGG_128(pl.LightningModule):
    def __init__(self, in_nc, base_nf, norm_type='batch', act_type='leakyrelu', mode='CNA', convtype='Conv2D', arch='ESRGAN'):
        super(Discriminator_VGG_128, self).__init__()
        # features
        # hxw, c
        # 128, 64
        conv0 = conv_block(in_nc, base_nf, kernel_size=3, norm_type=None, act_type=act_type, \
            mode=mode)
        conv1 = conv_block(base_nf, base_nf, kernel_size=4, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        # 64, 64
        conv2 = conv_block(base_nf, base_nf*2, kernel_size=3, stride=1, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        conv3 = conv_block(base_nf*2, base_nf*2, kernel_size=4, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        # 32, 128
        conv4 = conv_block(base_nf*2, base_nf*4, kernel_size=3, stride=1, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        conv5 = conv_block(base_nf*4, base_nf*4, kernel_size=4, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        # 16, 256
        conv6 = conv_block(base_nf*4, base_nf*8, kernel_size=3, stride=1, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        conv7 = conv_block(base_nf*8, base_nf*8, kernel_size=4, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        # 8, 512
        conv8 = conv_block(base_nf*8, base_nf*8, kernel_size=3, stride=1, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        conv9 = conv_block(base_nf*8, base_nf*8, kernel_size=4, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        # 4, 512
        self.features = sequential(conv0, conv1, conv2, conv3, conv4, conv5, conv6, conv7, conv8,\
            conv9)

        # classifier
        if arch=='PPON':
            self.classifier = nn.Sequential(
                nn.Linear(512 * 4 * 4, 128), nn.LeakyReLU(0.2, True), nn.Linear(128, 1))
        else: #arch='ESRGAN':
            self.classifier = nn.Sequential(
                nn.Linear(512 * 4 * 4, 100), nn.LeakyReLU(0.2, True), nn.Linear(100, 1))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# VGG style Discriminator with input size 192*192
class Discriminator_VGG_192(pl.LightningModule): #vic in PPON is called Discriminator_192
    def __init__(self, in_nc, base_nf, norm_type='batch', act_type='leakyrelu', mode='CNA', convtype='Conv2D', arch='ESRGAN'):
        super(Discriminator_VGG_192, self).__init__()
        # features
        # hxw, c
        # 192, 64
        conv0 = conv_block(in_nc, base_nf, kernel_size=3, norm_type=None, act_type=act_type, \
            mode=mode) # 3-->64
        conv1 = conv_block(base_nf, base_nf, kernel_size=4, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode) # 64-->64, 96*96
        # 96, 64
        conv2 = conv_block(base_nf, base_nf*2, kernel_size=3, stride=1, norm_type=norm_type, \
            act_type=act_type, mode=mode) # 64-->128
        conv3 = conv_block(base_nf*2, base_nf*2, kernel_size=4, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode) # 128-->128, 48*48
        # 48, 128
        conv4 = conv_block(base_nf*2, base_nf*4, kernel_size=3, stride=1, norm_type=norm_type, \
            act_type=act_type, mode=mode) # 128-->256
        conv5 = conv_block(base_nf*4, base_nf*4, kernel_size=4, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode) # 256-->256, 24*24
        # 24, 256
        conv6 = conv_block(base_nf*4, base_nf*8, kernel_size=3, stride=1, norm_type=norm_type, \
            act_type=act_type, mode=mode) # 256-->512
        conv7 = conv_block(base_nf*8, base_nf*8, kernel_size=4, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode) # 512-->512 12*12
        # 12, 512
        conv8 = conv_block(base_nf*8, base_nf*8, kernel_size=3, stride=1, norm_type=norm_type, \
            act_type=act_type, mode=mode) # 512-->512
        conv9 = conv_block(base_nf*8, base_nf*8, kernel_size=4, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode) # 512-->512 6*6
        # 6, 512
        conv10 = conv_block(base_nf*8, base_nf*8, kernel_size=3, stride=1, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        conv11 = conv_block(base_nf*8, base_nf*8, kernel_size=4, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode) # 3*3
        # 3, 512
        self.features = sequential(conv0, conv1, conv2, conv3, conv4, conv5, conv6, conv7, conv8,\
            conv9, conv10, conv11)

        # classifier
        if arch=='PPON':
            self.classifier = nn.Sequential(
                nn.Linear(512 * 3 * 3, 128), nn.LeakyReLU(0.2, True), nn.Linear(128, 1)) #vic PPON uses 128 and 128 instead of 100
        else: #arch='ESRGAN':
            self.classifier = nn.Sequential(
                nn.Linear(512 * 3 * 3, 100), nn.LeakyReLU(0.2, True), nn.Linear(100, 1))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# VGG style Discriminator with input size 256*256
class Discriminator_VGG_256(pl.LightningModule):
    def __init__(self, in_nc, base_nf, norm_type='batch', act_type='leakyrelu', mode='CNA', convtype='Conv2D', arch='ESRGAN'):
        super(Discriminator_VGG_256, self).__init__()
        # features
        # hxw, c
        # 256, 64
        conv0 = conv_block(in_nc, base_nf, kernel_size=3, norm_type=None, act_type=act_type, \
            mode=mode)
        conv1 = conv_block(base_nf, base_nf, kernel_size=4, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        # 128, 64
        conv2 = conv_block(base_nf, base_nf*2, kernel_size=3, stride=1, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        conv3 = conv_block(base_nf*2, base_nf*2, kernel_size=4, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        # 64, 128
        conv4 = conv_block(base_nf*2, base_nf*4, kernel_size=3, stride=1, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        conv5 = conv_block(base_nf*4, base_nf*4, kernel_size=4, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        # 32, 256
        conv6 = conv_block(base_nf*4, base_nf*8, kernel_size=3, stride=1, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        conv7 = conv_block(base_nf*8, base_nf*8, kernel_size=4, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        # 16, 512
        conv8 = conv_block(base_nf*8, base_nf*8, kernel_size=3, stride=1, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        conv9 = conv_block(base_nf*8, base_nf*8, kernel_size=4, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        # 8, 512
        conv10 = conv_block(base_nf*8, base_nf*8, kernel_size=3, stride=1, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        conv11 = conv_block(base_nf*8, base_nf*8, kernel_size=4, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode) # 3*3
        # 4, 512
        self.features = sequential(conv0, conv1, conv2, conv3, conv4, conv5, conv6, conv7, conv8,\
            conv9, conv10, conv11)

        # classifier
        if arch=='PPON':
            self.classifier = nn.Sequential(
                nn.Linear(512 * 4 * 4, 128), nn.LeakyReLU(0.2, True), nn.Linear(128, 1))
        else: #arch='ESRGAN':
            self.classifier = nn.Sequential(
                nn.Linear(512 * 4 * 4, 100), nn.LeakyReLU(0.2, True), nn.Linear(100, 1))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


####################
# Perceptual Network
####################


# Assume input range is [0, 1]
class VGGFeatureExtractor(pl.LightningModule):
    def __init__(self,
                 feature_layer=34,
                 use_bn=False,
                 use_input_norm=True,
                 device=torch.device('cpu'),
                 z_norm=False): #Note: PPON uses cuda instead of CPU
        super(VGGFeatureExtractor, self).__init__()
        if use_bn:
            model = torchvision.models.vgg19_bn(pretrained=True)
        else:
            model = torchvision.models.vgg19(pretrained=True)
        self.use_input_norm = use_input_norm
        if self.use_input_norm:
            if z_norm: # if input in range [-1,1]
                mean = torch.Tensor([0.485-1, 0.456-1, 0.406-1]).view(1, 3, 1, 1).to(device)
                std = torch.Tensor([0.229*2, 0.224*2, 0.225*2]).view(1, 3, 1, 1).to(device)
            else: # input in range [0,1]
                mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
                std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
            self.register_buffer('mean', mean)
            self.register_buffer('std', std)
        self.features = nn.Sequential(*list(model.features.children())[:(feature_layer + 1)])
        # No need to BP to variable
        for k, v in self.features.named_parameters():
            v.requires_grad = False

    def forward(self, x):
        if self.use_input_norm:
            x = (x - self.mean) / self.std
        output = self.features(x)
        return output

# Assume input range is [0, 1]
class ResNet101FeatureExtractor(pl.LightningModule):
    def __init__(self, use_input_norm=True, device=torch.device('cpu'), z_norm=False):
        super(ResNet101FeatureExtractor, self).__init__()
        model = torchvision.models.resnet101(pretrained=True)
        self.use_input_norm = use_input_norm
        if self.use_input_norm:
            if z_norm: # if input in range [-1,1]
                mean = torch.Tensor([0.485-1, 0.456-1, 0.406-1]).view(1, 3, 1, 1).to(device)
                std = torch.Tensor([0.229*2, 0.224*2, 0.225*2]).view(1, 3, 1, 1).to(device)
            else: # input in range [0,1]
                mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
                std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
            self.register_buffer('mean', mean)
            self.register_buffer('std', std)
        self.features = nn.Sequential(*list(model.children())[:8])
        # No need to BP to variable
        for k, v in self.features.named_parameters():
            v.requires_grad = False

    def forward(self, x):
        if self.use_input_norm:
            x = (x - self.mean) / self.std
        output = self.features(x)
        return output


class MINCNet(pl.LightningModule):
    def __init__(self):
        super(MINCNet, self).__init__()
        self.ReLU = nn.ReLU(True)
        self.conv11 = nn.Conv2d(3, 64, 3, 1, 1)
        self.conv12 = nn.Conv2d(64, 64, 3, 1, 1)
        self.maxpool1 = nn.MaxPool2d(2, stride=2, padding=0, ceil_mode=True)
        self.conv21 = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv22 = nn.Conv2d(128, 128, 3, 1, 1)
        self.maxpool2 = nn.MaxPool2d(2, stride=2, padding=0, ceil_mode=True)
        self.conv31 = nn.Conv2d(128, 256, 3, 1, 1)
        self.conv32 = nn.Conv2d(256, 256, 3, 1, 1)
        self.conv33 = nn.Conv2d(256, 256, 3, 1, 1)
        self.maxpool3 = nn.MaxPool2d(2, stride=2, padding=0, ceil_mode=True)
        self.conv41 = nn.Conv2d(256, 512, 3, 1, 1)
        self.conv42 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv43 = nn.Conv2d(512, 512, 3, 1, 1)
        self.maxpool4 = nn.MaxPool2d(2, stride=2, padding=0, ceil_mode=True)
        self.conv51 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv52 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv53 = nn.Conv2d(512, 512, 3, 1, 1)

    def forward(self, x):
        out = self.ReLU(self.conv11(x))
        out = self.ReLU(self.conv12(out))
        out = self.maxpool1(out)
        out = self.ReLU(self.conv21(out))
        out = self.ReLU(self.conv22(out))
        out = self.maxpool2(out)
        out = self.ReLU(self.conv31(out))
        out = self.ReLU(self.conv32(out))
        out = self.ReLU(self.conv33(out))
        out = self.maxpool3(out)
        out = self.ReLU(self.conv41(out))
        out = self.ReLU(self.conv42(out))
        out = self.ReLU(self.conv43(out))
        out = self.maxpool4(out)
        out = self.ReLU(self.conv51(out))
        out = self.ReLU(self.conv52(out))
        out = self.conv53(out)
        return out


# Assume input range is [0, 1]
class MINCFeatureExtractor(pl.LightningModule):
    def __init__(self, feature_layer=34, use_bn=False, use_input_norm=True, \
                device=torch.device('cpu')):
        super(MINCFeatureExtractor, self).__init__()

        self.features = MINCNet()
        self.features.load_state_dict(
            torch.load('../experiments/pretrained_models/VGG16minc_53.pth'), strict=True)
        self.features.eval()
        # No need to BP to variable
        for k, v in self.features.named_parameters():
            v.requires_grad = False

    def forward(self, x):
        output = self.features(x)
        return output


#TODO
# moved from models.modules.architectures.ASRResNet_arch, did not bring the self-attention layer
# VGG style Discriminator with input size 128*128, with feature_maps extraction and self-attention
class Discriminator_VGG_128_fea(pl.LightningModule):
    def __init__(self, in_nc, base_nf, norm_type='batch', act_type='leakyrelu', mode='CNA', convtype='Conv2D', \
         arch='ESRGAN', spectral_norm=False, self_attention = False, max_pool=False, poolsize = 4):
        super(Discriminator_VGG_128_fea, self).__init__()
        # features
        # hxw, c
        # 128, 64

        # Self-Attention configuration
        '''#TODO
        self.self_attention = self_attention
        self.max_pool = max_pool
        self.poolsize = poolsize
        '''

        # Remove BatchNorm2d if using spectral_norm
        if spectral_norm:
            norm_type = None

        self.conv0 = conv_block(in_nc, base_nf, kernel_size=3, norm_type=None, act_type=act_type, \
            mode=mode)
        self.conv1 = conv_block(base_nf, base_nf, kernel_size=4, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode, spectral_norm=spectral_norm)
        # 64, 64
        self.conv2 = conv_block(base_nf, base_nf*2, kernel_size=3, stride=1, norm_type=norm_type, \
            act_type=act_type, mode=mode, spectral_norm=spectral_norm)
        self.conv3 = conv_block(base_nf*2, base_nf*2, kernel_size=4, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode, spectral_norm=spectral_norm)
        # 32, 128
        self.conv4 = conv_block(base_nf*2, base_nf*4, kernel_size=3, stride=1, norm_type=norm_type, \
            act_type=act_type, mode=mode, spectral_norm=spectral_norm)
        self.conv5 = conv_block(base_nf*4, base_nf*4, kernel_size=4, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode, spectral_norm=spectral_norm)
        # 16, 256

        '''#TODO
        if self.self_attention:
            self.FSA = SelfAttentionBlock(in_dim = base_nf*4, max_pool=self.max_pool, poolsize = self.poolsize, spectral_norm=spectral_norm)
        '''

        self.conv6 = conv_block(base_nf*4, base_nf*8, kernel_size=3, stride=1, norm_type=norm_type, \
            act_type=act_type, mode=mode, spectral_norm=spectral_norm)
        self.conv7 = conv_block(base_nf*8, base_nf*8, kernel_size=4, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode, spectral_norm=spectral_norm)
        # 8, 512
        self.conv8 = conv_block(base_nf*8, base_nf*8, kernel_size=3, stride=1, norm_type=norm_type, \
            act_type=act_type, mode=mode, spectral_norm=spectral_norm)
        self.conv9 = conv_block(base_nf*8, base_nf*8, kernel_size=4, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode, spectral_norm=spectral_norm)
        # 4, 512
        # self.features = sequential(conv0, conv1, conv2, conv3, conv4, conv5, conv6, conv7, conv8,\
            # conv9)

        # classifier
        if arch=='PPON':
            self.classifier = nn.Sequential(
                nn.Linear(512 * 4 * 4, 128), nn.LeakyReLU(0.2, True), nn.Linear(128, 1))
        else: #arch='ESRGAN':
            self.classifier = nn.Sequential(
                nn.Linear(512 * 4 * 4, 100), nn.LeakyReLU(0.2, True), nn.Linear(100, 1))

    #TODO: modify to a listening dictionary like VGG_Model(), can select what maps to use
    def forward(self, x, return_maps=False):
        feature_maps = []
        # x = self.features(x)
        x = self.conv0(x)
        feature_maps.append(x)
        x = self.conv1(x)
        feature_maps.append(x)
        x = self.conv2(x)
        feature_maps.append(x)
        x = self.conv3(x)
        feature_maps.append(x)
        x = self.conv4(x)
        feature_maps.append(x)
        x = self.conv5(x)
        feature_maps.append(x)
        x = self.conv6(x)
        feature_maps.append(x)
        x = self.conv7(x)
        feature_maps.append(x)
        x = self.conv8(x)
        feature_maps.append(x)
        x = self.conv9(x)
        feature_maps.append(x)

        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        if return_maps:
            return [x, feature_maps]
        return x


class Discriminator_VGG_fea(pl.LightningModule):
    def __init__(self, size, in_nc, base_nf, norm_type='batch', act_type='leakyrelu', mode='CNA', convtype='Conv2D', \
         arch='ESRGAN', spectral_norm=False, self_attention = False, max_pool=False, poolsize = 4):
        super(Discriminator_VGG_fea, self).__init__()
        # features
        # hxw, c
        # 128, 64

        # Self-Attention configuration
        '''#TODO
        self.self_attention = self_attention
        self.max_pool = max_pool
        self.poolsize = poolsize
        '''

        # Remove BatchNorm2d if using spectral_norm
        if spectral_norm:
            norm_type = None

        self.conv_blocks = []
        self.conv_blocks.append(conv_block(in_nc, base_nf, kernel_size=3, norm_type=None, \
            act_type=act_type, mode=mode, spectral_norm=spectral_norm))
        self.conv_blocks.append(conv_block(base_nf, base_nf, kernel_size=4, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode, spectral_norm=spectral_norm))

        cur_size = size // 2
        cur_nc = base_nf
        while cur_size > 4:
            out_nc = cur_nc * 2 if cur_nc < 512 else cur_nc
            self.conv_blocks.append(conv_block(cur_nc, out_nc, kernel_size=3, stride=1, norm_type=norm_type, \
                act_type=act_type, mode=mode, spectral_norm=spectral_norm))
            self.conv_blocks.append(conv_block(out_nc, out_nc, kernel_size=4, stride=2, norm_type=norm_type, \
                act_type=act_type, mode=mode, spectral_norm=spectral_norm))
            cur_nc = out_nc
            cur_size //= 2

        '''#TODO
        if self.self_attention:
            self.FSA = SelfAttentionBlock(in_dim = base_nf*4, max_pool=self.max_pool, poolsize = self.poolsize, spectral_norm=spectral_norm)
        '''

        # self.features = sequential(*conv_blocks)

        # classifier
        if arch=='PPON':
            self.classifier = nn.Sequential(
                nn.Linear(cur_nc * cur_size * cur_size, 128), nn.LeakyReLU(0.2, True), nn.Linear(128, 1))
        else: #arch='ESRGAN':
            self.classifier = nn.Sequential(
                nn.Linear(cur_nc * cur_size * cur_size, 100), nn.LeakyReLU(0.2, True), nn.Linear(100, 1))

    #TODO: modify to a listening dictionary like VGG_Model(), can select what maps to use
    def forward(self, x, return_maps=False):
        feature_maps = []
        # x = self.features(x)
        for conv in self.conv_blocks:
            # Fixes incorrect device error
            device = x.device
            conv = conv.to(device)
            x = conv(x)
            feature_maps.append(x)

        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        if return_maps:
            return [x, feature_maps]
        return x


class NLayerDiscriminator(pl.LightningModule):
    r"""
    PatchGAN discriminator
    https://arxiv.org/pdf/1611.07004v3.pdf
    https://arxiv.org/pdf/1803.07422.pdf

    """

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d,
        use_sigmoid=False, getIntermFeat=False, patch=True, use_spectral_norm=False):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int): the number of channels in input images
            ndf (int): the number of filters in the last conv layer
            n_layers (int): the number of conv layers in the discriminator
            norm_layer (nn.Module): normalization layer (if not using Spectral Norm)
            patch (bool): Select between an patch or a linear output
            use_spectral_norm (bool): Select if Spectral Norm will be used
        """
        super(NLayerDiscriminator, self).__init__()
        '''
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        '''

        if use_spectral_norm:
            # disable Instance or Batch Norm if using Spectral Norm
            norm_layer = Identity

        #self.getIntermFeat = getIntermFeat # not used for now
        #use_sigmoid not used for now
        #TODO: test if there are benefits by incorporating the use of intermediate features from pix2pixHD

        use_bias = False
        kw = 4
        padw = 1 # int(np.ceil((kw-1.0)/2))

        sequence = [add_spectral_norm(
                        nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
                        use_spectral_norm),
                    nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                add_spectral_norm(
                    nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                    use_spectral_norm),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            add_spectral_norm(
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
                use_spectral_norm),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        if patch:
            # output patches as results
            sequence += [add_spectral_norm(
                nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw),
                use_spectral_norm)]  # output 1 channel prediction map
        else:
            # linear vector classification output
            sequence += [Mean([1, 2]), nn.Linear(ndf * nf_mult, 1)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        """Standard forward."""
        return self.model(x)


class MultiscaleDiscriminator(pl.LightningModule):
    r"""
    Multiscale PatchGAN discriminator
    https://arxiv.org/pdf/1711.11585.pdf

    """
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d,
                 use_sigmoid=False, num_D=3, getIntermFeat=False):
        """Construct a pyramid of PatchGAN discriminators
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
            use_sigmoid     -- boolean to use sigmoid in patchGAN discriminators
            num_D (int)     -- number of discriminators/downscales in the pyramid
            getIntermFeat   -- boolean to get intermediate features (unused for now)
        """
        super(MultiscaleDiscriminator, self).__init__()
        self.num_D = num_D
        self.n_layers = n_layers
        self.getIntermFeat = getIntermFeat

        for i in range(num_D):
            netD = NLayerDiscriminator(input_nc, ndf, n_layers, norm_layer, use_sigmoid, getIntermFeat)
            if getIntermFeat:
                for j in range(n_layers+2):
                    setattr(self, 'scale'+str(i)+'_layer'+str(j), getattr(netD, 'model'+str(j)))
            else:
                setattr(self, 'layer'+str(i), netD.model)

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def singleD_forward(self, model, input):
        if self.getIntermFeat:
            result = [input]
            for i in range(len(model)):
                result.append(model[i](result[-1]))
            return result[1:]
        else:
            return [model(input)]

    def forward(self, input):
        num_D = self.num_D
        result = []
        input_downsampled = input
        for i in range(num_D):
            if self.getIntermFeat:
                model = [getattr(self, 'scale'+str(num_D-1-i)+'_layer'+str(j)) for j in range(self.n_layers+2)]
            else:
                model = getattr(self, 'layer'+str(num_D-1-i))
            result.append(self.singleD_forward(model, input_downsampled))
            if i != (num_D-1):
                input_downsampled = self.downsample(input_downsampled)
        return result


class PixelDiscriminator(pl.LightningModule):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        """Construct a 1x1 PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(PixelDiscriminator, self).__init__()
        '''
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        '''
        use_bias = False

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        """Standard forward."""
        return self.net(input)


"""
models.py (21-12-20)
https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/context_encoder/models.py
"""

class context_encoder(pl.LightningModule):
    def __init__(self, channels=3):
        super(context_encoder, self).__init__()

        def discriminator_block(in_filters, out_filters, stride, normalize):
            """Returns layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 3, stride, 1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        layers = []
        in_filters = channels
        for out_filters, stride, normalize in [(64, 2, False), (128, 2, True), (256, 2, True), (512, 1, True)]:
            layers.extend(discriminator_block(in_filters, out_filters, stride, normalize))
            in_filters = out_filters

        layers.append(nn.Conv2d(out_filters, 1, 3, 1, 1))

        self.model = nn.Sequential(*layers)

    def forward(self, img):
        return self.model(img)


"""
discriminators.py (12-2-20)
https://github.com/JoeyBallentine/BasicSR/blob/resnet-discriminator/codes/models/modules/architectures/discriminators.py
"""
# Assume input range is [0, 1]
class ResNet101FeatureExtractor(pl.LightningModule):
    def __init__(self, use_input_norm=True, device=torch.device('cpu'), z_norm=False):
        super(ResNet101FeatureExtractor, self).__init__()
        model = torchvision.models.resnet101(pretrained=True)
        self.use_input_norm = use_input_norm
        if self.use_input_norm:
            if z_norm: # if input in range [-1,1]
                mean = torch.Tensor([0.485-1, 0.456-1, 0.406-1]).view(1, 3, 1, 1).to(device)
                std = torch.Tensor([0.229*2, 0.224*2, 0.225*2]).view(1, 3, 1, 1).to(device)
            else: # input in range [0,1]
                mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
                std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
            self.register_buffer('mean', mean)
            self.register_buffer('std', std)
        self.features = nn.Sequential(*list(model.children())[:8])
        # No need to BP to variable
        for k, v in self.features.named_parameters():
            v.requires_grad = False

    def forward(self, x):
        if self.use_input_norm:
            x = (x - self.mean) / self.std
        output = self.features(x)
        return output


# ResNet50 style Discriminator with input size 128*128
class Discriminator_ResNet_128(pl.LightningModule):
    """
    Structure based off of the ResNet50 configuration from this repository:
    https://github.com/bentrevett/pytorch-image-classification
    """
    def __init__(self, in_nc, base_nf, norm_type='batch', act_type='leakyrelu', mode='CNA'):
        super(Discriminator_ResNet_128, self).__init__()
        # features
        # hxw, c

        self.in_channels = base_nf

        # 128, 3
        conv0 = conv_block(in_nc, self.in_channels, kernel_size=7, norm_type=norm_type, act_type=act_type, \
            mode=mode, stride=2, bias=False)
        pool0 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # 32, 64

        layer1 = self.get_resnet_layer(Bottleneck, 3, base_nf) # 32, 64
        layer2 = self.get_resnet_layer(Bottleneck, 4, base_nf*2, stride = 2) # 16, 128
        layer3 = self.get_resnet_layer(Bottleneck, 6, base_nf*4, stride = 2) # 8, 256
        layer4 = self.get_resnet_layer(Bottleneck, 3, base_nf*8, stride = 2) # 4, 512

        avgpool = nn.AdaptiveAvgPool2d((1,1))

        self.features = sequential(conv0, pool0, layer1, layer2, layer3, layer4, avgpool)

        self.classifier = nn.Linear(self.in_channels, 1)

    def get_resnet_layer(self, block, n_blocks, channels, stride = 1):

        layers = []

        if self.in_channels != block.expansion * channels:
            downsample = True
        else:
            downsample = False

        layers.append(block(self.in_channels, channels, stride, downsample))

        for i in range(1, n_blocks):
            layers.append(block(block.expansion * channels, channels))

        self.in_channels = block.expansion * channels

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class Bottleneck(pl.LightningModule):

    expansion = 4

    def __init__(self, in_channels, out_channels, stride = 1, downsample = False):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 1,
                               stride = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3,
                               stride = stride, padding = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(out_channels, self.expansion * out_channels, kernel_size = 1,
                               stride = 1, bias = False)
        self.bn3 = nn.BatchNorm2d(self.expansion * out_channels)

        self.relu = nn.ReLU(inplace = True)

        if downsample:
            conv = nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size = 1,
                             stride = stride, bias = False)
            bn = nn.BatchNorm2d(self.expansion * out_channels)
            downsample = nn.Sequential(conv, bn)
        else:
            downsample = None

        self.downsample = downsample

    def forward(self, x):

        i = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)

        if self.downsample is not None:
            i = self.downsample(i)

        x += i
        x = self.relu(x)

        return x
