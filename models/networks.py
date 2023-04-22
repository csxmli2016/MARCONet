# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import warnings
warnings.filterwarnings('ignore')

from basicsr.ops.fused_act import FusedLeakyReLU, fused_leaky_relu

from models.textvit_arch import TextViT as TextEncoder
from models.resnet import resnet45stride as resnet45
import torch.nn.utils.spectral_norm as SpectralNorm

'''
This file defines the main network structures, including:
1) TextContextEncoderV2: using transformer encoder to predict the character labels, bounding boxes, and font style.
2) TSPGAN: Structure prior generation using StyleGAN
3) TSPSRNet: Text super-resolution using structure prior
'''


'''
1) Transformer encoder for predicting the character labels, bounding boxes, and font style.
'''
class TextContextEncoderV2(nn.Module):
    '''
    Input: LR image
    Output: character labels, character bounding boxes, and font style w
    '''
    def __init__(
        self,
        dim = 512,
        num_classes = 6736,
    ):
        super().__init__()
        self.resnet = resnet45()
        dim = 512
        max_length = 16 
        self.transformer = TextEncoder(num_classes=num_classes, dim=dim, max_length=max_length)
    def forward(self, lq):
        res_feature = self.resnet(lq) 
        logits, locs, w = self.transformer(res_feature) #
        return logits, locs, w
    

'''
2) Structure prior generation using StyleGAN
'''
class TSPGAN(nn.Module):
    def __init__(
        self,
        out_size=128,
        num_style_feat=512,
        class_num=6736,
        num_mlp=8,
    ):
        super().__init__()
        self.TextGenerator = TextGenerator(size=out_size, style_dim=num_style_feat, n_mlp=num_mlp, class_num=class_num)
    def forward(self, styles, labels, noise):
        return self.TextGenerator(styles, labels, noise)
    
class TextGenerator(nn.Module):
    '''
    Input: font style w and character labels
    Output: structure image, structure prior on 64*64 and 32*32
    '''
    def __init__(
        self,
        size,
        style_dim,
        n_mlp,
        class_num,
        channel_multiplier=1,
        blur_kernel=[1, 3, 3, 1],
        lr_mlp=0.01,
    ):
        super().__init__()
        self.size = size
        self.n_mlp = n_mlp
        self.style_dim = style_dim
        style_mlp_layers = [PixelNorm()]
        for i in range(n_mlp):
            style_mlp_layers.append(
                EqualLinear(
                    style_dim, style_dim, bias=True, bias_init_val=0, lr_mul=lr_mlp,
                    activation='fused_lrelu'))
        self.style_mlp = nn.Sequential(*style_mlp_layers)
        self.channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        self.input_text = SelectText(class_num, self.channels[4])
        self.conv1 = StyledConv(
            self.channels[4], self.channels[4], 3, style_dim, blur_kernel=blur_kernel
        )
        self.to_rgb1 = ToRGB(self.channels[4], style_dim, upsample=False)
        self.log_size = int(math.log(size, 2)) #7

        self.convs = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        in_channel = self.channels[4]

        for i in range(3, self.log_size + 1):
            out_channel = self.channels[2 ** i]
            self.convs.append(
                StyledConv(
                    in_channel,
                    out_channel,
                    3,
                    style_dim,
                    upsample=True,
                    blur_kernel=blur_kernel,
                )
            )
            self.convs.append(
                StyledConv(
                    out_channel, out_channel, 3, style_dim, blur_kernel=blur_kernel
                )
            )
            self.to_rgbs.append(ToRGB(out_channel, style_dim))
            in_channel = out_channel
        self.n_latent = self.log_size * 2 - 2
    def forward(
        self,
        styles,
        labels,
        noise=None,
    ):
        styles = self.style_mlp(styles)#
        latent = styles.unsqueeze(1).repeat(1, self.n_latent, 1)
        out = self.input_text(labels) #4*4

        out = self.conv1(out, latent[:, 0], noise=None)
        skip = self.to_rgb1(out, latent[:, 1])
        i = 1
        noise_i = 3
        for conv1, conv2, to_rgb in zip(
            self.convs[::2], self.convs[1::2], self.to_rgbs
        ):
            out = conv1(out, latent[:, i], noise=None) 
            out = conv2(out, latent[:, i + 1], noise=None) 
            skip = to_rgb(out.clone(), latent[:, i + 2], skip)
            if out.size(-1) == 64:
                prior_features64 = out.clone() # only 
                prior_rgb64 = skip.clone()
            if out.size(-1) == 32:
                prior_features32 = out.clone() # only 
                prior_rgb32 = skip.clone()
            i += 2
            noise_i += 2
        image = skip

        return image, prior_features64, prior_features32

class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)

class EqualLinear(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True, bias_init_val=0, lr_mul=1, activation=None):
        super(EqualLinear, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lr_mul = lr_mul
        self.activation = activation
        self.scale = (1 / math.sqrt(in_channels)) * lr_mul

        self.weight = nn.Parameter(torch.randn(out_channels, in_channels).div_(lr_mul))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels).fill_(bias_init_val))
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        if self.bias is None:
            bias = None
        else:
            bias = self.bias * self.lr_mul
        if self.activation == 'fused_lrelu':
            out = F.linear(x, self.weight * self.scale)
            out = fused_leaky_relu(out, bias)
        else:
            out = F.linear(x, self.weight * self.scale, bias=bias)
        return out

class SelectText(nn.Module):
    def __init__(self, class_num, channel, size=4):
        super().__init__()
        self.size = size
        self.TextEmbeddings = nn.Parameter(torch.randn(class_num, channel, 1, 1))
    def forward(self, labels):
        b, c = labels.size()
        TestEmbs = []
        for i in range(b):
            EmbTmps = []
            for j in range(c):
                EmbTmps.append(self.TextEmbeddings[labels[i][j]:labels[i][j]+1,...].repeat(1,1,self.size,self.size)) #
            Seqs = torch.cat(EmbTmps, dim=3)
            TestEmbs.append(Seqs)
        OutEmbs = torch.cat(TestEmbs, dim=0)
        return OutEmbs 


class StyledConv(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        style_dim,
        upsample=False,
        blur_kernel=[1, 3, 3, 1],
        demodulate=True,
    ):
        super().__init__()
        self.conv = ModulatedConv2d(
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
            upsample=upsample,
            blur_kernel=blur_kernel,
            demodulate=demodulate,
        )

        self.bias = nn.Parameter(torch.zeros(1, out_channel, 1, 1))
        self.activate = FusedLeakyReLU(out_channel)
    def forward(self, input, style, noise=None):
        out = self.conv(input, style)
        out = out + self.bias
        out = self.activate(out)
        return out


class ModulatedConv2d(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        style_dim,
        demodulate=True,
        upsample=False,
        downsample=False,
        blur_kernel=[1, 3, 3, 1],
    ):
        super().__init__()
        self.eps = 1e-8
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.upsample = upsample
        self.downsample = downsample
        self.up = nn.Upsample(scale_factor=2, mode='bilinear')


        fan_in = in_channel * kernel_size ** 2
        self.scale = 1 / math.sqrt(fan_in)
        self.padding = kernel_size // 2
        self.weight = nn.Parameter(
            torch.randn(1, out_channel, in_channel, kernel_size, kernel_size)
        )
        self.modulation = EqualLinear(style_dim, in_channel, bias=True, bias_init_val=1, lr_mul=1, activation=None)
        self.demodulate = demodulate


    def forward(self, input, style):
        batch, in_channel, height, width = input.shape
        style = self.modulation(style).view(batch, 1, in_channel, 1, 1)
        weight = self.scale * self.weight * style
        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
            weight = weight * demod.view(batch, self.out_channel, 1, 1, 1)
        weight = weight.view(
            batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size
        )
        if self.upsample:
            input = input.view(1, batch * in_channel, height, width)
            out = self.up(input)
            out = F.conv2d(out, weight, padding=1, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)
        else:
            input = input.view(1, batch * in_channel, height, width)
            out = F.conv2d(input, weight, padding=self.padding, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)
        return out


class ToRGB(nn.Module):
    def __init__(self, in_channel, style_dim, upsample=True, blur_kernel=[1, 3, 3, 1]):
        super().__init__()
        self.upsample = upsample

        self.conv = ModulatedConv2d(in_channel, 3, 1, style_dim, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))

    def forward(self, input, style, skip=None):
        out = self.conv(input, style)
        out = out + self.bias
        if skip is not None:
            if self.upsample:
                skip = F.interpolate(
                    skip, scale_factor=2, mode='bilinear', align_corners=False)
            out = out + skip
        return torch.tanh(out)


'''
3) Text super-resolution using structure prior
'''

class TSPSRNet(nn.Module):
    '''
    Input: LR features, structure prior on 64*64 and 32*32, character bounding boxes
    Output: SR results
    '''
    def __init__(self, in_channel=3, dim_channel=256):
        super().__init__()
        self.conv_first_32 = nn.Sequential(
            SpectralNorm(nn.Conv2d(in_channel, dim_channel//4, 3, 1, 1)),
            nn.LeakyReLU(0.2),
        )
        self.conv_first_16 = nn.Sequential(
            SpectralNorm(nn.Conv2d(dim_channel//4, dim_channel//2, 3, 2, 1)),
            nn.LeakyReLU(0.2),
        )
        self.conv_first_8 = nn.Sequential(
            SpectralNorm(nn.Conv2d(dim_channel//2, dim_channel, 3, 2, 1)),
            nn.LeakyReLU(0.2),
            SpectralNorm(nn.Conv2d(dim_channel, dim_channel, 3, 1, 1)),
        )
        self.conv_body_16 = nn.Sequential(
            SpectralNorm(nn.Conv2d(dim_channel+dim_channel//2, dim_channel, 3, 1, 1)),
            nn.LeakyReLU(0.2),
            SpectralNorm(nn.Conv2d(dim_channel, dim_channel, 3, 1, 1)),
        )
        self.conv_body_32 = nn.Sequential(
            SpectralNorm(nn.Conv2d(dim_channel+dim_channel//4, dim_channel, 3, 1, 1)),
            nn.LeakyReLU(0.2),
            SpectralNorm(nn.Conv2d(dim_channel, dim_channel, 3, 1, 1)),
        )

        self.conv_up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'), #64*64*256
            SpectralNorm(nn.Conv2d(dim_channel, dim_channel, 3, 1, 1)),
            nn.LeakyReLU(0.2),
            ResTextBlockV2(dim_channel, dim_channel),
            SpectralNorm(nn.Conv2d(dim_channel, dim_channel, 3, 1, 1)),
        )

        self.conv_final = nn.Sequential(
            SpectralNorm(nn.Conv2d(dim_channel, dim_channel//2, 3, 1, 1)),
            nn.LeakyReLU(0.2),
            nn.Upsample(scale_factor=2, mode='bilinear'), #128*128*256
            SpectralNorm(nn.Conv2d(dim_channel//2, dim_channel//4, 3, 1, 1)),
            nn.LeakyReLU(0.2),
            ResTextBlockV2(dim_channel//4, dim_channel//4),
            SpectralNorm(nn.Conv2d(dim_channel//4, 3, 3, 1, 1)),
            nn.Tanh()
        )
        self.conv_32_scale = nn.Sequential(
            SpectralNorm(nn.Conv2d(dim_channel, dim_channel, 3, 1, 1)),
            nn.LeakyReLU(0.2),
            SpectralNorm(nn.Conv2d(dim_channel, dim_channel, 3, 1, 1)),
        )
        self.conv_32_shift = nn.Sequential(
            SpectralNorm(nn.Conv2d(dim_channel, dim_channel, 3, 1, 1)),
            nn.LeakyReLU(0.2),
            SpectralNorm(nn.Conv2d(dim_channel, dim_channel, 3, 1, 1)),
        )
        self.conv_32_fuse = nn.Sequential(
            ResTextBlockV2(2*dim_channel, dim_channel)
        )

        self.conv_32_to256 = nn.Sequential(
            SpectralNorm(nn.Conv2d(512, dim_channel, 3, 1, 1)),
            nn.LeakyReLU(0.2),
            SpectralNorm(nn.Conv2d(dim_channel, dim_channel, 3, 1, 1)),
        )

        self.conv_64_scale = nn.Sequential(
            SpectralNorm(nn.Conv2d(dim_channel, dim_channel, 3, 1, 1)),
            nn.LeakyReLU(0.2),
            SpectralNorm(nn.Conv2d(dim_channel, dim_channel, 3, 1, 1)),
        )
        self.conv_64_shift = nn.Sequential(
            SpectralNorm(nn.Conv2d(dim_channel, dim_channel, 3, 1, 1)),
            nn.LeakyReLU(0.2),
            SpectralNorm(nn.Conv2d(dim_channel, dim_channel, 3, 1, 1)),
        )
        self.conv_64_fuse = nn.Sequential(
            ResTextBlockV2(2*dim_channel, dim_channel)
        )

    def forward(self, lq, priors64, priors32, locs): #
        lq_f_32 = self.conv_first_32(lq)
        lq_f_16 = self.conv_first_16(lq_f_32)
        lq_f_8 = self.conv_first_8(lq_f_16)
        sq_f_16 = self.conv_body_16(torch.cat([F.interpolate(lq_f_8, scale_factor=2, mode='bilinear'), lq_f_16], dim=1))
        sq_f_32 = self.conv_body_32(torch.cat([F.interpolate(sq_f_16, scale_factor=2, mode='bilinear'), lq_f_32], dim=1)) # 256*32*32

        '''
        Prior transformation on 32*32
        '''
        sq_f_32_ori = sq_f_32.clone()
        sq_f_32_res = sq_f_32.clone().detach()*0
        for b, p_32 in enumerate(priors32): 
            p_32_256 = self.conv_32_to256(p_32.clone()) 
            for c in range(p_32_256.size(0)): 
                center = (locs[b][2*c] * sq_f_32.size(-1)).int() 
                width = (locs[b][2*c+1] * sq_f_32.size(-1)).int() + 2 
                width = 16
                if center < width:
                    x1 = 0 
                    y1 = max(16 - center, 0)
                else:
                    x1 = center - width
                    y1 = max(16 - width, 0)
                if center + width > sq_f_32.size(-1):
                    x2 = sq_f_32.size(-1) 
                else:
                    x2 = center + width
                y2 = y1 + (x2 - x1)
                y1 = 16 - torch.div(x2-x1, 2, rounding_mode='trunc')
                y2 = y1 + x2 - x1
                char_prior_f = p_32_256[c:c+1, :, :, y1:y2].clone()
                char_lq_f = sq_f_32[b:b+1, :, :, x1:x2].clone()
                adain_prior_f = adaptive_instance_normalization(char_prior_f, char_lq_f)
                fuse_32_prior = self.conv_32_fuse(torch.cat((adain_prior_f, char_lq_f), dim=1))
                scale = self.conv_32_scale(fuse_32_prior)
                shift = self.conv_32_shift(fuse_32_prior)
                sq_f_32_res[b, :, :, x1:x2] = sq_f_32[b, :, :, x1:x2].clone() * scale[0,...] + shift[0,...]
        sq_pf_32_out = sq_f_32_ori + sq_f_32_res
        sq_f_64 = self.conv_up(sq_pf_32_out) 
        
        '''
        Prior transformation on 64*64
        '''
        sq_f_64_ori = sq_f_64.clone()
        sq_f_64_res = sq_f_64.clone().detach() * 0
        for b, p_64_prior in enumerate(priors64): 
            p_64 = p_64_prior.clone()
            for c in range(p_64.size(0)): 
                center = (locs[b][2*c] * sq_f_64.size(-1)).detach().int() 
                width = (locs[b][2*c+1] * sq_f_64.size(-1)).detach().int() + 4 
                width = 32
                if center < width:
                    x1 = 0
                    y1 = max(32 - center, 0)
                else:
                    x1 = center -width
                    y1 = max(32 - width, 0)
                if center + width > sq_f_64.size(-1):
                    x2 = sq_f_64.size(-1)
                else:
                    x2 = center + width
                y1 = 32 - torch.div(x2-x1, 2, rounding_mode='trunc')
                y2 = y1 + x2 - x1
                char_prior_f = p_64[c:c+1, :, :, y1:y2].clone()
                char_lq_f = sq_f_64[b:b+1, :, :, x1:x2].clone()
                adain_prior_f = adaptive_instance_normalization(char_prior_f, char_lq_f)
                fuse_64_prior = self.conv_64_fuse(torch.cat((adain_prior_f, char_lq_f), dim=1))
                scale = self.conv_64_scale(fuse_64_prior)
                shift = self.conv_64_shift(fuse_64_prior)
                sq_f_64_res[b, :, :, x1:x2] = sq_f_64[b, :, :, x1:x2].clone() * scale[0,...] + shift[0,...]
        sq_pf_64 = sq_f_64_ori + sq_f_64_res
        f256 = self.conv_final(sq_pf_64)

        return f256

def GroupNorm(in_channels):
    ec = 32
    assert in_channels % ec == 0
    return torch.nn.GroupNorm(num_groups=in_channels//32, num_channels=in_channels, eps=1e-6, affine=True)

def swish(x):
    return x*torch.sigmoid(x)

class ResTextBlockV2(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.norm1 = GroupNorm(in_channels)
        self.conv1 = SpectralNorm(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1))
        self.norm2 = GroupNorm(out_channels)
        self.conv2 = SpectralNorm(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1))
        if self.in_channels != self.out_channels:
            self.conv_out = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
    def forward(self, x_in):
        x = x_in
        x = self.norm1(x)
        x = swish(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = swish(x)
        x = self.conv2(x)
        if self.in_channels != self.out_channels:
            x_in = self.conv_out(x_in)
        return x + x_in
    
def calc_mean_std_4D(feat, eps=1e-5):
    size = feat.size()
    assert len(size) == 4, 'The input feature should be 4D tensor.'
    b, c = size[:2]
    feat_var = feat.view(b, c, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(b, c, 1, 1)
    feat_mean = feat.view(b, c, -1).mean(dim=2).view(b, c, 1, 1)
    return feat_mean, feat_std


def adaptive_instance_normalization(prior_feat, lq_feat):
    size = prior_feat.size()
    lq_mean, lq_std = calc_mean_std_4D(lq_feat)
    prior_mean, prior_std = calc_mean_std_4D(prior_feat)
    normalized_feat = (prior_feat - prior_mean.expand(size)) / prior_std.expand(size)
    return normalized_feat * lq_std.expand(size) + lq_mean.expand(size)


