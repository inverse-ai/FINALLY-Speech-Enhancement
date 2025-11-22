import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils import remove_weight_norm
from torch.nn.utils.parametrizations import weight_norm, spectral_norm
from utils.model_utils import init_weights, get_padding
from typing import Literal
from models.models import LRELU_SLOPE, models_registry
from tqdm import tqdm

class ResBlock1(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5),
                norm_type: Literal["weight", "spectral"] = "weight"):
        super(ResBlock1, self).__init__()
        self.norm = dict(weight=weight_norm, spectral=spectral_norm)[norm_type]
        self.convs1 = nn.ModuleList([
            self.norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                            padding=get_padding(kernel_size, dilation[0]))),
            self.norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                            padding=get_padding(kernel_size, dilation[1]))),
            self.norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=dilation[2],
                            padding=get_padding(kernel_size, dilation[2])))
        ])
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList([
            self.norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=1,
                            padding=get_padding(kernel_size, 1))),
            self.norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=1,
                            padding=get_padding(kernel_size, 1))),
            self.norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=1,
                            padding=get_padding(kernel_size, 1)))
        ])
        self.convs2.apply(init_weights)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        if self.norm != weight_norm:
            raise RuntimeWarning("No weight norm to remove")
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


class ResBlock2(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3),
                 norm_type: Literal["weight", "spectral"] = "weight"):
        super(ResBlock2, self).__init__()
        self.norm = dict(weight=weight_norm, spectral=spectral_norm)[norm_type]
        self.convs = nn.ModuleList([
            self.norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                            padding=get_padding(kernel_size, dilation[0]))),
            self.norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                            padding=get_padding(kernel_size, dilation[1])))
        ])
        self.convs.apply(init_weights)

    def forward(self, x):
        for c in self.convs:
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        if self.norm != weight_norm:
            raise RuntimeWarning("No weight norm to remove")
        for l in self.convs:
            remove_weight_norm(l)

@models_registry.add_to_registry(name='hifigan_gen')
class HifiGan_Generator(torch.nn.Module):
    def __init__(self, resblock, resblock_kernel_sizes, upsample_rates,
                upsample_initial_channel, upsample_kernel_sizes, resblock_dilation_sizes):
        super(HifiGan_Generator, self).__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.conv_pre = weight_norm(nn.Conv1d(80, upsample_initial_channel, 7, 1, padding=3))
        resblock = ResBlock1 if resblock == '1' else ResBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(weight_norm(
                nn.ConvTranspose1d(upsample_initial_channel//(2**i), upsample_initial_channel//(2**(i+1)),
                                k, u, padding=(k-u)//2)))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel//(2**(i+1))
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(resblock(ch, k, d))

        self.conv_post = weight_norm(nn.Conv1d(ch, 1, 7, 1, padding=3))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

    def forward(self, x):
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i*self.num_kernels+j](x)
                else:
                    xs += self.resblocks[i*self.num_kernels+j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        tqdm.write('Removing weight norm...')
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)

class HifiGan_DiscriminatorP(torch.nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super(HifiGan_DiscriminatorP, self).__init__()
        self.period = period
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(nn.Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(nn.Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(nn.Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(nn.Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(nn.Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(2, 0))),
        ])
        self.conv_post = norm_f(nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmaps = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0: # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmaps.append(x)
        x = self.conv_post(x)
        fmaps.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmaps

@models_registry.add_to_registry(name='mpd')
class HifiGan_MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self):
        super(HifiGan_MultiPeriodDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([
            HifiGan_DiscriminatorP(2),
            HifiGan_DiscriminatorP(3),
            HifiGan_DiscriminatorP(5),
            HifiGan_DiscriminatorP(7),
            HifiGan_DiscriminatorP(11),
        ])

    def forward(self, real_wav, gen_wav):
        discs_real_out = []
        discs_gen_out = []
        fmaps_real = []
        fmaps_gen = []
        for disc in self.discriminators:
            disc_real_out, fmap_real = disc(real_wav)
            disc_gen_out, fmap_gen = disc(gen_wav)
            discs_real_out.append(disc_real_out)
            fmaps_real.append(fmap_real)
            discs_gen_out.append(disc_gen_out)
            fmaps_gen.append(fmap_gen)

        return discs_real_out, discs_gen_out, fmaps_real, fmaps_gen


class HifiGan_DiscriminatorS(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(HifiGan_DiscriminatorS, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(nn.Conv1d(1, 128, 15, 1, padding=7)),
            norm_f(nn.Conv1d(128, 128, 41, 2, groups=4, padding=20)),
            norm_f(nn.Conv1d(128, 256, 41, 2, groups=16, padding=20)),
            norm_f(nn.Conv1d(256, 512, 41, 4, groups=16, padding=20)),
            norm_f(nn.Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
            norm_f(nn.Conv1d(1024, 1024, 41, 1, groups=16, padding=20)),
            norm_f(nn.Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.conv_post = norm_f(nn.Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmaps = []
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmaps.append(x)
        x = self.conv_post(x)
        fmaps.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmaps

@models_registry.add_to_registry(name='msd')
class HifiGan_MultiScaleDiscriminator(torch.nn.Module):
    def __init__(self, same_scale=False):
        super(HifiGan_MultiScaleDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([
            HifiGan_DiscriminatorS(use_spectral_norm=True),
            HifiGan_DiscriminatorS(),
            HifiGan_DiscriminatorS(),
        ])
        self.meanpools = nn.ModuleList([
            nn.AvgPool1d(4, 2, padding=2),
            nn.AvgPool1d(4, 2, padding=2)
        ]) if not same_scale else None

    def forward(self, real_wav, gen_wav):
        discs_real_out = []
        discs_gen_out = []
        fmaps_real = []
        fmaps_gen = []
        for i, d in enumerate(self.discriminators):
            if i != 0 and self.meanpools is not None:
                real_wav = self.meanpools[i-1](real_wav)
                gen_wav = self.meanpools[i-1](gen_wav)
            disc_real_out, fmap_real = d(real_wav)
            disc_gen_out, fmap_gen = d(gen_wav)
            discs_real_out.append(disc_real_out)
            fmaps_real.append(fmap_real)
            discs_gen_out.append(disc_gen_out)
            fmaps_gen.append(fmap_gen)

        return discs_real_out, discs_gen_out, fmaps_gen, fmaps_real
