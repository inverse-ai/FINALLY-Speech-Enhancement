from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, spectral_norm


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


LRELU_SLOPE = 0.1


class ResBlock1(torch.nn.Module):
    def __init__(
        self,
        channels,
        kernel_size=3,
        dilation=(1, 3, 5),
        norm_type: Literal["weight", "spectral"] = "weight"
    ):
        norm = dict(weight=weight_norm, spectral=spectral_norm)[norm_type]
        super(ResBlock1, self).__init__()
        self.convs1 = nn.ModuleList([
            norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=d, padding=get_padding(kernel_size, d)))
            for d in dilation
        ])
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList([
            norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1)))
            for _ in range(3)
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

class HiFiGeneratorBackbone(torch.nn.Module):
    def __init__(
            self,
            upsample_rates=(8, 8, 2, 2),
            upsample_kernel_sizes=(16, 16, 4, 4),
            upsample_initial_channel=512,
            resblock_kernel_sizes=(3, 7, 11),
            resblock_dilation_sizes=((1, 3, 5), (1, 3, 5), (1, 3, 5)),
            conv_pre_kernel_size=7,
            input_channels=512,
            norm_type: Literal["weight", "spectral"] = "weight",
    ):
        super().__init__()
        self.norm = dict(weight=weight_norm, spectral=spectral_norm)[norm_type]
        self.norm_type = norm_type
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.make_conv_pre(
            input_channels,
            upsample_initial_channel,
            conv_pre_kernel_size
        )

        self.ups = None
        self.resblocks = None
        self.out_channels = self.make_resblocks(
            upsample_rates,
            upsample_kernel_sizes,
            upsample_initial_channel,
            resblock_kernel_sizes,
            resblock_dilation_sizes,
        )

    def make_conv_pre(self, input_channels, upsample_initial_channel, kernel_size):
        assert kernel_size % 2 == 1
        self.conv_pre = self.norm(
            nn.Conv1d(
                input_channels, upsample_initial_channel, kernel_size, 1, padding=kernel_size // 2
            )
        )
        self.conv_pre.apply(init_weights)

    def make_resblocks(
        self,
        upsample_rates,
        upsample_kernel_sizes,
        upsample_initial_channel,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
    ):

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                self.norm(
                    nn.ConvTranspose1d(
                        upsample_initial_channel // (2 ** i),
                        upsample_initial_channel // (2 ** (i + 1)),
                        k,
                        u,
                        padding=(k - u) // 2,
                    )
                )
            )

        ch = None
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(
                zip(resblock_kernel_sizes, resblock_dilation_sizes)
            ):
                self.resblocks.append(
                    ResBlock1(ch, k, d, norm_type=self.norm_type)
                )
        self.conv_post = weight_norm(torch.nn.Conv1d(ch, 1, 7, 1, padding=3))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)
        return ch

    def forward(self, x):
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)
        return x.squeeze(1)