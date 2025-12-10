import torchaudio
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from utils.model_utils import NormConv2d, get_2d_padding
from models.hifigan_models import ResBlock1, ResBlock2
import typing as tp
from typing import Literal
from models.hifipp_models import MultiScaleResnet, A2AHiFiPlusGeneratorV2, build_block
from utils.model_utils import init_weights
import einops

from models.modules import WaveUnetUpsampler
from torch.nn.utils import weight_norm

LRELU_SLOPE = 0.1

FeatureMapType = tp.List[torch.Tensor]
LogitsType = torch.Tensor
DiscriminatorOutput = tp.Tuple[tp.List[LogitsType], tp.List[FeatureMapType]]

class DiscriminatorSTFT(nn.Module):
    """STFT sub-discriminator.
    Args:
        filters (int): Number of filters in convolutions
        in_channels (int): Number of input channels. Default: 1
        out_channels (int): Number of output channels. Default: 1
        n_fft (int): Size of FFT for each scale. Default: 1024
        hop_length (int): Length of hop between STFT windows for each scale. Default: 256
        kernel_size (tuple of int): Inner Conv2d kernel sizes. Default: ``(3, 9)``
        stride (tuple of int): Inner Conv2d strides. Default: ``(1, 2)``
        dilations (list of int): Inner Conv2d dilation on the time dimension. Default: ``[1, 2, 4]``
        win_length (int): Window size for each scale. Default: 1024
        normalized (bool): Whether to normalize by magnitude after stft. Default: True
        norm (str): Normalization method. Default: `'weight_norm'`
        activation (str): Activation function. Default: `'LeakyReLU'`
        activation_params (dict): Parameters to provide to the activation function.
        growth (int): Growth factor for the filters. Default: 1
    """
    def __init__(self, filters: int, in_channels: int = 1, out_channels: int = 1,
                 n_fft: int = 1024, hop_length: int = 256, win_length: int = 1024, max_filters: int = 1024,
                 filters_scale: int = 1, kernel_size: tp.Tuple[int, int] = (3, 9), dilations: tp.Tuple = (1, 2, 4),
                 stride: tp.Tuple[int, int] = (1, 2), normalized: bool = True, norm: str = 'weight_norm',
                 activation: str = 'LeakyReLU', activation_params: dict = {'negative_slope': 0.2}):
        super().__init__()
        assert len(kernel_size) == 2
        assert len(stride) == 2
        self.filters = filters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.normalized = normalized
        self.activation = getattr(torch.nn, activation)(**activation_params)
        self.spec_transform = torchaudio.transforms.Spectrogram(
            n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length, window_fn=torch.hann_window,
            normalized=self.normalized, center=False, pad_mode=None, power=None
        )
        spec_channels = 2 * self.in_channels
        self.convs = nn.ModuleList()
        self.convs.append(
            NormConv2d(spec_channels, self.filters, kernel_size=kernel_size, padding=get_2d_padding(kernel_size))
        )
        in_chs = min(filters_scale * self.filters, max_filters)
        for i, dilation in enumerate(dilations):
            out_chs = min((filters_scale ** (i + 1)) * self.filters, max_filters)
            self.convs.append(NormConv2d(in_chs, out_chs, kernel_size=kernel_size, stride=stride,
                                        dilation=(dilation, 1), padding=get_2d_padding(kernel_size, (dilation, 1)),
                                        norm=norm))
            in_chs = out_chs
        out_chs = min((filters_scale ** (len(dilations) + 1)) * self.filters, max_filters)
        self.convs.append(NormConv2d(in_chs, out_chs, kernel_size=(kernel_size[0], kernel_size[0]),
                                     padding=get_2d_padding((kernel_size[0], kernel_size[0])),
                                     norm=norm))
        self.conv_post = NormConv2d(out_chs, self.out_channels,
                                    kernel_size=(kernel_size[0], kernel_size[0]),
                                    padding=get_2d_padding((kernel_size[0], kernel_size[0])),
                                    norm=norm)

    def forward(self, x: torch.Tensor):
        fmap = []
        z = self.spec_transform(x)  # [B, 2, Freq, Frames, 2]
        z = torch.cat([z.real, z.imag], dim=1)
        z = einops.rearrange(z, 'b c w t -> b c t w')
        for i, layer in enumerate(self.convs):
            z = layer(z)
            z = self.activation(z)
            fmap.append(z)
        z = self.conv_post(z)
        return z, fmap

class MultiScaleSTFTDiscriminator(nn.Module):
    """Multi-Scale STFT (MS-STFT) discriminator.
    Args:
        filters (int): Number of filters in convolutions
        in_channels (int): Number of input channels. Default: 1
        out_channels (int): Number of output channels. Default: 1
        n_ffts (Sequence[int]): Size of FFT for each scale
        hop_lengths (Sequence[int]): Length of hop between STFT windows for each scale
        win_lengths (Sequence[int]): Window size for each scale
        **kwargs: additional args for STFTDiscriminator
    """
    def __init__(
        self,
        filters: int,
        in_channels: int = 1,
        out_channels: int = 1,
        n_ffts: tp.Tuple[int] = (4096, 2048, 1024, 512, 256),
        hop_lengths: tp.Tuple[int] = (1024, 512, 256, 128, 64),
        win_lengths: tp.Tuple[int] = (4096, 2048, 1024, 512, 256),
        **kwargs
    ): # defaults for 48kHz

        super().__init__()
        assert len(n_ffts) == len(hop_lengths) == len(win_lengths)
        self.discriminators = nn.ModuleList([
            DiscriminatorSTFT(filters, in_channels=in_channels, out_channels=out_channels,
                              n_fft=n_ffts[i], win_length=win_lengths[i], hop_length=hop_lengths[i], **kwargs)
            for i in range(len(n_ffts))
        ])
        self.num_discriminators = len(self.discriminators)

    def forward(self, x: torch.Tensor) -> DiscriminatorOutput:
        logits = []
        fmaps = []
        for disc in self.discriminators:
            logit, fmap = disc(x)
            logits.append(logit)
            fmaps.append(fmap)
        return logits, fmaps
    

class UpsampleWaveUnet(nn.Module):
    def __init__(
            self,
            block_widths,
            block_depth,
            in_width,
            out_width,
            waveunet_norm_type,
            upsample_factor,
            upsampler_norm
        ):
        super().__init__()
        self.waveunet = MultiScaleResnet(
                block_widths,
                block_depth,
                mode="waveunet_k5",
                in_width=in_width,
                out_width=out_width,
                norm_type=waveunet_norm_type
            )
        # self.activation = nn.LeakyReLU(LRELU_SLOPE)
        self.upsample_block = nn.Sequential(
                upsampler_norm(
                    nn.ConvTranspose1d(
                        self.waveunet.out_dims, 1, 
                        kernel_size=495,
                        stride=upsample_factor,
                        padding=(495 - upsample_factor) // 2
                    )
                ),
            )
        
    
        
        init_weights(self)

    def forward(self, x):
        x = self.waveunet(x)
        # x = self.activation(x)
        x = self.upsample_block(x)
        return x

class FinallyGenerator(A2AHiFiPlusGeneratorV2):
    def __init__(
        self,
        hifi_resblock="1", # good
        hifi_upsample_rates=(8, 8, 2, 2), # good
        hifi_upsample_kernel_sizes=(16, 16, 4, 4), # good
        hifi_upsample_initial_channel=512, # good? now v1, was 128 as in v2
        hifi_resblock_kernel_sizes=(3, 7, 11), # good
        hifi_resblock_dilation_sizes=((1, 3, 5), (1, 3, 5), (1, 3, 5)), # good
        hifi_input_channels=512, # was 128
        hifi_conv_pre_kernel_size=1, # good

        use_spectralunet=True, # good
        spectralunet_block_widths=(16, 32, 64, 128, 256), # was (8, 16, 24, 32, 64),
        spectralunet_block_depth=4, # was 5
        spectralunet_positional_encoding=True, # good
        spectralunet_out_channels=512, # good

        use_waveunet=True, # good
        waveunet_block_widths=(128, 128, 256, 512), # omg was (10, 20, 40, 80), 
        waveunet_block_depth=4, # good

        use_spectralmasknet=True, # good
        spectralmasknet_block_widths=(64, 128, 256, 512), # wtf was (8, 12, 24, 32),
        spectralmasknet_block_depth=4, # was 4

        use_upsamplewaveunet=False, # by default
        upsamplewaveunet_block_widths=(128,128,128,128,256), # as in paper
        upsamplewaveunet_block_depth=3, # as in paper
        upsamplewaveunet_upsample_factor=2, # from 16 to 48
        upsamplewaveunet_upsampler_features=512, # as in paper

        norm_type: Literal["weight", "spectral"] = "weight",
        use_skip_connect=True,

        use_conv_post=True,
        use_tanh=True,

        waveunet_input: Literal["waveform", "hifi", "both"] = "both",
    ):
        super().__init__(
            hifi_resblock=hifi_resblock,
            hifi_upsample_rates=hifi_upsample_rates,
            hifi_upsample_kernel_sizes=hifi_upsample_kernel_sizes,
            hifi_upsample_initial_channel=hifi_upsample_initial_channel,
            hifi_resblock_kernel_sizes=hifi_resblock_kernel_sizes,
            hifi_resblock_dilation_sizes=hifi_resblock_dilation_sizes,
            hifi_input_channels=hifi_input_channels,
            hifi_conv_pre_kernel_size=hifi_conv_pre_kernel_size,

            use_spectralunet=use_spectralunet,
            spectralunet_block_widths=spectralunet_block_widths,
            spectralunet_block_depth=spectralunet_block_depth,
            spectralunet_positional_encoding=spectralunet_positional_encoding,
            spectralunet_out_channels=spectralunet_out_channels,

            use_waveunet=use_waveunet,
            waveunet_block_widths=waveunet_block_widths,
            waveunet_block_depth=waveunet_block_depth,

            use_spectralmasknet=use_spectralmasknet,
            spectralmasknet_block_widths=spectralmasknet_block_widths,
            spectralmasknet_block_depth=spectralmasknet_block_depth,

            norm_type=norm_type,
            use_skip_connect=use_skip_connect,
            waveunet_before_spectralmasknet=True,

            waveunet_input=waveunet_input
        )
        self.use_conv_post = use_conv_post
        self.use_tanh = use_tanh
        
        self.wavlm_post_processing = nn.Sequential(
            build_block(inner_width=spectralunet_out_channels + 1024, block_depth=7,mode="waveunet_k5", norm=weight_norm, kernel=3),
            nn.LeakyReLU(LRELU_SLOPE),
            weight_norm(nn.Conv1d(in_channels=spectralunet_out_channels + 1024, out_channels=spectralunet_out_channels, kernel_size=1))
        )

        self.resblock_type = ResBlock1 if hifi_resblock == '1' else ResBlock2
        self.pre_upsampler_proccessing = nn.Sequential(
            self.resblock_type(spectralunet_out_channels, kernel_size=3),
            self.resblock_type(spectralunet_out_channels, kernel_size=7),
            self.resblock_type(spectralunet_out_channels, kernel_size=11)
        )
        init_weights(self.pre_upsampler_proccessing)

        self.upsamplewaveunet_block_widths = upsamplewaveunet_block_widths
        self.upsamplewaveunet_block_depth = upsamplewaveunet_block_depth
        self.upsamplewaveunet_upsampler_features = upsamplewaveunet_upsampler_features
        self.upsamplewaveunet_upsample_factor = upsamplewaveunet_upsample_factor
        self.upsamplewaveunet = None
        self.set_use_upsamplewaveunet(use_upsamplewaveunet)

    def set_use_upsamplewaveunet(self, use):
        self.use_upsamplewaveunet = use
        if self.use_upsamplewaveunet and self.upsamplewaveunet is None:
            self.upsamplewaveunet = WaveUnetUpsampler(in_channels=32, out_channels=32)
            self.conv_final = weight_norm(nn.Conv1d(32, 1, kernel_size=7, padding=3))

            
            

    def forward(self, x, wavlm_features):
        x_orig = x.clone()
        
        assert x_orig.shape[2] % 1024 == 0
        x_orig = x_orig[:, :, : x_orig.shape[2] // 1024 * 1024]

        x = self.get_melspec(x_orig)
        x = self.apply_spectralunet(x)

        assert not torch.isnan(x).any(), "SpectralUNet output contains NaNs"
        assert x.shape[1] == 512

        assert wavlm_features.shape[1] == 1024

        wavlm_features_interpolated = torch.nn.functional.interpolate(
            wavlm_features, size=x.shape[2], mode='nearest'
        )
        x = torch.cat([wavlm_features_interpolated, x], 1)
        assert x.shape[1] == 1536
        x = self.wavlm_post_processing(x)
        x = self.pre_upsampler_proccessing(x)
        x = self.hifi(x)

        if self.use_waveunet:
            x = self.apply_waveunet_a2a(x, x_orig)
        
        assert x.shape[1] == 32
        
        

        if self.use_spectralmasknet:
            x = self.apply_spectralmasknet(x)

        if self.use_upsamplewaveunet:
            x = self.upsamplewaveunet(x)
            x = self.conv_final(x)
        else:
            if self.use_conv_post:
                x = self.conv_post(x)

        if self.use_tanh:                 
            x = torch.tanh(x)

        return x
