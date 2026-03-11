import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

SPECTRALUNET_SCALE = 2
WAVEUNET_SCALE = 4
WAVEUNETUPSAMPLER_SCALE = 4
WAVEUNETUPSAMPLER_FACTOR = 3

class Downsample1d(nn.Module):

    def __init__(self, in_ch, out_ch, scale, layers=2) -> None:
        super().__init__()
        self.ResidualBlocks = nn.Sequential(*[ResidualBlock1d(in_ch) for _ in range(layers)])
        self.scale = scale
        self.conv = weight_norm(nn.Conv1d(in_ch, out_ch, scale, scale))

    def forward(self, x):
        return self.conv(self.ResidualBlocks(x))

class ResidualBlock1d(nn.Module):

    def __init__(self, width, kernel_size=5) -> None:
        super().__init__()
        self.conv = weight_norm(nn.Conv1d(width, width, kernel_size=kernel_size, padding=kernel_size // 2))

    def forward(self, x):
        x = x + self.conv(F.leaky_relu(x))
        return x

class Upsample1d(nn.Module):
    def __init__(self, in_ch, out_ch, scale, layers=2) -> None:
        super().__init__()
        self.upsample = weight_norm(nn.ConvTranspose1d(in_ch, out_ch, kernel_size=scale, stride=scale))
        self.ResidualBlocks = nn.Sequential(*[ResidualBlock1d(out_ch) for _ in range(layers)])
        self.out = weight_norm(nn.Conv1d(out_ch * 2, out_ch, kernel_size=1))

    def forward(self, x, skip):
        # Apply the transposed convolution (upsampling)
        x = self.upsample(x)
        x = self.ResidualBlocks(x)
        
        # Concatenate with skip connection and apply final conv
        x = self.out(torch.cat([x, skip], dim=1))
        return x
    
class WaveUnetUpsampler(nn.Module):

    def __init__(self, in_channels, out_channels, depth=3) -> None:
        super().__init__()
        self.input = weight_norm(nn.Conv1d(in_channels, 128, kernel_size=5, padding=2))
        self.down1 = Downsample1d(128, 128, WAVEUNETUPSAMPLER_SCALE, depth)
        self.down2 = Downsample1d(128, 128, WAVEUNETUPSAMPLER_SCALE, depth)
        self.down3 = Downsample1d(128, 128, WAVEUNETUPSAMPLER_SCALE, depth)
        self.down4 = Downsample1d(128, 256, WAVEUNETUPSAMPLER_SCALE, depth)

        self.bottleneck_pre = weight_norm(nn.Conv1d(256, 512, kernel_size=5, padding=2))
        self.bottleneck_lstm = nn.LSTM(
            input_size=512,
            hidden_size=512,
            num_layers=2,
            bidirectional=True,
            batch_first=True  # expects (batch, seq_len, features)
        )
        self.bottleneck_post = weight_norm(nn.Conv1d(1024, 256, kernel_size=5, padding=2))

        self.up4 = Upsample1d(256, 128, WAVEUNETUPSAMPLER_SCALE, depth)
        self.up3 = Upsample1d(128, 128, WAVEUNETUPSAMPLER_SCALE, depth)
        self.up2 = Upsample1d(128, 128, WAVEUNETUPSAMPLER_SCALE, depth)
        self.up1 = Upsample1d(128, 128, WAVEUNETUPSAMPLER_SCALE, depth)
        self.upsample_by_3 = weight_norm(nn.ConvTranspose1d(128, 512, kernel_size=WAVEUNETUPSAMPLER_FACTOR, stride=WAVEUNETUPSAMPLER_FACTOR))
        self.output = weight_norm(nn.Conv1d(512, out_channels, kernel_size=5, padding=2))

    def forward(self, x):
        # x shape: (32, 512, 65536)
        skip0 = self.input(x)  # (32, 128, 65536)
        skip1 = self.down1(skip0)   # (32, 128, 16384)
        skip2 = self.down2(skip1)   # (32, 128, 4096)
        skip3 = self.down3(skip2)   # (32, 128, 1024)
        bottleneck_pre_input = self.down4(skip3)   # (32, 256, 256)

        # Project 256 -> 512
        bottleneck = self.bottleneck_pre(bottleneck_pre_input)  # (32, 512, 256)

        # Reshape for LSTM: (seq_len, batch, features)
        bottleneck = bottleneck.permute(0, 2, 1)  # (32, 256, 512)

        # LSTM processing
        bottleneck, _ = self.bottleneck_lstm(bottleneck)  # (32, 256, 1024)
        
        # Reshape back: (batch, features, seq_len)
        bottleneck = bottleneck.permute(0, 2, 1)  # (32, 1024, 256)

        # Project 1024 -> 256
        bottleneck = self.bottleneck_post(bottleneck)  # (32, 256, 256)

        up4 = self.up4(bottleneck, skip3)   # (32, 128, 1024)
        up3 = self.up3(up4, skip2)  # (32, 128, 4096)
        up2 = self.up2(up3, skip1)  # (32, 128, 16384)
        up1 = self.up1(up2, skip0)  # (32, 128, 65536)

        upsamp = self.upsample_by_3(up1)    # (32, 512, 196608)
        out = self.output(upsamp)  # (32, 32, 196608)
        return out