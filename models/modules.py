import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from models.hifigan_models import ResBlock1

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
        x = x + F.leaky_relu(self.conv(x))
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
        
        # Padding to match the dimensions of the skip connection
        diffX = skip.size()[2] - x.size()[2]
        x = F.pad(x, [0, diffX])  # Pad the output to match skip size
        
        # Concatenate with skip connection and apply final conv
        x = self.out(torch.cat([x, skip], dim=1))
        return x


class ResidualBlock2d(nn.Module):

    def __init__(self, width, kernel_size=3) -> None:
        super().__init__()
        self.conv = weight_norm(nn.Conv2d(width, width, kernel_size=kernel_size, padding=kernel_size // 2))

    def forward(self, x):
        x = x + F.leaky_relu(self.conv(x))
        return x

class Downsample2d(nn.Module):

    def __init__(self, width, out_width, scale, layers=2) -> None:
        super().__init__()
        self.ResidualBlocks = nn.Sequential(*[ResidualBlock2d(width) for _ in range(layers)])
        self.conv = weight_norm(nn.Conv2d(width, out_width, kernel_size=scale, stride=scale))

    def forward(self, x):
        return self.conv(self.ResidualBlocks(x))

class Upsample2d(nn.Module):
    def __init__(self, in_width, width, scale, layers=2) -> None:
        super().__init__()
        self.upsample = weight_norm(nn.ConvTranspose2d(in_width, width, scale, scale))
        self.ResidualBlocks = nn.Sequential(*[ResidualBlock2d(width) for _ in range(layers)])
        self.out = weight_norm(nn.Conv2d(width * 2, width, kernel_size=1))

    def forward(self, x, skip):
        x = self.ResidualBlocks(self.upsample(x))
        diffY = skip.size()[2] - x.size()[2]
        diffX = skip.size()[3] - x.size()[3]
        x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        return self.out(torch.cat([x, skip], dim=1))
    
class WaveUnetUpsampler(nn.Module):

    def __init__(self, in_channels, out_channels, depth=3) -> None:
        super().__init__()
        self.input = weight_norm(nn.Conv1d(in_channels, 128, kernel_size=5, padding=2))
        self.down1 = Downsample1d(128, 128, WAVEUNETUPSAMPLER_SCALE, depth)
        self.down2 = Downsample1d(128, 128, WAVEUNETUPSAMPLER_SCALE, depth)
        self.down3 = Downsample1d(128, 128, WAVEUNETUPSAMPLER_SCALE, depth)
        self.down4 = Downsample1d(128, 256, WAVEUNETUPSAMPLER_SCALE, depth)
        self.bottleneck = nn.Sequential(*[ResidualBlock1d(256) for _ in range(depth)])
        self.up4 = Upsample1d(256, 128, WAVEUNETUPSAMPLER_SCALE, depth)
        self.up3 = Upsample1d(128, 128, WAVEUNETUPSAMPLER_SCALE, depth)
        self.up2 = Upsample1d(128, 128, WAVEUNETUPSAMPLER_SCALE, depth)
        self.up1 = Upsample1d(128, 128, WAVEUNETUPSAMPLER_SCALE, depth)
        self.upsample_by_3 = weight_norm(nn.ConvTranspose1d(128, 512, kernel_size=WAVEUNETUPSAMPLER_FACTOR, stride=WAVEUNETUPSAMPLER_FACTOR))
        self.ResidualBlocks = ResBlock1(channels=512, kernel_size=7)
        self.output = weight_norm(nn.Conv1d(512, out_channels, kernel_size=5, padding=2))

    def forward(self, x):
        skip1 = self.input(x)
        skip2 = self.down1(skip1)
        skip3 = self.down2(skip2)
        skip4 = self.down3(skip3)
        bottleneck = self.bottleneck(self.down4(skip4))
        up4 = self.up4(bottleneck, skip4)
        up3 = self.up3(up4, skip3)
        up2 = self.up2(up3, skip2)
        up1 = self.up1(up2, skip1)
        upsamp = self.upsample_by_3(up1)
        upsamp_res = self.ResidualBlocks(upsamp)
        out = self.output(upsamp_res)
        return out