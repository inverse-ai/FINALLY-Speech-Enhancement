import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from stft import STFT

SPECTRALUNET_SCALE = 2
WAVEUNET_SCALE = 4
WAVEUNETUPSAMPLER_SCALE = 4

class Downsample1d(nn.Module):

    def __init__(self, in_ch, out_ch, scale, layers=2) -> None:
        super().__init__()
        self.ResidualBlocks = nn.Sequential(*[ResidualBlock1d(in_ch) for _ in range(layers)])
        self.scale = scale
        self.conv = nn.Conv1d(in_ch, out_ch, scale, scale)

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
        self.upsample = nn.ConvTranspose1d(in_ch, out_ch, kernel_size=scale, stride=scale)
        self.ResidualBlocks = nn.Sequential(*[ResidualBlock1d(out_ch) for _ in range(layers)])
        self.out = nn.Conv1d(out_ch * 2, out_ch, kernel_size=1)

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
        self.conv = nn.Conv2d(width, out_width, kernel_size=scale, stride=scale)

    def forward(self, x):
        return self.conv(self.ResidualBlocks(x))

class Upsample2d(nn.Module):
    def __init__(self, in_width, width, scale, layers=2) -> None:
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_width, width, scale, scale)
        self.ResidualBlocks = nn.Sequential(*[ResidualBlock2d(width) for _ in range(layers)])
        self.out = nn.Conv2d(width * 2, width, kernel_size=1)

    def forward(self, x, skip):
        x = self.ResidualBlocks(self.upsample(x))
        diffY = skip.size()[2] - x.size()[2]
        diffX = skip.size()[3] - x.size()[3]
        x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        return self.out(torch.cat([x, skip], dim=1))

class SpectralUnet(nn.Module):

    def __init__(self, in_channels, out_channels, depth=4) -> None:
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, 1, 80, 128))  # 2D


        self.input = nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=3, padding=1)

        self.down1 = Downsample2d(16, 32, SPECTRALUNET_SCALE, depth)
        self.down2 = Downsample2d(32, 64, SPECTRALUNET_SCALE, depth)
        self.down3 = Downsample2d(64, 128, SPECTRALUNET_SCALE, depth)
        self.down4 = Downsample2d(128, 256, SPECTRALUNET_SCALE, depth)
        self.bottleneck = nn.Sequential(*[ResidualBlock2d(256) for _ in range(depth)])
        self.up4 = Upsample2d(256, 128, SPECTRALUNET_SCALE, depth)
        self.up3 = Upsample2d(128, 64, SPECTRALUNET_SCALE, depth)
        self.up2 = Upsample2d(64, 32, SPECTRALUNET_SCALE, depth)
        self.up1 = Upsample2d(32, 16, SPECTRALUNET_SCALE, depth)

        self.output = nn.Conv2d(16, out_channels=out_channels, kernel_size=3, padding=1)
        self.output2 = nn.Conv1d(in_channels=80, out_channels=512, kernel_size=1)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = x + self.pos_embedding[:, :, :, :x.size(-1)]  # [B, 1, 80, T]
        skip1 = self.input(x)
        skip2 = self.down1(skip1)
        skip3 = self.down2(skip2)
        skip4 = self.down3(skip3)
        bottleneck = self.bottleneck(self.down4(skip4))
        up4 = self.up4(bottleneck, skip4)
        up3 = self.up3(up4, skip3)
        up2 = self.up2(up3, skip2)
        up1 = self.up1(up2, skip1)
        x = self.output(up1).squeeze(1)
        x = self.output2(x)
        return x

class WaveUnet(nn.Module):

    def __init__(self, in_channels, out_channels, depth=4) -> None:
        super().__init__()
        self.input = nn.Conv1d(in_channels=in_channels, out_channels=128, kernel_size=5, padding=2)
        self.down1 = Downsample1d(128, 128, WAVEUNET_SCALE, depth)
        self.down2 = Downsample1d(128, 256, WAVEUNET_SCALE, depth)
        self.down3 = Downsample1d(256, 512, WAVEUNET_SCALE, depth)
        self.bottleneck = nn.Sequential(*[ResidualBlock1d(512) for _ in range(depth)])
        self.up3 = Upsample1d(512, 256, WAVEUNET_SCALE, depth)
        self.up2 = Upsample1d(256, 128, WAVEUNET_SCALE, depth)
        self.up1 = Upsample1d(128, 128, WAVEUNET_SCALE, depth)
        self.output = nn.Conv1d(128, out_channels=out_channels, kernel_size=5, padding=2)

    def forward(self, x):
        skip1 = self.input(x)
        skip2 = self.down1(skip1)
        skip3 = self.down2(skip2)
        bottleneck = self.bottleneck(self.down3(skip3))
        up3 = self.up3(bottleneck, skip3)
        up2 = self.up2(up3, skip2)
        up1 = self.up1(up2, skip1)
        return self.output(up1)
    
class SpectralUnet2(nn.Module):

    def __init__(self, in_channels, out_channels, depth=1) -> None:
        super().__init__()
        self.input = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, padding=1)
        self.down1 = Downsample2d(64, 128, 2, depth)
        self.down2 = Downsample2d(128, 256, 2, depth)
        self.down3 = Downsample2d(256, 512, 2, depth)
        self.bottleneck = nn.Sequential(*[ResidualBlock2d(512) for _ in range(depth)])
        self.up3 = Upsample2d(512, 256, 2, depth)
        self.up2 = Upsample2d(256, 128, 2, depth)
        self.up1 = Upsample2d(128, 64, 2, depth)
        self.output = nn.Conv2d(64, out_channels=out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = x.unsqueeze(1)
        skip1 = self.input(x)
        skip2 = self.down1(skip1)
        skip3 = self.down2(skip2)
        bottleneck = self.bottleneck(self.down3(skip3))
        up3 = self.up3(bottleneck, skip3)
        up2 = self.up2(up3, skip2)
        up1 = self.up1(up2, skip1)
        x = self.output(up1)
        return x.squeeze(1)

class SpectralMaskNet(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.conv1d = nn.Conv1d(80, 512, 1)
        self.spectralunet = SpectralUnet2(1, 1)
        self.stft = STFT(1024, 256, 1024)

    def forward(self, x):
        mag, phase = self.stft.transform(x)
        spec = self.spectralunet(mag)
        mul = F.softplus(spec)
        mag_ = mag * mul
        out = self.stft.inverse(mag_, phase)
        return out
    
class WaveUnetUpsampler(nn.Module):

    def __init__(self, in_channels, out_channels, depth=3) -> None:
        super().__init__()
        self.input = nn.Conv1d(in_channels, 128, kernel_size=5, padding=2)
        self.down1 = Downsample1d(128, 128, WAVEUNETUPSAMPLER_SCALE, depth)
        self.down2 = Downsample1d(128, 128, WAVEUNETUPSAMPLER_SCALE, depth)
        self.down3 = Downsample1d(128, 128, WAVEUNETUPSAMPLER_SCALE, depth)
        self.down4 = Downsample1d(128, 256, WAVEUNETUPSAMPLER_SCALE, depth)
        self.bottleneck = nn.Sequential(*[ResidualBlock1d(256) for _ in range(depth)])
        self.up4 = Upsample1d(256, 128, WAVEUNETUPSAMPLER_SCALE, depth)
        self.up3 = Upsample1d(128, 128, WAVEUNETUPSAMPLER_SCALE, depth)
        self.up2 = Upsample1d(128, 128, WAVEUNETUPSAMPLER_SCALE, depth)
        self.up1 = Upsample1d(128, 128, WAVEUNETUPSAMPLER_SCALE, depth)
        self.upsample_by_3 = nn.ConvTranspose1d(128, 512, kernel_size=3, stride=3)
        self.output = nn.Conv1d(512, out_channels, kernel_size=5, padding=2)

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
        out = self.output(upsamp)
        return out