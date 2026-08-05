import torch
import torch.nn as nn
import torch.nn.functional as F

class DenseBlock(nn.Module):
    def __init__(self, channels=64, growth_rate=32):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, growth_rate, 3, 1, 1)
        self.conv2 = nn.Conv2d(channels + growth_rate, growth_rate, 3, 1, 1)
        self.conv3 = nn.Conv2d(channels + 2 * growth_rate, growth_rate, 3, 1, 1)
        self.conv4 = nn.Conv2d(channels + 3 * growth_rate, growth_rate, 3, 1, 1)
        self.conv5 = nn.Conv2d(channels + 4 * growth_rate, channels, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x 

class RRDB(nn.Module):
    def __init__(self, channels=64, growth_rate=32):
        super().__init__()
        self.db1 = DenseBlock(channels, growth_rate)
        self.db2 = DenseBlock(channels, growth_rate)
        self.db3 = DenseBlock(channels, growth_rate)

    def forward(self, x):
        out = self.db1(x)
        out = self.db2(out)
        out = self.db3(out)
        return out * 0.2 + x

class RRDBNet_25D(nn.Module):
    def __init__(self, in_channels=9, out_channels=3, num_blocks=4):
        super().__init__()
        self.conv_first = nn.Conv2d(in_channels, 64, 3, 1, 1)
        self.rrdb_blocks = nn.Sequential(*[RRDB(channels=64, growth_rate=32) for _ in range(num_blocks)])
        self.conv_trunk = nn.Conv2d(64, 64, 3, 1, 1)
        
        self.up1 = nn.Sequential(nn.Conv2d(64, 256, 3, 1, 1), nn.PixelShuffle(2), nn.LeakyReLU(0.2, True))
        self.up2 = nn.Sequential(nn.Conv2d(64, 256, 3, 1, 1), nn.PixelShuffle(2), nn.LeakyReLU(0.2, True))
        self.conv_last = nn.Conv2d(64, out_channels, 3, 1, 1)

    def forward(self, x_stacked, x_center):
        base = F.interpolate(x_center, scale_factor=4, mode='bicubic', align_corners=False)
        feat = self.conv_first(x_stacked)
        trunk = self.conv_trunk(self.rrdb_blocks(feat))
        feat = feat + trunk
        
        feat = self.up1(feat)
        feat = self.up2(feat)
        out = self.conv_last(feat)
        return torch.clamp(out + base, 0.0, 1.0)
