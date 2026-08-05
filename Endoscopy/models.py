import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        return x + self.bn2(self.conv2(self.prelu(self.bn1(self.conv1(x)))))

class ResidualSR(nn.Module): 
    def __init__(self, in_channels=3, out_channels=3, n_residual_blocks=8):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=9, padding=4)
        self.prelu = nn.PReLU()
        
        res_blocks = [ResidualBlock(64) for _ in range(n_residual_blocks)]
        self.res_blocks = nn.Sequential(*res_blocks)
        
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.up1 = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.PReLU()
        )
        self.up2 = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.PReLU()
        )
        
        self.conv_final = nn.Conv2d(64, out_channels, kernel_size=9, padding=4)

    def forward(self, x):
        base = F.interpolate(x, scale_factor=4, mode='bicubic', align_corners=False)
        
        feat1 = self.prelu(self.conv1(x))
        residual = self.res_blocks(feat1)
        feat2 = self.bn2(self.conv2(residual))
        feat = feat1 + feat2 
        
        feat = self.up1(feat)
        feat = self.up2(feat)
        details = self.conv_final(feat)
        
        final = base + details
        return torch.clamp(final, 0.0, 1.0)
