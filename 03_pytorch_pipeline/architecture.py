import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import spectral_norm

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_sn=False):
        super().__init__()
        
        # Dynamically apply spectral normalization strictly to deep matrices
        def apply_sn(layer):
            return spectral_norm(layer) if use_sn else layer

        self.conv1 = apply_sn(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, padding_mode='reflect', bias=False))
        self.act = nn.SiLU(inplace=True)
        self.conv2 = apply_sn(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, padding_mode='reflect', bias=False))
        
        if in_channels != out_channels:
            self.identity_align = apply_sn(nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False))
        else:
            self.identity_align = nn.Identity()

    def forward(self, x):
        identity = self.identity_align(x)
        out = self.act(self.conv1(x))
        out = self.conv2(out)
        out += identity
        return self.act(out)

class ResUNet(nn.Module):
    def __init__(self, in_channels=102, out_channels=102, base_filters=128):
        super().__init__()
        
        # Outer Blocks: High-frequency processing, unnormalized
        self.enc1 = ResidualBlock(in_channels, base_filters, use_sn=False)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ResidualBlock(base_filters, base_filters * 2, use_sn=False)
        self.pool2 = nn.MaxPool2d(2)
        
        # Deep Blocks: Low-frequency processing, strictly Lipschitz bounded
        self.enc3 = ResidualBlock(base_filters * 2, base_filters * 4, use_sn=True)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = ResidualBlock(base_filters * 4, base_filters * 8, use_sn=True)
        self.pool4 = nn.MaxPool2d(2)
        
        self.bottleneck = ResidualBlock(base_filters * 8, base_filters * 16, use_sn=True)
        
        # Decoder: C1 continuous bilinear upsampling with strictly bounded deep matrices
        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            spectral_norm(nn.Conv2d(base_filters * 16, base_filters * 8, kernel_size=3, padding=1, padding_mode='reflect', bias=False))
        )
        self.dec4 = ResidualBlock(base_filters * 16, base_filters * 8, use_sn=True)
        
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            spectral_norm(nn.Conv2d(base_filters * 8, base_filters * 4, kernel_size=3, padding=1, padding_mode='reflect', bias=False))
        )
        self.dec3 = ResidualBlock(base_filters * 8, base_filters * 4, use_sn=True)
        
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(base_filters * 4, base_filters * 2, kernel_size=3, padding=1, padding_mode='reflect', bias=False)
        )
        self.dec2 = ResidualBlock(base_filters * 4, base_filters * 2, use_sn=False)
        
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(base_filters * 2, base_filters, kernel_size=3, padding=1, padding_mode='reflect', bias=False)
        )
        self.dec1 = ResidualBlock(base_filters * 2, base_filters, use_sn=False)
        
        self.final_conv = nn.Conv2d(base_filters, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        b = self.bottleneck(self.pool4(e4))
        
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.final_conv(d1)