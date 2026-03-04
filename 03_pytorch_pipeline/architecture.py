import torch
import torch.nn as nn

from torch.nn.utils.parametrizations import spectral_norm

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        # Apply Spectral Normalization directly to the convolutional weights
        self.conv1 = spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, padding_mode='replicate', bias=False))
        self.act = nn.SiLU(inplace=True)
        
        self.conv2 = spectral_norm(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, padding_mode='replicate', bias=False))
        
        # Identity mapping alignment strictly bounded
        if in_channels != out_channels:
            self.identity_align = spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False))
        else:
            self.identity_align = nn.Identity()

    def forward(self, x):
        identity = self.identity_align(x)
        
        # BN/IN layers are entirely removed. 
        # The thermodynamic state magnitude is strictly preserved.
        out = self.act(self.conv1(x))
        out = self.conv2(out)
        
        out += identity
        out = self.act(out)
        return out

class ResUNet(nn.Module):
    def __init__(self, in_channels=102, out_channels=102, base_filters=128):
        """
        Spatially symmetrical encoder-decoder network.
        Input: R^(B x 102 x 64 x 64)
        Output: R^(B x 102 x 64 x 64)
        """
        super().__init__()
        
        # ---------------------------------------------------------
        # ENCODER: Extracting Advection and Macroscopic Features
        # Spatial dimensions halve, channel depth doubles.
        # ---------------------------------------------------------
        self.enc1 = ResidualBlock(in_channels, base_filters)          # 64x64
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = ResidualBlock(base_filters, base_filters * 2)     # 32x32
        self.pool2 = nn.MaxPool2d(2)
        
        self.enc3 = ResidualBlock(base_filters * 2, base_filters * 4) # 16x16
        self.pool3 = nn.MaxPool2d(2)
        
        self.enc4 = ResidualBlock(base_filters * 4, base_filters * 8) # 8x8
        self.pool4 = nn.MaxPool2d(2)
        
        # ---------------------------------------------------------
        # BOTTLENECK: The Latent Thermodynamic State
        # Spatial resolution: 4x4. Receptive field maximizes here.
        # ---------------------------------------------------------
        self.bottleneck = ResidualBlock(base_filters * 8, base_filters * 16)
        
        # ---------------------------------------------------------
        # DECODER: Reconstructing the Spatiotemporal Matrix
        # Transposed convolutions double the spatial dimensions.
        # ---------------------------------------------------------
        self.up4 = nn.ConvTranspose2d(base_filters * 16, base_filters * 8, kernel_size=2, stride=2)
        self.dec4 = ResidualBlock(base_filters * 16, base_filters * 8) # Note: input is 16 (8 from up, 8 from skip)
        
        self.up3 = nn.ConvTranspose2d(base_filters * 8, base_filters * 4, kernel_size=2, stride=2)
        self.dec3 = ResidualBlock(base_filters * 8, base_filters * 4)
        
        self.up2 = nn.ConvTranspose2d(base_filters * 4, base_filters * 2, kernel_size=2, stride=2)
        self.dec2 = ResidualBlock(base_filters * 4, base_filters * 2)
        
        self.up1 = nn.ConvTranspose2d(base_filters * 2, base_filters, kernel_size=2, stride=2)
        self.dec1 = ResidualBlock(base_filters * 2, base_filters)
        
        # ---------------------------------------------------------
        # OUTPUT PROJECTION
        # ---------------------------------------------------------
        self.final_conv = nn.Conv2d(base_filters, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder Pass (Saving skip connections)
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        
        # Bottleneck
        b = self.bottleneck(self.pool4(e4))
        
        # Decoder Pass (Concatenating skip connections)
        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], dim=1)  # Merge macro state with high-res physical gradients
        d4 = self.dec4(d4)
        
        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        
        # Map back to 102 state variables
        out = self.final_conv(d1)
        return out

# Execution Test
if __name__ == "__main__":
    model = ResUNet(in_channels=102, out_channels=102)
    dummy_input = torch.randn(16, 102, 64, 64) # Batch size 16
    output = model(dummy_input)
    print(f"Input Matrix: {dummy_input.shape}")
    print(f"Output Prediction: {output.shape}")