import torch
import torch.nn as nn

class DECOEncoder(nn.Module):
    def __init__(self, in_channels, d_model=256, num_layers=3, kernel_size=7):
        """
        in_channels: channels from backbone (e.g., 512 for ResNet18)
        d_model: desired channels in the encoder
        num_layers: how many stacked 'conv blocks'
        kernel_size: e.g. 7 or 9 from the ablations
        """
        super().__init__()
        
        # 1x1 conv to reduce channels
        self.initial_conv = nn.Conv2d(in_channels, d_model, 1)
        
        # build stacked conv-based layers (similar to ConvNeXt blocks)
        self.layers = nn.ModuleList([
            ConvNeXtBlock(d_model, kernel_size) for _ in range(num_layers)
        ])
        
    def forward(self, x):
        # reduce channels
        x = self.initial_conv(x)
        # pass through stacked conv layers
        for layer in self.layers:
            x = layer(x)
        return x

class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, kernel_size=7):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size, padding=kernel_size//2, groups=dim)
        self.ln = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Conv2d(dim, 4 * dim, 1)
        self.gelu = nn.GELU()
        self.pwconv2 = nn.Conv2d(4 * dim, dim, 1)

    def forward(self, x):
        shortcut = x
        # depthwise conv
        x = self.dwconv(x)
        
        # (B, C, H, W) -> (B, H, W, C) for layernorm
        x = x.permute(0, 2, 3, 1)
        x = self.ln(x)
        # (B, H, W, C) -> (B, C, H, W)
        x = x.permute(0, 3, 1, 2)
        
        x = self.pwconv1(x)
        x = self.gelu(x)
        x = self.pwconv2(x)
        
        x = x + shortcut
        return x
