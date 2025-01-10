import torch.nn as nn

class DECOEncoder(nn.Module):
    """
    The convolution-based encoder that replaces Transformer encoder layers
    with ConvNeXt-like blocks, as described in Sec. 'DECO Encoder' of the paper.
    
    For the ResNet18-based model (lowest resource), you'll typically have:
    - in_channels = 512 (from the last stage of ResNet18)
    - d_model = 256
    - num_layers = 3 (or possibly 4, will need to recheck paper experiments)
    """
    def __init__(
        self,
        in_channels=512,
        d_model=256,
        num_layers=3,
        kernel_size=7
    ):
        super().__init__()
        
        # 1. reduce channels via 1x1
        self.initial_conv = nn.Conv2d(in_channels, d_model, kernel_size=1)
        
        # 2. stack of DECO encoder layers
        self.layers = nn.ModuleList([
            DECOEncoderLayer(d_model, kernel_size=kernel_size)
            for _ in range(num_layers)
        ])
        
    def forward(self, x):
        """
        x: backbone feature map, shape (B, C, H, W).
        Returns: shape (B, d_model, H, W).
        """
        # 1x1 conv to get (B, d_model, H, W)
        x = self.initial_conv(x)
        
        # pass through the stacked DECO layers
        for layer in self.layers:
            x = layer(x)
        
        return x


class DECOEncoderLayer(nn.Module):
    """
    A single 'ConvNeXt-like' block used in the DECO encoder:
    - Depthwise 7x7 (or other kernel_size)
    - LayerNorm
    - 1x1 conv
    - GELU
    - 1x1 conv
    - Skip connection
    """
    def __init__(
        self,
        dim: int,
        kernel_size: int = 7
    ):
        super().__init__()
        self.dwconv = nn.Conv2d(
            dim, 
            dim, 
            kernel_size=kernel_size, 
            padding=kernel_size // 2, 
            groups=dim  # depthwise
        )
        
        # Note: PyTorch's LayerNorm expects (N, C) or (N, C, H, W) if normalized_shape=C 
        # but we want to apply LN in "channel-last" format:
        self.ln = nn.LayerNorm(dim, eps=1e-6)
        
        # pointwise conv expansions
        self.pwconv1 = nn.Conv2d(dim, 4 * dim, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(4 * dim, dim, kernel_size=1)

    def forward(self, x):
        """
        x: (B, dim, H, W)
        returns: (B, dim, H, W)
        """
        shortcut = x
        
        # depthwise conv
        x = self.dwconv(x)
        
        # reshape to (B, H, W, C) for LN
        # because we want LN over the channel dimension in a "channel-last" style
        x = x.permute(0, 2, 3, 1)  # (B, H, W, C)
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2)  # back to (B, C, H, W)
        
        # pointwise conv → GELU → pointwise conv
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        
        # skip connection
        x = x + shortcut
        return x
