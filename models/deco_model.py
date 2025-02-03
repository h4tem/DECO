import torch
import torch.nn as nn

class DECOModel(nn.Module):
    """
    A wrapper that encapsulates:
      - a backbone (e.g., ResNet18)
      - a DECOEncoder
      - a DECODecoder
    so that forward(images) returns classification logits & box coords.
    """
    def __init__(self, backbone, encoder, decoder):
        super().__init__()
        self.backbone = backbone
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, images):
        """
        images: (B, 3, H, W)
        returns:
            class_logits: (B, num_queries, num_classes)
            box_coords:   (B, num_queries, 4)
        """
        # 1. Extract features
        feats = self.backbone(images)    # (B, C_in, H', W')
        
        # 2. Encode
        enc_feats = self.encoder(feats)  # (B, d_model, H', W')
        
        # 3. Decode
        class_logits, box_coords = self.decoder(enc_feats)
        
        return class_logits, box_coords
