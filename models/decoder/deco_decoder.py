import torch.nn as nn
import torch.nn.functional as F

class DECODecoder(nn.Module):
    """
    The top-level DECO decoder: 
      - holds a learnable query embedding
      - repeats DECODecoderLayer for 'num_layers' times 
      - final detection heads for classification & box regression
    """
    def __init__(
        self,
        d_model=256, 
        num_queries=100, 
        num_layers=3,
        num_classes=80,   # e.g. for COCO
        kernel_size=9,
    ):
        """
        d_model: channel dimension used for queries/features 
        num_queries: total number of learnable object queries 
        num_layers: how many times to repeat the SIM + CIM stack
        num_classes: number of object categories (excl. background)
        kernel_size: for depthwise conv in SIM & CIM
        """
        super().__init__()
        
        self.d_model = d_model
        self.num_queries = num_queries
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        
        # Learnable 1D query embed: (num_queries, d_model)
        #   or we could do 2D if we interpret queries as a small grid ? Will see later
        self.query_embed = nn.Embedding(num_queries, d_model)
        
        # We create multiple DECODecoderLayers that each do:
        #   SIM -> CIM -> feed-forward
        self.layers = nn.ModuleList([
            DECODecoderLayer(d_model, kernel_size) 
            for _ in range(num_layers)
        ])
        
        # FINAL HEADS: classify + regression 
        # We apply them after the final layer. 

        self.class_embed = nn.Linear(d_model, num_classes) 
        self.bbox_embed = MLP(d_model, d_model, 4, num_layers=3)
        # The MLP can be 3-layers: 
        #   see DETR or the paper for their exact dimension choices.

    def forward(self, encoder_feats):
        """
        encoder_feats: shape (B, d_model, H, W)
            from DECOEncoder output.
        
        returns:
            class_logits: (B, num_queries, num_classes)
            bbox_coords:  (B, num_queries, 4)
        """
        B, C, H, W = encoder_feats.shape
        assert C == self.d_model, "encoder_feats channels must match d_model"
        
        # Expand queries to (B, num_queries, d_model)
        queries = self.query_embed.weight.unsqueeze(0).repeat(B, 1, 1)

        # We can interpret queries as (B, d_model, QH, QW) if we want 
        #   a 2D shape.  Suppose num_queries=100 => 10 x 10
        Q_side = int(self.num_queries ** 0.5)  # e.g. 10 if 100 queries
        # reshape to (B, d_model, Q_side, Q_side)
        queries_2d = queries.view(B, Q_side, Q_side, self.d_model).permute(0, 3, 1, 2)

        # pass queries + encoder_feats through each decoder layer 
        for layer_idx, layer in enumerate(self.layers):
            queries_2d = layer(queries_2d, encoder_feats)

        # after the final layer, we flatten to feed class & box heads
        # shape: (B, d_model, Q_side*Q_side)
        queries_2d_flat = queries_2d.flatten(2)              # (B, d_model, N)
        queries_2d_flat = queries_2d_flat.transpose(1, 2)    # (B, N, d_model)

        class_logits = self.class_embed(queries_2d_flat)     # (B, N, num_classes)
        bbox_coords  = self.bbox_embed(queries_2d_flat)      # (B, N, 4)
        
        return class_logits, bbox_coords


class DECODecoderLayer(nn.Module):
    """
    One 'layer' of the DECO Decoder, including:
      - Self-Interaction Module (SIM) 
      - Cross-Interaction Module (CIM)
      - (Optional) small feed-forward / LN
    """
    def __init__(self, d_model=256, kernel_size=9):
        super().__init__()
        self.sim = SelfInteractionModule(d_model, kernel_size)
        self.cim = CrossInteractionModule(d_model, kernel_size)
        
        # Optional FFN after SIM + CIM
        self.FFN = nn.Sequential(
            nn.Conv2d(d_model, d_model, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(d_model, d_model, 1),
        )
        
    def forward(self, query_2d, encoder_feats):
        """
        query_2d: shape (B, d_model, Qh, Qw)
        encoder_feats: shape (B, d_model, H, W)
        """
        # 1) Self-Interaction 
        #    -> captures relationships among queries themselves
        out_sim = self.sim(query_2d)  # (B, d_model, Qh, Qw)
        
        # 2) Cross-Interaction
        #    -> upsample queries to (H, W), fuse with encoder_feats
        out_cim = self.cim(out_sim, encoder_feats)  # (B, d_model, H, W)

        # 3) Small feed-forward + residual and downsample back to (Qh, Qw)
        
        Qh, Qw = query_2d.shape[-2], query_2d.shape[-1]
        of = F.adaptive_max_pool2d(out_cim + self.FFN(out_cim), (Qh, Qw))  # (B, d_model, Qh, Qw)       
        
        return of


class SelfInteractionModule(nn.Module):
    """
    Replaces self-attention among queries with 
    a depthwise + pointwise conv.
    """
    def __init__(self, d_model=256, kernel_size=9):
        super().__init__()
        self.depthwise = nn.Conv2d(
            d_model, 
            d_model, 
            kernel_size=kernel_size, 
            padding=kernel_size // 2, 
            groups=d_model
        )
        self.pointwise = nn.Conv2d(d_model, d_model, kernel_size=1)
        self.act = nn.ReLU(inplace=True)  # or GELU ? Didn't specify in paper

    def forward(self, x):
        """
        x: (B, d_model, Qh, Qw)
        """
        out = self.depthwise(x)
        out = self.pointwise(out)
        out = self.act(out)
        
        # add a residual 
        out = out + x
        return out


class CrossInteractionModule(nn.Module):
    """
    Replaces cross-attention by fusing upsampled queries and encoder feats
    using convolution. Typically, queries + encoder_feats are summed 
    then depthwise + pointwise conv.
    """
    def __init__(self, d_model=256, kernel_size=9):
        super().__init__()
        self.depthwise = nn.Conv2d(
            d_model, 
            d_model, 
            kernel_size=kernel_size, 
            padding=kernel_size // 2, 
            groups=d_model
        )
        self.pointwise = nn.Conv2d(d_model, d_model, kernel_size=1)
        self.act = nn.ReLU(inplace=True)  # or GELU

    def forward(self, query_2d, encoder_feats):
        """
        query_2d: (B, d_model, some_H, some_W), after being upsampled 
                  to match encoder_feats shape
        encoder_feats: (B, d_model, H, W)
        """
        # 1) match encoder_feats shape
        H, W = encoder_feats.shape[-2:]
        upsampled = F.interpolate(query_2d, size=(H, W), mode='bilinear', align_corners=False)
        
        # 2) fuse 
        fused = upsampled + encoder_feats  # or concat + conv
        out = self.depthwise(fused)
        out = self.pointwise(out)
        out = self.act(out)

        # 3) Skip connection
        out = upsampled + out


        # If we followed the paper, it should look like this:

        # 2) fuse 
        # fused = upsampled + encoder_feats  # or concat + conv

        # 3) Depthwise conv + skip connection
        # out = upsampled + self.depthwise(fused)

        # But it feels sketchy to me. Not enough details in the paper.
        # So I'm doing what's above for now.
        return out


class MLP(nn.Module):
    """
    Just a small feed-forward for bounding box regression 
    or other small heads. 
    E.g., 3-layer MLP from DETR code.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3):
        super().__init__()
        layers = []
        in_dim = input_dim
        for i in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        # x: (B, N, input_dim)
        return self.mlp(x)
