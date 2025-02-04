import torch.nn as nn
import torch.nn.functional as F

class DECODecoder(nn.Module):
    """
    Following Section 3.1 of paper:
    The top-level DECO decoder:
      - Holds learnable object queries o ∈ ℝᴺˣᵈ
      - Processes through stacked SIM and CIM modules
      - Outputs class logits and box coordinates via FFNs
    """
    def __init__(
        self,
        d_model=256, 
        num_queries=100,  # Ablation study shows 100 is best
        num_layers=3,     # Table 4 shows 6 is a good choice, but we use 3
        num_classes=80,   # e.g. for COCO
        kernel_size=5,    # Table 5 shows 9x9 is best, we use 5 for now
    ):
        """
        Args:
            d_model: Channel dimension for queries/features 
            num_queries: Total number of object queries 
            num_layers: Number of SIM+CIM layers 
            num_classes: Number of object categories (dataset-dependent)
            kernel_size: For depthwise conv 
        """
        super().__init__()
        
        self.d_model = d_model
        self.num_queries = num_queries
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        
        # Paper Sec 3.1: "given N object queries o ∈ ℝᴺˣᵈ"
        self.query_embed = nn.Embedding(num_queries, d_model)
        
        # Stack of decoder layers (SIM → CIM → FFN) from Figure 2
        self.layers = nn.ModuleList([
            DECODecoderLayer(d_model, kernel_size) 
            for _ in range(num_layers)
        ])
        
        # Paper: "feed forward network (FFN)" for predictions
        # Note: Exact architectures not specified
        self.class_embed = nn.Linear(d_model, num_classes)
        self.bbox_embed = MLP(d_model, d_model, 4, num_layers=3)

    def forward(self, encoder_feats):
        """
        Following Section 3.1 of the paper:
        - Takes encoder feature map zₑ ∈ ℝᵈˣᴴˣᵂ
        - Processes object queries o ∈ ℝᴺˣᵈ through SIM and CIM
        - Returns class logits and box coordinates
        
        Args:
            encoder_feats: shape (B, d_model, H, W) from DECOEncoder output
        Returns:
            class_logits: (B, num_queries, num_classes)
            bbox_coords:  (B, num_queries, 4)
            
        Note: Some implementation details not specified in paper:
        - Batch dimension handling
        - Exact architecture of classification and box regression heads
        """
        B, C, H, W = encoder_feats.shape
        assert C == self.d_model, "encoder_feats channels must match d_model"
        
        # Expand queries to (B, N, d_model)
        queries = self.query_embed.weight.unsqueeze(0).repeat(B, 1, 1)

        # Reshape queries to (B, d_model, Qh, Qw) following paper Section 3.1 and Figure 2
        # For N=100 queries -> 10x10 shape as specified
        Q_side = int(self.num_queries ** 0.5)  # 10 for 100 queries
        queries_2d = queries.view(B, Q_side, Q_side, self.d_model).permute(0, 3, 1, 2)  # (B, d_model, Q_side, Q_side)

        # Paper Sec 3.1: Pass queries + encoder_feats through stacked SIM and CIM modules 
        for layer_idx, layer in enumerate(self.layers):
            queries_2d = layer(queries_2d, encoder_feats)

        # after the final layer, we flatten to feed class & box heads
        # shape: (B, d_model, Q_side*Q_side) [paper doesn't specify exact flattening procedure]
        queries_2d_flat = queries_2d.flatten(2)              # (B, d_model, N)
        queries_2d_flat = queries_2d_flat.transpose(1, 2)    # (B, N, d_model)

        # Paper mentions using "feed forward network (FFN)" for predictions
        # but doesn't specify exact architecture  
        class_logits = self.class_embed(queries_2d_flat)     # (B, N, num_classes)
        bbox_coords  = self.bbox_embed(queries_2d_flat)      # (B, N, 4)
        
        return class_logits, bbox_coords


class DECODecoderLayer(nn.Module):
    """
    One 'layer' of the DECO Decoder, following Section 3.1 of paper.
    Includes:
      - Self-Interaction Module (SIM) 
      - Cross-Interaction Module (CIM)
      - Feed-forward network (FFN)
    
    Note: Paper states "The output features further go through another FFN with skip connection"
    but doesn't specify:
      - The exact FFN architecture
      - The exact downsampling procedure
    """
    def __init__(self, d_model=256, kernel_size=9):
        super().__init__()
        self.sim = SelfInteractionModule(d_model, kernel_size)
        self.cim = CrossInteractionModule(d_model, kernel_size)
        
        # Paper mentions FFN but doesn't specify architecture
        # We use two 1x1 convs with ReLU, following standard practice
        self.FFN = nn.Sequential(
            nn.Conv2d(d_model, d_model, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(d_model, d_model, 1),
        )
        
    def forward(self, query_2d, encoder_feats):
        """
        Following paper Section 3.1 and Figure 2:
        1. Process queries through SIM
        2. Process through CIM with encoder features
        3. Apply FFN with skip connection, then downsample
        
        Args:
            query_2d: shape (B, d_model, Qh, Qw)
            encoder_feats: shape (B, d_model, H, W)
        """
        # 1) Self-Interaction 
        #    -> captures relationships among queries themselves
        out_sim = self.sim(query_2d)  # (B, d_model, Qh, Qw)
        
        # 2) Cross-Interaction
        #    -> upsample queries to (H, W), fuse with encoder_feats
        out_cim = self.cim(out_sim, encoder_feats)  # (B, d_model, H, W)

        # 3) Following Figure 2:
        #    CIM output → FFN → Skip connection → Adaptive MaxPool
        ffn_out = self.FFN(out_cim)
        out = out_cim + ffn_out  # Skip connection
        
        # Finally downsample back to query size
        Qh, Qw = query_2d.shape[-2], query_2d.shape[-1]
        out = F.adaptive_max_pool2d(out, (Qh, Qw)) # (B, d_model, Qh, Qw)
        
        return out


class SelfInteractionModule(nn.Module):
    """
    Following Section 3.1 and Figure 2:
    SIM replaces self-attention with:
    - Large kernel depthwise conv
    - Two pointwise (1x1) convs
    - Skip connection
    
    Paper specifies in Figure 2:
    - DWConv → PWConv → PWConv → Skip connection
    - Large kernel depthwise conv (9x9 from Table 5)
    
    Note: Paper doesn't specify:
    - Whether to use activation between convs
    """
    def __init__(self, d_model=256, kernel_size=9):
        super().__init__()
        # DWConv first
        self.depthwise = nn.Conv2d(
            d_model, 
            d_model, 
            kernel_size=kernel_size, 
            padding=kernel_size // 2, 
            groups=d_model
        )
        # Two PWConvs
        self.pointwise1 = nn.Conv2d(d_model, d_model, kernel_size=1)
        self.pointwise2 = nn.Conv2d(d_model, d_model, kernel_size=1)
        
        # Note: Paper doesn't specify activation
        # Using ReLU as a standard choice
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        Following Figure 2's SIM structure:
        x → DWConv → PWConv → PWConv → Skip connection
        
        Args:
            x: shape (B, d_model, Qh, Qw)
        Returns:
            out: shape (B, d_model, Qh, Qw)
        """
        out = self.depthwise(x)
        out = self.pointwise1(out)
        out = self.act(out)  # Note: Activation placement not specified in paper
        out = self.pointwise2(out)
        
        # Paper shows skip connection in Figure 2
        out = out + x
        return out


class CrossInteractionModule(nn.Module):
    """
    Following Section 3.1 and Figure 2:
    CIM replaces cross-attention through three steps:
    1. Upsample queries to match encoder feature size
    2. Fuse upsampled queries with encoder features (element-wise add)
    3. Apply depthwise conv with skip connection from upsampled queries
    
    Paper specifies in Eq(2):
    ôf = ô + dwconv(Fusion(ô, zₑ))
    where:
    - ô is upsampled queries
    - zₑ is encoder features
    - Fusion is element-wise addition (best in Table 3)
    
    Note: Unlike SIM, CIM's Figure 2 and Eq(2) only show depthwise conv,
    no pointwise conv or activation specified.
    """
    def __init__(self, d_model=256, kernel_size=9):
        super().__init__()
        # Paper specifies large kernel depthwise conv (9x9 from Table 5)
        self.depthwise = nn.Conv2d(
            d_model, 
            d_model, 
            kernel_size=kernel_size, 
            padding=kernel_size // 2, 
            groups=d_model
        )

    def forward(self, query_2d, encoder_feats):
        """
        Following paper equations (1,2):
        1. ô = Upsample(o)
        2. ôf = ô + dwconv(Fusion(ô, zₑ))
        where Fusion is element-wise addition
        
        Args:
            query_2d: shape (B, d_model, Qh, Qw)
            encoder_feats: shape (B, d_model, H, W)
        Returns:
            out: shape (B, d_model, H, W)
        """
        # 1) Eq(1): ô = Upsample(o) -> match encoder_feats shape
        H, W = encoder_feats.shape[-2:]
        upsampled = F.interpolate(query_2d, size=(H, W), mode='bilinear', align_corners=False)
        
        # 2) Fusion(ô, zₑ) = element-wise add (best in Table 3)
        fused = upsampled + encoder_feats
        
        # 3) Following Eq(2): ôf = ô + dwconv(Fusion(ô, zₑ))
        out = upsampled + self.depthwise(fused)
        
        return out


class MLP(nn.Module):
    """
    Following paper's detection head description:
    Feed-forward network (FFN) for bounding box regression.
    
    Note: Paper mentions using FFN for predictions but doesn't specify:
    - Number of layers (we use 3 following DETR)
    - Hidden dimension (we use same as d_model)
    - Activation function (we use ReLU)
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3):
        """
        Args:
            input_dim: Input dimension (d_model from paper)
            hidden_dim: Hidden dimension (not specified in paper)
            output_dim: Output dimension (4 for box coordinates)
            num_layers: Number of layers (not specified in paper, using 3 like DETR)
        """
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
        """
        Args:
            x: shape (B, N, input_dim) where N is number of queries
        Returns:
            Output tensor of shape (B, N, output_dim)
            For box regression: output_dim = 4 (coordinates)
        """
        return self.mlp(x)
