"""
ParticleVision — Model Architectures
======================================
1. Primary: U-Net with EfficientNet-B3 encoder (via segmentation_models_pytorch)
2. Secondary: SAM ViT-B encoder + custom FPN decoder (for comparison)
3. Attention U-Net: U-Net with Attention Gates on skip connections
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from typing import List, Optional
from .config import CFG


# ---------------------------------------------------------------------------
# Attention Gate (Oktay et al. 2018, Attention U-Net)
# ---------------------------------------------------------------------------

# Not used by the ParticleVision pipeline (kept from earlier experiments).
class AttentionGate(nn.Module):
    """
    Additive Attention Gate for skip connections.

    Takes a gating signal g (from deeper/coarser decoder layer) and a skip
    connection x (from encoder, finer/more spatial). Produces an attention
    map α ∈ [0,1] per spatial location that weights the skip features,
    suppressing irrelevant background and highlighting discriminative regions.

    Standard formulation:
        W_g(g) + W_x(x) → ReLU → ψ → Sigmoid → α
        output = x * α
    """

    def __init__(self, g_channels: int, x_channels: int, inter_channels: int = None):
        """
        Parameters
        ----------
        g_channels : int
            Number of channels in the gating signal (from deeper decoder layer).
        x_channels : int
            Number of channels in the skip connection (from encoder).
        inter_channels : int
            Intermediate channel count. Default: x_channels // 2.
        """
        super().__init__()
        if inter_channels is None:
            inter_channels = max(x_channels // 2, 1)

        # 1×1 convolutions to project g and x to the same intermediate space
        self.W_g = nn.Sequential(
            nn.Conv2d(g_channels, inter_channels, kernel_size=1, bias=True),
            nn.BatchNorm2d(inter_channels),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(x_channels, inter_channels, kernel_size=1, bias=True),
            nn.BatchNorm2d(inter_channels),
        )
        # Collapse to single-channel attention map
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        g : (B, g_channels, H_g, W_g) — gating signal (coarser resolution)
        x : (B, x_channels, H_x, W_x) — skip connection (finer resolution)

        Returns
        -------
        x_att : (B, x_channels, H_x, W_x) — attention-weighted skip features
        """
        # Upsample g to match x's spatial dimensions
        g_up = F.interpolate(g, size=x.shape[2:], mode='bilinear', align_corners=False)

        g_proj = self.W_g(g_up)   # (B, inter, H_x, W_x)
        x_proj = self.W_x(x)     # (B, inter, H_x, W_x)

        combined = self.relu(g_proj + x_proj)
        alpha = self.psi(combined)  # (B, 1, H_x, W_x)

        return x * alpha  # broadcast multiply


# ---------------------------------------------------------------------------
# Attention U-Net wrapper
# ---------------------------------------------------------------------------

# Not used by the ParticleVision pipeline (kept from earlier experiments).
class AttentionUnet(nn.Module):
    """
    Wraps a standard SMP U-Net and injects AttentionGates on each skip
    connection point. The encoder and decoder weights are loaded from a
    V1 checkpoint; only the new AttentionGate parameters need training.

    Architecture:
        - Encoder: EfficientNet-B3 (frozen, loaded from V1)
        - Decoder: Standard SMP UnetDecoder (loaded from V1)
        - Attention Gates: NEW, applied to each skip connection before concat
        - Segmentation Head: Standard (loaded from V1)
    """

    def __init__(self, base_model: nn.Module):
        """
        Parameters
        ----------
        base_model : smp.Unet
            A standard SMP U-Net model (already built, weights can be loaded later).
        """
        super().__init__()
        self.encoder = base_model.encoder
        self.decoder = base_model.decoder
        self.segmentation_head = base_model.segmentation_head

        # Encoder output channels: [3, 40, 32, 48, 136, 384] for EfficientNet-B3
        # After decoder processing:
        #   features[1:] reversed = [384, 136, 48, 32, 40]
        #   head = 384, skips = [136, 48, 32, 40]
        #
        # Gating channels at each decoder block:
        #   Block 0: g=384 (head),           skip=136
        #   Block 1: g=256 (dec_ch[0]),      skip=48
        #   Block 2: g=128 (dec_ch[1]),      skip=32
        #   Block 3: g=64  (dec_ch[2]),      skip=40
        #   Block 4: no skip connection

        enc_channels = list(self.encoder.out_channels)  # [3, 40, 32, 48, 136, 384]
        dec_channels = [256, 128, 64, 32, 16]  # CFG.decoder_channels

        # Skip channels in decoder order (reversed encoder, minus head)
        # features[1:] = [40, 32, 48, 136, 384]
        # reversed = [384, 136, 48, 32, 40]
        # skips = [136, 48, 32, 40]
        skip_channels = list(reversed(enc_channels[1:]))[1:]  # [136, 48, 32, 40]

        # Gating channels: head for first block, then decoder output channels
        g_channels = [enc_channels[-1]] + dec_channels[:-1]  # [384, 256, 128, 64, 32]

        # Create attention gates for blocks that have skip connections (first 4)
        self.attention_gates = nn.ModuleList()
        for i in range(len(skip_channels)):
            self.attention_gates.append(
                AttentionGate(
                    g_channels=g_channels[i],
                    x_channels=skip_channels[i],
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with attention-gated skip connections."""
        # 1. Encoder
        features = self.encoder(x)

        # 2. Process features (same as SMP UnetDecoder.forward)
        spatial_shapes = [f.shape[2:] for f in features]
        spatial_shapes = spatial_shapes[::-1]

        features = features[1:]   # remove first skip (same spatial res as input)
        features = features[::-1]  # reverse to start from deepest

        head = features[0]
        skip_connections = features[1:]

        # 3. Decoder with attention gates
        x = self.decoder.center(head)

        for i, decoder_block in enumerate(self.decoder.blocks):
            height, width = spatial_shapes[i + 1]

            if i < len(skip_connections):
                # Apply attention gate to skip connection
                skip = skip_connections[i]
                skip_attended = self.attention_gates[i](x, skip)
            else:
                skip_attended = None

            # Call decoder block with the attention-weighted skip
            x = decoder_block(x, height, width, skip_connection=skip_attended)

        # 4. Segmentation head
        return self.segmentation_head(x)


# ---------------------------------------------------------------------------
# 1. Primary architecture: U-Net + EfficientNet-B3
# ---------------------------------------------------------------------------

def build_unet(
    encoder_name: str = None,
    encoder_weights: str = None,
    num_classes: int = None,
    decoder_channels=None,
    in_channels: int = 3,
) -> nn.Module:
    """
    Build a U-Net with a pretrained encoder backbone.

    Default: EfficientNet-B3 pretrained on ImageNet, 4-class output.
    """
    if encoder_name is None:
        encoder_name = CFG.encoder_name
    if encoder_weights is None:
        encoder_weights = CFG.encoder_weights
    if num_classes is None:
        num_classes = CFG.num_classes
    if decoder_channels is None:
        decoder_channels = CFG.decoder_channels

    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=num_classes,
        decoder_channels=decoder_channels,
        activation=None,  # raw logits — loss function applies softmax
    )

    return model


# ---------------------------------------------------------------------------
# 2. Secondary architecture: SAM ViT-B encoder + lightweight decoder
# ---------------------------------------------------------------------------

# Not used by the ParticleVision pipeline (kept from earlier experiments).
class SAMEncoderDecoder(nn.Module):
    """
    Frozen SAM ViT-B encoder with a custom lightweight FPN-style decoder.

    The SAM encoder outputs a (B, 256, 64, 64) feature map for 1024×1024 input.
    For 512×512 input it will output (B, 256, 32, 32).
    The decoder upsamples this back to 512×512 with skip-less convolutions.

    This is intentionally simple — the point is to test whether SAM's
    pretrained features help vs. training from ImageNet features.
    """

    def __init__(self, num_classes: int = None, freeze_encoder: bool = True):
        super().__init__()
        self.num_classes = num_classes or CFG.num_classes
        self._freeze_encoder = freeze_encoder

        # Lazy import — SAM may not always be installed
        try:
            from segment_anything import sam_model_registry
        except ImportError:
            raise ImportError(
                "segment-anything is required for SAM model. "
                "Install with: pip install segment-anything"
            )

        # Load SAM ViT-B
        sam = sam_model_registry["vit_b"](checkpoint=None)
        self.encoder = sam.image_encoder

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # Decoder: upsample from (256, 32, 32) → (num_classes, 512, 512)
        # 32 → 64 → 128 → 256 → 512
        self.decoder = nn.Sequential(
            # Stage 1: 32 → 64
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # Stage 2: 64 → 128
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # Stage 3: 128 → 256
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # Stage 4: 256 → 512
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, self.num_classes, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, 3, 512, 512) or (B, 3, 1024, 1024)

        Returns
        -------
        logits : (B, num_classes, H, W) same spatial size as input
        """
        # SAM encoder expects (B, 3, 1024, 1024) but works with other sizes
        features = self.encoder(x)  # (B, 256, H/16, W/16)
        logits = self.decoder(features)
        return logits


# ---------------------------------------------------------------------------
# 3. Dual-Head Architecture with Stride-Capped Encoder
# ---------------------------------------------------------------------------

# Not used by the ParticleVision pipeline (kept from earlier experiments).
class DualHeadModel(nn.Module):
    """
    Dual-head model using DeepLabV3Plus as the base.
    Features a stride-capped encoder (output_stride=8) to preserve spatial resolution,
    and two heads:
      1. Segmentation head (4 classes)
      2. Heatmap head (1 class, fines centroid prediction)
    """
    def __init__(
        self,
        encoder_name: str = "efficientnet-b3",
        encoder_weights: str = "imagenet",
        in_channels: int = 3,
        num_classes: int = 4,
        encoder_output_stride: int = 16,
        decoder_channels: int = 256,
    ):
        super().__init__()
        self.base_model = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=num_classes,
            encoder_output_stride=encoder_output_stride,
            decoder_channels=decoder_channels,
        )
        
        # The base_model's segmentation head upsamples by 4 (from 1/4 resolution).
        # We duplicate its structure for the heatmap head.
        self.heatmap_head = smp.base.SegmentationHead(
            in_channels=decoder_channels,
            out_channels=1,
            activation=None,  # raw logits
            kernel_size=1,
            upsampling=4,
        )

    def forward(self, x: torch.Tensor) -> dict:
        """
        Returns a dict containing:
          - 'seg_logits': (B, num_classes, H, W)
          - 'heatmap_logits': (B, 1, H, W)
        """
        features = self.base_model.encoder(x)
        # DeepLabV3Plus decoder expects a list/tuple of features, not unpacked
        decoder_output = self.base_model.decoder(features)
        
        seg_logits = self.base_model.segmentation_head(decoder_output)
        heatmap_logits = self.heatmap_head(decoder_output)
        
        return {
            "seg_logits": seg_logits,
            "heatmap_logits": heatmap_logits,
        }


# ---------------------------------------------------------------------------
# 4. FinesDetector (Branch 2)
# ---------------------------------------------------------------------------

class FinesDetector(nn.Module):
    def __init__(self, base_channels=32, in_channels=3):
        super().__init__()
        c = base_channels
        layers = [
            nn.Conv2d(in_channels, c, 3, padding=1, dilation=1, bias=False), nn.BatchNorm2d(c), nn.ReLU(True),
            nn.Conv2d(c, c, 3, padding=1, dilation=1, bias=False), nn.BatchNorm2d(c), nn.ReLU(True),
            nn.Conv2d(c, c*2, 3, padding=2, dilation=2, bias=False), nn.BatchNorm2d(c*2), nn.ReLU(True),
            nn.Conv2d(c*2, c*2, 3, padding=2, dilation=2, bias=False), nn.BatchNorm2d(c*2), nn.ReLU(True),
            nn.Conv2d(c*2, c*2, 3, padding=4, dilation=4, bias=False), nn.BatchNorm2d(c*2), nn.ReLU(True),
            nn.Conv2d(c*2, c*2, 3, padding=4, dilation=4, bias=False), nn.BatchNorm2d(c*2), nn.ReLU(True),
            nn.Conv2d(c*2, c*2, 3, padding=8, dilation=8, bias=False), nn.BatchNorm2d(c*2), nn.ReLU(True),
            nn.Conv2d(c*2, c*2, 3, padding=8, dilation=8, bias=False), nn.BatchNorm2d(c*2), nn.ReLU(True),
            nn.Conv2d(c*2, c, 3, padding=4, dilation=4, bias=False), nn.BatchNorm2d(c), nn.ReLU(True),
            nn.Conv2d(c, 1, 1, padding=0, dilation=1, bias=True),
        ]
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

# Not used by the ParticleVision pipeline (kept from earlier experiments).
class SmallFinesDetector(nn.Module):
    def __init__(self, base_channels=16):
        super().__init__()
        c = base_channels
        layers = [
            nn.Conv2d(3, c, 3, padding=1, dilation=1, bias=False), nn.BatchNorm2d(c), nn.ReLU(True),
            nn.Conv2d(c, c*2, 3, padding=2, dilation=2, bias=False), nn.BatchNorm2d(c*2), nn.ReLU(True),
            nn.Conv2d(c*2, c*2, 3, padding=4, dilation=4, bias=False), nn.BatchNorm2d(c*2), nn.ReLU(True),
            nn.Conv2d(c*2, c, 3, padding=2, dilation=2, bias=False), nn.BatchNorm2d(c), nn.ReLU(True),
            nn.Conv2d(c, 1, 1, padding=0, dilation=1, bias=True),
        ]
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

# Not used by the ParticleVision pipeline (kept from earlier experiments).
def build_model(arch: str = "unet", **kwargs) -> nn.Module:
    """
    Build a model by name.

    Parameters
    ----------
    arch : 'unet', 'sam', or 'dual_head'
    **kwargs : forwarded to the builder function.
    """
    if arch == "unet":
        return build_unet(**kwargs)
    elif arch == "sam":
        return SAMEncoderDecoder(**kwargs)
    elif arch == "dual_head":
        return DualHeadModel(**kwargs)
    else:
        raise ValueError(f"Unknown architecture: {arch!r}")


# ---------------------------------------------------------------------------
# Utility: count parameters
# ---------------------------------------------------------------------------

# Not used by the ParticleVision pipeline (kept from earlier experiments).
def count_parameters(model: nn.Module) -> dict:
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
        "total_M": total / 1e6,
        "trainable_M": trainable / 1e6,
    }
