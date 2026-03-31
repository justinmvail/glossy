"""Autoregressive stroke prediction from glyph rasters.

Architecture: ResNet-18 encoder + autoregressive decoder that predicts one stroke
at a time, conditioned on what's already been drawn (the residual).

Each step:
    1. Encode the residual (what's left to draw) + current canvas
    2. Attend to glyph features + canvas state
    3. Predict one stroke (points, width, existence)
    4. Render stroke onto canvas
    5. Repeat until MAX_STROKES or existence < threshold
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


# Constants
MAX_STROKES = 8
MAX_POINTS = 40
NUM_CHARS = 62  # A-Z + a-z + 0-9
CHAR_EMBED_DIM = 32
CANVAS_SIZE = 224
RENDER_SIZE = 56
HIRES_RENDER_SIZE = 224  # full-res for high-res loss
SUBDIVISIONS = 4  # Catmull-Rom spline subdivisions per segment


class StrokeEncoder(nn.Module):
    """ResNet-18 backbone with character embedding fusion.

    Takes a 224x224 grayscale image and a character index, produces a
    sequence of spatial feature tokens for the transformer decoder.
    """

    def __init__(self, feature_dim: int = 256):
        super().__init__()
        self.feature_dim = feature_dim

        # ResNet-18 backbone (pretrained, modified for 1-channel input)
        self.backbone = timm.create_model(
            'resnet18', pretrained=True, features_only=True, out_indices=[4],
        )
        # Replace first conv: 3-channel -> 1-channel (average pretrained weights)
        old_conv = self.backbone.conv1
        self.backbone.conv1 = nn.Conv2d(
            1, old_conv.out_channels, kernel_size=old_conv.kernel_size,
            stride=old_conv.stride, padding=old_conv.padding, bias=False,
        )
        with torch.no_grad():
            self.backbone.conv1.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))

        # Character embedding
        self.char_embed = nn.Embedding(NUM_CHARS, CHAR_EMBED_DIM)

        # Project ResNet features (512) + char embed -> feature_dim
        self.proj = nn.Linear(512 + CHAR_EMBED_DIM, feature_dim)

    def forward(self, image: torch.Tensor, char_idx: torch.Tensor) -> torch.Tensor:
        """Encode image and character into feature sequence.

        Args:
            image: (B, 1, 224, 224) grayscale glyph image.
            char_idx: (B,) integer character indices.

        Returns:
            (B, S, feature_dim) feature sequence where S = 7*7 = 49 spatial tokens.
        """
        # Extract spatial features: (B, 512, 7, 7)
        features = self.backbone(image)[0]
        B, C, H, W = features.shape

        # Flatten spatial dims: (B, 49, 512)
        features = features.reshape(B, C, H * W).permute(0, 2, 1)

        # Character embedding: (B, char_embed_dim) -> (B, 49, char_embed_dim)
        char_emb = self.char_embed(char_idx).unsqueeze(1).expand(-1, H * W, -1)

        # Concatenate and project: (B, 49, feature_dim)
        fused = torch.cat([features, char_emb], dim=-1)
        return self.proj(fused)


class ResidualEncoder(nn.Module):
    """Lightweight CNN to encode current canvas state and residual.

    Input: 2-channel (canvas, residual) at RENDER_SIZE x RENDER_SIZE.
    Output: (B, 49, state_dim) spatial tokens aligned with encoder's 7x7 grid.
    """

    def __init__(self, state_dim: int = 64):
        super().__init__()
        # 3 stride-2 convs: 56 -> 28 -> 14 -> 7
        self.conv = nn.Sequential(
            nn.Conv2d(2, 32, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, state_dim, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, canvas_and_residual: torch.Tensor) -> torch.Tensor:
        """
        Args:
            canvas_and_residual: (B, 2, R, R) where R=RENDER_SIZE.

        Returns:
            (B, 49, state_dim) spatial tokens.
        """
        feat = self.conv(canvas_and_residual)  # (B, state_dim, 7, 7)
        B, C, H, W = feat.shape
        return feat.reshape(B, C, H * W).permute(0, 2, 1)  # (B, 49, state_dim)


class StrokeDecoderStep(nn.Module):
    """Single-stroke decoder: one query attends to glyph features + canvas state."""

    def __init__(self, feature_dim: int = 256, state_dim: int = 64,
                 num_heads: int = 4, num_layers: int = 2):
        super().__init__()

        # Project state tokens to match feature dim
        self.state_proj = nn.Linear(state_dim, feature_dim)

        # Single learned query
        self.stroke_query = nn.Parameter(torch.randn(1, feature_dim) * 0.02)

        # Step embedding so decoder knows which stroke number it's predicting
        self.step_embed = nn.Embedding(MAX_STROKES, feature_dim)

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=feature_dim, nhead=num_heads,
            dim_feedforward=feature_dim * 4, dropout=0.1, batch_first=True,
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

    def forward(self, encoder_features: torch.Tensor,
                state_tokens: torch.Tensor, step: int) -> torch.Tensor:
        """Decode a single stroke representation.

        Args:
            encoder_features: (B, 49, feature_dim) from StrokeEncoder.
            state_tokens: (B, 49, state_dim) from ResidualEncoder.
            step: int, which stroke number (0-7).

        Returns:
            (B, feature_dim) stroke feature vector.
        """
        B = encoder_features.shape[0]
        device = encoder_features.device

        # Project state and concatenate with glyph features along sequence dim
        state_proj = self.state_proj(state_tokens)  # (B, 49, feature_dim)
        memory = torch.cat([encoder_features, state_proj], dim=1)  # (B, 98, feature_dim)

        # Query with step embedding
        step_t = torch.tensor([step], device=device)
        query = self.stroke_query.unsqueeze(0).expand(B, -1, -1)  # (B, 1, feature_dim)
        query = query + self.step_embed(step_t).unsqueeze(0)  # add step identity

        # Decode
        out = self.transformer(query, memory)  # (B, 1, feature_dim)
        return out.squeeze(1)  # (B, feature_dim)


class StrokePredictor(nn.Module):
    """Autoregressive stroke predictor.

    Predicts strokes one at a time, rendering each onto a canvas and using
    the residual (what's left to cover) to inform the next stroke.
    """

    def __init__(self, feature_dim: int = 256, state_dim: int = 64):
        super().__init__()
        self.encoder = StrokeEncoder(feature_dim=feature_dim)
        self.residual_encoder = ResidualEncoder(state_dim=state_dim)
        self.decoder = StrokeDecoderStep(
            feature_dim=feature_dim, state_dim=state_dim,
        )

        # Output heads (single stroke)
        self.existence_head = nn.Linear(feature_dim, 1)
        self.points_head = nn.Linear(feature_dim, MAX_POINTS * 2)
        self.width_head = nn.Linear(feature_dim, MAX_POINTS)  # per-point widths
        self.point_count_head = nn.Linear(feature_dim, MAX_POINTS)

    def forward(self, image: torch.Tensor, char_idx: torch.Tensor,
                glyph_mask: torch.Tensor) -> dict:
        """Predict strokes autoregressively.

        Args:
            image: (B, 1, 224, 224) grayscale glyph image.
            char_idx: (B,) integer character indices.
            glyph_mask: (B, H, W) binary glyph mask (1=glyph, 0=bg).

        Returns:
            Dict with per-step predictions and final canvas.
        """
        from triton_render import render_single_stroke_triton

        B = image.shape[0]
        device = image.device
        R = RENDER_SIZE

        # Encode glyph once
        features = self.encoder(image, char_idx)  # (B, 49, 256)

        # Downsample glyph mask to render resolution
        target = F.interpolate(
            glyph_mask.unsqueeze(1), size=(R, R),
            mode='bilinear', align_corners=False,
        ).squeeze(1)  # (B, R, R)

        # Canvas in "inverse ink" space: 1=blank, 0=inked
        canvas_inv = torch.ones(B, R, R, device=device)

        all_existence = []
        all_points = []
        all_widths = []
        all_point_counts = []
        all_stroke_renders = []

        for step in range(MAX_STROKES):
            # Current ink and residual
            ink = 1.0 - canvas_inv  # 1=inked, 0=blank
            residual = (target - ink).clamp(0, 1)  # what still needs covering

            # Encode canvas state
            state_input = torch.stack([ink, residual], dim=1)  # (B, 2, R, R)
            state_tokens = self.residual_encoder(state_input)  # (B, 49, 64)

            # Decode single stroke
            stroke_feat = self.decoder(features, state_tokens, step)  # (B, 256)

            # Predict stroke parameters
            existence = torch.sigmoid(self.existence_head(stroke_feat).squeeze(-1))  # (B,)
            points_raw = self.points_head(stroke_feat)  # (B, 80)
            points = torch.sigmoid(points_raw.reshape(B, MAX_POINTS, 2))  # (B, 40, 2)
            widths = F.softplus(self.width_head(stroke_feat)) + 1.0  # (B, 40) per-point
            pc_logits = self.point_count_head(stroke_feat)  # (B, 40)
            n_pts = pc_logits.argmax(dim=-1) + 1  # (B,)
            n_pts = n_pts.clamp(min=2, max=MAX_POINTS)

            # Render this stroke: (B, R, R), 1=bg, 0=ink
            stroke_render = render_single_stroke_triton(
                points, widths, n_pts, CANVAS_SIZE, R,
            )

            # Composite with existence masking (differentiable multiply-blend)
            # If existence=1: canvas_inv *= stroke_render (add ink)
            # If existence=0: canvas_inv *= 1.0 (no change)
            blend = stroke_render * existence.unsqueeze(-1).unsqueeze(-1) + \
                    1.0 * (1.0 - existence.unsqueeze(-1).unsqueeze(-1))
            canvas_inv = canvas_inv * blend

            all_existence.append(existence)
            all_points.append(points)
            all_widths.append(widths)
            all_point_counts.append(pc_logits)
            all_stroke_renders.append(stroke_render)

        return {
            'existence': torch.stack(all_existence, dim=1),         # (B, MAX_STROKES)
            'points': torch.stack(all_points, dim=1),               # (B, MAX_STROKES, 40, 2)
            'widths': torch.stack(all_widths, dim=1),               # (B, MAX_STROKES, 40)
            'point_count_logits': torch.stack(all_point_counts, dim=1),
            'stroke_renders': torch.stack(all_stroke_renders, dim=1),  # (B, MAX_STROKES, R, R)
            'canvas_inv': canvas_inv,                               # (B, R, R)
            'target': target,                                        # (B, R, R)
            'glyph_mask': glyph_mask,                               # (B, H, W) full-res
        }

    def predict_strokes(self, image: torch.Tensor, char_idx: torch.Tensor,
                        canvas_size: int = CANVAS_SIZE,
                        existence_threshold: float = 0.5) -> tuple:
        """Predict strokes for inference (no ground truth mask needed).

        Args:
            image: (1, 1, 224, 224) single glyph image.
            char_idx: (1,) character index.
            canvas_size: Canvas size to scale coordinates to.
            existence_threshold: Minimum existence probability.

        Returns:
            Tuple of (strokes_list, avg_width).
        """
        self.eval()
        # Derive glyph mask from input image
        glyph_mask = (image.squeeze(1) < 0.5).float()  # (1, 224, 224)

        with torch.no_grad():
            out = self.forward(image, char_idx, glyph_mask)

        existence = out['existence'][0]
        points = out['points'][0]
        widths = out['widths'][0]
        point_count_logits = out['point_count_logits'][0]

        strokes = []
        stroke_widths = []
        for i in range(MAX_STROKES):
            if existence[i].item() < existence_threshold:
                break

            n_pts = torch.argmax(point_count_logits[i]).item() + 1
            n_pts = max(2, min(n_pts, MAX_POINTS))

            stroke_pts = points[i, :n_pts] * canvas_size  # (n, 2)

            # Smooth control points: 2-pass moving average on interior points
            # Removes barbs/spikes without introducing wobbles
            for _ in range(2):
                if stroke_pts.shape[0] > 2:
                    smoothed = stroke_pts.clone()
                    smoothed[1:-1] = (stroke_pts[:-2] + stroke_pts[1:-1] + stroke_pts[2:]) / 3
                    stroke_pts = smoothed

            stroke = [[round(float(p[0]), 1), round(float(p[1]), 1)]
                      for p in stroke_pts]
            strokes.append(stroke)
            stroke_w = [float(widths[i, j].item()) for j in range(n_pts)]
            stroke_widths.append(stroke_w)

        return strokes, stroke_widths


CHARS_LIST = list('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789')

def char_to_index(char: str) -> int:
    """Convert a character to its index (0-61)."""
    try:
        return CHARS_LIST.index(char)
    except ValueError:
        return 0
