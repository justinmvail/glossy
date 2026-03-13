"""Neural network for predicting centerline strokes from glyph rasters.

Architecture: ResNet-18 encoder + Transformer decoder with learned stroke queries.
The model takes a 224x224 grayscale glyph image and a character label, and predicts
up to 8 strokes with variable point counts.

Output per stroke query:
    - existence: sigmoid probability (does this stroke exist?)
    - points: up to 40 x 2 coordinates (normalized to [0, 1])
    - width: single float (stroke width in pixels)
    - point_count: how many of the 40 points are valid
"""

import torch
import torch.nn as nn
import timm


# Constants
MAX_STROKES = 8
MAX_POINTS = 40
NUM_CHARS = 62  # A-Z + a-z + 0-9
CHAR_EMBED_DIM = 32
CANVAS_SIZE = 224


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


class StrokeDecoder(nn.Module):
    """Transformer decoder with learned stroke queries.

    Uses MAX_STROKES learned queries to attend to encoder features
    and produce per-stroke predictions.
    """

    def __init__(self, feature_dim: int = 256, num_heads: int = 4,
                 num_layers: int = 3):
        super().__init__()
        self.feature_dim = feature_dim

        # Learned stroke queries: (MAX_STROKES, feature_dim)
        self.stroke_queries = nn.Parameter(
            torch.randn(MAX_STROKES, feature_dim) * 0.02,
        )

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=feature_dim, nhead=num_heads, dim_feedforward=feature_dim * 4,
            dropout=0.1, batch_first=True,
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

    def forward(self, encoder_features: torch.Tensor) -> torch.Tensor:
        """Decode stroke representations from encoder features.

        Args:
            encoder_features: (B, S, feature_dim) from encoder.

        Returns:
            (B, MAX_STROKES, feature_dim) decoded stroke features.
        """
        B = encoder_features.shape[0]
        # Expand queries for batch: (B, MAX_STROKES, feature_dim)
        queries = self.stroke_queries.unsqueeze(0).expand(B, -1, -1)

        return self.transformer(queries, encoder_features)


class StrokePredictor(nn.Module):
    """Full model: encoder + decoder + output heads.

    Predicts up to MAX_STROKES strokes from a glyph image and character label.
    Each stroke has: existence flag, point coordinates, width, point count.
    """

    def __init__(self, feature_dim: int = 256):
        super().__init__()
        self.encoder = StrokeEncoder(feature_dim=feature_dim)
        self.decoder = StrokeDecoder(feature_dim=feature_dim)

        # Output heads (per stroke query)
        self.existence_head = nn.Linear(feature_dim, 1)
        self.points_head = nn.Linear(feature_dim, MAX_POINTS * 2)
        self.width_head = nn.Linear(feature_dim, 1)
        self.point_count_head = nn.Linear(feature_dim, MAX_POINTS)

    def forward(self, image: torch.Tensor, char_idx: torch.Tensor) -> dict:
        """Predict strokes from glyph image and character label.

        Args:
            image: (B, 1, 224, 224) grayscale glyph image (white bg, black glyph).
            char_idx: (B,) integer character indices (0-61).

        Returns:
            Dict with:
                existence: (B, MAX_STROKES) sigmoid probabilities.
                points: (B, MAX_STROKES, MAX_POINTS, 2) coordinates in [0, 1].
                widths: (B, MAX_STROKES) stroke widths (positive).
                point_counts: (B, MAX_STROKES, MAX_POINTS) logits over point positions.
        """
        # Encode
        features = self.encoder(image, char_idx)

        # Decode
        stroke_features = self.decoder(features)

        # Output heads
        existence = torch.sigmoid(self.existence_head(stroke_features).squeeze(-1))
        points_raw = self.points_head(stroke_features)
        points = torch.sigmoid(points_raw.reshape(-1, MAX_STROKES, MAX_POINTS, 2))
        widths = torch.nn.functional.softplus(self.width_head(stroke_features).squeeze(-1)) + 2.0
        point_count_logits = self.point_count_head(stroke_features)

        return {
            'existence': existence,
            'points': points,
            'widths': widths,
            'point_count_logits': point_count_logits,
        }

    def predict_strokes(self, image: torch.Tensor, char_idx: torch.Tensor,
                        canvas_size: int = CANVAS_SIZE,
                        existence_threshold: float = 0.5) -> list:
        """Predict strokes and convert to list format for JSON output.

        Args:
            image: (1, 1, 224, 224) single glyph image.
            char_idx: (1,) character index.
            canvas_size: Canvas size to scale coordinates to.
            existence_threshold: Minimum existence probability to include a stroke.

        Returns:
            List of strokes: [[[x1, y1], [x2, y2], ...], ...]
        """
        self.eval()
        with torch.no_grad():
            out = self.forward(image, char_idx)

        existence = out['existence'][0]  # (MAX_STROKES,)
        points = out['points'][0]  # (MAX_STROKES, MAX_POINTS, 2)
        widths = out['widths'][0]  # (MAX_STROKES,)
        point_count_logits = out['point_count_logits'][0]  # (MAX_STROKES, MAX_POINTS)

        strokes = []
        for i in range(MAX_STROKES):
            if existence[i].item() < existence_threshold:
                continue

            # Determine point count from logits (argmax gives last valid index)
            n_points = torch.argmax(point_count_logits[i]).item() + 1
            n_points = max(2, min(n_points, MAX_POINTS))

            # Scale from [0, 1] to canvas coordinates and round
            stroke_pts = points[i, :n_points] * canvas_size
            stroke = [[round(float(p[0]), 1), round(float(p[1]), 1)]
                      for p in stroke_pts]
            strokes.append(stroke)

        return strokes, float(widths.mean().item())


CHARS_LIST = list('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789')

def char_to_index(char: str) -> int:
    """Convert a character to its index (0-61)."""
    try:
        return CHARS_LIST.index(char)
    except ValueError:
        return 0
