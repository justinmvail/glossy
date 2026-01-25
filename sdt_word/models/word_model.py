"""
SDT modified for word-level handwriting generation.

Key changes from original SDT:
1. Content encoder handles variable-width images (64 x W)
2. No mean pooling - preserve full spatial sequence for word structure
3. Content features attend to style features via cross-attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from einops import rearrange, repeat
import math


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)  # [max_len, 1, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x, step=None):
        """x: [seq_len, batch, d_model]"""
        if step is None:
            x = x + self.pe[:x.size(0)]
        else:
            x = x + self.pe[step]
        return self.dropout(x)


class WordContentEncoder(nn.Module):
    """
    Content encoder for word-level images.
    Handles variable-width input and outputs a sequence of content features.
    """
    def __init__(self, d_model=512, nhead=8, num_layers=2, dropout=0.1):
        super().__init__()

        # Feature extractor - modified ResNet18 for variable width
        # Output: (B, 512, H/32, W/32) for input (B, 1, H, W)
        resnet = models.resnet18(pretrained=True)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        # Project to d_model
        self.proj = nn.Linear(512, d_model)

        # Positional encoding
        self.pos_enc = PositionalEncoding(d_model, dropout)

        # Transformer encoder for content features
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4,
            dropout=dropout, activation='relu', batch_first=False
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        """
        x: [B, 1, H, W] - grayscale word image, H=64, W=variable

        Returns: [seq_len, B, d_model] - content feature sequence
        """
        # Feature extraction
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)  # [B, 512, H/32, W/32]

        B, C, H, W = x.shape
        # Flatten spatial dimensions: [B, 512, H*W] -> [H*W, B, 512]
        x = x.view(B, C, -1).permute(2, 0, 1)  # [seq_len, B, 512]

        # Project and add positional encoding
        x = self.proj(x)  # [seq_len, B, d_model]
        x = self.pos_enc(x)

        # Encode
        x = self.encoder(x)
        return x


class StyleEncoder(nn.Module):
    """
    Style encoder - extracts writer style from reference handwriting samples.
    Uses same architecture as original SDT.
    """
    def __init__(self, d_model=512, nhead=8, num_layers=2, dropout=0.1):
        super().__init__()

        # Feature extractor
        resnet = models.resnet18(pretrained=True)
        self.feat_encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
        )

        self.pos_enc = PositionalEncoding(d_model, dropout)

        # Base encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4,
            dropout=dropout, activation='relu', batch_first=False
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Writer and glyph heads
        self.writer_head = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, d_model*4, dropout, 'relu', batch_first=False),
            num_layers=1
        )
        self.glyph_head = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, d_model*4, dropout, 'relu', batch_first=False),
            num_layers=1
        )

    def forward(self, style_imgs):
        """
        style_imgs: [B, N, 1, 64, 64] - N reference samples per batch

        Returns:
            writer_style: [4*N, B, d_model] - writer-level style features
            glyph_style: [4*N, B, d_model] - glyph-level style features
        """
        B, N, C, H, W = style_imgs.shape
        # Process all style images
        x = style_imgs.view(B * N, C, H, W)  # [B*N, 1, 64, 64]
        x = self.feat_encoder(x)  # [B*N, 512, 2, 2]

        # Reshape and encode
        x = x.view(B * N, 512, -1).permute(2, 0, 1)  # [4, B*N, 512]
        x = self.pos_enc(x)
        x = self.encoder(x)  # [4, B*N, 512]

        # Split into writer and glyph features
        writer_feat = self.writer_head(x)  # [4, B*N, 512]
        glyph_feat = self.glyph_head(x)    # [4, B*N, 512]

        # Reshape for decoder: [4, B*N, C] -> [4*N, B, C]
        writer_feat = rearrange(writer_feat, 't (b n) c -> (t n) b c', b=B, n=N)
        glyph_feat = rearrange(glyph_feat, 't (b n) c -> (t n) b c', b=B, n=N)

        return writer_feat, glyph_feat


class StrokeDecoder(nn.Module):
    """
    Decoder that generates stroke sequences conditioned on content and style.
    """
    def __init__(self, d_model=512, nhead=8, num_layers=4, dropout=0.1, n_mixtures=20):
        super().__init__()

        self.d_model = d_model
        self.n_mixtures = n_mixtures

        # Sequence embedding (stroke -> d_model)
        self.seq_to_emb = nn.Sequential(
            nn.Linear(5, 256),  # (dx, dy, p1, p2, p3)
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, d_model)
        )

        # Positional encoding
        self.pos_enc = PositionalEncoding(d_model, dropout)

        # Transformer decoder layers
        # First attend to writer style, then to content+glyph
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4,
            dropout=dropout, activation='relu', batch_first=False
        )
        self.writer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers//2)
        self.content_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers//2)

        # Output projection to GMM parameters
        # For each mixture: pi, mu_x, mu_y, sigma_x, sigma_y, rho + pen states (3)
        # Total: n_mixtures * 6 + 3
        gmm_out = n_mixtures * 6 + 3
        self.emb_to_seq = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, gmm_out)
        )

    def forward(self, seq, content_feat, writer_style, glyph_style, tgt_mask=None):
        """
        seq: [B, T, 5] - input stroke sequence
        content_feat: [content_len, B, d_model] - content features
        writer_style: [style_len, B, d_model] - writer style features
        glyph_style: [style_len, B, d_model] - glyph style features

        Returns: [B, T, gmm_params] - GMM parameters for each timestep
        """
        # Embed sequence
        seq_emb = self.seq_to_emb(seq)  # [B, T, d_model]
        seq_emb = seq_emb.permute(1, 0, 2)  # [T, B, d_model]

        # Prepend content summary as first token
        content_summary = content_feat.mean(dim=0, keepdim=True)  # [1, B, d_model]
        tgt = torch.cat([content_summary, seq_emb], dim=0)  # [T+1, B, d_model]

        # Add positional encoding
        tgt = self.pos_enc(tgt)

        # Generate mask if not provided
        T = tgt.size(0)
        if tgt_mask is None:
            tgt_mask = self._generate_square_subsequent_mask(T).to(tgt.device)

        # Combine content and glyph features for memory
        combined_memory = torch.cat([content_feat, glyph_style], dim=0)

        # Decode: first with writer style, then with content+glyph
        h = self.writer_decoder(tgt, writer_style, tgt_mask=tgt_mask)
        h = self.content_decoder(h, combined_memory, tgt_mask=tgt_mask)

        # Output GMM parameters
        h = h.permute(1, 0, 2)  # [B, T+1, d_model]
        out = self.emb_to_seq(h[:, 1:])  # [B, T, gmm_params] - skip content token

        return out

    def _generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask


class WordSDT(nn.Module):
    """
    Complete SDT model modified for word-level generation.
    """
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=2,
                 num_decoder_layers=4, dropout=0.1, n_mixtures=20):
        super().__init__()

        self.content_encoder = WordContentEncoder(d_model, nhead, num_encoder_layers, dropout)
        self.style_encoder = StyleEncoder(d_model, nhead, num_encoder_layers, dropout)
        self.decoder = StrokeDecoder(d_model, nhead, num_decoder_layers, dropout, n_mixtures)

        self.n_mixtures = n_mixtures
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, content_img, style_imgs, seq):
        """
        Training forward pass.

        content_img: [B, 1, 64, W] - word content image
        style_imgs: [B, N, 1, 64, 64] - N style reference images
        seq: [B, T, 5] - target stroke sequence

        Returns: GMM parameters [B, T, gmm_params]
        """
        # Encode content and style
        content_feat = self.content_encoder(content_img)
        writer_style, glyph_style = self.style_encoder(style_imgs)

        # Decode
        output = self.decoder(seq, content_feat, writer_style, glyph_style)

        return output

    def generate(self, content_img, style_imgs, max_len=500, temperature=1.0):
        """
        Generate stroke sequence autoregressively.

        Returns: [B, T, 5] - generated stroke sequence
        """
        B = content_img.size(0)
        device = content_img.device

        # Encode content and style
        content_feat = self.content_encoder(content_img)
        writer_style, glyph_style = self.style_encoder(style_imgs)

        # Initialize sequence with zeros (start token)
        seq = torch.zeros(B, 1, 5).to(device)
        generated = []

        for _ in range(max_len):
            # Get prediction for next step
            output = self.decoder(seq, content_feat, writer_style, glyph_style)
            next_params = output[:, -1]  # [B, gmm_params]

            # Sample from GMM
            next_point = self._sample_from_gmm(next_params, temperature)
            generated.append(next_point)

            # Check for end of sequence (pen up with end)
            if next_point[:, 4].sum() == B:  # All ended
                break

            # Append to sequence
            seq = torch.cat([seq, next_point.unsqueeze(1)], dim=1)

        return torch.stack(generated, dim=1)  # [B, T, 5]

    def _sample_from_gmm(self, params, temperature=1.0):
        """Sample next point from GMM parameters."""
        B = params.size(0)
        M = self.n_mixtures

        # Parse parameters
        pi = params[:, :M]  # mixture weights
        mu_x = params[:, M:2*M]
        mu_y = params[:, 2*M:3*M]
        sigma_x = torch.exp(params[:, 3*M:4*M])
        sigma_y = torch.exp(params[:, 4*M:5*M])
        rho = torch.tanh(params[:, 5*M:6*M])
        pen = params[:, 6*M:]  # pen state logits

        # Sample mixture component
        pi = F.softmax(pi / temperature, dim=-1)
        k = torch.multinomial(pi, 1).squeeze(-1)  # [B]

        # Get parameters for selected component
        idx = torch.arange(B, device=params.device)
        mu_x_k = mu_x[idx, k]
        mu_y_k = mu_y[idx, k]
        sigma_x_k = sigma_x[idx, k] * temperature
        sigma_y_k = sigma_y[idx, k] * temperature
        rho_k = rho[idx, k]

        # Sample from bivariate Gaussian
        z1 = torch.randn(B, device=params.device)
        z2 = torch.randn(B, device=params.device)

        x = mu_x_k + sigma_x_k * z1
        y = mu_y_k + sigma_y_k * (rho_k * z1 + torch.sqrt(1 - rho_k**2) * z2)

        # Sample pen state
        pen_probs = F.softmax(pen / temperature, dim=-1)
        pen_state = torch.multinomial(pen_probs, 1).squeeze(-1)  # [B]

        # Convert to one-hot
        pen_onehot = F.one_hot(pen_state, num_classes=3).float()

        return torch.stack([x, y, pen_onehot[:, 0], pen_onehot[:, 1], pen_onehot[:, 2]], dim=-1)


if __name__ == "__main__":
    # Test the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = WordSDT(d_model=512, nhead=8).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass
    B, N, T = 2, 4, 100
    W = 256  # Variable width

    content_img = torch.randn(B, 1, 64, W).to(device)
    style_imgs = torch.randn(B, N, 1, 64, 64).to(device)
    seq = torch.randn(B, T, 5).to(device)

    output = model(content_img, style_imgs, seq)
    print(f"Output shape: {output.shape}")  # [B, T, gmm_params]

    # Test generation
    generated = model.generate(content_img, style_imgs, max_len=50)
    print(f"Generated shape: {generated.shape}")
