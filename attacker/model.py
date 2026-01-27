import math
from typing import Tuple

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer sequences.

    Adds position information to the input embeddings so the transformer
    knows the order of the sequence.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 100):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        position = torch.arange(max_len).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, seq_len, d_model)
        Returns:
            Tensor with positional encoding added
        """
        seq_len = x.size(1)
        # Add positional encoding to input
        x = x + self.pe[:seq_len, 0, :].unsqueeze(0)
        return self.dropout(x)


class AttentionInputProjection(nn.Module):
    def __init__(self, gradient_shape=(64, 512), d_model=512, nhead=8, dropout=0.1):
        super().__init__()
        self.num_rows = gradient_shape[0]  # 64
        self.row_dim = gradient_shape[1]  # 512

        # Project each row to d_model
        self.row_projection = nn.Linear(self.row_dim, d_model)

        self.num_queries = 1
        self.query_tokens = nn.Parameter(torch.randn(self.num_queries, d_model))

        # Cross-attention: queries attend to gradient rows
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, gradients: torch.Tensor) -> torch.Tensor:
        # gradients: (batch, seq_len, 64, 512)
        batch_size, seq_len = gradients.shape[:2]

        # Reshape: (batch * seq_len, 64, 512)
        x = gradients.view(batch_size * seq_len, self.num_rows, self.row_dim)

        # Project rows: (batch * seq_len, 64, d_model)
        x = self.row_projection(x)

        # Expand queries: (batch * seq_len, num_queries, d_model)
        queries = self.query_tokens.unsqueeze(0).expand(batch_size * seq_len, -1, -1)

        # Cross-attention: (batch * seq_len, num_queries, d_model)
        attended, _ = self.cross_attention(queries, x, x)

        # Output: (batch * seq_len, num_queries, d_model)
        attended = self.norm(attended)
        attended = self.dropout(attended)

        # Reshape back: (batch, seq_len, num_queries, d_model)
        attended = attended.view(batch_size, seq_len, self.num_queries, -1)
        if self.num_queries == 1:
            attended = attended.squeeze(2)
        return attended


class GradientToImageTransformer(nn.Module):
    def __init__(
        self,
        gradient_shape: Tuple[int, int] = (64, 512),  # grad w / grad b of fc2
        img_size: Tuple[int, int] = (120, 120),
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        # dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.gradient_shape = gradient_shape
        self.img_size = img_size
        self.gradient_dim = gradient_shape[0] * gradient_shape[1]  # 64 * 512 = 32,768
        self.img_pixels = img_size[0] * img_size[1]  # 120 * 120 = 14,400

        # Project gradients to transformer dimension
        # self.input_projection = nn.Sequential(
        #     nn.Linear(self.gradient_dim, d_model),
        #     nn.LayerNorm(d_model),
        #     nn.Dropout(dropout),
        # )
        self.input_projection = AttentionInputProjection(
            gradient_shape=gradient_shape,
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
        )

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=100)

        # Transformer encoder
        # This learns relationships between different actions in the same scene
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            # dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,  # (batch, seq, feature) format
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, norm=nn.LayerNorm(d_model)
        )

        # Project transformer output to initial feature map
        # d_model -> 256 * 8 * 8 (small spatial feature map)
        self.initial_channels = 256
        self.initial_size = 8
        self.feature_projection = nn.Sequential(
            nn.Linear(
                d_model, self.initial_channels * self.initial_size * self.initial_size
            ),
            nn.ReLU(),
        )

        # CNN decoder to upsample from 8x8 to 120x120
        # 8x8 -> 16x16 -> 30x30 -> 60x60 -> 120x120
        self.cnn_decoder = nn.Sequential(
            # 8x8 -> 16x16
            nn.ConvTranspose2d(
                256, 128, kernel_size=4, stride=2, padding=1
            ),  # 256x8x8 -> 128x16x16
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # 16x16 -> 30x30
            nn.ConvTranspose2d(
                128, 64, kernel_size=4, stride=2, padding=1
            ),  # 128x16x16 -> 64x32x32
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # 32x32 -> 60x60
            nn.ConvTranspose2d(
                64, 32, kernel_size=4, stride=2, padding=1
            ),  # 64x32x32 -> 32x64x64
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # 64x64 -> 120x120
            nn.ConvTranspose2d(
                32, 16, kernel_size=4, stride=2, padding=1
            ),  # 32x64x64 -> 16x128x128
            nn.BatchNorm2d(16),
            nn.ReLU(),
            # Final convolution to get exact size and single channel
            nn.Conv2d(
                16, 1, kernel_size=9, stride=1, padding=4
            ),  # 16x128x128 -> 1x128x128
            nn.AdaptiveAvgPool2d(
                (self.img_size[0], self.img_size[1])
            ),  # 1x128x128 -> 1x120x120
            nn.Sigmoid(),  # Output in [0, 1] range
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier uniform initialization."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, gradients: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the transformer.

        Args:
            gradients: Gradient sequences of shape (batch, seq_len, 64, 512)
                      These are the gradients from fc2.weight for each image
                      in a scene (same FloorPlan, different actions)

        Returns:
            Reconstructed images of shape (batch, seq_len, 150, 150)
        """
        batch_size, seq_len = gradients.shape[0], gradients.shape[1]

        # ! changed
        # # Flatten gradients: (batch, seq_len, 64, 512) -> (batch, seq_len, 32768)
        # x = gradients.reshape(batch_size, seq_len, -1)

        # Project to transformer dimension: (batch, seq_len, d_model)
        x = self.input_projection(gradients)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Apply transformer encoder
        # The transformer learns relationships between different actions
        x = self.transformer(x)  # (batch, seq_len, d_model)

        # Project to feature map: (batch, seq_len, d_model) -> (batch, seq_len, 256*8*8)
        x = self.feature_projection(x)

        # Reshape for CNN: (batch, seq_len, 256*8*8) -> (batch*seq_len, 256, 8, 8)
        x = x.view(
            batch_size * seq_len,
            self.initial_channels,
            self.initial_size,
            self.initial_size,
        )

        # Apply CNN decoder: (batch*seq_len, 256, 8, 8) -> (batch*seq_len, 1, 120, 120)
        x = self.cnn_decoder(x)

        # Reshape back: (batch*seq_len, 1, 120, 120) -> (batch, seq_len, 120, 120)
        x = x.view(batch_size, seq_len, self.img_size[0], self.img_size[1])

        return x

    def reconstruct_from_gradients(
        self, gradients: torch.Tensor, return_attention: bool = False
    ) -> torch.Tensor:
        """
        Convenience method for reconstruction.

        Args:
            gradients: Gradient sequences (batch, seq_len, 64, 512)
            return_attention: If True, also return attention weights

        Returns:
            Reconstructed images (batch, seq_len, 150, 150)
        """
        with torch.no_grad():
            reconstructed = self.forward(gradients)
        return reconstructed


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Example usage and testing
if __name__ == "__main__":
    print("=" * 80)
    print("Gradient-to-Image Transformer Architecture")
    print("=" * 80)

    # Create model
    model = GradientToImageTransformer(
        gradient_shape=(64, 512),  # fc2.weight shape
        img_size=(120, 120),
        d_model=512,
        nhead=8,
        num_layers=6,
        dropout=0.1,
    )

    print(f"\nModel Configuration:")
    print(f"  Gradient shape: (64, 512) = 32,768 values")
    print(f"  Image size: (120, 120) = 14,400 pixels")
    print(f"  Transformer dimension: 512")
    print(f"  Attention heads: 8")
    print(f"  Encoder layers: 6")
    print(f"  Total parameters: {count_parameters(model):,}")

    # Test with dummy data
    batch_size = 4  # 4 different scenes
    seq_len = 10  # 10 actions per scene

    print(f"\nTest Configuration:")
    print(f"  Batch size: {batch_size} scenes")
    print(f"  Sequence length: {seq_len} actions per scene")

    # Simulate gradient input from MLP
    dummy_gradients = torch.randn(batch_size, seq_len, 64, 512)

    print(f"\nInput (Gradients from fc2.weight):")
    print(f"  Shape: {dummy_gradients.shape}")
    print(
        f"  Description: {batch_size} scenes x {seq_len} actions x (64 x 512) gradients"
    )

    # Forward pass
    with torch.no_grad():
        reconstructed = model(dummy_gradients)

    print(f"\nOutput (Reconstructed Images):")
    print(f"  Shape: {reconstructed.shape}")
    print(
        f"  Description: {batch_size} scenes x {seq_len} actions x (120 x 120) images"
    )
    print(f"  Value range: [{reconstructed.min():.3f}, {reconstructed.max():.3f}]")

    # Information flow analysis
    print(f"\n" + "=" * 80)
    print("Information Flow Analysis")
    print("=" * 80)

    gradient_info = 32768
    image_info = 14400
    compression_ratio = gradient_info / image_info

    print(f"\nInput information: {gradient_info:,} gradient values")
    print(f"Output information: {image_info:,} pixel values")
    print(f"Compression ratio: {compression_ratio:.2f}x")

    if compression_ratio > 1:
        print(f"✅ Input has {compression_ratio:.2f}x MORE information than output")
        print(f"   The transformer has enough data to reconstruct images!")
    else:
        print(f"⚠️  Input has LESS information than output")
        print(f"   Reconstruction may be challenging")

    print(f"\n" + "=" * 80)
    print("Architecture Details")
    print("=" * 80)
    print(
        """
    1. Input Processing:
       - Flatten gradients: (batch, seq, 64, 512) → (batch, seq, 32768)
       - Linear projection: 32768 → 512 (d_model)
       - Layer normalization + dropout
    
    2. Positional Encoding:
       - Adds sinusoidal position information to sequence
       - Allows model to understand temporal/action order
       - Max sequence length: 100
    
    3. Transformer Encoder (6 layers):
       - Multi-head self-attention (8 heads)
       - Learns relationships between different actions in same scene
       - Feed-forward network with GELU activation
       - Layer normalization for stable training
    
    4. Feature Projection:
       - Linear projection: 512 → 16,384 (256 x 8 x 8)
       - Reshape to spatial feature map: (256, 8, 8)
    
    5. CNN Decoder:
       - ConvTranspose 256→128: 8x8 → 16x16
       - ConvTranspose 128→64:  16x16 → 32x32
       - ConvTranspose 64→32:   32x32 → 64x64
       - ConvTranspose 32→16:   64x64 → 128x128
       - Conv2d 16→1: Final channel reduction
       - AdaptiveAvgPool2d: 128x128 → 120x120
       - Sigmoid: Output in [0, 1] range
    
    6. Normalization Layers:
       - BatchNorm2d after each ConvTranspose layer
       - ReLU activations throughout decoder
    
    Next Steps:
    - Load gradients from features/all_scenes.pkl
    - Create GradientDataset that feeds (batch, 10, 64, 512) tensors
    - Train with MSE loss between reconstructed and original images
    - The CNN decoder provides spatial structure learning!
    """
    )

    print("\n✅ Model architecture validated and ready for training!")
