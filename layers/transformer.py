from layers.attention import BidirectionalTemporalAttention
import torch
import torch.nn as nn


class BTTransformer(nn.Module):
    def __init__(self, embed_dim, num_heads, lookback_heads, lookahead_heads, frame_size, 
                 ff_hidden_dim, dropout=0.1):
        super().__init__()
        self.pre_ln1 = nn.LayerNorm(embed_dim)
        self.attention = BidirectionalTemporalAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            lookback_heads=lookback_heads,
            lookahead_heads=lookahead_heads,
            frame_size=frame_size,
        )
        self.dropout1 = nn.Dropout(dropout)

        self.pre_ln2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_hidden_dim),
            nn.GELU(),
            nn.Linear(ff_hidden_dim, embed_dim)
        )
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, T, F, D)
        Returns:
            Output tensor of shape (B, T, F, D)
        """
        # Pre-LN and Attention
        ln1 = self.pre_ln1(x)
        attn_out = self.attention(ln1)
        attn_out = self.dropout1(attn_out)
        x = x + attn_out  # Residual connection

        # Pre-LN and Feedforward
        ln2 = self.pre_ln2(x)
        ffn_out = self.ffn(ln2)
        ffn_out = self.dropout2(ffn_out)
        x = x + ffn_out  # Residual connection

        return x
