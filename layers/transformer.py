from layers.attention import BidirectionalTemporalAttention
import torch
import torch.nn as nn


class BTTransformer(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_attention_heads,
        num_lookback_heads,
        num_lookahead_heads,
        ff_hidden_dim=None,
        local_frame_range=16,
        dropout=0.1,
    ):
        super().__init__()
        if ff_hidden_dim is None:
            ff_hidden_dim = 4 * embed_dim
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attention = BidirectionalTemporalAttention(
            embed_dim=embed_dim,
            num_attention_heads=num_attention_heads,
            num_lookback_heads=num_lookback_heads,
            num_lookahead_heads=num_lookahead_heads,
            local_frame_range=local_frame_range,
        )
        self.dropout1 = nn.Dropout(dropout)

        self.ln2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_hidden_dim),
            nn.GELU(),
            nn.Linear(ff_hidden_dim, embed_dim),
        )
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, T, F, D)
        Returns:
            Output tensor of shape (B, T, F, D)
        """
        # Pre-LN and Attention and Residual connection
        x = x + self.dropout1(self.attention(self.ln1(x)))

        # Pre-LN and Feedforward and Residual connection
        x = x + self.dropout2(self.ffn(self.ln2(x)))

        return x
