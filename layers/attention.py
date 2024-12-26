import torch
import torch.nn as nn
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

class BidirectionalTemporalAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, lookback_heads, lookahead_heads, block_size=128):
        super().__init__()
        assert num_heads == lookback_heads + lookahead_heads, \
            "Total heads must equal the sum of lookback and lookahead heads."
        assert embed_dim % num_heads == 0, \
            "Embedding dimension must be divisible by the number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.lookback_heads = lookback_heads
        self.lookahead_heads = lookahead_heads
        self.head_dim = embed_dim // num_heads
        self.block_size = block_size

        # Separate QKV projections for lookback heads
        self.q_projs_lb = nn.ModuleList([
            nn.Linear(embed_dim, self.head_dim, bias=False) for _ in range(self.lookback_heads)
        ])
        self.k_projs_lb = nn.ModuleList([
            nn.Linear(embed_dim, self.head_dim, bias=False) for _ in range(self.lookback_heads)
        ])
        self.v_projs_lb = nn.ModuleList([
            nn.Linear(embed_dim, self.head_dim, bias=False) for _ in range(self.lookback_heads)
        ])

        # Separate QKV projections for lookahead heads
        self.q_projs_la = nn.ModuleList([
            nn.Linear(embed_dim, self.head_dim, bias=False) for _ in range(self.lookahead_heads)
        ])
        self.k_projs_la = nn.ModuleList([
            nn.Linear(embed_dim, self.head_dim, bias=False) for _ in range(self.lookahead_heads)
        ])
        self.v_projs_la = nn.ModuleList([
            nn.Linear(embed_dim, self.head_dim, bias=False) for _ in range(self.lookahead_heads)
        ])

        # Output projection similar to MHA
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        # Initialize weights as in standard MHA
        nn.init.xavier_uniform_(self.out_proj.weight)
        for proj in self.q_projs_lb + self.k_projs_lb + self.v_projs_lb + \
                   self.q_projs_la + self.k_projs_la + self.v_projs_la:
            nn.init.xavier_uniform_(proj.weight)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, T, F, D)
                B: Batch size
                T: Sequence length (temporal dimension, number of frames)
                F: Number of spatial patches per frame
                D: Embedding dimension
        Returns:
            Output tensor of shape (B, T, F, D)
        """
        B, T, F, D = x.shape
        x = x.reshape(B, T * F, D)  # Flatten frames into a combined temporal-spatial sequence
        TF = T * F  # Combined temporal-spatial sequence length

        # Process Lookback Heads
        # Collect Q, K, V for all lookback heads
        Q_lb = torch.stack([proj(x) for proj in self.q_projs_lb], dim=1)  # (B, lookback_heads, TF, head_dim)
        K_lb = torch.stack([proj(x) for proj in self.k_projs_lb], dim=1)  # (B, lookback_heads, TF, head_dim)
        V_lb = torch.stack([proj(x) for proj in self.v_projs_lb], dim=1)  # (B, lookback_heads, TF, head_dim)

        # Define lookback mask
        def lookback_mask(batch, head, q_idx, kv_idx):
            q_frame, q_patch = divmod(q_idx, F)
            k_frame, k_patch = divmod(kv_idx, F)
            return (q_frame > k_frame) or (q_frame == k_frame and q_patch >= k_patch)

        lookback_block_mask = create_block_mask(
            lookback_mask,
            B=B,
            H=self.lookback_heads,
            Q_LEN=TF,
            KV_LEN=TF,
            device=x.device,
            BLOCK_SIZE=self.block_size
        )  # (B, lookback_heads, TF, TF)

        # Apply FlexAttention to lookback heads
        attn_output_lb = flex_attention(Q_lb, K_lb, V_lb, block_mask=lookback_block_mask)  # (B, lookback_heads, TF, head_dim)

        # Process Lookahead Heads
        # Collect Q, K, V for all lookahead heads
        Q_la = torch.stack([proj(x) for proj in self.q_projs_la], dim=1)  # (B, lookahead_heads, TF, head_dim)
        K_la = torch.stack([proj(x) for proj in self.k_projs_la], dim=1)  # (B, lookahead_heads, TF, head_dim)
        V_la = torch.stack([proj(x) for proj in self.v_projs_la], dim=1)  # (B, lookahead_heads, TF, head_dim)

        # Define lookahead mask
        def lookahead_mask(batch, head, q_idx, kv_idx):
            q_frame, q_patch = divmod(q_idx, F)
            k_frame, k_patch = divmod(kv_idx, F)
            return (q_frame < k_frame) or (q_frame == k_frame and q_patch <= k_patch)

        lookahead_block_mask = create_block_mask(
            lookahead_mask,
            B=B,
            H=self.lookahead_heads,
            Q_LEN=TF,
            KV_LEN=TF,
            device=x.device,
            BLOCK_SIZE=self.block_size
        )  # (B, lookahead_heads, TF, TF)

        # Apply FlexAttention to lookahead heads
        attn_output_la = flex_attention(Q_la, K_la, V_la, block_mask=lookahead_block_mask)  # (B, lookahead_heads, TF, head_dim)

        # Concatenate Lookback and Lookahead Outputs: (B, num_heads, TF, head_dim)
        attn_output = torch.cat([attn_output_lb, attn_output_la], dim=1)  # (B, num_heads, TF, head_dim)

        # Reshape to (B, TF, embed_dim)
        attn_output = attn_output.reshape(B, TF, D)

        # Apply Output Projection (MHA-like)
        output = self.out_proj(attn_output)  # Shape: (B, TF, D)

        # Reshape back to (B, T, F, D)
        output = output.reshape(B, T, F, D)

        return output
