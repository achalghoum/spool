import torch
import torch.nn as nn
from torch.nn.attention.flex_attention import flex_attention, create_mask


class BidirectionalTemporalAttention(nn.Module):
    def __init__(self, embed_dim, num_attention_heads, num_lookback_heads, num_lookahead_heads):
        super().__init__()
        assert num_attention_heads == num_lookback_heads + num_lookahead_heads, \
            "Total heads must equal the sum of lookback and lookahead heads."
        assert embed_dim % num_attention_heads == 0, \
            "Embedding dimension must be divisible by the number of heads."

        self.embed_dim = embed_dim
        self.num_attention_heads = num_attention_heads
        self.num_lookback_heads = num_lookback_heads
        self.num_lookahead_heads = num_lookahead_heads
        self.head_dim = embed_dim // num_attention_heads

        # Separate QKV projections for each head
        self.q_projs = nn.ModuleList(
            [nn.Linear(embed_dim, self.head_dim) for _ in range(num_attention_heads)])
        self.k_projs = nn.ModuleList(
            [nn.Linear(embed_dim, self.head_dim) for _ in range(num_attention_heads)])
        self.v_projs = nn.ModuleList(
            [nn.Linear(embed_dim, self.head_dim) for _ in range(num_attention_heads)])

        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, T , F, D)
                B: Batch size
                T: Sequence length (temporal dimension, number of frames)
                F: Number of spatial patches per frame
                D: Embedding dimension
        """
        B, T, F, D = x.shape
        # Flatten frames into a combined temporal-spatial sequence
        x = x.reshape(B, T * F, D)
        TF = T * F  # Combined temporal-spatial sequence length
        # Define mask_mod functions for lookback and lookahead with spatial awareness
        def lookback_mask(batch, head, q_idx, kv_idx):
            q_frame, q_patch = divmod(q_idx, F)
            k_frame, k_patch = divmod(kv_idx, F)
            return (q_frame > k_frame) or (q_frame == k_frame and q_patch >= k_patch)

        def lookahead_mask(batch, head, q_idx, kv_idx):
            q_frame, q_patch = divmod(q_idx, F)
            k_frame, k_patch = divmod(kv_idx, F)
            return (q_frame < k_frame) or (q_frame == k_frame and q_patch <= k_patch)

        # Generate masks for lookback and lookahead
        lookback_mask = create_mask(
            lookback_mask, B=B, H=self.num_lookback_heads, Q_LEN=TF, KV_LEN=TF, device=x.device
        )
        lookahead_mask = create_mask(
            lookahead_mask, B=B, H=self.num_lookahead_heads, Q_LEN=TF, KV_LEN=TF, device=x.device
        )

        def get_mask(
            i): return lookback_mask if i < self.num_lookback_heads else lookahead_mask
        # Compute Q, K, V for all heads at once
        # Shape: (B, num_heads, TF, head_dim)
        q = torch.stack([proj(x) for proj in self.q_projs], dim=1)
        # Shape: (B, num_heads, TF, head_dim)
        k = torch.stack([proj(x) for proj in self.k_projs], dim=1)
        # Shape: (B, num_heads, TF, head_dim)
        v = torch.stack([proj(x) for proj in self.v_projs], dim=1)

        # Apply FlexAttention with the appropriate masks for all heads and concatenate all head outputs
        combined_output = torch.cat([flex_attention(q[:, i], k[:, i], v[:, i], 
                                                    mask=get_mask(i)) for i in range(self.num_attention_heads)], dim=-1)  # Shape: (B, TF, embed_dim)

        output = self.out_proj(combined_output)  # Shape: (B, T, F, D)

        return output.reshape(B, T, F, D)
