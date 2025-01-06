import torch
import torch.nn as nn
from torch.nn.attention.flex_attention import flex_attention, create_block_mask, or_masks, and_masks


class BidirectionalTemporalAttention(nn.Module):
    """
    Optimized module for bidirectional temporal attention with spatial awareness.
    This version supports multiple masks and caches them in a hashmap.
    """

    def __init__(self,
                 embed_dim,
                 num_attention_heads,
                 num_lookback_heads,
                 num_lookahead_heads,
                 local_frame_range):
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

        # Unified QKV projections for efficiency
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)

        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.local_frame_range = local_frame_range
        # Cached masks in a hashmap
        self.mask_cache = {}

    def create_attention_mask(self, B, TF, F, device):
        """Generates and caches attention masks."""
        # Generate a unique key for the mask based on its parameters
        cache_key = (B, TF, F, device)

        # Check if the mask is already cached
        if cache_key in self.mask_cache:
            return self.mask_cache[cache_key]

        # Define mask functions
        def is_lookahead(batch, head, q_idx, kv_idx):
            return head >= self.num_lookback_heads

        def is_lookback(batch, head, q_idx, kv_idx):
            return head < self.num_lookback_heads

        def lookback_cls_key_mask(batch, head, q_idx, kv_idx):
            return kv_idx >= TF - F - 1

        def lookback_cls_query_mask(batch, head, q_idx, kv_idx):
            return q_idx >= TF - F - 1

        lookback_cls_mask = or_masks(lookback_cls_key_mask, lookback_cls_query_mask)

        def lookahead_cls_key_mask(batch, head, q_idx, kv_idx):
            return kv_idx < F

        def lookahead_cls_query_mask(batch, head, q_idx, kv_idx):
            return q_idx < F

        lookahead_cls_mask = or_masks(lookahead_cls_key_mask, lookahead_cls_query_mask)

        def lookback_mask(batch, head, q_idx, kv_idx):
            q_frame, q_patch = divmod(q_idx, F)
            k_frame, k_patch = divmod(kv_idx, F)
            return q_frame >= k_frame >= q_frame - self.local_frame_range

        def lookahead_mask(batch, head, q_idx, kv_idx):
            q_frame, q_patch = divmod(q_idx, F)
            k_frame, k_patch = divmod(kv_idx, F)
            return q_frame < k_frame <= q_frame + self.local_frame_range

        # Build mask function
        mask_func = or_masks(and_masks(is_lookahead, or_masks(lookahead_mask, lookahead_cls_mask)),
                             and_masks(is_lookback, or_masks(lookback_mask, lookback_cls_mask)))

        # Create the mask using FlexAttention's create_block_mask
        block_mask = create_block_mask(
            mask_func, B=B, H=self.num_attention_heads, Q_LEN=TF, KV_LEN=TF, device=device
        )

        # Cache the mask
        self.mask_cache[cache_key] = block_mask

        return block_mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, F, D = x.shape
        assert D == self.embed_dim, "Input embedding dimension mismatch."

        # Flatten frames into a combined temporal-spatial sequence
        x = x.reshape(B, T * F, D)
        TF = T * F  # Combined temporal-spatial sequence length

        # Generate masks for lookback and lookahead
        attention_mask = self.create_attention_mask(B, TF, F, x.device)

        # Unified QKV projections
        qkv = self.qkv_proj(x).chunk(3, dim=-1)
        q, k, v = qkv  # Shape: (B, TF, D)

        # Split Q, K, V into lookback and lookahead heads
        q = q.view(B, TF, self.num_attention_heads, self.head_dim).transpose(1,
                                                                             2)  # (B, H, TF, head_dim)
        k = k.view(B, TF, self.num_attention_heads, self.head_dim).transpose(1,
                                                                             2)  # (B, H, TF, head_dim)
        v = v.view(B, TF, self.num_attention_heads, self.head_dim).transpose(1,
                                                                             2)  # (B, H, TF, head_dim)

        # Compute lookback and lookahead attention
        attn_output = flex_attention(q, k, v, block_mask=attention_mask)

        # Combine outputs
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, TF, D)  # (B, TF, D)

        # Output projection
        output = self.out_proj(attn_output)  # (B, TF, D)

        return output.reshape(B, T, F, D)
