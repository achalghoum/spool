import torch
import torch.nn as nn

class AddCLSFrames(nn.Module):
    def __init__(self, embed_dim):
        """
        Module to prepend and append CLS frames to the input sequence.

        Args:
            embed_dim (int): Dimension of the embeddings.
            frame_size (int): Number of spatial patches per frame.
        """
        super().__init__()
        self.embed_dim = embed_dim

        # CLS token parameters for start and end frames
        self.cls_start = nn.Parameter(torch.randn(1, 1, embed_dim))  # (1, 1, embed_dim)
        self.cls_end = nn.Parameter(torch.randn(1, 1, embed_dim))    # (1, 1, embed_dim)

    def forward(self, x):
        """
        Adds CLS frames to the input sequence.

        Args:
            x (Tensor): Input tensor of shape (B, T, F, D), where:
                B: Batch size
                T: Number of temporal frames.
                F: Number of spatial patches per frame.
                D: Embedding dimension.

        Returns:
            Tensor: Output tensor with CLS frames added, of shape (B, T+2, F, D).
        """
        B, T, F, D = x.shape
        assert D == self.embed_dim, "Input embedding dimension must match initialized embed_dim."

        # Expand CLS tokens to create CLS frames
        cls_start_frame = self.cls_start.expand(B, F, D)  # (B, F, D)
        cls_end_frame = self.cls_end.expand(B, F, D)      # (B, F, D)

        # Add CLS frames at the beginning and end
        cls_start_frame = cls_start_frame.unsqueeze(1)  # (B, 1, F, D)
        cls_end_frame = cls_end_frame.unsqueeze(1)      # (B, 1, F, D)

        # Concatenate CLS frames to the input sequence
        x_with_cls = torch.cat([cls_start_frame, x, cls_end_frame], dim=1)  # (B, T+2, F, D)

        return x_with_cls

class CLSPooling(nn.Module):
    def __init__(self, embed_dim):
        """
        Module to derive the final CLS token by averaging the CLS frames.

        Args:
            embed_dim (int): Dimension of the embeddings.
            frame_size (int): Number of spatial patches per frame.
        """
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, x):
        """
        Derives the CLS token from CLS frames.

        Args:
            x (Tensor): Input tensor with CLS frames, of shape (B, T+2, F, D), where:
                B: Batch size
                T+2: Sequence length (including CLS frames).
                F: Number of spatial patches per frame.
                D: Embedding dimension.

        Returns:
            Tensor: Final CLS token of shape (B, 2 * D).
        """
        B, T_with_cls, F, D = x.shape
        assert T_with_cls >= 2, "Input must include at least 2 CLS frames."
        assert D == self.embed_dim, "Input embedding dimension must match initialized embed_dim."

        # Extract CLS start and CLS end frames
        cls_start_frame = x[:, 0, :, :]  # (B, F, D)
        cls_end_frame = x[:, -1, :, :]   # (B, F, D)

        # Global average pooling over the spatial dimension
        cls_start_avg = cls_start_frame.mean(dim=1)  # (B, D)
        cls_end_avg = cls_end_frame.mean(dim=1)      # (B, D)

        # Concatenate both averages to form the final CLS token
        cls_token = torch.cat([cls_start_avg, cls_end_avg], dim=-1)  # (B, 2 * D)

        return cls_token
