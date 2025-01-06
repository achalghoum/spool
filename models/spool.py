from torch import nn
import torch
from layers.backbone import BackboneFrameEncoder
from layers.context import AddCLSFrames, CLSPooling
from layers.transformer import BTTransformer


class Spool(nn.Module):
    """
    Spool model for video understanding.

    This model is designed to process video data by first extracting features from each frame using a backbone model, then adding context frames for global information,
    and finally applying a series of transformer layers to model temporal relationships.
    The final output includes a comprehensive sequence-level representation derived from the context frames.

    Args:
        num_hidden_layers (int, optional): Number of hidden layers in the transformer. Defaults to 12.
        num_attention_heads (int, optional): Total number of attention heads. Defaults to 12.
        num_lookahead_heads (int, optional): Number of lookahead attention heads. Defaults to 6.
        num_lookback_heads (int, optional): Number of lookback attention heads. Defaults to 6.
        image_size (int, optional): Size of the input images. Defaults to 244.
        backbone_name (str, optional): Name of the backbone model. Defaults to "facebook/dinov2-base".
    """

    def __init__(self,
                 num_hidden_layers: int = 12,  # Number of hidden layers in the transformer
                 num_attention_heads:int =12,  # Total number of attention heads
                 num_lookahead_heads: int = 6,  # Number of lookahead attention heads
                 num_lookback_heads: int = 6,  # Number of lookback attention heads
                 image_size: int = 244,  # Size of the input images
                 local_frame_range: int = 16,
                 backbone_name: str = "facebook/dinov2-base"):  # Name of the backbone model
        super().__init__()
        self.backbone = BackboneFrameEncoder(
            backbone_name=backbone_name,  # Initialize the backbone with the specified name
            image_size=image_size  # Set the image size for the backbone
        )
        self.context_frames = AddCLSFrames(
            embed_dim=self.backbone.embed_dim  # Use the backbone's embedding dimension
        )
        # Initialize a list of transformer modules with the specified configuration
        self.transformers = nn.ModuleList([
            BTTransformer(embed_dim=self.backbone.embed_dim,
                          num_attention_heads=num_attention_heads,
                          num_lookahead_heads=num_lookahead_heads,
                          local_frame_range = local_frame_range,
                          num_lookback_heads=num_lookback_heads)
            for i in range(num_hidden_layers)
        ])
        # Define the sequential processing pipeline
        self.sequential_process = nn.Sequential(
            self.backbone,  # Backbone for feature extraction
            self.context_frames,  # Add CLS frames for context
            *self.transformers  # Apply transformer layers
        )
        self.cls_pooling = CLSPooling(
            embed_dim=self.backbone.embed_dim)  # For final CLS token

    def forward(self, x: torch.Tensor):
        """
        Forward pass through the Spool model.

        Args:
            x (torch.Tensor): Input tensor to the model.

        Returns:
            tuple: A tuple containing the output tensor and the final CLS token.
        """
        x = self.sequential_process(
            x)  # Process the input through the sequential pipeline
        cls_token = self.cls_pooling(x)  # Derive the final CLS token
        return x, cls_token  # Return the output and the CLS token
