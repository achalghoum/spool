import torch
import torch.nn as nn
from torchvision.transforms import Resize, Normalize, Compose, ToTensor
from transformers import AutoModel, AutoImageProcessor


class BackboneFrameEncoder(nn.Module):
    def __init__(self, backbone_name, image_size=224, device:str="cuda"):
        """
        Initializes the BackboneFrameEncoder with a Hugging Face backbone.

        Args:
            backbone_name (str): Name of the Hugging Face model to load as the backbone.
            image_size (int): Size to which frames are resized for encoding.
            device (str): Device to use for computation ('cuda' or 'cpu').
        """
        super(BackboneFrameEncoder, self).__init__()
        self.device = device
        self.image_size = image_size
        # Load Hugging Face model and feature extractor
        self.backbone = AutoModel.from_pretrained(backbone_name).to(self.device)
        self.feature_extractor = AutoImageProcessor.from_pretrained(backbone_name)
        self.embed_dim = self.backbone.config.hidden_size  # Token embedding dimension

        # Preprocessing for video frames
        self.preprocess = Compose([
            Resize((image_size, image_size)),
            ToTensor(),  # Converts to [C, H, W]
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Standard ImageNet normalization
        ])

    def preprocess_frames(self, video_frames):
        """
        Preprocess video frames for the backbone model.

        Args:
            video_frames (torch.Tensor): Video frames of shape (B, F, H, W, C).

        Returns:
            torch.Tensor: Preprocessed frames of shape (B * F, C, H', W').
        """
        B, F, H, W, C = video_frames.shape
        video_frames = video_frames.view(B * F, H, W, C)  # Flatten batch and frames
        video_frames = video_frames.permute(0, 3, 1, 2)  # Convert to (B * F, C, H, W)
        preprocessed = torch.stack([self.preprocess(frame) for frame in video_frames])  # Apply transforms
        return preprocessed.to(self.device)

    def forward(self, video_frames):
        """
        Encodes video frames to tokens using the backbone.

        Args:
            video_frames (torch.Tensor): Video tensor of shape (B, F, H, W, C).

        Returns:
            torch.Tensor: Token embeddings of shape (B, F, L, D),
                          where L is the number of tokens and D is the embedding dimension.
        """
        B, F, H, W, C = video_frames.shape
        preprocessed_frames = self.preprocess_frames(video_frames)

        # Extract features with the backbone
        with torch.no_grad():
            outputs = self.backbone(preprocessed_frames)

        # Extract token embeddings
        if hasattr(outputs, "last_hidden_state"):
            frame_embeddings = outputs.last_hidden_state  # Shape: (B * F, L, D)
        else:
            frame_embeddings = outputs[0]

        # Reshape to (B, F, L, D)
        frame_embeddings = frame_embeddings.view(B, F, -1, self.embed_dim)

        return frame_embeddings
