# FastBack

FastBack is a framework for video understanding and generation. It uses transformers optimized for temporal and spatial modeling. The design includes bidirectional temporal attention, CLS token aggregation, and Hugging Face backbones, making it suitable for tasks like action recognition, video captioning, and video generation.

## Key Features

### 1. **Bidirectional Temporal Attention**
FastBack uses the `BidirectionalTemporalAttention` module with:
- **Lookback Heads**: Focused on past frames.
- **Lookahead Heads**: Focused on future frames.

This approach models long-range dependencies and uses block-based masking for efficiency.

### 2. **CLS Frames for Context Aggregation**
Special start and end CLS frames capture global context:
- **AddCLSFrames**: Adds start and end CLS frames.
- **CLSPooling**: Aggregates embeddings from CLS frames for a sequence-level representation.

### 3. **Transformer Design**
Multiple `BTTransformer` blocks combine attention, normalization, and feedforward layers for temporal-spatial modeling.

### 4. **Backbone Encoder**
The backbone uses Hugging Face’s `AutoModel` for extracting features from video frames:
- Pre-trained models for high performance.
- Flexible support for different architectures.
- Efficient preprocessing with resizing, normalization, and tensor conversion.

### 5. **Modular and Scalable Design**
FastBack components can be customized for specific tasks and scaled for various hardware and datasets.

## Installation

To install FastBack:

```bash
# Clone the repository
git clone https://github.com/achalghoum/fastback.git
cd fastback

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Example Code
The following demonstrates FastBack in a video processing pipeline:

```python
import torch
from fastback import FastBack

# Define input tensor (batch_size, num_frames, height, width, channels)
batch_size, num_frames, height, width, channels = 8, 16, 224, 224, 3
x = torch.randn(batch_size, num_frames, height, width, channels)

# Initialize FastBack model
model = FastBack(
    num_hidden_layers=12,
    num_attention_heads=12,
    num_lookahead_heads=6,
    num_lookback_heads=6,
    image_size=224,
    backbone_name="facebook/dinov2-base"
)

# Forward pass
outputs, cls_token = model(x)
print("Outputs Shape:", outputs.shape)
print("CLS Token Shape:", cls_token.shape)
```

## Design and Architecture

### 1. **Preprocessing**
- `BackboneFrameEncoder`: Encodes video frames using a pre-trained Hugging Face model.
- Performs resizing, normalization, and tensor conversion.

### 2. **Context Augmentation**
- `AddCLSFrames`: Adds start and end CLS tokens for global context learning.

### 3. **Temporal-Spatial Attention**
- `BidirectionalTemporalAttention`: Focuses on past and future frames with spatial awareness via masking.

### 4. **Transformer Layers**
- `BTTransformer`: Combines normalization, attention, feedforward layers, and residual connections for robust modeling.

### 5. **Global Representation**
- `CLSPooling`: Aggregates CLS frames to produce a sequence-level representation.

## Applications
FastBack supports various video tasks:

- **Action Recognition**: Identifying activities in videos.
- **Video Captioning**: Generating descriptions for video content.
- **Video Generation**: Synthesizing video sequences.
- **Multimodal Integration**: Combining video, text, and audio for tasks like question answering.

## Future Directions
1. **Cross-Modal Integration**: Adding support for audio and text.
2. **Pre-Trained Models**: Offering task-specific pre-trained versions.
3. **Efficiency Improvements**: Using optimized attention mechanisms like Linformer.
4. **Community Engagement**: Adding tutorials and examples.

FastBack is designed for versatility in video tasks, using advanced transformers for performance and scalability.

