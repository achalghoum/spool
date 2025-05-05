# Spool: A Bidirectional Temporal Transformer Architecture for Video Understanding

Spool presents a neural architecture that introduces bidirectional temporal attention for video understanding. The model combines local temporal relationships with global contextual information, enabling comprehensive video sequence analysis.

## Technical Overview

The architecture consists of three primary components that work in concert to process video sequences:

1. **Dense Visual Feature Extraction**
   - Leverages foundation vision models for frame-level representation
   - Default backbone: DINOv2, chosen for its strong performance in different vision tasks and its stable learned features

2. **Temporal Context Framing**
   - Introduces specialized context frames at sequence boundaries
   - Enables bidirectional information flow through the temporal dimension
   - Facilitates global-local feature interaction through learned embeddings

3. **Bidirectional Temporal Attention Mechanism**
   - Implements separate lookback and lookahead attention paths
   - Supports asymmetric attention allocation for temporal modeling
   - Local frame range constraints to focus on relevant temporal windows
   - Global attention attention through context frames to complement local attention at frame level


## Implementation

```python
model = Spool(
    num_hidden_layers=12,      # Depth of temporal processing
    num_attention_heads=12,    # Total attention heads
    num_lookahead_heads=6,     # Future context heads
    num_lookback_heads=6,      # Past context heads
    image_size=224,           # Spatial dimension
    backbone_name="facebook/dinov2-base"
)

# Input shape: [batch_size, temporal_frames, height, width, channels]
# Output shapes: 
# - features: [batch_size, temporal_frames + 2, spatial_tokens, embedding_dim]
# - context: [batch_size, 2 * embedding_dim]
```

## Model Parameters

The architecture can be configured through the following parameters:

- `num_hidden_layers`: Controls the depth of temporal processing
- `num_attention_heads`: Total number of attention paths
- `num_lookahead_heads`: Determines future context capacity
- `num_lookback_heads`: Determines past context capacity
- `image_size`: Spatial dimension of input frames tp resize to
- `local_frame_range`: Size of frame neighborhood in local attention computation
- `backbone_name`: Selection of visual feature extractor

## Technical Requirements

- Python 3.11+
- PyTorch 2.5+
- CUDA-compatible GPU

## Project Structure

```
spool/
   ├── layers/
   │   ├── attention.py      # Bidirectional attention implementation
   │   ├── backbone.py       # Visual feature extraction
   │   ├── context.py        # Context frame mechanisms
   │   └── transformer.py    # Feature processing blocks
   └── models/
       └── spool.py       # Architecture definition
```
