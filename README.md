# FastBack

FastBack is an advanced framework for video understanding and generation, designed around a novel transformer architecture optimized for the simultaneous modeling of temporal and spatial dimensions. 
## Key Features

### 1. **Bidirectional Temporal Attention**
FastBack leverages a `BidirectionalTemporalAttention` module that divides attention heads into two categories: lookback (past frames) and lookahead (future frames). This design provides precise temporal modeling capabilities, enabling:

- Robust encoding of long-range temporal dependencies.
- Computational efficiency through the use of block-based masking strategies.

### 2. **CLS Frames for Context Aggregation**
The architecture employs two CLS (classification) frames: one prepended to the start and one appended to the end of the sequence. These frames facilitate the capture of global contextual information. Core components include:

- **AddCLSFrames**: Augments the input sequence by adding the CLS frames.
- **CLSPooling**: Aggregates embeddings from both CLS frames to provide a comprehensive sequence-level representation.

### 3. **Transformer Block Architecture**
Each transformer block integrates essential components for stable and efficient training:

- **Layer Normalization** to mitigate gradient instabilities.
- **Feedforward Neural Network (FFN)** with GELU activation to enhance non-linearity.
- Residual connections and dropout to promote robust optimization and generalization.

### 4. **Modular and Scalable Design**
The modular architecture, with components such as `AddCLSFrames`, `BidirectionalTemporalAttention`, and `TransformerBlock`, enables scalability and adaptability for diverse video tasks, including:

- Action recognition.
- Video captioning.
- Video generation.

## Installation

To deploy FastBack, follow these steps:

```bash
# Clone the repository
git clone https://github.com/your-repo/fastback.git
cd fastback

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Example Code
Below is a concise example demonstrating how FastBack can be employed in a video processing pipeline:

```python
import torch
from fastback import BTTransformer, AddCLSFrames, CLSPooling

# Define input tensor with dimensions: (batch_size, num_frames, num_patches, embed_dim)
batch_size, num_frames, num_patches, embed_dim = 8, 16, 64, 128
x = torch.randn(batch_size, num_frames, num_patches, embed_dim)

# Incorporate CLS Frames
add_cls = AddCLSFrames(embed_dim=embed_dim, frame_size=num_patches)
x_with_cls = add_cls(x)

# Apply Transformer Block
transformer_block = BTTransformerBlock(
    embed_dim=embed_dim,
    num_heads=8,
    lookback_heads=4,
    lookahead_heads=4,
    frame_size=num_patches,
    ff_hidden_dim=512,
    block_size=128,
    dropout=0.1
)
out = transformer_block(x_with_cls)

# Perform CLS Pooling
cls_pool = CLSPooling(embed_dim=embed_dim)
cls_token = cls_pool(out)

print("CLS Token Shape:", cls_token.shape)
```

## Scientific Design and Architecture

FastBack’s design integrates:

1. **Preprocessing Module**
   - `AddCLSFrames`: Introduces CLS tokens to enhance global representation learning.

2. **Temporal-Spatial Attention Mechanism**
   - `BidirectionalTemporalAttention`: Implements sophisticated masking to manage temporal dynamics, enabling selective attention across past and future frames.

3. **Transformer Blocks**
   - Combines advanced normalization, attention, and feedforward techniques with residual connections for efficient backpropagation and training stability.

4. **Global Representation Aggregation**
   - `CLSPooling`: Aggregates context across sequences using the two CLS frames for high-level representation suitable for downstream tasks.

## Applications

FastBack is applicable across diverse domains, including:

- **Video Understanding**: Supporting tasks like action recognition, video captioning, and video question answering (QA).
- **Video Generation**: Facilitating the synthesis of coherent video sequences from latent representations.
- **Multimodal Integration**: Enabling cross-modal tasks involving video, text, and audio for enhanced capabilities.
