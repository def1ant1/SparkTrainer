# Architecture Builder Guide

## Overview

The Architecture Builder is a visual, Figma-like interface for designing neural network architectures. It supports all major AI model types including Language Models, Vision Models, Audio Models, and Multimodal Models.

## Features

### 🎨 Figma-like Canvas
- **Zoom**: Scroll to zoom in/out (10% - 500%)
- **Pan**: Space + Drag or Middle-click + Drag to pan
- **Keyboard Shortcuts**:
  - `Ctrl/Cmd + 0`: Reset view
  - `Ctrl/Cmd + =`: Zoom in
  - `Ctrl/Cmd + -`: Zoom out
  - `Space + Drag`: Pan canvas

### 📐 60+ Layer Types

Organized by category:

#### Input Layers
- `Input1D`: 1D sequences (text, time series)
- `Input2D`: 2D images
- `Input3D`: 3D videos
- `InputAudio`: Audio spectrograms

#### Core Layers
- `Embedding`: Token embeddings
- `Linear`: Fully connected layers
- `MLP`: Multi-layer perceptron blocks

#### Attention Mechanisms
- `Attention`: Standard multi-head attention
- `CrossAttention`: Cross-attention for encoder-decoder
- `FlashAttention`: Memory-efficient attention
- `GroupedQueryAttn`: GQA for efficient inference (LLaMA/Mistral style)

#### Normalization
- `LayerNorm`: Standard layer normalization
- `RMSNorm`: Root mean square normalization (LLaMA style)
- `BatchNorm2d`: Batch normalization for 2D
- `GroupNorm`: Group normalization

#### Convolutional Layers
- `Conv1d`, `Conv2d`, `Conv3d`: Standard convolutions
- `DepthwiseConv2d`: Depthwise separable convolutions
- `ConvTranspose2d`: Transposed convolutions for upsampling

#### Pooling & Upsampling
- `Pool2d`, `AvgPool2d`: Max and average pooling
- `AdaptiveAvgPool2d`: Adaptive pooling
- `GlobalAvgPool`: Global average pooling
- `Upsample2d`: Upsampling
- `PixelShuffle`: Sub-pixel upsampling

#### Blocks
- `ResBlock1d`, `ResBlock2d`: Residual blocks
- `TransformerBlock`: Complete transformer block
- `SEBlock`: Squeeze-and-Excitation blocks

#### Activation Functions
- `GELU`, `ReLU`, `SiLU`, `Sigmoid`, `Tanh`, `Softmax`

#### Positional Encodings
- `RotaryEmbedding`: Rotary position embeddings (RoPE)
- `LearnedPosEmbed`: Learned positional embeddings
- `SinusoidalPosEmbed`: Sinusoidal position embeddings

#### Recurrent
- `LSTM`, `GRU`: Recurrent neural networks

#### Operations
- `Add`: Residual connections
- `Concat`: Concatenation
- `Flatten`, `Reshape`: Shape manipulation

#### Adapters & Quantization
- `LoRA`: Low-rank adaptation layers
- `Quantize`: Quantization nodes

### 🎯 Pre-built Templates

10 production-ready architectures:

1. **GPT-2 Style**: Autoregressive language model
2. **LLaMA / Mistral Style**: Modern LLM with GQA and RMSNorm
3. **Vision Transformer (ViT)**: Image classification with transformers
4. **ResNet-50**: Classic CNN architecture
5. **UNet (Diffusion)**: U-Net for diffusion models
6. **CLIP (Vision + Text)**: Multimodal vision-language model
7. **Whisper (Audio)**: Speech-to-text encoder-decoder
8. **GAN (DCGAN)**: Deep convolutional GAN
9. **MoE (Mixture of Experts)**: Sparse mixture of experts
10. **BERT Encoder**: Bidirectional transformer encoder

### ⚙️ Advanced Model Features

#### Quantization
Enable to reduce model size and improve inference speed:
- **4-bit**: NF4, FP4
- **8-bit**: INT8, INT4

#### LoRA (Low-Rank Adaptation)
Parameter-efficient fine-tuning:
- **Rank**: Controls adapter size (default: 8)
- **Alpha**: Scaling factor (default: 16)
- **Dropout**: Regularization (default: 0.05)

#### Context Extension
Extend sequence length beyond training:
- **RoPE Scaling**: Rotary position embedding scaling
- **ALiBi**: Attention with Linear Biases
- **YaRN**: Yet another RoPE extension
- **Max Length**: Target sequence length

#### Optimization
- **Gradient Checkpointing**: Trade compute for memory
- **Flash Attention**: 2-4x faster attention with less memory

## Workflow

### 1. Start from a Template
Click any template to load a complete architecture. The canvas will automatically zoom out to fit the entire graph.

### 2. Add Layers
Drag layers from the palette onto the canvas. Layers are organized by category for easy access.

### 3. Connect Layers
1. Click "Connect Mode" or click the `•→` button on a node
2. Click the target node to create an edge
3. Edges are drawn as curved lines with arrows

### 4. Edit Parameters
Select a node to edit its parameters in the right sidebar:
- All layer-specific parameters are editable
- Changes update automatically
- Type inference handles numbers vs strings

### 5. Configure Model Extensions
Enable advanced features in the left sidebar:
- Toggle quantization and select precision
- Enable LoRA with custom rank/alpha
- Configure context extension
- Enable optimizations

### 6. Export
Click "Export PyTorch" to download generated code with your configurations.

## Examples

### Creating a Custom Language Model

1. **Load GPT-2 template** as starting point
2. **Adjust parameters**:
   - Increase hidden size to 2048
   - Add more transformer blocks
   - Adjust vocabulary size
3. **Enable LoRA** for efficient fine-tuning
4. **Enable Flash Attention** for faster training
5. **Export** to get PyTorch code

### Building a Vision-Language Model

1. **Load CLIP template**
2. **Customize vision tower**:
   - Replace conv stem with ViT patches
   - Adjust number of layers
3. **Customize text tower**:
   - Modify vocabulary size
   - Adjust hidden dimensions
4. **Add fusion layers**:
   - Add CrossAttention between towers
   - Add projection layers
5. **Export** to get complete model code

### Creating a Diffusion Model

1. **Load UNet template**
2. **Add time embeddings**:
   - Insert SinusoidalPosEmbed nodes
   - Connect to each block
3. **Add attention**:
   - Insert Attention layers at bottleneck
   - Add cross-attention for conditioning
4. **Configure**:
   - Enable gradient checkpointing
   - Set precision to FP16
5. **Export** to get model code

## Tips & Best Practices

### Canvas Navigation
- **Always use Space+Drag** for panning to avoid accidental node moves
- **Zoom out first** when loading templates to see the full architecture
- **Use keyboard shortcuts** for quick zoom adjustments

### Building Models
- **Start with templates** - They provide proven architectures
- **Group similar layers** - Keep the graph organized
- **Name layers consistently** - Use the parameter editor
- **Test incrementally** - Export and test small changes

### Performance
- **Use Flash Attention** for transformers when possible
- **Enable Gradient Checkpointing** for large models
- **Consider quantization** for deployment
- **Use LoRA** instead of full fine-tuning when applicable

### Common Patterns

#### Residual Connection
```
Input → LayerNorm → Attention → Add ←┐
  └─────────────────────────────────┘
```

#### Bottleneck Block
```
Input → Conv1x1(reduce) → Conv3x3 → Conv1x1(expand) → Add ←┐
  └────────────────────────────────────────────────────────┘
```

#### Encoder-Decoder Attention
```
Encoder Output → CrossAttention(kv) ←─ Decoder(q)
```

## Model Statistics

The right sidebar shows real-time statistics:
- **Parameters**: Total trainable parameters (in millions)
- **FLOPs**: Computational complexity (in GFLOPs)
- **Memory**: Activation memory estimate (in MiB)
- **Zoom**: Current zoom level

## Validation

The builder performs automatic validation:
- ✅ Shape compatibility between connected layers
- ✅ Parameter constraints (e.g., in_features must match)
- ✅ Dimension mismatches
- ⚠️ Errors shown in red in the stats panel

## Export Formats

Currently supports:
- **PyTorch**: Full nn.Module code with your architecture
- Includes import statements for enabled features (LoRA, Flash Attention)
- Contains configuration comments for quantization

## Keyboard Shortcuts Reference

| Shortcut | Action |
|----------|--------|
| `Space + Drag` | Pan canvas |
| `Scroll` | Zoom in/out |
| `Ctrl/Cmd + 0` | Reset view to 100% |
| `Ctrl/Cmd + =` | Zoom in |
| `Ctrl/Cmd + -` | Zoom out |
| `Middle Click + Drag` | Pan canvas |
| `Click node` | Select node |
| `Drag node` | Move node |
| `Drag from palette` | Add new node |

## Supported Model Types

### Language Models
- ✅ GPT (autoregressive)
- ✅ BERT (bidirectional)
- ✅ LLaMA/Mistral (modern LLMs)
- ✅ T5 (encoder-decoder)

### Vision Models
- ✅ ResNet, VGG, DenseNet (CNNs)
- ✅ Vision Transformer (ViT)
- ✅ U-Net (segmentation, diffusion)
- ✅ DETR (object detection)

### Audio Models
- ✅ Whisper (speech-to-text)
- ✅ Wav2Vec (self-supervised)
- ✅ Conformer (speech recognition)

### Multimodal Models
- ✅ CLIP (vision + text)
- ✅ Flamingo (vision + language)
- ✅ PerceiveIO (unified architecture)

### Generative Models
- ✅ GAN (DCGAN, StyleGAN)
- ✅ VAE (variational autoencoders)
- ✅ Diffusion (UNet-based)

### Sparse Models
- ✅ Mixture of Experts (MoE)
- ✅ Switch Transformers

## Advanced Topics

### Custom Layer Types
To add your own layer types, edit the `LAYERS` object in `ArchitectureBuilder.jsx`:

```javascript
const LAYERS = {
  YourLayer: {
    kind: 'op',
    shapeType: '1d',
    params: { your_param: 256 },
    category: 'Custom'
  },
  // ...
};
```

### Custom Templates
Add templates to the `TEMPLATES` object:

```javascript
const TEMPLATES = {
  'Your Model': () => {
    const nodes = [];
    const edges = [];
    let nid = 1;
    const add = (type, x, y, params = {}) => { /* ... */ };
    const conn = (a, b) => { /* ... */ };

    // Build your architecture
    const n0 = add('Input1D', 100, 100);
    const n1 = add('Linear', 300, 100);
    conn(n0, n1);

    return { nodes, edges };
  },
};
```

## Troubleshooting

### Canvas not responding
- Refresh the page
- Check console for errors
- Try resetting the view (Ctrl/Cmd + 0)

### Nodes overlapping
- Zoom out to see full graph
- Manually drag nodes to organize
- Consider starting from a clean template

### Validation errors
- Check error messages in stats panel
- Verify parameter dimensions match
- Ensure all required parameters are set

### Export not working
- Check browser console for errors
- Ensure nodes and edges are valid
- Try exporting a template first to verify

## Future Enhancements

Planned features:
- Real-time code preview
- Model deployment integration
- Automatic hyperparameter suggestions
- Performance profiling
- Multi-framework export (TensorFlow, JAX)
- Collaborative editing
- Version control for architectures

## Support

For issues or questions:
- Check validation errors in the stats panel
- Review this guide for common patterns
- Test with templates first
- Export frequently to save progress
