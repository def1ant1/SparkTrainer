import React, { useEffect, useMemo, useRef, useState, useCallback } from 'react';

// Extended layer catalog with more model types
const LAYERS = {
  // Inputs
  Input1D: { kind: 'input', shapeType: '1d', params: { seq_len: 128, hidden: 256 }, category: 'Input' },
  Input2D: { kind: 'input', shapeType: '2d', params: { channels: 3, height: 224, width: 224 }, category: 'Input' },
  Input3D: { kind: 'input', shapeType: '3d', params: { channels: 3, depth: 16, height: 112, width: 112 }, category: 'Input' },
  InputAudio: { kind: 'input', shapeType: '1d', params: { n_mels: 80, seq_len: 3000 }, category: 'Input' },

  // Core layers
  Embedding: { kind: 'op', shapeType: '1d', params: { vocab_size: 50257, embed_dim: 256 }, category: 'Core' },
  Linear: { kind: 'op', shapeType: '1d', params: { in_features: 256, out_features: 256, bias: true }, category: 'Core' },
  MLP: { kind: 'block', shapeType: '1d', params: { hidden: 256, mlp_ratio: 4, bias: true }, category: 'Core' },

  // Attention
  Attention: { kind: 'op', shapeType: '1d', params: { hidden: 256, num_heads: 8, seq_len: 128 }, category: 'Attention' },
  CrossAttention: { kind: 'op', shapeType: '1d', params: { hidden: 256, num_heads: 8, kv_hidden: 256 }, category: 'Attention' },
  FlashAttention: { kind: 'op', shapeType: '1d', params: { hidden: 256, num_heads: 8, seq_len: 128 }, category: 'Attention' },
  GroupedQueryAttn: { kind: 'op', shapeType: '1d', params: { hidden: 256, num_heads: 8, num_kv_heads: 2 }, category: 'Attention' },

  // Normalization
  LayerNorm: { kind: 'op', shapeType: '1d', params: { hidden: 256, eps: 1e-5 }, category: 'Norm' },
  RMSNorm: { kind: 'op', shapeType: '1d', params: { hidden: 256, eps: 1e-6 }, category: 'Norm' },
  BatchNorm2d: { kind: 'op', shapeType: '2d', params: { num_features: 64, eps: 1e-5, momentum: 0.1 }, category: 'Norm' },
  GroupNorm: { kind: 'op', shapeType: 'any', params: { num_groups: 32, num_channels: 64, eps: 1e-5 }, category: 'Norm' },

  // Regularization
  Dropout: { kind: 'op', shapeType: '1d', params: { p: 0.1 }, category: 'Regularization' },
  DropPath: { kind: 'op', shapeType: 'any', params: { drop_prob: 0.1 }, category: 'Regularization' },

  // Conv layers
  Conv1d: { kind: 'op', shapeType: '1d', params: { in_ch: 256, out_ch: 256, k: 3, stride: 1, pad: 1, bias: true }, category: 'Conv' },
  Conv2d: { kind: 'op', shapeType: '2d', params: { in_ch: 3, out_ch: 64, k: 3, stride: 1, pad: 1, bias: true }, category: 'Conv' },
  Conv3d: { kind: 'op', shapeType: '3d', params: { in_ch: 3, out_ch: 64, k: 3, stride: 1, pad: 1, bias: true }, category: 'Conv' },
  DepthwiseConv2d: { kind: 'op', shapeType: '2d', params: { channels: 64, k: 3, stride: 1, pad: 1 }, category: 'Conv' },
  ConvTranspose2d: { kind: 'op', shapeType: '2d', params: { in_ch: 64, out_ch: 64, k: 2, stride: 2, pad: 0, bias: true }, category: 'Conv' },

  // Pooling
  Pool2d: { kind: 'op', shapeType: '2d', params: { k: 2, stride: 2 }, category: 'Pooling' },
  AvgPool2d: { kind: 'op', shapeType: '2d', params: { k: 2, stride: 2 }, category: 'Pooling' },
  AdaptiveAvgPool2d: { kind: 'op', shapeType: '2d', params: { output_size: 1 }, category: 'Pooling' },
  GlobalAvgPool: { kind: 'op', shapeType: 'any', params: {}, category: 'Pooling' },

  // Upsampling
  Upsample2d: { kind: 'op', shapeType: '2d', params: { scale: 2 }, category: 'Upsample' },
  PixelShuffle: { kind: 'op', shapeType: '2d', params: { upscale_factor: 2 }, category: 'Upsample' },

  // Blocks
  ResBlock2d: { kind: 'block', shapeType: '2d', params: { ch: 64, k: 3 }, category: 'Blocks' },
  ResBlock1d: { kind: 'block', shapeType: '1d', params: { hidden: 256 }, category: 'Blocks' },
  TransformerBlock: { kind: 'block', shapeType: '1d', params: { hidden: 256, num_heads: 8, mlp_ratio: 4 }, category: 'Blocks' },
  SEBlock: { kind: 'block', shapeType: '2d', params: { channels: 64, reduction: 16 }, category: 'Blocks' },

  // Activation
  GELU: { kind: 'op', shapeType: 'any', params: {}, category: 'Activation' },
  ReLU: { kind: 'op', shapeType: 'any', params: {}, category: 'Activation' },
  SiLU: { kind: 'op', shapeType: 'any', params: {}, category: 'Activation' },
  Sigmoid: { kind: 'op', shapeType: 'any', params: {}, category: 'Activation' },
  Tanh: { kind: 'op', shapeType: 'any', params: {}, category: 'Activation' },
  Softmax: { kind: 'op', shapeType: 'any', params: { dim: -1 }, category: 'Activation' },

  // Recurrent
  LSTM: { kind: 'op', shapeType: '1d', params: { input_size: 256, hidden_size: 256, num_layers: 1, bidirectional: false, batch_first: true }, category: 'Recurrent' },
  GRU: { kind: 'op', shapeType: '1d', params: { input_size: 256, hidden_size: 256, num_layers: 1, bidirectional: false, batch_first: true }, category: 'Recurrent' },

  // Operations
  Add: { kind: 'op', shapeType: 'any', params: {}, category: 'Operations' },
  Concat: { kind: 'op', shapeType: 'any', params: { axis: -1 }, category: 'Operations' },
  Flatten: { kind: 'op', shapeType: 'any', params: { start_dim: 1, end_dim: -1 }, category: 'Operations' },
  Reshape: { kind: 'op', shapeType: 'any', params: { to: '(B, -1)' }, category: 'Operations' },

  // Positional encodings
  RotaryEmbedding: { kind: 'op', shapeType: '1d', params: { hidden: 256, max_len: 2048 }, category: 'Positional' },
  LearnedPosEmbed: { kind: 'op', shapeType: '1d', params: { max_len: 512, hidden: 256 }, category: 'Positional' },
  SinusoidalPosEmbed: { kind: 'op', shapeType: '1d', params: { max_len: 512, hidden: 256 }, category: 'Positional' },

  // Special
  LoRA: { kind: 'adapter', shapeType: 'any', params: { rank: 8, alpha: 16, dropout: 0.05 }, category: 'Adapter' },
  Quantize: { kind: 'quantize', shapeType: 'any', params: { bits: 8, symmetric: true }, category: 'Quantization' },
};

// Extensive templates covering all major model architectures
const TEMPLATES = {
  'GPT-2 Style': () => {
    const nodes = []; const edges = []; let nid = 1;
    const add = (type, x, y, params = {}) => { const id = String(nid++); nodes.push({ id, type, x, y, params: { ...LAYERS[type].params, ...params } }); return id; };
    const conn = (a, b) => edges.push({ id: `${a}-${b}`, from: a, to: b });

    const n0 = add('Input1D', 100, 100, { seq_len: 1024, hidden: 768 });
    const n1 = add('Embedding', 300, 100, { vocab_size: 50257, embed_dim: 768 });
    const n2 = add('LearnedPosEmbed', 500, 100, { max_len: 1024, hidden: 768 });
    const n3 = add('Add', 700, 100, {});
    const n4 = add('LayerNorm', 900, 100, { hidden: 768 });
    const n5 = add('Attention', 1100, 80, { hidden: 768, num_heads: 12, seq_len: 1024 });
    const n6 = add('Add', 1300, 100, {});
    const n7 = add('LayerNorm', 1500, 100, { hidden: 768 });
    const n8 = add('MLP', 1700, 100, { hidden: 768, mlp_ratio: 4 });
    const n9 = add('Add', 1900, 100, {});
    const n10 = add('Linear', 2100, 100, { in_features: 768, out_features: 50257 });

    conn(n0, n1); conn(n1, n3); conn(n2, n3); conn(n3, n4); conn(n4, n5); conn(n5, n6); conn(n4, n6);
    conn(n6, n7); conn(n7, n8); conn(n8, n9); conn(n6, n9); conn(n9, n10);
    return { nodes, edges };
  },

  'LLaMA / Mistral Style': () => {
    const nodes = []; const edges = []; let nid = 1;
    const add = (type, x, y, params = {}) => { const id = String(nid++); nodes.push({ id, type, x, y, params: { ...LAYERS[type].params, ...params } }); return id; };
    const conn = (a, b) => edges.push({ id: `${a}-${b}`, from: a, to: b });

    const n0 = add('Input1D', 100, 100, { seq_len: 4096, hidden: 4096 });
    const n1 = add('Embedding', 300, 100, { vocab_size: 32000, embed_dim: 4096 });
    const n2 = add('RMSNorm', 500, 100, { hidden: 4096 });
    const n3 = add('GroupedQueryAttn', 700, 80, { hidden: 4096, num_heads: 32, num_kv_heads: 8 });
    const n4 = add('Add', 900, 100, {});
    const n5 = add('RMSNorm', 1100, 100, { hidden: 4096 });
    const n6 = add('MLP', 1300, 100, { hidden: 4096, mlp_ratio: 2.7 });
    const n7 = add('Add', 1500, 100, {});
    const n8 = add('RMSNorm', 1700, 100, { hidden: 4096 });
    const n9 = add('Linear', 1900, 100, { in_features: 4096, out_features: 32000 });

    conn(n0, n1); conn(n1, n2); conn(n2, n3); conn(n3, n4); conn(n2, n4);
    conn(n4, n5); conn(n5, n6); conn(n6, n7); conn(n4, n7);
    conn(n7, n8); conn(n8, n9);
    return { nodes, edges };
  },

  'Vision Transformer (ViT)': () => {
    const nodes = []; const edges = []; let nid = 1;
    const add = (type, x, y, params = {}) => { const id = String(nid++); nodes.push({ id, type, x, y, params: { ...LAYERS[type].params, ...params } }); return id; };
    const conn = (a, b) => edges.push({ id: `${a}-${b}`, from: a, to: b });

    const n0 = add('Input2D', 100, 100, { channels: 3, height: 224, width: 224 });
    const n1 = add('Conv2d', 300, 100, { in_ch: 3, out_ch: 768, k: 16, stride: 16, pad: 0 });
    const n2 = add('LearnedPosEmbed', 500, 100, { max_len: 197, hidden: 768 });
    const n3 = add('Add', 700, 100, {});
    const n4 = add('TransformerBlock', 900, 100, { hidden: 768, num_heads: 12, mlp_ratio: 4 });
    const n5 = add('LayerNorm', 1100, 100, { hidden: 768 });
    const n6 = add('GlobalAvgPool', 1300, 100, {});
    const n7 = add('Linear', 1500, 100, { in_features: 768, out_features: 1000 });

    conn(n0, n1); conn(n1, n3); conn(n2, n3); conn(n3, n4); conn(n4, n5); conn(n5, n6); conn(n6, n7);
    return { nodes, edges };
  },

  'ResNet-50': () => {
    const nodes = []; const edges = []; let nid = 1;
    const add = (type, x, y, params = {}) => { const id = String(nid++); nodes.push({ id, type, x, y, params: { ...LAYERS[type].params, ...params } }); return id; };
    const conn = (a, b) => edges.push({ id: `${a}-${b}`, from: a, to: b });

    const n0 = add('Input2D', 100, 120, { channels: 3, height: 224, width: 224 });
    const n1 = add('Conv2d', 280, 120, { in_ch: 3, out_ch: 64, k: 7, stride: 2, pad: 3 });
    const n2 = add('BatchNorm2d', 460, 120, { num_features: 64 });
    const n3 = add('ReLU', 640, 120, {});
    const n4 = add('Pool2d', 820, 120, { k: 3, stride: 2 });
    const n5 = add('ResBlock2d', 1000, 120, { ch: 64, k: 3 });
    const n6 = add('ResBlock2d', 1180, 120, { ch: 128, k: 3 });
    const n7 = add('ResBlock2d', 1360, 120, { ch: 256, k: 3 });
    const n8 = add('ResBlock2d', 1540, 120, { ch: 512, k: 3 });
    const n9 = add('AdaptiveAvgPool2d', 1720, 120, { output_size: 1 });
    const n10 = add('Flatten', 1900, 120, {});
    const n11 = add('Linear', 2080, 120, { in_features: 512, out_features: 1000 });

    conn(n0, n1); conn(n1, n2); conn(n2, n3); conn(n3, n4); conn(n4, n5);
    conn(n5, n6); conn(n6, n7); conn(n7, n8); conn(n8, n9); conn(n9, n10); conn(n10, n11);
    return { nodes, edges };
  },

  'UNet (Diffusion)': () => {
    const nodes = []; const edges = []; let nid = 1;
    const add = (type, x, y, params = {}) => { const id = String(nid++); nodes.push({ id, type, x, y, params: { ...LAYERS[type].params, ...params } }); return id; };
    const conn = (a, b) => edges.push({ id: `${a}-${b}`, from: a, to: b });

    const i = add('Input2D', 100, 200, { channels: 3, height: 256, width: 256 });
    const d1 = add('Conv2d', 300, 200, { in_ch: 3, out_ch: 64, k: 3, stride: 1, pad: 1 });
    const p1 = add('Pool2d', 500, 200, { k: 2, stride: 2 });
    const d2 = add('Conv2d', 700, 200, { in_ch: 64, out_ch: 128, k: 3, stride: 1, pad: 1 });
    const p2 = add('Pool2d', 900, 200, { k: 2, stride: 2 });
    const b = add('ResBlock2d', 1100, 200, { ch: 128 });
    const u1 = add('Upsample2d', 1300, 200, { scale: 2 });
    const c1 = add('Concat', 1500, 160, { axis: 1 });
    const uconv1 = add('Conv2d', 1700, 200, { in_ch: 256, out_ch: 64, k: 3, stride: 1, pad: 1 });
    const u2 = add('Upsample2d', 1900, 200, { scale: 2 });
    const c2 = add('Concat', 2100, 160, { axis: 1 });
    const out = add('Conv2d', 2300, 200, { in_ch: 128, out_ch: 3, k: 3, stride: 1, pad: 1 });

    conn(i, d1); conn(d1, p1); conn(p1, d2); conn(d2, p2); conn(p2, b);
    conn(b, u1); conn(u1, c1); conn(d2, c1); conn(c1, uconv1);
    conn(uconv1, u2); conn(u2, c2); conn(d1, c2); conn(c2, out);
    return { nodes, edges };
  },

  'CLIP (Vision + Text)': () => {
    const nodes = []; const edges = []; let nid = 1;
    const add = (type, x, y, params = {}) => { const id = String(nid++); nodes.push({ id, type, x, y, params: { ...LAYERS[type].params, ...params } }); return id; };
    const conn = (a, b) => edges.push({ id: `${a}-${b}`, from: a, to: b });

    // Vision tower
    const v0 = add('Input2D', 100, 80, { channels: 3, height: 224, width: 224 });
    const v1 = add('Conv2d', 300, 80, { in_ch: 3, out_ch: 768, k: 16, stride: 16, pad: 0 });
    const v2 = add('TransformerBlock', 500, 80, { hidden: 768, num_heads: 12, mlp_ratio: 4 });
    const v3 = add('LayerNorm', 700, 80, { hidden: 768 });
    const v4 = add('GlobalAvgPool', 900, 80, {});

    // Text tower
    const t0 = add('Input1D', 100, 220, { seq_len: 77, hidden: 512 });
    const t1 = add('Embedding', 300, 220, { vocab_size: 49408, embed_dim: 512 });
    const t2 = add('TransformerBlock', 500, 220, { hidden: 512, num_heads: 8, mlp_ratio: 4 });
    const t3 = add('LayerNorm', 700, 220, { hidden: 512 });

    conn(v0, v1); conn(v1, v2); conn(v2, v3); conn(v3, v4);
    conn(t0, t1); conn(t1, t2); conn(t2, t3);
    return { nodes, edges };
  },

  'Whisper (Audio)': () => {
    const nodes = []; const edges = []; let nid = 1;
    const add = (type, x, y, params = {}) => { const id = String(nid++); nodes.push({ id, type, x, y, params: { ...LAYERS[type].params, ...params } }); return id; };
    const conn = (a, b) => edges.push({ id: `${a}-${b}`, from: a, to: b });

    // Encoder
    const e0 = add('InputAudio', 100, 100, { n_mels: 80, seq_len: 3000 });
    const e1 = add('Conv1d', 300, 100, { in_ch: 80, out_ch: 512, k: 3, stride: 1, pad: 1 });
    const e2 = add('GELU', 500, 100, {});
    const e3 = add('Conv1d', 700, 100, { in_ch: 512, out_ch: 512, k: 3, stride: 2, pad: 1 });
    const e4 = add('SinusoidalPosEmbed', 900, 100, { max_len: 1500, hidden: 512 });
    const e5 = add('Add', 1100, 100, {});
    const e6 = add('TransformerBlock', 1300, 100, { hidden: 512, num_heads: 8, mlp_ratio: 4 });

    // Decoder
    const d0 = add('Input1D', 100, 240, { seq_len: 448, hidden: 512 });
    const d1 = add('Embedding', 300, 240, { vocab_size: 51865, embed_dim: 512 });
    const d2 = add('LearnedPosEmbed', 500, 240, { max_len: 448, hidden: 512 });
    const d3 = add('Add', 700, 240, {});
    const d4 = add('Attention', 900, 240, { hidden: 512, num_heads: 8, seq_len: 448 });
    const d5 = add('CrossAttention', 1100, 220, { hidden: 512, num_heads: 8, kv_hidden: 512 });
    const d6 = add('MLP', 1300, 240, { hidden: 512, mlp_ratio: 4 });

    conn(e0, e1); conn(e1, e2); conn(e2, e3); conn(e3, e5); conn(e4, e5); conn(e5, e6);
    conn(d0, d1); conn(d1, d3); conn(d2, d3); conn(d3, d4); conn(d4, d5); conn(e6, d5); conn(d5, d6);
    return { nodes, edges };
  },

  'GAN (DCGAN)': () => {
    const nodes = []; const edges = []; let nid = 1;
    const add = (type, x, y, params = {}) => { const id = String(nid++); nodes.push({ id, type, x, y, params: { ...LAYERS[type].params, ...params } }); return id; };
    const conn = (a, b) => edges.push({ id: `${a}-${b}`, from: a, to: b });

    const z = add('Input1D', 100, 100, { seq_len: 1, hidden: 100 });
    const g1 = add('Linear', 300, 100, { in_features: 100, out_features: 256 * 8 * 8 });
    const g2 = add('Reshape', 500, 100, { to: '(B, 256, 8, 8)' });
    const g3 = add('ConvTranspose2d', 700, 100, { in_ch: 256, out_ch: 128, k: 4, stride: 2, pad: 1 });
    const g4 = add('BatchNorm2d', 900, 100, { num_features: 128 });
    const g5 = add('ReLU', 1100, 100, {});
    const g6 = add('ConvTranspose2d', 1300, 100, { in_ch: 128, out_ch: 64, k: 4, stride: 2, pad: 1 });
    const g7 = add('BatchNorm2d', 1500, 100, { num_features: 64 });
    const g8 = add('ReLU', 1700, 100, {});
    const g9 = add('ConvTranspose2d', 1900, 100, { in_ch: 64, out_ch: 3, k: 4, stride: 2, pad: 1 });
    const g10 = add('Tanh', 2100, 100, {});

    conn(z, g1); conn(g1, g2); conn(g2, g3); conn(g3, g4); conn(g4, g5);
    conn(g5, g6); conn(g6, g7); conn(g7, g8); conn(g8, g9); conn(g9, g10);
    return { nodes, edges };
  },

  'MoE (Mixture of Experts)': () => {
    const nodes = []; const edges = []; let nid = 1;
    const add = (type, x, y, params = {}) => { const id = String(nid++); nodes.push({ id, type, x, y, params: { ...LAYERS[type].params, ...params } }); return id; };
    const conn = (a, b) => edges.push({ id: `${a}-${b}`, from: a, to: b });

    const inp = add('Input1D', 100, 100, { seq_len: 128, hidden: 512 });
    const gate = add('Linear', 300, 40, { in_features: 512, out_features: 8 });
    const e1 = add('MLP', 300, 180, { hidden: 512, mlp_ratio: 4 });
    const e2 = add('MLP', 500, 180, { hidden: 512, mlp_ratio: 4 });
    const e3 = add('MLP', 700, 180, { hidden: 512, mlp_ratio: 4 });
    const e4 = add('MLP', 900, 180, { hidden: 512, mlp_ratio: 4 });
    const comb = add('Add', 1100, 100, {});

    conn(inp, gate); conn(inp, e1); conn(inp, e2); conn(inp, e3); conn(inp, e4);
    conn(e1, comb); conn(e2, comb); conn(e3, comb); conn(e4, comb);
    return { nodes, edges };
  },

  'BERT Encoder': () => {
    const nodes = []; const edges = []; let nid = 1;
    const add = (type, x, y, params = {}) => { const id = String(nid++); nodes.push({ id, type, x, y, params: { ...LAYERS[type].params, ...params } }); return id; };
    const conn = (a, b) => edges.push({ id: `${a}-${b}`, from: a, to: b });

    const n0 = add('Input1D', 100, 100, { seq_len: 512, hidden: 768 });
    const n1 = add('Embedding', 300, 100, { vocab_size: 30522, embed_dim: 768 });
    const n2 = add('LearnedPosEmbed', 500, 100, { max_len: 512, hidden: 768 });
    const n3 = add('Add', 700, 100, {});
    const n4 = add('LayerNorm', 900, 100, { hidden: 768 });
    const n5 = add('Attention', 1100, 80, { hidden: 768, num_heads: 12, seq_len: 512 });
    const n6 = add('Add', 1300, 100, {});
    const n7 = add('LayerNorm', 1500, 100, { hidden: 768 });
    const n8 = add('MLP', 1700, 100, { hidden: 768, mlp_ratio: 4 });
    const n9 = add('Add', 1900, 100, {});

    conn(n0, n1); conn(n1, n3); conn(n2, n3); conn(n3, n4); conn(n4, n5);
    conn(n5, n6); conn(n4, n6); conn(n6, n7); conn(n7, n8); conn(n8, n9); conn(n6, n9);
    return { nodes, edges };
  },
};

function genId() { return Math.random().toString(36).slice(2, 9); }

// Estimation functions
function estimateParams(node) {
  const p = node.params || {};
  switch (node.type) {
    case 'Embedding': return p.vocab_size * p.embed_dim;
    case 'Linear': return p.in_features * p.out_features + (p.bias ? p.out_features : 0);
    case 'MLP': return (p.hidden * (p.mlp_ratio * p.hidden)) + ((p.mlp_ratio * p.hidden) * p.hidden);
    case 'Conv2d': return p.out_ch * p.in_ch * (p.k * p.k) + (p.bias ? p.out_ch : 0);
    case 'Conv1d': return p.out_ch * p.in_ch * (p.k) + (p.bias ? p.out_ch : 0);
    case 'LayerNorm': case 'RMSNorm': return 2 * (p.hidden || 0);
    case 'BatchNorm2d': return 2 * (p.num_features || 0);
    case 'Attention': case 'FlashAttention': case 'CrossAttention': {
      const h = p.hidden || 0;
      const proj = 3 * h * h + h * h;
      return proj;
    }
    case 'GroupedQueryAttn': {
      const h = p.hidden || 0;
      const num_kv_heads = p.num_kv_heads || 1;
      const num_heads = p.num_heads || 8;
      return h * h + 2 * h * (h * num_kv_heads / num_heads) + h * h;
    }
    case 'LSTM': case 'GRU': {
      const I = p.input_size || 0, H = p.hidden_size || 0, L = p.num_layers || 1, dir = (p.bidirectional ? 2 : 1);
      const per = 4 * (I * H + H * H + H);
      return dir * L * per;
    }
    case 'LoRA': {
      const r = p.rank || 8;
      const alpha = p.alpha || 16;
      return 2 * r * (p.in_features || 256); // simplified
    }
    default: return 0;
  }
}

function estimateFlops(node, inShape) {
  const p = node.params || {};
  if (!inShape) return 0;
  switch (node.type) {
    case 'Linear': return 2 * (p.in_features || 0) * (p.out_features || 0);
    case 'Conv2d': {
      const [C, H, W] = inShape; const k = p.k || 1, s = p.stride || 1, pad = p.pad || 0;
      const Ho = Math.floor((H + 2 * pad - k) / s) + 1; const Wo = Math.floor((W + 2 * pad - k) / s) + 1;
      return 2 * Ho * Wo * (p.out_ch || 0) * (k * k * (p.in_ch || 0));
    }
    case 'Attention': case 'FlashAttention': case 'CrossAttention': {
      const L = p.seq_len || inShape[0] || 0, h = p.hidden || inShape[1] || 0;
      return 2 * L * L * h + 6 * L * h * h;
    }
    default: return 0;
  }
}

function propagateShape(node, inShape) {
  const p = node.params || {};
  switch (node.type) {
    case 'Embedding': return [p.seq_len || inShape?.[0] || 128, p.embed_dim];
    case 'Linear': return [inShape?.[0] || 1, p.out_features];
    case 'MLP': return [inShape?.[0] || 1, p.hidden];
    case 'LayerNorm': case 'RMSNorm': case 'Dropout': case 'DropPath': return inShape;
    case 'Attention': case 'FlashAttention': case 'GroupedQueryAttn': return [p.seq_len || inShape?.[0] || 128, p.hidden || inShape?.[1] || 256];
    case 'CrossAttention': return [inShape?.[0] || 128, p.hidden || inShape?.[1] || 256];
    case 'Conv2d': {
      const C = inShape?.[0] ?? p.in_ch, H = inShape?.[1] ?? 224, W = inShape?.[2] ?? 224;
      const k = p.k || 1, s = p.stride || 1, pad = p.pad || 0;
      const Ho = Math.floor((H + 2 * pad - k) / s) + 1; const Wo = Math.floor((W + 2 * pad - k) / s) + 1;
      return [p.out_ch, Ho, Wo];
    }
    case 'Pool2d': case 'AvgPool2d': {
      const [C, H, W] = inShape || [p.in_ch || 64, 224, 224];
      const k = p.k || 2, s = p.stride || 2;
      const Ho = Math.floor((H - k) / s) + 1; const Wo = Math.floor((W - k) / s) + 1;
      return [C, Ho, Wo];
    }
    case 'Upsample2d': {
      const [C, H, W] = inShape || [64, 56, 56]; const sc = p.scale || 2;
      return [C, H * sc, W * sc];
    }
    case 'Input1D': return [p.seq_len, p.hidden];
    case 'Input2D': return [p.channels, p.height, p.width];
    case 'InputAudio': return [p.seq_len, p.n_mels];
    case 'GlobalAvgPool': case 'AdaptiveAvgPool2d': return [inShape?.[0] || 1, 1, 1];
    case 'Flatten': return [inShape?.reduce((a, b) => a * b, 1) || 1];
    default: return inShape;
  }
}

export default function ArchitectureBuilder() {
  const [nodes, setNodes] = useState([]);
  const [edges, setEdges] = useState([]);
  const [selected, setSelected] = useState(null);
  const [connecting, setConnecting] = useState(null);
  const [batchSize, setBatchSize] = useState(8);
  const [dtype, setDtype] = useState('fp16');

  // Zoom and pan state
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [isPanning, setIsPanning] = useState(false);
  const [panStart, setPanStart] = useState({ x: 0, y: 0 });

  const canvasRef = useRef(null);
  const containerRef = useRef(null);

  // Model configuration extensions
  const [modelConfig, setModelConfig] = useState({
    quantization: { enabled: false, bits: 8, method: 'int8' },
    lora: { enabled: false, rank: 8, alpha: 16, dropout: 0.05 },
    contextExtension: { enabled: false, method: 'rope', max_length: 4096 },
    optimization: { gradient_checkpointing: false, flash_attention: false },
  });

  // Figma-like controls
  const handleWheel = useCallback((e) => {
    e.preventDefault();
    const delta = e.deltaY * -0.001;
    const newZoom = Math.min(Math.max(0.1, zoom + delta), 5);
    setZoom(newZoom);
  }, [zoom]);

  const handleMouseDown = useCallback((e) => {
    if (e.button === 1 || (e.button === 0 && e.spaceKey)) { // Middle mouse or space+click
      setIsPanning(true);
      setPanStart({ x: e.clientX - pan.x, y: e.clientY - pan.y });
    }
  }, [pan]);

  const handleMouseMove = useCallback((e) => {
    if (isPanning) {
      setPan({ x: e.clientX - panStart.x, y: e.clientY - panStart.y });
    }
  }, [isPanning, panStart]);

  const handleMouseUp = useCallback(() => {
    setIsPanning(false);
  }, []);

  const handleKeyDown = useCallback((e) => {
    if (e.key === ' ') {
      e.preventDefault();
      e.spaceKey = true;
    }
    // Zoom shortcuts
    if (e.ctrlKey || e.metaKey) {
      if (e.key === '0') { setZoom(1); setPan({ x: 0, y: 0 }); } // Reset
      if (e.key === '=') { setZoom(z => Math.min(z + 0.1, 5)); } // Zoom in
      if (e.key === '-') { setZoom(z => Math.max(z - 0.1, 0.1)); } // Zoom out
    }
  }, []);

  const handleKeyUp = useCallback((e) => {
    if (e.key === ' ') {
      e.spaceKey = false;
    }
  }, []);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    container.addEventListener('wheel', handleWheel, { passive: false });
    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', handleMouseUp);
    document.addEventListener('keydown', handleKeyDown);
    document.addEventListener('keyup', handleKeyUp);

    return () => {
      container.removeEventListener('wheel', handleWheel);
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
      document.removeEventListener('keydown', handleKeyDown);
      document.removeEventListener('keyup', handleKeyUp);
    };
  }, [handleWheel, handleMouseMove, handleMouseUp, handleKeyDown, handleKeyUp]);

  const addNode = (type, x = 100, y = 100) => {
    const id = genId();
    const params = { ...(LAYERS[type]?.params || {}) };
    // Adjust for zoom and pan
    const adjustedX = (x - pan.x) / zoom;
    const adjustedY = (y - pan.y) / zoom;
    setNodes(n => [...n, { id, type, x: adjustedX, y: adjustedY, params }]);
  };

  const addEdge = (from, to) => {
    if (!from || !to || from === to) return;
    if (edges.some(e => e.from === from && e.to === to)) return;
    setEdges(e => [...e, { id: genId(), from, to }]);
  };

  const onDropPalette = (e) => {
    const type = e.dataTransfer.getData('text/plain');
    const rect = canvasRef.current?.getBoundingClientRect();
    const x = e.clientX - (rect?.left || 0);
    const y = e.clientY - (rect?.top || 0);
    if (LAYERS[type]) addNode(type, x, y);
  };

  const onDragStartPalette = (e, type) => {
    e.dataTransfer.setData('text/plain', type);
  };

  const onDragNode = (id, dx, dy) => {
    setNodes(ns => ns.map(n => n.id === id ? { ...n, x: n.x + dx / zoom, y: n.y + dy / zoom } : n));
  };

  const loadTemplate = (key) => {
    const tpl = TEMPLATES[key];
    if (!tpl) return;
    const { nodes: ns, edges: es } = tpl();
    const idMap = {};
    const newNodes = ns.map(n => { const newId = genId(); idMap[n.id] = newId; return { ...n, id: newId }; });
    const newEdges = es.map(e => ({ id: genId(), from: idMap[e.from], to: idMap[e.to] }));
    setNodes(newNodes);
    setEdges(newEdges);
    // Reset view
    setZoom(0.5);
    setPan({ x: 0, y: 0 });
  };

  const { paramCount, flops, memMB, errors, shapes } = useMemo(() => {
    const inputNodes = nodes.filter(n => n.type.startsWith('Input'));
    const inShapes = Object.fromEntries(inputNodes.map(n => [n.id, propagateShape(n, null)]));
    let shapes = { ...inShapes };
    let changed = true; let iters = 0;
    const adj = {}; edges.forEach(e => { (adj[e.from] = adj[e.from] || []).push(e.to); });

    while (changed && iters < 64) {
      changed = false; iters += 1;
      for (const n of nodes) {
        const preds = edges.filter(e => e.to === n.id).map(e => shapes[e.from]).filter(Boolean);
        const inShape = preds[0] || shapes[n.id] || null;
        const newShape = propagateShape(n, inShape);
        if (newShape && JSON.stringify(shapes[n.id]) !== JSON.stringify(newShape)) {
          shapes[n.id] = newShape; changed = true;
        }
      }
    }

    let params = 0; let f = 0; let mbytes = 0;
    const bytesPer = dtype === 'fp16' ? 2 : (dtype === 'bf16' ? 2 : 4);
    for (const n of nodes) {
      params += estimateParams(n);
      const inEdges = edges.filter(e => e.to === n.id);
      const inShape = inEdges.length ? shapes[inEdges[0].from] : shapes[n.id];
      f += estimateFlops(n, inShape);
      const outShape = shapes[n.id];
      if (outShape) {
        const actSize = outShape.reduce((a, b) => a * b, 1) * batchSize * bytesPer;
        mbytes += actSize / (1024 * 1024);
      }
    }

    const errs = [];
    // Add validation...
    return { paramCount: params, flops: f, memMB: mbytes, errors: errs, shapes };
  }, [nodes, edges, batchSize, dtype]);

  const updateParam = (id, key, value) => {
    setNodes(ns => ns.map(n => n.id === id ? { ...n, params: { ...n.params, [key]: value } } : n));
  };

  const exportPyTorch = () => {
    const code = generatePyTorch(nodes, edges, modelConfig);
    downloadFile('model.py', code);
  };

  const downloadFile = (filename, content) => {
    const blob = new Blob([content], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a'); a.href = url; a.download = filename; a.click();
    URL.revokeObjectURL(url);
  };

  // Group layers by category
  const layersByCategory = Object.entries(LAYERS).reduce((acc, [name, config]) => {
    const cat = config.category || 'Other';
    if (!acc[cat]) acc[cat] = [];
    acc[cat].push(name);
    return acc;
  }, {});

  return (
    <div className="flex h-[calc(100vh-4rem)] gap-4 p-4 overflow-hidden">
      {/* Left Sidebar - Organized Palette */}
      <div className="w-64 flex flex-col gap-3 overflow-y-auto">
        <div className="p-3 border border-border rounded bg-surface">
          <div className="text-sm font-semibold mb-2">Templates</div>
          <div className="space-y-1">
            {Object.keys(TEMPLATES).map(k => (
              <button key={k} className="w-full text-left px-2 py-1.5 text-xs border border-border rounded hover:bg-muted transition-colors" onClick={() => loadTemplate(k)}>{k}</button>
            ))}
          </div>
        </div>

        <div className="p-3 border border-border rounded bg-surface">
          <div className="text-sm font-semibold mb-2">Layers</div>
          <div className="space-y-3">
            {Object.entries(layersByCategory).map(([category, layers]) => (
              <div key={category}>
                <div className="text-xs font-medium text-text/60 mb-1">{category}</div>
                <div className="grid grid-cols-2 gap-1.5">
                  {layers.map(type => (
                    <div
                      key={type}
                      draggable
                      onDragStart={(e) => onDragStartPalette(e, type)}
                      className="px-2 py-1 text-[10px] border border-border rounded cursor-grab bg-muted hover:bg-muted/70 text-center truncate"
                      title={type}
                    >
                      {type}
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="p-3 border border-border rounded bg-surface">
          <div className="text-sm font-semibold mb-2">Model Extensions</div>
          <div className="space-y-2 text-xs">
            <label className="flex items-center gap-2">
              <input type="checkbox" checked={modelConfig.quantization.enabled} onChange={e => setModelConfig({ ...modelConfig, quantization: { ...modelConfig.quantization, enabled: e.target.checked } })} />
              <span>Quantization</span>
            </label>
            {modelConfig.quantization.enabled && (
              <div className="pl-5 space-y-1">
                <select className="w-full border border-border rounded px-2 py-1 text-xs" value={modelConfig.quantization.bits} onChange={e => setModelConfig({ ...modelConfig, quantization: { ...modelConfig.quantization, bits: parseInt(e.target.value) } })}>
                  <option value="4">4-bit</option>
                  <option value="8">8-bit</option>
                </select>
                <select className="w-full border border-border rounded px-2 py-1 text-xs" value={modelConfig.quantization.method} onChange={e => setModelConfig({ ...modelConfig, quantization: { ...modelConfig.quantization, method: e.target.value } })}>
                  <option value="int8">INT8</option>
                  <option value="int4">INT4</option>
                  <option value="nf4">NF4</option>
                  <option value="fp4">FP4</option>
                </select>
              </div>
            )}

            <label className="flex items-center gap-2">
              <input type="checkbox" checked={modelConfig.lora.enabled} onChange={e => setModelConfig({ ...modelConfig, lora: { ...modelConfig.lora, enabled: e.target.checked } })} />
              <span>LoRA</span>
            </label>
            {modelConfig.lora.enabled && (
              <div className="pl-5 space-y-1">
                <label>Rank: <input type="number" className="w-full border border-border rounded px-2 py-1 text-xs" value={modelConfig.lora.rank} onChange={e => setModelConfig({ ...modelConfig, lora: { ...modelConfig.lora, rank: parseInt(e.target.value) } })} /></label>
                <label>Alpha: <input type="number" className="w-full border border-border rounded px-2 py-1 text-xs" value={modelConfig.lora.alpha} onChange={e => setModelConfig({ ...modelConfig, lora: { ...modelConfig.lora, alpha: parseInt(e.target.value) } })} /></label>
              </div>
            )}

            <label className="flex items-center gap-2">
              <input type="checkbox" checked={modelConfig.contextExtension.enabled} onChange={e => setModelConfig({ ...modelConfig, contextExtension: { ...modelConfig.contextExtension, enabled: e.target.checked } })} />
              <span>Context Extension</span>
            </label>
            {modelConfig.contextExtension.enabled && (
              <div className="pl-5 space-y-1">
                <select className="w-full border border-border rounded px-2 py-1 text-xs" value={modelConfig.contextExtension.method} onChange={e => setModelConfig({ ...modelConfig, contextExtension: { ...modelConfig.contextExtension, method: e.target.value } })}>
                  <option value="rope">RoPE Scaling</option>
                  <option value="alibi">ALiBi</option>
                  <option value="yarn">YaRN</option>
                </select>
                <label>Max Length: <input type="number" className="w-full border border-border rounded px-2 py-1 text-xs" value={modelConfig.contextExtension.max_length} onChange={e => setModelConfig({ ...modelConfig, contextExtension: { ...modelConfig.contextExtension, max_length: parseInt(e.target.value) } })} /></label>
              </div>
            )}

            <label className="flex items-center gap-2">
              <input type="checkbox" checked={modelConfig.optimization.gradient_checkpointing} onChange={e => setModelConfig({ ...modelConfig, optimization: { ...modelConfig.optimization, gradient_checkpointing: e.target.checked } })} />
              <span>Gradient Checkpointing</span>
            </label>

            <label className="flex items-center gap-2">
              <input type="checkbox" checked={modelConfig.optimization.flash_attention} onChange={e => setModelConfig({ ...modelConfig, optimization: { ...modelConfig.optimization, flash_attention: e.target.checked } })} />
              <span>Flash Attention</span>
            </label>
          </div>
        </div>
      </div>

      {/* Center Canvas - Figma-like */}
      <div className="flex-1 flex flex-col gap-2">
        <div className="flex items-center justify-between gap-2 px-3 py-2 bg-surface border border-border rounded">
          <div className="flex items-center gap-2 text-xs">
            <button className="px-2 py-1 border border-border rounded hover:bg-muted" onClick={() => { setZoom(1); setPan({ x: 0, y: 0 }); }}>Reset View</button>
            <button className="px-2 py-1 border border-border rounded hover:bg-muted" onClick={() => setZoom(z => Math.max(z - 0.1, 0.1))}>−</button>
            <span className="px-2">{(zoom * 100).toFixed(0)}%</span>
            <button className="px-2 py-1 border border-border rounded hover:bg-muted" onClick={() => setZoom(z => Math.min(z + 0.1, 5))}>+</button>
            <span className="text-text/60 ml-2">| Space+Drag to Pan | Scroll to Zoom</span>
          </div>
          <div className="flex gap-2">
            <button className={`px-2 py-1 rounded border text-xs ${connecting ? 'border-primary text-primary' : 'border-border'}`} onClick={() => setConnecting(null)}>{connecting ? 'Click target node…' : 'Connect Mode'}</button>
            <button className="px-2 py-1 rounded border border-border hover:bg-muted text-xs" onClick={() => { setNodes([]); setEdges([]); setSelected(null); }}>Clear</button>
            <button className="px-2 py-1 rounded border border-border hover:bg-muted text-xs" onClick={exportPyTorch}>Export PyTorch</button>
          </div>
        </div>

        <div
          ref={containerRef}
          className="flex-1 relative border border-border rounded bg-[linear-gradient(90deg,rgba(0,0,0,0.03)_1px,transparent_1px),linear-gradient(180deg,rgba(0,0,0,0.03)_1px,transparent_1px)] bg-[length:20px_20px] overflow-hidden cursor-grab"
          onMouseDown={handleMouseDown}
          style={{ cursor: isPanning ? 'grabbing' : 'grab' }}
        >
          <div
            ref={canvasRef}
            onDragOver={(e) => e.preventDefault()}
            onDrop={onDropPalette}
            className="absolute inset-0"
            style={{
              transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
              transformOrigin: '0 0',
              width: '10000px',
              height: '10000px'
            }}
          >
            <svg className="absolute inset-0 pointer-events-none" style={{ width: '10000px', height: '10000px' }}>
              {edges.map(e => {
                const a = nodes.find(n => n.id === e.from); const b = nodes.find(n => n.id === e.to);
                if (!a || !b) return null;
                const x1 = a.x + 80, y1 = a.y + 20; const x2 = b.x, y2 = b.y + 20;
                const mx = (x1 + x2) / 2;
                return <path key={e.id} d={`M ${x1} ${y1} C ${mx} ${y1}, ${mx} ${y2}, ${x2} ${y2}`} stroke="#888" strokeWidth={2 / zoom} fill="none" markerEnd="url(#arrow)" />;
              })}
              <defs>
                <marker id="arrow" markerWidth="10" markerHeight="10" refX="10" refY="3" orient="auto" markerUnits="strokeWidth">
                  <path d="M0,0 L0,6 L9,3 z" fill="#888" />
                </marker>
              </defs>
            </svg>
            {nodes.map(n => (
              <DraggableNode
                key={n.id}
                node={n}
                selected={selected === n.id}
                onSelect={() => setSelected(n.id)}
                onDrag={(dx, dy) => onDragNode(n.id, dx, dy)}
                onConnect={() => {
                  if (!connecting) setConnecting(n.id); else { addEdge(connecting, n.id); setConnecting(null); }
                }}
                shapeOut={shapes[n.id]}
                zoom={zoom}
              />
            ))}
          </div>
        </div>
      </div>

      {/* Right Sidebar - Inspector */}
      <div className="w-80 flex flex-col gap-3 overflow-y-auto">
        <div className="p-3 border border-border rounded bg-surface">
          <div className="text-sm font-semibold mb-2">Global Settings</div>
          <div className="grid grid-cols-2 gap-2 text-sm">
            <label>Batch<input type="number" className="w-full border border-border rounded px-2 py-1 text-xs" value={batchSize} onChange={e => setBatchSize(parseInt(e.target.value) || 1)} /></label>
            <label>Precision<select className="w-full border border-border rounded px-2 py-1 text-xs" value={dtype} onChange={e => setDtype(e.target.value)}><option value="fp16">FP16</option><option value="bf16">BF16</option><option value="fp32">FP32</option></select></label>
          </div>
        </div>

        <div className="p-3 border border-border rounded bg-surface">
          <div className="text-sm font-semibold mb-2">Node Properties</div>
          {!selected && <div className="text-xs text-text/60">Select a node to edit</div>}
          {selected && (() => {
            const n = nodes.find(x => x.id === selected);
            if (!n) return <div />;
            const keys = Object.keys(n.params || {});
            return (
              <div className="space-y-2 text-xs">
                <div className="text-xs font-medium">{n.type}</div>
                {keys.map(k => (
                  <label key={k} className="block">
                    <div className="text-[10px] text-text/60">{k}</div>
                    <input className="w-full border border-border rounded px-2 py-1 text-xs" value={n.params[k]} onChange={e => updateParam(n.id, k, inferType(e.target.value, n.params[k]))} />
                  </label>
                ))}
                <button className="w-full px-2 py-1 border border-danger text-danger rounded hover:bg-danger/10 text-xs" onClick={() => setNodes(ns => ns.filter(x => x.id !== n.id))}>Delete Node</button>
              </div>
            );
          })()}
        </div>

        <div className="p-3 border border-border rounded bg-surface">
          <div className="text-sm font-semibold mb-1">Model Stats</div>
          <div className="space-y-1 text-xs text-text/70">
            <div>Parameters: {(paramCount / 1e6).toFixed(2)}M</div>
            <div>FLOPs: {(flops / 1e9).toFixed(2)} GFLOPs</div>
            <div>Memory: {memMB.toFixed(1)} MiB</div>
            <div>Zoom: {(zoom * 100).toFixed(0)}%</div>
          </div>
          {errors.length > 0 ? (
            <div className="mt-2 text-xs text-red-600 space-y-1">
              {errors.slice(0, 3).map((e, i) => (<div key={i}>• {e}</div>))}
              {errors.length > 3 && <div>… {errors.length - 3} more</div>}
            </div>
          ) : (
            <div className="mt-2 text-xs text-green-700">✓ No errors</div>
          )}
        </div>
      </div>
    </div>
  );
}

function inferType(val, prev) {
  if (typeof prev === 'number') {
    const n = Number(val);
    return isNaN(n) ? prev : n;
  }
  if (typeof prev === 'boolean') return val === 'true' || val === true || val === 'on';
  return val;
}

function DraggableNode({ node, onDrag, onSelect, selected, onConnect, shapeOut, zoom }) {
  const ref = useRef(null);
  useEffect(() => {
    const el = ref.current; if (!el) return;
    let last = null;
    const onMouseDown = (e) => {
      if (e.button !== 0) return;
      last = { x: e.clientX, y: e.clientY };
      document.addEventListener('mousemove', onMove);
      document.addEventListener('mouseup', onUp);
      e.stopPropagation();
      e.preventDefault();
    };
    const onMove = (e) => {
      if (!last) return;
      const dx = e.clientX - last.x;
      const dy = e.clientY - last.y;
      last = { x: e.clientX, y: e.clientY };
      onDrag(dx, dy);
    };
    const onUp = () => {
      last = null;
      document.removeEventListener('mousemove', onMove);
      document.removeEventListener('mouseup', onUp);
    };
    el.addEventListener('mousedown', onMouseDown);
    return () => { el.removeEventListener('mousedown', onMouseDown); };
  }, [onDrag]);

  return (
    <div
      ref={ref}
      onClick={(e) => { e.stopPropagation(); onSelect(); }}
      className={`absolute border rounded shadow-sm ${selected ? 'border-primary ring-2 ring-primary/30' : 'border-border'} bg-surface cursor-move`}
      style={{ left: node.x, top: node.y, width: 160 }}
    >
      <div className="px-2 py-1 text-xs font-semibold bg-muted border-b border-border flex items-center justify-between">
        <span className="truncate">{node.type}</span>
        <button className="px-1 text-[10px] border border-border rounded hover:bg-surface" onClick={(e) => { e.stopPropagation(); onConnect(); }}>•→</button>
      </div>
      <div className="p-2 text-[10px] text-text/70">
        {shapeOut && (<div className="mb-1 text-[9px] text-text/60 font-mono">shape: [{Array.isArray(shapeOut) ? shapeOut.join('×') : String(shapeOut)}]</div>)}
        {Object.entries(node.params || {}).slice(0, 3).map(([k, v]) => (
          <div key={k} className="truncate"><span className="text-text/50">{k}:</span> {String(v)}</div>
        ))}
      </div>
    </div>
  );
}

// Code generation with model config support
function generatePyTorch(nodes, edges, modelConfig) {
  let code = `# Auto-generated by Architecture Builder
import torch
import torch.nn as nn
`;

  if (modelConfig.lora?.enabled) {
    code += `from peft import LoraConfig, get_peft_model\n`;
  }

  if (modelConfig.optimization?.flash_attention) {
    code += `# Note: Install flash-attn for FlashAttention support\n`;
  }

  code += `\nclass BuiltModel(nn.Module):
    def __init__(self):
        super().__init__()
        # TODO: Initialize layers based on architecture
        pass

    def forward(self, x):
        # TODO: Implement forward pass
        return x

if __name__ == '__main__':
    model = BuiltModel()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
`;

  if (modelConfig.quantization?.enabled) {
    code += `\n# Quantization Config
# Method: ${modelConfig.quantization.method}
# Bits: ${modelConfig.quantization.bits}
`;
  }

  if (modelConfig.lora?.enabled) {
    code += `\n# LoRA Config
lora_config = LoraConfig(
    r=${modelConfig.lora.rank},
    lora_alpha=${modelConfig.lora.alpha},
    lora_dropout=${modelConfig.lora.dropout},
    target_modules=["q_proj", "v_proj"]  # Adjust based on your model
)
model = get_peft_model(model, lora_config)
`;
  }

  return code;
}
