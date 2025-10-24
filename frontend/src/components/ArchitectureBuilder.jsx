import React, { useEffect, useMemo, useRef, useState } from 'react';

// Simple layer catalog with defaults and param schemas
const LAYERS = {
  Input1D: { kind: 'input', shapeType: '1d', params: { seq_len: 128, hidden: 256 } },
  Input2D: { kind: 'input', shapeType: '2d', params: { channels: 3, height: 224, width: 224 } },
  Embedding: { kind: 'op', shapeType: '1d', params: { vocab_size: 50257, embed_dim: 256 } },
  Linear: { kind: 'op', shapeType: '1d', params: { in_features: 256, out_features: 256, bias: true } },
  MLP: { kind: 'block', shapeType: '1d', params: { hidden: 256, mlp_ratio: 4, bias: true } },
  Attention: { kind: 'op', shapeType: '1d', params: { hidden: 256, num_heads: 8, seq_len: 128 } },
  LayerNorm: { kind: 'op', shapeType: '1d', params: { hidden: 256, eps: 1e-5 } },
  Dropout: { kind: 'op', shapeType: '1d', params: { p: 0.1 } },
  Conv2d: { kind: 'op', shapeType: '2d', params: { in_ch: 3, out_ch: 64, k: 3, stride: 1, pad: 1, bias: true } },
  Pool2d: { kind: 'op', shapeType: '2d', params: { k: 2, stride: 2 } },
  Upsample2d: { kind: 'op', shapeType: '2d', params: { scale: 2 } },
  ResBlock2d: { kind: 'block', shapeType: '2d', params: { ch: 64, k: 3 } },
  Add: { kind: 'op', shapeType: 'any', params: {} },
  Concat: { kind: 'op', shapeType: 'any', params: { axis: -1 } },
  BatchNorm2d: { kind: 'op', shapeType: '2d', params: { num_features: 64, eps: 1e-5, momentum: 0.1 } },
  GELU: { kind: 'op', shapeType: 'any', params: {} },
  ReLU: { kind: 'op', shapeType: 'any', params: {} },
  Sigmoid: { kind: 'op', shapeType: 'any', params: {} },
  Tanh: { kind: 'op', shapeType: 'any', params: {} },
  Softmax: { kind: 'op', shapeType: 'any', params: { dim: -1 } },
  Conv1d: { kind: 'op', shapeType: '1d', params: { in_ch: 256, out_ch: 256, k: 3, stride: 1, pad: 1, bias: true } },
  LSTM: { kind: 'op', shapeType: '1d', params: { input_size: 256, hidden_size: 256, num_layers: 1, bidirectional: false, batch_first: true } },
  ConvTranspose2d: { kind: 'op', shapeType: '2d', params: { in_ch: 64, out_ch: 64, k: 2, stride: 2, pad: 0, bias: true } },
  GroupNorm: { kind: 'op', shapeType: 'any', params: { num_groups: 32, num_channels: 64, eps: 1e-5 } },
  Flatten: { kind: 'op', shapeType: 'any', params: { start_dim: 1, end_dim: -1 } },
  Reshape: { kind: 'op', shapeType: 'any', params: { to: '(B, -1)' } },
};

// Templates: nodes and edges with positions
const TEMPLATES = {
  'Transformer Encoder': () => {
    const nodes = [];
    const edges = [];
    let nid = 1;
    const add = (type, x, y, params={}) => { const id = String(nid++); nodes.push({ id, type, x, y, params: { ...LAYERS[type].params, ...params } }); return id; };
    const conn = (a,b) => edges.push({ id: `${a}-${b}`, from: a, to: b });
    const n0 = add('Input1D', 80, 60, { seq_len: 128, hidden: 512 });
    const n1 = add('Embedding', 260, 60, { vocab_size: 32000, embed_dim: 512 });
    const n2 = add('LayerNorm', 440, 60, { hidden: 512 });
    const n3 = add('Attention', 620, 40, { hidden: 512, num_heads: 8, seq_len: 128 });
    const n4 = add('Add', 800, 60, {});
    const n5 = add('LayerNorm', 980, 60, { hidden: 512 });
    const n6 = add('MLP', 1160, 60, { hidden: 512, mlp_ratio: 4 });
    const n7 = add('Add', 1340, 60, {});
    ;[n0,n1,n2,n3,n4,n5,n6,n7];
    conn(n0,n1); conn(n1,n2); conn(n2,n3); conn(n2,n4); conn(n3,n4); conn(n4,n5); conn(n5,n6); conn(n6,n7); conn(n4,n7);
    return { nodes, edges };
  },
  'Transformer Decoder': () => {
    const { nodes, edges } = TEMPLATES['Transformer Encoder']();
    // leave same shape; in real decoder add cross-attn node
    return { nodes, edges };
  },
  'Vision Transformer (ViT)': () => {
    const nodes=[]; const edges=[]; let nid=1;
    const add=(type,x,y,params={})=>{ const id=String(nid++); nodes.push({id,type,x,y,params:{...LAYERS[type].params,...params}}); return id; };
    const conn=(a,b)=>edges.push({id:`${a}-${b}`,from:a,to:b});
    const n0=add('Input2D',80,60,{channels:3,height:224,width:224});
    const n1=add('Conv2d',260,60,{in_ch:3,out_ch:768,k:16,stride:16,pad:0});
    const n2=add('LayerNorm',440,60,{hidden:768});
    const n3=add('Attention',620,40,{hidden:768,num_heads:12,seq_len:196});
    const n4=add('Add',800,60,{});
    const n5=add('MLP',980,60,{hidden:768,mlp_ratio:4});
    const n6=add('Add',1160,60,{});
    conn(n0,n1); conn(n1,n2); conn(n2,n3); conn(n2,n4); conn(n3,n4); conn(n4,n5); conn(n5,n6); conn(n4,n6);
    return { nodes, edges };
  },
  'UNet (Diffusion)': () => {
    const nodes=[]; const edges=[]; let nid=1;
    const add=(type,x,y,params={})=>{ const id=String(nid++); nodes.push({id,type,x,y,params:{...LAYERS[type].params,...params}}); return id; };
    const conn=(a,b)=>edges.push({id:`${a}-${b}`,from:a,to:b});
    const i=add('Input2D',80,120,{channels:3,height:256,width:256});
    const d1=add('Conv2d',260,120,{in_ch:3,out_ch:64,k:3,stride:1,pad:1});
    const p1=add('Pool2d',420,120,{k:2,stride:2});
    const d2=add('Conv2d',580,120,{in_ch:64,out_ch:128,k:3,stride:1,pad:1});
    const p2=add('Pool2d',740,120,{k:2,stride:2});
    const b=add('ResBlock2d',900,120,{ch:128});
    const u1=add('Upsample2d',1060,120,{scale:2});
    const uconv1=add('Conv2d',1220,120,{in_ch:128,out_ch:64,k:3,stride:1,pad:1});
    const u2=add('Upsample2d',1380,120,{scale:2});
    const out=add('Conv2d',1540,120,{in_ch:64,out_ch:3,k:3,stride:1,pad:1});
    conn(i,d1); conn(d1,p1); conn(p1,d2); conn(d2,p2); conn(p2,b); conn(b,u1); conn(u1,uconv1); conn(uconv1,u2); conn(u2,out);
    return { nodes, edges };
  },
  'GAN (DCGAN)': () => {
    const nodes=[]; const edges=[]; let nid=1;
    const add=(type,x,y,params={})=>{ const id=String(nid++); nodes.push({id,type,x,y,params:{...LAYERS[type].params,...params}}); return id; };
    const conn=(a,b)=>edges.push({id:`${a}-${b}`,from:a,to:b});
    const z=add('Input1D',80,80,{seq_len:1,hidden:100});
    const g1=add('Linear',260,80,{in_features:100,out_features:1024});
    const g2=add('Linear',420,80,{in_features:1024,out_features:64*64*3});
    const img=add('Input2D',600,80,{channels:3,height:64,width:64}); // placeholder output
    conn(z,g1); conn(g1,g2); conn(g2,img);
    return { nodes, edges };
  },
  'Mixture of Experts (MoE)': () => {
    const nodes=[]; const edges=[]; let nid=1;
    const add=(type,x,y,params={})=>{ const id=String(nid++); nodes.push({id,type,x,y,params:{...LAYERS[type].params,...params}}); return id; };
    const conn=(a,b)=>edges.push({id:`${a}-${b}`,from:a,to:b});
    const inp=add('Input1D',80,60,{seq_len:128,hidden:512});
    const gate=add('Linear',260,20,{in_features:512,out_features:4});
    const e1=add('MLP',260,120,{hidden:512,mlp_ratio:4});
    const e2=add('MLP',260,200,{hidden:512,mlp_ratio:4});
    const comb=add('Add',520,120,{});
    conn(inp,gate); conn(inp,e1); conn(inp,e2); conn(e1,comb); conn(e2,comb);
    return { nodes, edges };
  }
};

function genId() { return Math.random().toString(36).slice(2, 9); }

function estimateParams(node) {
  const p = node.params || {};
  switch (node.type) {
    case 'Embedding': return p.vocab_size * p.embed_dim;
    case 'Linear': return p.in_features * p.out_features + (p.bias ? p.out_features : 0);
    case 'MLP': return (p.hidden * (p.mlp_ratio*p.hidden)) + ((p.mlp_ratio*p.hidden) * p.hidden);
    case 'Conv2d': return p.out_ch * p.in_ch * (p.k*p.k) + (p.bias ? p.out_ch : 0);
    case 'Conv1d': return p.out_ch * p.in_ch * (p.k) + (p.bias ? p.out_ch : 0);
    case 'LayerNorm': return 2 * (p.hidden || 0);
    case 'BatchNorm2d': return 2 * (p.num_features || 0);
    case 'Attention': {
      const h = p.hidden || 0, L = p.seq_len || 0; // projections + attention
      const proj = 3*h*h + h*h; // qkv + out
      const attn = 2 * L * L * h; // QK^T + softmax(V)
      return proj + Math.floor(attn/10); // params: only projections; include small term
    }
    case 'LSTM': {
      const I=p.input_size||0, H=p.hidden_size||0, L=p.num_layers||1, dir=(p.bidirectional?2:1);
      const per = 4*(I*H + H*H + H);
      return dir * L * per;
    }
    default: return 0;
  }
}

function estimateFlops(node, inShape) {
  const p = node.params || {};
  if (!inShape) return 0;
  switch (node.type) {
    case 'Linear': return 2 * (p.in_features||0) * (p.out_features||0);
    case 'Conv2d': {
      const [C,H,W] = inShape; const k=p.k||1, s=p.stride||1, pad=p.pad||0; const Ho=Math.floor((H+2*pad-k)/s)+1; const Wo=Math.floor((W+2*pad-k)/s)+1;
      return 2 * Ho * Wo * (p.out_ch||0) * (k*k*(p.in_ch||0));
    }
    case 'Conv1d': {
      const L=inShape?.[0]||0; const k=p.k||1, s=p.stride||1, pad=p.pad||0; const Lo=Math.floor((L+2*pad-k)/s)+1;
      return 2 * Lo * (p.out_ch||0) * (k*(p.in_ch||0));
    }
    case 'Attention': {
      const L=p.seq_len||inShape[0]||0, h=p.hidden||inShape[1]||0;
      return 2 * L * L * h + 6 * L * h * h; // rough
    }
    case 'LSTM': {
      const L=inShape?.[0]||0; const I=p.input_size||0; const H=p.hidden_size||0; const layers=p.num_layers||1; const dir=(p.bidirectional?2:1);
      const per_step = 8*(I*H + H*H); // approx
      return dir * layers * L * per_step;
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
    case 'LayerNorm': return inShape;
    case 'BatchNorm2d': return inShape;
    case 'Attention': return [p.seq_len || inShape?.[0] || 128, p.hidden || inShape?.[1] || 256];
    case 'Conv2d': {
      const C=inShape?.[0]??p.in_ch, H=inShape?.[1]??224, W=inShape?.[2]??224; const k=p.k||1, s=p.stride||1, pad=p.pad||0; const Ho=Math.floor((H+2*pad-k)/s)+1; const Wo=Math.floor((W+2*pad-k)/s)+1; return [p.out_ch, Ho, Wo];
    }
    case 'Pool2d': {
      const [C,H,W]=inShape||[p.in_ch||64,224,224]; const k=p.k||2, s=p.stride||2; const Ho=Math.floor((H-k)/s)+1; const Wo=Math.floor((W-k)/s)+1; return [C,Ho,Wo];
    }
    case 'Upsample2d': {
      const [C,H,W]=inShape||[64,56,56]; const sc=p.scale||2; return [C, H*sc, W*sc];
    }
    case 'ResBlock2d': return inShape;
    case 'Add': return inShape;
    case 'Concat': {
      if (!inShape || !Array.isArray(inShape)) return null; // handled elsewhere
      return inShape; // simplified
    }
    case 'Conv1d': {
      const L=inShape?.[0]??128; const H=inShape?.[1]??p.in_ch; const k=p.k||1, s=p.stride||1, pad=p.pad||0; const Lo=Math.floor((L+2*pad-k)/s)+1; return [Lo, p.out_ch];
    }
    case 'LSTM': {
      const L=inShape?.[0]??128; const H=p.hidden_size||inShape?.[1]||256; return [L, H];
    }
    case 'Input1D': return [p.seq_len, p.hidden];
    case 'Input2D': return [p.channels, p.height, p.width];
    default: return inShape;
  }
}

export default function ArchitectureBuilder(){
  const [nodes, setNodes] = useState([]);
  const [edges, setEdges] = useState([]);
  const [selected, setSelected] = useState(null);
  const [connecting, setConnecting] = useState(null);
  const [batchSize, setBatchSize] = useState(8);
  const [dtype, setDtype] = useState('fp16');
  const canvasRef = useRef(null);

  const addNode = (type, x=100, y=100) => {
    const id = genId();
    const params = { ...(LAYERS[type]?.params||{}) };
    setNodes(n => [...n, { id, type, x, y, params }]);
  };
  const addEdge = (from, to) => {
    if (!from || !to || from===to) return;
    if (edges.some(e => e.from===from && e.to===to)) return;
    setEdges(e => [...e, { id: genId(), from, to }]);
  };

  const onDropPalette = (e) => {
    const type = e.dataTransfer.getData('text/plain');
    const rect = canvasRef.current?.getBoundingClientRect();
    const x = e.clientX - (rect?.left||0); const y = e.clientY - (rect?.top||0);
    if (LAYERS[type]) addNode(type, x, y);
  };

  const onDragStartPalette = (e, type) => {
    e.dataTransfer.setData('text/plain', type);
  };

  const onDragNode = (id, dx, dy) => {
    setNodes(ns => ns.map(n => n.id===id ? { ...n, x: n.x + dx, y: n.y + dy } : n));
  };

  const palette = Object.keys(LAYERS);

  const loadTemplate = (key) => {
    const tpl = TEMPLATES[key];
    if (!tpl) return;
    const { nodes: ns, edges: es } = tpl();
    const idMap = {};
    const newNodes = ns.map(n => { const newId = genId(); idMap[n.id] = newId; return { ...n, id: newId }; });
    const newEdges = es.map(e => ({ id: genId(), from: idMap[e.from], to: idMap[e.to] }));
    setNodes(newNodes);
    setEdges(newEdges);
  };

  const { paramCount, flops, memMB, errors, shapes } = useMemo(() => {
    // topo-ish: iterate until no change; simple single-input graph support
    const inputNodes = nodes.filter(n => n.type.startsWith('Input'));
    const inShapes = Object.fromEntries(inputNodes.map(n => [n.id, propagateShape(n, null)]));
    let shapes = { ...inShapes };
    let changed = true; let iters = 0;
    const adj = {}; edges.forEach(e => { (adj[e.from] = adj[e.from]||[]).push(e.to); });
    while (changed && iters < 64) {
      changed = false; iters += 1;
      for (const n of nodes) {
        const preds = edges.filter(e => e.to===n.id).map(e => shapes[e.from]).filter(Boolean);
        const inShape = preds[0] || shapes[n.id] || null;
        const newShape = propagateShape(n, inShape);
        if (newShape && JSON.stringify(shapes[n.id]) !== JSON.stringify(newShape)) {
          shapes[n.id] = newShape; changed = true;
        }
      }
    }
    // param count + FLOPs + mem
    let params = 0; let f = 0; let mbytes = 0; const bytesPer = dtype==='fp16'?2:(dtype==='bf16'?2:4);
    for (const n of nodes) {
      params += estimateParams(n);
      const inEdges = edges.filter(e => e.to===n.id);
      const inShape = inEdges.length ? shapes[inEdges[0].from] : shapes[n.id];
      f += estimateFlops(n, inShape);
      const outShape = shapes[n.id];
      if (outShape) {
        const actSize = outShape.reduce((a,b)=>a*b, 1) * batchSize * bytesPer;
        mbytes += actSize / (1024*1024);
      }
    }
    // simple validations
    const errs = [];
    for (const e of edges) {
      const a = shapes[e.from]; const b = shapes[e.to];
      if (!a || !b) continue;
      // For Linear: in_features must match previous hidden
      const nodeB = nodes.find(n => n.id===e.to);
      if (nodeB?.type==='Linear') {
        const need = nodeB.params?.in_features; const have = a?.[1]||a?.[0];
        if (need && have && need !== have) errs.push(`Linear in_features ${need} != previous dim ${have} at node ${e.to}`);
      }
      if (nodeB?.type==='Conv2d') {
        const need = nodeB.params?.in_ch; const have = a?.[0];
        if (need && have && need !== have) errs.push(`Conv2d in_ch ${need} != previous channels ${have} at node ${e.to}`);
      }
      if (nodeB?.type==='BatchNorm2d') {
        const need = nodeB.params?.num_features; const have = a?.[0];
        if (need && have && need !== have) errs.push(`BatchNorm2d num_features ${need} != previous channels ${have} at node ${e.to}`);
      }
      if (nodeB?.type==='Conv1d') {
        const need = nodeB.params?.in_ch; const have = a?.[1];
        if (need && have && need !== have) errs.push(`Conv1d in_ch ${need} != previous hidden ${have} at node ${e.to}`);
      }
      if (nodeB?.type==='LSTM') {
        const need = nodeB.params?.input_size; const have = a?.[1];
        if (need && have && need !== have) errs.push(`LSTM input_size ${need} != previous hidden ${have} at node ${e.to}`);
      }
      if (nodeB?.type==='Attention') {
        const h = nodeB.params?.hidden; const have = a?.[1]||0; if (h && have && h !== have) errs.push(`Attention hidden ${h} != previous hidden ${have} at node ${e.to}`);
        const L = nodeB.params?.seq_len; const haveL = a?.[0]||0; if (L && haveL && L !== haveL) errs.push(`Attention seq_len ${L} != previous seq_len ${haveL} at node ${e.to}`);
      }
    }
    return { paramCount: params, flops: f, memMB: mbytes, errors: errs, shapes };
  }, [nodes, edges, batchSize, dtype]);

  const updateParam = (id, key, value) => {
    setNodes(ns => ns.map(n => n.id===id ? { ...n, params: { ...n.params, [key]: value } } : n));
  };

  const exportPyTorch = () => {
    const code = generatePyTorch(nodes, edges);
    showCode('PyTorch nn.Module', code, 'model.py');
  };
  const exportHF = () => {
    const code = generateHuggingFace(nodes, edges);
    showCode('HuggingFace Compatible', code, 'hf_model.py');
  };
  const exportTF = () => {
    const code = generateKeras(nodes, edges);
    showCode('TensorFlow / Keras', code, 'model_keras.py');
  };

  const exportZip = async () => {
    const py = generatePyTorch(nodes, edges);
    const hf = generateHuggingFace(nodes, edges);
    const tf = generateKeras(nodes, edges);
    const readme = `# Architecture Export\n\nThis archive contains code generated by the Architecture Builder.\n\nContents:\n- code/pytorch/model.py\n- code/huggingface/hf_model.py\n- code/keras/model_keras.py\n\nStats (approx):\n- Parameters: ${paramCount.toLocaleString()}\n- FLOPs: ${(flops/1e9).toFixed(2)} GFLOPs\n- Activations memory: ${memMB.toFixed(1)} MiB (batch=${batchSize}, dtype=${dtype})\n\nNotes:\n- Validation errors (if any) should be resolved before training.\n- The HF skeleton is a starting point; adapt to your exact architecture.\n`;
    const torchTrain = `# Minimal PyTorch training loop\nimport torch\nfrom torch import nn, optim\nfrom code.pytorch.model import BuiltModel\n\nif __name__ == '__main__':\n    model = BuiltModel()\n    optimiz = optim.AdamW(model.parameters(), lr=3e-4)\n    crit = nn.MSELoss()\n    model.train()\n    for step in range(100):\n        x = torch.randn(8, 16, 256)  # adjust shape to your model\n        y = torch.randn(8, 16, 256)\n        optimiz.zero_grad()\n        out = model(x)\n        loss = crit(out, y)\n        loss.backward()\n        optimiz.step()\n        if step % 10 == 0:\n            print(f'step {step} loss {loss.item():.4f}')\n`;
    const req = `# Suggested requirements\ntorch\ntransformers\ntensorflow\n`;
    const files = [
      { path: 'README.md', content: readme },
      { path: 'code/pytorch/model.py', content: py },
      { path: 'code/huggingface/hf_model.py', content: hf },
      { path: 'code/keras/model_keras.py', content: tf },
      { path: 'code/train/torch_train.py', content: torchTrain },
      { path: 'code/requirements.txt', content: req },
    ];
    try {
      const res = await fetch('/api/export/zip', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ files, name: 'architecture_export.zip' }) });
      if (!res.ok) throw new Error('Zip export failed');
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a'); a.href = url; a.download = 'architecture_export.zip'; a.click();
      URL.revokeObjectURL(url);
    } catch (e) { alert(String(e.message||e)); }
  };

  // Save/Load via localStorage
  const STORAGE_KEY = 'arch_builder.designs';
  const [designs, setDesigns] = useState([]);
  const loadDesigns = () => {
    try { const raw = localStorage.getItem(STORAGE_KEY); setDesigns(raw ? JSON.parse(raw) : []); } catch { setDesigns([]); }
  };
  useEffect(()=>{ loadDesigns(); }, []);
  const saveDesign = () => {
    const name = prompt('Name for this design?');
    if (!name) return;
    const item = { id: genId(), name, ts: Date.now(), nodes, edges, batchSize, dtype };
    const next = [item, ...designs].slice(0, 100);
    localStorage.setItem(STORAGE_KEY, JSON.stringify(next));
    setDesigns(next);
  };
  const loadDesignItem = (id) => {
    const it = designs.find(d => d.id===id); if (!it) return;
    setNodes(it.nodes||[]); setEdges(it.edges||[]); setBatchSize(it.batchSize||8); setDtype(it.dtype||'fp16');
  };
  const deleteDesignItem = (id) => {
    const next = designs.filter(d => d.id!==id);
    localStorage.setItem(STORAGE_KEY, JSON.stringify(next));
    setDesigns(next);
  };
  const exportJSON = () => {
    const payload = JSON.stringify({ nodes, edges, batchSize, dtype }, null, 2);
    const blob = new Blob([payload], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a'); a.href = url; a.download = 'architecture.json'; a.click(); URL.revokeObjectURL(url);
  };
  const importJSON = async (file) => {
    try { const text = await file.text(); const data = JSON.parse(text); setNodes(data.nodes||[]); setEdges(data.edges||[]); setBatchSize(data.batchSize||8); setDtype(data.dtype||'fp16'); } catch { alert('Invalid JSON'); }
  };

  const showCode = (title, code, filename) => {
    const blob = new Blob([code], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a'); a.href = url; a.download = filename; a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="grid grid-cols-12 gap-4">
      {/* Palette */}
      <div className="col-span-12 lg:col-span-2 space-y-3">
        <div className="p-3 border border-border rounded bg-surface">
          <div className="text-sm font-semibold mb-2">Layer Palette</div>
          <div className="grid grid-cols-2 gap-2 text-xs">
            {palette.map(type => (
              <div key={type}
                   draggable
                   onDragStart={(e)=>onDragStartPalette(e,type)}
                   className="px-2 py-1 border border-border rounded cursor-grab bg-muted hover:bg-muted/70 text-center">
                {type}
              </div>
            ))}
          </div>
        </div>
        <div className="p-3 border border-border rounded bg-surface">
          <div className="text-sm font-semibold mb-2">Designs</div>
          <div className="flex gap-2 mb-2">
            <button className="px-2 py-1 border border-border rounded hover:bg-muted text-sm" onClick={saveDesign}>Save</button>
            <button className="px-2 py-1 border border-border rounded hover:bg-muted text-sm" onClick={exportJSON}>Export JSON</button>
            <label className="px-2 py-1 border border-border rounded hover:bg-muted text-sm cursor-pointer">
              Import JSON<input type="file" accept="application/json" className="hidden" onChange={e=>{ const f=e.target.files?.[0]; if (f) importJSON(f); e.target.value=''; }} />
            </label>
          </div>
          <div className="space-y-1 max-h-40 overflow-auto text-sm">
            {designs.length===0 && (<div className="text-text/60 text-xs">No saved designs</div>)}
            {designs.map(d => (
              <div key={d.id} className="flex items-center justify-between gap-2">
                <button className="flex-1 text-left truncate hover:underline" onClick={()=>loadDesignItem(d.id)} title={new Date(d.ts).toLocaleString()}>{d.name}</button>
                <button className="text-xs text-danger" onClick={()=>deleteDesignItem(d.id)}>Delete</button>
              </div>
            ))}
          </div>
        </div>
        <div className="p-3 border border-border rounded bg-surface">
          <div className="text-sm font-semibold mb-2">Templates</div>
          <div className="space-y-2">
            {Object.keys(TEMPLATES).map(k => (
              <button key={k} className="w-full text-left px-2 py-1 border border-border rounded hover:bg-muted text-sm" onClick={()=>loadTemplate(k)}>{k}</button>
            ))}
          </div>
        </div>
        <div className="p-3 border border-border rounded bg-surface">
          <div className="text-sm font-semibold mb-2">Export</div>
          <div className="flex flex-col gap-2">
            <button className="px-2 py-1 border border-border rounded hover:bg-muted text-sm" onClick={exportPyTorch}>PyTorch nn.Module</button>
            <button className="px-2 py-1 border border-border rounded hover:bg-muted text-sm" onClick={exportHF}>HuggingFace Compatible</button>
            <button className="px-2 py-1 border border-border rounded hover:bg-muted text-sm" onClick={exportTF}>TensorFlow / Keras</button>
            <button className="px-2 py-1 border border-border rounded hover:bg-muted text-sm" onClick={exportZip}>Export ZIP (all)</button>
          </div>
        </div>
      </div>

      {/* Canvas */}
      <div className="col-span-12 lg:col-span-7">
        <div
          ref={canvasRef}
          onDragOver={(e)=>e.preventDefault()}
          onDrop={onDropPalette}
          className="relative h-[640px] border border-border rounded bg-[linear-gradient(90deg,rgba(0,0,0,0.03)_1px,transparent_1px),linear-gradient(180deg,rgba(0,0,0,0.03)_1px,transparent_1px)] bg-[length:20px_20px] overflow-hidden">
          {/* edges as SVG */}
          <svg className="absolute inset-0 w-full h-full pointer-events-none">
            {edges.map(e => {
              const a = nodes.find(n => n.id===e.from); const b = nodes.find(n => n.id===e.to);
              if (!a || !b) return null;
              const x1 = a.x + 80, y1 = a.y + 20; const x2 = b.x, y2 = b.y + 20;
              const mx = (x1+x2)/2;
              return <path key={e.id} d={`M ${x1} ${y1} C ${mx} ${y1}, ${mx} ${y2}, ${x2} ${y2}`} stroke="#888" strokeWidth={2} fill="none" markerEnd="url(#arrow)"/>;
            })}
            <defs>
              <marker id="arrow" markerWidth="10" markerHeight="10" refX="10" refY="3" orient="auto" markerUnits="strokeWidth">
                <path d="M0,0 L0,6 L9,3 z" fill="#888" />
              </marker>
            </defs>
          </svg>
          {/* nodes */}
          {nodes.map(n => (
            <DraggableNode key={n.id} node={n}
              selected={selected===n.id}
              onSelect={()=>setSelected(n.id)}
              onDrag={(dx,dy)=>onDragNode(n.id,dx,dy)}
              onConnect={() => {
                if (!connecting) setConnecting(n.id); else { addEdge(connecting, n.id); setConnecting(null); }
              }}
              shapeOut={shapes[n.id]}
            />
          ))}
        </div>
        <div className="mt-2 flex items-center gap-2 text-sm">
          <button className={`px-2 py-1 rounded border ${connecting?'border-primary text-primary':'border-border'}`} onClick={()=>setConnecting(null)}>{connecting? 'Click target node…' : 'Add Connection'}</button>
          <button className="px-2 py-1 rounded border border-border" onClick={()=>{ setNodes([]); setEdges([]); setSelected(null); }}>Clear</button>
        </div>
      </div>

      {/* Inspector */}
      <div className="col-span-12 lg:col-span-3 space-y-3">
        <div className="p-3 border border-border rounded bg-surface">
          <div className="text-sm font-semibold mb-2">Global</div>
          <div className="grid grid-cols-2 gap-2 text-sm">
            <label>Batch<input type="number" className="w-full border border-border rounded px-2 py-1" value={batchSize} onChange={e=>setBatchSize(parseInt(e.target.value)||1)} /></label>
            <label>Precision<select className="w-full border border-border rounded px-2 py-1" value={dtype} onChange={e=>setDtype(e.target.value)}><option value="fp16">FP16</option><option value="bf16">BF16</option><option value="fp32">FP32</option></select></label>
          </div>
        </div>
        <div className="p-3 border border-border rounded bg-surface">
          <div className="text-sm font-semibold mb-2">Node Properties</div>
          {!selected && <div className="text-sm text-text/60">Select a node to edit parameters</div>}
          {selected && (()=>{
            const n = nodes.find(x=>x.id===selected);
            if (!n) return <div/>;
            const keys = Object.keys(n.params||{});
            return (
              <div className="space-y-2 text-sm">
                <div className="text-xs text-text/60">{n.type}</div>
                {keys.map(k => (
                  <label key={k} className="block">
                    <div className="text-xs">{k}</div>
                    <input className="w-full border border-border rounded px-2 py-1" value={n.params[k]} onChange={e=>updateParam(n.id, k, inferType(e.target.value, n.params[k]))}/>
                  </label>
                ))}
                <div className="flex gap-2">
                  <button className="px-2 py-1 border border-border rounded hover:bg-muted" onClick={()=>setNodes(ns=>ns.filter(x=>x.id!==n.id))}>Delete</button>
                </div>
              </div>
            );
          })()}
        </div>
        <div className="p-3 border border-border rounded bg-surface">
          <div className="text-sm font-semibold mb-1">Validation & Stats</div>
          <div className="text-xs text-text/70">Params: {paramCount.toLocaleString()}</div>
          <div className="text-xs text-text/70">Approx FLOPs: {(flops/1e9).toFixed(2)} GFLOPs</div>
          <div className="text-xs text-text/70">Activations (est.): {memMB.toFixed(1)} MiB</div>
          {errors.length>0 ? (
            <div className="mt-2 text-xs text-red-600 space-y-1">
              {errors.slice(0,5).map((e,i)=>(<div key={i}>• {e}</div>))}
              {errors.length>5 && <div>… {errors.length-5} more</div>}
            </div>
          ) : (
            <div className="mt-2 text-xs text-green-700">No validation errors found.</div>
          )}
        </div>
      </div>
    </div>
  );
}

function inferType(val, prev){
  if (typeof prev === 'number') {
    const n = Number(val);
    return isNaN(n) ? prev : n;
  }
  if (typeof prev === 'boolean') return val === 'true' || val === true || val === 'on';
  return val;
}

function DraggableNode({ node, onDrag, onSelect, selected, onConnect, shapeOut }){
  const ref = useRef(null);
  useEffect(() => {
    const el = ref.current; if (!el) return;
    let last = null;
    const onMouseDown = (e) => { if (e.button!==0) return; last = { x: e.clientX, y: e.clientY }; document.addEventListener('mousemove', onMove); document.addEventListener('mouseup', onUp); e.preventDefault(); };
    const onMove = (e) => { if (!last) return; const dx = e.clientX - last.x; const dy = e.clientY - last.y; last = { x: e.clientX, y: e.clientY }; onDrag(dx,dy); };
    const onUp = () => { last = null; document.removeEventListener('mousemove', onMove); document.removeEventListener('mouseup', onUp); };
    el.addEventListener('mousedown', onMouseDown);
    return () => { el.removeEventListener('mousedown', onMouseDown); };
  }, [onDrag]);
  return (
    <div ref={ref} onClick={onSelect} className={`absolute border rounded shadow-sm ${selected?'border-primary ring-2 ring-primary/30':'border-border'} bg-surface`} style={{ left: node.x, top: node.y, width: 160 }}>
      <div className="px-2 py-1 text-xs font-semibold bg-muted border-b border-border flex items-center justify-between">
        <span>{node.type}</span>
        <button className="px-1 text-[11px] border border-border rounded" onClick={(e)=>{ e.stopPropagation(); onConnect(); }}>Connect</button>
      </div>
      <div className="p-2 text-[11px] text-text/70">
        {shapeOut && (<div className="mb-1 text-[10px] text-text/60">shape: [{Array.isArray(shapeOut) ? shapeOut.join('×') : String(shapeOut)}]</div>)}
        {Object.entries(node.params||{}).slice(0,3).map(([k,v]) => (
          <div key={k} className="truncate">{k}: {String(v)}</div>
        ))}
      </div>
    </div>
  );
}

// Code generators (simplified)
function topologicalOrder(nodes, edges){
  const indeg = Object.fromEntries(nodes.map(n => [n.id, 0]));
  for (const e of edges) indeg[e.to] = (indeg[e.to]||0) + 1;
  const q = nodes.filter(n => (indeg[n.id]||0) === 0).map(n=>n.id);
  const order=[]; const adj={}; edges.forEach(e=>{ (adj[e.from]=adj[e.from]||[]).push(e.to); });
  while(q.length){ const u=q.shift(); order.push(u); for (const v of (adj[u]||[])){ indeg[v]--; if (indeg[v]===0) q.push(v);} }
  return order.map(id => nodes.find(n=>n.id===id)).filter(Boolean);
}

function generatePyTorch(nodes, edges){
  const order = topologicalOrder(nodes, edges);
  const layerLines=[]; const forwardLines=[];
  let idx=0; const nameMap={};
  for (const n of order){
    const name = `${n.type.toLowerCase()}_${idx++}`.replace(/[^a-z0-9_]+/g,'_');
    nameMap[n.id]=name;
    const p = n.params||{};
    switch(n.type){
      case 'Embedding': layerLines.push(`${name} = nn.Embedding(${p.vocab_size}, ${p.embed_dim})`); break;
      case 'Linear': layerLines.push(`${name} = nn.Linear(${p.in_features}, ${p.out_features}, bias=${!!p.bias})`); break;
      case 'LayerNorm': layerLines.push(`${name} = nn.LayerNorm(${p.hidden})`); break;
      case 'Dropout': layerLines.push(`${name} = nn.Dropout(p=${p.p})`); break;
      case 'Conv2d': layerLines.push(`${name} = nn.Conv2d(${p.in_ch}, ${p.out_ch}, kernel_size=${p.k}, stride=${p.stride}, padding=${p.pad}, bias=${!!p.bias})`); break;
      case 'BatchNorm2d': layerLines.push(`${name} = nn.BatchNorm2d(${p.num_features}, eps=${p.eps}, momentum=${p.momentum})`); break;
      case 'Pool2d': layerLines.push(`${name} = nn.MaxPool2d(kernel_size=${p.k}, stride=${p.stride})`); break;
      case 'Upsample2d': layerLines.push(`${name} = nn.Upsample(scale_factor=${p.scale}, mode='nearest')`); break;
      case 'Conv1d': layerLines.push(`${name} = nn.Conv1d(${p.in_ch}, ${p.out_ch}, kernel_size=${p.k}, stride=${p.stride}, padding=${p.pad}, bias=${!!p.bias})`); break;
      case 'LSTM': layerLines.push(`${name} = nn.LSTM(input_size=${p.input_size}, hidden_size=${p.hidden_size}, num_layers=${p.num_layers}, bidirectional=${!!p.bidirectional}, batch_first=${!!p.batch_first})`); break;
      case 'GELU': layerLines.push(`${name} = nn.GELU()`); break;
      case 'ReLU': layerLines.push(`${name} = nn.ReLU()`); break;
      case 'Sigmoid': layerLines.push(`${name} = nn.Sigmoid()`); break;
      case 'Tanh': layerLines.push(`${name} = nn.Tanh()`); break;
      case 'Softmax': layerLines.push(`${name} = nn.Softmax(dim=${p.dim})`); break;
      case 'ConvTranspose2d': layerLines.push(`${name} = nn.ConvTranspose2d(${p.in_ch}, ${p.out_ch}, kernel_size=${p.k}, stride=${p.stride}, padding=${p.pad}, bias=${!!p.bias})`); break;
      case 'GroupNorm': layerLines.push(`${name} = nn.GroupNorm(${p.num_groups}, ${p.num_channels}, eps=${p.eps})`); break;
      case 'Flatten': layerLines.push(`${name} = nn.Flatten(start_dim=${p.start_dim}, end_dim=${p.end_dim})`); break;
      case 'Reshape': layerLines.push(`# ${name}: Reshape to ${p.to}`); break;
      case 'Attention': layerLines.push(`${name} = nn.MultiheadAttention(embed_dim=${p.hidden}, num_heads=${p.num_heads}, batch_first=True)`); break;
      case 'MLP': layerLines.push(`${name} = nn.Sequential(nn.Linear(${p.hidden}, ${p.mlp_ratio*p.hidden}), nn.GELU(), nn.Linear(${p.mlp_ratio*p.hidden}, ${p.hidden}))`); break;
      case 'Add': layerLines.push(`# ${name}: Add`); break;
      case 'Concat': layerLines.push(`# ${name}: Concat(dim=${p.axis})`); break;
      case 'Input1D': case 'Input2D': default: layerLines.push(`# ${name}: ${n.type}`); break;
    }
  }
  // forward pass
  for (const n of order){
    const name = nameMap[n.id];
    const preds = edges.filter(e=>e.to===n.id).map(e=> nameMap[e.from]);
    if (n.type==='Input1D') { forwardLines.push(`# ${name}: expects (B, L, H) or ids`); forwardLines.push(`${name}_out = x if 'x' in locals() else x`); continue; }
    if (n.type==='Input2D') { forwardLines.push(`# ${name}: expects (B, C, H, W)`); forwardLines.push(`${name}_out = x if 'x' in locals() else x`); continue; }
    const inName = preds[0] ? `${preds[0]}_out` : 'x';
    switch(n.type){
      case 'Conv1d': forwardLines.push(`${name}_out = ${name}(${inName}.transpose(1,2)).transpose(1,2)`); break;
      case 'LSTM': forwardLines.push(`${name}_out, _ = ${name}(${inName})`); break;
      case 'GELU': forwardLines.push(`${name}_out = ${name}(${inName})`); break;
      case 'ReLU': forwardLines.push(`${name}_out = ${name}(${inName})`); break;
      case 'Sigmoid': forwardLines.push(`${name}_out = ${name}(${inName})`); break;
      case 'Tanh': forwardLines.push(`${name}_out = ${name}(${inName})`); break;
      case 'Softmax': forwardLines.push(`${name}_out = ${name}(${inName})`); break;
      case 'Embedding': forwardLines.push(`${name}_out = ${name}(${inName})`); break;
      case 'Linear': forwardLines.push(`${name}_out = ${name}(${inName})`); break;
      case 'LayerNorm': forwardLines.push(`${name}_out = ${name}(${inName})`); break;
      case 'Dropout': forwardLines.push(`${name}_out = ${name}(${inName})`); break;
      case 'Conv2d': forwardLines.push(`${name}_out = ${name}(${inName})`); break;
      case 'BatchNorm2d': forwardLines.push(`${name}_out = ${name}(${inName})`); break;
      case 'Pool2d': forwardLines.push(`${name}_out = ${name}(${inName})`); break;
      case 'Upsample2d': forwardLines.push(`${name}_out = ${name}(${inName})`); break;
      case 'Attention': forwardLines.push(`${name}_out, _ = ${name}(${inName}, ${inName}, ${inName})`); break;
      case 'MLP': forwardLines.push(`${name}_out = ${name}(${inName})`); break;
      case 'Add': {
        const a = preds[0] ? `${preds[0]}_out` : 'x'; const b = preds[1] ? `${preds[1]}_out` : a; forwardLines.push(`${name}_out = ${a} + ${b}`); break; }
      case 'Concat': {
        const a = preds[0] ? `${preds[0]}_out` : 'x'; const b = preds[1] ? `${preds[1]}_out` : a; const dim = (nodes.find(nn=>nn.id===n.id)?.params?.axis)||-1; forwardLines.push(`${name}_out = torch.cat([${a}, ${b}], dim=${dim})`); break; }
      default: forwardLines.push(`${name}_out = ${inName}`);
    }
  }

  const code = `# Auto-generated by Architecture Builder\n\nimport torch\nimport torch.nn as nn\n\nclass BuiltModel(nn.Module):\n    def __init__(self):\n        super().__init__()\n${layerLines.map(l=>`        self.${l}`).join('\n')}\n\n    def forward(self, x):\n${forwardLines.map(l=>`        ${l}`).join('\n')}\n        return ${order.length? `${nameMap[order[order.length-1].id]}_out` : 'x'}\n\nif __name__ == '__main__':\n    m = BuiltModel()\n    print(m)\n`;
  return code;
}

function generateHuggingFace(nodes, edges){
  // For transformer-like templates, emit a config + AutoModel suggestion; otherwise fallback to PyTorch code
  const hasAttn = nodes.some(n => n.type==='Attention');
  const ln = nodes.find(n => n.type==='LayerNorm');
  const hid = (ln?.params?.hidden) || 256;
  if (hasAttn) {
    return `# Hugging Face compatible skeleton\nfrom transformers import PretrainedConfig, PreTrainedModel\nimport torch, torch.nn as nn\n\nclass BuilderConfig(PretrainedConfig):\n    def __init__(self, hidden_size=${hid}, num_attention_heads=8, **kwargs):\n        super().__init__(**kwargs)\n        self.hidden_size = hidden_size\n        self.num_attention_heads = num_attention_heads\n\nclass BuilderModel(PreTrainedModel):\n    config_class = BuilderConfig\n    def __init__(self, config: BuilderConfig):\n        super().__init__(config)\n        self.ln = nn.LayerNorm(config.hidden_size)\n        self.attn = nn.MultiheadAttention(config.hidden_size, config.num_attention_heads, batch_first=True)\n        self.mlp = nn.Sequential(nn.Linear(config.hidden_size, 4*config.hidden_size), nn.GELU(), nn.Linear(4*config.hidden_size, config.hidden_size))\n        self.post_init()\n    def forward(self, input_ids=None, inputs_embeds=None, attention_mask=None):\n        x = inputs_embeds if inputs_embeds is not None else input_ids\n        x = self.ln(x)\n        y,_ = self.attn(x,x,x)\n        x = x + y\n        x = x + self.mlp(x)\n        return x\n`;
  }
  return generatePyTorch(nodes, edges);
}

function generateKeras(nodes, edges){
  const order = topologicalOrder(nodes, edges);
  const lines = [
    'import tensorflow as tf',
    'from tensorflow import keras',
    'from tensorflow.keras import layers',
    '',
    'def build_model():',
    '    x = keras.Input(shape=(None, 256))  # adjust to your input',
    '    t = x',
  ];
  let idx=0;
  for (const n of order){
    const name = `${n.type.toLowerCase()}_${idx++}`.replace(/[^a-z0-9_]+/g,'_');
    const p = n.params||{};
    switch(n.type){
      case 'Linear': lines.push(`    t = layers.Dense(${p.out_features}, use_bias=${!!p.bias}, name='${name}')(t)`); break;
      case 'LayerNorm': lines.push(`    t = layers.LayerNormalization(name='${name}')(t)`); break;
      case 'Dropout': lines.push(`    t = layers.Dropout(${p.p}, name='${name}')(t)`); break;
      case 'Attention': lines.push(`    t = layers.MultiHeadAttention(num_heads=${p.num_heads}, key_dim=${Math.max(1, Math.floor((p.hidden||256)/(p.num_heads||1)))}, name='${name}')(t, t)`); break;
      case 'Conv2d': lines.push(`    t = layers.Conv2D(${p.out_ch}, kernel_size=${p.k}, strides=${p.stride}, padding='same', use_bias=${!!p.bias}, name='${name}')(t)`); break;
      case 'BatchNorm2d': lines.push(`    t = layers.BatchNormalization(name='${name}')(t)`); break;
      case 'Pool2d': lines.push(`    t = layers.MaxPooling2D(pool_size=${p.k}, strides=${p.stride}, name='${name}')(t)`); break;
      case 'ConvTranspose2d': lines.push(`    # Keras: use Conv2DTranspose`); lines.push(`    t = layers.Conv2DTranspose(${p.out_ch}, kernel_size=${p.k}, strides=${p.stride}, padding='same', use_bias=${!!p.bias}, name='${name}')(t)`); break;
      case 'GroupNorm': lines.push(`    # GroupNorm not in core Keras; consider tfa.layers.GroupNormalization`); break;
      case 'Flatten': lines.push(`    t = layers.Flatten(name='${name}')(t)`); break;
      case 'Reshape': lines.push(`    t = layers.Reshape(target_shape=( -1, ), name='${name}')(t)  # adjust shape`); break;
      case 'Upsample2d': lines.push(`    t = layers.UpSampling2D(size=${p.scale}, name='${name}')(t)`); break;
      case 'MLP': lines.push(`    t = layers.Dense(${(p.mlp_ratio||4)*(p.hidden||256)}, activation='gelu', name='${name}_fc1')(t)`);
                     lines.push(`    t = layers.Dense(${p.hidden||256}, name='${name}_fc2')(t)`); break;
      case 'GELU': lines.push(`    t = layers.Activation('gelu', name='${name}')(t)`); break;
      case 'ReLU': lines.push(`    t = layers.ReLU(name='${name}')(t)`); break;
      case 'Sigmoid': lines.push(`    t = layers.Activation('sigmoid', name='${name}')(t)`); break;
      case 'Tanh': lines.push(`    t = layers.Activation('tanh', name='${name}')(t)`); break;
      case 'Softmax': lines.push(`    t = layers.Softmax(axis=${p.dim||-1}, name='${name}')(t)`); break;
      case 'Conv1d': lines.push(`    t = layers.Conv1D(${p.out_ch}, kernel_size=${p.k}, strides=${p.stride}, padding='same', use_bias=${!!p.bias}, name='${name}')(t)`); break;
      case 'LSTM': lines.push(`    t = layers.LSTM(${p.hidden_size}, return_sequences=True, name='${name}')(t)`); break;
      default: lines.push(`    # ${name}: ${n.type}`);
    }
  }
  lines.push('    return keras.Model(inputs=x, outputs=t)');
  lines.push('');
  lines.push('if __name__ == "__main__":');
  lines.push('    m = build_model()');
  lines.push('    m.summary()');
  return lines.join('\n');
}
