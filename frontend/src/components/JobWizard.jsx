import React, { useEffect, useMemo, useState } from 'react';

export default function JobWizard({ onNavigate, frameworks, partitions, api }) {
  const [step, setStep] = useState(0);
  const [jobType, setJobType] = useState('train');
  const [framework, setFramework] = useState('pytorch');
  const [arch, setArch] = useState('custom');
  const [preset, setPreset] = useState('');
  const [name, setName] = useState('');
  const [errors, setErrors] = useState([]);
  const [warnings, setWarnings] = useState([]);
  const [validating, setValidating] = useState(false);
  const [gpuMode, setGpuMode] = useState('auto');
  const [gpuAutoPrefer, setGpuAutoPrefer] = useState('mig_first');
  const [gpuSelection, setGpuSelection] = useState('');

  // Data config
  const [dataSource, setDataSource] = useState('local');
  const [localPath, setLocalPath] = useState('');
  const [hfDataset, setHfDataset] = useState('');
  const [s3Uri, setS3Uri] = useState('');
  const [gcsUri, setGcsUri] = useState('');
  const [split, setSplit] = useState({ train: 0.8, val: 0.1, test: 0.1 });
  const [augment, setAugment] = useState({ flip: true, rotate: false, normalize: true });
  const [loader, setLoader] = useState({ num_workers: 4, prefetch: 2, pin_memory: true });
  const [localFiles, setLocalFiles] = useState([]);

  // Training config
  const [epochs, setEpochs] = useState(10);
  const [batchSize, setBatchSize] = useState(32);
  const [lr, setLr] = useState(0.001);
  const [optimizer, setOptimizer] = useState('adam');
  const [scheduler, setScheduler] = useState('none');
  const [amp, setAmp] = useState('none'); // none|fp16|bf16|tf32
  const [gradAccum, setGradAccum] = useState(1);
  const [earlyStop, setEarlyStop] = useState({ enabled: false, patience: 3 });
  const [checkpoint, setCheckpoint] = useState({ strategy: 'best', every_epochs: 1 });
  // Precision + grad clip
  const [precision, setPrecision] = useState(''); // '', fp16, bf16, fp8
  const [gradClip, setGradClip] = useState({ type: 'norm', max_grad_norm: 1.0, max_value: 1.0 });

  // Fine-tune specific
  const [hfModel, setHfModel] = useState('bert-base-uncased');
  const [freezeLayers, setFreezeLayers] = useState(false);
  const [lora, setLora] = useState({ enabled: false, r: 8, alpha: 16, dropout: 0.05, qlora: false, target_modules: ['q_proj','v_proj'], merge_adapters: false, name: '', load_path: '' });
  const [adapters, setAdapters] = useState({ method: 'none' }); // none|prefix|ptuning
  const [quant, setQuant] = useState('none'); // none|int8|int4
  const [fourbitType, setFourbitType] = useState('nf4'); // nf4|fp4
  const [doubleQuant, setDoubleQuant] = useState(true);
  const [computeDtype, setComputeDtype] = useState('bfloat16'); // bfloat16|float16|float32
  const [gradCheckpoint, setGradCheckpoint] = useState(false);

  // Advanced Training Strategies
  const [enableStages, setEnableStages] = useState(false);
  const [stages, setStages] = useState([
    { name: 'warmup', epochs: 1, learning_rate: 5e-5, batch_size: 8, curriculum: { incremental: true, mode: 'length', start_frac: 0.4, end_frac: 0.8 } },
    { name: 'main', epochs: 2, learning_rate: 3e-5, batch_size: 8 },
    { name: 'refinement', epochs: 1, learning_rate: 1e-5, batch_size: 8 }
  ]);

  // Knowledge Distillation
  const [distill, setDistill] = useState({ enabled: false, teacher_model: '', multi_teachers: '', temperature: 2.0, alpha_distill: 0.5, alpha_ce: 0.5 });

  // Distributed
  const [dsPath, setDsPath] = useState('');
  const [fsdp, setFsdp] = useState('');
  const [ddpFindUnused, setDdpFindUnused] = useState(false);
  const [fsdpMinParams, setFsdpMinParams] = useState(0);

  // Context window extension (HF only)
  const [ctxEnabled, setCtxEnabled] = useState(false);
  const [ctxTargetLen, setCtxTargetLen] = useState(8192);
  const [ctxRopeMethod, setCtxRopeMethod] = useState('linear'); // linear|dynamic|yarn
  const [ctxSchedulePreset, setCtxSchedulePreset] = useState('4-8-16-32');
  const [ctxCustomSchedule, setCtxCustomSchedule] = useState('4096,8192,16384,32768');
  const [longloraEnabled, setLongloraEnabled] = useState(false);
  const [longloraShifted, setLongloraShifted] = useState(false);
  const [evalPpl, setEvalPpl] = useState(true);
  const [evalNeedle, setEvalNeedle] = useState(true);
  const [evalLongQA, setEvalLongQA] = useState(false);
  const [evalLengths, setEvalLengths] = useState('4096,8192,16384');
  const [preferCtxScript, setPreferCtxScript] = useState(true);

  // Architecture custom params (simple)
  const [customArch, setCustomArch] = useState({ input_size: 784, output_size: 10, hidden_layers: '512,256,128' });

  useEffect(() => {
    if (gpuMode !== 'select') return;
    if (gpuSelection) return;
    let pick = '';
    const gpus = partitions?.gpus || [];
    for (const g of gpus) {
      for (const inst of (g.instances||[])) {
        if (!(inst.allocated_by_jobs||[]).length) {
          pick = JSON.stringify({type:'mig', gpu_index:g.index, gpu_uuid:g.uuid, mig_uuid:inst.uuid});
          break;
        }
      }
      if (pick) break;
    }
    if (!pick && gpus.length) pick = JSON.stringify({type:'gpu', gpu_index:gpus[0].index, gpu_uuid:gpus[0].uuid});
    setGpuSelection(pick);
  }, [gpuMode, partitions]);

  const presets = useMemo(() => ({
    imagenet: {
      framework: 'pytorch', arch: 'resnet', epochs: 90, batchSize: 256, lr: 0.1,
      data: { source: 'local', augment: { flip: true, normalize: true }, loader: { num_workers: 8, pin_memory: true } }
    },
    bert_cls: {
      framework: 'huggingface', arch: 'transformer', model: 'bert-base-uncased', epochs: 3, batchSize: 16, lr: 2e-5,
      data: { source: 'huggingface', dataset_name: 'imdb' }
    },
    gpt2_ft: {
      framework: 'huggingface', arch: 'transformer', model: 'gpt2', epochs: 3, batchSize: 8, lr: 5e-5,
      data: { source: 'huggingface', dataset_name: 'wikitext' }
    },
    hf_advanced_balanced: {
      framework: 'huggingface', arch: 'transformer', model: 'gpt2', epochs: 4, batchSize: 8, lr: 3e-5,
      advanced: {
        stages: [
          { name: 'warmup', epochs: 1, learning_rate: 5e-5, batch_size: 8, curriculum: { incremental: true, mode: 'length', start_frac: 0.4, end_frac: 0.8 } },
          { name: 'main', epochs: 2, learning_rate: 3e-5, batch_size: 8 },
          { name: 'refinement', epochs: 1, learning_rate: 1e-5, batch_size: 8 }
        ],
        distillation: { enabled: true, teacher_model: 'gpt2-medium', temperature: 2.0, alpha_distill: 0.5, alpha_ce: 0.5 },
        precision: 'bf16',
        grad_clip: { type: 'norm', max_grad_norm: 1.0 }
      }
    }
  }), []);

  const applyPreset = (key) => {
    const p = presets[key];
    if (!p) return;
    setFramework(p.framework);
    setArch(p.arch);
    if (p.model) setHfModel(p.model);
    setEpochs(p.epochs);
    setBatchSize(p.batchSize);
    setLr(p.lr);
    if (p.advanced) {
      setEnableStages(true);
      setStages(p.advanced.stages || stages);
      setDistill(pv => ({...pv, ...(p.advanced.distillation||{}), enabled: !!(p.advanced.distillation?.enabled)}));
      if (p.advanced.precision) setPrecision(p.advanced.precision);
      if (p.advanced.grad_clip) setGradClip(p.advanced.grad_clip);
    }
    if (p.data) {
      if (p.data.source) setDataSource(p.data.source);
      if (p.data.dataset_name) setHfDataset(p.data.dataset_name);
      if (p.data.augment) setAugment(a => ({...a, ...p.data.augment}));
      if (p.data.loader) setLoader(l => ({...l, ...p.data.loader}));
    }
    setPreset(key);
  };

  const next = () => setStep(s => Math.min(4, s+1));
  const back = () => setStep(s => Math.max(0, s-1));

  const buildConfig = () => {
    const cfg = {
      architecture: arch,
      epochs, batch_size: batchSize, learning_rate: lr,
      optimizer, scheduler,
      mixed_precision: amp,
      precision: precision || undefined,
      gradient_accumulation_steps: gradAccum,
      early_stopping: earlyStop,
      checkpoint: checkpoint,
      data: {
        source: dataSource,
        local_path: dataSource==='local'?localPath:undefined,
        dataset_name: dataSource==='huggingface'?hfDataset:undefined,
        s3_uri: dataSource==='s3'?s3Uri:undefined,
        gcs_uri: dataSource==='gcs'?gcsUri:undefined,
        split,
        augment,
        loader,
      },
    };
    if (framework === 'huggingface') {
      cfg.model_name = hfModel;
      cfg.freeze_layers = freezeLayers;
      cfg.lora = lora;
      cfg.adapters = adapters;
      cfg.quantization = quant;
      if (quant === 'int4' || lora.qlora) {
        cfg.fourbit_quant_type = fourbitType;
        cfg.double_quant = doubleQuant;
        cfg.compute_dtype = computeDtype;
      }
      cfg.gradient_checkpointing = gradCheckpoint;
      if (enableStages) cfg.stages = stages.map(s => ({
        name: s.name,
        epochs: parseInt(s.epochs)||1,
        learning_rate: parseFloat(s.learning_rate)||lr,
        batch_size: parseInt(s.batch_size)||batchSize,
        ...(s.curriculum ? { curriculum: {
          incremental: !!s.curriculum.incremental,
          mode: s.curriculum.mode || 'length',
          start_frac: parseFloat(s.curriculum.start_frac ?? 0.5),
          end_frac: parseFloat(s.curriculum.end_frac ?? 1.0)
        }} : {})
      }));
      if (distill.enabled) {
        cfg.distillation = {
          enabled: true,
          teacher_model: distill.teacher_model || undefined,
          multi_teachers: (distill.multi_teachers||'').split(',').map(s=>s.trim()).filter(Boolean),
          temperature: parseFloat(distill.temperature)||2.0,
          alpha_distill: parseFloat(distill.alpha_distill)||0.5,
          alpha_ce: parseFloat(distill.alpha_ce)||0.5,
        };
      }
      cfg.grad_clip = gradClip;
      // Distributed opts
      const dist = {};
      if (dsPath) dist.deepspeed = dsPath;
      if (fsdp) dist.fsdp = fsdp;
      if (ddpFindUnused) dist.ddp_find_unused_parameters = true;
      if (fsdpMinParams) dist.fsdp_min_num_params = parseInt(fsdpMinParams);
      if (Object.keys(dist).length) cfg.distributed = dist;
      if (ctxEnabled) {
        const sched = ctxSchedulePreset==='custom' ? ctxCustomSchedule.split(',').map(s=>parseInt(s.trim())).filter(Boolean) : (ctxSchedulePreset==='8-16-32-64' ? [8192,16384,32768,65536] : [4096,8192,16384,32768]);
        cfg.context_extension = {
          enabled: true,
          rope: { method: ctxRopeMethod, target_length: Math.max(2048, Math.min(131072, parseInt(ctxTargetLen)||8192)) },
          longlora: { enabled: longloraEnabled, shifted_sparse_attention: longloraShifted },
          route: preferCtxScript,
          schedule: sched,
          eval: { run_ppl: evalPpl, run_needle: evalNeedle, run_long_qa: evalLongQA, lengths: evalLengths.split(',').map(s=>parseInt(s.trim())).filter(Boolean) }
        };
      }
    }
    if (framework === 'pytorch' && arch === 'custom') {
      cfg.input_size = customArch.input_size;
      cfg.output_size = customArch.output_size;
      cfg.hidden_layers = customArch.hidden_layers.split(',').map(s=>parseInt(s.trim())).filter(Boolean);
    }
    return cfg;
  };

  const onSubmit = async () => {
    setErrors([]); setWarnings([]); setValidating(true);
    const payload = {
      name: name || undefined,
      type: jobType,
      framework,
      config: buildConfig(),
      ...(gpuMode==='auto' ? { gpu_prefer: gpuAutoPrefer } : {}),
      ...(gpuMode==='select' && gpuSelection ? { gpu: JSON.parse(gpuSelection) } : {}),
    };
    try {
      const v = await api.validateJob(payload);
      if (!v.ok) {
        setErrors(v.errors||[]); setWarnings(v.warnings||[]); setValidating(false);
        return;
      }
      const res = await api.createJob(payload);
      setValidating(false);
      alert(`Job created: ${res.id}`);
      onNavigate('jobs');
    } catch (e) {
      setValidating(false);
      setErrors([String(e.message||e)]);
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-bold">New Training Job (Wizard)</h1>
        <button onClick={()=>onNavigate('dashboard')} className="text-primary hover:brightness-110">← Dashboard</button>
      </div>

      {/* Stepper */}
      <div className="flex items-center gap-3 text-sm">
        {['Framework','Architecture','Data','Training','Review'].map((t,i)=>(
          <div key={i} className={`px-3 py-1 rounded ${step===i?'bg-primary text-on-primary':'bg-muted'}`}>{i+1}. {t}</div>
        ))}
      </div>

      {/* Step content */}
      {step===0 && (
        <div className="bg-surface p-6 rounded-lg shadow-md border border-border space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <label className="block text-sm font-semibold mb-2">Job Type</label>
              <select value={jobType} onChange={e=>setJobType(e.target.value)} className="w-full border border-border rounded px-3 py-2 bg-surface">
                <option value="train">Train from scratch</option>
                <option value="finetune">Fine-tune</option>
              </select>
            </div>
            <div>
              <label className="block text-sm font-semibold mb-2">Framework</label>
              <select value={framework} onChange={e=>setFramework(e.target.value)} className="w-full border border-border rounded px-3 py-2 bg-surface">
                <option value="pytorch">PyTorch</option>
                <option value="huggingface">Hugging Face</option>
                <option value="tensorflow">TensorFlow</option>
              </select>
            </div>
          </div>

          <div>
            <label className="block text-sm font-semibold mb-2">Preset Templates</label>
            <div className="flex flex-wrap gap-2">
              <button onClick={()=>applyPreset('imagenet')} className="px-3 py-2 border border-border rounded bg-surface hover:bg-muted">ImageNet (ResNet)</button>
              <button onClick={()=>applyPreset('bert_cls')} className="px-3 py-2 border border-border rounded bg-surface hover:bg-muted">BERT Fine-tune</button>
              <button onClick={()=>applyPreset('gpt2_ft')} className="px-3 py-2 border border-border rounded bg-surface hover:bg-muted">GPT-2 Fine-tune</button>
              <button onClick={()=>applyPreset('hf_advanced_balanced')} className="px-3 py-2 border border-border rounded bg-surface hover:bg-muted">HF Advanced (Curriculum + KD)</button>
            </div>
          </div>

          <div className="flex justify-end gap-2"><button onClick={next} className="px-4 py-2 bg-primary text-on-primary rounded">Next</button></div>
        </div>
      )}

      {step===1 && (
        <div className="bg-surface p-6 rounded-lg shadow-md border border-border space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {['custom','resnet','transformer'].map(a => (
              <button key={a} onClick={()=>setArch(a)} className={`p-4 border border-border rounded ${arch===a?'ring-2 ring-primary/40':''}`}>
                <div className="font-semibold capitalize mb-2">{a}</div>
                {/* Simple diagram */}
                {a==='custom' && <div className="h-24 bg-gradient-to-b from-gray-100 to-gray-200 rounded flex items-center justify-center">MLP</div>}
                {a==='resnet' && <div className="h-24 bg-gradient-to-b from-gray-100 to-gray-200 rounded flex items-center justify-center">Conv ↔ Residual</div>}
                {a==='transformer' && <div className="h-24 bg-gradient-to-b from-gray-100 to-gray-200 rounded flex items-center justify-center">Attention Blocks</div>}
              </button>
            ))}
          </div>

          {framework==='pytorch' && arch==='custom' && (
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div><label className="text-sm">Input Size</label><input type="number" className="w-full border rounded px-3 py-2" value={customArch.input_size} onChange={e=>setCustomArch({...customArch, input_size: parseInt(e.target.value)})}/></div>
              <div><label className="text-sm">Output Size</label><input type="number" className="w-full border rounded px-3 py-2" value={customArch.output_size} onChange={e=>setCustomArch({...customArch, output_size: parseInt(e.target.value)})}/></div>
              <div><label className="text-sm">Hidden Layers</label><input type="text" className="w-full border rounded px-3 py-2" value={customArch.hidden_layers} onChange={e=>setCustomArch({...customArch, hidden_layers: e.target.value})}/></div>
            </div>
          )}

          {framework==='huggingface' && (
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="md:col-span-2"><label className="text-sm">Base Model</label><input className="w-full border rounded px-3 py-2" value={hfModel} onChange={e=>setHfModel(e.target.value)} placeholder="bert-base-uncased, gpt2, ..."/></div>
              <div className="flex items-end"><label className="inline-flex items-center gap-2"><input type="checkbox" checked={freezeLayers} onChange={e=>setFreezeLayers(e.target.checked)}/> Freeze Layers</label></div>
            </div>
          )}

          <div className="flex justify-between">
            <button onClick={back} className="px-4 py-2 border rounded">Back</button>
            <button onClick={next} className="px-4 py-2 bg-blue-600 text-white rounded">Next</button>
          </div>
        </div>
      )}

      {step===2 && (
        <div className="bg-surface p-6 rounded-lg shadow-md border border-border space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div>
              <label className="block text-sm font-semibold mb-2">Data Source</label>
              <select className="w-full border border-border rounded px-3 py-2 bg-surface" value={dataSource} onChange={e=>setDataSource(e.target.value)}>
                <option value="local">Local Path</option>
                <option value="huggingface">Hugging Face Dataset</option>
                <option value="s3">S3</option>
                <option value="gcs">GCS</option>
              </select>
            </div>
            {dataSource==='local' && (
              <div className="md:col-span-2">
                <label className="text-sm">Local Path (server-visible)</label>
                <input className="w-full border rounded px-3 py-2" value={localPath} onChange={e=>setLocalPath(e.target.value)} placeholder="./data/imagenet"/>
                <div className="mt-2 text-xs text-text/70">Optional preview from browser: <input type="file" directory="" webkitdirectory="" multiple onChange={e=>setLocalFiles(Array.from(e.target.files||[]).map(f=>f.name).slice(0,5))} /></div>
                {localFiles.length>0 && (<div className="text-xs text-text/70 mt-1">Preview: {localFiles.join(', ')}</div>)}
              </div>
            )}
            {dataSource==='huggingface' && (
              <div className="md:col-span-2"><label className="text-sm">Dataset Name</label><input className="w-full border border-border rounded px-3 py-2 bg-surface" value={hfDataset} onChange={e=>setHfDataset(e.target.value)} placeholder="imdb, ag_news, ..."/></div>
            )}
            {dataSource==='s3' && (
              <div className="md:col-span-2"><label className="text-sm">S3 URI</label><input className="w-full border border-border rounded px-3 py-2 bg-surface" value={s3Uri} onChange={e=>setS3Uri(e.target.value)} placeholder="s3://bucket/prefix"/></div>
            )}
            {dataSource==='gcs' && (
              <div className="md:col-span-2"><label className="text-sm">GCS URI</label><input className="w-full border border-border rounded px-3 py-2 bg-surface" value={gcsUri} onChange={e=>setGcsUri(e.target.value)} placeholder="gs://bucket/prefix"/></div>
            )}
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div>
              <label className="text-sm">Split (train/val/test)</label>
              <div className="flex gap-2 mt-1">
                <input type="number" step="0.01" className="w-full border border-border rounded px-2 py-1 bg-surface" value={split.train} onChange={e=>setSplit({...split, train: parseFloat(e.target.value)})}/>
                <input type="number" step="0.01" className="w-full border border-border rounded px-2 py-1 bg-surface" value={split.val} onChange={e=>setSplit({...split, val: parseFloat(e.target.value)})}/>
                <input type="number" step="0.01" className="w-full border border-border rounded px-2 py-1 bg-surface" value={split.test} onChange={e=>setSplit({...split, test: parseFloat(e.target.value)})}/>
              </div>
            </div>
            <div>
              <label className="text-sm">Augmentations</label>
              <div className="grid grid-cols-2 gap-2 mt-1">
                <label className="inline-flex items-center gap-2"><input type="checkbox" checked={augment.flip} onChange={e=>setAugment({...augment, flip: e.target.checked})}/> Flip</label>
                <label className="inline-flex items-center gap-2"><input type="checkbox" checked={augment.rotate} onChange={e=>setAugment({...augment, rotate: e.target.checked})}/> Rotate</label>
                <label className="inline-flex items-center gap-2"><input type="checkbox" checked={augment.normalize} onChange={e=>setAugment({...augment, normalize: e.target.checked})}/> Normalize</label>
              </div>
            </div>
            <div>
              <label className="text-sm">Data Loader</label>
              <div className="grid grid-cols-2 gap-2 mt-1">
                <label className="text-xs">Workers<input type="number" className="w-full border border-border rounded px-2 py-1 bg-surface" value={loader.num_workers} onChange={e=>setLoader({...loader, num_workers: parseInt(e.target.value)})}/></label>
                <label className="text-xs">Prefetch<input type="number" className="w-full border border-border rounded px-2 py-1 bg-surface" value={loader.prefetch} onChange={e=>setLoader({...loader, prefetch: parseInt(e.target.value)})}/></label>
                <label className="inline-flex items-center gap-2 col-span-2"><input type="checkbox" checked={loader.pin_memory} onChange={e=>setLoader({...loader, pin_memory: e.target.checked})}/> Pin Memory</label>
              </div>
            </div>
          </div>

          <div className="flex justify-between">
            <button onClick={back} className="px-4 py-2 border border-border rounded bg-surface hover:bg-muted">Back</button>
            <button onClick={next} className="px-4 py-2 bg-primary text-on-primary rounded">Next</button>
          </div>
        </div>
      )}

      {step===3 && (
        <div className="bg-surface p-6 rounded-lg shadow-md border border-border space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div><label className="text-sm">Epochs</label><input type="number" className="w-full border rounded px-3 py-2" value={epochs} onChange={e=>setEpochs(parseInt(e.target.value))}/></div>
            <div><label className="text-sm">Batch Size</label><input type="number" className="w-full border rounded px-3 py-2" value={batchSize} onChange={e=>setBatchSize(parseInt(e.target.value))}/></div>
            <div><label className="text-sm">Learning Rate</label><input type="number" step="0.00001" className="w-full border rounded px-3 py-2" value={lr} onChange={e=>setLr(parseFloat(e.target.value))}/></div>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div>
              <label className="text-sm">Optimizer</label>
              <select className="w-full border rounded px-3 py-2" value={optimizer} onChange={e=>setOptimizer(e.target.value)}>
                <option value="adam">Adam</option>
                <option value="adamw">AdamW</option>
                <option value="sgd">SGD (momentum)</option>
                <option value="lamb">LAMB</option>
              </select>
            </div>
            <div>
              <label className="text-sm">Scheduler</label>
              <select className="w-full border rounded px-3 py-2" value={scheduler} onChange={e=>setScheduler(e.target.value)}>
                <option value="none">None</option>
                <option value="steplr">StepLR</option>
                <option value="cosine">CosineAnnealing</option>
                <option value="onecycle">OneCycle</option>
              </select>
            </div>
            <div>
              <label className="text-sm">Mixed Precision</label>
              <select className="w-full border rounded px-3 py-2" value={amp} onChange={e=>setAmp(e.target.value)}>
                <option value="none">None</option>
                <option value="fp16">FP16</option>
                <option value="bf16">BF16</option>
                <option value="tf32">TF32</option>
              </select>
            </div>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div><label className="text-sm">Grad Accum</label><input type="number" className="w-full border rounded px-3 py-2" value={gradAccum} onChange={e=>setGradAccum(parseInt(e.target.value))}/></div>
            <div className="flex items-end"><label className="inline-flex items-center gap-2"><input type="checkbox" checked={earlyStop.enabled} onChange={e=>setEarlyStop({...earlyStop, enabled: e.target.checked})}/> Early Stopping</label></div>
            <div><label className="text-sm">Patience</label><input type="number" className="w-full border rounded px-3 py-2" value={earlyStop.patience} onChange={e=>setEarlyStop({...earlyStop, patience: parseInt(e.target.value)})}/></div>
          </div>

          <details className="mt-2">
            <summary className="cursor-pointer text-sm font-semibold">Advanced Options</summary>
            <div className="mt-2 grid grid-cols-1 md:grid-cols-3 gap-4">
              <div>
                <label className="text-sm">Checkpoint Strategy</label>
                <select className="w-full border rounded px-3 py-2" value={checkpoint.strategy} onChange={e=>setCheckpoint({...checkpoint, strategy: e.target.value})}>
                  <option value="best">Best only</option>
                  <option value="every_n">Every N epochs</option>
                </select>
              </div>
              {checkpoint.strategy==='every_n' && (
                <div><label className="text-sm">Every N epochs</label><input type="number" className="w-full border rounded px-3 py-2" value={checkpoint.every_epochs} onChange={e=>setCheckpoint({...checkpoint, every_epochs: parseInt(e.target.value)})}/></div>
              )}
              {framework==='huggingface' && (
                <>
                  <div className="md:col-span-3 space-y-3">
                    <div className="flex items-center justify-between">
                      <label className="inline-flex items-center gap-2"><input type="checkbox" checked={lora.enabled} onChange={e=>setLora({...lora, enabled: e.target.checked})}/> Enable LoRA</label>
                      <div className="flex gap-2 text-xs">
                        <button type="button" className="px-2 py-1 border border-border rounded" onClick={()=>setLora(l=>({...l, enabled:true, r:8, alpha:16}))}>Quick LoRA</button>
                        <button type="button" className="px-2 py-1 border border-border rounded" onClick={()=>setLora(l=>({...l, enabled:true, r:16, alpha:32}))}>Balanced</button>
                        <button type="button" className="px-2 py-1 border border-border rounded" onClick={()=>setLora(l=>({...l, enabled:true, r:64, alpha:128}))}>Full</button>
                      </div>
                    </div>
                    {lora.enabled && (
                      <div className="grid grid-cols-1 md:grid-cols-4 gap-3">
                        <div>
                          <label className="text-xs">LoRA rank (r): {lora.r}</label>
                          <input type="range" min="1" max="256" value={lora.r} onChange={e=>setLora({...lora, r: parseInt(e.target.value)})} className="w-full" />
                        </div>
                        <label className="text-xs">alpha<input type="number" className="w-full border rounded px-2 py-1" value={lora.alpha} onChange={e=>setLora({...lora, alpha: parseInt(e.target.value)})}/></label>
                        <label className="text-xs">dropout<input type="number" step="0.01" className="w-full border rounded px-2 py-1" value={lora.dropout} onChange={e=>setLora({...lora, dropout: parseFloat(e.target.value)})}/></label>
                        <div className="flex items-end"><label className="inline-flex items-center gap-2"><input type="checkbox" checked={lora.qlora} onChange={e=>setLora({...lora, qlora: e.target.checked})}/> QLoRA (4-bit)</label></div>
                        <div className="md:col-span-4">
                          <label className="text-xs block mb-1">Target Modules</label>
                          <div className="flex flex-wrap gap-2 text-xs">
                            {['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj','lm_head'].map(m => (
                              <label key={m} className={`px-2 py-1 rounded border cursor-pointer ${lora.target_modules?.includes(m)?'bg-primary/10 border-primary text-primary':'border-border'}`}>
                                <input type="checkbox" className="hidden" checked={!!(lora.target_modules||[]).includes(m)} onChange={(e)=>{
                                  setLora(l=> ({...l, target_modules: e.target.checked ? Array.from(new Set([...(l.target_modules||[]), m])) : (l.target_modules||[]).filter(x=>x!==m) }));
                                }}/>
                                {m}
                              </label>
                            ))}
                          </div>
                          <div className="text-[11px] text-text/60 mt-1">Tip: For LLaMA/LLM families use q_proj and v_proj; for GPT‑2 often q_proj,v_proj; for BERT tasks down_proj/up_proj sometimes apply depending on head. You can start with q_proj+v_proj (Quick LoRA) and iterate.</div>
                        </div>
                        <div className="md:col-span-2"><label className="text-xs">Adapter Name (for saving)</label><input className="w-full border rounded px-2 py-1" value={lora.name} onChange={e=>setLora({...lora, name: e.target.value})} placeholder="my-adapter"/></div>
                        <div className="md:col-span-2"><label className="text-xs">Load Existing Adapter (server path)</label><input className="w-full border rounded px-2 py-1" value={lora.load_path} onChange={e=>setLora({...lora, load_path: e.target.value})} placeholder="./models/<id>/adapters/<name>"/></div>
                        <div className="md:col-span-2 flex items-end"><label className="inline-flex items-center gap-2"><input type="checkbox" checked={lora.merge_adapters} onChange={e=>setLora({...lora, merge_adapters: e.target.checked})}/> Merge adapter into base model after training</label></div>
                      </div>
                    )}
                  </div>
                  <div>
                    <label className="text-sm">Adapter Method</label>
                    <select className="w-full border rounded px-3 py-2" value={adapters.method} onChange={e=>setAdapters({method: e.target.value})}>
                      <option value="none">None</option>
                      <option value="prefix">Prefix Tuning</option>
                      <option value="ptuning">P-Tuning</option>
                    </select>
                  </div>
                  <div>
                    <label className="text-sm">Quantization</label>
                    <select className="w-full border rounded px-3 py-2" value={quant} onChange={e=>setQuant(e.target.value)}>
                      <option value="none">None</option>
                      <option value="int8">8-bit</option>
                      <option value="int4">4-bit</option>
                    </select>
                    {(quant==='int4' || lora.qlora) && (
                      <div className="grid grid-cols-1 gap-2 mt-2 text-xs">
                        <label>4-bit type
                          <select className="w-full border rounded px-2 py-1" value={fourbitType} onChange={e=>setFourbitType(e.target.value)}>
                            <option value="nf4">NF4</option>
                            <option value="fp4">FP4</option>
                          </select>
                        </label>
                        <label className="inline-flex items-center gap-2"><input type="checkbox" checked={doubleQuant} onChange={e=>setDoubleQuant(e.target.checked)}/> Double quantization</label>
                        <label>Compute dtype
                          <select className="w-full border rounded px-2 py-1" value={computeDtype} onChange={e=>setComputeDtype(e.target.value)}>
                            <option value="bfloat16">bfloat16</option>
                            <option value="float16">float16</option>
                            <option value="float32">float32</option>
                          </select>
                        </label>
                      </div>
                    )}
                  </div>
                  <div className="flex items-end"><label className="inline-flex items-center gap-2"><input type="checkbox" checked={gradCheckpoint} onChange={e=>setGradCheckpoint(e.target.checked)}/> Gradient Checkpointing</label></div>
              </>
          )}
          </div>
        </details>

        {/* Advanced Training Strategies */}
        {framework==='huggingface' && (
          <details className="mt-2">
            <summary className="cursor-pointer text-sm font-semibold">Advanced Training Strategies</summary>
            <div className="mt-2 grid grid-cols-1 gap-4">
              <div className="flex items-end"><label className="inline-flex items-center gap-2"><input type="checkbox" checked={enableStages} onChange={e=>setEnableStages(e.target.checked)}/> Enable Multi‑stage Pipeline</label></div>
              {enableStages && (
                <div className="space-y-2">
                  {stages.map((s, idx)=> (
                    <div key={idx} className="border border-border rounded p-3">
                      <div className="grid grid-cols-1 md:grid-cols-5 gap-2">
                        <label className="text-xs">Name<input className="w-full border rounded px-2 py-1" value={s.name} onChange={e=>{const a=[...stages]; a[idx]={...a[idx], name:e.target.value}; setStages(a);}}/></label>
                        <label className="text-xs">Epochs<input type="number" className="w-full border rounded px-2 py-1" value={s.epochs} onChange={e=>{const a=[...stages]; a[idx]={...a[idx], epochs: parseInt(e.target.value)||1}; setStages(a);}}/></label>
                        <label className="text-xs">LR<input type="number" step="0.000001" className="w-full border rounded px-2 py-1" value={s.learning_rate} onChange={e=>{const a=[...stages]; a[idx]={...a[idx], learning_rate: parseFloat(e.target.value)||0}; setStages(a);}}/></label>
                        <label className="text-xs">Batch<input type="number" className="w-full border rounded px-2 py-1" value={s.batch_size} onChange={e=>{const a=[...stages]; a[idx]={...a[idx], batch_size: parseInt(e.target.value)||1}; setStages(a);}}/></label>
                        <div className="flex items-end justify-end gap-2">
                          <button type="button" className="text-xs px-2 py-1 border rounded" onClick={()=>{
                            const a=[...stages]; a.splice(idx,1); setStages(a);
                          }}>Remove</button>
                        </div>
                      </div>
                      <div className="mt-2 grid grid-cols-1 md:grid-cols-5 gap-2 text-xs">
                        <div className="md:col-span-5 font-semibold">Curriculum</div>
                        <label className="inline-flex items-center gap-2"><input type="checkbox" checked={!!(s.curriculum?.incremental)} onChange={e=>{const a=[...stages]; a[idx]={...a[idx], curriculum:{...(s.curriculum||{}), incremental:e.target.checked}}; setStages(a);}}/> Incremental</label>
                        <label>Mode
                          <select className="w-full border rounded px-2 py-1" value={s.curriculum?.mode||'length'} onChange={e=>{const a=[...stages]; a[idx]={...a[idx], curriculum:{...(s.curriculum||{}), mode:e.target.value}}; setStages(a);}}>
                            <option value="length">Length (easy→hard)</option>
                            <option value="random">Random</option>
                          </select>
                        </label>
                        <label>Start frac<input type="number" step="0.05" className="w-full border rounded px-2 py-1" value={s.curriculum?.start_frac ?? 0.5} onChange={e=>{const a=[...stages]; a[idx]={...a[idx], curriculum:{...(s.curriculum||{}), start_frac: parseFloat(e.target.value)||0.5}}; setStages(a);}}/></label>
                        <label>End frac<input type="number" step="0.05" className="w-full border rounded px-2 py-1" value={s.curriculum?.end_frac ?? 1.0} onChange={e=>{const a=[...stages]; a[idx]={...a[idx], curriculum:{...(s.curriculum||{}), end_frac: parseFloat(e.target.value)||1.0}}; setStages(a);}}/></label>
                      </div>
                    </div>
                  ))}
                  <div className="flex items-center gap-2">
                    <button type="button" className="px-2 py-1 border rounded" onClick={()=>setStages([...stages, { name: `stage${stages.length+1}`, epochs: 1, learning_rate: lr, batch_size: batchSize }])}>Add Stage</button>
                    <div className="text-xs text-text/70">Stages map to context schedule when Context Extension is enabled; otherwise they run sequentially.</div>
                  </div>
                </div>
              )}

              <div className="border-t border-border pt-2">
                <div className="text-sm font-semibold mb-1">Knowledge Distillation</div>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-2 text-sm">
                  <label className="inline-flex items-center gap-2"><input type="checkbox" checked={distill.enabled} onChange={e=>setDistill({...distill, enabled:e.target.checked})}/> Enable KD</label>
                  <div className="md:col-span-2"><label className="text-xs">Teacher model (HF id or path)</label><input className="w-full border rounded px-2 py-1" value={distill.teacher_model} onChange={e=>setDistill({...distill, teacher_model:e.target.value})} placeholder="gpt2-medium"/></div>
                  <div className="md:col-span-3"><label className="text-xs">Multi-teachers (comma‑separated)</label><input className="w-full border rounded px-2 py-1" value={distill.multi_teachers} onChange={e=>setDistill({...distill, multi_teachers:e.target.value})} placeholder="modelA, modelB"/></div>
                  <label className="text-xs">Temperature<input type="number" step="0.1" className="w-full border rounded px-2 py-1" value={distill.temperature} onChange={e=>setDistill({...distill, temperature: parseFloat(e.target.value)||2.0})}/></label>
                  <label className="text-xs">Alpha KD<input type="number" step="0.05" className="w-full border rounded px-2 py-1" value={distill.alpha_distill} onChange={e=>setDistill({...distill, alpha_distill: parseFloat(e.target.value)||0.5})}/></label>
                  <label className="text-xs">Alpha CE<input type="number" step="0.05" className="w-full border rounded px-2 py-1" value={distill.alpha_ce} onChange={e=>setDistill({...distill, alpha_ce: parseFloat(e.target.value)||0.5})}/></label>
                </div>
              </div>
            </div>
          </details>
        )}

        {/* Precision & Distributed */}
        {framework==='huggingface' && (
          <details className="mt-2">
            <summary className="cursor-pointer text-sm font-semibold">Precision & Distributed</summary>
            <div className="mt-2 grid grid-cols-1 md:grid-cols-3 gap-4">
              <div>
                <label className="text-sm">Precision</label>
                <select className="w-full border rounded px-3 py-2" value={precision} onChange={e=>setPrecision(e.target.value)}>
                  <option value="">Default</option>
                  <option value="fp16">FP16</option>
                  <option value="bf16">BF16</option>
                  <option value="fp8">FP8 (env support required)</option>
                </select>
              </div>
              <div>
                <label className="text-sm">Grad Clip</label>
                <div className="grid grid-cols-2 gap-2 text-xs mt-1">
                  <select className="w-full border rounded px-2 py-1" value={gradClip.type} onChange={e=>setGradClip({...gradClip, type:e.target.value})}>
                    <option value="norm">By Norm</option>
                    <option value="value">By Value</option>
                  </select>
                  {gradClip.type==='norm' ? (
                    <input type="number" step="0.1" className="w-full border rounded px-2 py-1" value={gradClip.max_grad_norm} onChange={e=>setGradClip({...gradClip, max_grad_norm: parseFloat(e.target.value)||1.0})}/>
                  ) : (
                    <input type="number" step="0.1" className="w-full border rounded px-2 py-1" value={gradClip.max_value} onChange={e=>setGradClip({...gradClip, max_value: parseFloat(e.target.value)||1.0})}/>
                  )}
                </div>
              </div>
              <div>
                <label className="text-sm">Grad Accumulation</label>
                <input type="number" className="w-full border rounded px-3 py-2" value={gradAccum} onChange={e=>setGradAccum(parseInt(e.target.value)||1)}/>
              </div>
              <div className="md:col-span-3 border-t border-border pt-2">
                <div className="text-sm font-semibold mb-1">Distributed</div>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-3 text-sm">
                  <div className="md:col-span-2"><label className="text-xs">DeepSpeed config path</label><input className="w-full border rounded px-2 py-1" value={dsPath} onChange={e=>setDsPath(e.target.value)} placeholder="training_scripts/deepspeed/zero2.json"/></div>
                  <div className="flex items-end"><button type="button" className="px-2 py-1 border rounded" onClick={()=>setDsPath('training_scripts/deepspeed/zero2.json')}>Use ZeRO‑2 template</button></div>
                  <div className="md:col-span-2"><label className="text-xs">FSDP config string</label><input className="w-full border rounded px-2 py-1" value={fsdp} onChange={e=>setFsdp(e.target.value)} placeholder="full_shard auto_wrap"/></div>
                  <label className="inline-flex items-center gap-2"><input type="checkbox" checked={ddpFindUnused} onChange={e=>setDdpFindUnused(e.target.checked)}/> DDP find_unused_parameters</label>
                  <label className="text-xs">FSDP min params<input type="number" className="w-full border rounded px-2 py-1" value={fsdpMinParams} onChange={e=>setFsdpMinParams(parseInt(e.target.value)||0)}/></label>
                </div>
                <div className="text-[11px] text-text/60 mt-1">Note: Multi‑node/multi‑GPU usually requires launching with torchrun externally.</div>
              </div>
            </div>
          </details>
        )}

          {framework==='huggingface' && (
            <details className="mt-2">
              <summary className="cursor-pointer text-sm font-semibold">Context Window Extension</summary>
              <div className="mt-2 grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="flex items-end"><label className="inline-flex items-center gap-2"><input type="checkbox" checked={ctxEnabled} onChange={e=>setCtxEnabled(e.target.checked)}/> Enable Context Extension</label></div>
              <div className="flex items-end"><label className="inline-flex items-center gap-2"><input type="checkbox" checked={preferCtxScript} onChange={e=>setPreferCtxScript(e.target.checked)} disabled={!ctxEnabled}/> Prefer context extension training path</label></div>
                <div><label className="text-sm">Target Context Length</label><input type="number" min={2048} max={131072} className="w-full border rounded px-3 py-2" value={ctxTargetLen} onChange={e=>setCtxTargetLen(parseInt(e.target.value)||8192)} /></div>
                <div><label className="text-sm">RoPE Scaling</label><select className="w-full border rounded px-3 py-2" value={ctxRopeMethod} onChange={e=>setCtxRopeMethod(e.target.value)}><option value="linear">Linear</option><option value="dynamic">Dynamic</option><option value="yarn">YaRN</option></select></div>
                <div><label className="text-sm">Schedule</label><select className="w-full border rounded px-3 py-2" value={ctxSchedulePreset} onChange={e=>setCtxSchedulePreset(e.target.value)}><option value="4-8-16-32">4k→8k→16k→32k</option><option value="8-16-32-64">8k→16k→32k→64k</option><option value="custom">Custom</option></select></div>
                {ctxSchedulePreset==='custom' && (<div className="md:col-span-2"><label className="text-sm">Custom schedule (comma‑sep tokens)</label><input className="w-full border rounded px-3 py-2" value={ctxCustomSchedule} onChange={e=>setCtxCustomSchedule(e.target.value)} placeholder="4096,8192,16384,32768"/></div>)}
                <div className="flex items-end"><label className="inline-flex items-center gap-2"><input type="checkbox" checked={longloraEnabled} onChange={e=>setLongloraEnabled(e.target.checked)}/> Enable LongLoRA</label></div>
                <div className="flex items-end"><label className="inline-flex items-center gap-2"><input type="checkbox" checked={longloraShifted} onChange={e=>setLongloraShifted(e.target.checked)}/> Shifted sparse attention</label></div>
                <div className="md:col-span-3 border-t border-border pt-2">
                  <div className="text-sm font-semibold mb-1">Long‑context Evaluation</div>
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-3 text-sm">
                    <label className="inline-flex items-center gap-2"><input type="checkbox" checked={evalPpl} onChange={e=>setEvalPpl(e.target.checked)}/> Perplexity</label>
                    <label className="inline-flex items-center gap-2"><input type="checkbox" checked={evalNeedle} onChange={e=>setEvalNeedle(e.target.checked)}/> Needle‑in‑haystack</label>
                    <label className="inline-flex items-center gap-2"><input type="checkbox" checked={evalLongQA} onChange={e=>setEvalLongQA(e.target.checked)}/> Long QA</label>
                    <div className="md:col-span-2"><label className="text-xs">Eval lengths</label><input className="w-full border rounded px-2 py-1" value={evalLengths} onChange={e=>setEvalLengths(e.target.value)} placeholder="4096,8192,16384"/></div>
                  </div>
                </div>
              </div>
            </details>
          )}

          <div className="flex justify-between">
            <button onClick={back} className="px-4 py-2 border rounded">Back</button>
            <button onClick={next} className="px-4 py-2 bg-blue-600 text-white rounded">Next</button>
          </div>
        </div>
      )}

      {step===4 && (
        <div className="bg-surface p-6 rounded-lg shadow-md border border-border space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="text-sm">Job Name</label>
              <input className="w-full border border-border rounded px-3 py-2 bg-surface" value={name} onChange={e=>setName(e.target.value)} placeholder="My Training Job"/>
            </div>
            <div>
              <label className="text-sm">Compute</label>
              <div className="flex gap-3">
                <label className="inline-flex items-center gap-2"><input type="radio" name="gw_gpu" checked={gpuMode==='auto'} onChange={()=>setGpuMode('auto')}/> Auto</label>
                <label className="inline-flex items-center gap-2"><input type="radio" name="gw_gpu" checked={gpuMode==='select'} onChange={()=>setGpuMode('select')}/> Select</label>
              </div>
              {gpuMode==='auto' && (
                <div className="flex gap-3 mt-2">
                  <label className="inline-flex items-center gap-2"><input type="radio" name="gw_pref" checked={gpuAutoPrefer==='mig_first'} onChange={()=>setGpuAutoPrefer('mig_first')}/> MIG first</label>
                  <label className="inline-flex items-center gap-2"><input type="radio" name="gw_pref" checked={gpuAutoPrefer==='gpu_first'} onChange={()=>setGpuAutoPrefer('gpu_first')}/> GPU first</label>
                </div>
              )}
              {gpuMode==='select' && (
                <select className="w-full border border-border rounded px-3 py-2 mt-2 bg-surface" value={gpuSelection} onChange={e=>setGpuSelection(e.target.value)}>
                  <option value="">Choose...</option>
                  {(partitions?.gpus || []).map((g) => (
                    <option key={`g-${g.index}`} value={JSON.stringify({type:'gpu', gpu_index:g.index, gpu_uuid:g.uuid})}>{`GPU ${g.index} — ${g.name}`}</option>
                  ))}
                  {(partitions?.gpus || []).flatMap(g => (g.instances||[]).map((inst, k) => (
                    <option key={`m-${g.index}-${k}`} disabled={(inst.allocated_by_jobs||[]).length>0} value={JSON.stringify({type:'mig', gpu_index:g.index, gpu_uuid:g.uuid, mig_uuid:inst.uuid})}>{`GPU ${g.index} — ${inst.profile} (MIG ${inst.device_id})${(inst.allocated_by_jobs||[]).length>0 ? ' — Allocated' : ''}`}</option>
                  )))}
                </select>
              )}
            </div>
          </div>

          <div>
            <label className="text-sm font-semibold">Summary</label>
            <pre className="bg-muted p-4 rounded text-xs overflow-auto">{JSON.stringify({ name, type: jobType, framework, config: buildConfig(), gpu: (gpuMode==='select' && gpuSelection ? JSON.parse(gpuSelection) : undefined), gpu_prefer: gpuMode==='auto'?gpuAutoPrefer:undefined }, null, 2)}</pre>
          </div>

          {warnings.length>0 && (
            <div className="p-3 bg-yellow-50 border border-yellow-200 rounded text-sm">{warnings.join('\n')}</div>
          )}
          {errors.length>0 && (
            <div className="p-3 bg-red-50 border border-red-200 rounded text-sm text-red-700">{errors.join('\n')}</div>
          )}

          <div className="flex justify-between items-center">
            <button onClick={back} className="px-4 py-2 border border-border rounded bg-surface hover:bg-muted">Back</button>
            <button disabled={validating} onClick={onSubmit} className={`px-4 py-2 rounded ${validating?'bg-muted text-text/60':'bg-primary text-on-primary hover:brightness-110'}`}>{validating?'Validating...':'Create Job'}</button>
          </div>
        </div>
      )}
    </div>
  );
}
