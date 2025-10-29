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

  // Reproducibility
  const [seed, setSeed] = useState('');
  const [deterministic, setDeterministic] = useState(false);

  // Experiment tracking
  const [expName, setExpName] = useState('');
  const [expTags, setExpTags] = useState('');
  const [expNotes, setExpNotes] = useState('');
  const [trackMlflow, setTrackMlflow] = useState(false);
  const [trackWandb, setTrackWandb] = useState(false);
  const [trackProject, setTrackProject] = useState('trainer');
  const [trackRun, setTrackRun] = useState('');
  const [mlflowUri, setMlflowUri] = useState('');
  const [wandbApiKey, setWandbApiKey] = useState('');
  const [wandbEntity, setWandbEntity] = useState('');

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

  // Hyperparameter Optimization (Optuna)
  const [hpoEnabled, setHpoEnabled] = useState(false);
  const [hpoMetric, setHpoMetric] = useState('eval_loss');
  const [hpoDirection, setHpoDirection] = useState('minimize');
  const [hpoMetricsText, setHpoMetricsText] = useState(''); // comma-separated multi-metrics
  const [hpoMaxTrials, setHpoMaxTrials] = useState(10);
  const [hpoTimeout, setHpoTimeout] = useState(0);
  const [hpoWorkers, setHpoWorkers] = useState(1);
  const [hpoSampler, setHpoSampler] = useState('tpe'); // tpe|random
  const [hpoPruner, setHpoPruner] = useState('median'); // median|asha|none
  const [hpoTrialEpochs, setHpoTrialEpochs] = useState(1);
  const [hpoSpace, setHpoSpace] = useState([
    { name: 'learning_rate', type: 'float', low: 1e-6, high: 5e-4, log: true },
    { name: 'batch_size', type: 'int', low: 4, high: 32, step: 4 },
  ]);

  // Architecture custom params (simple)
  const [customArch, setCustomArch] = useState({ input_size: 784, output_size: 10, hidden_layers: '512,256,128' });

  // Gating mechanisms for dynamic capacity and smarter compute
  const [gating, setGating] = useState({
    enabled: false,
    type: 'moe', // moe | moe_lora | routerless | film_gates | span_routing | mixture_of_depths
    num_experts: 8,
    num_selected: 2,
    capacity_factor: 1.25,
    gate_temp: 1.0,
    z_loss_coef: 0.01,
    lora_rank: 8,
    lora_alpha: 16.0,
    depth_threshold: 0.8,
    min_layers: 1,
    span_size: 32,
    span_overlap: 8,
    num_modalities: 3,
    dropout: 0.1,
    enable_metrics: true,
  });

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
    },
    hf_zero3_perf: {
      framework: 'huggingface', arch: 'transformer', model: 'gpt2', epochs: 3, batchSize: 8, lr: 3e-5,
      distributed: { deepspeed: 'training_scripts/deepspeed/zero3.json', gradient_accumulation_steps: 8 },
      precision: 'bf16'
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
    if (p.distributed) {
      if (p.distributed.deepspeed) setDsPath(p.distributed.deepspeed);
      if (p.distributed.gradient_accumulation_steps) setGradAccum(p.distributed.gradient_accumulation_steps);
    }
    if (p.precision) setPrecision(p.precision);
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
      if (seed !== '') cfg.seed = parseInt(seed) || undefined;
      if (deterministic) cfg.deterministic = true;
      cfg.tracking = {
        mlflow: { enabled: trackMlflow, experiment: expName || undefined },
        wandb: { enabled: trackWandb },
        project: trackProject || undefined,
        name: trackRun || undefined,
        env: {
          ...(mlflowUri ? { MLFLOW_TRACKING_URI: mlflowUri } : {}),
          ...(wandbApiKey ? { WANDB_API_KEY: wandbApiKey } : {}),
          ...(wandbEntity ? { WANDB_ENTITY: wandbEntity } : {}),
        }
      };
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
      if (hpoEnabled) {
        cfg.hpo = {
          enabled: true,
          metric: hpoMetric,
          direction: hpoDirection,
          metrics: (hpoMetricsText||'').split(',').map(s=>s.trim()).filter(Boolean),
          max_trials: parseInt(hpoMaxTrials)||10,
          timeout_seconds: parseInt(hpoTimeout)||0,
          workers: parseInt(hpoWorkers)||1,
          sampler: hpoSampler,
          pruner: hpoPruner,
          trial_epochs: parseInt(hpoTrialEpochs)||1,
          space: hpoSpace,
        };
      }
    }
    if (framework === 'pytorch' && arch === 'custom') {
      cfg.input_size = customArch.input_size;
      cfg.output_size = customArch.output_size;
      cfg.hidden_layers = customArch.hidden_layers.split(',').map(s=>parseInt(s.trim())).filter(Boolean);
    }
    // Add gating configuration if enabled
    if (gating.enabled) {
      cfg.gating = {
        type: gating.type,
        num_experts: parseInt(gating.num_experts) || 8,
        num_selected: parseInt(gating.num_selected) || 2,
        capacity_factor: parseFloat(gating.capacity_factor) || 1.25,
        gate_temp: parseFloat(gating.gate_temp) || 1.0,
        z_loss_coef: parseFloat(gating.z_loss_coef) || 0.01,
        lora_rank: parseInt(gating.lora_rank) || 8,
        lora_alpha: parseFloat(gating.lora_alpha) || 16.0,
        depth_threshold: parseFloat(gating.depth_threshold) || 0.8,
        min_layers: parseInt(gating.min_layers) || 1,
        span_size: parseInt(gating.span_size) || 32,
        span_overlap: parseInt(gating.span_overlap) || 8,
        num_modalities: parseInt(gating.num_modalities) || 3,
        dropout: parseFloat(gating.dropout) || 0.1,
        enable_metrics: gating.enable_metrics,
      };
    }
    return cfg;
  };

  const autoSuggestHpo = () => {
    // Heuristic: infer GPU memory and model size hints
    let memGiB = 16;
    try {
      const g = (partitions?.gpus||[])[0];
      if (g && g.memory_total_mib) memGiB = Math.max(8, Math.floor((g.memory_total_mib||0)/1024));
    } catch {}
    const isLargeModel = /llama|t5-(?:(?:3|11)b)|gpt\d+-?(?:xl|large|neo)/i.test(hfModel);
    const baseLR = isLargeModel ? 1e-5 : 3e-5;
    const maxBS = Math.max(4, Math.min(64, Math.floor(memGiB/2)));
    setHpoSpace([
      { name:'learning_rate', type:'float', low: baseLR/10, high: baseLR*10, log:true },
      { name:'batch_size', type:'int', low: 4, high: maxBS, step: 4 },
      ...(lora.enabled ? [{ name:'lora.r', type:'int', low: 4, high: 64, step: 4 }] : [])
    ]);
    setHpoMaxTrials(20);
    setHpoTrialEpochs(1);
    setHpoSampler('tpe');
    setHpoPruner('median');
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
    if (expName) {
      payload.experiment = { name: expName, tags: expTags.split(',').map(s=>s.trim()).filter(Boolean), notes: expNotes };
    }
    try {
      const v = await api.validateJob(payload);
      if (!v.ok) {
        setErrors(v.errors||[]); setWarnings(v.warnings||[]); setValidating(false);
        return;
      }
      const res = await api.createJob(payload);
      try {
        if (hpoEnabled) {
          const metrics = (hpoMetricsText||'').split(',').map(s=>s.trim()).filter(Boolean);
          await api.saveHpoStudy({
            job_id: res?.id,
            framework,
            model: hfModel,
            metrics: metrics.length>0 ? metrics : [hpoMetric],
            base_metric: hpoMetric,
            direction: hpoDirection,
            max_trials: hpoMaxTrials,
            timeout: hpoTimeout,
            workers: hpoWorkers,
            sampler: hpoSampler,
            pruner: hpoPruner,
            space: hpoSpace,
          });
        }
      } catch {}
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
              <button onClick={()=>applyPreset('hf_zero3_perf')} className="px-3 py-2 border border-border rounded bg-surface hover:bg-muted">HF ZeRO‑3 Performance</button>
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
            <button onClick={back} className="px-4 py-2 border border-border rounded bg-surface hover:bg-muted">Back</button>
            <button onClick={next} className="px-4 py-2 bg-primary text-on-primary rounded hover:brightness-110">Next</button>
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

          {framework==='huggingface' && (
            <details className="mt-2">
              <summary className="cursor-pointer text-sm font-semibold">Hyperparameter Optimization (Optuna)</summary>
              <div className="mt-2 grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="flex items-end"><label className="inline-flex items-center gap-2"><input type="checkbox" checked={hpoEnabled} onChange={e=>setHpoEnabled(e.target.checked)}/> Enable HPO</label></div>
                <div><label className="text-sm">Metric</label><input className="w-full border rounded px-3 py-2" value={hpoMetric} onChange={e=>setHpoMetric(e.target.value)} placeholder="eval_loss"/></div>
                <div><label className="text-sm">Direction</label><select className="w-full border rounded px-3 py-2" value={hpoDirection} onChange={e=>setHpoDirection(e.target.value)}><option value="minimize">Minimize</option><option value="maximize">Maximize</option></select></div>
                <div className="md:col-span-2"><label className="text-sm">Metrics (multi, comma‑sep)</label><input className="w-full border rounded px-3 py-2" value={hpoMetricsText} onChange={e=>setHpoMetricsText(e.target.value)} placeholder="eval_loss,accuracy"/></div>
                <div>
                  <label className="text-sm">Presets</label>
                  <select className="w-full border rounded px-3 py-2" onChange={e=>{
                    const p = e.target.value;
                    if (p==='loss_acc'){ setHpoMetricsText('eval_loss,accuracy'); setHpoMetric('eval_loss'); setHpoDirection('minimize'); }
                    if (p==='bleu_rouge'){ setHpoMetricsText('bleu,rougeL'); setHpoMetric('bleu'); setHpoDirection('maximize'); }
                    if (p==='f1_latency'){ setHpoMetricsText('f1,latency_ms'); setHpoMetric('f1'); setHpoDirection('maximize'); }
                    e.target.selectedIndex=0;
                  }}>
                    <option value="">Select…</option>
                    <option value="loss_acc">Loss + Accuracy</option>
                    <option value="bleu_rouge">BLEU + ROUGE</option>
                    <option value="f1_latency">F1 + Latency</option>
                  </select>
                </div>
                <div><label className="text-sm">Max Trials</label><input type="number" className="w-full border rounded px-3 py-2" value={hpoMaxTrials} onChange={e=>setHpoMaxTrials(parseInt(e.target.value)||0)} /></div>
                <div><label className="text-sm">Timeout (sec)</label><input type="number" className="w-full border rounded px-3 py-2" value={hpoTimeout} onChange={e=>setHpoTimeout(parseInt(e.target.value)||0)} /></div>
                <div><label className="text-sm">Workers</label><input type="number" className="w-full border rounded px-3 py-2" value={hpoWorkers} onChange={e=>setHpoWorkers(parseInt(e.target.value)||1)} /></div>
                <div><label className="text-sm">Sampler</label><select className="w-full border rounded px-3 py-2" value={hpoSampler} onChange={e=>setHpoSampler(e.target.value)}><option value="tpe">Bayesian (TPE)</option><option value="random">Random</option></select></div>
                <div><label className="text-sm">Pruner</label><select className="w-full border rounded px-3 py-2" value={hpoPruner} onChange={e=>setHpoPruner(e.target.value)}><option value="median">Median</option><option value="asha">ASHA</option><option value="none">None</option></select></div>
                <div><label className="text-sm">Trial Epochs</label><input type="number" className="w-full border rounded px-3 py-2" value={hpoTrialEpochs} onChange={e=>setHpoTrialEpochs(parseInt(e.target.value)||1)} /></div>
              </div>
              <div className="mt-2">
                <div className="text-sm font-semibold mb-1">Search Space</div>
                <div className="space-y-2">
                  {hpoSpace.map((p, idx) => (
                    <div key={idx} className="grid grid-cols-1 md:grid-cols-7 gap-2 text-xs items-end">
                      <label className="md:col-span-2">Name<input className="w-full border rounded px-2 py-1" value={p.name} onChange={e=>{const a=[...hpoSpace]; a[idx]={...a[idx], name:e.target.value}; setHpoSpace(a);}} placeholder="learning_rate, batch_size, lora.r, ..."/></label>
                      <label>Type<select className="w-full border rounded px-2 py-1" value={p.type} onChange={e=>{const a=[...hpoSpace]; a[idx]={...a[idx], type:e.target.value}; setHpoSpace(a);}}><option value="float">float</option><option value="int">int</option><option value="categorical">categorical</option></select></label>
                      {p.type!=='categorical' ? (<>
                        <label>Low<input type="number" className="w-full border rounded px-2 py-1" value={p.low} onChange={e=>{const a=[...hpoSpace]; a[idx]={...a[idx], low: parseFloat(e.target.value)}; setHpoSpace(a);}}/></label>
                        <label>High<input type="number" className="w-full border rounded px-2 py-1" value={p.high} onChange={e=>{const a=[...hpoSpace]; a[idx]={...a[idx], high: parseFloat(e.target.value)}; setHpoSpace(a);}}/></label>
                        <label>Step<input type="number" className="w-full border rounded px-2 py-1" value={p.step||''} onChange={e=>{const a=[...hpoSpace]; a[idx]={...a[idx], step: e.target.value===''?undefined:parseFloat(e.target.value)}; setHpoSpace(a);}}/></label>
                        <label className="inline-flex items-center gap-2"><input type="checkbox" checked={!!p.log} onChange={e=>{const a=[...hpoSpace]; a[idx]={...a[idx], log: e.target.checked}; setHpoSpace(a);}}/> log</label>
                      </>): (
                        <label className="md:col-span-4">Choices<input className="w-full border rounded px-2 py-1" placeholder="comma-separated" value={(p.choices||[]).join(',')} onChange={e=>{const a=[...hpoSpace]; a[idx]={...a[idx], choices: e.target.value.split(',').map(s=>s.trim()).filter(Boolean)}; setHpoSpace(a);}}/></label>
                      )}
                      <button className="px-2 py-1 border rounded" onClick={()=>{const a=[...hpoSpace]; a.splice(idx,1); setHpoSpace(a);}}>Remove</button>
                    </div>
                  ))}
                  <div className="flex gap-2">
                    <button className="px-2 py-1 border rounded" onClick={()=>setHpoSpace([...hpoSpace, { name:'', type:'float', low:0, high:1 }])}>Add Param</button>
                    <button className="px-2 py-1 border rounded" onClick={autoSuggestHpo}>Auto-suggest</button>
                    {framework==='pytorch' && (
                      <button className="px-2 py-1 border rounded" onClick={()=>{
                        setHpoEnabled(true);
                        setHpoMetric('eval_loss');
                        setHpoSpace([
                          { name:'learning_rate', type:'float', low:1e-5, high:1e-2, log:true },
                          { name:'batch_size', type:'int', low:8, high:128, step:8 },
                        ]);
                        setHpoMaxTrials(20);
                      }}>Torch HPO Preset</button>
                    )}
                  </div>
                </div>
              </div>
            </details>
          )}

        {/* Gating Mechanisms for Dynamic Capacity */}
        <details className="mt-2">
          <summary className="cursor-pointer text-sm font-semibold">🚀 Gating Mechanisms (MoE, Mixture-of-Depths, Multi-modal)</summary>
          <div className="mt-2 space-y-4">
            <div className="flex items-center gap-2 mb-3">
              <label className="inline-flex items-center gap-2">
                <input type="checkbox" checked={gating.enabled} onChange={e=>setGating({...gating, enabled:e.target.checked})}/>
                <span className="font-semibold">Enable Gating for Dynamic Compute & Smarter Routing</span>
              </label>
            </div>

            {gating.enabled && (
              <div className="space-y-4 border border-primary/20 rounded-lg p-4 bg-primary/5">
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div>
                    <label className="text-sm font-semibold">Gating Type</label>
                    <select className="w-full border rounded px-3 py-2 mt-1" value={gating.type} onChange={e=>setGating({...gating, type: e.target.value})}>
                      <option value="moe">Token-level MoE (Top-K routing)</option>
                      <option value="moe_lora">MoE-LoRA (Low VRAM footprint)</option>
                      <option value="routerless">Routerless MoE (DeepSeek-style)</option>
                      <option value="mixture_of_depths">Mixture-of-Depths (Early exit)</option>
                      <option value="film_gates">FiLM Gating (Multi-modal fusion)</option>
                      <option value="span_routing">Span Routing (Efficiency)</option>
                    </select>
                    <p className="text-xs text-text/60 mt-1">
                      {gating.type === 'moe' && 'Switch/Top-K experts with capacity factor & z-loss'}
                      {gating.type === 'moe_lora' && 'Per-expert LoRA adapters - dramatically smaller VRAM'}
                      {gating.type === 'routerless' && 'No explicit routing - simpler training, no collapse'}
                      {gating.type === 'mixture_of_depths' && 'Dynamic layer selection - easy tokens exit early'}
                      {gating.type === 'film_gates' && 'Modality-conditioned gates for text↔image↔audio'}
                      {gating.type === 'span_routing' && 'Route contiguous spans to share KV/memory'}
                    </p>
                  </div>

                  {(gating.type === 'moe' || gating.type === 'moe_lora' || gating.type === 'routerless' || gating.type === 'span_routing') && (
                    <div>
                      <label className="text-sm">Number of Experts</label>
                      <input type="number" min="2" max="256" className="w-full border rounded px-3 py-2 mt-1" value={gating.num_experts} onChange={e=>setGating({...gating, num_experts: parseInt(e.target.value)||8})}/>
                      <p className="text-xs text-text/60 mt-1">Total expert modules (typically 4-16)</p>
                    </div>
                  )}

                  {(gating.type === 'moe' || gating.type === 'moe_lora') && (
                    <div>
                      <label className="text-sm">Top-K Selected</label>
                      <input type="number" min="1" max="8" className="w-full border rounded px-3 py-2 mt-1" value={gating.num_selected} onChange={e=>setGating({...gating, num_selected: parseInt(e.target.value)||2})}/>
                      <p className="text-xs text-text/60 mt-1">Number of experts per token (usually 1-2)</p>
                    </div>
                  )}
                </div>

                {/* MoE-specific parameters */}
                {(gating.type === 'moe' || gating.type === 'moe_lora') && (
                  <div className="border-t border-border pt-3">
                    <div className="text-sm font-semibold mb-2">MoE Load Balancing</div>
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                      <div>
                        <label className="text-xs">Capacity Factor: {gating.capacity_factor}x</label>
                        <input type="range" min="1.0" max="2.0" step="0.05" value={gating.capacity_factor} onChange={e=>setGating({...gating, capacity_factor: parseFloat(e.target.value)})} className="w-full"/>
                        <p className="text-xs text-text/60">Expert capacity multiplier (higher = less overflow)</p>
                      </div>
                      <div>
                        <label className="text-xs">Gate Temperature: {gating.gate_temp}</label>
                        <input type="range" min="0.1" max="2.0" step="0.1" value={gating.gate_temp} onChange={e=>setGating({...gating, gate_temp: parseFloat(e.target.value)})} className="w-full"/>
                        <p className="text-xs text-text/60">Routing sharpness (lower = more peaked)</p>
                      </div>
                      <div>
                        <label className="text-xs">Z-loss Coefficient</label>
                        <input type="number" step="0.001" className="w-full border rounded px-2 py-1" value={gating.z_loss_coef} onChange={e=>setGating({...gating, z_loss_coef: parseFloat(e.target.value)||0.01})}/>
                        <p className="text-xs text-text/60">Auxiliary loss for stability (0.001-0.1)</p>
                      </div>
                    </div>
                  </div>
                )}

                {/* MoE-LoRA specific */}
                {gating.type === 'moe_lora' && (
                  <div className="border-t border-border pt-3">
                    <div className="text-sm font-semibold mb-2">LoRA Configuration</div>
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                      <div>
                        <label className="text-xs">LoRA Rank: {gating.lora_rank}</label>
                        <input type="range" min="1" max="128" value={gating.lora_rank} onChange={e=>setGating({...gating, lora_rank: parseInt(e.target.value)})} className="w-full"/>
                      </div>
                      <div>
                        <label className="text-xs">LoRA Alpha</label>
                        <input type="number" className="w-full border rounded px-2 py-1" value={gating.lora_alpha} onChange={e=>setGating({...gating, lora_alpha: parseFloat(e.target.value)||16.0})}/>
                      </div>
                      <div>
                        <label className="text-xs">Dropout</label>
                        <input type="number" step="0.01" min="0" max="0.5" className="w-full border rounded px-2 py-1" value={gating.dropout} onChange={e=>setGating({...gating, dropout: parseFloat(e.target.value)||0.1})}/>
                      </div>
                    </div>
                  </div>
                )}

                {/* Mixture-of-Depths specific */}
                {gating.type === 'mixture_of_depths' && (
                  <div className="border-t border-border pt-3">
                    <div className="text-sm font-semibold mb-2">Depth Gating Configuration</div>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                      <div>
                        <label className="text-xs">Confidence Threshold: {gating.depth_threshold}</label>
                        <input type="range" min="0.1" max="0.99" step="0.05" value={gating.depth_threshold} onChange={e=>setGating({...gating, depth_threshold: parseFloat(e.target.value)})} className="w-full"/>
                        <p className="text-xs text-text/60">Higher = more tokens exit early</p>
                      </div>
                      <div>
                        <label className="text-xs">Minimum Layers</label>
                        <input type="number" min="1" max="12" className="w-full border rounded px-2 py-1" value={gating.min_layers} onChange={e=>setGating({...gating, min_layers: parseInt(e.target.value)||1})}/>
                        <p className="text-xs text-text/60">All tokens process through these layers</p>
                      </div>
                    </div>
                  </div>
                )}

                {/* FiLM gating specific */}
                {gating.type === 'film_gates' && (
                  <div className="border-t border-border pt-3">
                    <div className="text-sm font-semibold mb-2">Multi-modal Configuration</div>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                      <div>
                        <label className="text-xs">Number of Modalities</label>
                        <select className="w-full border rounded px-2 py-1" value={gating.num_modalities} onChange={e=>setGating({...gating, num_modalities: parseInt(e.target.value)||3})}>
                          <option value="2">2 (e.g., text + image)</option>
                          <option value="3">3 (text + image + audio)</option>
                          <option value="4">4 (text + image + audio + video)</option>
                        </select>
                      </div>
                      <div>
                        <label className="text-xs">Dropout</label>
                        <input type="number" step="0.01" min="0" max="0.5" className="w-full border rounded px-2 py-1" value={gating.dropout} onChange={e=>setGating({...gating, dropout: parseFloat(e.target.value)||0.1})}/>
                      </div>
                    </div>
                  </div>
                )}

                {/* Span routing specific */}
                {gating.type === 'span_routing' && (
                  <div className="border-t border-border pt-3">
                    <div className="text-sm font-semibold mb-2">Span Configuration</div>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                      <div>
                        <label className="text-xs">Span Size (tokens)</label>
                        <input type="number" min="8" max="512" step="8" className="w-full border rounded px-2 py-1" value={gating.span_size} onChange={e=>setGating({...gating, span_size: parseInt(e.target.value)||32})}/>
                        <p className="text-xs text-text/60">Contiguous tokens routed together</p>
                      </div>
                      <div>
                        <label className="text-xs">Overlap (tokens)</label>
                        <input type="number" min="0" max="64" step="4" className="w-full border rounded px-2 py-1" value={gating.span_overlap} onChange={e=>setGating({...gating, span_overlap: parseInt(e.target.value)||8})}/>
                        <p className="text-xs text-text/60">Overlap between consecutive spans</p>
                      </div>
                    </div>
                  </div>
                )}

                {/* Metrics monitoring */}
                <div className="border-t border-border pt-3">
                  <label className="inline-flex items-center gap-2 text-sm">
                    <input type="checkbox" checked={gating.enable_metrics} onChange={e=>setGating({...gating, enable_metrics:e.target.checked})}/>
                    <span>Track expert utilization, capacity overflow, and routing statistics</span>
                  </label>
                  <p className="text-xs text-text/60 ml-6 mt-1">
                    View heatmaps and metrics in Jobs dashboard after training starts
                  </p>
                </div>

                {/* Info box */}
                <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded p-3 text-xs">
                  <div className="font-semibold mb-1">💡 Gating Benefits:</div>
                  <ul className="list-disc list-inside space-y-1 text-text/80">
                    <li><strong>MoE:</strong> Scales model capacity while keeping inference cost constant</li>
                    <li><strong>MoE-LoRA:</strong> 10-100x less VRAM than full MoE - perfect for consumer GPUs</li>
                    <li><strong>Routerless:</strong> Easier training, no routing collapse issues</li>
                    <li><strong>Mixture-of-Depths:</strong> 2-3x faster inference by skipping layers for easy tokens</li>
                    <li><strong>FiLM:</strong> Adaptive multi-modal fusion (text↔vision↔audio)</li>
                    <li><strong>Span Routing:</strong> Reduced routing overhead for long contexts</li>
                  </ul>
                </div>
              </div>
            )}
          </div>
        </details>

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
                  <div className="flex items-end gap-2">
                    <button type="button" className="px-2 py-1 border rounded" onClick={()=>setDsPath('training_scripts/deepspeed/zero2.json')}>Use ZeRO‑2 template</button>
                    <button type="button" className="px-2 py-1 border rounded" onClick={()=>setDsPath('training_scripts/deepspeed/zero3.json')}>Use ZeRO‑3 template</button>
                  </div>
                  <div className="md:col-span-2"><label className="text-xs">FSDP config string</label><input className="w-full border rounded px-2 py-1" value={fsdp} onChange={e=>setFsdp(e.target.value)} placeholder="full_shard auto_wrap"/></div>
                  <label className="inline-flex items-center gap-2"><input type="checkbox" checked={ddpFindUnused} onChange={e=>setDdpFindUnused(e.target.checked)}/> DDP find_unused_parameters</label>
                  <label className="text-xs">FSDP min params<input type="number" className="w-full border rounded px-2 py-1" value={fsdpMinParams} onChange={e=>setFsdpMinParams(parseInt(e.target.value)||0)}/></label>
                </div>
                <div className="text-[11px] text-text/60 mt-1">Note: Multi‑node/multi‑GPU usually requires launching with torchrun externally.</div>
              </div>
            </div>
          </details>
        )}

        {/* Experiment & Tracking */}
        <details className="mt-2">
          <summary className="cursor-pointer text-sm font-semibold">Experiment & Tracking</summary>
          <div className="mt-2 grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
            <div>
              <label className="text-xs">Experiment Name</label>
              <input className="w-full border rounded px-2 py-1" value={expName} onChange={e=>setExpName(e.target.value)} placeholder="my-experiment" />
            </div>
            <div>
              <label className="text-xs">Tags (comma)</label>
              <input className="w-full border rounded px-2 py-1" value={expTags} onChange={e=>setExpTags(e.target.value)} placeholder="baseline, lora, test" />
            </div>
            <div>
              <label className="text-xs">Notes</label>
              <input className="w-full border rounded px-2 py-1" value={expNotes} onChange={e=>setExpNotes(e.target.value)} placeholder="Short description" />
            </div>
            <div>
              <label className="text-xs">Seed</label>
              <input className="w-full border rounded px-2 py-1" value={seed} onChange={e=>setSeed(e.target.value)} placeholder="e.g., 42" />
              <label className="inline-flex items-center gap-2 mt-1"><input type="checkbox" checked={deterministic} onChange={e=>setDeterministic(e.target.checked)} /> Deterministic</label>
            </div>
            <div className="md:col-span-2">
              <div className="text-xs font-semibold mb-1">Logging</div>
              <div className="flex items-center gap-4">
                <label className="inline-flex items-center gap-2"><input type="checkbox" checked={trackMlflow} onChange={e=>setTrackMlflow(e.target.checked)} /> MLflow</label>
                <label className="inline-flex items-center gap-2"><input type="checkbox" checked={trackWandb} onChange={e=>setTrackWandb(e.target.checked)} /> Weights & Biases</label>
              </div>
              <div className="grid grid-cols-2 gap-2 mt-2">
                <label className="text-xs">Project<input className="w-full border rounded px-2 py-1" value={trackProject} onChange={e=>setTrackProject(e.target.value)} placeholder="trainer" /></label>
                <label className="text-xs">Run Name<input className="w-full border rounded px-2 py-1" value={trackRun} onChange={e=>setTrackRun(e.target.value)} placeholder="run-001" /></label>
              </div>
              <div className="grid grid-cols-2 gap-2 mt-2">
                <label className="text-xs">MLflow Tracking URI<input className="w-full border rounded px-2 py-1" value={mlflowUri} onChange={e=>setMlflowUri(e.target.value)} placeholder="http://mlflow:5000" /></label>
                <label className="text-xs">W&B Entity<input className="w-full border rounded px-2 py-1" value={wandbEntity} onChange={e=>setWandbEntity(e.target.value)} placeholder="org-or-username" /></label>
                <label className="text-xs md:col-span-2">W&B API Key<input className="w-full border rounded px-2 py-1" value={wandbApiKey} onChange={e=>setWandbApiKey(e.target.value)} placeholder="xxxxxxxx" /></label>
              </div>
              <div className="text-[11px] text-text/60 mt-1">These are passed to the training process as environment variables (if provided). Logging degrades gracefully if unavailable.</div>
            </div>
          </div>
        </details>

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
            <button onClick={back} className="px-4 py-2 border border-border rounded bg-surface hover:bg-muted">Back</button>
            <button onClick={next} className="px-4 py-2 bg-primary text-on-primary rounded hover:brightness-110">Next</button>
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
            <div className="p-3 bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 rounded text-sm text-yellow-800 dark:text-yellow-200">{warnings.join('\n')}</div>
          )}
          {errors.length>0 && (
            <div className="p-3 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded text-sm text-red-700 dark:text-red-300">{errors.join('\n')}</div>
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
