import argparse
import json
import os
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import math
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    TrainerCallback,
)

try:
    from transformers import BitsAndBytesConfig
except Exception:
    BitsAndBytesConfig = None  # type: ignore

try:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
except Exception:
    LoraConfig = None  # type: ignore
    get_peft_model = None  # type: ignore
    prepare_model_for_kbit_training = None  # type: ignore
    PeftModel = None  # type: ignore

from datasets import Dataset
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# -----------------------------
# Utilities and helpers
# -----------------------------

def _to_dtype(s: str):
    m = {
        'bfloat16': torch.bfloat16,
        'bf16': torch.bfloat16,
        'float16': torch.float16,
        'fp16': torch.float16,
        'float32': torch.float32,
        'fp32': torch.float32,
    }
    return m.get((s or '').lower(), torch.float32)


def create_dummy_text_dataset(config):
    num_samples = config.get('num_samples', 1000)
    max_length = config.get('max_length', 128)
    num_classes = config.get('num_classes', 2)
    data = {
        'text': [f"Sample text {i} for fine-tuning." for i in range(num_samples)],
        'label': np.random.randint(0, num_classes, num_samples).tolist(),
    }
    return Dataset.from_dict(data)


def build_model(config):
    model_name = config.get('model_name', 'bert-base-uncased')
    task_type = config.get('task_type', 'classification')

    quant = (config.get('quantization') or '').lower()
    lora_cfg = config.get('lora') or {}
    use_qlora = bool(lora_cfg.get('qlora', False)) or quant == 'int4'

    quant_config = None
    if use_qlora and BitsAndBytesConfig is not None:
        qtype = (config.get('fourbit_quant_type') or 'nf4').lower()  # 'nf4'|'fp4'
        double_q = bool(config.get('double_quant', lora_cfg.get('double_quant', False)))
        compute_dtype = str(config.get('compute_dtype', 'bfloat16')).lower()  # 'bfloat16'|'float16'|'float32'
        dtype_map = {
            'bfloat16': torch.bfloat16,
            'float16': torch.float16,
            'fp16': torch.float16,
            'float32': torch.float32,
            'fp32': torch.float32,
        }
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4' if qtype == 'nf4' else 'fp4',
            bnb_4bit_use_double_quant=double_q,
            bnb_4bit_compute_dtype=dtype_map.get(compute_dtype, torch.bfloat16),
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    common_kwargs = {}
    if quant_config is not None:
        common_kwargs['quantization_config'] = quant_config
        common_kwargs['device_map'] = 'auto'

    if task_type == 'classification':
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=config.get('num_classes', 2),
            **common_kwargs,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, **common_kwargs)

    # Apply LoRA if requested
    if lora_cfg.get('enabled') and LoraConfig is not None and get_peft_model is not None:
        if quant_config is not None and prepare_model_for_kbit_training is not None:
            model = prepare_model_for_kbit_training(model)
        target_modules = lora_cfg.get('target_modules') or []
        if isinstance(target_modules, str):
            target_modules = [x.strip() for x in target_modules.split(',') if x.strip()]
        peft_cfg = LoraConfig(
            r=int(lora_cfg.get('r', 8)),
            lora_alpha=int(lora_cfg.get('alpha', 16)),
            lora_dropout=float(lora_cfg.get('dropout', 0.05)),
            target_modules=target_modules or None,
            bias='none',
            task_type='SEQ_CLS' if task_type == 'classification' else 'CAUSAL_LM',
        )
        model = get_peft_model(model, peft_cfg)
        # Optionally load an existing adapter to continue training
        load_path = lora_cfg.get('load_path')
        if load_path and PeftModel is not None:
            try:
                model = PeftModel.from_pretrained(model, load_path)
                print(f"Loaded existing LoRA adapter from: {load_path}")
            except Exception as e:
                print(f"Warning: failed to load adapter from {load_path}: {e}")

    return tokenizer, model


class AdvancedTrainer(Trainer):
    """HF Trainer extension with optional Knowledge Distillation and custom grad clipping.

    Config keys used (under `config`):
      - distillation: {
            enabled: bool,
            teacher_model: str | path,
            temperature: float,
            alpha_distill: float,    # KD loss weight
            alpha_ce: float,         # hard-label CE weight
            multi_teachers: [str],   # optional multiple teacher models
        }
      - grad_clip: { type: 'norm'|'value', max_norm: float, max_value: float }
    """
    def __init__(self, *args, **kwargs):
        self._adv_cfg: Dict[str, Any] = kwargs.pop('adv_config', {}) or {}
        super().__init__(*args, **kwargs)
        self._teacher_models: List[Any] = []
        self._distill_cfg = (self._adv_cfg.get('distillation') or {}) if isinstance(self._adv_cfg, dict) else {}
        if self._distill_cfg.get('enabled'):
            self._init_teachers()

    def _init_teachers(self):
        tcfg = self._distill_cfg
        teacher_ids: List[str] = []
        if tcfg.get('multi_teachers'):
            teacher_ids = [str(x) for x in tcfg.get('multi_teachers')]
        elif tcfg.get('teacher_model'):
            teacher_ids = [str(tcfg.get('teacher_model'))]
        if not teacher_ids:
            print('KD enabled but no teacher specified; disabling.')
            self._distill_cfg['enabled'] = False
            return
        task_type = (self._adv_cfg.get('task_type') or '').lower()
        for tid in teacher_ids:
            try:
                if task_type == 'classification':
                    tm = AutoModelForSequenceClassification.from_pretrained(tid)
                else:
                    tm = AutoModelForCausalLM.from_pretrained(tid)
            except Exception:
                # Fallback try both
                try:
                    tm = AutoModelForSequenceClassification.from_pretrained(tid)
                except Exception:
                    tm = AutoModelForCausalLM.from_pretrained(tid)
            tm.eval()
            for p in tm.parameters():
                p.requires_grad_(False)
            device = self.model.device if hasattr(self.model, 'device') else torch.device('cpu')
            tm.to(device)
            self._teacher_models.append(tm)
        print(f"Loaded {len(self._teacher_models)} teacher model(s) for KD")

    def clip_gradients(self, optimizer, max_norm: float):
        gcfg = (self._adv_cfg.get('grad_clip') or {}) if isinstance(self._adv_cfg, dict) else {}
        mode = (gcfg.get('type') or 'norm').lower()
        if mode == 'value':
            max_val = float(gcfg.get('max_value', 1.0))
            for p in self.model.parameters():
                if p.grad is not None:
                    p.grad.data.clamp_(min=-max_val, max=max_val)
            return
        # default norm clipping
        return super().clip_gradients(optimizer, max_norm)

    def compute_loss(self, model, inputs, return_outputs=False):
        # Base loss
        outputs = model(**inputs)
        loss = outputs.get('loss') if isinstance(outputs, dict) else outputs[0]

        # Knowledge Distillation
        if self._distill_cfg.get('enabled') and self._teacher_models:
            temperature = float(self._distill_cfg.get('temperature', 2.0))
            alpha_kd = float(self._distill_cfg.get('alpha_distill', 0.5))
            alpha_ce = float(self._distill_cfg.get('alpha_ce', 0.5))
            # Student logits
            s_logits = outputs.get('logits') if isinstance(outputs, dict) else outputs[1]
            if s_logits is None:
                # try to recompute forward to extract logits
                with torch.no_grad():
                    s_out = model(**{k: v for k, v in inputs.items() if k in ('input_ids','attention_mask','token_type_ids','labels')})
                    s_logits = s_out.get('logits') if isinstance(s_out, dict) else s_out[1]
            if s_logits is not None:
                # Teachers average logits
                with torch.no_grad():
                    t_logits_sum = None
                    for tm in self._teacher_models:
                        tout = tm(**{k: v for k, v in inputs.items() if k in ('input_ids','attention_mask','token_type_ids')})
                        tl = tout.get('logits') if isinstance(tout, dict) else tout[0]
                        t_logits_sum = tl if t_logits_sum is None else (t_logits_sum + tl)
                    t_logits = t_logits_sum / float(len(self._teacher_models))
                # KL Divergence with temperature
                kd_loss_fn = torch.nn.KLDivLoss(reduction='batchmean')
                s_log_probs = torch.log_softmax(s_logits / temperature, dim=-1)
                t_probs = torch.softmax(t_logits / temperature, dim=-1)
                kd_loss = kd_loss_fn(s_log_probs, t_probs) * (temperature ** 2)
                # Combine
                loss = alpha_ce * loss + alpha_kd * kd_loss
                try:
                    print('METRIC:' + json.dumps({'kind':'kd','kd_loss': float(kd_loss.detach().cpu().item())}))
                except Exception:
                    pass

        return (loss, outputs) if return_outputs else loss


def _score_difficulty_texts(texts: List[str], mode: str = 'length') -> List[float]:
    if mode == 'length':
        return [float(len(t or '')) for t in texts]
    if mode == 'random':
        return np.random.rand(len(texts)).astype(float).tolist()
    # default: length
    return [float(len(t or '')) for t in texts]


def _apply_curriculum_subset(dataset: Dataset, curriculum: Dict[str, Any], epoch_idx: int, total_epochs: int) -> Dataset:
    """Return a subset of the dataset according to curriculum schedule.

    curriculum: { mode: 'length'|'random'|'provided', start_frac: 0.3, end_frac: 1.0 }
    Grows the portion of harder examples over the stage epochs.
    """
    if not curriculum:
        return dataset
    mode = (curriculum.get('mode') or 'length').lower()
    start_frac = float(curriculum.get('start_frac', 0.5))
    end_frac = float(curriculum.get('end_frac', 1.0))
    start_frac = min(max(0.05, start_frac), 1.0)
    end_frac = min(max(start_frac, end_frac), 1.0)
    if total_epochs <= 1:
        frac = end_frac
    else:
        frac = start_frac + (end_frac - start_frac) * (epoch_idx / max(1, total_epochs - 1))
    frac = min(max(0.05, frac), 1.0)

    # compute difficulty scores
    texts = dataset['text'] if 'text' in dataset.column_names else []
    if mode == 'provided' and 'difficulty' in dataset.column_names:
        scores = [float(x) for x in dataset['difficulty']]
    else:
        scores = _score_difficulty_texts(texts, mode=mode)
    # lower score = easier by convention
    # select first N easiest based on frac
    idx_sorted = np.argsort(np.array(scores))
    n = max(1, int(math.ceil(len(idx_sorted) * frac)))
    chosen = idx_sorted[:n].tolist()
    return dataset.select(chosen)


def _prepare_tokenized_dataset(tokenizer, raw_dataset: Dataset, task_type: str, max_length: int) -> Tuple[Dataset, Dataset]:
    split = raw_dataset.train_test_split(test_size=0.2)
    train_dataset = split['train']
    eval_dataset = split['test']

    def tokenize_fn(examples):
        return tokenizer(
            examples['text'],
            padding='max_length',
            truncation=True,
            max_length=max_length,
        )

    train_dataset = train_dataset.map(tokenize_fn, batched=True)
    eval_dataset = eval_dataset.map(tokenize_fn, batched=True)
    if task_type == 'classification':
        cols = ['input_ids', 'attention_mask', 'label']
    else:
        def set_labels(batch):
            batch['labels'] = batch['input_ids']
            return batch
        train_dataset = train_dataset.map(set_labels, batched=True)
        eval_dataset = eval_dataset.map(set_labels, batched=True)
        cols = ['input_ids', 'attention_mask', 'labels']
    train_dataset.set_format('torch', columns=cols)
    eval_dataset.set_format('torch', columns=cols)
    return train_dataset, eval_dataset


def _build_training_args(output_dir: str, config: Dict[str, Any], stage_cfg: Optional[Dict[str, Any]] = None) -> TrainingArguments:
    s = stage_cfg or {}
    dist = config.get('distributed') or {}
    # precision
    precision = str(s.get('precision', config.get('precision', ''))).lower()
    compute_dtype = str(s.get('compute_dtype', config.get('compute_dtype',''))).lower()
    if not compute_dtype and precision in ('fp16','float16'):
        compute_dtype = 'fp16'
    if not compute_dtype and precision in ('bf16','bfloat16'):
        compute_dtype = 'bf16'
    if precision == 'fp8':
        # HF Trainer doesn't natively toggle fp8; warn user
        print('Warning: FP8 requested; ensure model implements FP8 and environment supports it (Transformer Engine). Proceeding without explicit Trainer flag.')
    # gradient clipping
    max_grad_norm = float(s.get('max_grad_norm', config.get('max_grad_norm', 1.0)))
    args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=float(s.get('learning_rate', config.get('learning_rate', 2e-5))),
        per_device_train_batch_size=int(s.get('batch_size', config.get('batch_size', 8))),
        per_device_eval_batch_size=int(s.get('eval_batch_size', s.get('batch_size', config.get('batch_size', 8)))) ,
        num_train_epochs=int(s.get('epochs', config.get('epochs', 1))),
        logging_dir=os.path.join(BASE_DIR, 'logs'),
        logging_steps=int(config.get('logging_steps', 10)),
        evaluation_strategy=str(config.get('evaluation_strategy', 'epoch')),
        save_strategy=str(s.get('save_strategy', config.get('save_strategy', 'epoch'))),
        save_total_limit=int(config.get('save_total_limit', 2)),
        load_best_model_at_end=bool(config.get('load_best_model_at_end', False)),
        gradient_checkpointing=bool(s.get('gradient_checkpointing', config.get('gradient_checkpointing', False))),
        bf16=compute_dtype in ('bfloat16','bf16'),
        fp16=compute_dtype in ('float16','fp16'),
        max_grad_norm=max_grad_norm,
        gradient_accumulation_steps=int(s.get('gradient_accumulation_steps', config.get('gradient_accumulation_steps', 1))),
        ddp_find_unused_parameters=bool(dist.get('ddp_find_unused_parameters', False)) if dist else None,
        deepspeed=dist.get('deepspeed') if isinstance(dist.get('deepspeed'), (str, dict)) else None,
        fsdp=dist.get('fsdp') if isinstance(dist.get('fsdp'), str) else None,
        fsdp_min_num_params=int(dist.get('fsdp_min_num_params', 0)) if dist else None,
    )
    return args


def finetune_hf(config, job_id):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    tokenizer, model = build_model(config)
    task_type = config.get('task_type', 'classification')
    # Distributed execution caveats
    dist = config.get('distributed') or {}
    if dist.get('nnodes') or dist.get('nproc_per_node'):
        print('Note: Multi-process/multi-node execution typically requires launching with torchrun. Backend launches single process. Configure external launcher if needed.')
    if dist.get('pipeline_parallelism'):
        print('Note: Pipeline parallelism is not directly supported by HF Trainer unless using specific libraries. This script does not configure it automatically.')
    output_dir = os.path.join(BASE_DIR, 'models', job_id)

    # Raw dataset + tokenization helper
    raw_dataset = create_dummy_text_dataset(config)

    # Multi-stage pipeline configuration
    stages: List[Dict[str, Any]] = config.get('stages') or []
    if not stages:
        # default single stage using top-level params
        stages = [
            {
                'name': 'main',
                'epochs': int(config.get('epochs', 3)),
                'learning_rate': float(config.get('learning_rate', 2e-5)),
                'batch_size': int(config.get('batch_size', 8)),
            }
        ]
    # ensure names
    for i, st in enumerate(stages):
        st['name'] = st.get('name') or f'stage{i+1}'

    adv_cfg = {
        'distillation': config.get('distillation') or {},
        'grad_clip': config.get('grad_clip') or {},
        'task_type': task_type,
    }

    # Train through stages
    global_step = 0
    for idx, stage in enumerate(stages):
        stage_name = stage.get('name', f'stage{idx+1}')
        stage_epochs = int(stage.get('epochs', 1))
        stage_out = os.path.join(output_dir, f'{stage_name}')
        os.makedirs(stage_out, exist_ok=True)

        # Curriculum subset before tokenization, then tokenize per-epoch when incremental
        curriculum = stage.get('curriculum') or {}

        # If curriculum incremental, we run loop of 1 epoch at a time to adjust subset
        incremental = bool(curriculum.get('incremental', False))
        total_loops = stage_epochs if incremental else 1
        loops_epochs = 1 if incremental else stage_epochs

        for loop_idx in range(total_loops):
            # Apply curriculum subset for this loop
            subset_raw = _apply_curriculum_subset(raw_dataset, curriculum, loop_idx, total_loops) if curriculum else raw_dataset
            train_dataset, eval_dataset = _prepare_tokenized_dataset(tokenizer, subset_raw, task_type, config.get('max_length', 128))

            training_args = _build_training_args(stage_out, config, {**stage, 'epochs': loops_epochs})

            data_collator = DataCollatorWithPadding(tokenizer=tokenizer) if task_type == 'classification' else None

            class JsonLogger(TrainerCallback):
                def on_log(self, args, state, control, logs=None, **kwargs):
                    try:
                        payload = {
                            'kind': 'log',
                            'stage': stage_name,
                            'stage_index': idx,
                            'step': int(state.global_step),
                            'epoch': float(state.epoch) if state.epoch is not None else None,
                            'logs': logs or {},
                            'time': datetime.now().isoformat(),
                        }
                        print('METRIC:' + json.dumps(payload), flush=True)
                    except Exception:
                        pass
                def on_epoch_end(self, args, state, control, **kwargs):
                    try:
                        payload = {
                            'kind': 'epoch',
                            'stage': stage_name,
                            'stage_index': idx,
                            'epoch': float(state.epoch) if state.epoch is not None else None,
                            'step': int(state.global_step),
                            'time': datetime.now().isoformat(),
                        }
                        print('METRIC:' + json.dumps(payload), flush=True)
                    except Exception:
                        pass

            trainer = AdvancedTrainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                tokenizer=tokenizer,
                data_collator=data_collator,
                callbacks=[JsonLogger()],
                adv_config=adv_cfg,
            )

            # Log planned steps to the job stream
            try:
                total_steps = trainer.state.max_steps if trainer.state and trainer.state.max_steps else None
                print('METRIC:' + json.dumps({'kind':'plan','stage':stage_name,'stage_index':idx,'total_steps':total_steps,'start_time':datetime.now().isoformat()}), flush=True)
            except Exception:
                pass

            trainer.train()
            global_step = trainer.state.global_step

        # Save checkpoint at each stage end
        ckpt_dir = os.path.join(stage_out, 'checkpoint-final')
        os.makedirs(ckpt_dir, exist_ok=True)
        # If PEFT model, save adapter; else full model
        lora_cfg = config.get('lora') or {}
        try:
            if PeftModel is not None and isinstance(model, PeftModel):
                adapter_name = lora_cfg.get('name') or f"adapter-{stage_name}"
                adapter_dir = os.path.join(stage_out, 'adapters', adapter_name)
                os.makedirs(adapter_dir, exist_ok=True)
                model.save_pretrained(adapter_dir)
                print(f"Saved LoRA adapter for stage '{stage_name}' to: {adapter_dir}")
            else:
                model.save_pretrained(ckpt_dir)
                tokenizer.save_pretrained(ckpt_dir)
        except Exception as e:
            print(f"Warning: failed to save stage checkpoint: {e}")

        # Emit stage completion metric
        try:
            print('METRIC:' + json.dumps({'kind':'stage_complete','stage':stage_name,'stage_index':idx,'global_step':global_step,'time':datetime.now().isoformat()}), flush=True)
        except Exception:
            pass

    # Final save (post last stage)
    os.makedirs(output_dir, exist_ok=True)
    lora_cfg = config.get('lora') or {}
    adapter_name = lora_cfg.get('name') or f"adapter-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    adapter_dir = os.path.join(output_dir, 'adapters', adapter_name)

    if PeftModel is not None and isinstance(model, PeftModel):
        os.makedirs(adapter_dir, exist_ok=True)
        model.save_pretrained(adapter_dir)
        print(f"Saved LoRA adapter to: {adapter_dir}")
        if bool(lora_cfg.get('merge_adapters', False)):
            base_model = model.merge_and_unload()
            base_model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            print("Merged adapter into base model and saved full weights.")
    else:
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

    # Persist job config snapshot
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump({
            'name': config.get('name', f'HF Finetuned {job_id[:8]}'),
            'framework': 'huggingface',
            'base_model': config.get('model_name'),
            'task_type': config.get('task_type', 'classification'),
            'created': datetime.now().isoformat(),
            'lora': lora_cfg,
            'quantization': config.get('quantization'),
            'stages': config.get('stages'),
            'distillation': config.get('distillation'),
            'distributed': config.get('distributed'),
            'precision': config.get('precision') or config.get('compute_dtype'),
        }, f, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--job-id', required=True)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()

    cfg = json.loads(args.config)
    finetune_hf(cfg, args.job_id)
