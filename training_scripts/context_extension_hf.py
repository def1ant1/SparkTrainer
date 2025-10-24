import argparse
import json
import os
from datetime import datetime
import math
import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    TrainerCallback,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ==============================
# Advanced Trainer + helpers
# ==============================

class AdvancedTrainer(Trainer):
    """HF Trainer extension with optional Knowledge Distillation and custom grad clipping.

    adv_config keys:
      - distillation: { enabled, teacher_model|multi_teachers[], temperature, alpha_distill, alpha_ce }
      - grad_clip: { type: 'norm'|'value', max_norm, max_value }
      - task_type: 'generation'|'classification'
    """
    def __init__(self, *args, **kwargs):
        self._adv_cfg = kwargs.pop('adv_config', {}) or {}
        super().__init__(*args, **kwargs)
        self._teacher_models = []
        self._distill_cfg = (self._adv_cfg.get('distillation') or {}) if isinstance(self._adv_cfg, dict) else {}
        if self._distill_cfg.get('enabled'):
            self._init_teachers()

    def _init_teachers(self):
        tcfg = self._distill_cfg
        tids = []
        if tcfg.get('multi_teachers'):
            tids = [str(x) for x in tcfg.get('multi_teachers')]
        elif tcfg.get('teacher_model'):
            tids = [str(tcfg.get('teacher_model'))]
        if not tids:
            print('KD enabled but no teacher specified; disabling.')
            self._distill_cfg['enabled'] = False
            return
        ttype = (self._adv_cfg.get('task_type') or '').lower()
        for tid in tids:
            try:
                if ttype == 'classification':
                    tm = AutoModelForSequenceClassification.from_pretrained(tid)
                else:
                    tm = AutoModelForCausalLM.from_pretrained(tid)
            except Exception:
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
        return super().clip_gradients(optimizer, max_norm)

    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        loss = outputs.get('loss') if isinstance(outputs, dict) else outputs[0]
        if self._distill_cfg.get('enabled') and self._teacher_models:
            temperature = float(self._distill_cfg.get('temperature', 2.0))
            alpha_kd = float(self._distill_cfg.get('alpha_distill', 0.5))
            alpha_ce = float(self._distill_cfg.get('alpha_ce', 0.5))
            s_logits = outputs.get('logits') if isinstance(outputs, dict) else (outputs[1] if len(outputs) > 1 else None)
            if s_logits is not None:
                with torch.no_grad():
                    t_logits_sum = None
                    for tm in self._teacher_models:
                        tout = tm(**{k: v for k, v in inputs.items() if k in ('input_ids','attention_mask','token_type_ids')})
                        tl = tout.get('logits') if isinstance(tout, dict) else tout[0]
                        t_logits_sum = tl if t_logits_sum is None else (t_logits_sum + tl)
                    t_logits = t_logits_sum / float(len(self._teacher_models))
                kd_loss_fn = torch.nn.KLDivLoss(reduction='batchmean')
                s_log_probs = torch.log_softmax(s_logits / temperature, dim=-1)
                t_probs = torch.softmax(t_logits / temperature, dim=-1)
                kd_loss = kd_loss_fn(s_log_probs, t_probs) * (temperature ** 2)
                loss = alpha_ce * loss + alpha_kd * kd_loss
                try:
                    print('METRIC:' + json.dumps({'kind':'kd','kd_loss': float(kd_loss.detach().cpu().item())}))
                except Exception:
                    pass
        return (loss, outputs) if return_outputs else loss


def _build_training_args(output_dir: str, base_logs_dir: str, config: dict, stage_cfg: dict) -> TrainingArguments:
    s = stage_cfg or {}
    dist = config.get('distributed') or {}
    precision = str(s.get('precision', config.get('precision', ''))).lower()
    compute_dtype = str(s.get('compute_dtype', config.get('compute_dtype',''))).lower()
    if not compute_dtype and precision in ('fp16','float16'):
        compute_dtype = 'fp16'
    if not compute_dtype and precision in ('bf16','bfloat16'):
        compute_dtype = 'bf16'
    if precision == 'fp8':
        print('Warning: FP8 requested; ensure model and environment support. Proceeding without explicit Trainer flag.')
    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=int(s.get('batch_size', config.get('batch_size', 1))),
        per_device_eval_batch_size=int(s.get('eval_batch_size', s.get('batch_size', config.get('batch_size', 1)))) ,
        num_train_epochs=int(s.get('epochs', config.get('stage_epochs', 1))),
        logging_dir=base_logs_dir,
        logging_steps=int(config.get('logging_steps', 10)),
        evaluation_strategy=str(config.get('evaluation_strategy', 'epoch')),
        save_strategy=str(s.get('save_strategy', config.get('save_strategy', 'no'))),
        gradient_checkpointing=bool(s.get('gradient_checkpointing', config.get('gradient_checkpointing', True))),
        bf16=compute_dtype in ('bfloat16','bf16'),
        fp16=compute_dtype in ('float16','fp16'),
        gradient_accumulation_steps=int(s.get('gradient_accumulation_steps', config.get('gradient_accumulation_steps', 1))),
        ddp_find_unused_parameters=bool((dist or {}).get('ddp_find_unused_parameters', False)) if dist else None,
        deepspeed=dist.get('deepspeed') if isinstance(dist.get('deepspeed'), (str, dict)) else None,
        fsdp=dist.get('fsdp') if isinstance(dist.get('fsdp'), str) else None,
        fsdp_min_num_params=int(dist.get('fsdp_min_num_params', 0)) if dist else None,
    )
    return args


def _apply_curriculum_subset(dataset: Dataset, curriculum: dict, epoch_idx: int, total_epochs: int) -> Dataset:
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
    texts = dataset['text'] if 'text' in dataset.column_names else []
    if mode == 'random':
        scores = np.random.rand(len(texts)).astype(float).tolist()
    else:
        scores = [float(len(t or '')) for t in texts]
    idx_sorted = np.argsort(np.array(scores))
    n = max(1, int(np.ceil(len(idx_sorted) * frac)))
    chosen = idx_sorted[:n].tolist()
    return dataset.select(chosen)


def _dummy_long_dataset(tokenizer, length_tokens=4096, samples=64):
    # Generate synthetic long sequences of tokens by repeating a pattern
    text = ("Lorem ipsum dolor sit amet, consectetur adipiscing elit. ") * (length_tokens // 8)
    data = { 'text': [text for _ in range(samples)], 'label': [0 for _ in range(samples)] }
    return Dataset.from_dict(data)


def _apply_rope_scaling(model, tokenizer, rope_cfg):
    if not rope_cfg:
        return
    method = (rope_cfg.get('method') or 'linear').lower()  # linear|dynamic|yarn
    target = int(rope_cfg.get('target_length') or (tokenizer.model_max_length or 2048))
    base_ctx = getattr(model.config, 'max_position_embeddings', None) or getattr(tokenizer, 'model_max_length', 2048)
    try:
        base_ctx = int(base_ctx)
    except Exception:
        base_ctx = 2048
    factor = max(1.0, float(target) / float(base_ctx))
    try:
        model.config.rope_scaling = { 'name': 'yarn' if method == 'yarn' else method, 'factor': factor }
        # For some models, also bump tokenizer limit
        try:
            tokenizer.model_max_length = target
        except Exception:
            pass
        print(f"Applied RoPE scaling: method={method}, factor={factor:.3f}, target={target}")
    except Exception as e:
        print(f"Warning: Failed to set rope_scaling: {e}")


def _configure_longlora(model, longlora_cfg):
    if not longlora_cfg or not longlora_cfg.get('enabled'):
        return
    # Best-effort toggles; true LongLoRA requires specialized kernels.
    # Enable gradient checkpointing and try sliding window if supported
    try:
        model.gradient_checkpointing_enable()
    except Exception:
        pass
    if getattr(model.config, 'sliding_window', None) is not None:
        try:
            model.config.sliding_window = max(2048, int(longlora_cfg.get('sliding_window') or 4096))
        except Exception:
            pass
    # Shifted sparse attention not implemented; placeholder toggle
    if longlora_cfg.get('shifted_sparse_attention'):
        print("Note: shifted_sparse_attention requested; using default attention as placeholder.")


def _compute_perplexity(model, tokenizer, texts, max_length):
    model.eval()
    nlls = []
    with torch.no_grad():
        for t in texts:
            enc = tokenizer(t, return_tensors='pt', truncation=True, max_length=max_length)
            input_ids = enc['input_ids'].to(model.device)
            labels = input_ids.clone()
            out = model(input_ids=input_ids, labels=labels)
            nlls.append(out.loss.item())
    ppl = math.exp(sum(nlls) / max(1, len(nlls)))
    return ppl


def _needle_in_haystack_eval(model, tokenizer, ctx_length):
    # Synthetic: place "NEEDLE:XYZ" near start, ask to output XYZ
    token = str(np.random.randint(10000, 99999))
    context = ("word " * (ctx_length//5)) + f" START NEEDLE:{token} END " + (" filler" * (ctx_length//5))
    question = "\nQ: What is the NEEDLE? A:"
    full = context + question
    enc = tokenizer(full, return_tensors='pt', truncation=True, max_length=ctx_length)
    input_ids = enc['input_ids'].to(model.device)
    with torch.no_grad():
        out = model(input_ids=input_ids)
        logits = out.logits[:, -1, :]
        # Score digits; approximate correctness if first digit has high prob
        top_id = int(torch.argmax(logits, dim=-1).item())
        pred = tokenizer.decode([top_id]).strip()
    correct = pred.startswith(token[0])  # weak proxy
    return {'pred_token': pred, 'target': token, 'correct_head': bool(correct)}


def _long_qa_eval(model, tokenizer, ctx_length):
    # Synthetic QA: place a fact and ask for it
    fact = "The color of the sky is cerulean."
    question = "\nQ: What is the color of the sky? A:"
    padding = (" context" * (ctx_length//7))
    full = padding + " " + fact + padding + question
    enc = tokenizer(full, return_tensors='pt', truncation=True, max_length=ctx_length)
    input_ids = enc['input_ids'].to(model.device)
    with torch.no_grad():
        logits = model(input_ids=input_ids).logits[:, -1, :]
        probs = torch.softmax(logits, dim=-1)
        tok = tokenizer.encode("cerulean", add_special_tokens=False)
        score = float(probs[0, tok[0]].item()) if tok else 0.0
    return {'score_cerulean': score}


class JsonLogger(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        try:
            payload = { 'kind': 'log', 'step': int(state.global_step), 'epoch': float(state.epoch) if state.epoch is not None else None, 'logs': logs or {}, 'time': datetime.now().isoformat() }
            print('METRIC:' + json.dumps(payload), flush=True)
        except Exception:
            pass
    def on_epoch_end(self, args, state, control, **kwargs):
        try:
            payload = { 'kind': 'epoch', 'epoch': float(state.epoch) if state.epoch is not None else None, 'step': int(state.global_step), 'time': datetime.now().isoformat() }
            print('METRIC:' + json.dumps(payload), flush=True)
        except Exception:
            pass


def main(job_id, config):
    model_name = config.get('model_name', 'gpt2')
    task_type = config.get('task_type', 'generation')
    ctx = config.get('context_extension') or {}
    rope = ctx.get('rope') or {}
    longlora = ctx.get('longlora') or {}
    schedule = ctx.get('schedule') or [4096, 8192, 16384, 32768]
    eval_cfg = ctx.get('eval') or {'run_ppl': True, 'run_needle': True, 'run_long_qa': True, 'lengths': schedule}

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if task_type == 'classification':
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=config.get('num_classes', 2))
        data_collator = DataCollatorWithPadding(tokenizer)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name)
        data_collator = None

    # Apply long-context configs
    _apply_rope_scaling(model, tokenizer, rope)
    _configure_longlora(model, longlora)

    output_dir = os.path.join(BASE_DIR, 'models', job_id)
    os.makedirs(output_dir, exist_ok=True)

    # Optional external "stages" hyperparams to pair with schedule
    stages_cfg = config.get('stages') or []
    base_logs_dir = os.path.join(BASE_DIR, 'logs', job_id)

    # Distributed execution note
    dist = config.get('distributed') or {}
    if dist.get('nnodes') or dist.get('nproc_per_node'):
        print('Note: Multi-process/multi-node typically requires torchrun. Backend launches single process.')

    # Multi-stage training over scheduled context lengths
    for i, tgt in enumerate(schedule):
        print(f"\n=== Stage {i+1}/{len(schedule)}: target_ctx={tgt} ===")
        _apply_rope_scaling(model, tokenizer, { 'method': rope.get('method', 'linear'), 'target_length': tgt })
        dataset = _dummy_long_dataset(tokenizer, length_tokens=tgt, samples=config.get('stage_samples', 128))

        # Determine hyperparams for this stage (fallback to top-level)
        stage_hp = (stages_cfg[i] if i < len(stages_cfg) else {}) if isinstance(stages_cfg, list) else {}
        stage_name = stage_hp.get('name', f'stage{i+1}')

        # Curriculum support (incremental loops)
        curriculum = stage_hp.get('curriculum') or {}
        incremental = bool(curriculum.get('incremental', False))
        loops = int(stage_hp.get('epochs', config.get('stage_epochs', 1))) if incremental else 1
        epochs_per_loop = 1 if incremental else int(stage_hp.get('epochs', config.get('stage_epochs', 1)))

        for loop_idx in range(loops):
            ds_loop = _apply_curriculum_subset(dataset, curriculum, loop_idx, loops) if curriculum else dataset
            split = ds_loop.train_test_split(test_size=0.1)
            train_dataset = split['train']; eval_dataset = split['test']
            if task_type == 'generation':
                def set_labels(batch): batch['labels'] = batch['input_ids']; return batch
                train_dataset = train_dataset.map(lambda x: tokenizer(x['text'], truncation=True, max_length=tgt), batched=True)
                eval_dataset = eval_dataset.map(lambda x: tokenizer(x['text'], truncation=True, max_length=tgt), batched=True)
                train_dataset = train_dataset.map(set_labels, batched=True)
                eval_dataset = eval_dataset.map(set_labels, batched=True)
                cols = ['input_ids', 'attention_mask', 'labels']
            else:
                train_dataset = train_dataset.map(lambda x: tokenizer(x['text'], truncation=True, max_length=tgt), batched=True)
                eval_dataset = eval_dataset.map(lambda x: tokenizer(x['text'], truncation=True, max_length=tgt), batched=True)
                cols = ['input_ids', 'attention_mask', 'label']
            train_dataset.set_format('torch', columns=cols)
            eval_dataset.set_format('torch', columns=cols)

            args = _build_training_args(output_dir, base_logs_dir, config, {**stage_hp, 'epochs': epochs_per_loop})
            adv_cfg = {
                'distillation': config.get('distillation') or {},
                'grad_clip': config.get('grad_clip') or {},
                'task_type': task_type,
            }
            trainer = AdvancedTrainer(
                model=model,
                args=args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                tokenizer=tokenizer,
                data_collator=data_collator,
                callbacks=[JsonLogger()],
                adv_config=adv_cfg,
            )
            try:
                total_steps = trainer.state.max_steps if trainer.state and trainer.state.max_steps else None
                print('METRIC:' + json.dumps({'kind':'plan','stage':stage_name,'stage_index':i,'total_steps':total_steps,'start_time':datetime.now().isoformat()}), flush=True)
            except Exception:
                pass
            trainer.train()

        # Evaluate long-context metrics at this stage
        if eval_cfg.get('run_ppl') and eval_cfg.get('lengths'):
            lens = eval_cfg.get('lengths') or [tgt]
            texts = [ ("abcdef ") * (L//2) for L in lens ]
            ppl = { int(L): _compute_perplexity(model, tokenizer, [texts[i]], max_length=int(L)) for i, L in enumerate(lens) }
            print('METRIC:' + json.dumps({'kind':'long_eval','metric':'perplexity','ctx_lengths':list(ppl.keys()),'values':list(ppl.values()),'stage':i+1}))
        if eval_cfg.get('run_needle'):
            res = _needle_in_haystack_eval(model, tokenizer, int(tgt))
            print('METRIC:' + json.dumps({'kind':'long_eval','metric':'needle','result':res,'stage':i+1}))
        if eval_cfg.get('run_long_qa'):
            res = _long_qa_eval(model, tokenizer, int(tgt))
            print('METRIC:' + json.dumps({'kind':'long_eval','metric':'long_qa','result':res,'stage':i+1}))

    # Save final model snapshot/config
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump({
            'name': config.get('name', f'CTX-Extended {job_id[:8]}'),
            'framework': 'huggingface',
            'base_model': model_name,
            'task_type': task_type,
            'created': datetime.now().isoformat(),
            'context_extension': ctx,
            'stages': config.get('stages'),
            'distillation': config.get('distillation'),
            'distributed': config.get('distributed'),
            'precision': config.get('precision') or config.get('compute_dtype'),
        }, f, indent=2)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--job-id', required=True)
    p.add_argument('--config', required=True)
    a = p.parse_args()
    cfg = json.loads(a.config)
    main(a.job_id, cfg)
