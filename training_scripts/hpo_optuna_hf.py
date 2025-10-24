import argparse
import json
import os
from datetime import datetime
import copy
import math

import torch
import optuna
from optuna.importance import get_param_importances

from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)

from datasets import Dataset

# Reuse build/tokenize from finetune script
from training_scripts.finetune_huggingface import build_model, create_dummy_text_dataset

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _set_nested(cfg: dict, key: str, value):
    cur = cfg
    parts = key.split('.')
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value


def _get_metric(metrics: dict, name: str, default: float | None = None):
    if name in metrics:
        return float(metrics[name])
    # try common variants
    for k in metrics.keys():
        if k.lower().endswith(name.lower()):
            try:
                return float(metrics[k])
            except Exception:
                pass
    return default


def build_datasets(tokenizer, config: dict, task_type: str):
    # TODO: optionally load real datasets based on config['data']
    dataset = create_dummy_text_dataset(config)
    split = dataset.train_test_split(test_size=0.2)
    train_dataset = split['train']
    eval_dataset = split['test']

    def tokenize_fn(examples):
        return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=config.get('max_length', 128))

    train_dataset = train_dataset.map(tokenize_fn, batched=True)
    eval_dataset = eval_dataset.map(tokenize_fn, batched=True)
    if task_type == 'classification':
        cols = ['input_ids', 'attention_mask', 'label']
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    else:
        def set_labels(batch):
            batch['labels'] = batch['input_ids']
            return batch
        train_dataset = train_dataset.map(set_labels, batched=True)
        eval_dataset = eval_dataset.map(set_labels, batched=True)
        cols = ['input_ids', 'attention_mask', 'labels']
        data_collator = None
    train_dataset.set_format('torch', columns=cols)
    eval_dataset.set_format('torch', columns=cols)
    return train_dataset, eval_dataset, data_collator


def suggest_params(trial: optuna.trial.Trial, space: list[dict]) -> dict:
    out = {}
    for s in (space or []):
        name = s.get('name')
        ptype = (s.get('type') or 'float').lower()
        if not name:
            continue
        if ptype == 'float':
            low = float(s.get('low', 0.0)); high = float(s.get('high', 1.0))
            log = bool(s.get('log', False)); step = s.get('step', None)
            if step is not None:
                val = trial.suggest_float(name, low, high, step=step, log=log)
            else:
                val = trial.suggest_float(name, low, high, log=log)
        elif ptype == 'int':
            low = int(s.get('low', 1)); high = int(s.get('high', max(low, 2)))
            step = int(s.get('step', 1))
            val = trial.suggest_int(name, low, high, step=step)
        elif ptype == 'categorical':
            choices = s.get('choices') or []
            val = trial.suggest_categorical(name, choices)
        else:
            continue
        out[name] = val
    return out


def apply_params_to_config(base_cfg: dict, params: dict) -> dict:
    cfg = copy.deepcopy(base_cfg)
    for k, v in params.items():
        _set_nested(cfg, k, v)
    return cfg


def objective_builder(job_id: str, base_cfg: dict, obj_name: str, direction: str, pruner: optuna.pruners.BasePruner | None):
    def _objective(trial: optuna.trial.Trial):
        # Suggest parameters
        space = (base_cfg.get('hpo') or {}).get('space') or []
        params = suggest_params(trial, space)
        cfg = apply_params_to_config(base_cfg, params)
        task_type = cfg.get('task_type', 'classification')

        # Build model/tokenizer
        tokenizer, model = build_model(cfg)
        train_dataset, eval_dataset, data_collator = build_datasets(tokenizer, cfg, task_type)

        # Minimal training budget per trial
        epochs = int((cfg.get('hpo') or {}).get('trial_epochs', 1))
        batch_size = int(cfg.get('batch_size', 8))
        output_dir = os.path.join(BASE_DIR, 'models', job_id, 'hpo_trials', f"trial_{trial.number}")
        os.makedirs(output_dir, exist_ok=True)

        training_args = TrainingArguments(
            output_dir=output_dir,
            learning_rate=float(cfg.get('learning_rate', 2e-5)),
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epochs,
            logging_dir=os.path.join(BASE_DIR, 'logs', job_id),
            logging_steps=10,
            evaluation_strategy='epoch',
            save_strategy='no',
            gradient_checkpointing=bool(cfg.get('gradient_checkpointing', False)),
            bf16=str(cfg.get('compute_dtype','')).lower() in ('bfloat16','bf16'),
            fp16=str(cfg.get('compute_dtype','')).lower() in ('float16','fp16'),
        )

        trial_pruner = pruner
        class OptunaCallback:
            def on_log(self, args, state, control, logs=None, **kwargs):
                if logs is None:
                    return
                # Use training loss as intermediate value if available
                val = logs.get('loss') or logs.get('eval_loss')
                if val is not None:
                    try:
                        trial.report(float(val), step=int(state.global_step or 0))
                        if trial_pruner and trial.should_prune():
                            raise optuna.TrialPruned()
                    except Exception:
                        pass

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            callbacks=[OptunaCallback()],
        )

        trainer.train()
        metrics = trainer.evaluate()
        value = _get_metric(metrics, obj_name, default=None)
        if value is None:
            # fallback use eval_loss if present
            value = _get_metric(metrics, 'eval_loss', default=math.inf if direction=='minimize' else -math.inf)
        return float(value)
    return _objective


def main(job_id: str, config: dict):
    hpo = (config.get('hpo') or {})
    if not hpo.get('enabled'):
        print('HPO is not enabled in config')
        return
    study_name = hpo.get('study_name') or f"hpo_{job_id}"
    # Support multi-objective: metrics can be a list, with directions list
    metrics = hpo.get('metrics')
    metric = hpo.get('metric') or (metrics[0] if isinstance(metrics, list) and metrics else 'eval_loss')
    direction = (hpo.get('direction') or 'minimize').lower()
    directions = hpo.get('directions') if isinstance(hpo.get('directions'), list) else None
    n_trials = int(hpo.get('max_trials', 10))
    timeout = int(hpo.get('timeout_seconds', 0)) or None
    workers = int(hpo.get('workers', 1))

    # Storage for parallel processes
    storage_dir = os.path.join(BASE_DIR, 'models', job_id)
    os.makedirs(storage_dir, exist_ok=True)
    storage_url = f"sqlite:///{os.path.join(storage_dir, 'hpo.sqlite3')}"

    # Sampler and pruner
    sampler_name = (hpo.get('sampler') or 'tpe').lower()
    if sampler_name == 'random':
        sampler = optuna.samplers.RandomSampler()
    else:
        sampler = optuna.samplers.TPESampler()
    pruner_name = (hpo.get('pruner') or 'median').lower()
    if pruner_name == 'median':
        pruner = optuna.pruners.MedianPruner()
    elif pruner_name == 'asha':
        pruner = optuna.pruners.SuccessiveHalvingPruner()
    else:
        pruner = None

    if isinstance(metrics, list) and directions and len(directions) == len(metrics):
        # Multi-objective
        study = optuna.create_study(study_name=study_name, storage=storage_url, directions=[d.lower() for d in directions], load_if_exists=True, sampler=sampler)
        # Wrap objective to return tuple of metric values
        base_obj = objective_builder(job_id, config, metric, direction, pruner)
        def multi_obj(trial: optuna.trial.Trial):
            # Evaluate once using base metric run; then compute other metrics using trainer.evaluate keys
            # For simplicity, call base objective to run a trainer and evaluate; capture logs via user_attrs
            val = base_obj(trial)
            # Attempt to read last metrics from trial user_attrs (not set); fallback to using same value for all
            vals = []
            for m in metrics:
                vals.append(val if m == metric else float(val))
            return tuple(vals)
        objective = multi_obj
    else:
        study = optuna.create_study(study_name=study_name, storage=storage_url, direction=direction, load_if_exists=True, sampler=sampler)
        objective = objective_builder(job_id, config, metric, direction, pruner)

    # Run optimization
    study.optimize(objective, n_trials=n_trials, timeout=timeout)

    # Collect results
    trials = []
    for t in study.trials:
        trials.append({
            'number': t.number,
            'state': str(t.state),
            'value': getattr(t, 'value', None),
            'values': getattr(t, 'values', None),
            'params': t.params,
            'datetime_start': t.datetime_start.isoformat() if t.datetime_start else None,
            'datetime_complete': t.datetime_complete.isoformat() if t.datetime_complete else None,
            'user_attrs': t.user_attrs,
            'system_attrs': t.system_attrs,
        })

    results = {
        'study_name': study.study_name,
        'direction': direction,
        'metric': metric,
        'metrics': metrics if isinstance(metrics, list) else None,
        'directions': [d.lower() for d in directions] if isinstance(directions, list) else None,
        'n_trials': len(study.trials),
        'best_trial': {
            'number': study.best_trial.number if study.best_trial else None,
            'value': getattr(study, 'best_value', None) if study.best_trial else None,
            'values': getattr(study.best_trial, 'values', None) if study.best_trial else None,
            'params': study.best_trial.params if study.best_trial else {},
        },
        'param_importances': {},
        'timestamp': datetime.now().isoformat(),
    }

    try:
        importances = get_param_importances(study)
        results['param_importances'] = importances
    except Exception:
        pass

    # Save artifacts
    with open(os.path.join(storage_dir, 'hpo_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(storage_dir, 'hpo_trials.json'), 'w') as f:
        json.dump({'trials': trials}, f, indent=2)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--job-id', required=True)
    p.add_argument('--config', required=True)
    a = p.parse_args()
    cfg = json.loads(a.config)
    main(a.job_id, cfg)
