import argparse
import json
import os
from datetime import datetime
import math
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from training_scripts.train_pytorch import CustomModel, CustomResNet


def build_dataset(cfg: dict):
    num_samples = int(cfg.get('num_samples', 1000))
    input_size = int(cfg.get('input_size', 784))
    output_size = int(cfg.get('output_size', 10))
    X = torch.randn(num_samples, input_size)
    y = torch.randint(0, output_size, (num_samples,))
    return TensorDataset(X, y)


def objective_builder(cfg: dict, device: torch.device, metric: str, direction: str):
    def _objective(trial: optuna.trial.Trial):
        # suggest
        space = (cfg.get('hpo') or {}).get('space') or [
            { 'name':'learning_rate', 'type': 'float', 'low': 1e-5, 'high': 1e-2, 'log': True },
            { 'name':'batch_size', 'type': 'int', 'low': 8, 'high': 128, 'step': 8 },
        ]
        params = {}
        for s in space:
            name = s.get('name'); typ=(s.get('type') or 'float').lower()
            if not name: continue
            if typ=='float':
                low=float(s.get('low',0.0)); high=float(s.get('high',1.0)); log=bool(s.get('log',False)); step=s.get('step')
                v = trial.suggest_float(name, low, high, log=log) if step is None else trial.suggest_float(name, low, high, step=step, log=log)
            elif typ=='int':
                low=int(s.get('low',1)); high=int(s.get('high',2)); step=int(s.get('step',1))
                v = trial.suggest_int(name, low, high, step=step)
            elif typ=='categorical':
                v = trial.suggest_categorical(name, s.get('choices') or [])
            else:
                continue
            params[name]=v
        lr = float(params.get('learning_rate', cfg.get('learning_rate', 1e-3)))
        batch_size = int(params.get('batch_size', cfg.get('batch_size', 32)))

        # model
        arch = cfg.get('architecture','custom')
        model = (CustomResNet(cfg) if arch=='resnet' else CustomModel(cfg)).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=lr)

        ds = build_dataset(cfg)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
        epochs = int((cfg.get('hpo') or {}).get('trial_epochs', 1))
        global_step = 0
        best_metric = math.inf if direction=='minimize' else -math.inf
        for ep in range(epochs):
            model.train(); total_loss=0; correct=0; total=0
            for i,(x,y) in enumerate(loader):
                x=x.to(device); y=y.to(device)
                optimizer.zero_grad(); out=model(x); loss=criterion(out,y); loss.backward(); optimizer.step()
                total_loss += float(loss.item())
                pred = out.argmax(dim=1); total += y.size(0); correct += (pred==y).sum().item()
                global_step += 1
                trial.report(float(loss.item()), step=global_step)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            avg_loss = total_loss / max(1,len(loader))
            acc = correct / max(1,total)
            val = avg_loss if metric=='loss' or metric=='eval_loss' else (acc if metric=='accuracy' else avg_loss)
            if direction=='minimize': best_metric = min(best_metric, val)
            else: best_metric = max(best_metric, val)
        return best_metric
    return _objective


def main(job_id: str, config: dict):
    hpo = (config.get('hpo') or {})
    if not hpo.get('enabled'):
        print('HPO disabled')
        return
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    metric = (hpo.get('metric') or 'eval_loss')
    direction = (hpo.get('direction') or 'minimize').lower()
    n_trials = int(hpo.get('max_trials', 10))
    timeout = int(hpo.get('timeout_seconds', 0)) or None
    study_name = hpo.get('study_name') or f"hpo_torch_{job_id}"
    storage_dir = os.path.join(BASE_DIR, 'models', job_id)
    os.makedirs(storage_dir, exist_ok=True)
    storage_url = f"sqlite:///{os.path.join(storage_dir, 'hpo_torch.sqlite3')}"
    sampler = optuna.samplers.TPESampler() if (hpo.get('sampler') or 'tpe').lower()=='tpe' else optuna.samplers.RandomSampler()
    pruner = (optuna.pruners.MedianPruner() if (hpo.get('pruner') or 'median').lower()=='median' else optuna.pruners.SuccessiveHalvingPruner()) if (hpo.get('pruner') or '').lower()!='none' else None

    study = optuna.create_study(study_name=study_name, storage=storage_url, direction=direction, load_if_exists=True, sampler=sampler)
    objective = objective_builder(config, device, metric, direction)
    study.optimize(objective, n_trials=n_trials, timeout=timeout)

    trials = [{ 'number': t.number, 'state': str(t.state), 'value': t.value, 'params': t.params, 'datetime_start': t.datetime_start.isoformat() if t.datetime_start else None, 'datetime_complete': t.datetime_complete.isoformat() if t.datetime_complete else None } for t in study.trials]
    results = {
        'study_name': study.study_name,
        'direction': direction,
        'metric': metric,
        'n_trials': len(trials),
        'best_trial': { 'number': study.best_trial.number if study.best_trial else None, 'value': study.best_value if study.best_trial else None, 'params': study.best_trial.params if study.best_trial else {} },
        'timestamp': datetime.now().isoformat()
    }
    with open(os.path.join(storage_dir, 'hpo_results_torch.json'), 'w') as f: json.dump(results, f, indent=2)
    with open(os.path.join(storage_dir, 'hpo_trials_torch.json'), 'w') as f: json.dump({'trials':trials}, f, indent=2)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--job-id', required=True)
    p.add_argument('--config', required=True)
    a = p.parse_args()
    cfg = json.loads(a.config)
    main(a.job_id, cfg)

