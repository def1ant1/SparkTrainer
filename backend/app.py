from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import json
import subprocess
import uuid
from datetime import datetime, timedelta
import threading
import signal
import time
from typing import Dict, Any, Optional
from collections import deque
import zipfile
import io
import re
import shutil
import hashlib
import csv
import random
import secrets
try:
    from PIL import Image
except Exception:
    Image = None

app = Flask(__name__)
CORS(app)

# Global error handlers to ensure JSON responses
@app.errorhandler(404)
def not_found_error(error):
    return jsonify({'error': 'Endpoint not found', 'status': 404}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error', 'status': 500, 'message': str(error)}), 500

@app.errorhandler(Exception)
def handle_exception(e):
    # Handle all unhandled exceptions
    return jsonify({'error': 'Unexpected error', 'message': str(e)}), 500

# Configuration
# Use project-relative paths by default, with environment variable overrides.
BASE_DIR = os.environ.get(
    'DGX_TRAINER_BASE_DIR',
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

def _resolve_dir(env_value: str | None, default_name: str) -> str:
    if env_value:
        return env_value if os.path.isabs(env_value) else os.path.abspath(os.path.join(BASE_DIR, env_value))
    return os.path.join(BASE_DIR, default_name)

JOBS_DIR = _resolve_dir(os.environ.get('JOBS_DIR'), 'jobs')
MODELS_DIR = _resolve_dir(os.environ.get('MODELS_DIR'), 'models')
LOGS_DIR = _resolve_dir(os.environ.get('LOGS_DIR'), 'logs')
TRAINING_SCRIPTS_DIR = _resolve_dir(os.environ.get('TRAINING_SCRIPTS_DIR'), 'training_scripts')
DATASETS_DIR = _resolve_dir(os.environ.get('DATASETS_DIR'), 'datasets')
EXPERIMENTS_DIR = _resolve_dir(os.environ.get('EXPERIMENTS_DIR'), os.path.join('jobs','experiments'))

os.makedirs(JOBS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(DATASETS_DIR, exist_ok=True)
os.makedirs(EXPERIMENTS_DIR, exist_ok=True)

# In-memory job tracking
jobs = {}

# Metrics history storage (last hour)
METRICS_INTERVAL_SECONDS = int(os.environ.get('METRICS_INTERVAL_SECONDS', '5'))
METRICS_WINDOW_SECONDS = int(os.environ.get('METRICS_WINDOW_SECONDS', '3600'))
_metrics_history = deque(maxlen=max(10, METRICS_WINDOW_SECONDS // max(1, METRICS_INTERVAL_SECONDS) + 5))
_metrics_thread = None
_metrics_thread_started = False
_metrics_lock = threading.Lock()
_last_net_totals = {'rx': None, 'tx': None, 'ts': None}
_last_io_totals = {'reads': None, 'writes': None, 'ts': None}
_scheduler_started = False

PIPELINES_PATH = os.path.join(JOBS_DIR, 'pipelines.json')
SCHEDULES_PATH = os.path.join(JOBS_DIR, 'schedules.json')
USERS_PATH = os.path.join(JOBS_DIR, 'users.json')
TEAMS_PATH = os.path.join(JOBS_DIR, 'teams.json')
BILLING_PATH = os.path.join(JOBS_DIR, 'billing.json')

# Simple in-memory session storage (replace with Redis/database in production)
_sessions: Dict[str, Dict[str, Any]] = {}
_api_tokens: Dict[str, str] = {}  # token -> user_id mapping

def _load_json(path: str, default: Any):
    if os.path.exists(path):
        try:
            return json.load(open(path))
        except Exception:
            return default
    return default

def _save_json(path: str, data: Any):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

def load_jobs():
    """Load existing jobs from disk"""
    if os.path.exists(os.path.join(JOBS_DIR, 'jobs.json')):
        with open(os.path.join(JOBS_DIR, 'jobs.json'), 'r') as f:
            return json.load(f)
    return {}

def save_jobs():
    """Save jobs to disk"""
    with open(os.path.join(JOBS_DIR, 'jobs.json'), 'w') as f:
        json.dump(jobs, f, indent=2)

jobs = load_jobs()

# ------------ Experiments -------------
_EXPS_INDEX = os.path.join(EXPERIMENTS_DIR, 'experiments.json')

def _load_experiments() -> Dict[str, Any]:
    if os.path.exists(_EXPS_INDEX):
        try:
            return json.load(open(_EXPS_INDEX))
        except Exception:
            return {}
    return {}

def _save_experiments(exps: Dict[str, Any]):
    with open(_EXPS_INDEX, 'w', encoding='utf-8') as f:
        json.dump(exps, f, indent=2)

def _find_or_create_experiment_by_name(name: str, tags: list[str] | None = None, description: str | None = None) -> Dict[str, Any]:
    exps = _load_experiments()
    for eid, e in exps.items():
        if str(e.get('name','')).strip().lower() == str(name).strip().lower():
            return {'id': eid, **e}
    eid = str(uuid.uuid4())
    now = datetime.now().isoformat()
    rec = {'id': eid, 'name': name, 'tags': tags or [], 'description': description or '', 'favorite': False, 'created': now, 'updated': now}
    exps[eid] = rec
    _save_experiments(exps)
    return rec

@app.route('/api/experiments', methods=['GET','POST'])
def experiments_root():
    if request.method == 'POST':
        data = request.json or {}
        name = (data.get('name') or '').strip()
        if not name:
            return jsonify({'error': 'name required'}), 400
        rec = _find_or_create_experiment_by_name(name, data.get('tags') or [], data.get('description') or '')
        return jsonify(rec), 201
    # GET
    exps = _load_experiments()
    items = list(exps.values())
    q = (request.args.get('q') or '').lower().strip()
    if q:
        items = [e for e in items if q in (e.get('name','').lower() + ' ' + (e.get('description','').lower()))]
    # Attach run counts
    runs_by_exp: Dict[str, int] = {}
    for j in jobs.values():
        eid = (j.get('experiment') or {}).get('id') if isinstance(j.get('experiment'), dict) else None
        if eid:
            runs_by_exp[eid] = runs_by_exp.get(eid, 0) + 1
    for e in items:
        e['run_count'] = runs_by_exp.get(e.get('id'), 0)
    return jsonify({'items': items})

@app.route('/api/experiments/<exp_id>', methods=['GET','PUT'])
def experiments_detail(exp_id):
    exps = _load_experiments()
    if exp_id not in exps:
        return jsonify({'error': 'Not found'}), 404
    if request.method == 'PUT':
        data = request.json or {}
        e = exps[exp_id]
        for k in ('name','description','tags','favorite'):
            if k in data:
                e[k] = data[k]
        e['updated'] = datetime.now().isoformat()
        exps[exp_id] = e
        _save_experiments(exps)
    runs = [j for j in jobs.values() if (isinstance(j.get('experiment'), dict) and j['experiment'].get('id') == exp_id)]
    out = dict(exps[exp_id])
    out['runs'] = runs
    return jsonify(out)

@app.route('/api/experiments/<exp_id>/star', methods=['POST'])
def experiments_star(exp_id):
    exps = _load_experiments()
    if exp_id not in exps:
        return jsonify({'error': 'Not found'}), 404
    e = exps[exp_id]
    e['favorite'] = bool((request.json or {}).get('favorite', True))
    e['updated'] = datetime.now().isoformat()
    exps[exp_id] = e
    _save_experiments(exps)
    return jsonify({'status': 'ok', 'favorite': e['favorite']})

@app.route('/api/experiments/<exp_id>', methods=['DELETE'])
def delete_experiment(exp_id):
    """Delete an experiment"""
    exps = _load_experiments()
    if exp_id not in exps:
        return jsonify({'error': 'Not found'}), 404
    del exps[exp_id]
    _save_experiments(exps)
    return jsonify({'status': 'ok', 'deleted': exp_id})

@app.route('/api/experiments/<exp_id>/jobs', methods=['GET'])
def experiment_jobs(exp_id):
    """Get all jobs associated with an experiment"""
    exps = _load_experiments()
    if exp_id not in exps:
        return jsonify({'error': 'Not found'}), 404

    # Filter jobs by experiment ID
    exp_jobs = [
        j for j in jobs.values()
        if isinstance(j.get('experiment'), dict) and j['experiment'].get('id') == exp_id
    ]

    # Sort by created date (newest first)
    exp_jobs.sort(key=lambda x: x.get('created', ''), reverse=True)

    return jsonify({'jobs': exp_jobs, 'count': len(exp_jobs)})

@app.route('/api/experiments/<exp_id>/metrics', methods=['GET'])
def experiment_metrics(exp_id):
    """Get aggregated metrics for all jobs in an experiment"""
    exps = _load_experiments()
    if exp_id not in exps:
        return jsonify({'error': 'Not found'}), 404

    # Get all jobs for this experiment
    exp_jobs = [
        j for j in jobs.values()
        if isinstance(j.get('experiment'), dict) and j['experiment'].get('id') == exp_id
    ]

    # Aggregate metrics
    metrics = {
        'total_jobs': len(exp_jobs),
        'completed': sum(1 for j in exp_jobs if j.get('status') == 'completed'),
        'running': sum(1 for j in exp_jobs if j.get('status') == 'running'),
        'failed': sum(1 for j in exp_jobs if j.get('status') == 'failed'),
        'pending': sum(1 for j in exp_jobs if j.get('status') == 'pending'),
        'best_metrics': {},
        'job_metrics': []
    }

    # Collect individual job metrics
    for job in exp_jobs:
        job_id = job.get('id')
        job_metrics = {
            'job_id': job_id,
            'name': job.get('name'),
            'status': job.get('status'),
            'created': job.get('created'),
            'metrics': {}
        }

        # Try to load metrics from job output directory
        if job_id:
            model_dir = os.path.join(MODELS_DIR, job_id)
            config_path = os.path.join(model_dir, 'config.json')
            metadata_path = os.path.join(model_dir, 'metadata.json')

            # Try config.json first (PyTorch custom models)
            if os.path.exists(config_path):
                try:
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                        if 'best_val_loss' in config:
                            job_metrics['metrics']['val_loss'] = config['best_val_loss']
                        if 'best_val_accuracy' in config:
                            job_metrics['metrics']['val_accuracy'] = config['best_val_accuracy']
                except Exception:
                    pass

            # Try metadata.json (HuggingFace models)
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                        if 'best_val_loss' in metadata:
                            job_metrics['metrics']['val_loss'] = metadata['best_val_loss']
                        if 'best_val_accuracy' in metadata:
                            job_metrics['metrics']['val_accuracy'] = metadata['best_val_accuracy']
                        if 'eval_results' in metadata:
                            job_metrics['metrics'].update(metadata['eval_results'])
                except Exception:
                    pass

        if job_metrics['metrics']:
            metrics['job_metrics'].append(job_metrics)

            # Track best metrics across all jobs
            for metric_name, metric_value in job_metrics['metrics'].items():
                if isinstance(metric_value, (int, float)):
                    if metric_name not in metrics['best_metrics']:
                        metrics['best_metrics'][metric_name] = {
                            'value': metric_value,
                            'job_id': job_id,
                            'job_name': job.get('name')
                        }
                    else:
                        # For loss metrics, lower is better; for accuracy/f1/etc, higher is better
                        is_loss_metric = 'loss' in metric_name.lower()
                        current_best = metrics['best_metrics'][metric_name]['value']
                        if (is_loss_metric and metric_value < current_best) or \
                           (not is_loss_metric and metric_value > current_best):
                            metrics['best_metrics'][metric_name] = {
                                'value': metric_value,
                                'job_id': job_id,
                                'job_name': job.get('name')
                            }

    return jsonify(metrics)

@app.route('/api/experiments/compare', methods=['POST'])
def compare_experiments():
    """Compare metrics across multiple experiments"""
    data = request.json or {}
    exp_ids = data.get('experiment_ids', [])

    if not exp_ids or not isinstance(exp_ids, list):
        return jsonify({'error': 'experiment_ids array required'}), 400

    exps = _load_experiments()
    comparison = {
        'experiments': [],
        'metrics_comparison': {}
    }

    for exp_id in exp_ids:
        if exp_id not in exps:
            continue

        exp = exps[exp_id]

        # Get jobs for this experiment
        exp_jobs = [
            j for j in jobs.values()
            if isinstance(j.get('experiment'), dict) and j['experiment'].get('id') == exp_id
        ]

        exp_data = {
            'id': exp_id,
            'name': exp.get('name'),
            'description': exp.get('description'),
            'total_jobs': len(exp_jobs),
            'completed_jobs': sum(1 for j in exp_jobs if j.get('status') == 'completed'),
            'metrics': {}
        }

        # Collect metrics from all jobs
        all_metrics = {}
        for job in exp_jobs:
            job_id = job.get('id')
            if not job_id:
                continue

            model_dir = os.path.join(MODELS_DIR, job_id)
            config_path = os.path.join(model_dir, 'config.json')
            metadata_path = os.path.join(model_dir, 'metadata.json')

            for path in [config_path, metadata_path]:
                if os.path.exists(path):
                    try:
                        with open(path, 'r') as f:
                            data_dict = json.load(f)
                            for key in ['best_val_loss', 'best_val_accuracy', 'eval_results']:
                                if key in data_dict:
                                    if key == 'eval_results' and isinstance(data_dict[key], dict):
                                        for metric_name, value in data_dict[key].items():
                                            if isinstance(value, (int, float)):
                                                if metric_name not in all_metrics:
                                                    all_metrics[metric_name] = []
                                                all_metrics[metric_name].append(value)
                                    elif isinstance(data_dict[key], (int, float)):
                                        metric_name = key.replace('best_', '')
                                        if metric_name not in all_metrics:
                                            all_metrics[metric_name] = []
                                        all_metrics[metric_name].append(data_dict[key])
                    except Exception:
                        pass

        # Calculate aggregate metrics
        for metric_name, values in all_metrics.items():
            if values:
                exp_data['metrics'][metric_name] = {
                    'mean': sum(values) / len(values),
                    'min': min(values),
                    'max': max(values),
                    'count': len(values)
                }

        comparison['experiments'].append(exp_data)

    # Build cross-experiment metric comparison
    all_metric_names = set()
    for exp in comparison['experiments']:
        all_metric_names.update(exp['metrics'].keys())

    for metric_name in all_metric_names:
        comparison['metrics_comparison'][metric_name] = {
            'experiments': {}
        }
        for exp in comparison['experiments']:
            if metric_name in exp['metrics']:
                comparison['metrics_comparison'][metric_name]['experiments'][exp['id']] = {
                    'name': exp['name'],
                    'mean': exp['metrics'][metric_name]['mean'],
                    'min': exp['metrics'][metric_name]['min'],
                    'max': exp['metrics'][metric_name]['max']
                }

    return jsonify(comparison)

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

@app.route('/api/frameworks', methods=['GET'])
def get_frameworks():
    """Get available frameworks and their configurations"""
    frameworks = {
        'pytorch': {
            'name': 'PyTorch',
            'versions': ['2.1.0', '2.0.1', '1.13.1'],
            'architectures': ['resnet', 'vgg', 'densenet', 'transformer', 'custom']
        },
        'tensorflow': {
            'name': 'TensorFlow',
            'versions': ['2.14.0', '2.13.0', '2.12.0'],
            'architectures': ['resnet', 'efficientnet', 'mobilenet', 'transformer', 'custom']
        },
        'huggingface': {
            'name': 'Hugging Face Transformers',
            'versions': ['4.35.0', '4.34.0'],
            'architectures': ['bert', 'gpt2', 'llama', 't5', 'custom']
        }
    }
    return jsonify(frameworks)

@app.route('/api/models', methods=['GET'])
def list_models():
    """List models with optional filtering, search, sort."""
    q = (request.args.get('q') or '').strip().lower()
    framework = request.args.get('framework')
    architecture = request.args.get('architecture')
    sort = request.args.get('sort') or 'date'  # date|size|accuracy|name
    order = request.args.get('order') or 'desc'  # asc|desc

    def _size_category(params: int | None, size_bytes: int | None) -> str | None:
        if params is not None:
            try:
                p = int(params)
                if p < 100_000_000: return 'small'
                if p < 500_000_000: return 'base'
                if p < 2_000_000_000: return 'large'
                return 'xl'
            except Exception:
                pass
        if size_bytes is not None:
            if size_bytes < 500*1024*1024: return 'small'
            if size_bytes < 2*1024*1024*1024: return 'base'
            if size_bytes < 8*1024*1024*1024: return 'large'
            return 'xl'
        return None

    def _auto_tags(arch: str | None, cfg: Dict[str, Any], meta: Dict[str, Any]) -> list[str]:
        tags = []
        a = (arch or '').lower()
        if any(k in a for k in ['bert','gpt','llama','t5','transformer','mpt','neox']):
            tags.append('transformer')
        if any(k in a for k in ['resnet','vgg','densenet','vit','conv','cnn']):
            tags.append('vision')
        task = (cfg.get('task_type') or meta.get('task_type') or '').lower()
        if task:
            if any(k in task for k in ['classif','cls']): tags.append('classification')
            if any(k in task for k in ['causal','lm','generation']): tags.append('language-model')
            if 'seg' in task: tags.append('segmentation')
            if 'qa' in task: tags.append('qa')
        dom = (meta.get('domain') or '').lower()
        if dom: tags.append(dom)
        return sorted(list(dict.fromkeys(tags)))

    def _read_stats(path: str) -> Dict[str, Any]:
        sp = os.path.join(path, 'stats.json')
        if os.path.exists(sp):
            try:
                return json.load(open(sp))
            except Exception:
                return {}
        return {}

    def _card_text(path: str, limit: int = 800) -> str:
        p = os.path.join(path, 'card.md')
        if os.path.exists(p):
            try:
                t = open(p, 'r', encoding='utf-8').read()
                return t[:limit]
            except Exception:
                return ''
        return ''

    def load_model(model_dir):
        path = os.path.join(MODELS_DIR, model_dir)
        if not os.path.isdir(path):
            return None
        info = {'id': model_dir}
        cfg = {}
        cfg_path = os.path.join(path, 'config.json')
        if os.path.exists(cfg_path):
            try:
                with open(cfg_path, 'r') as f:
                    cfg = json.load(f)
            except Exception:
                cfg = {}
        meta = {}
        meta_path = os.path.join(path, 'metadata.json')
        if os.path.exists(meta_path):
            try:
                with open(meta_path, 'r') as f:
                    meta = json.load(f)
            except Exception:
                meta = {}
        metrics = {}
        metrics_path = os.path.join(path, 'metrics.json')
        if os.path.exists(metrics_path):
            try:
                with open(metrics_path, 'r') as f:
                    metrics = json.load(f)
            except Exception:
                metrics = {}
        # size
        total_size = 0
        for root, _, files in os.walk(path):
            for fn in files:
                try:
                    total_size += os.path.getsize(os.path.join(root, fn))
                except Exception:
                    pass
        stats = _read_stats(path)
        size_cat = _size_category(cfg.get('parameters'), total_size)
        auto = _auto_tags(cfg.get('architecture') or meta.get('architecture'), cfg, meta)
        license_name = meta.get('license') or cfg.get('license')
        card_excerpt = _card_text(path)
        info.update({
            'name': cfg.get('name', model_dir),
            'framework': cfg.get('framework', meta.get('framework', 'unknown')),
            'architecture': cfg.get('architecture', meta.get('architecture')),
            'created': cfg.get('created', meta.get('created')),
            'parameters': cfg.get('parameters'),
            'metrics': metrics,
            'tags': sorted(list(dict.fromkeys((meta.get('tags') or []) + auto))),
            'size_bytes': total_size,
            'size_category': size_cat,
            'license': license_name,
            'popularity': {
                'views': int(stats.get('views', 0)),
                'exports': int(stats.get('exports', 0)),
                'used_in_jobs': int(stats.get('used_in_jobs', 0)),
            },
            'card_excerpt': card_excerpt,
        })
        return info

    items = []
    if os.path.exists(MODELS_DIR):
        for d in os.listdir(MODELS_DIR):
            m = load_model(d)
            if not m:
                continue
            if framework and m.get('framework') != framework:
                continue
            if architecture and (m.get('architecture') or '') != architecture:
                continue
            if q:
                hay = ' '.join([
                    m.get('name') or '', m.get('framework') or '', m.get('architecture') or '',
                    ' '.join(m.get('tags') or []),
                    (m.get('card_excerpt') or '')
                ]).lower()
                if q not in hay:
                    continue
            items.append(m)

    # Faceted filters
    lic = (request.args.get('license') or '').strip().lower()
    size_f = (request.args.get('size') or '').strip().lower()
    tag_f = (request.args.get('tag') or '').strip().lower()
    domain_f = (request.args.get('domain') or '').strip().lower()
    if lic:
        items = [m for m in items if (str(m.get('license') or '').lower() == lic)]
    if size_f:
        items = [m for m in items if (str(m.get('size_category') or '').lower() == size_f)]
    if tag_f:
        items = [m for m in items if tag_f in [str(t).lower() for t in (m.get('tags') or [])]]
    if domain_f:
        items = [m for m in items if domain_f in [str(t).lower() for t in (m.get('tags') or [])]]

    def sort_key(x):
        if sort == 'size':
            return x.get('size_bytes') or 0
        if sort == 'accuracy':
            return (x.get('metrics') or {}).get('eval_accuracy') or 0
        if sort == 'popular':
            p = x.get('popularity') or {}
            return (p.get('views') or 0) + 2*(p.get('exports') or 0) + (p.get('used_in_jobs') or 0)
        if sort == 'name':
            return (x.get('name') or '').lower()
        # default date
        return (x.get('created') or '')

    items.sort(key=sort_key, reverse=(order != 'asc'))
    return jsonify(items)


def _model_detail(model_id: str) -> Dict[str, Any]:
    path = os.path.join(MODELS_DIR, model_id)
    if not os.path.isdir(path):
        return {}
    cfg = {}
    for fn in ('config.json',):
        p = os.path.join(path, fn)
        if os.path.exists(p):
            try:
                cfg = json.load(open(p))
            except Exception:
                pass
    meta = {}
    p = os.path.join(path, 'metadata.json')
    if os.path.exists(p):
        try:
            meta = json.load(open(p))
        except Exception:
            pass
    metrics = {}
    p = os.path.join(path, 'metrics.json')
    if os.path.exists(p):
        try:
            metrics = json.load(open(p))
        except Exception:
            pass
    # files
    files = []
    for root, _, fns in os.walk(path):
        for fn in fns:
            fp = os.path.join(root, fn)
            try:
                size = os.path.getsize(fp)
            except Exception:
                size = None
            rel = os.path.relpath(fp, path)
            files.append({'path': rel.replace('\\','/'), 'size': size})
    # card
    card_md = None
    p = os.path.join(path, 'card.md')
    if os.path.exists(p):
        try:
            card_md = open(p, 'r', encoding='utf-8').read()
        except Exception:
            card_md = None
    # card versions
    versions = []
    vdir = os.path.join(path, 'cards')
    if os.path.isdir(vdir):
        for fn in sorted(os.listdir(vdir)):
            if fn.endswith('.md'):
                versions.append(fn)
    # size and params
    size_bytes = 0
    for f in files:
        size_bytes += f.get('size') or 0
    return {
        'id': model_id,
        'config': cfg,
        'metadata': meta,
        'metrics': metrics,
        'files': files,
        'card_md': card_md,
        'card_versions': versions,
        'size_bytes': size_bytes,
    }


@app.route('/api/models/<model_id>', methods=['GET'])
def get_model_detail(model_id):
    base = os.path.join(MODELS_DIR, model_id)
    if not os.path.isdir(base):
        return jsonify({'error': 'Model not found'}), 404
    # increment view counter
    try:
        sp = os.path.join(base, 'stats.json')
        stats = {}
        if os.path.exists(sp):
            try: stats = json.load(open(sp))
            except Exception: stats = {}
        stats['views'] = int(stats.get('views', 0)) + 1
        with open(sp, 'w', encoding='utf-8') as f:
            json.dump(stats, f)
    except Exception:
        pass
    # enrich detail with assets
    detail = _model_detail(model_id)
    try:
        images = []
        videos = []
        for sub in ('assets','screenshots','media'):
            ap = os.path.join(base, sub)
            if not os.path.isdir(ap):
                continue
            for r, _, fns in os.walk(ap):
                for fn in fns:
                    rel = os.path.relpath(os.path.join(r, fn), base).replace('\\','/')
                    ext = os.path.splitext(fn)[1].lower()
                    if ext in {'.png','.jpg','.jpeg','.gif'}:
                        images.append(rel)
                    if ext in {'.mp4','.webm','.mov'}:
                        videos.append(rel)
        detail['assets'] = {'images': images[:24], 'videos': videos[:8]}
    except Exception:
        detail['assets'] = {'images': [], 'videos': []}
    return jsonify(detail)


@app.route('/api/models/<model_id>/metadata', methods=['PUT'])
def update_model_metadata(model_id):
    path = os.path.join(MODELS_DIR, model_id)
    if not os.path.isdir(path):
        return jsonify({'error': 'Model not found'}), 404
    data = request.json or {}
    meta_path = os.path.join(path, 'metadata.json')
    cur = {}
    if os.path.exists(meta_path):
        try:
            cur = json.load(open(meta_path))
        except Exception:
            cur = {}
    cur.update(data)
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(cur, f, indent=2)
    return jsonify({'status': 'ok', 'metadata': cur})


@app.route('/api/models/<model_id>/card', methods=['PUT'])
def update_model_card(model_id):
    path = os.path.join(MODELS_DIR, model_id)
    if not os.path.isdir(path):
        return jsonify({'error': 'Model not found'}), 404
    data = request.json or {}
    content = data.get('content') or ''
    author = data.get('author') or ''
    note = data.get('message') or ''
    # Write latest
    with open(os.path.join(path, 'card.md'), 'w', encoding='utf-8') as f:
        f.write(content)
    # Versioned
    vdir = os.path.join(path, 'cards')
    os.makedirs(vdir, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d-%H%M%S')
    vname = f'{ts}.md'
    with open(os.path.join(vdir, vname), 'w', encoding='utf-8') as f:
        meta = f"<!-- author: {author} | note: {note} | time: {ts} -->\n"
        f.write(meta + content)
    return jsonify({'status': 'ok', 'version': vname})


@app.route('/api/models/bulk_delete', methods=['POST'])
def bulk_delete_models():
    ids = (request.json or {}).get('ids') or []
    deleted = []
    for mid in ids:
        p = os.path.join(MODELS_DIR, mid)
        if os.path.isdir(p):
            try:
                # destructive: remove tree
                for root, dirs, files in os.walk(p, topdown=False):
                    for fn in files:
                        try: os.remove(os.path.join(root, fn))
                        except Exception: pass
                    for d in dirs:
                        try: os.rmdir(os.path.join(root, d))
                        except Exception: pass
                os.rmdir(p)
                deleted.append(mid)
            except Exception:
                pass
    return jsonify({'status': 'ok', 'deleted': deleted})


@app.route('/api/models/export', methods=['GET'])
def export_models():
    ids = (request.args.get('ids') or '').split(',') if request.args.get('ids') else []
    if not ids:
        return jsonify({'error': 'No ids provided'}), 400
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, 'w', zipfile.ZIP_DEFLATED) as z:
        for mid in ids:
            base = os.path.join(MODELS_DIR, mid)
            if not os.path.isdir(base):
                continue
            for root, _, files in os.walk(base):
                for fn in files:
                    fp = os.path.join(root, fn)
                    arc = os.path.relpath(fp, MODELS_DIR)
                    try:
                        z.write(fp, arcname=arc)
                    except Exception:
                        pass
            # increment export count
            try:
                sp = os.path.join(base, 'stats.json')
                stats = {}
                if os.path.exists(sp):
                    try: stats = json.load(open(sp))
                    except Exception: stats = {}
                stats['exports'] = int(stats.get('exports', 0)) + 1
                with open(sp, 'w', encoding='utf-8') as f:
                    json.dump(stats, f)
            except Exception:
                pass
    mem.seek(0)
    return send_file(mem, mimetype='application/zip', as_attachment=True, download_name='models_export.zip')


@app.route('/api/models/save', methods=['POST'])
def save_model():
    """Save a model architecture created in the builder"""
    data = request.json or {}
    name = data.get('name', 'custom_model')
    architecture = data.get('architecture', {})
    config = data.get('config', {})
    code = data.get('code', '')
    metadata = data.get('metadata', {})

    # Create unique model ID
    import uuid
    import datetime
    model_id = f"{_safe_name(name)}_{uuid.uuid4().hex[:8]}"
    model_dir = os.path.join(MODELS_DIR, model_id)
    os.makedirs(model_dir, exist_ok=True)

    # Save architecture JSON
    with open(os.path.join(model_dir, 'architecture.json'), 'w', encoding='utf-8') as f:
        json.dump(architecture, f, indent=2)

    # Save config JSON
    config_path = os.path.join(model_dir, 'config.json')
    full_config = {
        'model_id': model_id,
        'name': name,
        'architecture': 'custom',
        'created_from': metadata.get('created_from', 'builder'),
        'batch_size': metadata.get('batch_size', 8),
        'dtype': metadata.get('dtype', 'fp16'),
        **config
    }
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(full_config, f, indent=2)

    # Save generated PyTorch code
    if code:
        with open(os.path.join(model_dir, 'model.py'), 'w', encoding='utf-8') as f:
            f.write(code)

    # Save metadata
    meta_path = os.path.join(model_dir, 'metadata.json')
    full_metadata = {
        'name': name,
        'model_id': model_id,
        'architecture': 'Custom',
        'framework': 'PyTorch',
        'created_at': datetime.datetime.now().isoformat(),
        'created_from': 'builder',
        'tags': ['custom', 'builder'],
        **metadata
    }
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(full_metadata, f, indent=2)

    return jsonify({'status': 'ok', 'model_id': model_id, 'message': f'Model saved as {model_id}'})


@app.route('/api/models/<model_id>/card.html', methods=['GET'])
def model_card_html(model_id):
    path = os.path.join(MODELS_DIR, model_id)
    if not os.path.isdir(path):
        return jsonify({'error': 'Model not found'}), 404
    md_path = os.path.join(path, 'card.md')
    content = ''
    if os.path.exists(md_path):
        try:
            content = open(md_path, 'r', encoding='utf-8').read()
        except Exception:
            content = ''
    # simple html wrapper; for full markdown render, open in browser or export tool
    html = f"<html><head><meta charset='utf-8'><title>Model Card {model_id}</title></head><body><pre>{content}</pre></body></html>"
    return html


@app.route('/api/models/<model_id>/file', methods=['GET'])
def model_file_serve(model_id):
    base = _model_dir(_safe_name(model_id))
    if not os.path.isdir(base):
        return jsonify({'error': 'Model not found'}), 404
    rel = request.args.get('path') or ''
    p = os.path.normpath(os.path.join(base, rel))
    if not p.startswith(base):
        return jsonify({'error': 'Invalid path'}), 400
    if not os.path.exists(p):
        return jsonify({'error': 'Not found'}), 404
    try:
        return send_file(p)
    except Exception as e:
        return jsonify({'error': 'Failed to send', 'detail': str(e)}), 500


@app.route('/api/models/<model_id>/evals', methods=['GET', 'POST'])
def model_evals(model_id):
    base = _model_dir(_safe_name(model_id))
    if not os.path.isdir(base):
        return jsonify({'error': 'Model not found'}), 404
    ev_path = os.path.join(base, 'evals.jsonl')
    if request.method == 'POST':
        data = request.json or {}
        rec = {
            'ts': datetime.now().isoformat(),
            'name': data.get('name') or 'eval',
            'metrics': data.get('metrics') or {},
            'notes': data.get('notes') or '',
        }
        try:
            with open(ev_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(rec) + '\n')
            return jsonify({'status': 'ok'})
        except Exception as e:
            return jsonify({'error': 'Failed to save', 'detail': str(e)}), 500
    # GET
    items = []
    if os.path.exists(ev_path):
        try:
            with open(ev_path, 'r', encoding='utf-8') as f:
                for ln in f:
                    try: items.append(json.loads(ln))
                    except: pass
        except Exception:
            pass
    return jsonify({'items': items})


def _model_similarity(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    # Jaccard on tags + architecture + size_category
    sa = set([str(a.get('architecture') or '').lower(), str(a.get('size_category') or '').lower()] + [str(t).lower() for t in (a.get('tags') or [])])
    sb = set([str(b.get('architecture') or '').lower(), str(b.get('size_category') or '').lower()] + [str(t).lower() for t in (b.get('tags') or [])])
    inter = len(sa & sb)
    uni = len(sa | sb) or 1
    return inter / uni


@app.route('/api/models/<model_id>/similar', methods=['GET'])
def similar_models(model_id):
    # Build model list and compute similarity with given id.
    base = os.path.join(MODELS_DIR, model_id)
    if not os.path.isdir(base):
        return jsonify({'error': 'Model not found'}), 404
    # Load models
    models = []
    for d in os.listdir(MODELS_DIR):
        m = d
        p = os.path.join(MODELS_DIR, m)
        if not os.path.isdir(p):
            continue
        models.append(m)
    def load_info(mid):
        res = None
        try:
            # Reuse list_models loader by calling underlying helper again
            # Duplicate lightweight reader
            path = os.path.join(MODELS_DIR, mid)
            cfg = {}
            meta = {}
            try:
                cp = os.path.join(path, 'config.json')
                if os.path.exists(cp): cfg = json.load(open(cp))
            except Exception: pass
            try:
                mp = os.path.join(path, 'metadata.json')
                if os.path.exists(mp): meta = json.load(open(mp))
            except Exception: pass
            # tags/arch/size
            total_size = 0
            for r,_,fns in os.walk(path):
                for fn in fns:
                    try: total_size += os.path.getsize(os.path.join(r,fn))
                    except: pass
            arch = cfg.get('architecture', meta.get('architecture'))
            tags = (meta.get('tags') or [])
            size_cat = None
            try:
                p = int(cfg.get('parameters')) if cfg.get('parameters') is not None else None
            except Exception:
                p = None
            if p is not None:
                if p < 100_000_000: size_cat='small'
                elif p < 500_000_000: size_cat='base'
                elif p < 2_000_000_000: size_cat='large'
                else: size_cat='xl'
            elif total_size:
                if total_size < 500*1024*1024: size_cat='small'
                elif total_size < 2*1024*1024*1024: size_cat='base'
                elif total_size < 8*1024*1024*1024: size_cat='large'
                else: size_cat='xl'
            res = {'id': mid, 'architecture': arch, 'tags': tags, 'size_category': size_cat}
        except Exception:
            res = {'id': mid}
        return res
    infos = { mid: load_info(mid) for mid in models }
    src = infos.get(model_id)
    sims = []
    if src:
        for mid, info in infos.items():
            if mid == model_id: continue
            sims.append((mid, _model_similarity(src, info)))
    sims.sort(key=lambda x: x[1], reverse=True)
    return jsonify({'similar': [{'id': mid, 'score': float(f'{score:.3f}')} for mid, score in sims[:10]]})


def _model_dir(model_id: str) -> str:
    return os.path.join(MODELS_DIR, model_id)


def _dir_size_bytes(path: str) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for fn in files:
            fp = os.path.join(root, fn)
            try:
                total += os.path.getsize(fp)
            except Exception:
                pass
    return total


@app.route('/api/models/<model_id>/adapters', methods=['GET'])
def list_model_adapters(model_id):
    base = _model_dir(model_id)
    if not os.path.isdir(base):
        return jsonify({'error': 'Model not found'}), 404
    adir = os.path.join(base, 'adapters')
    items = []
    if os.path.isdir(adir):
        for nm in sorted(os.listdir(adir)):
            p = os.path.join(adir, nm)
            if not os.path.isdir(p):
                continue
            try:
                st = os.stat(p)
                items.append({
                    'name': nm,
                    'path': os.path.relpath(p, base),
                    'size_bytes': _dir_size_bytes(p),
                    'created': datetime.fromtimestamp(st.st_mtime).isoformat()
                })
            except Exception:
                items.append({'name': nm, 'path': os.path.relpath(p, base)})
    return jsonify({'adapters': items})


@app.route('/api/models/<model_id>/adapters/merge', methods=['POST'])
def merge_model_adapter(model_id):
    base = _model_dir(model_id)
    if not os.path.isdir(base):
        return jsonify({'error': 'Model not found'}), 404
    data = request.json or {}
    name = (data.get('name') or '').strip()
    if not name:
        return jsonify({'error': 'Missing adapter name'}), 400
    adir = os.path.join(base, 'adapters', name)
    if not os.path.isdir(adir):
        return jsonify({'error': 'Adapter not found'}), 404
    # Determine task type to pick model class
    task_type = 'classification'
    try:
        cfg_path = os.path.join(base, 'config.json')
        if os.path.exists(cfg_path):
            cfg = json.load(open(cfg_path))
            tt = (cfg.get('task_type') or '').lower()
            if tt in ('generation', 'causal_lm', 'lm'):
                task_type = 'generation'
    except Exception:
        pass
    try:
        from transformers import AutoModelForSequenceClassification, AutoModelForCausalLM
        from peft import PeftModel
    except Exception as e:
        return jsonify({'error': 'Missing dependencies for merge: transformers/peft not available', 'detail': str(e)}), 500
    try:
        if task_type == 'generation':
            base_model = AutoModelForCausalLM.from_pretrained(base)
        else:
            base_model = AutoModelForSequenceClassification.from_pretrained(base)
        peft_model = PeftModel.from_pretrained(base_model, adir)
        merged = peft_model.merge_and_unload()
        merged.save_pretrained(base)
        return jsonify({'status': 'ok', 'merged': True, 'adapter': name})
    except Exception as e:
        return jsonify({'error': 'Merge failed', 'detail': str(e)}), 500

@app.route('/api/jobs', methods=['GET'])
def list_jobs():
    """List all training jobs"""
    return jsonify(list(jobs.values()))

@app.route('/api/jobs/<job_id>', methods=['GET'])
def get_job(job_id):
    """Get specific job details"""
    if job_id not in jobs:
        return jsonify({'error': 'Job not found'}), 404
    
    job = jobs[job_id]
    
    # Get latest logs
    log_file = os.path.join(LOGS_DIR, f'{job_id}.log')
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            logs = f.readlines()
            job['logs'] = ''.join(logs[-100:])  # Last 100 lines
            # parse metrics for progress/eta from METRIC lines
            try:
                total_steps = None
                last_step = None
                start_time = None
                for line in logs[-100:]:
                    if line.startswith('METRIC:'):
                        m = json.loads(line[len('METRIC:'):].strip())
                        if 'total_steps' in m:
                            total_steps = m.get('total_steps')
                        if 'step' in m:
                            last_step = m.get('step')
                        if 'start_time' in m:
                            start_time = m.get('start_time')
                if total_steps and last_step:
                    pct = round((last_step / max(1, total_steps)) * 100, 1)
                    job['progress'] = pct
                    if start_time:
                        try:
                            st = datetime.fromisoformat(start_time)
                            elapsed = (datetime.now() - st).total_seconds()
                            if last_step > 0:
                                eta = elapsed * (total_steps - last_step) / last_step
                                job['eta_seconds'] = int(eta)
                        except Exception:
                            pass
            except Exception:
                pass
    
    # Resource usage per job (best-effort)
    try:
        if 'pid' in job:
            pid = job['pid']
            res = {}
            # Memory RSS from /proc
            try:
                with open(f'/proc/{pid}/status','r') as f:
                    for ln in f:
                        if ln.startswith('VmRSS:'):
                            parts = ln.split()
                            if len(parts) >= 2:
                                rss_kib = int(parts[1])
                                res['rss_mib'] = int(rss_kib/1024)
                            break
            except Exception:
                pass
            # CPU time
            try:
                with open(f'/proc/{pid}/stat','r') as f:
                    parts = f.read().split()
                    utime = int(parts[13]); stime = int(parts[14])
                    clk = os.sysconf(os.sysconf_names['SC_CLK_TCK'])
                    cpu_sec = (utime + stime) / float(clk)
                    res['cpu_time_sec'] = round(cpu_sec, 2)
            except Exception:
                pass
            # GPU memory
            try:
                out = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits']).decode()
                for ln in out.strip().split('\n'):
                    if not ln.strip():
                        continue
                    p_s, mem_s = [x.strip() for x in ln.split(',')[:2]]
                    if str(pid) == p_s:
                        res['gpu_mem_mib'] = int(float(mem_s))
                        break
            except Exception:
                pass
            if res:
                job['resource'] = res
    except Exception:
        pass

    return jsonify(job)


@app.route('/api/jobs/<job_id>/experiment', methods=['PUT'])
def job_set_experiment(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404
    data = request.json or {}
    exp = data.get('experiment') or {}
    if isinstance(exp, dict) and (exp.get('id') or exp.get('name')):
        if exp.get('id'):
            exps = _load_experiments()
            e = exps.get(str(exp['id']))
            if not e:
                return jsonify({'error': 'Experiment not found'}), 404
            job['experiment'] = {'id': e['id'], 'name': e['name']}
        else:
            e = _find_or_create_experiment_by_name(str(exp.get('name')), exp.get('tags') or [], exp.get('notes') or '')
            job['experiment'] = {'id': e['id'], 'name': e['name']}
    if 'experiment_tags' in data:
        job['experiment_tags'] = data.get('experiment_tags') or []
    if 'experiment_notes' in data:
        job['experiment_notes'] = data.get('experiment_notes') or ''
    save_jobs()
    return jsonify({'status': 'ok', 'job': job})


def _tail_file_lines(path: str, max_lines: int = 200):
    try:
        with open(path, 'rb') as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            block = 1024
            data = b''
            while size > 0 and data.count(b'\n') <= max_lines:
                step = min(block, size)
                size -= step
                f.seek(size)
                data = f.read(step) + data
            lines = data.splitlines()[-max_lines:]
            return [l.decode(errors='ignore') for l in lines]
    except Exception:
        return []


def _parse_metrics_from_lines(lines):
    out = []
    for line in lines:
        if line.startswith('METRIC:'):
            try:
                m = json.loads(line[len('METRIC:'):].strip())
                out.append(m)
            except Exception:
                continue
    return out


@app.route('/api/jobs/<job_id>/metrics', methods=['GET'])
def job_metrics(job_id):
    log_file = os.path.join(LOGS_DIR, f'{job_id}.log')
    if not os.path.exists(log_file):
        return jsonify({'metrics': []})
    lines = _tail_file_lines(log_file, max_lines=2000)
    metrics = _parse_metrics_from_lines(lines)
    if (request.args.get('export') or '').lower() == 'csv':
        # Build CSV with union of keys
        keys = []
        for m in metrics:
            for k in m.keys():
                if k not in keys:
                    keys.append(k)
        import csv as _csv
        from flask import Response
        sio = io.StringIO()
        w = _csv.DictWriter(sio, fieldnames=keys)
        w.writeheader()
        for m in metrics:
            w.writerow({k: m.get(k) for k in keys})
        data = sio.getvalue()
        return Response(data, headers={'Content-Type':'text/csv','Content-Disposition':f'attachment; filename={job_id}.metrics.csv'})
    return jsonify({'metrics': metrics})


@app.route('/api/jobs/<job_id>/metrics/stream', methods=['GET'])
def job_metrics_stream(job_id):
    log_file = os.path.join(LOGS_DIR, f'{job_id}.log')
    from flask import Response
    def gen():
        last_count = 0
        while True:
            if os.path.exists(log_file):
                lines = _tail_file_lines(log_file, max_lines=500)
                metrics = _parse_metrics_from_lines(lines)
                # send only the last metric snapshot
                payload = {'metrics': metrics[-50:]}
                yield f"data: {json.dumps(payload)}\n\n"
            time.sleep(2)
    headers = {'Content-Type':'text/event-stream','Cache-Control':'no-cache','Connection':'keep-alive','X-Accel-Buffering':'no'}
    return Response(gen(), headers=headers)


[...]
@app.route('/api/jobs/<job_id>/logs', methods=['GET'])
def job_logs(job_id):
    log_file = os.path.join(LOGS_DIR, f'{job_id}.log')
    if not os.path.exists(log_file):
        return jsonify({'logs': ''})
    search = request.args.get('q')
    level = (request.args.get('level') or '').lower()  # info|warn|error
    max_lines = int(request.args.get('max', 500))
    lines = _tail_file_lines(log_file, max_lines=max_lines)
    def match_level(line: str) -> bool:
        if not level:
            return True
        if level == 'error':
            return 'error' in line.lower()
        if level in ('warn','warning'):
            return 'warn' in line.lower()
        if level == 'info':
            return True
        return True
    if search:
        lines = [ln for ln in lines if search.lower() in ln.lower()]
    lines = [ln for ln in lines if match_level(ln)]
    # export
    export = request.args.get('export')
    if export == 'txt':
        from flask import Response
        data = '\n'.join(lines)
        return Response(data, headers={'Content-Type':'text/plain','Content-Disposition':f'attachment; filename={job_id}.log.txt'})
    if export == 'json':
        return jsonify({'lines': lines})
    return jsonify({'lines': lines})


@app.route('/api/jobs/<job_id>/logs/stream', methods=['GET'])
def job_logs_stream(job_id):
    log_file = os.path.join(LOGS_DIR, f'{job_id}.log')
    from flask import Response
    def gen():
        last = 0
        while True:
            if os.path.exists(log_file):
                lines = _tail_file_lines(log_file, max_lines=200)
                payload = {'lines': lines[-50:]}
                yield f"data: {json.dumps(payload)}\n\n"
            time.sleep(2)
    headers = {'Content-Type':'text/event-stream','Cache-Control':'no-cache','Connection':'keep-alive','X-Accel-Buffering':'no'}
    return Response(gen(), headers=headers)


@app.route('/api/jobs/<job_id>/pause', methods=['POST'])
def job_pause(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({'error':'Job not found'}), 404
    if 'pid' in job:
        try:
            os.kill(job['pid'], signal.SIGSTOP)
            job['status'] = 'paused'
            save_jobs()
            return jsonify(job)
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    return jsonify({'error':'No PID'}), 400


@app.route('/api/jobs/<job_id>/resume', methods=['POST'])
def job_resume(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({'error':'Job not found'}), 404
    if 'pid' in job:
        try:
            os.kill(job['pid'], signal.SIGCONT)
            job['status'] = 'running'
            save_jobs()
            return jsonify(job)
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    return jsonify({'error':'No PID'}), 400


@app.route('/api/jobs/<job_id>/priority', methods=['POST'])
def job_priority(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({'error':'Job not found'}), 404
    pr = (request.json or {}).get('priority')
    job['priority'] = pr
    save_jobs()
    return jsonify(job)


@app.route('/api/jobs/<job_id>/clone', methods=['POST'])
def job_clone(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({'error':'Job not found'}), 404
    payload = {
        'name': (job.get('name','') + ' (clone)')[:100],
        'type': job.get('type'),
        'framework': job.get('framework'),
        'config': job.get('config'),
        'gpu': job.get('gpu'),
        'gpu_prefer': None,
        'priority': job.get('priority'),
        'depends_on': None,
    }
    with app.test_request_context(json=payload):
        return create_job()


@app.route('/api/jobs/<job_id>/checkpoints', methods=['GET'])
def job_checkpoints(job_id):
    model_dir = os.path.join(MODELS_DIR, job_id)
    res = []
    if os.path.isdir(model_dir):
        for fn in os.listdir(model_dir):
            if fn.endswith('.pth'):
                fp = os.path.join(model_dir, fn)
                try:
                    sz = os.path.getsize(fp)
                except Exception:
                    sz = None
                res.append({'file': fn, 'size': sz})
    return jsonify({'checkpoints': res})


@app.route('/api/jobs/<job_id>/checkpoint/rollback', methods=['POST'])
def job_checkpoint_rollback(job_id):
    filename = (request.json or {}).get('file')
    model_dir = os.path.join(MODELS_DIR, job_id)
    if not filename:
        return jsonify({'error':'file required'}), 400
    src = os.path.join(model_dir, filename)
    dst = os.path.join(model_dir, 'model.pth')
    if not os.path.exists(src):
        return jsonify({'error':'checkpoint not found'}), 404
    try:
        import shutil
        shutil.copy2(src, dst)
        return jsonify({'status':'ok'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/jobs/<job_id>/checkpoint/save', methods=['POST'])
def job_checkpoint_save(job_id):
    model_dir = os.path.join(MODELS_DIR, job_id)
    src = os.path.join(model_dir, 'model.pth')
    if not os.path.exists(src):
        return jsonify({'error':'no model.pth'}), 404
    ts = datetime.now().strftime('%Y%m%d-%H%M%S')
    dst = os.path.join(model_dir, f'model-{ts}.pth')
    try:
        import shutil
        shutil.copy2(src, dst)
        return jsonify({'status':'ok', 'file': os.path.basename(dst)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/jobs', methods=['POST'])
def create_job():
    """Create a new training job"""
    data = request.json

    job_id = str(uuid.uuid4())
    
    gpu_sel = data.get('gpu') or None
    # Normalize gpu selection if provided
    gpu_meta = None
    if isinstance(gpu_sel, dict):
        gtype = gpu_sel.get('type')
        if gtype == 'mig' and gpu_sel.get('mig_uuid'):
            gpu_meta = {
                'type': 'mig',
                'mig_uuid': gpu_sel.get('mig_uuid'),
                'gpu_index': gpu_sel.get('gpu_index'),
                'gpu_uuid': gpu_sel.get('gpu_uuid'),
            }
        elif gtype == 'gpu' and (gpu_sel.get('gpu_index') is not None or gpu_sel.get('gpu_uuid')):
            gpu_meta = {
                'type': 'gpu',
                'gpu_index': gpu_sel.get('gpu_index'),
                'gpu_uuid': gpu_sel.get('gpu_uuid'),
            }

    # If no explicit GPU selection provided, attempt an auto-assignment:
    # Prefer a free MIG instance, otherwise a free whole GPU.
    if gpu_meta is None:
        try:
            parts = _detect_partitions()
            prefer = (data.get('gpu_prefer') or '').lower()
            gpu_first = prefer in ('gpu_first', 'gpu')
            chosen = None
            def pick_mig():
                for g in parts.get('gpus', []):
                    for inst in g.get('instances', []):
                        if not inst.get('allocated_by_jobs'):
                            return {
                                'type': 'mig',
                                'gpu_index': g.get('index'),
                                'gpu_uuid': g.get('uuid'),
                                'mig_uuid': inst.get('uuid'),
                            }
                return None
            def pick_gpu():
                for g in parts.get('gpus', []):
                    if not g.get('allocated_by_jobs'):
                        return {
                            'type': 'gpu',
                            'gpu_index': g.get('index'),
                            'gpu_uuid': g.get('uuid'),
                        }
                return None
            if gpu_first:
                chosen = pick_gpu() or pick_mig()
            else:
                chosen = pick_mig() or pick_gpu()
            gpu_meta = chosen
        except Exception:
            gpu_meta = None

    # Resolve dataset selection from registry into a concrete path if provided
    cfg = data.get('config', {}) or {}
    dcfg = cfg.get('data') or {}
    ds_id = cfg.get('dataset_id') or dcfg.get('dataset_id')
    ds_ver = cfg.get('dataset_version') or dcfg.get('dataset_version')
    dataset_resolved_path = None
    if ds_id:
        try:
            ds_path = _dataset_version_dir(str(ds_id), str(ds_ver) if ds_ver else None)
            if os.path.isdir(ds_path):
                dataset_resolved_path = ds_path
                dcfg = dict(dcfg)
                dcfg.setdefault('source', 'local')
                dcfg['local_path'] = ds_path
                cfg['data'] = dcfg
        except Exception:
            pass

    # Attach experiment metadata (optional)
    exp_meta = None
    exp_in = data.get('experiment') or {}
    if isinstance(exp_in, dict) and (exp_in.get('id') or exp_in.get('name')):
        if exp_in.get('id'):
            # ensure exists
            exps = _load_experiments()
            e = exps.get(str(exp_in['id']))
            if e:
                exp_meta = {'id': e['id'], 'name': e['name']}
        else:
            e = _find_or_create_experiment_by_name(str(exp_in.get('name')), exp_in.get('tags') or [], exp_in.get('notes') or '')
            exp_meta = {'id': e['id'], 'name': e['name']}

    job = {
        'id': job_id,
        'name': data.get('name', f'Training Job {job_id[:8]}'),
        'type': data.get('type', 'train'),  # 'train' or 'finetune'
        'framework': data.get('framework', 'pytorch'),
        'status': 'queued',
        'created': datetime.now().isoformat(),
        'config': cfg,
        'progress': 0,
        'metrics': {},
        'gpu': gpu_meta,
        'priority': data.get('priority', 0),
        'depends_on': data.get('depends_on'),
        'dataset_resolved_path': dataset_resolved_path,
        'experiment': exp_meta,
        'experiment_tags': (exp_in.get('tags') if isinstance(exp_in, dict) else None) or [],
        'experiment_notes': (exp_in.get('notes') if isinstance(exp_in, dict) else None) or '',
    }
    
    jobs[job_id] = job
    save_jobs()
    # model usage tracking (if job references a local model id)
    try:
        mid = (cfg or {}).get('model_id')
        if mid and os.path.isdir(os.path.join(MODELS_DIR, _safe_name(str(mid)))):
            sp = os.path.join(MODELS_DIR, _safe_name(str(mid)), 'stats.json')
            stats = {}
            if os.path.exists(sp):
                try: stats = json.load(open(sp))
                except Exception: stats = {}
            stats['used_in_jobs'] = int(stats.get('used_in_jobs', 0)) + 1
            with open(sp, 'w', encoding='utf-8') as f:
                json.dump(stats, f)
    except Exception:
        pass
    
    # Start training in background thread unless blocked by dependencies
    def deps_done(dep_ids):
        if not dep_ids:
            return True
        for did in (dep_ids if isinstance(dep_ids, list) else [dep_ids]):
            dj = jobs.get(did)
            if not dj or dj.get('status') not in ('completed', 'failed', 'cancelled'):
                return False
        return True
    if deps_done(job.get('depends_on')):
        thread = threading.Thread(target=run_training_job, args=(job_id,))
        thread.daemon = True
        thread.start()
    else:
        job['status'] = 'blocked'
    
    return jsonify(job), 201


@app.route('/api/jobs/allocate', methods=['POST'])
def jobs_allocate():
    """Suggest a GPU/MIG allocation based on model config and available memory with 20% headroom.

    Payload mirrors create_job. Returns: { gpu: {...}, rationale } or { error }.
    """
    data = request.json or {}
    cfg = (data.get('config') or {})
    bs = int((cfg.get('batch_size') or 8))
    seq = int(((cfg.get('context_extension') or {}).get('rope') or {}).get('target_length') or 2048)
    training = True
    # Estimate params when framework is HF and model_name provided; else unknown
    params = None
    if data.get('framework') == 'huggingface' and cfg.get('model_name'):
        # cannot fetch HF remotely; fallback
        params = None
    # Use local model_id if specified
    model_id = cfg.get('model_id') or data.get('model_id')
    if model_id:
        est = _estimate_model_size_params(_model_dir(_safe_name(model_id)))
        params = est.get('parameters') or params
    mem = _estimate_memory_requirements(params, bs, seq, training)
    need = mem['total_bytes'] * 1.2  # 20% headroom
    # Detect partitions and GPU memory
    parts = _detect_partitions()
    # Best-effort memory per MIG profile
    def mig_mem_gb(profile: str) -> int:
        return _MIG_PROFILE_MEM_GB.get(str(profile).lower(), 0)
    # Query GPUs total mem via nvidia-smi
    gmem = {}
    try:
        out = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.total', '--format=csv,noheader,nounits']).decode()
        for line in out.strip().split('\n'):
            idx_s, mib_s = [x.strip() for x in line.split(',')]
            gmem[int(idx_s)] = int(mib_s) * 1024 * 1024
    except Exception:
        pass
    # Prefer MIG instance that fits
    for g in parts.get('gpus', []):
        for inst in g.get('instances', []):
            prof = inst.get('profile')
            if mig_mem_gb(prof) * (1024**3) >= need and not inst.get('allocated_by_jobs'):
                return jsonify({'gpu': {'type':'mig','gpu_index':g.get('index'),'gpu_uuid':g.get('uuid'),'mig_uuid':inst.get('uuid')}, 'rationale': f'MIG {prof} meets memory need'}), 200
    # Else prefer full GPU that fits
    for g in parts.get('gpus', []):
        if not g.get('allocated_by_jobs'):
            tot = gmem.get(g.get('index'))
            if tot and tot >= need:
                return jsonify({'gpu': {'type':'gpu','gpu_index':g.get('index'),'gpu_uuid':g.get('uuid')}, 'rationale': 'Full GPU meets memory need'}), 200
    return jsonify({'error': 'No suitable GPU/MIG found', 'need_bytes': int(need)}), 200

@app.route('/api/jobs/<job_id>/cancel', methods=['POST'])
def cancel_job(job_id):
    """Cancel a running job"""
    if job_id not in jobs:
        return jsonify({'error': 'Job not found'}), 404
    
    job = jobs[job_id]
    if job['status'] in ['completed', 'failed', 'cancelled']:
        return jsonify({'error': 'Job already finished'}), 400
    
    # Try to kill the process
    if 'pid' in job:
        try:
            os.kill(job['pid'], signal.SIGTERM)
        except:
            pass
    
    job['status'] = 'cancelled'
    save_jobs()
    
    return jsonify(job)

@app.route('/api/jobs/<job_id>', methods=['DELETE'])
def delete_job(job_id):
    """Delete a job (only if completed, failed, or cancelled)"""
    if job_id not in jobs:
        return jsonify({'error': 'Job not found'}), 404

    job = jobs[job_id]
    if job['status'] in ['running', 'queued', 'paused']:
        return jsonify({'error': 'Cannot delete running or queued jobs. Cancel first.'}), 400

    # Remove from jobs dict
    del jobs[job_id]
    save_jobs()

    # Optionally clean up job files (logs, checkpoints) - keeping them for now
    # model_dir = os.path.join(MODELS_DIR, job_id)
    # log_file = os.path.join(LOGS_DIR, f'{job_id}.log')

    return jsonify({'message': 'Job deleted successfully'}), 200

@app.route('/api/jobs/<job_id>/restart', methods=['POST'])
def restart_job(job_id):
    """Restart a failed or cancelled job"""
    if job_id not in jobs:
        return jsonify({'error': 'Job not found'}), 404

    job = jobs[job_id]
    if job['status'] not in ['failed', 'cancelled']:
        return jsonify({'error': 'Can only restart failed or cancelled jobs'}), 400

    # Reset job status and clear error info
    job['status'] = 'queued'
    job['started'] = None
    job['finished'] = None
    job['error'] = None
    job['progress'] = 0
    job['eta_seconds'] = None
    save_jobs()

    # Re-enqueue the job
    from threading import Thread
    Thread(target=run_training_job, args=(job_id,), daemon=True).start()

    return jsonify(job)

def run_training_job(job_id):
    """Execute the training job"""
    job = jobs[job_id]
    config = job['config']
    
    try:
        job['status'] = 'running'
        job['started'] = datetime.now().isoformat()
        save_jobs()
        # Prepare model output dir and snapshot config/env
        model_dir = os.path.join(MODELS_DIR, job_id)
        os.makedirs(model_dir, exist_ok=True)
        try:
            with open(os.path.join(model_dir, 'config.json'), 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2)
        except Exception:
            pass
        # Env snapshot
        snap = {
            'created': datetime.now().isoformat(),
            'python_version': None,
            'packages': [],
            'cuda': {},
            'system': _system_info_snapshot(),
            'dataset_path': job.get('dataset_resolved_path'),
        }
        try:
            import sys
            snap['python_version'] = sys.version
        except Exception:
            pass
        try:
            out = subprocess.check_output(['pip','freeze'], stderr=subprocess.STDOUT)
            snap['packages'] = out.decode(errors='ignore').splitlines()
        except Exception:
            pass
        try:
            out = subprocess.check_output(['nvidia-smi','--query-gpu=driver_version,cuda_version','--format=csv,noheader'], stderr=subprocess.STDOUT).decode()
            parts = out.strip().split('\n')[0].split(',')
            if len(parts) >= 2:
                snap['cuda'] = {'driver_version': parts[0].strip(), 'cuda_version': parts[1].strip()}
        except Exception:
            pass
        try:
            with open(os.path.join(model_dir, 'env.json'), 'w', encoding='utf-8') as f:
                json.dump(snap, f, indent=2)
        except Exception:
            pass

        # Prepare training command
        script_path = TRAINING_SCRIPTS_DIR
        
        if job['type'] == 'train':
            # Route to HPO for PyTorch if enabled
            cfg = job.get('config') or {}
            if job['framework'] == 'pytorch' and isinstance(cfg.get('hpo'), dict) and cfg['hpo'].get('enabled'):
                script = os.path.join(script_path, 'hpo_optuna_torch.py')
            else:
                script = os.path.join(script_path, f"train_{job['framework']}.py")
        else:
            # Prefer context extension script when enabled for HF
            cfg = job.get('config') or {}
            if job['framework'] == 'huggingface' and isinstance(cfg.get('hpo'), dict) and cfg['hpo'].get('enabled'):
                script = os.path.join(script_path, 'hpo_optuna_hf.py')
            elif job['framework'] == 'huggingface' and isinstance(cfg.get('context_extension'), dict) and cfg['context_extension'].get('enabled'):
                script = os.path.join(script_path, 'context_extension_hf.py')
            else:
                script = os.path.join(script_path, f"finetune_{job['framework']}.py")
        
        # Build command
        cmd = [
            'python', script,
            '--job-id', job_id,
            '--config', json.dumps(config)
        ]
        
        log_file = os.path.join(LOGS_DIR, f'{job_id}.log')
        
        with open(log_file, 'w') as f:
            # Set GPU visibility if specified
            env = os.environ.copy()
            gpu_meta = job.get('gpu') or {}
            if gpu_meta.get('type') == 'mig' and gpu_meta.get('mig_uuid'):
                env['CUDA_VISIBLE_DEVICES'] = gpu_meta['mig_uuid']
            elif gpu_meta.get('type') == 'gpu':
                if gpu_meta.get('gpu_uuid'):
                    env['CUDA_VISIBLE_DEVICES'] = gpu_meta['gpu_uuid']
                elif gpu_meta.get('gpu_index') is not None:
                    env['CUDA_VISIBLE_DEVICES'] = str(gpu_meta['gpu_index'])

            # Apply tracking env overrides from job config (e.g., MLFLOW_TRACKING_URI, WANDB_API_KEY)
            try:
                tr_env = ((config.get('tracking') or {}).get('env') or {})
                if isinstance(tr_env, dict):
                    for k, v in tr_env.items():
                        if isinstance(k, str) and v is not None:
                            env[str(k)] = str(v)
            except Exception:
                pass

            process = subprocess.Popen(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                cwd=BASE_DIR,
                env=env
            )
            
            job['pid'] = process.pid
            save_jobs()
            
            # Wait for completion
            return_code = process.wait()
            
            if return_code == 0:
                job['status'] = 'completed'
            else:
                job['status'] = 'failed'
                job['error'] = f'Process exited with code {return_code}'
        
        job['completed'] = datetime.now().isoformat()
        
    except Exception as e:
        job['status'] = 'failed'
        job['error'] = str(e)
        job['completed'] = datetime.now().isoformat()
    
    finally:
        save_jobs()
        # Try to start blocked jobs whose dependencies are satisfied
        try:
            for jid, j in list(jobs.items()):
                if j.get('status') == 'blocked':
                    deps = j.get('depends_on')
                    def deps_done(dep_ids):
                        if not dep_ids:
                            return True
                        for did in (dep_ids if isinstance(dep_ids, list) else [dep_ids]):
                            dj = jobs.get(did)
                            if not dj or dj.get('status') not in ('completed','failed','cancelled'):
                                return False
                        return True
                    if deps_done(deps):
                        j['status'] = 'queued'
                        save_jobs()
                        t = threading.Thread(target=run_training_job, args=(jid,))
                        t.daemon = True
                        t.start()
        except Exception:
            pass

def _parse_gpu_info() -> Dict[str, Any]:
    """Read GPU info via nvidia-smi and compute memory percentages.

    Returns a dict with a 'gpus' list. If nvidia-smi is not available, returns an empty list.
    """
    gpus = []
    try:
        # Ask for raw numbers (no units) to avoid parsing issues
        out = subprocess.check_output([
            'nvidia-smi',
            '--query-gpu=name,index,memory.total,memory.free,utilization.gpu,temperature.gpu,power.draw,power.limit',
            '--format=csv,noheader,nounits'
        ]).decode()
        for line in out.strip().split('\n'):
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 6:
                # Power metrics may be missing on some GPUs; guard by length
                name = parts[0]
                idx_s = parts[1]
                total_s = parts[2]
                free_s = parts[3]
                util_s = parts[4]
                temp_s = parts[5]
                power_draw_s = parts[6] if len(parts) > 6 else None
                power_limit_s = parts[7] if len(parts) > 7 else None
                try:
                    idx = int(idx_s)
                except Exception:
                    idx = 0
                def to_int(s: str) -> int:
                    try:
                        return int(float(s))
                    except Exception:
                        return 0
                def to_float_or_none(s: str):
                    try:
                        return float(s)
                    except Exception:
                        return None
                total = to_int(total_s)      # MiB
                free = to_int(free_s)        # MiB
                used = max(0, total - free)  # MiB
                util = to_int(util_s)        # %
                temp = to_int(temp_s)        # Celsius
                p_draw = to_float_or_none(power_draw_s) if power_draw_s and power_draw_s not in ('N/A', '') else None
                p_limit = to_float_or_none(power_limit_s) if power_limit_s and power_limit_s not in ('N/A', '') else None
                used_pct = round((used / total) * 100, 1) if total > 0 else 0.0
                free_pct = round((free / total) * 100, 1) if total > 0 else 0.0
                gpus.append({
                    'name': name,
                    'index': idx,
                    'memory_total_mib': total,
                    'memory_used_mib': used,
                    'memory_free_mib': free,
                    'memory_used_pct': used_pct,
                    'memory_free_pct': free_pct,
                    'utilization_gpu_pct': util,
                    'temperature_gpu_c': temp,
                    'power_draw_w': p_draw,
                    'power_limit_w': p_limit,
                })
    except Exception:
        gpus = []
    total_mem = sum(g.get('memory_total_mib', 0) for g in gpus)
    used_mem = sum(g.get('memory_used_mib', 0) for g in gpus)
    summary = {
        'gpus': gpus,
        'memory_total_mib': total_mem,
        'memory_used_mib': used_mem,
        'memory_used_pct': round((used_mem / total_mem) * 100, 1) if total_mem > 0 else 0.0,
    }
    return summary


def _system_info_snapshot() -> Dict[str, Any]:
    base = _parse_gpu_info()
    base.update({
        'timestamp': datetime.now().isoformat(),
        'jobs_running': len([j for j in jobs.values() if j['status'] == 'running']),
        'jobs_queued': len([j for j in jobs.values() if j['status'] == 'queued']),
        'models_available': len(os.listdir(MODELS_DIR)) if os.path.exists(MODELS_DIR) else 0,
    })
    # CPU count and load averages
    try:
        base['cpu'] = {
            'count': os.cpu_count(),
            'load_avg': list(os.getloadavg()) if hasattr(os, 'getloadavg') else None,
        }
    except Exception:
        base['cpu'] = {'count': os.cpu_count()}

    # Memory via /proc/meminfo
    try:
        meminfo = {}
        with open('/proc/meminfo', 'r') as f:
            for line in f:
                k, v = line.split(':', 1)
                meminfo[k.strip()] = v.strip()
        def to_kib(s: str) -> int:
            return int(s.split()[0]) if s else 0
        total_kib = to_kib(meminfo.get('MemTotal', '0'))
        avail_kib = to_kib(meminfo.get('MemAvailable', '0'))
        free_kib = to_kib(meminfo.get('MemFree', '0'))
        used_kib = max(0, total_kib - avail_kib)
        total_mib = total_kib // 1024
        base['memory'] = {
            'total_mib': int(total_mib),
            'used_mib': int(used_kib // 1024),
            'available_mib': int(avail_kib // 1024),
            'free_mib': int(free_kib // 1024),
            'used_pct': round((used_kib / total_kib) * 100, 1) if total_kib else 0.0,
        }
    except Exception:
        base['memory'] = {}

    # Swap via /proc/meminfo
    try:
        if 'meminfo' not in locals():
            meminfo = {}
            with open('/proc/meminfo', 'r') as f:
                for line in f:
                    k, v = line.split(':', 1)
                    meminfo[k.strip()] = v.strip()
        def to_kib(s: str) -> int:
            return int(s.split()[0]) if s else 0
        swap_total_kib = to_kib(meminfo.get('SwapTotal', '0'))
        swap_free_kib = to_kib(meminfo.get('SwapFree', '0'))
        swap_used_kib = max(0, swap_total_kib - swap_free_kib)
        base['swap'] = {
            'total_mib': int(swap_total_kib // 1024),
            'used_mib': int(swap_used_kib // 1024),
            'free_mib': int(swap_free_kib // 1024),
            'used_pct': round((swap_used_kib / swap_total_kib) * 100, 1) if swap_total_kib else 0.0,
        }
    except Exception:
        base['swap'] = {}

    # Disks via statvfs
    try:
        def disk_usage(path: str) -> Dict[str, Any]:
            st = os.statvfs(path)
            total = st.f_frsize * st.f_blocks
            free = st.f_frsize * st.f_bavail
            used = total - free
            to_gib = lambda b: round(b / (1024 ** 3), 1)
            used_pct = round((used / total) * 100, 1) if total else 0.0
            return {
                'path': path,
                'total_gib': to_gib(total),
                'used_gib': to_gib(used),
                'free_gib': to_gib(free),
                'used_pct': used_pct,
            }
        disks = [disk_usage('/')]
        if os.path.exists('/app'):
            disks.append(disk_usage('/app'))
        base['disks'] = disks
    except Exception:
        base['disks'] = []

    # Net totals via /proc/net/dev (+rates)
    try:
        rx = tx = 0
        with open('/proc/net/dev', 'r') as f:
            lines = f.read().strip().splitlines()[2:]
        for ln in lines:
            if ':' not in ln:
                continue
            iface, rest = ln.split(':', 1)
            iface = iface.strip()
            cols = rest.split()
            if len(cols) >= 16:
                rx_bytes = int(cols[0])
                tx_bytes = int(cols[8])
                # include all interfaces, including loopback for completeness
                rx += rx_bytes
                tx += tx_bytes
        now = time.time()
        rate_rx = rate_tx = 0.0
        if _last_net_totals['rx'] is not None and _last_net_totals['ts'] is not None:
            dt = max(1e-6, now - _last_net_totals['ts'])
            rate_rx = (rx - _last_net_totals['rx']) / dt
            rate_tx = (tx - _last_net_totals['tx']) / dt
        _last_net_totals.update({'rx': rx, 'tx': tx, 'ts': now})
        base['net'] = {'rx_bytes': rx, 'tx_bytes': tx, 'rx_rate_bps': int(rate_rx), 'tx_rate_bps': int(rate_tx)}
    except Exception:
        base['net'] = {}
    # Active GPU allocations derived from jobs metadata
    allocations = []
    try:
        for j in jobs.values():
            if j.get('status') == 'running' and j.get('gpu'):
                g = j['gpu']
                allocations.append({
                    'job_id': j['id'],
                    'job_name': j.get('name'),
                    'type': g.get('type'),
                    'gpu_index': g.get('gpu_index'),
                    'gpu_uuid': g.get('gpu_uuid'),
                    'mig_uuid': g.get('mig_uuid'),
                })
    except Exception:
        pass
    base['gpu_allocations'] = allocations
    return base


# ========== Dataset Registry ==========

def _safe_name(name: str) -> str:
    return re.sub(r'[^a-zA-Z0-9._\-]+', '_', name or '').strip('_')


def _dataset_dir(name: str) -> str:
    return os.path.join(DATASETS_DIR, _safe_name(name))


def _dataset_version_dir(name: str, version: str | None = None) -> str:
    base = _dataset_dir(name)
    if version:
        return os.path.join(base, _safe_name(version))
    # latest by mtime
    if not os.path.isdir(base):
        return base
    subs = [d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))]
    if not subs:
        return base
    subs.sort(key=lambda d: os.path.getmtime(os.path.join(base, d)), reverse=True)
    return os.path.join(base, subs[0])


def _uploads_dir() -> str:
    p = os.path.join(DATASETS_DIR, '_uploads')
    os.makedirs(p, exist_ok=True)
    return p


def _compute_dataset_stats(root: str) -> Dict[str, Any]:
    total_files = 0
    total_bytes = 0
    by_ext: Dict[str, int] = {}
    sample_csv_header = None
    image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
    image_count = 0
    for r, _, files in os.walk(root):
        for fn in files:
            total_files += 1
            fp = os.path.join(r, fn)
            try:
                total_bytes += os.path.getsize(fp)
            except Exception:
                pass
            ext = os.path.splitext(fn)[1].lower()
            by_ext[ext] = by_ext.get(ext, 0) + 1
            if sample_csv_header is None and ext == '.csv':
                try:
                    with open(fp, 'r', encoding='utf-8', errors='ignore') as f:
                        header = f.readline().strip()
                        sample_csv_header = header
                except Exception:
                    pass
            if ext in image_exts:
                image_count += 1
    return {
        'total_files': total_files,
        'total_bytes': total_bytes,
        'by_extension': by_ext,
        'image_count': image_count,
        'csv_header_sample': sample_csv_header,
        'healthy': total_files > 0,
    }


@app.route('/api/datasets', methods=['GET'])
def list_datasets():
    q = (request.args.get('q') or '').strip().lower()
    items = []
    if os.path.isdir(DATASETS_DIR):
        for nm in sorted(os.listdir(DATASETS_DIR)):
            p = os.path.join(DATASETS_DIR, nm)
            if not os.path.isdir(p):
                continue
            if q and q not in nm.lower():
                continue
            # versions
            versions = [d for d in os.listdir(p) if os.path.isdir(os.path.join(p, d))]
            latest = None
            if versions:
                versions.sort(key=lambda d: os.path.getmtime(os.path.join(p, d)), reverse=True)
                latest = versions[0]
            meta = {}
            meta_path = os.path.join(p, 'metadata.json')
            if os.path.exists(meta_path):
                try:
                    meta = json.load(open(meta_path))
                except Exception:
                    meta = {}
            stats = _compute_dataset_stats(os.path.join(p, latest)) if latest else {
                'total_files': 0,
                'total_bytes': 0,
                'by_extension': {},
                'image_count': 0,
                'healthy': False,
            }
            # DVC detection
            dvc_detected = False
            try:
                dvc_detected = os.path.exists(os.path.join(p, 'dvc.yaml')) or os.path.exists(os.path.join(p, '.dvc')) or any(fn.endswith('.dvc') for fn in os.listdir(p))
            except Exception:
                dvc_detected = False
            items.append({
                'name': nm,
                'latest': latest,
                'versions': len(versions),
                'meta': { **meta, **({'dvc_detected': True} if dvc_detected else {}) },
                'stats': stats,
            })
    return jsonify(items)


@app.route('/api/datasets/<name>', methods=['GET', 'DELETE'])
def dataset_detail(name):
    name = _safe_name(name)
    base = _dataset_dir(name)
    if not os.path.isdir(base):
        return jsonify({'error': 'Dataset not found'}), 404

    if request.method == 'DELETE':
        # Delete the entire dataset directory
        import shutil
        try:
            shutil.rmtree(base)
            return jsonify({'status': 'ok', 'message': 'Dataset deleted successfully'})
        except Exception as e:
            return jsonify({'error': f'Failed to delete dataset: {str(e)}'}), 500

    # GET
    versions = [d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))]
    versions.sort(key=lambda d: os.path.getmtime(os.path.join(base, d)), reverse=True)
    out_vers = []
    for v in versions:
        vp = os.path.join(base, v)
        try:
            st = os.path.getmtime(vp)
        except Exception:
            st = None
        out_vers.append({'version': v, 'created': datetime.fromtimestamp(st).isoformat() if st else None, 'stats': _compute_dataset_stats(vp)})
    meta = {}
    meta_path = os.path.join(base, 'metadata.json')
    if os.path.exists(meta_path):
        try:
            meta = json.load(open(meta_path))
        except Exception:
            meta = {}
    # DVC detection for detail
    try:
        dvc_detected = os.path.exists(os.path.join(base, 'dvc.yaml')) or os.path.exists(os.path.join(base, '.dvc')) or any(fn.endswith('.dvc') for fn in os.listdir(base))
    except Exception:
        dvc_detected = False
    if dvc_detected:
        meta['dvc_detected'] = True
    return jsonify({'name': name, 'meta': meta, 'versions': out_vers})


@app.route('/api/datasets/<name>/metadata', methods=['PUT'])
def dataset_update_metadata(name):
    name = _safe_name(name)
    base = _dataset_dir(name)
    if not os.path.isdir(base):
        return jsonify({'error': 'Dataset not found'}), 404
    data = request.json or {}
    meta_path = os.path.join(base, 'metadata.json')
    cur = {}
    if os.path.exists(meta_path):
        try:
            cur = json.load(open(meta_path))
        except Exception:
            cur = {}
    for k in ('description','tags','categories','type'):
        if k in data:
            cur[k] = data.get(k)
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(cur, f, indent=2)
    return jsonify({'status': 'ok', 'metadata': cur})


@app.route('/api/datasets/upload', methods=['POST'])
def dataset_upload():
    """Upload a dataset archive or single file; creates a new version directory.

    form fields: name (required), version (optional), file (required)
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    file = request.files['file']
    name = _safe_name(request.form.get('name') or '')
    if not name:
        return jsonify({'error': 'Missing dataset name'}), 400
    version_in = request.form.get('version')
    version = _safe_name(version_in) if version_in else datetime.now().strftime('%Y%m%d-%H%M%S')
    dest_dir = _dataset_version_dir(name, version)
    os.makedirs(dest_dir, exist_ok=True)
    # Save to a temp path
    tmp_path = os.path.join(dest_dir, file.filename)
    file.save(tmp_path)
    # If it's a zip, extract and remove
    extracted = False
    try:
        if zipfile.is_zipfile(tmp_path):
            with zipfile.ZipFile(tmp_path, 'r') as z:
                z.extractall(dest_dir)
            os.remove(tmp_path)
            extracted = True
    except Exception:
        pass
    stats = _compute_dataset_stats(dest_dir)
    return jsonify({'status': 'ok', 'name': name, 'version': version, 'extracted': extracted, 'stats': stats})


# Streaming ingestion for large files
@app.route('/api/datasets/ingest/stream_start', methods=['POST'])
def datasets_ingest_stream_start():
    """Start a streaming upload session.

    JSON: { name, version?, filename, total_size?, type?("csv"|"jsonl") }
    Returns: { session, path }
    """
    data = request.json or {}
    name = _safe_name(data.get('name') or '')
    if not name:
        return jsonify({'error': 'Missing name'}), 400
    version = _safe_name(data.get('version') or datetime.now().strftime('%Y%m%d-%H%M%S'))
    filename = data.get('filename') or 'upload.bin'
    sess = str(uuid.uuid4())
    sess_dir = os.path.join(_uploads_dir(), sess)
    os.makedirs(sess_dir, exist_ok=True)
    # Store session metadata
    meta = {
        'name': name,
        'version': version,
        'filename': filename,
        'type': (data.get('type') or '').lower(),
        'created': datetime.now().isoformat(),
    }
    with open(os.path.join(sess_dir, 'meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f)
    # Ensure chunk file exists
    open(os.path.join(sess_dir, 'chunks.part'), 'ab').close()
    return jsonify({'session': sess, 'path': f"{sess_dir}"})


@app.route('/api/datasets/ingest/stream_chunk', methods=['POST'])
def datasets_ingest_stream_chunk():
    """Append a chunk to an existing session.

    form-data: session, index, file (chunk)
    """
    sess = request.form.get('session') or ''
    if not sess or 'file' not in request.files:
        return jsonify({'error': 'Missing session or file'}), 400
    sess_dir = os.path.join(_uploads_dir(), _safe_name(sess))
    if not os.path.isdir(sess_dir):
        return jsonify({'error': 'Session not found'}), 404
    chunk = request.files['file']
    try:
        with open(os.path.join(sess_dir, 'chunks.part'), 'ab') as f:
            f.write(chunk.read())
        return jsonify({'status': 'ok'})
    except Exception as e:
        return jsonify({'error': 'Write failed', 'detail': str(e)}), 500


@app.route('/api/datasets/ingest/stream_finalize', methods=['POST'])
def datasets_ingest_stream_finalize():
    """Finalize a session: map uploaded CSV/JSONL to dataset JSONL on server side without loading into memory.

    JSON: { session, mapping:{ new: src }, header?: [csv headers], type?: 'csv'|'jsonl' }
    Returns: { status, name, version, out }
    """
    data = request.json or {}
    sess = _safe_name(data.get('session') or '')
    if not sess:
        return jsonify({'error': 'Missing session'}), 400
    sess_dir = os.path.join(_uploads_dir(), sess)
    meta_path = os.path.join(sess_dir, 'meta.json')
    if not os.path.exists(meta_path):
        return jsonify({'error': 'Session not found'}), 404
    meta = json.load(open(meta_path))
    name = meta['name']
    version = meta['version']
    typ = (data.get('type') or meta.get('type') or '').lower()
    mapping = data.get('mapping') or {}
    header = data.get('header') or []
    src_file = os.path.join(sess_dir, 'chunks.part')
    dst_dir = _dataset_version_dir(name, version)
    os.makedirs(dst_dir, exist_ok=True)
    outp = os.path.join(dst_dir, 'data.jsonl')
    try:
        with open(outp, 'w', encoding='utf-8') as out_f:
            if typ == 'csv':
                # Stream CSV rows
                with open(src_file, 'r', encoding='utf-8', errors='ignore', newline='') as f:
                    reader = csv.reader(f)
                    hdr = header or (next(reader, []) or [])
                    for row in reader:
                        row_map = { hdr[i]: row[i] for i in range(min(len(hdr), len(row))) }
                        obj = { k: row_map.get(v) for k, v in mapping.items() } if mapping else row_map
                        out_f.write(json.dumps(obj, ensure_ascii=False) + '\n')
            elif typ == 'jsonl':
                with open(src_file, 'r', encoding='utf-8', errors='ignore') as f:
                    for ln in f:
                        ln = ln.strip()
                        if not ln:
                            continue
                        try:
                            r = json.loads(ln)
                        except Exception:
                            continue
                        obj = { k: r.get(v) for k, v in mapping.items() } if mapping else r
                        out_f.write(json.dumps(obj, ensure_ascii=False) + '\n')
            else:
                return jsonify({'error': 'Unsupported type'}), 400
        # Clean up session dir
        try:
            shutil.rmtree(sess_dir, ignore_errors=True)
        except Exception:
            pass
        stats = _compute_dataset_stats(dst_dir)
        return jsonify({'status': 'ok', 'name': name, 'version': version, 'path': outp, 'stats': stats})
    except Exception as e:
        return jsonify({'error': 'Finalize failed', 'detail': str(e)}), 500


@app.route('/api/datasets/<name>/samples', methods=['GET'])
def dataset_samples(name):
    name = _safe_name(name)
    version = _safe_name(request.args.get('version') or '')
    kind = (request.args.get('kind') or 'any').lower()  # any|image|text
    offset = int(request.args.get('offset') or 0)
    limit = int(request.args.get('limit') or 24)
    root = _dataset_version_dir(name, version if version else None)
    if not os.path.isdir(root):
        return jsonify({'error': 'Not found'}), 404
    image_exts = {'.jpg','.jpeg','.png','.bmp','.gif'}
    text_exts = {'.txt','.json','.jsonl','.csv'}
    files = []
    for r, _, fns in os.walk(root):
        for fn in fns:
            ext = os.path.splitext(fn)[1].lower()
            p = os.path.join(r, fn)
            rel = os.path.relpath(p, root).replace('\\','/')
            fkind = 'image' if ext in image_exts else ('text' if ext in text_exts else 'other')
            if kind != 'any' and fkind != kind:
                continue
            try:
                sz = os.path.getsize(p)
            except Exception:
                sz = None
            files.append({'path': rel, 'kind': fkind, 'size': sz})
    files.sort(key=lambda x: x['path'])
    return jsonify({'total': len(files), 'items': files[offset:offset+limit]})


@app.route('/api/datasets/<name>/file', methods=['GET'])
def dataset_file(name):
    name = _safe_name(name)
    version = _safe_name(request.args.get('version') or '')
    rel = request.args.get('path') or ''
    root = _dataset_version_dir(name, version if version else None)
    p = os.path.normpath(os.path.join(root, rel))
    if not p.startswith(root):
        return jsonify({'error': 'Invalid path'}), 400
    if not os.path.isfile(p):
        return jsonify({'error': 'Not found'}), 404
    return send_file(p)


@app.route('/api/datasets/<name>/version/create', methods=['POST'])
def dataset_version_create(name):
    name = _safe_name(name)
    data = request.json or {}
    base_ver = _safe_name(data.get('base') or '')
    new_ver = _safe_name(data.get('new') or '')
    if not new_ver:
        return jsonify({'error': 'Missing new version name'}), 400
    src = _dataset_version_dir(name, base_ver if base_ver else None)
    dst = _dataset_version_dir(name, new_ver)
    if not os.path.isdir(src):
        return jsonify({'error': 'Base version not found'}), 404
    if os.path.exists(dst):
        return jsonify({'error': 'New version already exists'}), 400
    try:
        shutil.copytree(src, dst)
        return jsonify({'status': 'ok', 'version': new_ver})
    except Exception as e:
        return jsonify({'error': 'Copy failed', 'detail': str(e)}), 500


@app.route('/api/datasets/<name>/version/diff', methods=['POST'])
def dataset_version_diff(name):
    name = _safe_name(name)
    data = request.json or {}
    a = _safe_name(data.get('a') or '')
    b = _safe_name(data.get('b') or '')
    if not a or not b:
        return jsonify({'error': 'Missing versions'}), 400
    ra = _dataset_version_dir(name, a)
    rb = _dataset_version_dir(name, b)
    if not os.path.isdir(ra) or not os.path.isdir(rb):
        return jsonify({'error': 'Version not found'}), 404
    def map_files(root):
        m = {}
        for r, _, fns in os.walk(root):
            for fn in fns:
                p = os.path.join(r, fn)
                rel = os.path.relpath(p, root).replace('\\','/')
                try:
                    m[rel] = os.path.getsize(p)
                except Exception:
                    m[rel] = None
        return m
    ma = map_files(ra); mb = map_files(rb)
    added = [k for k in mb.keys() if k not in ma]
    removed = [k for k in ma.keys() if k not in mb]
    changed = [k for k in mb.keys() if k in ma and ma[k] != mb[k]]
    return jsonify({'a': a, 'b': b, 'added': added, 'removed': removed, 'changed': changed})


@app.route('/api/datasets/<name>/version/rollback', methods=['POST'])
def dataset_version_rollback(name):
    name = _safe_name(name)
    data = request.json or {}
    target = _safe_name(data.get('target') or '')
    if not target:
        return jsonify({'error': 'Missing target'}), 400
    new_ver = f"rollback-{target}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    # copy target to new_ver
    src = _dataset_version_dir(name, target)
    if not os.path.isdir(src):
        return jsonify({'error': 'Target version not found'}), 404
    dst = _dataset_version_dir(name, new_ver)
    if os.path.exists(dst):
        return jsonify({'error': 'Rollback version already exists'}), 400
    try:
        shutil.copytree(src, dst)
        return jsonify({'status': 'ok', 'version': new_ver, 'from': target})
    except Exception as e:
        return jsonify({'error': 'Rollback failed', 'detail': str(e)}), 500


@app.route('/api/datasets/template', methods=['POST'])
def dataset_create_template():
    data = request.json or {}
    name = _safe_name(data.get('name') or '')
    typ = (data.get('template') or 'image_classification').lower()
    if not name:
        return jsonify({'error': 'Missing name'}), 400
    version = data.get('version') or 'v1'
    base = _dataset_version_dir(name, version)
    try:
        os.makedirs(base, exist_ok=False)
    except FileExistsError:
        return jsonify({'error': 'Dataset/version already exists'}), 400
    # Create skeleton
    if typ == 'image_classification':
        # base/class_{0,1}/img_*.png (placeholders)
        for cls in ('class_a','class_b'):
            d = os.path.join(base, cls); os.makedirs(d, exist_ok=True)
            open(os.path.join(d,'README.txt'),'w').write('Put images of '+cls)
    elif typ == 'text_generation':
        open(os.path.join(base,'data.jsonl'),'w').write('\n'.join([
            json.dumps({'text': 'Hello world'}),
            json.dumps({'text': 'Goodbye world'})
        ]))
    elif typ == 'qa':
        open(os.path.join(base,'qa.jsonl'),'w').write('\n'.join([
            json.dumps({'question': 'What is the capital of France?', 'answer': 'Paris'}),
            json.dumps({'question': '2+2?', 'answer': '4'})
        ]))
    elif typ == 'instruction_tuning':
        open(os.path.join(base,'instructions.jsonl'),'w').write('\n'.join([
            json.dumps({'instruction': 'Summarize the text', 'input': 'Long text here', 'output': 'Summary'}),
            json.dumps({'instruction': 'Translate to German', 'input': 'Hello', 'output': 'Hallo'})
        ]))
    elif typ == 'conversational':
        open(os.path.join(base,'conversations.jsonl'),'w').write('\n'.join([
            json.dumps({'messages': [{'role':'user','content':'Hi'},{'role':'assistant','content':'Hello!'}]}),
        ]))
    # Write metadata
    meta = {'type': typ, 'tags': [typ.replace('_',' ')], 'created': datetime.now().isoformat()}
    with open(os.path.join(_dataset_dir(name),'metadata.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    return jsonify({'status': 'ok', 'name': name, 'version': version, 'meta': meta})


@app.route('/api/datasets/<name>/download', methods=['GET'])
def dataset_download(name):
    name = _safe_name(name)
    version = _safe_name(request.args.get('version') or '')
    root = _dataset_version_dir(name, version if version else None)
    if not os.path.isdir(root):
        return jsonify({'error': 'Not found'}), 404
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, 'w', zipfile.ZIP_DEFLATED) as z:
        for r, _, files in os.walk(root):
            for fn in files:
                fp = os.path.join(r, fn)
                arc = os.path.relpath(fp, root)
                try:
                    z.write(fp, arcname=arc)
                except Exception:
                    pass
    mem.seek(0)
    return send_file(mem, mimetype='application/zip', as_attachment=True, download_name=f'{name}-{os.path.basename(root)}.zip')


@app.route('/api/datasets/<name>/validate', methods=['GET'])
def dataset_validate(name):
    name = _safe_name(name)
    version = _safe_name(request.args.get('version') or '')
    root = _dataset_version_dir(name, version if version else None)
    if not os.path.isdir(root):
        return jsonify({'error': 'Not found'}), 404
    stats = _compute_dataset_stats(root)
    issues = []
    if stats['total_files'] == 0:
        issues.append('Dataset contains no files')
    return jsonify({'name': name, 'version': os.path.basename(root), 'stats': stats, 'issues': issues})


@app.route('/api/datasets/<name>/quality', methods=['GET'])
def dataset_quality(name):
    name = _safe_name(name)
    version = _safe_name(request.args.get('version') or '')
    root = _dataset_version_dir(name, version if version else None)
    if not os.path.isdir(root):
        return jsonify({'error': 'Not found'}), 404
    # Exact duplicate detection via md5 of first N MiB (fast path)
    limit = int(request.args.get('limit') or 10000)
    include_near = (request.args.get('near') or '0') in ('1','true','yes')
    near_thresh = int(request.args.get('near_thresh') or 6)
    hashes = {}
    dup_groups = {}
    ahashes = []  # (hash_int, relpath)
    for idx, (r, _, fns) in enumerate(os.walk(root)):
        for fn in fns:
            fp = os.path.join(r, fn)
            try:
                h = hashlib.md5()
                with open(fp, 'rb') as f:
                    h.update(f.read(2*1024*1024))  # first 2 MiB
                digest = h.hexdigest()
                rel = os.path.relpath(fp, root).replace('\\','/')
                hashes.setdefault(digest, []).append(rel)
                # Perceptual aHash for images
                ext = os.path.splitext(fn)[1].lower()
                if include_near and Image is not None and ext in {'.jpg','.jpeg','.png','.bmp','.gif'}:
                    try:
                        with Image.open(fp) as im:
                            im = im.convert('L').resize((8,8))
                            pix = list(im.getdata())
                            avg = sum(pix)/len(pix)
                            bits = 0
                            for i,pv in enumerate(pix):
                                if pv >= avg:
                                    bits |= (1 << i)
                            ahashes.append((bits, rel))
                    except Exception:
                        pass
            except Exception:
                pass
            if len(hashes) > limit:
                break
    for k, paths in hashes.items():
        if len(paths) > 1:
            dup_groups[k] = paths
    near_groups = []
    if include_near and ahashes:
        # Bucket by top 12 bits to reduce comparisons
        buckets = {}
        for hv, rel in ahashes:
            buckets.setdefault(hv >> 52, []).append((hv, rel))
        def hamming(a,b):
            return bin(a ^ b).count('1')
        visited = set()
        for _, items in buckets.items():
            n = len(items)
            for i in range(n):
                if items[i][1] in visited:
                    continue
                group = [items[i][1]]
                for j in range(i+1, n):
                    if hamming(items[i][0], items[j][0]) <= near_thresh:
                        group.append(items[j][1])
                        visited.add(items[j][1])
                if len(group) > 1:
                    near_groups.append(group)
    # Class counts (image classification style) by top-level subfolders
    class_counts = {}
    for d in os.listdir(root):
        p = os.path.join(root, d)
        if os.path.isdir(p):
            cnt = 0
            for _, _, fns in os.walk(p):
                cnt += len(fns)
            class_counts[d] = cnt
    # Imbalance warning
    total = sum(class_counts.values()) or 1
    ratios = {k: v/total for k,v in class_counts.items()} if class_counts else {}
    max_cls = max(ratios.items(), key=lambda x:x[1])[0] if ratios else None
    imbalance = None
    if ratios and (max(ratios.values()) > 0.8 or (len(ratios)>=2 and (max(ratios.values())/max(1e-6, min(ratios.values()))) > 10)):
        imbalance = {'message': f'Class imbalance detected; class {max_cls} dominates', 'ratios': ratios}
    # Suggestions
    suggestions = {}
    dup_remove = []
    for paths in dup_groups.values():
        if len(paths) > 1:
            dup_remove.extend(paths[1:])
    suggestions['duplicates_to_remove'] = dup_remove
    if class_counts:
        counts = list(class_counts.values())
        median = sorted(counts)[len(counts)//2]
        suggestions['balance_plan'] = { c: min(median, n) for c, n in class_counts.items() }
    return jsonify({'duplicates': dup_groups, 'near_duplicates': near_groups, 'class_counts': class_counts, 'imbalance': imbalance, 'suggestions': suggestions})


@app.route('/api/datasets/<name>/quality/apply', methods=['POST'])
def dataset_quality_apply(name):
    """Apply quality suggestions by moving files to a quarantine folder.

    JSON: { version, remove?: [paths], balance?: {class: target_count} }
    """
    name = _safe_name(name)
    data = request.json or {}
    version = _safe_name(data.get('version') or '')
    base = _dataset_version_dir(name, version if version else None)
    if not os.path.isdir(base):
        return jsonify({'error': 'Not found'}), 404
    quarantine = os.path.join(base, '_quarantine')
    os.makedirs(quarantine, exist_ok=True)
    removed = []
    for rel in data.get('remove') or []:
        src = os.path.join(base, rel)
        if os.path.exists(src):
            dst = os.path.join(quarantine, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            try:
                shutil.move(src, dst)
                removed.append(rel)
            except Exception:
                pass
    balanced = {}
    for cls, target in (data.get('balance') or {}).items():
        cls_dir = os.path.join(base, cls)
        if not os.path.isdir(cls_dir):
            continue
        files = []
        for r, _, fns in os.walk(cls_dir):
            for fn in fns:
                files.append(os.path.join(r, fn))
        if len(files) <= int(target):
            continue
        extra = files[int(target):]
        moved = []
        for fp in extra:
            rel = os.path.relpath(fp, base)
            dst = os.path.join(quarantine, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            try:
                shutil.move(fp, dst)
                moved.append(rel)
            except Exception:
                pass
        balanced[cls] = {'moved': len(moved)}
    return jsonify({'status': 'ok', 'removed': removed, 'balanced': balanced})


@app.route('/api/datasets/ingest/preview', methods=['POST'])
def datasets_ingest_preview():
    data = request.json or {}
    typ = (data.get('type') or '').lower()
    text = data.get('text') or ''
    out = {'type': typ, 'columns': [], 'rows': []}
    try:
        if typ == 'csv':
            reader = csv.reader(text.splitlines())
            rows = list(reader)[:6]
            if rows:
                out['columns'] = rows[0]
                out['rows'] = rows[1:]
        elif typ == 'jsonl':
            rows = [json.loads(ln) for ln in text.splitlines() if ln.strip()][:6]
            cols = set()
            for r in rows: cols.update(r.keys())
            out['columns'] = sorted(cols)
            out['rows'] = rows
        else:
            return jsonify({'error':'Unsupported type'}), 400
    except Exception as e:
        return jsonify({'error':'Parse failed','detail':str(e)}), 400
    return jsonify(out)


@app.route('/api/datasets/ingest/apply', methods=['POST'])
def datasets_ingest_apply():
    data = request.json or {}
    name = _safe_name(data.get('name') or '')
    version = _safe_name(data.get('version') or datetime.now().strftime('%Y%m%d-%H%M%S'))
    typ = (data.get('type') or '').lower()
    mapping = data.get('mapping') or {}  # { new_field: source_field }
    rows = data.get('rows') or []  # for small in-memory apply
    if not name:
        return jsonify({'error':'Missing name'}), 400
    dst = _dataset_version_dir(name, version)
    os.makedirs(dst, exist_ok=True)
    outp = os.path.join(dst, 'data.jsonl')
    try:
        with open(outp, 'w', encoding='utf-8') as f:
            for r in rows:
                if typ == 'csv' and isinstance(r, list):
                    # assume header provided in mapping.source_header
                    header = data.get('header') or []
                    row_map = { header[i]: r[i] for i in range(min(len(header), len(r))) }
                    obj = { k: row_map.get(v) for k, v in mapping.items() } if mapping else row_map
                elif typ == 'jsonl' and isinstance(r, dict):
                    obj = { k: r.get(v) for k, v in mapping.items() } if mapping else r
                else:
                    obj = r
                f.write(json.dumps(obj, ensure_ascii=False) + '\n')
        return jsonify({'status':'ok','name': name, 'version': version, 'path': outp})
    except Exception as e:
        return jsonify({'error':'Failed to write','detail':str(e)}), 500


@app.route('/api/datasets/<name>/annotations', methods=['GET'])
def dataset_annotations_get(name):
    name = _safe_name(name)
    version = _safe_name(request.args.get('version') or '')
    base = _dataset_version_dir(name, version if version else None)
    p = os.path.join(base, 'annotations.jsonl')
    items = []
    if os.path.exists(p):
        try:
            with open(p,'r',encoding='utf-8') as f:
                for ln in f:
                    try: items.append(json.loads(ln))
                    except: pass
        except Exception:
            pass
    return jsonify({'items': items})


@app.route('/api/datasets/<name>/annotations/save', methods=['POST'])
def dataset_annotations_save(name):
    name = _safe_name(name)
    data = request.json or {}
    version = _safe_name(data.get('version') or '')
    base = _dataset_version_dir(name, version if version else None)
    if not os.path.isdir(base):
        return jsonify({'error':'Version not found'}), 404
    items = data.get('items') or []
    p = os.path.join(base, 'annotations.jsonl')
    try:
        with open(p,'w',encoding='utf-8') as f:
            for it in items:
                f.write(json.dumps(it, ensure_ascii=False)+'\n')
        return jsonify({'status':'ok','saved': len(items)})
    except Exception as e:
        return jsonify({'error':'Save failed','detail':str(e)}), 500


@app.route('/api/datasets/<name>/annotations/export/yolo', methods=['GET'])
def dataset_annotations_export_yolo(name):
    name = _safe_name(name)
    version = _safe_name(request.args.get('version') or '')
    base = _dataset_version_dir(name, version if version else None)
    ann_path = os.path.join(base, 'annotations.jsonl')
    if not os.path.exists(ann_path):
        return jsonify({'error': 'No annotations found'}), 404
    items = []
    labels = set()
    with open(ann_path, 'r', encoding='utf-8') as f:
        for ln in f:
            try:
                obj = json.loads(ln)
                if obj.get('bboxes'):
                    items.append(obj)
                    for b in obj['bboxes']:
                        if 'label' in b: labels.add(b['label'])
            except Exception:
                continue
    classes = sorted(labels)
    class_to_id = { c:i for i,c in enumerate(classes) }
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, 'w', zipfile.ZIP_DEFLATED) as z:
        z.writestr('classes.txt', '\n'.join(classes).encode('utf-8'))
        for it in items:
            ipath = os.path.join(base, it['path'])
            w = h = None
            if Image is not None and os.path.exists(ipath):
                try:
                    with Image.open(ipath) as im:
                        w, h = im.size
                except Exception:
                    pass
            if not w or not h:
                w = (it.get('image_size') or {}).get('w')
                h = (it.get('image_size') or {}).get('h')
            if not w or not h:
                continue
            lines = []
            for b in it.get('bboxes') or []:
                cx = (b['x'] + b['w']/2.0) / max(1.0, w)
                cy = (b['y'] + b['h']/2.0) / max(1.0, h)
                nw = b['w'] / max(1.0, w)
                nh = b['h'] / max(1.0, h)
                cls = class_to_id.get(b.get('label'), 0)
                lines.append(f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
            rel = it['path']
            normalized_path = rel.replace('\\\\', '/').replace('\\', '/')
            lbl_path = f"labels/{os.path.splitext(normalized_path)[0]}.txt"
            z.writestr(lbl_path, ('\n'.join(lines)+'\n').encode('utf-8'))
    mem.seek(0)
    return send_file(mem, mimetype='application/zip', as_attachment=True, download_name=f"{name}_{version or 'latest'}_yolo.zip")


@app.route('/api/datasets/<name>/annotations/export/coco', methods=['GET'])
def dataset_annotations_export_coco(name):
    name = _safe_name(name)
    version = _safe_name(request.args.get('version') or '')
    base = _dataset_version_dir(name, version if version else None)
    ann_path = os.path.join(base, 'annotations.jsonl')
    if not os.path.exists(ann_path):
        return jsonify({'error': 'No annotations found'}), 404
    items = []
    labels = set()
    with open(ann_path, 'r', encoding='utf-8') as f:
        for ln in f:
            try:
                obj = json.loads(ln)
                if obj.get('bboxes'):
                    items.append(obj)
                    for b in obj['bboxes']:
                        if 'label' in b: labels.add(b['label'])
            except Exception:
                continue
    categories = [{'id': i+1, 'name': c} for i, c in enumerate(sorted(labels))]
    cat_to_id = { c['name']: c['id'] for c in categories }
    images = []
    anns = []
    img_id = 1
    ann_id = 1
    for it in items:
        ipath = os.path.join(base, it['path'])
        w = h = None
        if Image is not None and os.path.exists(ipath):
            try:
                with Image.open(ipath) as im:
                    w, h = im.size
            except Exception:
                pass
        if not w or not h:
            w = (it.get('image_size') or {}).get('w')
            h = (it.get('image_size') or {}).get('h')
        if not w or not h:
            continue
        images.append({'id': img_id, 'file_name': it['path'], 'width': w, 'height': h})
        for b in it.get('bboxes') or []:
            anns.append({
                'id': ann_id,
                'image_id': img_id,
                'category_id': cat_to_id.get(b.get('label'), 1),
                'bbox': [float(b['x']), float(b['y']), float(b['w']), float(b['h'])],
                'area': float(b['w']) * float(b['h']),
                'iscrowd': 0,
            })
            ann_id += 1
        img_id += 1
    coco = {'images': images, 'annotations': anns, 'categories': categories}
    return jsonify(coco)


@app.route('/api/datasets/<name>/annotations/queue', methods=['GET'])
def dataset_annotations_queue(name):
    name = _safe_name(name)
    version = _safe_name(request.args.get('version') or '')
    strategy = (request.args.get('strategy') or 'missing').lower()
    limit = int(request.args.get('limit') or 50)
    base = _dataset_version_dir(name, version if version else None)
    if not os.path.isdir(base):
        return jsonify({'error': 'Not found'}), 404
    annotated = set()
    ann_path = os.path.join(base, 'annotations.jsonl')
    if os.path.exists(ann_path):
        try:
            for ln in open(ann_path, 'r', encoding='utf-8'):
                try:
                    it = json.loads(ln)
                    if it.get('path'):
                        annotated.add(it['path'])
                except Exception:
                    pass
        except Exception:
            pass
    image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
    all_paths = []
    for r, _, fns in os.walk(base):
        for fn in fns:
            ext = os.path.splitext(fn)[1].lower()
            if ext in image_exts:
                rel = os.path.relpath(os.path.join(r, fn), base).replace('\\', '/')
                all_paths.append(rel)
    cand = [p for p in all_paths if p not in annotated]
    if strategy == 'random':
        random.shuffle(cand)
    out = cand[:limit]
    return jsonify({'items': out, 'total_unlabeled': len(cand)})


@app.route('/api/datasets/<name>/annotations/prelabel', methods=['POST'])
def dataset_annotations_prelabel(name):
    data = request.json or {}
    version = _safe_name(data.get('version') or '')
    provider = (data.get('provider') or 'stub').lower()
    task = (data.get('task') or 'bbox').lower()
    tasks_path = os.path.join(JOBS_DIR, 'curation_tasks.jsonl')
    rec = {
        'id': str(uuid.uuid4()),
        'dataset': _safe_name(name),
        'version': version,
        'provider': provider,
        'task': task,
        'created': datetime.now().isoformat(),
        'status': 'accepted',
    }
    try:
        with open(tasks_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(rec) + '\n')
    except Exception:
        pass
    return jsonify({'status': 'accepted', 'provider': provider, 'note': 'Prelabeling hook queued (stub).'}), 202


@app.route('/api/datasets/sync', methods=['POST'])
def datasets_sync():
    """Best-effort sync to/from external storage using available CLI tools.

    JSON body: { provider: 's3'|'gcs'|'azure'|'minio', direction:'upload'|'download', bucket:'...', prefix:'', dataset:'optional' }
    No-op if required CLI tools are missing.
    """
    data = request.json or {}
    provider = (data.get('provider') or '').lower()
    direction = (data.get('direction') or 'upload').lower()
    bucket = data.get('bucket') or ''
    prefix = data.get('prefix') or ''
    ds = data.get('dataset')
    src = ''
    dst = ''
    note = ''
    if ds:
        src_dir = _dataset_dir(ds)
    else:
        src_dir = DATASETS_DIR
    if not os.path.isdir(src_dir):
        return jsonify({'error': 'Source directory not found'}), 404
    if provider == 's3':
        cli = shutil.which('aws')
        if not cli:
            return jsonify({'status': 'skipped', 'message': 'aws CLI not found'}), 200
        target = f's3://{bucket}/{prefix}'.rstrip('/')
        if direction == 'upload':
            cmd = [cli, 's3', 'sync', src_dir, target]
        else:
            cmd = [cli, 's3', 'sync', target, src_dir]
    elif provider == 'gcs':
        cli = shutil.which('gsutil')
        if not cli:
            return jsonify({'status': 'skipped', 'message': 'gsutil not found'}), 200
        target = f'gs://{bucket}/{prefix}'.rstrip('/')
        if direction == 'upload':
            cmd = [cli, '-m', 'rsync', '-r', src_dir, target]
        else:
            cmd = [cli, '-m', 'rsync', '-r', target, src_dir]
    elif provider == 'azure':
        cli = shutil.which('az')
        if not cli:
            return jsonify({'status': 'skipped', 'message': 'az CLI not found'}), 200
        # Use az storage blob sync (requires AZ env/args configured)
        if direction == 'upload':
            cmd = [cli, 'storage', 'blob', 'upload-batch', '--destination', bucket, '--source', src_dir]
        else:
            cmd = [cli, 'storage', 'blob', 'download-batch', '--destination', src_dir, '--source', bucket]
    elif provider == 'minio':
        cli = shutil.which('mc')
        if not cli:
            return jsonify({'status': 'skipped', 'message': 'minio mc not found'}), 200
        target = f'{bucket}/{prefix}'.rstrip('/')
        if direction == 'upload':
            cmd = [cli, 'mirror', src_dir, target]
        else:
            cmd = [cli, 'mirror', target, src_dir]
    else:
        return jsonify({'error': 'Unknown provider'}), 400
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        return jsonify({'status': 'ok', 'provider': provider, 'direction': direction, 'output': out.decode(errors='ignore')})
    except subprocess.CalledProcessError as e:
        return jsonify({'status': 'error', 'provider': provider, 'direction': direction, 'rc': e.returncode, 'output': (e.output or b'').decode(errors='ignore')}), 500


# ================= Video Dataset Processing =================

# Background job tracking for video indexing
_video_index_jobs = {}  # job_id -> {status, progress, manifest_path, etc}

@app.route('/api/datasets/index', methods=['POST'])
def dataset_index_video():
    """
    Index a video dataset directory.

    Creates a background job that:
    1. Scans directory for video files
    2. Extracts metadata (duration, fps, resolution, etc.)
    3. Generates JSONL manifest file

    POST body: {
        name: dataset name,
        source_path: path to video directory,
        version: optional version name,
        recursive: scan subdirectories (default: true),
        extract_metadata: extract full metadata (default: true),
        extensions: list of video extensions (default: [mp4, avi, mov, mkv, webm])
    }
    """
    data = request.json or {}
    name = _safe_name(data.get('name') or '')
    source_path = data.get('source_path', '')
    version = _safe_name(data.get('version') or datetime.now().strftime('%Y%m%d-%H%M%S'))
    recursive = data.get('recursive', True)
    extract_metadata = data.get('extract_metadata', True)
    extensions = data.get('extensions')

    if not name:
        return jsonify({'error': 'Dataset name required'}), 400

    if not source_path or not os.path.isdir(source_path):
        return jsonify({'error': 'Valid source_path directory required'}), 400

    # Check if ffmpeg is available
    try:
        from backend.utils import video as video_utils
        if not video_utils.check_ffmpeg_installed():
            return jsonify({'error': 'FFmpeg not installed. Please install ffmpeg and ffprobe.'}), 400
    except ImportError:
        return jsonify({'error': 'Video utils not available'}), 500

    # Create dataset version directory
    dataset_dir = _dataset_version_dir(name, version)
    os.makedirs(dataset_dir, exist_ok=True)

    # Create job
    job_id = str(uuid.uuid4())
    manifest_path = os.path.join(dataset_dir, 'manifest.jsonl')

    job = {
        'id': job_id,
        'name': name,
        'version': version,
        'source_path': source_path,
        'manifest_path': manifest_path,
        'status': 'running',
        'progress': 0,
        'total_videos': 0,
        'processed': 0,
        'created': datetime.now().isoformat(),
        'log': []
    }

    _video_index_jobs[job_id] = job

    # Start background indexing
    def index_worker():
        try:
            job['log'].append(f"Scanning {source_path}...")

            # Scan for videos
            videos = video_utils.scan_video_directory(
                source_path,
                extensions=extensions,
                recursive=recursive
            )

            job['total_videos'] = len(videos)
            job['log'].append(f"Found {len(videos)} videos")

            if len(videos) == 0:
                job['status'] = 'completed'
                job['log'].append("No videos found")
                return

            # Build manifest
            def progress_callback(current, total):
                job['processed'] = current
                job['progress'] = int((current / total) * 100) if total > 0 else 0

            video_utils.build_video_manifest(
                videos,
                manifest_path,
                extract_metadata=extract_metadata,
                progress_callback=progress_callback
            )

            # Get statistics
            stats = video_utils.get_video_stats(manifest_path)
            job['stats'] = stats
            job['status'] = 'completed'
            job['progress'] = 100
            job['log'].append(f"Completed: {stats['total_videos']} videos indexed")

            # Save metadata
            meta_path = os.path.join(dataset_dir, 'metadata.json')
            with open(meta_path, 'w') as f:
                json.dump({
                    'name': name,
                    'version': version,
                    'type': 'video_classification',
                    'source_path': source_path,
                    'created': job['created'],
                    'stats': stats
                }, f, indent=2)

        except Exception as e:
            job['status'] = 'failed'
            job['error'] = str(e)
            job['log'].append(f"Error: {str(e)}")

    # Start in background thread
    thread = threading.Thread(target=index_worker, daemon=True)
    thread.start()

    return jsonify({
        'status': 'ok',
        'job_id': job_id,
        'message': 'Video indexing started in background'
    }), 202


@app.route('/api/datasets/index/<job_id>', methods=['GET'])
def get_index_job_status(job_id):
    """Get status of video indexing job"""
    job = _video_index_jobs.get(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404

    return jsonify(job)


@app.route('/api/datasets/<name>/process', methods=['POST'])
def dataset_process_video(name):
    """
    Process videos in a dataset (extract frames, transcribe audio).

    POST body: {
        version: dataset version,
        num_frames: frames to extract per video (default: 16),
        extract_audio: extract audio tracks (default: false),
        transcribe: transcribe with Whisper (default: false),
        whisper_model: Whisper model size (default: 'base'),
        max_videos: limit processing to first N videos (optional)
    }
    """
    name = _safe_name(name)
    data = request.json or {}
    version = _safe_name(data.get('version') or '')
    num_frames = data.get('num_frames', 16)
    extract_audio_flag = data.get('extract_audio', False)
    transcribe = data.get('transcribe', False)
    whisper_model = data.get('whisper_model', 'base')
    max_videos = data.get('max_videos')

    dataset_dir = _dataset_version_dir(name, version if version else None)
    manifest_path = os.path.join(dataset_dir, 'manifest.jsonl')

    if not os.path.exists(manifest_path):
        return jsonify({'error': 'No manifest found. Run indexing first.'}), 404

    # Load videos from manifest
    videos = []
    with open(manifest_path, 'r') as f:
        for idx, line in enumerate(f):
            if max_videos and idx >= max_videos:
                break
            videos.append(json.loads(line))

    # Create processing job
    job_id = str(uuid.uuid4())
    job = {
        'id': job_id,
        'name': name,
        'version': version,
        'status': 'running',
        'progress': 0,
        'total_videos': len(videos),
        'processed': 0,
        'created': datetime.now().isoformat(),
        'log': []
    }

    _video_index_jobs[job_id] = job

    def process_worker():
        try:
            from backend.utils import video as video_utils

            processed_manifest = os.path.join(dataset_dir, 'processed_manifest.jsonl')

            with open(processed_manifest, 'w') as out_f:
                for idx, video in enumerate(videos):
                    try:
                        video_path = video['path']
                        output_dir = os.path.join(dataset_dir, 'processed', f'video_{idx:06d}')

                        result = video_utils.process_video_for_training(
                            video_path,
                            output_dir,
                            num_frames=num_frames,
                            extract_audio=extract_audio_flag,
                            transcribe=transcribe,
                            whisper_model=whisper_model
                        )

                        # Merge with original video metadata
                        processed_record = {**video, **result, 'output_dir': output_dir}
                        out_f.write(json.dumps(processed_record) + '\n')

                        job['processed'] = idx + 1
                        job['progress'] = int(((idx + 1) / len(videos)) * 100)

                    except Exception as e:
                        job['log'].append(f"Error processing {video.get('filename', 'unknown')}: {str(e)}")

            job['status'] = 'completed'
            job['progress'] = 100
            job['log'].append(f"Processed {len(videos)} videos")

        except Exception as e:
            job['status'] = 'failed'
            job['error'] = str(e)
            job['log'].append(f"Processing failed: {str(e)}")

    thread = threading.Thread(target=process_worker, daemon=True)
    thread.start()

    return jsonify({
        'status': 'ok',
        'job_id': job_id,
        'message': 'Video processing started in background'
    }), 202


# ================= HPO studies persistence =================
@app.route('/api/hpo/studies', methods=['GET'])
def hpo_studies_list():
    p = os.path.join(JOBS_DIR, 'hpo_studies.jsonl')
    items = []
    if os.path.exists(p):
        try:
            with open(p, 'r', encoding='utf-8') as f:
                for ln in f:
                    try:
                        items.append(json.loads(ln))
                    except Exception:
                        pass
        except Exception:
            pass
    return jsonify({'items': items[-100:]})


@app.route('/api/hpo/studies/save', methods=['POST'])
def hpo_studies_save():
    data = request.json or {}
    rec = {
        'id': str(uuid.uuid4()),
        'created': datetime.now().isoformat(),
        'study': data,
    }
    p = os.path.join(JOBS_DIR, 'hpo_studies.jsonl')
    try:
        with open(p, 'a', encoding='utf-8') as f:
            f.write(json.dumps(rec) + '\n')
        return jsonify({'status': 'ok', 'id': rec['id']})
    except Exception as e:
        return jsonify({'error': 'Failed to save', 'detail': str(e)}), 500


def _sample_metrics_once():
    """Take one metrics sample and append to history."""
    try:
        snap = _system_info_snapshot()
        ts = datetime.now().isoformat()
        gpus = []
        for g in snap.get('gpus', []):
            gpus.append({
                'index': g.get('index'),
                'util_pct': g.get('utilization_gpu_pct'),
                'mem_pct': g.get('memory_used_pct'),
            })
        sample = {
            'ts': ts,
            'gpus': gpus,
            'gpu_mem_used_pct': snap.get('memory_used_pct'),
            'sys_mem_used_pct': (snap.get('memory') or {}).get('used_pct'),
        }
        with _metrics_lock:
            _metrics_history.append(sample)
    except Exception:
        pass


def _ensure_metrics_thread():
    global _metrics_thread, _metrics_thread_started
    if _metrics_thread_started:
        return
    _metrics_thread_started = True

    def _run():
        while True:
            _sample_metrics_once()
            time.sleep(max(1, METRICS_INTERVAL_SECONDS))

    t = threading.Thread(target=_run, daemon=True)
    _metrics_thread = t
    t.start()


def _ensure_scheduler_thread():
    global _scheduler_started
    if _scheduler_started:
        return
    _scheduler_started = True

    def _run_sched():
        while True:
            try:
                scheds = _load_json(SCHEDULES_PATH, [])
                changed = False
                now = time.time()
                for s in scheds:
                    last = float(s.get('last_run_ts') or 0)
                    interval = int(s.get('interval_seconds') or 0)
                    if interval > 0 and (now - last) >= interval:
                        payload = s.get('payload') or {}
                        try:
                            with app.test_request_context(json=payload):
                                create_job()
                            s['last_run_ts'] = now
                            changed = True
                        except Exception:
                            pass
                if changed:
                    _save_json(SCHEDULES_PATH, scheds)
            except Exception:
                pass
            time.sleep(30)

    t = threading.Thread(target=_run_sched, daemon=True)
    t.start()


@app.route('/api/system/info', methods=['GET'])
def system_info():
    """Get DGX system information (single snapshot)."""
    _ensure_metrics_thread()
    _ensure_scheduler_thread()
    return jsonify(_system_info_snapshot())


@app.route('/api/system/io', methods=['GET'])
def system_io():
    """Basic I/O profiling from /proc/diskstats (aggregate reads/writes per second)."""
    reads = writes = 0
    try:
        with open('/proc/diskstats', 'r') as f:
            for ln in f:
                parts = ln.split()
                if len(parts) < 14:
                    continue
                # fields: https://www.kernel.org/doc/Documentation/ABI/testing/procfs-diskstats
                # reads completed (3), sectors read (5); writes completed (7), sectors written (9)
                try:
                    reads += int(parts[5])
                    writes += int(parts[9])
                except Exception:
                    pass
    except Exception:
        pass
    now = time.time()
    rps = wps = 0.0
    if _last_io_totals['reads'] is not None and _last_io_totals['ts'] is not None:
        dt = max(1e-6, now - _last_io_totals['ts'])
        rps = (reads - _last_io_totals['reads']) / dt
        wps = (writes - _last_io_totals['writes']) / dt
    _last_io_totals.update({'reads': reads, 'writes': writes, 'ts': now})
    return jsonify({'reads_per_sec': rps, 'writes_per_sec': wps})


# ---------------- Pipelines & Scheduling ----------------
@app.route('/api/pipelines', methods=['GET','POST'])
def pipelines_root():
    if request.method == 'POST':
        data = request.json or {}
        pipes = _load_json(PIPELINES_PATH, [])
        pid = data.get('id') or str(uuid.uuid4())
        data['id'] = pid
        data['created'] = datetime.now().isoformat()
        pipes.append(data)
        _save_json(PIPELINES_PATH, pipes)
        return jsonify({'status': 'ok', 'id': pid}), 201
    return jsonify({'items': _load_json(PIPELINES_PATH, [])})


@app.route('/api/pipelines/<pid>', methods=['GET', 'PUT', 'DELETE'])
def pipeline_detail(pid):
    """Get, update, or delete a specific pipeline"""
    pipes = _load_json(PIPELINES_PATH, [])
    pipe_idx = next((i for i, p in enumerate(pipes) if p.get('id') == pid), None)

    if pipe_idx is None:
        return jsonify({'error': 'Pipeline not found'}), 404

    if request.method == 'DELETE':
        pipes.pop(pipe_idx)
        _save_json(PIPELINES_PATH, pipes)
        return jsonify({'status': 'ok', 'deleted': pid})

    if request.method == 'PUT':
        data = request.json or {}
        # Preserve ID and created timestamp
        data['id'] = pid
        data['created'] = pipes[pipe_idx].get('created', datetime.now().isoformat())
        data['updated'] = datetime.now().isoformat()
        pipes[pipe_idx] = data
        _save_json(PIPELINES_PATH, pipes)
        return jsonify(data)

    # GET
    return jsonify(pipes[pipe_idx])

@app.route('/api/pipelines/<pid>/run', methods=['POST'])
@app.route('/api/pipelines/<pid>/execute', methods=['POST'])
def pipelines_execute(pid):
    """Execute a pipeline (creates jobs for all nodes)"""
    pipes = _load_json(PIPELINES_PATH, [])
    pipe = next((p for p in pipes if p.get('id') == pid), None)
    if not pipe:
        return jsonify({'error': 'Pipeline not found'}), 404

    nodes = pipe.get('nodes') or []
    edges = pipe.get('edges') or []

    # Compute dependencies map
    deps = { n.get('id'): [] for n in nodes }
    for e in edges:
        deps.setdefault(e.get('to'), []).append(e.get('from'))

    # Store execution metadata
    execution_id = str(uuid.uuid4())
    execution_start = datetime.now().isoformat()

    id_map = {}
    errors = []

    for n in nodes:
        node_id = n.get('id')
        payload = (n.get('job') or {}).copy()

        # Map dependencies to created job IDs
        depends_on_nodes = deps.get(node_id) or []
        dep_ids = [id_map[d] for d in depends_on_nodes if d in id_map]
        if dep_ids:
            payload['depends_on'] = dep_ids

        # Tag job with pipeline and execution info
        if 'metadata' not in payload:
            payload['metadata'] = {}
        payload['metadata']['pipeline_id'] = pid
        payload['metadata']['pipeline_name'] = pipe.get('name', 'Unnamed Pipeline')
        payload['metadata']['execution_id'] = execution_id
        payload['metadata']['node_id'] = node_id
        payload['metadata']['node_name'] = n.get('name', 'Unnamed Node')

        try:
            with app.test_request_context(json=payload):
                resp, code = create_job()
                if code == 201:
                    job_obj = resp.get_json()
                    id_map[node_id] = job_obj.get('id')
                else:
                    errors.append({'node_id': node_id, 'error': 'Failed to create job'})
        except Exception as e:
            errors.append({'node_id': node_id, 'error': str(e)})
            continue

    # Save execution record
    if not hasattr(pipe, 'executions'):
        pipe['executions'] = []

    execution_record = {
        'id': execution_id,
        'started': execution_start,
        'node_jobs': id_map,
        'errors': errors,
        'status': 'running' if id_map else 'failed'
    }

    # Update pipeline with execution record
    for i, p in enumerate(pipes):
        if p.get('id') == pid:
            if 'executions' not in p:
                p['executions'] = []
            p['executions'].append(execution_record)
            pipes[i] = p
            break

    _save_json(PIPELINES_PATH, pipes)

    return jsonify({
        'status': 'ok',
        'execution_id': execution_id,
        'jobs': id_map,
        'errors': errors
    })

@app.route('/api/pipelines/<pid>/status', methods=['GET'])
def pipeline_status(pid):
    """Get execution status of a pipeline"""
    pipes = _load_json(PIPELINES_PATH, [])
    pipe = next((p for p in pipes if p.get('id') == pid), None)

    if not pipe:
        return jsonify({'error': 'Pipeline not found'}), 404

    executions = pipe.get('executions', [])

    # Get latest execution or all executions
    execution_id = request.args.get('execution_id')

    if execution_id:
        execution = next((e for e in executions if e.get('id') == execution_id), None)
        if not execution:
            return jsonify({'error': 'Execution not found'}), 404
        executions_to_check = [execution]
    else:
        # Return status of latest execution
        executions_to_check = [executions[-1]] if executions else []

    result = []
    for execution in executions_to_check:
        node_jobs = execution.get('node_jobs', {})
        job_statuses = {}

        for node_id, job_id in node_jobs.items():
            if job_id in jobs:
                job = jobs[job_id]
                job_statuses[node_id] = {
                    'job_id': job_id,
                    'status': job.get('status'),
                    'progress': job.get('progress', 0),
                    'name': job.get('name')
                }
            else:
                job_statuses[node_id] = {
                    'job_id': job_id,
                    'status': 'unknown',
                    'progress': 0
                }

        # Determine overall execution status
        statuses = [js['status'] for js in job_statuses.values()]
        if all(s == 'completed' for s in statuses):
            overall_status = 'completed'
        elif any(s == 'failed' for s in statuses):
            overall_status = 'failed'
        elif any(s == 'running' for s in statuses):
            overall_status = 'running'
        elif all(s == 'pending' for s in statuses):
            overall_status = 'pending'
        else:
            overall_status = 'partial'

        result.append({
            'execution_id': execution.get('id'),
            'started': execution.get('started'),
            'status': overall_status,
            'nodes': job_statuses,
            'errors': execution.get('errors', [])
        })

    if execution_id:
        return jsonify(result[0] if result else {})
    else:
        return jsonify({
            'pipeline_id': pid,
            'latest_execution': result[0] if result else None,
            'total_executions': len(executions)
        })

@app.route('/api/pipelines/<pid>/stages/<stage_id>/retry', methods=['POST'])
def pipeline_stage_retry(pid, stage_id):
    """Retry a failed stage in a pipeline"""
    pipes = _load_json(PIPELINES_PATH, [])
    pipe = next((p for p in pipes if p.get('id') == pid), None)

    if not pipe:
        return jsonify({'error': 'Pipeline not found'}), 404

    # Get the execution ID from request
    data = request.json or {}
    execution_id = data.get('execution_id')

    if not execution_id:
        return jsonify({'error': 'execution_id required'}), 400

    executions = pipe.get('executions', [])
    execution = next((e for e in executions if e.get('id') == execution_id), None)

    if not execution:
        return jsonify({'error': 'Execution not found'}), 404

    # Find the node/stage in the pipeline
    nodes = pipe.get('nodes', [])
    node = next((n for n in nodes if n.get('id') == stage_id), None)

    if not node:
        return jsonify({'error': 'Stage not found'}), 404

    # Get the original job ID for this stage
    node_jobs = execution.get('node_jobs', {})
    original_job_id = node_jobs.get(stage_id)

    if not original_job_id:
        return jsonify({'error': 'No job found for this stage'}), 404

    # Create a new job with the same configuration
    payload = (node.get('job') or {}).copy()

    # Tag with retry metadata
    if 'metadata' not in payload:
        payload['metadata'] = {}
    payload['metadata']['pipeline_id'] = pid
    payload['metadata']['execution_id'] = execution_id
    payload['metadata']['node_id'] = stage_id
    payload['metadata']['retry_of'] = original_job_id

    try:
        with app.test_request_context(json=payload):
            resp, code = create_job()
            if code == 201:
                job_obj = resp.get_json()
                new_job_id = job_obj.get('id')

                # Update the execution record
                for i, p in enumerate(pipes):
                    if p.get('id') == pid:
                        for j, e in enumerate(p.get('executions', [])):
                            if e.get('id') == execution_id:
                                if 'retries' not in e:
                                    e['retries'] = []
                                e['retries'].append({
                                    'stage_id': stage_id,
                                    'original_job_id': original_job_id,
                                    'retry_job_id': new_job_id,
                                    'timestamp': datetime.now().isoformat()
                                })
                                # Update the node_jobs mapping to point to new job
                                e['node_jobs'][stage_id] = new_job_id
                                pipes[i]['executions'][j] = e
                                break
                        break

                _save_json(PIPELINES_PATH, pipes)

                return jsonify({
                    'status': 'ok',
                    'original_job_id': original_job_id,
                    'retry_job_id': new_job_id
                })
            else:
                return jsonify({'error': 'Failed to create retry job'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/pipelines/templates', methods=['GET'])
def pipeline_templates():
    """Get predefined pipeline templates"""
    templates = [
        {
            'id': 'simple-train-eval',
            'name': 'Simple Train & Evaluate',
            'description': 'Train a model and evaluate it on a test set',
            'category': 'basic',
            'nodes': [
                {
                    'id': 'train',
                    'name': 'Train Model',
                    'type': 'train',
                    'position': {'x': 100, 'y': 100},
                    'job': {
                        'name': 'Training Job',
                        'type': 'train',
                        'framework': 'pytorch'
                    }
                },
                {
                    'id': 'eval',
                    'name': 'Evaluate Model',
                    'type': 'evaluate',
                    'position': {'x': 400, 'y': 100},
                    'job': {
                        'name': 'Evaluation Job',
                        'type': 'finetune',
                        'framework': 'pytorch'
                    }
                }
            ],
            'edges': [
                {'from': 'train', 'to': 'eval'}
            ]
        },
        {
            'id': 'data-prep-train',
            'name': 'Data Prep → Train',
            'description': 'Prepare dataset then train a model',
            'category': 'basic',
            'nodes': [
                {
                    'id': 'prep',
                    'name': 'Data Preparation',
                    'type': 'preprocess',
                    'position': {'x': 100, 'y': 100}
                },
                {
                    'id': 'train',
                    'name': 'Train Model',
                    'type': 'train',
                    'position': {'x': 400, 'y': 100}
                }
            ],
            'edges': [
                {'from': 'prep', 'to': 'train'}
            ]
        },
        {
            'id': 'hyperparameter-sweep',
            'name': 'Hyperparameter Sweep',
            'description': 'Train multiple models with different hyperparameters',
            'category': 'optimization',
            'nodes': [
                {
                    'id': 'train-lr1',
                    'name': 'Train (LR=0.001)',
                    'type': 'train',
                    'position': {'x': 100, 'y': 50}
                },
                {
                    'id': 'train-lr2',
                    'name': 'Train (LR=0.01)',
                    'type': 'train',
                    'position': {'x': 100, 'y': 150}
                },
                {
                    'id': 'train-lr3',
                    'name': 'Train (LR=0.1)',
                    'type': 'train',
                    'position': {'x': 100, 'y': 250}
                },
                {
                    'id': 'compare',
                    'name': 'Compare Results',
                    'type': 'evaluate',
                    'position': {'x': 400, 'y': 150}
                }
            ],
            'edges': [
                {'from': 'train-lr1', 'to': 'compare'},
                {'from': 'train-lr2', 'to': 'compare'},
                {'from': 'train-lr3', 'to': 'compare'}
            ]
        },
        {
            'id': 'ensemble',
            'name': 'Model Ensemble',
            'description': 'Train multiple models and ensemble them',
            'category': 'advanced',
            'nodes': [
                {
                    'id': 'model1',
                    'name': 'Model 1 (ResNet)',
                    'type': 'train',
                    'position': {'x': 100, 'y': 50}
                },
                {
                    'id': 'model2',
                    'name': 'Model 2 (VGG)',
                    'type': 'train',
                    'position': {'x': 100, 'y': 150}
                },
                {
                    'id': 'model3',
                    'name': 'Model 3 (DenseNet)',
                    'type': 'train',
                    'position': {'x': 100, 'y': 250}
                },
                {
                    'id': 'ensemble',
                    'name': 'Ensemble',
                    'type': 'evaluate',
                    'position': {'x': 400, 'y': 150}
                }
            ],
            'edges': [
                {'from': 'model1', 'to': 'ensemble'},
                {'from': 'model2', 'to': 'ensemble'},
                {'from': 'model3', 'to': 'ensemble'}
            ]
        },
        {
            'id': 'model-merging',
            'name': 'Model Merging (SLERP/TIES)',
            'description': 'Merge multiple pre-trained models into a single unified model',
            'category': 'advanced',
            'nodes': [
                {
                    'id': 'model1',
                    'name': 'Load Model 1',
                    'type': 'load',
                    'position': {'x': 100, 'y': 50},
                    'job': {
                        'name': 'Load Base Model 1',
                        'type': 'load',
                        'framework': 'pytorch',
                        'config': {'model_source': 'huggingface', 'model_name': 'bert-base-uncased'}
                    }
                },
                {
                    'id': 'model2',
                    'name': 'Load Model 2',
                    'type': 'load',
                    'position': {'x': 100, 'y': 150},
                    'job': {
                        'name': 'Load Base Model 2',
                        'type': 'load',
                        'framework': 'pytorch',
                        'config': {'model_source': 'huggingface', 'model_name': 'roberta-base'}
                    }
                },
                {
                    'id': 'model3',
                    'name': 'Load Model 3',
                    'type': 'load',
                    'position': {'x': 100, 'y': 250},
                    'job': {
                        'name': 'Load Base Model 3',
                        'type': 'load',
                        'framework': 'pytorch',
                        'config': {'model_source': 'huggingface', 'model_name': 'distilbert-base-uncased'}
                    }
                },
                {
                    'id': 'merge',
                    'name': 'Merge Models',
                    'type': 'merge',
                    'position': {'x': 400, 'y': 150},
                    'job': {
                        'name': 'Merge Models (SLERP)',
                        'type': 'merge',
                        'framework': 'pytorch',
                        'config': {
                            'merge_method': 'slerp',  # slerp, ties, dare, passthrough
                            'weights': [0.4, 0.4, 0.2],
                            'base_model': 'model1'
                        }
                    }
                },
                {
                    'id': 'eval',
                    'name': 'Evaluate Merged',
                    'type': 'evaluate',
                    'position': {'x': 700, 'y': 150},
                    'job': {
                        'name': 'Evaluate Merged Model',
                        'type': 'evaluate',
                        'framework': 'pytorch'
                    }
                }
            ],
            'edges': [
                {'from': 'model1', 'to': 'merge'},
                {'from': 'model2', 'to': 'merge'},
                {'from': 'model3', 'to': 'merge'},
                {'from': 'merge', 'to': 'eval'}
            ]
        },
        {
            'id': 'video-processing',
            'name': 'Video Dataset Processing',
            'description': 'Index videos, extract frames, generate transcripts, and train classifier',
            'category': 'video',
            'nodes': [
                {
                    'id': 'index',
                    'name': 'Index Videos',
                    'type': 'index',
                    'position': {'x': 100, 'y': 100},
                    'job': {
                        'name': 'Index Video Dataset',
                        'type': 'index',
                        'config': {
                            'source_path': '/data/videos',
                            'extract_audio': True,
                            'generate_manifest': True
                        }
                    }
                },
                {
                    'id': 'extract-frames',
                    'name': 'Extract Frames',
                    'type': 'preprocess',
                    'position': {'x': 400, 'y': 50},
                    'job': {
                        'name': 'Extract Frames',
                        'type': 'preprocess',
                        'config': {
                            'fps': 1,
                            'format': 'jpg',
                            'quality': 95
                        }
                    }
                },
                {
                    'id': 'transcribe',
                    'name': 'Generate Transcripts',
                    'type': 'transcribe',
                    'position': {'x': 400, 'y': 200},
                    'job': {
                        'name': 'Whisper Transcription',
                        'type': 'transcribe',
                        'config': {
                            'model': 'whisper-large-v3',
                            'language': 'en'
                        }
                    }
                },
                {
                    'id': 'train',
                    'name': 'Train Video Classifier',
                    'type': 'train',
                    'position': {'x': 700, 'y': 125},
                    'job': {
                        'name': 'VideoMAE Classification',
                        'type': 'train',
                        'framework': 'pytorch',
                        'config': {
                            'model': 'videomae-base',
                            'num_frames': 16,
                            'batch_size': 8,
                            'epochs': 10
                        }
                    }
                }
            ],
            'edges': [
                {'from': 'index', 'to': 'extract-frames'},
                {'from': 'index', 'to': 'transcribe'},
                {'from': 'extract-frames', 'to': 'train'},
                {'from': 'transcribe', 'to': 'train'}
            ]
        },
        {
            'id': 'transfer-learning',
            'name': 'Transfer Learning Pipeline',
            'description': 'Pre-train on large dataset, then fine-tune on specific task',
            'category': 'advanced',
            'nodes': [
                {
                    'id': 'pretrain',
                    'name': 'Pre-training',
                    'type': 'train',
                    'position': {'x': 100, 'y': 100},
                    'job': {
                        'name': 'Pre-train on ImageNet',
                        'type': 'train',
                        'framework': 'pytorch',
                        'config': {
                            'dataset': 'imagenet-1k',
                            'epochs': 100,
                            'lr': 0.1
                        }
                    }
                },
                {
                    'id': 'finetune',
                    'name': 'Fine-tuning',
                    'type': 'finetune',
                    'position': {'x': 400, 'y': 100},
                    'job': {
                        'name': 'Fine-tune on CIFAR-10',
                        'type': 'finetune',
                        'framework': 'pytorch',
                        'config': {
                            'dataset': 'cifar10',
                            'epochs': 20,
                            'lr': 0.001,
                            'freeze_layers': True
                        }
                    }
                },
                {
                    'id': 'eval',
                    'name': 'Evaluate',
                    'type': 'evaluate',
                    'position': {'x': 700, 'y': 100}
                }
            ],
            'edges': [
                {'from': 'pretrain', 'to': 'finetune'},
                {'from': 'finetune', 'to': 'eval'}
            ]
        },
        {
            'id': 'distillation',
            'name': 'Knowledge Distillation',
            'description': 'Train a smaller student model from a larger teacher model',
            'category': 'optimization',
            'nodes': [
                {
                    'id': 'teacher',
                    'name': 'Train Teacher',
                    'type': 'train',
                    'position': {'x': 100, 'y': 100},
                    'job': {
                        'name': 'Train Large Teacher Model',
                        'type': 'train',
                        'framework': 'pytorch',
                        'config': {
                            'model': 'resnet152',
                            'epochs': 100
                        }
                    }
                },
                {
                    'id': 'student',
                    'name': 'Distill to Student',
                    'type': 'distill',
                    'position': {'x': 400, 'y': 100},
                    'job': {
                        'name': 'Train Small Student Model',
                        'type': 'distill',
                        'framework': 'pytorch',
                        'config': {
                            'student_model': 'resnet18',
                            'temperature': 3.0,
                            'alpha': 0.7,
                            'epochs': 50
                        }
                    }
                },
                {
                    'id': 'compare',
                    'name': 'Compare Performance',
                    'type': 'evaluate',
                    'position': {'x': 700, 'y': 100}
                }
            ],
            'edges': [
                {'from': 'teacher', 'to': 'student'},
                {'from': 'student', 'to': 'compare'}
            ]
        },
        {
            'id': 'data-augmentation',
            'name': 'Data Augmentation Pipeline',
            'description': 'Apply various augmentations and train multiple models',
            'category': 'preprocessing',
            'nodes': [
                {
                    'id': 'original',
                    'name': 'Original Data',
                    'type': 'data',
                    'position': {'x': 100, 'y': 150}
                },
                {
                    'id': 'aug1',
                    'name': 'Augmentation 1',
                    'type': 'augment',
                    'position': {'x': 300, 'y': 50},
                    'job': {
                        'config': {'method': 'mixup'}
                    }
                },
                {
                    'id': 'aug2',
                    'name': 'Augmentation 2',
                    'type': 'augment',
                    'position': {'x': 300, 'y': 150},
                    'job': {
                        'config': {'method': 'cutmix'}
                    }
                },
                {
                    'id': 'aug3',
                    'name': 'Augmentation 3',
                    'type': 'augment',
                    'position': {'x': 300, 'y': 250},
                    'job': {
                        'config': {'method': 'randaugment'}
                    }
                },
                {
                    'id': 'train',
                    'name': 'Train with Best Aug',
                    'type': 'train',
                    'position': {'x': 500, 'y': 150}
                }
            ],
            'edges': [
                {'from': 'original', 'to': 'aug1'},
                {'from': 'original', 'to': 'aug2'},
                {'from': 'original', 'to': 'aug3'},
                {'from': 'aug1', 'to': 'train'},
                {'from': 'aug2', 'to': 'train'},
                {'from': 'aug3', 'to': 'train'}
            ]
        },
        {
            'id': 'multimodal-training',
            'name': 'Multimodal Training (Vision + Text)',
            'description': 'Train a model on both images and text (CLIP-style)',
            'category': 'advanced',
            'nodes': [
                {
                    'id': 'image-encoder',
                    'name': 'Image Encoder',
                    'type': 'train',
                    'position': {'x': 100, 'y': 50},
                    'job': {
                        'name': 'Train Vision Encoder',
                        'config': {'model': 'vit-base'}
                    }
                },
                {
                    'id': 'text-encoder',
                    'name': 'Text Encoder',
                    'type': 'train',
                    'position': {'x': 100, 'y': 200},
                    'job': {
                        'name': 'Train Text Encoder',
                        'config': {'model': 'bert-base'}
                    }
                },
                {
                    'id': 'contrastive',
                    'name': 'Contrastive Learning',
                    'type': 'train',
                    'position': {'x': 400, 'y': 125},
                    'job': {
                        'name': 'CLIP-style Training',
                        'config': {
                            'loss': 'contrastive',
                            'temperature': 0.07
                        }
                    }
                },
                {
                    'id': 'eval',
                    'name': 'Evaluate',
                    'type': 'evaluate',
                    'position': {'x': 700, 'y': 125}
                }
            ],
            'edges': [
                {'from': 'image-encoder', 'to': 'contrastive'},
                {'from': 'text-encoder', 'to': 'contrastive'},
                {'from': 'contrastive', 'to': 'eval'}
            ]
        },
        {
            'id': 'continual-learning',
            'name': 'Continual Learning Pipeline',
            'description': 'Train on multiple tasks sequentially without forgetting',
            'category': 'advanced',
            'nodes': [
                {
                    'id': 'task1',
                    'name': 'Task 1',
                    'type': 'train',
                    'position': {'x': 100, 'y': 125},
                    'job': {
                        'name': 'Train on Task 1',
                        'config': {'dataset': 'task1'}
                    }
                },
                {
                    'id': 'task2',
                    'name': 'Task 2 + Replay',
                    'type': 'train',
                    'position': {'x': 350, 'y': 125},
                    'job': {
                        'name': 'Train on Task 2 with Replay',
                        'config': {
                            'dataset': 'task2',
                            'replay_buffer': True,
                            'ewc_lambda': 0.4
                        }
                    }
                },
                {
                    'id': 'task3',
                    'name': 'Task 3 + Replay',
                    'type': 'train',
                    'position': {'x': 600, 'y': 125},
                    'job': {
                        'name': 'Train on Task 3 with Replay',
                        'config': {
                            'dataset': 'task3',
                            'replay_buffer': True,
                            'ewc_lambda': 0.4
                        }
                    }
                },
                {
                    'id': 'eval-all',
                    'name': 'Evaluate All Tasks',
                    'type': 'evaluate',
                    'position': {'x': 850, 'y': 125}
                }
            ],
            'edges': [
                {'from': 'task1', 'to': 'task2'},
                {'from': 'task2', 'to': 'task3'},
                {'from': 'task3', 'to': 'eval-all'}
            ]
        },
        {
            'id': 'automl',
            'name': 'AutoML Pipeline',
            'description': 'Automatically search for best architecture and hyperparameters',
            'category': 'optimization',
            'nodes': [
                {
                    'id': 'search-space',
                    'name': 'Define Search Space',
                    'type': 'config',
                    'position': {'x': 100, 'y': 125}
                },
                {
                    'id': 'nas',
                    'name': 'Neural Architecture Search',
                    'type': 'search',
                    'position': {'x': 350, 'y': 50},
                    'job': {
                        'config': {
                            'method': 'darts',
                            'search_epochs': 50
                        }
                    }
                },
                {
                    'id': 'hpo',
                    'name': 'Hyperparameter Optimization',
                    'type': 'optimize',
                    'position': {'x': 350, 'y': 200},
                    'job': {
                        'config': {
                            'method': 'optuna',
                            'n_trials': 100
                        }
                    }
                },
                {
                    'id': 'best-model',
                    'name': 'Train Best Model',
                    'type': 'train',
                    'position': {'x': 600, 'y': 125}
                }
            ],
            'edges': [
                {'from': 'search-space', 'to': 'nas'},
                {'from': 'search-space', 'to': 'hpo'},
                {'from': 'nas', 'to': 'best-model'},
                {'from': 'hpo', 'to': 'best-model'}
            ]
        }
    ]

    return jsonify({'templates': templates})


@app.route('/api/schedules', methods=['GET','POST'])
def schedules_root():
    if request.method == 'POST':
        data = request.json or {}
        scheds = _load_json(SCHEDULES_PATH, [])
        data['id'] = data.get('id') or str(uuid.uuid4())
        data['created'] = datetime.now().isoformat()
        scheds.append(data)
        _save_json(SCHEDULES_PATH, scheds)
        return jsonify({'status': 'ok', 'id': data['id']}), 201
    return jsonify({'items': _load_json(SCHEDULES_PATH, [])})


# ---------------- Auth & Users ----------------

def _hash_password(password: str) -> str:
    """Simple password hashing (use bcrypt/argon2 in production)"""
    return hashlib.sha256(password.encode()).hexdigest()

def _generate_token() -> str:
    """Generate a secure random token"""
    return secrets.token_urlsafe(32)

def _get_current_user() -> Optional[Dict[str, Any]]:
    """Get current user from session or API token"""
    # Check for API token in Authorization header
    auth_header = request.headers.get('Authorization', '')
    if auth_header.startswith('Bearer '):
        token = auth_header[7:]
        user_id = _api_tokens.get(token)
        if user_id:
            users = _load_json(USERS_PATH, {})
            return users.get(user_id)

    # Check for session token
    session_token = request.headers.get('X-Session-Token') or request.cookies.get('session_token')
    if session_token and session_token in _sessions:
        session = _sessions[session_token]
        if datetime.fromisoformat(session['expires']) > datetime.now():
            users = _load_json(USERS_PATH, {})
            return users.get(session['user_id'])
        else:
            # Session expired
            del _sessions[session_token]

    return None

@app.route('/api/auth/register', methods=['POST'])
def auth_register():
    """Register a new user"""
    data = request.json or {}
    email = (data.get('email') or '').strip().lower()
    password = data.get('password', '')
    name = (data.get('name') or '').strip()

    if not email or not password:
        return jsonify({'error': 'Email and password required'}), 400

    if len(password) < 6:
        return jsonify({'error': 'Password must be at least 6 characters'}), 400

    users = _load_json(USERS_PATH, {})

    # Check if email already exists
    if any(u.get('email') == email for u in users.values()):
        return jsonify({'error': 'Email already registered'}), 400

    user_id = str(uuid.uuid4())
    user = {
        'id': user_id,
        'email': email,
        'name': name or email.split('@')[0],
        'password_hash': _hash_password(password),
        'created': datetime.now().isoformat(),
        'role': 'user',  # 'user', 'admin'
        'avatar': None,
        'preferences': {
            'theme': 'dark',
            'notifications': {'email': True, 'web': True}
        },
        'teams': []
    }

    users[user_id] = user
    _save_json(USERS_PATH, users)

    # Don't return password hash
    user_response = {k: v for k, v in user.items() if k != 'password_hash'}

    return jsonify(user_response), 201

@app.route('/api/auth/login', methods=['POST'])
def auth_login():
    """Login user and create session"""
    data = request.json or {}
    email = (data.get('email') or '').strip().lower()
    password = data.get('password', '')

    if not email or not password:
        return jsonify({'error': 'Email and password required'}), 400

    users = _load_json(USERS_PATH, {})
    user = next((u for u in users.values() if u.get('email') == email), None)

    if not user or user.get('password_hash') != _hash_password(password):
        return jsonify({'error': 'Invalid email or password'}), 401

    # Create session
    session_token = _generate_token()
    _sessions[session_token] = {
        'user_id': user['id'],
        'created': datetime.now().isoformat(),
        'expires': (datetime.now() + timedelta(days=7)).isoformat()
    }

    user_response = {k: v for k, v in user.items() if k != 'password_hash'}

    return jsonify({
        'user': user_response,
        'session_token': session_token,
        'expires': _sessions[session_token]['expires']
    })

@app.route('/api/auth/logout', methods=['POST'])
def auth_logout():
    """Logout user and destroy session"""
    session_token = request.headers.get('X-Session-Token') or request.cookies.get('session_token')

    if session_token and session_token in _sessions:
        del _sessions[session_token]

    return jsonify({'status': 'ok'})

@app.route('/api/user/profile', methods=['GET', 'PUT'])
def user_profile():
    """Get or update user profile"""
    user = _get_current_user()
    if not user:
        return jsonify({'error': 'Unauthorized'}), 401

    if request.method == 'PUT':
        data = request.json or {}
        users = _load_json(USERS_PATH, {})

        # Update allowed fields
        for field in ['name', 'avatar', 'preferences']:
            if field in data:
                users[user['id']][field] = data[field]

        users[user['id']]['updated'] = datetime.now().isoformat()
        _save_json(USERS_PATH, users)

        user = users[user['id']]

    user_response = {k: v for k, v in user.items() if k != 'password_hash'}
    return jsonify(user_response)

@app.route('/api/user/avatar', methods=['POST'])
def user_avatar():
    """Upload user avatar"""
    user = _get_current_user()
    if not user:
        return jsonify({'error': 'Unauthorized'}), 401

    # In production, handle actual file upload
    # For now, accept base64 data URL
    data = request.json or {}
    avatar_data = data.get('avatar')

    users = _load_json(USERS_PATH, {})
    users[user['id']]['avatar'] = avatar_data
    users[user['id']]['updated'] = datetime.now().isoformat()
    _save_json(USERS_PATH, users)

    return jsonify({'status': 'ok', 'avatar': avatar_data})

@app.route('/api/user/tokens', methods=['GET', 'POST'])
def user_tokens():
    """Manage API tokens"""
    user = _get_current_user()
    if not user:
        return jsonify({'error': 'Unauthorized'}), 401

    users = _load_json(USERS_PATH, {})

    if request.method == 'POST':
        # Create new API token
        token = _generate_token()
        _api_tokens[token] = user['id']

        if 'api_tokens' not in users[user['id']]:
            users[user['id']]['api_tokens'] = []

        token_record = {
            'id': str(uuid.uuid4()),
            'token': token,
            'created': datetime.now().isoformat(),
            'name': (request.json or {}).get('name', 'API Token')
        }

        users[user['id']]['api_tokens'].append(token_record)
        _save_json(USERS_PATH, users)

        return jsonify(token_record), 201

    # GET - list tokens (without showing full token)
    tokens = users[user['id']].get('api_tokens', [])
    tokens_response = [
        {
            'id': t['id'],
            'name': t['name'],
            'created': t['created'],
            'token_preview': t['token'][:8] + '...' if 'token' in t else None
        }
        for t in tokens
    ]

    return jsonify({'tokens': tokens_response})

@app.route('/api/user/tokens/<token_id>', methods=['DELETE'])
def delete_user_token(token_id):
    """Delete an API token"""
    user = _get_current_user()
    if not user:
        return jsonify({'error': 'Unauthorized'}), 401

    users = _load_json(USERS_PATH, {})
    tokens = users[user['id']].get('api_tokens', [])

    token_to_delete = next((t for t in tokens if t['id'] == token_id), None)
    if not token_to_delete:
        return jsonify({'error': 'Token not found'}), 404

    # Remove from memory
    if token_to_delete.get('token') in _api_tokens:
        del _api_tokens[token_to_delete['token']]

    # Remove from user record
    users[user['id']]['api_tokens'] = [t for t in tokens if t['id'] != token_id]
    _save_json(USERS_PATH, users)

    return jsonify({'status': 'ok'})

@app.route('/api/user/preferences', methods=['GET', 'PUT'])
def user_preferences():
    """Get or update user preferences"""
    user = _get_current_user()
    if not user:
        return jsonify({'error': 'Unauthorized'}), 401

    users = _load_json(USERS_PATH, {})

    if request.method == 'PUT':
        data = request.json or {}
        users[user['id']]['preferences'] = data
        users[user['id']]['updated'] = datetime.now().isoformat()
        _save_json(USERS_PATH, users)

    return jsonify(users[user['id']].get('preferences', {}))

@app.route('/api/user/settings', methods=['GET', 'PUT'])
def user_settings():
    """Get or update user settings (includes API keys, preferences, etc.)"""
    user = _get_current_user()
    if not user:
        # Return default settings for unauthenticated users
        return jsonify({
            'name': '',
            'email': '',
            'organization': '',
            'default_framework': 'pytorch',
            'auto_save_interval': 300,
            'notification_enabled': True,
            'theme': 'dark'
        })

    users = _load_json(USERS_PATH, {})
    user_id = user['id']

    if request.method == 'PUT':
        data = request.json or {}
        # Store settings
        if user_id not in users:
            users[user_id] = {'id': user_id, 'created': datetime.now().isoformat()}

        users[user_id]['settings'] = data
        users[user_id]['updated'] = datetime.now().isoformat()
        _save_json(USERS_PATH, users)

    return jsonify(users.get(user_id, {}).get('settings', {
        'name': '',
        'email': '',
        'organization': '',
        'default_framework': 'pytorch',
        'auto_save_interval': 300,
        'notification_enabled': True,
        'theme': 'dark'
    }))


@app.route('/api/user/dashboard', methods=['GET'])
def user_dashboard():
    """Get user dashboard statistics"""
    try:
        # Count datasets
        datasets_count = 0
        if os.path.exists(DATASETS_DIR):
            datasets_count = len([d for d in os.listdir(DATASETS_DIR)
                                if os.path.isdir(os.path.join(DATASETS_DIR, d))])

        # Count models
        models_count = 0
        if os.path.exists(MODELS_DIR):
            models_count = len([m for m in os.listdir(MODELS_DIR)
                              if os.path.isdir(os.path.join(MODELS_DIR, m))])

        # Recent jobs (last 10)
        recent_jobs = sorted(
            jobs.values(),
            key=lambda j: j.get('created', ''),
            reverse=True
        )[:10]

        # Job statistics
        job_stats = {
            'total': len(jobs),
            'running': len([j for j in jobs.values() if j['status'] == 'running']),
            'completed': len([j for j in jobs.values() if j['status'] == 'completed']),
            'failed': len([j for j in jobs.values() if j['status'] == 'failed']),
            'queued': len([j for j in jobs.values() if j['status'] == 'queued']),
            'cancelled': len([j for j in jobs.values() if j['status'] == 'cancelled']),
        }

        # Get system info for environment summary
        system_info = _system_info_snapshot()

        # Environment summary
        env_summary = {
            'python_version': os.popen('python3 --version 2>&1').read().strip(),
            'pytorch_version': 'N/A',
            'cuda_available': False,
            'cuda_version': 'N/A',
            'gpu_count': len(system_info.get('gpus', [])),
            'cpu_count': system_info.get('cpu', {}).get('count', 0),
            'memory_total_gb': round(system_info.get('memory', {}).get('total_mib', 0) / 1024, 1),
        }

        try:
            import torch
            env_summary['pytorch_version'] = torch.__version__
            env_summary['cuda_available'] = torch.cuda.is_available()
            if torch.cuda.is_available():
                env_summary['cuda_version'] = torch.version.cuda or 'N/A'
        except Exception:
            pass

        return jsonify({
            'datasets_count': datasets_count,
            'models_count': models_count,
            'job_stats': job_stats,
            'recent_jobs': recent_jobs,
            'environment': env_summary,
            'system': {
                'cpu': system_info.get('cpu', {}),
                'memory': system_info.get('memory', {}),
                'gpus': system_info.get('gpus', []),
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/config/persistent', methods=['GET', 'PUT'])
def persistent_config():
    """Manage persistent configuration at ~/.spark_trainer/config.json"""
    config_dir = os.path.expanduser('~/.spark_trainer')
    config_path = os.path.join(config_dir, 'config.json')

    if request.method == 'GET':
        # Load persistent config
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    return jsonify(json.load(f))
            except Exception as e:
                return jsonify({'error': f'Failed to load config: {str(e)}'}), 500
        else:
            # Return default config
            return jsonify({
                'version': '1.0',
                'user': {
                    'name': '',
                    'email': '',
                    'organization': ''
                },
                'paths': {
                    'datasets': DATASETS_DIR,
                    'models': MODELS_DIR,
                    'logs': LOGS_DIR
                },
                'defaults': {
                    'framework': 'pytorch',
                    'precision': 'fp16',
                    'batch_size': 8
                }
            })

    elif request.method == 'PUT':
        # Save persistent config
        data = request.json or {}
        try:
            os.makedirs(config_dir, exist_ok=True)
            with open(config_path, 'w') as f:
                json.dump(data, f, indent=2)
            return jsonify({'status': 'ok', 'path': config_path})
        except Exception as e:
            return jsonify({'error': f'Failed to save config: {str(e)}'}), 500


@app.route('/api/config/profiles', methods=['GET', 'POST'])
def config_profiles():
    """Manage reusable configuration profiles (profiles.yaml)"""
    profiles_path = os.path.join(BASE_DIR, 'profiles.yaml')

    if request.method == 'GET':
        # Load profiles
        if os.path.exists(profiles_path):
            try:
                import yaml
                with open(profiles_path, 'r') as f:
                    profiles = yaml.safe_load(f) or {}
                return jsonify({'profiles': profiles})
            except Exception as e:
                return jsonify({'error': f'Failed to load profiles: {str(e)}'}), 500
        else:
            # Return default profiles
            default_profiles = {
                'default': {
                    'framework': 'pytorch',
                    'precision': 'fp16',
                    'batch_size': 8,
                    'learning_rate': 1e-5,
                    'optimizer': 'adamw'
                },
                'vision-language': {
                    'framework': 'huggingface',
                    'precision': 'bf16',
                    'batch_size': 4,
                    'learning_rate': 5e-6,
                    'optimizer': 'adamw',
                    'model_type': 'vision_language'
                },
                'diffusion': {
                    'framework': 'pytorch',
                    'precision': 'fp16',
                    'batch_size': 1,
                    'learning_rate': 1e-4,
                    'optimizer': 'adam',
                    'model_type': 'diffusion'
                },
                'llm-finetuning': {
                    'framework': 'huggingface',
                    'precision': 'bf16',
                    'batch_size': 2,
                    'learning_rate': 2e-5,
                    'optimizer': 'adamw',
                    'use_lora': True,
                    'lora_r': 16,
                    'lora_alpha': 32
                }
            }
            return jsonify({'profiles': default_profiles})

    elif request.method == 'POST':
        # Save profiles
        data = request.json or {}
        profiles = data.get('profiles', {})
        try:
            import yaml
            with open(profiles_path, 'w') as f:
                yaml.dump(profiles, f, default_flow_style=False)
            return jsonify({'status': 'ok', 'path': profiles_path})
        except Exception as e:
            return jsonify({'error': f'Failed to save profiles: {str(e)}'}), 500


@app.route('/api/models/templates', methods=['GET'])
def model_templates():
    """Get available model templates from templates.yaml"""
    templates_path = os.path.join(BASE_DIR, 'src', 'spark_trainer', 'models', 'templates.yaml')

    try:
        import yaml
    except ImportError:
        return jsonify({
            'error': 'PyYAML not installed',
            'templates': {},
            'categories': {}
        }), 500

    try:
        if not os.path.exists(templates_path):
            return jsonify({
                'error': f'Templates file not found at {templates_path}',
                'templates': {},
                'categories': {}
            }), 404

        with open(templates_path, 'r') as f:
            templates_data = yaml.safe_load(f)

        if templates_data is None:
            return jsonify({
                'error': 'Templates file is empty',
                'templates': {},
                'categories': {}
            }), 500

        # Ensure the data has the expected structure
        if not isinstance(templates_data, dict):
            return jsonify({
                'error': 'Invalid templates file format',
                'templates': {},
                'categories': {}
            }), 500

        # Provide defaults if keys are missing
        if 'templates' not in templates_data:
            templates_data['templates'] = {}
        if 'categories' not in templates_data:
            templates_data['categories'] = {}

        return jsonify(templates_data)
    except FileNotFoundError:
        return jsonify({
            'error': 'Templates file not found',
            'templates': {},
            'categories': {}
        }), 404
    except yaml.YAMLError as e:
        return jsonify({
            'error': f'YAML parsing error: {str(e)}',
            'templates': {},
            'categories': {}
        }), 500
    except Exception as e:
        return jsonify({
            'error': f'Failed to load templates: {str(e)}',
            'templates': {},
            'categories': {}
        }), 500


@app.route('/api/huggingface/download-model', methods=['POST'])
def huggingface_download_model():
    """Download a model from HuggingFace Hub"""
    data = request.json or {}
    model_id = data.get('model_id')
    token = data.get('token')

    if not model_id:
        return jsonify({'error': 'model_id required'}), 400

    try:
        # Create a background job to download the model
        job_id = str(uuid.uuid4())

        def download_model_task():
            try:
                from transformers import AutoModel, AutoTokenizer
                import os

                # Set HF token if provided
                if token:
                    os.environ['HF_TOKEN'] = token

                # Download to models directory
                model_path = os.path.join(MODELS_DIR, model_id.replace('/', '_'))
                os.makedirs(model_path, exist_ok=True)

                # Download model and tokenizer
                AutoModel.from_pretrained(model_id, cache_dir=model_path)
                AutoTokenizer.from_pretrained(model_id, cache_dir=model_path)

                return {'status': 'completed', 'path': model_path}
            except Exception as e:
                return {'status': 'failed', 'error': str(e)}

        # Start download in background
        from threading import Thread
        Thread(target=download_model_task, daemon=True).start()

        return jsonify({
            'status': 'ok',
            'job_id': job_id,
            'message': f'Downloading {model_id} in background'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/huggingface/download-dataset', methods=['POST'])
def huggingface_download_dataset():
    """Download a dataset from HuggingFace Hub"""
    data = request.json or {}
    dataset_id = data.get('dataset_id')
    token = data.get('token')

    if not dataset_id:
        return jsonify({'error': 'dataset_id required'}), 400

    try:
        # Create a background job to download the dataset
        job_id = str(uuid.uuid4())

        def download_dataset_task():
            try:
                from datasets import load_dataset
                import os

                # Set HF token if provided
                if token:
                    os.environ['HF_TOKEN'] = token

                # Download to datasets directory
                dataset_name = dataset_id.replace('/', '_')
                dataset_path = os.path.join(DATASETS_DIR, dataset_name)
                os.makedirs(dataset_path, exist_ok=True)

                # Download dataset
                ds = load_dataset(dataset_id, cache_dir=dataset_path)

                # Save to disk
                ds.save_to_disk(os.path.join(dataset_path, 'v1'))

                return {'status': 'completed', 'path': dataset_path}
            except Exception as e:
                return {'status': 'failed', 'error': str(e)}

        # Start download in background
        from threading import Thread
        Thread(target=download_dataset_task, daemon=True).start()

        return jsonify({
            'status': 'ok',
            'job_id': job_id,
            'message': f'Downloading {dataset_id} in background'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ---------------- Teams ----------------

@app.route('/api/teams', methods=['GET', 'POST'])
def teams_root():
    """List or create teams"""
    user = _get_current_user()
    if not user:
        return jsonify({'error': 'Unauthorized'}), 401

    teams = _load_json(TEAMS_PATH, {})

    if request.method == 'POST':
        data = request.json or {}
        name = (data.get('name') or '').strip()

        if not name:
            return jsonify({'error': 'Team name required'}), 400

        team_id = str(uuid.uuid4())
        team = {
            'id': team_id,
            'name': name,
            'description': data.get('description', ''),
            'created': datetime.now().isoformat(),
            'owner_id': user['id'],
            'members': [
                {
                    'user_id': user['id'],
                    'role': 'owner',  # 'owner', 'admin', 'member'
                    'joined': datetime.now().isoformat()
                }
            ],
            'quota': {
                'max_gpus': data.get('max_gpus', 4),
                'max_storage_gb': data.get('max_storage_gb', 1000),
                'max_jobs_per_month': data.get('max_jobs_per_month', 100)
            },
            'usage': {
                'current_gpus': 0,
                'storage_gb': 0,
                'jobs_this_month': 0
            }
        }

        teams[team_id] = team
        _save_json(TEAMS_PATH, teams)

        # Add team to user
        users = _load_json(USERS_PATH, {})
        if 'teams' not in users[user['id']]:
            users[user['id']]['teams'] = []
        users[user['id']]['teams'].append(team_id)
        _save_json(USERS_PATH, users)

        return jsonify(team), 201

    # GET - list teams where user is a member
    user_teams = [
        t for t in teams.values()
        if any(m['user_id'] == user['id'] for m in t.get('members', []))
    ]

    return jsonify({'teams': user_teams})

@app.route('/api/teams/<team_id>', methods=['GET', 'PUT', 'DELETE'])
def team_detail(team_id):
    """Get, update, or delete a team"""
    user = _get_current_user()
    if not user:
        return jsonify({'error': 'Unauthorized'}), 401

    teams = _load_json(TEAMS_PATH, {})

    if team_id not in teams:
        return jsonify({'error': 'Team not found'}), 404

    team = teams[team_id]

    # Check if user is a member
    is_member = any(m['user_id'] == user['id'] for m in team.get('members', []))
    is_owner = team.get('owner_id') == user['id']

    if not is_member:
        return jsonify({'error': 'Not a team member'}), 403

    if request.method == 'DELETE':
        if not is_owner:
            return jsonify({'error': 'Only owner can delete team'}), 403

        del teams[team_id]
        _save_json(TEAMS_PATH, teams)

        # Remove from all users
        users = _load_json(USERS_PATH, {})
        for user_data in users.values():
            if 'teams' in user_data and team_id in user_data['teams']:
                user_data['teams'].remove(team_id)
        _save_json(USERS_PATH, users)

        return jsonify({'status': 'ok', 'deleted': team_id})

    if request.method == 'PUT':
        if not is_owner:
            return jsonify({'error': 'Only owner can update team'}), 403

        data = request.json or {}
        for field in ['name', 'description', 'quota']:
            if field in data:
                team[field] = data[field]

        team['updated'] = datetime.now().isoformat()
        teams[team_id] = team
        _save_json(TEAMS_PATH, teams)

    return jsonify(team)

@app.route('/api/teams/<team_id>/members', methods=['POST'])
def team_add_member(team_id):
    """Add a member to a team"""
    user = _get_current_user()
    if not user:
        return jsonify({'error': 'Unauthorized'}), 401

    teams = _load_json(TEAMS_PATH, {})

    if team_id not in teams:
        return jsonify({'error': 'Team not found'}), 404

    team = teams[team_id]

    # Check if user is owner or admin
    user_member = next((m for m in team.get('members', []) if m['user_id'] == user['id']), None)
    if not user_member or user_member['role'] not in ['owner', 'admin']:
        return jsonify({'error': 'Insufficient permissions'}), 403

    data = request.json or {}
    new_user_email = (data.get('email') or '').strip().lower()
    role = data.get('role', 'member')

    if not new_user_email:
        return jsonify({'error': 'User email required'}), 400

    # Find user by email
    users = _load_json(USERS_PATH, {})
    new_user = next((u for u in users.values() if u.get('email') == new_user_email), None)

    if not new_user:
        return jsonify({'error': 'User not found'}), 404

    # Check if already a member
    if any(m['user_id'] == new_user['id'] for m in team.get('members', [])):
        return jsonify({'error': 'User already a member'}), 400

    member = {
        'user_id': new_user['id'],
        'role': role,
        'joined': datetime.now().isoformat()
    }

    team['members'].append(member)
    teams[team_id] = team
    _save_json(TEAMS_PATH, teams)

    # Add team to user
    if 'teams' not in users[new_user['id']]:
        users[new_user['id']]['teams'] = []
    users[new_user['id']]['teams'].append(team_id)
    _save_json(USERS_PATH, users)

    return jsonify(member), 201

@app.route('/api/teams/<team_id>/members/<member_user_id>', methods=['DELETE'])
def team_remove_member(team_id, member_user_id):
    """Remove a member from a team"""
    user = _get_current_user()
    if not user:
        return jsonify({'error': 'Unauthorized'}), 401

    teams = _load_json(TEAMS_PATH, {})

    if team_id not in teams:
        return jsonify({'error': 'Team not found'}), 404

    team = teams[team_id]

    # Owner can't be removed
    if member_user_id == team.get('owner_id'):
        return jsonify({'error': 'Cannot remove team owner'}), 400

    # Check permissions
    user_member = next((m for m in team.get('members', []) if m['user_id'] == user['id']), None)
    if not user_member:
        return jsonify({'error': 'Not a team member'}), 403

    # Users can remove themselves, or admins/owners can remove others
    if member_user_id != user['id'] and user_member['role'] not in ['owner', 'admin']:
        return jsonify({'error': 'Insufficient permissions'}), 403

    # Remove member
    team['members'] = [m for m in team.get('members', []) if m['user_id'] != member_user_id]
    teams[team_id] = team
    _save_json(TEAMS_PATH, teams)

    # Remove team from user
    users = _load_json(USERS_PATH, {})
    if member_user_id in users and 'teams' in users[member_user_id]:
        users[member_user_id]['teams'] = [t for t in users[member_user_id]['teams'] if t != team_id]
        _save_json(USERS_PATH, users)

    return jsonify({'status': 'ok'})

@app.route('/api/teams/<team_id>/members/<member_user_id>/role', methods=['PUT'])
def team_update_member_role(team_id, member_user_id):
    """Update a member's role"""
    user = _get_current_user()
    if not user:
        return jsonify({'error': 'Unauthorized'}), 401

    teams = _load_json(TEAMS_PATH, {})

    if team_id not in teams:
        return jsonify({'error': 'Team not found'}), 404

    team = teams[team_id]

    # Only owner can change roles
    if team.get('owner_id') != user['id']:
        return jsonify({'error': 'Only owner can change member roles'}), 403

    # Can't change owner's role
    if member_user_id == team.get('owner_id'):
        return jsonify({'error': 'Cannot change owner role'}), 400

    data = request.json or {}
    new_role = data.get('role', 'member')

    if new_role not in ['admin', 'member']:
        return jsonify({'error': 'Invalid role'}), 400

    # Update role
    for member in team.get('members', []):
        if member['user_id'] == member_user_id:
            member['role'] = new_role
            break
    else:
        return jsonify({'error': 'Member not found'}), 404

    teams[team_id] = team
    _save_json(TEAMS_PATH, teams)

    return jsonify({'status': 'ok', 'role': new_role})

@app.route('/api/teams/<team_id>/quota', methods=['GET', 'PUT'])
def team_quota(team_id):
    """Get or update team quota"""
    user = _get_current_user()
    if not user:
        return jsonify({'error': 'Unauthorized'}), 401

    teams = _load_json(TEAMS_PATH, {})

    if team_id not in teams:
        return jsonify({'error': 'Team not found'}), 404

    team = teams[team_id]

    # Check if user is a member
    is_member = any(m['user_id'] == user['id'] for m in team.get('members', []))
    if not is_member:
        return jsonify({'error': 'Not a team member'}), 403

    if request.method == 'PUT':
        # Only owner can update quota
        if team.get('owner_id') != user['id']:
            return jsonify({'error': 'Only owner can update quota'}), 403

        data = request.json or {}
        team['quota'] = data
        teams[team_id] = team
        _save_json(TEAMS_PATH, teams)

    return jsonify({
        'quota': team.get('quota', {}),
        'usage': team.get('usage', {})
    })


# ---------------- Billing ----------------

def _track_job_cost(job_id: str):
    """Track cost for a completed job"""
    if job_id not in jobs:
        return

    job = jobs[job_id]

    # Simple cost calculation (customize based on your pricing)
    gpu_type = job.get('gpu', 'unknown')
    duration_seconds = 0

    if job.get('started') and job.get('completed'):
        try:
            started = datetime.fromisoformat(job['started'])
            completed = datetime.fromisoformat(job['completed'])
            duration_seconds = (completed - started).total_seconds()
        except Exception:
            pass

    # Example pricing ($/hour)
    gpu_pricing = {
        'A100': 3.00,
        'V100': 2.00,
        'T4': 0.50,
        'unknown': 1.00
    }

    hourly_rate = gpu_pricing.get(gpu_type, 1.00)
    cost = (duration_seconds / 3600) * hourly_rate

    billing = _load_json(BILLING_PATH, {'records': []})

    record = {
        'id': str(uuid.uuid4()),
        'job_id': job_id,
        'user_id': job.get('user_id', 'unknown'),
        'team_id': job.get('team_id'),
        'timestamp': datetime.now().isoformat(),
        'gpu_type': gpu_type,
        'duration_seconds': duration_seconds,
        'cost': round(cost, 4),
        'description': f"Job: {job.get('name', job_id)}"
    }

    billing['records'].append(record)
    _save_json(BILLING_PATH, billing)

@app.route('/api/billing/usage', methods=['GET'])
def billing_usage():
    """Get billing usage summary"""
    user = _get_current_user()
    if not user:
        return jsonify({'error': 'Unauthorized'}), 401

    billing = _load_json(BILLING_PATH, {'records': []})

    # Filter by user
    user_records = [r for r in billing['records'] if r.get('user_id') == user['id']]

    # Calculate totals
    total_cost = sum(r.get('cost', 0) for r in user_records)
    total_hours = sum(r.get('duration_seconds', 0) for r in user_records) / 3600

    # Group by month
    monthly_usage = {}
    for record in user_records:
        try:
            month_key = datetime.fromisoformat(record['timestamp']).strftime('%Y-%m')
            if month_key not in monthly_usage:
                monthly_usage[month_key] = {'cost': 0, 'hours': 0, 'jobs': 0}
            monthly_usage[month_key]['cost'] += record.get('cost', 0)
            monthly_usage[month_key]['hours'] += record.get('duration_seconds', 0) / 3600
            monthly_usage[month_key]['jobs'] += 1
        except Exception:
            pass

    return jsonify({
        'total_cost': round(total_cost, 2),
        'total_hours': round(total_hours, 2),
        'total_jobs': len(user_records),
        'monthly_usage': monthly_usage,
        'current_month': datetime.now().strftime('%Y-%m')
    })

@app.route('/api/billing/breakdown', methods=['GET'])
def billing_breakdown():
    """Get detailed billing breakdown"""
    user = _get_current_user()
    if not user:
        return jsonify({'error': 'Unauthorized'}), 401

    billing = _load_json(BILLING_PATH, {'records': []})

    # Filter by user
    user_records = [r for r in billing['records'] if r.get('user_id') == user['id']]

    # Group by GPU type
    by_gpu = {}
    for record in user_records:
        gpu = record.get('gpu_type', 'unknown')
        if gpu not in by_gpu:
            by_gpu[gpu] = {'cost': 0, 'hours': 0, 'jobs': 0}
        by_gpu[gpu]['cost'] += record.get('cost', 0)
        by_gpu[gpu]['hours'] += record.get('duration_seconds', 0) / 3600
        by_gpu[gpu]['jobs'] += 1

    # Get recent records (last 30 days)
    thirty_days_ago = datetime.now() - timedelta(days=30)
    recent_records = [
        r for r in user_records
        if datetime.fromisoformat(r['timestamp']) > thirty_days_ago
    ]

    return jsonify({
        'by_gpu_type': by_gpu,
        'recent_records': recent_records[-100:],  # Last 100 records
        'total_records': len(user_records)
    })

@app.route('/api/billing/invoices', methods=['GET'])
def billing_invoices():
    """Get invoices"""
    user = _get_current_user()
    if not user:
        return jsonify({'error': 'Unauthorized'}), 401

    # Generate monthly invoices from billing records
    billing = _load_json(BILLING_PATH, {'records': []})
    user_records = [r for r in billing['records'] if r.get('user_id') == user['id']]

    invoices = {}
    for record in user_records:
        try:
            month_key = datetime.fromisoformat(record['timestamp']).strftime('%Y-%m')
            if month_key not in invoices:
                invoices[month_key] = {
                    'id': f"INV-{month_key}",
                    'month': month_key,
                    'total': 0,
                    'items': 0,
                    'status': 'paid'
                }
            invoices[month_key]['total'] += record.get('cost', 0)
            invoices[month_key]['items'] += 1
        except Exception:
            pass

    invoice_list = sorted(invoices.values(), key=lambda x: x['month'], reverse=True)

    return jsonify({'invoices': invoice_list})

@app.route('/api/billing/invoices/<invoice_id>', methods=['GET'])
def billing_invoice_detail(invoice_id):
    """Get invoice details"""
    user = _get_current_user()
    if not user:
        return jsonify({'error': 'Unauthorized'}), 401

    # Extract month from invoice ID (format: INV-YYYY-MM)
    if not invoice_id.startswith('INV-'):
        return jsonify({'error': 'Invalid invoice ID'}), 400

    month_key = invoice_id[4:]  # Remove 'INV-' prefix

    billing = _load_json(BILLING_PATH, {'records': []})
    user_records = [
        r for r in billing['records']
        if r.get('user_id') == user['id'] and
        datetime.fromisoformat(r['timestamp']).strftime('%Y-%m') == month_key
    ]

    if not user_records:
        return jsonify({'error': 'Invoice not found'}), 404

    total = sum(r.get('cost', 0) for r in user_records)

    return jsonify({
        'id': invoice_id,
        'month': month_key,
        'total': round(total, 2),
        'records': user_records,
        'status': 'paid'
    })

@app.route('/api/billing/alerts', methods=['POST'])
def billing_alerts():
    """Set billing alerts"""
    user = _get_current_user()
    if not user:
        return jsonify({'error': 'Unauthorized'}), 401

    data = request.json or {}

    # Store alert preferences in user preferences
    users = _load_json(USERS_PATH, {})
    if 'preferences' not in users[user['id']]:
        users[user['id']]['preferences'] = {}

    users[user['id']]['preferences']['billing_alerts'] = {
        'monthly_limit': data.get('monthly_limit'),
        'daily_limit': data.get('daily_limit'),
        'enabled': data.get('enabled', True)
    }

    _save_json(USERS_PATH, users)

    return jsonify({'status': 'ok', 'alerts': users[user['id']]['preferences']['billing_alerts']})


def _detect_partitions() -> Dict[str, Any]:
    """Detect GPUs and MIG instances.

    Returns structure:
    {
      'gpus': [
         { 'index': int, 'name': str, 'uuid': str, 'mig_mode': str, 'instances': [
             { 'uuid': str, 'profile': str, 'device_id': int }
         ]}
      ]
    }
    """
    result = {'gpus': []}
    try:
        # Basic GPU info including MIG mode
        q = subprocess.check_output([
            'nvidia-smi',
            '--query-gpu=index,uuid,name,mig.mode.current',
            '--format=csv,noheader,nounits'
        ]).decode()
        gpu_list = []
        for line in q.strip().split('\n'):
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 4:
                idx_s, uuid, name, mig_mode = parts[:4]
                try:
                    idx = int(idx_s)
                except Exception:
                    idx = len(gpu_list)
                gpu_list.append({
                    'index': idx,
                    'uuid': uuid,
                    'name': name,
                    'mig_mode': mig_mode,
                    'instances': []
                })
        # Parse nvidia-smi -L for MIG instances
        try:
            outL = subprocess.check_output(['nvidia-smi', '-L']).decode()
            current_gpu_idx = None
            for line in outL.splitlines():
                line = line.rstrip()
                if line.startswith('GPU '):
                    # Example: GPU 0: A100 (UUID: GPU-xxxx)
                    try:
                        idx = int(line.split(':', 1)[0].split()[1])
                    except Exception:
                        idx = None
                    current_gpu_idx = idx
                elif line.strip().startswith('MIG '):
                    # Example: MIG 1g.5gb Device 0: (UUID: MIG-GPU-.../1/0)
                    prof_part = line.strip().split('Device')[0].strip()
                    profile = prof_part.replace('MIG', '').strip()
                    # Extract device id
                    try:
                        after_dev = line.split('Device', 1)[1]
                        dev_id = int(after_dev.split(':', 1)[0].strip())
                    except Exception:
                        dev_id = None
                    # Extract UUID inside parentheses
                    mig_uuid = None
                    if 'UUID:' in line:
                        try:
                            mig_uuid = line.split('UUID:', 1)[1].split(')')[0].strip()
                        except Exception:
                            pass
                    if current_gpu_idx is not None and mig_uuid:
                        for g in gpu_list:
                            if g['index'] == current_gpu_idx:
                                g['instances'].append({
                                    'uuid': mig_uuid,
                                    'profile': profile,
                                    'device_id': dev_id
                                })
                                break
        except Exception:
            pass
        result['gpus'] = gpu_list
    except Exception:
        pass
    # Mark allocations from running jobs
    try:
        running = {j['id']: j for j in jobs.values() if j.get('status') == 'running'}
        for g in result.get('gpus', []):
            # GPU-level allocations
            g['allocated_by_jobs'] = [
                {'job_id': j['id'], 'job_name': j.get('name')}
                for j in running.values()
                if j.get('gpu', {}).get('type') == 'gpu' and (
                    j['gpu'].get('gpu_index') == g['index'] or j['gpu'].get('gpu_uuid') == g['uuid']
                )
            ]
            # MIG-level allocations
            for inst in g.get('instances', []):
                inst['allocated_by_jobs'] = [
                    {'job_id': j['id'], 'job_name': j.get('name')}
                    for j in running.values()
                    if j.get('gpu', {}).get('type') == 'mig' and j['gpu'].get('mig_uuid') == inst['uuid']
                ]
    except Exception:
        pass
    return result


@app.route('/api/gpu/partitions', methods=['GET'])
def gpu_partitions():
    return jsonify(_detect_partitions())

# Backward-compatible singular alias
@app.route('/api/gpu/partition', methods=['GET'])
def gpu_partition_alias():
    return jsonify(_detect_partitions())


@app.route('/api/gpu/partition/config', methods=['GET'])
def gpu_partition_config():
    """Expose current GPU/MIG topology and supported profiles. Admin flag indicates whether apply is permitted."""
    supported_profiles = [
        '1g.5gb', '2g.10gb', '3g.20gb', '4g.20gb', '7g.40gb',
        '1g.10gb', '2g.20gb', '3g.40gb', '7g.80gb'
    ]
    return jsonify({
        'admin_enabled': os.environ.get('ENABLE_MIG_ADMIN') == '1',
        'partitions': _detect_partitions(),
        'supported_profiles': supported_profiles,
    })


_MIG_PROFILE_MEM_GB = {
    '1g.5gb': 5,
    '2g.10gb': 10,
    '3g.20gb': 20,
    '4g.20gb': 20,
    '7g.40gb': 40,
    '1g.10gb': 10,
    '2g.20gb': 20,
    '3g.40gb': 40,
    '7g.80gb': 80,
}


@app.route('/api/gpu/partition/presets', methods=['GET'])
def gpu_partition_presets():
    presets = {
        '4x_small_jobs': {'label': '4x Small Jobs', 'config': {'1g.5gb': 4}},
        '2x_medium_infer': {'label': '2x Medium + Inference', 'config': {'3g.20gb': 2}},
        '1x_large_training': {'label': '1x Large Training', 'config': {'7g.40gb': 1}},
        'mixed_workload': {'label': 'Mixed Workload', 'config': {'3g.20gb': 1, '1g.5gb': 2}},
    }
    return jsonify({'presets': presets})


def _estimate_model_size_params(model_dir: str) -> Dict[str, Any]:
    # Best-effort estimation from config.json
    cfg_path = os.path.join(model_dir, 'config.json')
    params = None
    hidden_size = num_layers = vocab = None
    if os.path.exists(cfg_path):
        try:
            cfg = json.load(open(cfg_path))
            params = cfg.get('n_parameters') or cfg.get('parameter_count')
            hidden_size = cfg.get('hidden_size') or cfg.get('n_embd')
            num_layers = cfg.get('num_hidden_layers') or cfg.get('n_layer')
            vocab = cfg.get('vocab_size')
        except Exception:
            pass
    if params is None and hidden_size and num_layers and vocab:
        # crude transformer estimate: embeddings + MHA + MLP per layer
        emb = hidden_size * vocab
        per_layer = 12 * (hidden_size ** 2)  # very rough
        params = emb + per_layer * num_layers
    return {
        'parameters': int(params) if isinstance(params, (int, float)) else None,
        'hidden_size': hidden_size,
        'num_layers': num_layers,
        'vocab_size': vocab,
    }


def _estimate_memory_requirements(params: int | None, batch_size: int, seq_len: int, training: bool, precision: str = 'fp16') -> Dict[str, Any]:
    # simple heuristics
    bytes_per_param = 2 if precision in ('fp16', 'bf16') else 4
    model_mem = (params or 0) * bytes_per_param
    # optimizer states ~ 2-3x model params during training
    opt_mem = model_mem * (2.0 if training else 0.0)
    # activations ~ batch * seq * hidden_size * bytes (unknown hidden, approximate from params)
    act_mem = 0.0
    if params:
        # assume hidden_size ~ sqrt(params / (12 * layers)) -> fallback scale
        act_scale = 0.0005  # heuristic factor
        act_mem = batch_size * seq_len * bytes_per_param * act_scale * params
    total = model_mem + opt_mem + act_mem
    return {
        'model_bytes': int(model_mem),
        'optimizer_bytes': int(opt_mem),
        'activation_bytes': int(act_mem),
        'total_bytes': int(total),
    }


@app.route('/api/models/<model_id>/resource/estimate', methods=['GET'])
def model_resource_estimate(model_id):
    base = _model_dir(model_id)
    if not os.path.isdir(base):
        return jsonify({'error': 'Model not found'}), 404
    batch_size = int(request.args.get('batch_size') or 8)
    seq_len = int(request.args.get('seq_len') or 2048)
    training = (request.args.get('training') or '0') in ('1','true','yes')
    precision = request.args.get('precision') or 'fp16'
    est = _estimate_model_size_params(base)
    mem = _estimate_memory_requirements(est.get('parameters'), batch_size, seq_len, training, precision)
    return jsonify({'model': model_id, 'estimate': est, 'memory': mem})


@app.route('/api/gpu/partition/recommend', methods=['POST'])
def gpu_partition_recommend():
    """Recommend a MIG partition layout based on model size/use-case.

    JSON: { model_id?: string, params?: int, batch_size?: int, seq_len?: int, training?: bool }
    """
    data = request.json or {}
    params = data.get('params')
    if not params and data.get('model_id'):
        est = _estimate_model_size_params(_model_dir(_safe_name(data['model_id'])))
        params = est.get('parameters')
    batch_size = int(data.get('batch_size') or 8)
    seq_len = int(data.get('seq_len') or 2048)
    training = bool(data.get('training') or False)
    mem = _estimate_memory_requirements(params, batch_size, seq_len, training)
    need_gb = max(1, int(mem['total_bytes'] / (1024**3)))
    # Pick profiles with >= need_gb, limited by 4 instances.
    candidates = sorted(_MIG_PROFILE_MEM_GB.items(), key=lambda x: x[1])
    cfg: Dict[str, int] = {}
    for prof, gb in candidates:
        if gb >= need_gb:
            cfg[prof] = 1
            break
    if not cfg:
        # too large: suggest full GPU (no MIG)
        return jsonify({'config': {}, 'full_gpu': True, 'rationale': f'model needs ~{need_gb} GB; consider full GPU allocation'}), 200
    # Try to fill remaining capacity with smaller profiles if asked for parallelism
    parallel = int(data.get('parallel') or 1)
    total = sum(cfg.values())
    if parallel > 1:
        small = '1g.5gb'
        add = min(4-total, max(0, parallel-1))
        if add > 0:
            cfg[small] = cfg.get(small, 0) + add
    # Cap to 4
    while sum(cfg.values()) > 4:
        # remove smallest
        smallest = sorted(cfg.items(), key=lambda x: _MIG_PROFILE_MEM_GB.get(x[0], 0))[0][0]
        cfg[smallest] -= 1
        if cfg[smallest] <= 0:
            del cfg[smallest]
    return jsonify({'config': cfg, 'need_gb': need_gb, 'rationale': 'Heuristic based on parameter count and batch/seq settings'})


@app.route('/api/gpu/partition/apply', methods=['POST'])
def gpu_partition_apply():
    """Apply a MIG partitioning configuration. Requires ENABLE_MIG_ADMIN=1.

    Expected payload example:
    {
      "gpu_index": 0,
      "enable_mig": true,
      "config": { "1g.5gb": 2, "3g.20gb": 1 }
    }
    This implementation is a safe stub (dry-run) unless ENABLE_MIG_ADMIN=1 and the server is configured appropriately.
    """
    if os.environ.get('ENABLE_MIG_ADMIN') != '1':
        return jsonify({'error': 'Admin operations disabled'}), 403
    data = request.json or {}
    gpu_index = data.get('gpu_index')
    enable = bool(data.get('enable_mig', True))
    config = data.get('config') or {}
    # Enforce max 4 instances per GPU
    try:
        total_instances = sum(int(v or 0) for v in (config or {}).values())
    except Exception:
        return jsonify({'error': 'Invalid config: counts must be integers >= 0'}), 400
    if total_instances > 4:
        return jsonify({'error': 'Too many partitions requested', 'detail': 'Maximum 4 MIG instances per GPU are allowed', 'requested_total': total_instances, 'limit': 4}), 400
    # Support dry-run by default; execute only if MIG_APPLY_EXECUTE=1
    if os.environ.get('MIG_APPLY_EXECUTE') != '1':
        return jsonify({
            'status': 'accepted',
            'note': 'Dry-run: set MIG_APPLY_EXECUTE=1 to execute. No changes applied.',
            'requested': {
                'gpu_index': gpu_index,
                'enable_mig': enable,
                'config': config
            },
            'constraints': {'max_instances_per_gpu': 4, 'requested_total': total_instances}
        }), 202

    # Real execution (best-effort, requires privileges)
    import re
    def run(cmd):
        try:
            out = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
            return 0, out.decode()
        except subprocess.CalledProcessError as e:
            return e.returncode, e.output.decode() if e.output else ''

    # Enable/disable MIG mode
    rc, out = run(['nvidia-smi', '-i', str(gpu_index), '-mig', '1' if enable else '0'])
    steps = [{'cmd': f"nvidia-smi -i {gpu_index} -mig {'1' if enable else '0'}", 'rc': rc, 'out': out}]
    if rc != 0:
        return jsonify({'status': 'error', 'step': steps[-1], 'message': 'Failed to toggle MIG mode'}), 500

    created = []
    created_ci = []
    if enable and config:
        # Destroy existing GPU instances
        rc, out = run(['nvidia-smi', 'mig', '-dgi', '-i', str(gpu_index)])
        steps.append({'cmd': f'nvidia-smi mig -dgi -i {gpu_index}', 'rc': rc, 'out': out})
        # Map profile string to GIP ID
        rc, out = run(['nvidia-smi', 'mig', '-lgci', '-i', str(gpu_index)])
        steps.append({'cmd': f'nvidia-smi mig -lgci -i {gpu_index}', 'rc': rc, 'out': out})
        gip_map = {}
        try:
            for line in out.splitlines():
                m = re.search(r'(\d+g\.\d+gb).*?ID[:\s]+(\d+)', line, re.IGNORECASE)
                if m:
                    gip_map[m.group(1).lower()] = int(m.group(2))
        except Exception:
            pass

        created_total = 0
        for prof, count in (config or {}).items():
            try:
                c = int(count)
            except Exception:
                continue
            if c <= 0:
                continue
            gip_id = gip_map.get(str(prof).lower())
            if not gip_id:
                steps.append({'cmd': f'# skip: unknown profile {prof}', 'rc': -1, 'out': ''})
                continue
            for _ in range(c):
                if created_total >= 4:
                    steps.append({'cmd': f'# cap reached (4 instances); skipping extra {prof}', 'rc': 0, 'out': ''})
                    break
                rc, out = run(['nvidia-smi', 'mig', '-cgi', str(gip_id), '-i', str(gpu_index), '-C'])
                created.append({'profile': prof, 'gip_id': gip_id, 'rc': rc, 'out': out})
                steps.append({'cmd': f'nvidia-smi mig -cgi {gip_id} -i {gpu_index} -C', 'rc': rc, 'out': out})
                created_total += 1

        # Attempt compute instance creation for any GPU instance lacking CI
        # List GPU instances to retrieve GI IDs
        rc, out = run(['nvidia-smi', 'mig', '-lgi', '-i', str(gpu_index)])
        steps.append({'cmd': f'nvidia-smi mig -lgi -i {gpu_index}', 'rc': rc, 'out': out})
        gi_ids = []
        try:
            for line in out.splitlines():
                m = re.search(r'GPU instance ID\s*:\s*(\d+)', line)
                if m:
                    gi_ids.append(int(m.group(1)))
        except Exception:
            pass
        # List current compute instances
        rc, out = run(['nvidia-smi', 'mig', '-lci', '-i', str(gpu_index)])
        steps.append({'cmd': f'nvidia-smi mig -lci -i {gpu_index}', 'rc': rc, 'out': out})
        existing_ci = out
        for gi in gi_ids:
            if str(gi) in existing_ci:
                continue
            # Try to create a compute instance for this GI (profile auto)
            rc, out = run(['nvidia-smi', 'mig', '-cci', '-gi', str(gi), '-i', str(gpu_index)])
            created_ci.append({'gi_id': gi, 'rc': rc, 'out': out})
            steps.append({'cmd': f'nvidia-smi mig -cci -gi {gi} -i {gpu_index}', 'rc': rc, 'out': out})

    return jsonify({
        'status': 'ok',
        'enable_mig': enable,
        'gpu_index': gpu_index,
        'applied': created,
        'compute_instances': created_ci,
        'steps': steps
    })


# ---------------------------
# Job configuration validation
# ---------------------------
def _validate_job_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    errors = []
    warnings = []
    name = (payload or {}).get('name')
    jtype = (payload or {}).get('type')
    framework = (payload or {}).get('framework')
    cfg = (payload or {}).get('config') or {}

    if not name:
        warnings.append('Job name is empty; a default will be used.')
    if jtype not in ('train', 'finetune'):
        errors.append('type must be one of: train, finetune')
    if framework not in ('pytorch', 'tensorflow', 'huggingface'):
        errors.append('framework must be one of: pytorch, tensorflow, huggingface')

    # Framework-specific checks
    if framework == 'pytorch':
        arch = cfg.get('architecture', 'custom')
        if arch not in ('custom', 'resnet', 'vgg', 'densenet', 'transformer'):
            errors.append('Unsupported architecture for PyTorch: ' + str(arch))
        # common numeric checks
        for k in ('epochs', 'batch_size'):
            v = cfg.get(k)
            if v is None or not isinstance(v, (int, float)) or v <= 0:
                errors.append(f'{k} must be a positive number')
        lr = cfg.get('learning_rate')
        if lr is None or not isinstance(lr, (int, float)) or lr <= 0:
            errors.append('learning_rate must be a positive number')
    elif framework == 'huggingface':
        if jtype == 'finetune':
            if not cfg.get('model_name'):
                errors.append('model_name is required for Hugging Face fine-tuning')
        for k in ('epochs', 'batch_size'):
            v = cfg.get(k)
            if v is None or not isinstance(v, (int, float)) or v <= 0:
                errors.append(f'{k} must be a positive number')

    # Model source validation for fine-tuning
    if jtype == 'finetune':
        model_source = cfg.get('model_source', 'torchvision' if framework == 'pytorch' else 'huggingface')

        if model_source == 'model_id':
            # Validate that model_id exists
            model_id = cfg.get('model_id')
            if not model_id:
                errors.append("model_id is required when model_source is 'model_id'")
            else:
                model_dir = os.path.join(MODELS_DIR, str(model_id))
                if not os.path.isdir(model_dir):
                    errors.append(f"Model with ID '{model_id}' not found in models directory")
                else:
                    # Validate model files exist
                    if framework == 'pytorch':
                        model_file = os.path.join(model_dir, 'model.pth')
                        if not os.path.exists(model_file):
                            errors.append(f"Model file 'model.pth' not found for model '{model_id}'")

                        # Validate config exists for reconstruction
                        config_file = os.path.join(model_dir, 'config.json')
                        if not os.path.exists(config_file):
                            errors.append(f"Model config file 'config.json' not found for model '{model_id}'")
                        else:
                            # Try to load and validate architecture info
                            try:
                                with open(config_file, 'r') as f:
                                    model_config = json.load(f)
                                    if 'architecture' not in model_config:
                                        warnings.append(f"Model '{model_id}' is missing architecture information")
                            except Exception as e:
                                errors.append(f"Failed to read model config: {str(e)}")

                    elif framework == 'huggingface':
                        hf_config_file = os.path.join(model_dir, 'config.json')
                        if not os.path.exists(hf_config_file):
                            errors.append(f"HuggingFace model config not found for model '{model_id}'. "
                                        "Make sure the model was trained with HuggingFace framework.")

        elif model_source == 'custom':
            # Validate that model_path is provided and exists
            model_path = cfg.get('model_path')
            if not model_path:
                errors.append("model_path is required when model_source is 'custom'")
            elif not os.path.exists(model_path):
                warnings.append(f"Model file not found at: {model_path}")

        elif model_source == 'torchvision' and framework == 'pytorch':
            # Validate model_name is valid
            valid_models = ['resnet18', 'resnet50', 'vgg16', 'densenet121']
            model_name = cfg.get('model_name')
            if model_name and model_name not in valid_models:
                errors.append(f"Invalid torchvision model '{model_name}'. Must be one of: {', '.join(valid_models)}")

    # Data source validation (best-effort)
    data_cfg = cfg.get('data', {}) if isinstance(cfg.get('data'), dict) else {}
    src = data_cfg.get('source')
    if src not in (None, 'local', 's3', 'gcs', 'huggingface'):
        errors.append('data.source must be one of: local, s3, gcs, huggingface')
    if src == 'local':
        path = data_cfg.get('local_path')
        if not path:
            errors.append('data.local_path is required for local data source')
        else:
            # Only validate existence if path is absolute or relative to BASE_DIR
            p = path if os.path.isabs(path) else os.path.join(BASE_DIR, path)
            if not os.path.exists(p):
                warnings.append(f'Local path not found on server: {p}')
    if src == 'huggingface':
        if not data_cfg.get('dataset_name'):
            errors.append('data.dataset_name is required for Hugging Face datasets')

    # Splits
    split = data_cfg.get('split', {}) if isinstance(data_cfg.get('split'), dict) else {}
    if split:
        tot = float(split.get('train', 0)) + float(split.get('val', 0)) + float(split.get('test', 0))
        if abs(tot - 1.0) > 1e-6 and abs(tot - 100.0) > 1e-3:
            warnings.append('Train/Val/Test split should sum to 1.0 (or 100%).')

    return {
        'ok': len(errors) == 0,
        'errors': errors,
        'warnings': warnings,
        'normalized_config': cfg,
    }


@app.route('/api/jobs/validate', methods=['POST'])
def validate_job():
    payload = request.json or {}
    result = _validate_job_payload(payload)
    code = 200 if result['ok'] else 400
    return jsonify(result), code


@app.route('/api/system/stream', methods=['GET'])
def system_stream():
    """Server-Sent Events stream of system info updated every 2 seconds."""
    def event_stream():
        prev_cpu = None
        prev_net = None
        prev_t = None
        while True:
            payload = _system_info_snapshot()
            # push into history too so connected clients keep history fresh
            try:
                ts = datetime.now().isoformat()
                gpus = [{
                    'index': g.get('index'),
                    'util_pct': g.get('utilization_gpu_pct'),
                    'mem_pct': g.get('memory_used_pct')
                } for g in payload.get('gpus', [])]
                sample = {
                    'ts': ts,
                    'gpus': gpus,
                    'gpu_mem_used_pct': payload.get('memory_used_pct'),
                    'sys_mem_used_pct': (payload.get('memory') or {}).get('used_pct'),
                }
                with _metrics_lock:
                    _metrics_history.append(sample)
            except Exception:
                pass
            # Per-core CPU percent via /proc/stat deltas
            try:
                def read_cpu_times():
                    times = []
                    with open('/proc/stat', 'r') as f:
                        for line in f:
                            if line.startswith('cpu'):
                                parts = line.split()
                                if parts[0] == 'cpu':
                                    continue  # aggregate skip
                                vals = list(map(int, parts[1:8]))  # user nice system idle iowait irq softirq
                                times.append(vals)
                    return times
                now = time.time()
                cur = read_cpu_times()
                if prev_cpu is not None and cur and prev_cpu and len(cur) == len(prev_cpu):
                    per_core_pct = []
                    total_pct = []
                    for prev_row, cur_row in zip(prev_cpu, cur):
                        prev_idle = prev_row[3] + prev_row[4]
                        idle = cur_row[3] + cur_row[4]
                        prev_total = sum(prev_row)
                        total = sum(cur_row)
                        diff_total = max(1, total - prev_total)
                        diff_idle = idle - prev_idle
                        usage = 100.0 * (diff_total - diff_idle) / diff_total
                        per_core_pct.append(round(usage, 1))
                    # estimate overall average
                    if per_core_pct:
                        total_pct = round(sum(per_core_pct) / len(per_core_pct), 1)
                        payload.setdefault('cpu', {})['total_pct'] = total_pct
                        payload['cpu']['per_core_pct'] = per_core_pct
                prev_cpu = cur
                # Net rates from delta of totals
                if 'net' in payload:
                    rx = payload['net'].get('rx_bytes')
                    tx = payload['net'].get('tx_bytes')
                    if prev_net is not None and prev_t is not None and rx is not None and tx is not None:
                        dt = max(1e-3, now - prev_t)
                        payload['net']['rx_rate_bps'] = (rx - prev_net[0]) / dt
                        payload['net']['tx_rate_bps'] = (tx - prev_net[1]) / dt
                    prev_net = (rx, tx)
                prev_t = now
            except Exception:
                pass
            yield f"data: {json.dumps(payload)}\n\n"
            time.sleep(2)

    from flask import Response
    headers = {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
        'X-Accel-Buffering': 'no',  # for nginx
    }
    return Response(event_stream(), headers=headers)


@app.route('/api/system/metrics/history', methods=['GET'])
def metrics_history():
    _ensure_metrics_thread()
    with _metrics_lock:
        samples = list(_metrics_history)
    # Attach jobs in the last hour window
    now = datetime.now()
    win_start = now.timestamp() - METRICS_WINDOW_SECONDS
    def parse_ts(s):
        try:
            return datetime.fromisoformat(s).timestamp()
        except Exception:
            return None
    jobs_out = []
    for j in jobs.values():
        started = j.get('started')
        completed = j.get('completed')
        s_ts = parse_ts(started) if started else None
        c_ts = parse_ts(completed) if completed else None
        # include jobs that overlap the window
        if s_ts is None and c_ts is None:
            continue
        s = s_ts if s_ts is not None else now.timestamp()
        e = c_ts if c_ts is not None else now.timestamp()
        if e >= win_start and s <= now.timestamp():
            jobs_out.append({
                'id': j.get('id'),
                'name': j.get('name'),
                'gpu': j.get('gpu'),
                'started': started,
                'completed': completed,
                'status': j.get('status'),
            })
    return jsonify({
        'window_seconds': METRICS_WINDOW_SECONDS,
        'interval_seconds': METRICS_INTERVAL_SECONDS,
        'samples': samples,
        'jobs': jobs_out,
    })


@app.route('/api/system/metrics/history.csv', methods=['GET'])
def metrics_history_csv():
    from flask import Response
    _ensure_metrics_thread()
    with _metrics_lock:
        samples = list(_metrics_history)
    # CSV header: timestamp,gpu_index,util_pct,mem_pct,sys_mem_used_pct
    lines = ['timestamp,gpu_index,util_pct,mem_pct,sys_mem_used_pct']
    for s in samples:
        ts = s.get('ts')
        sysm = s.get('sys_mem_used_pct')
        gpus = s.get('gpus') or []
        if not gpus:
            lines.append(f"{ts},,,,{sysm if sysm is not None else ''}")
        else:
            for g in gpus:
                gi = g.get('index')
                up = g.get('util_pct')
                mp = g.get('mem_pct')
                lines.append(f"{ts},{'' if gi is None else gi},{'' if up is None else up},{'' if mp is None else mp},{'' if sysm is None else sysm}")
    data = '\n'.join(lines)
    return Response(data, headers={'Content-Type': 'text/csv', 'Content-Disposition': 'attachment; filename=metrics_history.csv'})


# ==========================================
# Profile and Settings API Endpoints
# ==========================================

@app.route('/api/settings', methods=['GET'])
def get_settings():
    """Get user settings"""
    try:
        import sys
        sys.path.insert(0, os.path.join(BASE_DIR, 'src'))
        from spark_trainer.profiles import get_profile_manager

        manager = get_profile_manager()
        settings = manager.get_settings()
        return jsonify({
            'success': True,
            'settings': settings.__dict__
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/settings', methods=['PUT'])
def update_settings():
    """Update user settings"""
    try:
        import sys
        sys.path.insert(0, os.path.join(BASE_DIR, 'src'))
        from spark_trainer.profiles import get_profile_manager

        data = request.get_json()
        manager = get_profile_manager()
        settings = manager.update_settings(**data)
        return jsonify({
            'success': True,
            'settings': settings.__dict__
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/settings/environment', methods=['GET'])
def get_environment_info():
    """Get comprehensive environment information"""
    try:
        import sys
        sys.path.insert(0, os.path.join(BASE_DIR, 'src'))
        from spark_trainer.profiles import get_profile_manager

        manager = get_profile_manager()
        env_info = manager.get_environment_info()
        return jsonify({
            'success': True,
            'environment': env_info
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/profiles', methods=['GET'])
def get_profiles():
    """Get all configuration profiles"""
    try:
        import sys
        sys.path.insert(0, os.path.join(BASE_DIR, 'src'))
        from spark_trainer.profiles import get_profile_manager

        manager = get_profile_manager()
        tag = request.args.get('tag')

        if tag:
            profile_names = manager.list_profiles(tag=tag)
            profiles = {name: manager.get_profile(name) for name in profile_names}
        else:
            profiles = manager.get_profiles()

        return jsonify({
            'success': True,
            'profiles': profiles
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/profiles/<profile_name>', methods=['GET'])
def get_profile(profile_name):
    """Get a specific profile"""
    try:
        import sys
        sys.path.insert(0, os.path.join(BASE_DIR, 'src'))
        from spark_trainer.profiles import get_profile_manager

        manager = get_profile_manager()
        profile = manager.get_profile(profile_name)

        if profile is None:
            return jsonify({'error': 'Profile not found'}), 404

        return jsonify({
            'success': True,
            'profile': profile
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/profiles', methods=['POST'])
def create_profile():
    """Create a new configuration profile"""
    try:
        import sys
        sys.path.insert(0, os.path.join(BASE_DIR, 'src'))
        from spark_trainer.profiles import get_profile_manager

        data = request.get_json()
        name = data.get('name')
        config = data.get('config')
        description = data.get('description', '')
        tags = data.get('tags', [])

        if not name or not config:
            return jsonify({'error': 'Name and config are required'}), 400

        manager = get_profile_manager()
        manager.create_profile(name, config, description, tags)

        return jsonify({
            'success': True,
            'message': f'Profile {name} created successfully'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/profiles/<profile_name>', methods=['PUT'])
def update_profile(profile_name):
    """Update an existing profile"""
    try:
        import sys
        sys.path.insert(0, os.path.join(BASE_DIR, 'src'))
        from spark_trainer.profiles import get_profile_manager

        data = request.get_json()
        config = data.get('config')

        if not config:
            return jsonify({'error': 'Config is required'}), 400

        manager = get_profile_manager()
        manager.update_profile(profile_name, config)

        return jsonify({
            'success': True,
            'message': f'Profile {profile_name} updated successfully'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/profiles/<profile_name>', methods=['DELETE'])
def delete_profile(profile_name):
    """Delete a profile"""
    try:
        import sys
        sys.path.insert(0, os.path.join(BASE_DIR, 'src'))
        from spark_trainer.profiles import get_profile_manager

        manager = get_profile_manager()
        success = manager.delete_profile(profile_name)

        if not success:
            return jsonify({'error': 'Profile not found'}), 404

        return jsonify({
            'success': True,
            'message': f'Profile {profile_name} deleted successfully'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/settings/recent/<resource_type>', methods=['POST'])
def add_recent_resource(resource_type):
    """Add a resource to recent activity"""
    try:
        import sys
        sys.path.insert(0, os.path.join(BASE_DIR, 'src'))
        from spark_trainer.profiles import get_profile_manager

        data = request.get_json()
        resource_id = data.get('id')

        if not resource_id:
            return jsonify({'error': 'Resource ID is required'}), 400

        manager = get_profile_manager()

        if resource_type == 'dataset':
            manager.add_recent_dataset(resource_id)
        elif resource_type == 'model':
            manager.add_recent_model(resource_id)
        elif resource_type == 'experiment':
            manager.add_recent_experiment(resource_id)
        else:
            return jsonify({'error': 'Invalid resource type'}), 400

        return jsonify({
            'success': True,
            'message': f'Added {resource_id} to recent {resource_type}s'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/trainers/registry', methods=['GET'])
def get_trainer_registry():
    """Get the autodiscovered trainer registry"""
    try:
        import sys
        sys.path.insert(0, os.path.join(BASE_DIR, 'src'))
        from spark_trainer.trainer_registry import get_trainer_registry

        registry = get_trainer_registry()
        registry_data = registry.get_registry()

        # Optional filters
        model_type = request.args.get('model_type')
        tag = request.args.get('tag')

        if model_type or tag:
            trainers = registry.list_trainers(model_type=model_type, tag=tag)
            registry_data = {
                t.name: {
                    'name': t.name,
                    'description': t.description,
                    'module_path': t.module_path,
                    'class_name': t.class_name,
                    'model_types': t.model_types,
                    'tags': t.tags,
                    'author': t.author,
                    'version': t.version,
                    'requirements': t.requirements,
                    'discovered_at': t.discovered_at
                }
                for t in trainers
            }

        return jsonify({
            'success': True,
            'trainers': registry_data,
            'count': len(registry_data)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/trainers/registry/reload', methods=['POST'])
def reload_trainer_registry():
    """Reload the trainer registry (useful for development)"""
    try:
        import sys
        sys.path.insert(0, os.path.join(BASE_DIR, 'src'))
        from spark_trainer.trainer_registry import reload_registry

        registry = reload_registry()
        registry_data = registry.get_registry()

        return jsonify({
            'success': True,
            'message': f'Reloaded trainer registry with {len(registry_data)} trainers',
            'trainers': registry_data
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/trainers/<trainer_name>', methods=['GET'])
def get_trainer_info(trainer_name):
    """Get information about a specific trainer"""
    try:
        import sys
        sys.path.insert(0, os.path.join(BASE_DIR, 'src'))
        from spark_trainer.trainer_registry import get_trainer_registry

        registry = get_trainer_registry()
        metadata = registry.get_trainer_metadata(trainer_name)

        if metadata is None:
            return jsonify({'error': 'Trainer not found'}), 404

        return jsonify({
            'success': True,
            'trainer': {
                'name': metadata.name,
                'description': metadata.description,
                'module_path': metadata.module_path,
                'class_name': metadata.class_name,
                'model_types': metadata.model_types,
                'tags': metadata.tags,
                'author': metadata.author,
                'version': metadata.version,
                'requirements': metadata.requirements,
                'discovered_at': metadata.discovered_at
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/hpo/<job_id>', methods=['GET'])
def hpo_results(job_id):
    base = os.path.join(MODELS_DIR, job_id)
    if not os.path.isdir(base):
        return jsonify({'error': 'Job not found'}), 404
    out = {'results': None, 'trials': []}
    try:
        p = os.path.join(base, 'hpo_results.json')
        if os.path.exists(p):
            out['results'] = json.load(open(p))
    except Exception:
        out['results'] = None
    try:
        p = os.path.join(base, 'hpo_trials.json')
        if os.path.exists(p):
            t = json.load(open(p))
            out['trials'] = t.get('trials') or []
    except Exception:
        pass
    return jsonify(out)

@app.route('/api/export/zip', methods=['POST'])
def export_zip():
    """Create a ZIP archive from posted files and return it.

    Request JSON format:
      { "files": [ {"path": "path/in/zip.txt", "content": "..."}, ... ], "name": "archive.zip" }
    """
    try:
        payload = request.json or {}
        files = payload.get('files') or []
        if not isinstance(files, list) or not files:
            return jsonify({'error': 'No files provided'}), 400
        name = payload.get('name') or 'export.zip'
        mem = io.BytesIO()
        with zipfile.ZipFile(mem, 'w', zipfile.ZIP_DEFLATED) as z:
            for f in files:
                try:
                    p = (f.get('path') or '').strip()
                    c = f.get('content')
                    if not p:
                        continue
                    # sanitize path: keep relative, no parent traversals
                    p = os.path.normpath(p).replace('\\', '/')
                    if p.startswith('../') or p.startswith('/'):
                        p = p.lstrip('./')
                    if isinstance(c, str):
                        data = c.encode('utf-8')
                    elif isinstance(c, bytes):
                        data = c
                    else:
                        data = json.dumps(c, indent=2).encode('utf-8')
                    z.writestr(p, data)
                except Exception:
                    continue
        mem.seek(0)
        return send_file(mem, mimetype='application/zip', as_attachment=True, download_name=name)
    except Exception as e:
        return jsonify({'error': 'Failed to build zip', 'detail': str(e)}), 500


@app.route('/api/curation/ollama/run', methods=['POST'])
def ollama_curation():
    """Stub endpoint for Ollama-based curation tasks.

    JSON: { model: 'llama3', prompt: '...', input: {...}, task: 'label|score|dedupe', max_tokens: 256 }
    """
    data = request.json or {}
    host = os.environ.get('OLLAMA_HOST')
    if not host:
        return jsonify({'status': 'skipped', 'message': 'OLLAMA_HOST not configured; install and set OLLAMA_HOST to enable'}), 200
    return jsonify({'status': 'accepted', 'note': 'Ollama integration not fully implemented in this build'}), 202


@app.route('/api/curation/openai/run', methods=['POST'])
def openai_curation():
    """Stub endpoint for OpenAI-based curation tasks.

    JSON: { model: 'gpt-4o-mini', messages: [...], task: 'synth|augment|assess|bias' }
    """
    if not os.environ.get('OPENAI_API_KEY'):
        return jsonify({'status': 'skipped', 'message': 'OPENAI_API_KEY not configured'}), 200
    return jsonify({'status': 'accepted', 'note': 'OpenAI integration not fully implemented in this build'}), 202


# ================= Storage Management =================
def _dir_usage_bytes(path: str, limit_files: int | None = None):
    total = 0
    largest = []  # list of (size, rel)
    start = path
    for r, _, fns in os.walk(path):
        for fn in fns:
            fp = os.path.join(r, fn)
            try:
                sz = os.path.getsize(fp)
            except Exception:
                continue
            total += sz
            rel = os.path.relpath(fp, start)
            largest.append((sz, rel))
    largest.sort(reverse=True)
    if limit_files is not None:
        largest = largest[:limit_files]
    return total, [{'file': rel, 'size_bytes': sz} for sz, rel in largest]


@app.route('/api/storage/usage', methods=['GET'])
def storage_usage():
    dirs = {
        'models': MODELS_DIR,
        'datasets': DATASETS_DIR,
        'logs': LOGS_DIR,
        'jobs': JOBS_DIR,
    }
    out = {}
    for k, p in dirs.items():
        try:
            total, largest = _dir_usage_bytes(p, limit_files=20)
            out[k] = {'path': p, 'total_bytes': total, 'largest_files': largest}
        except Exception:
            out[k] = {'path': p, 'total_bytes': 0, 'largest_files': []}
    # Persist a trend point
    trend_path = os.path.join(BASE_DIR, 'storage_stats.jsonl')
    rec = {'ts': datetime.now().isoformat(), 'totals': {k: v['total_bytes'] for k,v in out.items()}}
    try:
        with open(trend_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(rec) + '\n')
    except Exception:
        pass
    # Cleanup suggestions
    suggestions = []
    try:
        # Large logs
        for it in out.get('logs', {}).get('largest_files', [])[:5]:
            if it['size_bytes'] > 50*1024*1024:
                suggestions.append({'action': 'compress', 'target': os.path.join(LOGS_DIR, it['file']), 'reason': 'Large log file'})
        # Datasets: suggest deleting quarantine
        for root, dirs, _ in os.walk(DATASETS_DIR):
            if '_quarantine' in dirs:
                suggestions.append({'action': 'review_quarantine', 'target': os.path.join(root, '_quarantine')})
    except Exception:
        pass
    return jsonify({'usage': out, 'suggestions': suggestions})


@app.route('/api/storage/trends', methods=['GET'])
def storage_trends():
    path = os.path.join(BASE_DIR, 'storage_stats.jsonl')
    items = []
    if os.path.exists(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                for ln in f:
                    try: items.append(json.loads(ln))
                    except: pass
        except Exception:
            pass
    return jsonify({'items': items[-200:]})


@app.route('/api/storage/checkpoints/cleanup', methods=['POST'])
def storage_checkpoints_cleanup():
    data = request.json or {}
    keep = int(data.get('keep') or 3)
    model_id = data.get('model_id')
    targets = []
    if model_id:
        targets = [os.path.join(MODELS_DIR, _safe_name(model_id))]
    else:
        # all models
        targets = [os.path.join(MODELS_DIR, d) for d in os.listdir(MODELS_DIR) if os.path.isdir(os.path.join(MODELS_DIR, d))]
    cleaned = []
    for t in targets:
        try:
            files = []
            for fn in os.listdir(t):
                if fn.endswith('.pth') or fn.startswith('checkpoint-'):
                    fp = os.path.join(t, fn)
                    try:
                        st = os.stat(fp)
                        files.append((st.st_mtime, fp))
                    except Exception:
                        pass
            files.sort(reverse=True)
            for _, fp in files[keep:]:
                # compress then remove
                z = fp + '.zip'
                try:
                    with zipfile.ZipFile(z, 'w', zipfile.ZIP_DEFLATED) as zz:
                        zz.write(fp, arcname=os.path.basename(fp))
                    os.remove(fp)
                    cleaned.append({'path': fp, 'archived': z})
                except Exception:
                    pass
        except Exception:
            pass
    return jsonify({'status': 'ok', 'cleaned': cleaned})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
