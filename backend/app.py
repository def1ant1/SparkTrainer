from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import json
import subprocess
import uuid
from datetime import datetime
import threading
import signal
import time
from typing import Dict, Any
from collections import deque
import zipfile
import io
import re
import shutil

app = Flask(__name__)
CORS(app)

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

os.makedirs(JOBS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(DATASETS_DIR, exist_ok=True)

# In-memory job tracking
jobs = {}

# Metrics history storage (last hour)
METRICS_INTERVAL_SECONDS = int(os.environ.get('METRICS_INTERVAL_SECONDS', '5'))
METRICS_WINDOW_SECONDS = int(os.environ.get('METRICS_WINDOW_SECONDS', '3600'))
_metrics_history = deque(maxlen=max(10, METRICS_WINDOW_SECONDS // max(1, METRICS_INTERVAL_SECONDS) + 5))
_metrics_thread = None
_metrics_thread_started = False
_metrics_lock = threading.Lock()

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
        info.update({
            'name': cfg.get('name', model_dir),
            'framework': cfg.get('framework', meta.get('framework', 'unknown')),
            'architecture': cfg.get('architecture', meta.get('architecture')),
            'created': cfg.get('created', meta.get('created')),
            'parameters': cfg.get('parameters'),
            'metrics': metrics,
            'tags': meta.get('tags', []),
            'size_bytes': total_size,
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
                    ' '.join(m.get('tags') or [])
                ]).lower()
                if q not in hay:
                    continue
            items.append(m)

    def sort_key(x):
        if sort == 'size':
            return x.get('size_bytes') or 0
        if sort == 'accuracy':
            return (x.get('metrics') or {}).get('eval_accuracy') or 0
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
    if not os.path.isdir(os.path.join(MODELS_DIR, model_id)):
        return jsonify({'error': 'Model not found'}), 404
    return jsonify(_model_detail(model_id))


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
    mem.seek(0)
    return send_file(mem, mimetype='application/zip', as_attachment=True, download_name='models_export.zip')


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
    }
    
    jobs[job_id] = job
    save_jobs()
    
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

def run_training_job(job_id):
    """Execute the training job"""
    job = jobs[job_id]
    config = job['config']
    
    try:
        job['status'] = 'running'
        job['started'] = datetime.now().isoformat()
        save_jobs()
        
        # Prepare training command
        script_path = TRAINING_SCRIPTS_DIR
        
        if job['type'] == 'train':
            script = os.path.join(script_path, f"train_{job['framework']}.py")
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
            '--query-gpu=name,index,memory.total,memory.free,utilization.gpu,temperature.gpu',
            '--format=csv,noheader,nounits'
        ]).decode()
        for line in out.strip().split('\n'):
            parts = [p.strip() for p in line.split(',')]
            if len(parts) == 6:
                name, idx_s, total_s, free_s, util_s, temp_s = parts
                try:
                    idx = int(idx_s)
                except Exception:
                    idx = 0
                def to_int(s: str) -> int:
                    try:
                        return int(float(s))
                    except Exception:
                        return 0
                total = to_int(total_s)      # MiB
                free = to_int(free_s)        # MiB
                used = max(0, total - free)  # MiB
                util = to_int(util_s)        # %
                temp = to_int(temp_s)        # Celsius
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

    # Net totals via /proc/net/dev
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
        base['net'] = {'rx_bytes': rx, 'tx_bytes': tx}
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
            items.append({
                'name': nm,
                'latest': latest,
                'versions': len(versions),
                'meta': meta,
                'stats': stats,
            })
    return jsonify(items)


@app.route('/api/datasets/<name>', methods=['GET'])
def dataset_detail(name):
    name = _safe_name(name)
    base = _dataset_dir(name)
    if not os.path.isdir(base):
        return jsonify({'error': 'Dataset not found'}), 404
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
    return jsonify({'name': name, 'meta': meta, 'versions': out_vers})


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


@app.route('/api/system/info', methods=['GET'])
def system_info():
    """Get DGX system information (single snapshot)."""
    _ensure_metrics_thread()
    return jsonify(_system_info_snapshot())


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
    # Support dry-run by default; execute only if MIG_APPLY_EXECUTE=1
    if os.environ.get('MIG_APPLY_EXECUTE') != '1':
        return jsonify({
            'status': 'accepted',
            'note': 'Dry-run: set MIG_APPLY_EXECUTE=1 to execute. No changes applied.',
            'requested': {
                'gpu_index': gpu_index,
                'enable_mig': enable,
                'config': config
            }
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
                rc, out = run(['nvidia-smi', 'mig', '-cgi', str(gip_id), '-i', str(gpu_index), '-C'])
                created.append({'profile': prof, 'gip_id': gip_id, 'rc': rc, 'out': out})
                steps.append({'cmd': f'nvidia-smi mig -cgi {gip_id} -i {gpu_index} -C', 'rc': rc, 'out': out})

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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
