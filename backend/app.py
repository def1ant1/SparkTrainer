from flask import Flask, request, jsonify
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

os.makedirs(JOBS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# In-memory job tracking
jobs = {}

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
    """List available pre-trained models"""
    models = []
    if os.path.exists(MODELS_DIR):
        for model_dir in os.listdir(MODELS_DIR):
            model_path = os.path.join(MODELS_DIR, model_dir)
            if os.path.isdir(model_path):
                config_path = os.path.join(model_path, 'config.json')
                if os.path.exists(config_path):
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                        models.append({
                            'id': model_dir,
                            'name': config.get('name', model_dir),
                            'framework': config.get('framework', 'unknown'),
                            'created': config.get('created', 'unknown')
                        })
    return jsonify(models)

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
    
    return jsonify(job)

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

    job = {
        'id': job_id,
        'name': data.get('name', f'Training Job {job_id[:8]}'),
        'type': data.get('type', 'train'),  # 'train' or 'finetune'
        'framework': data.get('framework', 'pytorch'),
        'status': 'queued',
        'created': datetime.now().isoformat(),
        'config': data.get('config', {}),
        'progress': 0,
        'metrics': {},
        'gpu': gpu_meta
    }
    
    jobs[job_id] = job
    save_jobs()
    
    # Start training in background thread
    thread = threading.Thread(target=run_training_job, args=(job_id,))
    thread.daemon = True
    thread.start()
    
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


@app.route('/api/system/info', methods=['GET'])
def system_info():
    """Get DGX system information (single snapshot)."""
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


@app.route('/api/system/stream', methods=['GET'])
def system_stream():
    """Server-Sent Events stream of system info updated every 2 seconds."""
    def event_stream():
        prev_cpu = None
        prev_net = None
        prev_t = None
        while True:
            payload = _system_info_snapshot()
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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
