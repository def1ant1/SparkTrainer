from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
import subprocess
import uuid
from datetime import datetime
import threading
import signal

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
    
    job = {
        'id': job_id,
        'name': data.get('name', f'Training Job {job_id[:8]}'),
        'type': data.get('type', 'train'),  # 'train' or 'finetune'
        'framework': data.get('framework', 'pytorch'),
        'status': 'queued',
        'created': datetime.now().isoformat(),
        'config': data.get('config', {}),
        'progress': 0,
        'metrics': {}
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
            process = subprocess.Popen(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                cwd=BASE_DIR
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

@app.route('/api/system/info', methods=['GET'])
def system_info():
    """Get DGX system information"""
    try:
        # Try to get GPU info
        gpu_info = subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total,memory.used', '--format=csv,noheader']).decode()
        gpus = []
        for line in gpu_info.strip().split('\n'):
            parts = line.split(',')
            if len(parts) == 3:
                gpus.append({
                    'name': parts[0].strip(),
                    'memory_total': parts[1].strip(),
                    'memory_used': parts[2].strip()
                })
    except:
        gpus = []
    
    return jsonify({
        'gpus': gpus,
        'jobs_running': len([j for j in jobs.values() if j['status'] == 'running']),
        'jobs_queued': len([j for j in jobs.values() if j['status'] == 'queued']),
        'models_available': len(os.listdir(MODELS_DIR)) if os.path.exists(MODELS_DIR) else 0
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
