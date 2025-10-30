"""
Flask-RESTX API Documentation for SparkTrainer

This module provides interactive Swagger/OpenAPI documentation for all SparkTrainer API endpoints.
Access the documentation at: http://localhost:5000/api/docs
"""

from flask import Blueprint
from flask_restx import Api, Resource, fields, Namespace
from functools import wraps
import os

# Create Blueprint for API documentation
api_bp = Blueprint('api', __name__, url_prefix='/api')

# Initialize Flask-RESTX with comprehensive documentation
api = Api(
    api_bp,
    version='1.0.0',
    title='SparkTrainer API',
    description="""
    **SparkTrainer** - GPU-Accelerated Multimodal AI Training Platform

    ## Overview

    SparkTrainer provides a comprehensive REST API for:
    - **Training Jobs**: Submit, monitor, and manage ML training jobs
    - **Experiments**: Track experiments with MLflow integration
    - **Models**: Browse, version, and serve trained models
    - **Datasets**: Ingest, version, and manage multimodal datasets
    - **Recipes**: Configure training recipes (LoRA, QLoRA, Full Fine-tuning)
    - **Monitoring**: Real-time metrics, GPU utilization, system health

    ## Authentication

    Most endpoints require JWT authentication:
    1. Login via `/api/auth/login` to get access token
    2. Include token in `Authorization: Bearer <token>` header

    ## Rate Limiting

    API requests are rate-limited per user/IP. See response headers for limits.

    ## WebSocket Support

    Real-time updates available via WebSocket at `/ws`:
    - Job status updates
    - Training metrics streaming
    - System health monitoring

    ## SDKs and Examples

    - Python SDK: `pip install spark-trainer-client`
    - Jupyter Notebooks: `/examples/` directory
    - CLI: `spark-trainer` command-line tool
    """,
    doc='/docs',  # Swagger UI path
    authorizations={
        'Bearer': {
            'type': 'apiKey',
            'in': 'header',
            'name': 'Authorization',
            'description': 'JWT Bearer token. Format: Bearer <token>'
        }
    },
    security='Bearer',
    contact='https://github.com/def1ant1/SparkTrainer',
    license='Apache 2.0',
    license_url='https://www.apache.org/licenses/LICENSE-2.0.html',
)

# ============================================================================
# Namespaces - Organize endpoints by domain
# ============================================================================

ns_jobs = Namespace('jobs', description='Training job operations')
ns_experiments = Namespace('experiments', description='MLflow experiment tracking')
ns_models = Namespace('models', description='Model registry and versioning')
ns_datasets = Namespace('datasets', description='Dataset ingestion and management')
ns_recipes = Namespace('recipes', description='Training recipe configuration')
ns_adapters = Namespace('adapters', description='LoRA/QLoRA adapter management')
ns_system = Namespace('system', description='System metrics and health')
ns_auth = Namespace('auth', description='Authentication and authorization')
ns_pipelines = Namespace('pipelines', description='Training pipeline orchestration')
ns_hpo = Namespace('hpo', description='Hyperparameter optimization')

api.add_namespace(ns_jobs)
api.add_namespace(ns_experiments)
api.add_namespace(ns_models)
api.add_namespace(ns_datasets)
api.add_namespace(ns_recipes)
api.add_namespace(ns_adapters)
api.add_namespace(ns_system)
api.add_namespace(ns_auth)
api.add_namespace(ns_pipelines)
api.add_namespace(ns_hpo)

# ============================================================================
# Data Models - Request/Response schemas
# ============================================================================

# Job Models
job_config = api.model('JobConfig', {
    'name': fields.String(required=True, description='Job name', example='my-training-job'),
    'recipe': fields.String(required=True, description='Training recipe', example='lora_qlora'),
    'base_model': fields.String(required=True, description='Base model name', example='meta-llama/Llama-2-7b-hf'),
    'dataset': fields.String(required=True, description='Dataset name', example='my-dataset'),
    'hyperparameters': fields.Raw(description='Training hyperparameters', example={
        'learning_rate': 2e-4,
        'num_epochs': 3,
        'batch_size': 4,
        'lora_r': 16,
        'lora_alpha': 32
    }),
    'resources': fields.Raw(description='Resource allocation', example={
        'gpu_count': 1,
        'memory_gb': 32
    })
})

job_status = api.model('JobStatus', {
    'id': fields.String(description='Job ID'),
    'name': fields.String(description='Job name'),
    'status': fields.String(description='Current status', enum=['pending', 'queued', 'running', 'completed', 'failed', 'cancelled']),
    'progress': fields.Float(description='Progress percentage (0-100)'),
    'created_at': fields.DateTime(description='Creation timestamp'),
    'started_at': fields.DateTime(description='Start timestamp'),
    'completed_at': fields.DateTime(description='Completion timestamp'),
    'metrics': fields.Raw(description='Training metrics'),
    'error': fields.String(description='Error message if failed')
})

# Experiment Models
experiment_create = api.model('ExperimentCreate', {
    'name': fields.String(required=True, description='Experiment name'),
    'base_model': fields.String(required=True, description='Base model'),
    'recipe': fields.String(required=True, description='Training recipe'),
    'dataset': fields.String(required=True, description='Dataset name'),
    'adapter_type': fields.String(description='Adapter type (lora/qlora)', example='lora'),
    'hyperparameters': fields.Raw(description='Hyperparameters')
})

experiment_detail = api.model('ExperimentDetail', {
    'id': fields.String(description='Experiment ID'),
    'name': fields.String(description='Experiment name'),
    'mlflow_run_id': fields.String(description='MLflow run ID'),
    'status': fields.String(description='Status'),
    'metrics': fields.Raw(description='Metrics'),
    'parameters': fields.Raw(description='Parameters'),
    'artifacts': fields.List(fields.String, description='Artifact URLs')
})

# Model Models
model_info = api.model('ModelInfo', {
    'id': fields.String(description='Model ID'),
    'name': fields.String(description='Model name'),
    'family': fields.String(description='Model family', example='llama'),
    'modality': fields.String(description='Modality', example='text'),
    'parameters': fields.String(description='Parameter count', example='7B'),
    'context_length': fields.Integer(description='Context window size'),
    'quantization': fields.String(description='Quantization', example='4bit'),
    'adapters': fields.List(fields.String, description='Attached adapters'),
    'created_at': fields.DateTime(description='Creation timestamp')
})

# Dataset Models
dataset_info = api.model('DatasetInfo', {
    'name': fields.String(description='Dataset name'),
    'type': fields.String(description='Dataset type', enum=['text', 'image', 'video', 'audio', 'multimodal']),
    'size': fields.Integer(description='Number of samples'),
    'size_bytes': fields.Integer(description='Size in bytes'),
    'split': fields.Raw(description='Train/val/test split'),
    'created_at': fields.DateTime(description='Creation timestamp'),
    'checksum': fields.String(description='Dataset checksum')
})

dataset_upload = api.model('DatasetUpload', {
    'name': fields.String(required=True, description='Dataset name'),
    'type': fields.String(required=True, description='Dataset type'),
    'description': fields.String(description='Dataset description'),
    'quality_gates': fields.List(fields.String, description='Quality gates to apply', example=['dedupe', 'pii_redaction'])
})

# System Models
system_info = api.model('SystemInfo', {
    'gpu': fields.Raw(description='GPU information', example={
        'count': 4,
        'models': ['NVIDIA A100-SXM4-40GB'],
        'utilization': [45, 67, 23, 89],
        'memory_used_mb': [12000, 18000, 8000, 32000]
    }),
    'memory': fields.Raw(description='System memory', example={
        'total_gb': 256,
        'used_gb': 128,
        'available_gb': 128
    }),
    'cpu': fields.Raw(description='CPU information'),
    'network': fields.Raw(description='Network I/O'),
    'celery': fields.Raw(description='Celery worker status')
})

# Recipe Models
recipe_info = api.model('RecipeInfo', {
    'name': fields.String(description='Recipe name', example='lora_qlora'),
    'type': fields.String(description='Recipe type', example='adapter'),
    'description': fields.String(description='Recipe description'),
    'compatible_models': fields.List(fields.String, description='Compatible model families'),
    'hyperparameters': fields.Raw(description='Default hyperparameters')
})

# ============================================================================
# API Endpoints Documentation
# ============================================================================

@ns_jobs.route('/')
class JobList(Resource):
    @ns_jobs.doc('list_jobs',
                 description='List all training jobs with optional filtering',
                 params={
                     'status': 'Filter by status (pending/queued/running/completed/failed/cancelled)',
                     'limit': 'Maximum number of jobs to return (default: 50)',
                     'offset': 'Pagination offset'
                 })
    @ns_jobs.marshal_list_with(job_status)
    def get(self):
        """List all training jobs"""
        pass

    @ns_jobs.doc('create_job',
                 description='Submit a new training job to the queue')
    @ns_jobs.expect(job_config)
    @ns_jobs.marshal_with(job_status, code=201)
    def post(self):
        """Create a new training job"""
        pass


@ns_jobs.route('/<string:job_id>')
@ns_jobs.param('job_id', 'The job identifier')
class Job(Resource):
    @ns_jobs.doc('get_job', description='Get detailed job information')
    @ns_jobs.marshal_with(job_status)
    def get(self, job_id):
        """Get job details by ID"""
        pass

    @ns_jobs.doc('delete_job', description='Cancel and delete a job')
    @ns_jobs.response(204, 'Job deleted successfully')
    def delete(self, job_id):
        """Delete a job"""
        pass


@ns_jobs.route('/<string:job_id>/cancel')
@ns_jobs.param('job_id', 'The job identifier')
class JobCancel(Resource):
    @ns_jobs.doc('cancel_job',
                 description='Cancel a running or queued job')
    @ns_jobs.response(200, 'Job cancelled successfully')
    def post(self, job_id):
        """Cancel a job"""
        pass


@ns_jobs.route('/<string:job_id>/metrics')
@ns_jobs.param('job_id', 'The job identifier')
class JobMetrics(Resource):
    @ns_jobs.doc('get_job_metrics',
                 description='Get real-time training metrics for a job')
    @ns_jobs.response(200, 'Metrics retrieved successfully')
    def get(self, job_id):
        """Get job training metrics"""
        pass


@ns_jobs.route('/<string:job_id>/logs')
@ns_jobs.param('job_id', 'The job identifier')
class JobLogs(Resource):
    @ns_jobs.doc('get_job_logs',
                 description='Stream training logs for a job',
                 params={
                     'tail': 'Number of lines to return from end of log (default: 100)',
                     'follow': 'Stream logs in real-time (SSE)'
                 })
    @ns_jobs.response(200, 'Logs retrieved successfully')
    def get(self, job_id):
        """Get job logs"""
        pass


@ns_experiments.route('/')
class ExperimentList(Resource):
    @ns_experiments.doc('list_experiments',
                        description='List all MLflow experiments')
    @ns_experiments.marshal_list_with(experiment_detail)
    def get(self):
        """List all experiments"""
        pass

    @ns_experiments.doc('create_experiment',
                        description='Create a new experiment')
    @ns_experiments.expect(experiment_create)
    @ns_experiments.marshal_with(experiment_detail, code=201)
    def post(self):
        """Create a new experiment"""
        pass


@ns_experiments.route('/<string:experiment_id>')
@ns_experiments.param('experiment_id', 'The experiment identifier')
class Experiment(Resource):
    @ns_experiments.doc('get_experiment')
    @ns_experiments.marshal_with(experiment_detail)
    def get(self, experiment_id):
        """Get experiment details"""
        pass


@ns_models.route('/')
class ModelList(Resource):
    @ns_models.doc('list_models',
                   description='Browse available models',
                   params={
                       'family': 'Filter by model family (llama/mistral/gpt/etc)',
                       'modality': 'Filter by modality (text/vision/audio/multimodal)',
                       'trainable': 'Filter by trainable status (true/false)'
                   })
    @ns_models.marshal_list_with(model_info)
    def get(self):
        """List all models"""
        pass


@ns_models.route('/<string:model_id>')
@ns_models.param('model_id', 'The model identifier')
class Model(Resource):
    @ns_models.doc('get_model')
    @ns_models.marshal_with(model_info)
    def get(self, model_id):
        """Get model details"""
        pass


@ns_models.route('/<string:model_id>/adapters')
@ns_models.param('model_id', 'The model identifier')
class ModelAdapters(Resource):
    @ns_models.doc('list_model_adapters',
                   description='List all LoRA/QLoRA adapters for a model')
    @ns_models.response(200, 'Adapters retrieved successfully')
    def get(self, model_id):
        """Get model adapters"""
        pass


@ns_datasets.route('/')
class DatasetList(Resource):
    @ns_datasets.doc('list_datasets',
                     description='List all datasets')
    @ns_datasets.marshal_list_with(dataset_info)
    def get(self):
        """List all datasets"""
        pass

    @ns_datasets.doc('create_dataset',
                     description='Create a new dataset (metadata only)')
    @ns_datasets.expect(dataset_upload)
    @ns_datasets.marshal_with(dataset_info, code=201)
    def post(self):
        """Create a new dataset"""
        pass


@ns_datasets.route('/upload')
class DatasetUpload(Resource):
    @ns_datasets.doc('upload_dataset',
                     description='Upload dataset files (multipart/form-data)',
                     params={
                         'file': 'Dataset file(s) to upload',
                         'name': 'Dataset name',
                         'type': 'Dataset type (text/image/video/audio)'
                     })
    @ns_datasets.response(201, 'Dataset uploaded successfully')
    def post(self):
        """Upload dataset files"""
        pass


@ns_datasets.route('/<string:dataset_name>/samples')
@ns_datasets.param('dataset_name', 'The dataset name')
class DatasetSamples(Resource):
    @ns_datasets.doc('get_dataset_samples',
                     description='Preview dataset samples',
                     params={
                         'limit': 'Number of samples to return (default: 10)',
                         'offset': 'Pagination offset'
                     })
    @ns_datasets.response(200, 'Samples retrieved successfully')
    def get(self, dataset_name):
        """Get dataset samples"""
        pass


@ns_system.route('/info')
class SystemInfo(Resource):
    @ns_system.doc('get_system_info',
                   description='Get current system metrics (GPU, memory, network, etc.)')
    @ns_system.marshal_with(system_info)
    def get(self):
        """Get system information"""
        pass


@ns_system.route('/metrics/history')
class MetricsHistory(Resource):
    @ns_system.doc('get_metrics_history',
                   description='Get historical system metrics',
                   params={
                       'window': 'Time window in seconds (default: 3600)',
                       'metric': 'Specific metric to retrieve (gpu/memory/network)'
                   })
    @ns_system.response(200, 'Metrics retrieved successfully')
    def get(self):
        """Get metrics history"""
        pass


@ns_system.route('/health')
class Health(Resource):
    @ns_system.doc('health_check',
                   description='Health check endpoint for load balancers')
    @ns_system.response(200, 'System healthy')
    @ns_system.response(503, 'System unhealthy')
    def get(self):
        """Health check"""
        pass


@ns_recipes.route('/')
class RecipeList(Resource):
    @ns_recipes.doc('list_recipes',
                    description='List all available training recipes')
    @ns_recipes.marshal_list_with(recipe_info)
    def get(self):
        """List all recipes"""
        pass


@ns_recipes.route('/<string:recipe_name>')
@ns_recipes.param('recipe_name', 'The recipe name')
class Recipe(Resource):
    @ns_recipes.doc('get_recipe',
                    description='Get detailed recipe configuration')
    @ns_recipes.marshal_with(recipe_info)
    def get(self, recipe_name):
        """Get recipe details"""
        pass


@ns_auth.route('/login')
class Login(Resource):
    @ns_auth.doc('login',
                 description='Authenticate and get JWT token')
    @ns_auth.expect(api.model('LoginCredentials', {
        'username': fields.String(required=True, description='Username'),
        'password': fields.String(required=True, description='Password')
    }))
    @ns_auth.response(200, 'Login successful')
    @ns_auth.response(401, 'Invalid credentials')
    def post(self):
        """Login to get JWT token"""
        pass


@ns_auth.route('/logout')
class Logout(Resource):
    @ns_auth.doc('logout',
                 description='Invalidate current session')
    @ns_auth.response(200, 'Logout successful')
    def post(self):
        """Logout current session"""
        pass


@ns_pipelines.route('/')
class PipelineList(Resource):
    @ns_pipelines.doc('list_pipelines',
                      description='List all training pipelines')
    @ns_pipelines.response(200, 'Pipelines retrieved successfully')
    def get(self):
        """List all pipelines"""
        pass


@ns_hpo.route('/studies')
class HPOStudies(Resource):
    @ns_hpo.doc('list_hpo_studies',
                description='List hyperparameter optimization studies')
    @ns_hpo.response(200, 'Studies retrieved successfully')
    def get(self):
        """List HPO studies"""
        pass

    @ns_hpo.doc('create_hpo_study',
                description='Create a new HPO study')
    @ns_hpo.response(201, 'Study created successfully')
    def post(self):
        """Create HPO study"""
        pass


# ============================================================================
# Helper function to register API blueprint
# ============================================================================

def register_api_docs(app):
    """
    Register the API documentation blueprint with the Flask app.

    Args:
        app: Flask application instance

    Example:
        from api_docs import register_api_docs
        register_api_docs(app)
    """
    app.register_blueprint(api_bp)

    print("=" * 80)
    print("API Documentation registered successfully!")
    print("=" * 80)
    print(f"Swagger UI available at: http://localhost:{os.environ.get('PORT', 5000)}/api/docs")
    print("=" * 80)
