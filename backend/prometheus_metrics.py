"""
Prometheus Metrics Exporter for SparkTrainer

This module provides comprehensive Prometheus metrics for monitoring:
- Training job metrics (throughput, queue depth, success rate)
- System metrics (GPU, memory, CPU)
- API request metrics
- Celery worker metrics
- MLflow integration metrics

Access metrics at: http://localhost:5000/metrics
"""

from prometheus_client import (
    Counter, Gauge, Histogram, Summary,
    generate_latest, CollectorRegistry, CONTENT_TYPE_LATEST
)
from prometheus_flask_exporter import PrometheusMetrics
from flask import Response
import time
import psutil
try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False
    pynvml = None
from typing import Dict, Any, Optional
import threading


class SparkTrainerMetrics:
    """
    Centralized metrics collection for SparkTrainer.

    This class provides a comprehensive set of Prometheus metrics
    for monitoring all aspects of the SparkTrainer platform.
    """

    def __init__(self, app=None, registry=None):
        """
        Initialize Prometheus metrics.

        Args:
            app: Flask application instance (optional)
            registry: Prometheus CollectorRegistry (optional)
        """
        self.registry = registry or CollectorRegistry()

        # Initialize NVML for GPU metrics
        self.nvml_initialized = False
        if NVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.nvml_initialized = True
            except Exception as e:
                print(f"Warning: Could not initialize NVML: {e}")

        # ===================================================================
        # Job Metrics
        # ===================================================================

        self.jobs_total = Counter(
            'sparktrainer_jobs_total',
            'Total number of training jobs submitted',
            ['recipe', 'status'],
            registry=self.registry
        )

        self.jobs_active = Gauge(
            'sparktrainer_jobs_active',
            'Number of currently running jobs',
            registry=self.registry
        )

        self.jobs_queued = Gauge(
            'sparktrainer_jobs_queued',
            'Number of jobs waiting in queue',
            registry=self.registry
        )

        self.job_duration_seconds = Histogram(
            'sparktrainer_job_duration_seconds',
            'Training job duration in seconds',
            ['recipe', 'status'],
            buckets=(60, 300, 900, 1800, 3600, 7200, 14400, 28800, 86400),
            registry=self.registry
        )

        self.job_training_loss = Gauge(
            'sparktrainer_job_training_loss',
            'Current training loss for active jobs',
            ['job_id', 'job_name'],
            registry=self.registry
        )

        self.job_training_accuracy = Gauge(
            'sparktrainer_job_training_accuracy',
            'Current training accuracy for active jobs',
            ['job_id', 'job_name'],
            registry=self.registry
        )

        self.job_gpu_memory_allocated_bytes = Gauge(
            'sparktrainer_job_gpu_memory_allocated_bytes',
            'GPU memory allocated by job',
            ['job_id', 'gpu_id'],
            registry=self.registry
        )

        # ===================================================================
        # Celery Queue Metrics
        # ===================================================================

        self.celery_queue_depth = Gauge(
            'sparktrainer_celery_queue_depth',
            'Number of tasks in Celery queue',
            ['queue_name'],
            registry=self.registry
        )

        self.celery_workers_active = Gauge(
            'sparktrainer_celery_workers_active',
            'Number of active Celery workers',
            registry=self.registry
        )

        self.celery_tasks_total = Counter(
            'sparktrainer_celery_tasks_total',
            'Total number of Celery tasks processed',
            ['task_name', 'status'],
            registry=self.registry
        )

        self.celery_task_duration_seconds = Histogram(
            'sparktrainer_celery_task_duration_seconds',
            'Celery task execution time',
            ['task_name'],
            buckets=(1, 5, 10, 30, 60, 300, 900, 1800, 3600),
            registry=self.registry
        )

        # ===================================================================
        # GPU Metrics
        # ===================================================================

        self.gpu_utilization_percent = Gauge(
            'sparktrainer_gpu_utilization_percent',
            'GPU utilization percentage',
            ['gpu_id', 'gpu_name'],
            registry=self.registry
        )

        self.gpu_memory_used_bytes = Gauge(
            'sparktrainer_gpu_memory_used_bytes',
            'GPU memory used in bytes',
            ['gpu_id', 'gpu_name'],
            registry=self.registry
        )

        self.gpu_memory_total_bytes = Gauge(
            'sparktrainer_gpu_memory_total_bytes',
            'GPU memory total in bytes',
            ['gpu_id', 'gpu_name'],
            registry=self.registry
        )

        self.gpu_temperature_celsius = Gauge(
            'sparktrainer_gpu_temperature_celsius',
            'GPU temperature in Celsius',
            ['gpu_id', 'gpu_name'],
            registry=self.registry
        )

        self.gpu_power_watts = Gauge(
            'sparktrainer_gpu_power_watts',
            'GPU power consumption in watts',
            ['gpu_id', 'gpu_name'],
            registry=self.registry
        )

        self.gpu_compute_processes = Gauge(
            'sparktrainer_gpu_compute_processes',
            'Number of compute processes on GPU',
            ['gpu_id'],
            registry=self.registry
        )

        # ===================================================================
        # System Metrics
        # ===================================================================

        self.system_cpu_percent = Gauge(
            'sparktrainer_system_cpu_percent',
            'System CPU utilization percentage',
            registry=self.registry
        )

        self.system_memory_used_bytes = Gauge(
            'sparktrainer_system_memory_used_bytes',
            'System memory used in bytes',
            registry=self.registry
        )

        self.system_memory_total_bytes = Gauge(
            'sparktrainer_system_memory_total_bytes',
            'System memory total in bytes',
            registry=self.registry
        )

        self.system_disk_used_bytes = Gauge(
            'sparktrainer_system_disk_used_bytes',
            'Disk space used in bytes',
            ['mount_point'],
            registry=self.registry
        )

        self.system_disk_total_bytes = Gauge(
            'sparktrainer_system_disk_total_bytes',
            'Disk space total in bytes',
            ['mount_point'],
            registry=self.registry
        )

        self.system_network_bytes_sent = Counter(
            'sparktrainer_system_network_bytes_sent_total',
            'Total network bytes sent',
            registry=self.registry
        )

        self.system_network_bytes_recv = Counter(
            'sparktrainer_system_network_bytes_recv_total',
            'Total network bytes received',
            registry=self.registry
        )

        # ===================================================================
        # Dataset Metrics
        # ===================================================================

        self.datasets_total = Gauge(
            'sparktrainer_datasets_total',
            'Total number of datasets',
            ['type'],
            registry=self.registry
        )

        self.dataset_size_bytes = Gauge(
            'sparktrainer_dataset_size_bytes',
            'Dataset size in bytes',
            ['dataset_name', 'type'],
            registry=self.registry
        )

        self.dataset_samples_total = Gauge(
            'sparktrainer_dataset_samples_total',
            'Total number of samples in dataset',
            ['dataset_name', 'type'],
            registry=self.registry
        )

        self.dataset_ingestion_duration_seconds = Histogram(
            'sparktrainer_dataset_ingestion_duration_seconds',
            'Dataset ingestion duration',
            ['type'],
            buckets=(10, 60, 300, 900, 1800, 3600, 7200),
            registry=self.registry
        )

        # ===================================================================
        # Model Metrics
        # ===================================================================

        self.models_total = Gauge(
            'sparktrainer_models_total',
            'Total number of models',
            ['family', 'modality'],
            registry=self.registry
        )

        self.model_parameters_total = Gauge(
            'sparktrainer_model_parameters_total',
            'Total model parameters',
            ['model_id', 'model_name'],
            registry=self.registry
        )

        self.adapters_total = Gauge(
            'sparktrainer_adapters_total',
            'Total number of LoRA adapters',
            ['base_model'],
            registry=self.registry
        )

        # ===================================================================
        # MLflow Metrics
        # ===================================================================

        self.experiments_total = Gauge(
            'sparktrainer_experiments_total',
            'Total number of MLflow experiments',
            registry=self.registry
        )

        self.experiment_runs_total = Counter(
            'sparktrainer_experiment_runs_total',
            'Total number of experiment runs',
            ['experiment_name', 'status'],
            registry=self.registry
        )

        self.mlflow_artifact_size_bytes = Gauge(
            'sparktrainer_mlflow_artifact_size_bytes',
            'Total size of MLflow artifacts',
            registry=self.registry
        )

        # ===================================================================
        # API Metrics
        # ===================================================================

        self.api_requests_total = Counter(
            'sparktrainer_api_requests_total',
            'Total API requests',
            ['method', 'endpoint', 'status'],
            registry=self.registry
        )

        self.api_request_duration_seconds = Histogram(
            'sparktrainer_api_request_duration_seconds',
            'API request duration',
            ['method', 'endpoint'],
            buckets=(0.001, 0.01, 0.1, 0.5, 1, 2, 5, 10, 30),
            registry=self.registry
        )

        # ===================================================================
        # Training Throughput Metrics
        # ===================================================================

        self.training_throughput_samples_per_second = Gauge(
            'sparktrainer_training_throughput_samples_per_second',
            'Training throughput in samples per second',
            ['job_id', 'recipe'],
            registry=self.registry
        )

        self.training_throughput_tokens_per_second = Gauge(
            'sparktrainer_training_throughput_tokens_per_second',
            'Training throughput in tokens per second',
            ['job_id', 'recipe'],
            registry=self.registry
        )

        # Store last network stats for delta calculation
        self._last_network_stats = None

        if app:
            self.init_app(app)

    def init_app(self, app):
        """
        Initialize metrics with Flask app.

        Args:
            app: Flask application instance
        """
        # Add metrics endpoint
        @app.route('/metrics')
        def metrics():
            """Prometheus metrics endpoint"""
            # Update metrics before serving
            self.update_system_metrics()
            self.update_gpu_metrics()

            return Response(
                generate_latest(self.registry),
                mimetype=CONTENT_TYPE_LATEST
            )

        # Initialize PrometheusMetrics for automatic API metrics
        PrometheusMetrics(
            app,
            registry=self.registry,
            group_by='endpoint'
        )

    def update_gpu_metrics(self):
        """Update GPU metrics from NVML"""
        if not self.nvml_initialized:
            return

        try:
            device_count = pynvml.nvmlDeviceGetCount()

            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(handle)

                # Utilization
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                self.gpu_utilization_percent.labels(
                    gpu_id=str(i),
                    gpu_name=name
                ).set(util.gpu)

                # Memory
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                self.gpu_memory_used_bytes.labels(
                    gpu_id=str(i),
                    gpu_name=name
                ).set(mem_info.used)

                self.gpu_memory_total_bytes.labels(
                    gpu_id=str(i),
                    gpu_name=name
                ).set(mem_info.total)

                # Temperature
                try:
                    temp = pynvml.nvmlDeviceGetTemperature(
                        handle, pynvml.NVML_TEMPERATURE_GPU
                    )
                    self.gpu_temperature_celsius.labels(
                        gpu_id=str(i),
                        gpu_name=name
                    ).set(temp)
                except:
                    pass

                # Power
                try:
                    power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                    self.gpu_power_watts.labels(
                        gpu_id=str(i),
                        gpu_name=name
                    ).set(power)
                except:
                    pass

                # Process count
                try:
                    processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                    self.gpu_compute_processes.labels(
                        gpu_id=str(i)
                    ).set(len(processes))
                except:
                    pass

        except Exception as e:
            print(f"Error updating GPU metrics: {e}")

    def update_system_metrics(self):
        """Update system-level metrics"""
        try:
            # CPU
            self.system_cpu_percent.set(psutil.cpu_percent(interval=0.1))

            # Memory
            mem = psutil.virtual_memory()
            self.system_memory_used_bytes.set(mem.used)
            self.system_memory_total_bytes.set(mem.total)

            # Disk
            disk = psutil.disk_usage('/')
            self.system_disk_used_bytes.labels(mount_point='/').set(disk.used)
            self.system_disk_total_bytes.labels(mount_point='/').set(disk.total)

            # Network
            net = psutil.net_io_counters()
            if self._last_network_stats:
                bytes_sent_delta = net.bytes_sent - self._last_network_stats.bytes_sent
                bytes_recv_delta = net.bytes_recv - self._last_network_stats.bytes_recv

                if bytes_sent_delta > 0:
                    self.system_network_bytes_sent.inc(bytes_sent_delta)
                if bytes_recv_delta > 0:
                    self.system_network_bytes_recv.inc(bytes_recv_delta)

            self._last_network_stats = net

        except Exception as e:
            print(f"Error updating system metrics: {e}")

    def record_job_start(self, job_id: str, recipe: str):
        """Record job start event"""
        self.jobs_total.labels(recipe=recipe, status='started').inc()
        self.jobs_active.inc()

    def record_job_complete(self, job_id: str, recipe: str, duration: float, status: str):
        """Record job completion event"""
        self.jobs_total.labels(recipe=recipe, status=status).inc()
        self.jobs_active.dec()
        self.job_duration_seconds.labels(recipe=recipe, status=status).observe(duration)

    def update_job_metrics(self, job_id: str, job_name: str, metrics: Dict[str, Any]):
        """Update metrics for a running job"""
        if 'loss' in metrics:
            self.job_training_loss.labels(
                job_id=job_id,
                job_name=job_name
            ).set(metrics['loss'])

        if 'accuracy' in metrics:
            self.job_training_accuracy.labels(
                job_id=job_id,
                job_name=job_name
            ).set(metrics['accuracy'])

        if 'throughput_samples_per_sec' in metrics:
            recipe = metrics.get('recipe', 'unknown')
            self.training_throughput_samples_per_second.labels(
                job_id=job_id,
                recipe=recipe
            ).set(metrics['throughput_samples_per_sec'])

    def update_celery_metrics(self, queue_stats: Dict[str, Any]):
        """Update Celery worker metrics"""
        for queue_name, depth in queue_stats.get('queues', {}).items():
            self.celery_queue_depth.labels(queue_name=queue_name).set(depth)

        active_workers = queue_stats.get('active_workers', 0)
        self.celery_workers_active.set(active_workers)

    def __del__(self):
        """Cleanup NVML on deletion"""
        if self.nvml_initialized:
            try:
                pynvml.nvmlShutdown()
            except:
                pass


# Global metrics instance
metrics = None


def init_metrics(app):
    """
    Initialize Prometheus metrics for the application.

    Args:
        app: Flask application instance

    Returns:
        SparkTrainerMetrics instance
    """
    global metrics
    metrics = SparkTrainerMetrics(app)

    # Start background metrics update thread
    def update_metrics_loop():
        while True:
            try:
                metrics.update_system_metrics()
                metrics.update_gpu_metrics()
            except Exception as e:
                print(f"Error in metrics update loop: {e}")
            time.sleep(15)  # Update every 15 seconds

    metrics_thread = threading.Thread(target=update_metrics_loop, daemon=True)
    metrics_thread.start()

    print("Prometheus metrics initialized at /metrics")
    return metrics


def get_metrics():
    """Get the global metrics instance"""
    return metrics
