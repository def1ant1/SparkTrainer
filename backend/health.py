"""
Health Check Module for SparkTrainer

Implements comprehensive health and readiness checks for all dependencies:
- Database (PostgreSQL)
- Redis
- Celery workers
- MLflow
- Disk space
- GPU availability
"""

import os
import psutil
import subprocess
from typing import Dict, Any, Tuple
from datetime import datetime


def check_database() -> Tuple[bool, str]:
    """Check PostgreSQL database connection"""
    try:
        from sqlalchemy import create_engine, text

        db_url = os.environ.get(
            'DATABASE_URL',
            'postgresql://sparktrainer:sparktrainer@localhost:5432/sparktrainer'
        )

        engine = create_engine(db_url)
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            result.fetchone()

        return True, "Connected"
    except Exception as e:
        return False, f"Database error: {str(e)}"


def check_redis() -> Tuple[bool, str]:
    """Check Redis connection"""
    try:
        import redis

        redis_url = os.environ.get('REDIS_URL', 'redis://localhost:6379/0')
        r = redis.from_url(redis_url)
        r.ping()

        return True, "Connected"
    except Exception as e:
        return False, f"Redis error: {str(e)}"


def check_celery() -> Tuple[bool, str]:
    """Check Celery workers"""
    try:
        from celery import Celery

        redis_url = os.environ.get('REDIS_URL', 'redis://localhost:6379/0')
        celery_app = Celery('tasks', broker=redis_url)

        # Inspect active workers
        inspect = celery_app.control.inspect()
        active_workers = inspect.active()

        if active_workers:
            worker_count = len(active_workers)
            return True, f"{worker_count} workers active"
        else:
            return False, "No active workers"
    except Exception as e:
        return False, f"Celery error: {str(e)}"


def check_mlflow() -> Tuple[bool, str]:
    """Check MLflow tracking server"""
    try:
        import requests

        mlflow_url = os.environ.get('MLFLOW_TRACKING_URI', 'http://localhost:5000')
        response = requests.get(f"{mlflow_url}/health", timeout=5)

        if response.status_code == 200:
            return True, "Connected"
        else:
            return False, f"MLflow returned status {response.status_code}"
    except Exception as e:
        return False, f"MLflow error: {str(e)}"


def check_disk_space() -> Tuple[bool, str]:
    """Check available disk space"""
    try:
        disk = psutil.disk_usage('/')
        percent_used = disk.percent
        free_gb = disk.free / (1024 ** 3)

        # Warn if less than 10% free or less than 10GB
        if percent_used > 90 or free_gb < 10:
            return False, f"Low disk space: {free_gb:.1f}GB free ({100-percent_used:.1f}% available)"
        else:
            return True, f"{free_gb:.1f}GB free ({100-percent_used:.1f}% available)"
    except Exception as e:
        return False, f"Disk check error: {str(e)}"


def check_gpu() -> Tuple[bool, str]:
    """Check GPU availability"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=count', '--format=csv,noheader'],
            capture_output=True,
            text=True,
            timeout=5
        )

        if result.returncode == 0:
            gpu_count = int(result.stdout.strip())
            return True, f"{gpu_count} GPUs available"
        else:
            return False, "nvidia-smi command failed"
    except FileNotFoundError:
        return False, "nvidia-smi not found (no GPU support)"
    except Exception as e:
        return False, f"GPU check error: {str(e)}"


def get_system_metrics() -> Dict[str, Any]:
    """Get current system metrics"""
    try:
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        network = psutil.net_io_counters()

        return {
            "cpu": {
                "percent": cpu_percent,
                "count": psutil.cpu_count(),
            },
            "memory": {
                "total": memory.total,
                "available": memory.available,
                "percent": memory.percent,
            },
            "disk": {
                "total": disk.total,
                "used": disk.used,
                "free": disk.free,
                "percent": disk.percent,
            },
            "network": {
                "bytes_sent": network.bytes_sent,
                "bytes_recv": network.bytes_recv,
                "packets_sent": network.packets_sent,
                "packets_recv": network.packets_recv,
            }
        }
    except Exception as e:
        return {"error": str(e)}


def healthz() -> Tuple[Dict[str, Any], int]:
    """
    Basic health check - returns 200 if service is alive

    Returns:
        Tuple of (response_dict, status_code)
    """
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "sparktrainer-api",
        "version": "1.0.0"
    }, 200


def readyz() -> Tuple[Dict[str, Any], int]:
    """
    Readiness check - returns 200 only if all dependencies are ready

    Returns:
        Tuple of (response_dict, status_code)
    """
    checks = {}
    all_healthy = True

    # Check database
    db_ok, db_msg = check_database()
    checks["database"] = {"healthy": db_ok, "message": db_msg}
    if not db_ok:
        all_healthy = False

    # Check Redis
    redis_ok, redis_msg = check_redis()
    checks["redis"] = {"healthy": redis_ok, "message": redis_msg}
    if not redis_ok:
        all_healthy = False

    # Check Celery (warning only, not critical)
    celery_ok, celery_msg = check_celery()
    checks["celery"] = {"healthy": celery_ok, "message": celery_msg}

    # Check MLflow (warning only, not critical)
    mlflow_ok, mlflow_msg = check_mlflow()
    checks["mlflow"] = {"healthy": mlflow_ok, "message": mlflow_msg}

    # Check disk space
    disk_ok, disk_msg = check_disk_space()
    checks["disk"] = {"healthy": disk_ok, "message": disk_msg}

    # Check GPU (warning only, not critical)
    gpu_ok, gpu_msg = check_gpu()
    checks["gpu"] = {"healthy": gpu_ok, "message": gpu_msg}

    response = {
        "status": "ready" if all_healthy else "not_ready",
        "timestamp": datetime.utcnow().isoformat(),
        "checks": checks,
        "overall": all_healthy
    }

    status_code = 200 if all_healthy else 503
    return response, status_code


def livez() -> Tuple[Dict[str, Any], int]:
    """
    Liveness check - returns 200 if process is alive
    Similar to healthz but can include process-specific checks

    Returns:
        Tuple of (response_dict, status_code)
    """
    try:
        process = psutil.Process()
        uptime_seconds = (datetime.now() - datetime.fromtimestamp(process.create_time())).total_seconds()

        return {
            "status": "alive",
            "timestamp": datetime.utcnow().isoformat(),
            "uptime_seconds": uptime_seconds,
            "pid": process.pid,
            "memory_rss_mb": process.memory_info().rss / (1024 ** 2),
        }, 200
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }, 500


# Export functions
__all__ = [
    'healthz',
    'readyz',
    'livez',
    'get_system_metrics',
    'check_database',
    'check_redis',
    'check_celery',
    'check_mlflow',
    'check_disk_space',
    'check_gpu',
]
