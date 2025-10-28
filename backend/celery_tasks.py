"""
Celery tasks for training, preprocessing, and evaluation.
"""
import os
import sys
import json
import subprocess
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

import mlflow
from celery import Task, current_task
from celery.signals import task_prerun, task_postrun, task_failure

from celery_app import celery
from database import get_db
from models import Job, Experiment, Artifact, JobStatus


class MLflowTask(Task):
    """Base task class with MLflow integration."""

    def __init__(self):
        super().__init__()
        self._mlflow_client = None

    @property
    def mlflow_client(self):
        """Lazy initialization of MLflow client."""
        if self._mlflow_client is None:
            mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
            mlflow.set_tracking_uri(mlflow_uri)
            self._mlflow_client = mlflow.tracking.MlflowClient()
        return self._mlflow_client

    def before_start(self, task_id, args, kwargs):
        """Called before task execution."""
        pass

    def on_success(self, retval, task_id, args, kwargs):
        """Called on successful task completion."""
        pass

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Called on task failure."""
        pass


@celery.task(base=MLflowTask, bind=True, name="celery_tasks.train_model")
def train_model(self, job_id: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute training job with MLflow tracking.

    Args:
        job_id: Job ID from database
        config: Training configuration

    Returns:
        Result dictionary with status and metrics
    """
    with get_db() as db:
        # Get job and experiment
        job = db.query(Job).filter(Job.id == job_id).first()
        if not job:
            raise ValueError(f"Job {job_id} not found")

        experiment = db.query(Experiment).filter(Experiment.id == job.experiment_id).first()
        if not experiment:
            raise ValueError(f"Experiment {job.experiment_id} not found")

        try:
            # Transition to running
            job.transition_to(db, JobStatus.RUNNING, reason="Task started by Celery worker")
            job.celery_task_id = self.request.id
            db.commit()

            # Create or get MLflow experiment
            mlflow_exp_name = f"{experiment.project_id}/{experiment.name}"
            mlflow_exp = mlflow.get_experiment_by_name(mlflow_exp_name)
            if mlflow_exp is None:
                mlflow_exp_id = mlflow.create_experiment(mlflow_exp_name)
            else:
                mlflow_exp_id = mlflow_exp.experiment_id

            # Start MLflow run
            with mlflow.start_run(experiment_id=mlflow_exp_id, run_name=f"job_{job_id}") as run:
                mlflow_run_id = run.info.run_id

                # Update experiment with MLflow IDs
                experiment.mlflow_run_id = mlflow_run_id
                experiment.mlflow_experiment_id = mlflow_exp_id
                experiment.status = JobStatus.RUNNING
                experiment.started_at = datetime.utcnow()
                db.commit()

                # Log parameters
                mlflow.log_params(config.get("hyperparameters", {}))
                mlflow.set_tags({
                    "job_id": job_id,
                    "experiment_id": experiment.id,
                    "model_type": experiment.model_type or "unknown",
                    "recipe": experiment.recipe_name or "unknown"
                })

                # Prepare training command
                train_script = config.get("train_script", "training_scripts/train_huggingface.py")
                config_path = f"/tmp/train_config_{job_id}.json"

                # Write config to file
                with open(config_path, "w") as f:
                    json.dump(config, f, indent=2)

                # Build command
                command = [
                    sys.executable,
                    train_script,
                    "--config", config_path,
                    "--mlflow-run-id", mlflow_run_id,
                    "--job-id", job_id
                ]

                # Add GPU selection if specified
                if job.gpu_ids:
                    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, job.gpu_ids))

                # Setup log file
                log_dir = Path(config.get("log_dir", "/app/logs"))
                log_dir.mkdir(parents=True, exist_ok=True)
                log_file = log_dir / f"job_{job_id}.log"
                job.log_file = str(log_file)
                db.commit()

                # Execute training
                print(f"Starting training: {' '.join(command)}")
                with open(log_file, "w") as log_f:
                    process = subprocess.Popen(
                        command,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        universal_newlines=True,
                        bufsize=1
                    )

                    # Stream output
                    for line in process.stdout:
                        log_f.write(line)
                        log_f.flush()

                        # Update progress from logs (basic parsing)
                        if "epoch" in line.lower() or "step" in line.lower():
                            # Parse and update progress
                            # This is a simplified version - actual implementation would be more robust
                            pass

                        # Update task state for monitoring
                        self.update_state(
                            state="PROGRESS",
                            meta={"job_id": job_id, "status": "training"}
                        )

                    # Wait for completion
                    return_code = process.wait()

                    if return_code != 0:
                        raise RuntimeError(f"Training failed with exit code {return_code}")

                # Log final metrics
                final_metrics = config.get("final_metrics", {})
                for key, value in final_metrics.items():
                    mlflow.log_metric(key, value)

                # Log model artifacts
                model_path = config.get("output_dir", f"/app/models/{experiment.id}")
                if Path(model_path).exists():
                    mlflow.log_artifacts(model_path, artifact_path="model")

                    # Create artifact record
                    artifact = Artifact(
                        id=f"artifact_{job_id}_{datetime.utcnow().timestamp()}",
                        experiment_id=experiment.id,
                        artifact_type="model",
                        name=f"{experiment.name}_final",
                        file_path=model_path,
                        mlflow_artifact_path=f"runs:/{mlflow_run_id}/model",
                        storage_backend="local"
                    )
                    db.add(artifact)

                # Mark as completed
                job.transition_to(db, JobStatus.COMPLETED, reason="Training completed successfully")
                job.progress = 100.0
                job.completed_at = datetime.utcnow()

                experiment.status = JobStatus.COMPLETED
                experiment.completed_at = datetime.utcnow()
                experiment.model_path = model_path

                db.commit()

                return {
                    "status": "completed",
                    "job_id": job_id,
                    "mlflow_run_id": mlflow_run_id,
                    "model_path": model_path
                }

        except Exception as e:
            # Mark as failed
            error_msg = f"{str(e)}\n{traceback.format_exc()}"
            job.transition_to(db, JobStatus.FAILED, reason="Training failed", metadata={"error": error_msg})
            job.error_message = error_msg
            job.completed_at = datetime.utcnow()

            experiment.status = JobStatus.FAILED
            experiment.completed_at = datetime.utcnow()

            db.commit()

            raise


@celery.task(base=MLflowTask, bind=True, name="celery_tasks.preprocess_data")
def preprocess_data(self, dataset_id: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Preprocess dataset (video extraction, transcription, etc.).

    Args:
        dataset_id: Dataset ID from database
        config: Preprocessing configuration

    Returns:
        Result dictionary with preprocessing stats
    """
    from database import get_db
    from models import Dataset

    with get_db() as db:
        dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
        if not dataset:
            raise ValueError(f"Dataset {dataset_id} not found")

        try:
            # Get preprocessing parameters
            input_path = config.get("input_path")
            output_path = config.get("output_path", f"/app/datasets/{dataset_id}")
            modality = config.get("modality", "video")

            # Setup output directory
            output_dir = Path(output_path)
            output_dir.mkdir(parents=True, exist_ok=True)

            # Build preprocessing command
            preprocess_cmd = [
                sys.executable,
                "-m", "spark_trainer.cli",
                "preprocess",
                "--video-dir", input_path,
                "--output-dir", output_path,
            ]

            # Add optional parameters
            if "fps" in config:
                preprocess_cmd.extend(["--fps", str(config["fps"])])
            if "resolution" in config:
                preprocess_cmd.extend(["--resolution", config["resolution"]])
            if "captioner_backend" in config:
                preprocess_cmd.extend(["--captioner-backend", config["captioner_backend"]])

            # Execute preprocessing
            print(f"Starting preprocessing: {' '.join(preprocess_cmd)}")
            result = subprocess.run(
                preprocess_cmd,
                capture_output=True,
                text=True,
                check=True
            )

            # Update dataset with results
            manifest_path = output_dir / "manifest.jsonl"
            if manifest_path.exists():
                dataset.manifest_path = str(manifest_path)

                # Count samples
                with open(manifest_path) as f:
                    num_samples = sum(1 for _ in f)
                dataset.num_samples = num_samples

            dataset.storage_path = output_path
            dataset.integrity_checked = True
            dataset.integrity_passed = True

            db.commit()

            return {
                "status": "completed",
                "dataset_id": dataset_id,
                "num_samples": dataset.num_samples,
                "output_path": output_path
            }

        except Exception as e:
            dataset.integrity_checked = True
            dataset.integrity_passed = False
            dataset.integrity_report = {"error": str(e)}
            db.commit()
            raise


@celery.task(base=MLflowTask, bind=True, name="celery_tasks.evaluate_model")
def evaluate_model(self, experiment_id: str, benchmark: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluate model on benchmark (MMLU, COCO, etc.).

    Args:
        experiment_id: Experiment ID
        benchmark: Benchmark name (mmlu, coco, glue, etc.)
        config: Evaluation configuration

    Returns:
        Evaluation results
    """
    from database import get_db
    from models import Experiment, Evaluation

    with get_db() as db:
        experiment = db.query(Experiment).filter(Experiment.id == experiment_id).first()
        if not experiment:
            raise ValueError(f"Experiment {experiment_id} not found")

        try:
            # Build evaluation command
            eval_script = f"/app/src/spark_trainer/evaluation/{benchmark}_eval.py"
            model_path = experiment.model_path or config.get("model_path")

            eval_cmd = [
                sys.executable,
                eval_script,
                "--model-path", model_path,
                "--output-dir", f"/app/evaluations/{experiment_id}",
            ]

            # Add benchmark-specific parameters
            if "num_samples" in config:
                eval_cmd.extend(["--num-samples", str(config["num_samples"])])
            if "batch_size" in config:
                eval_cmd.extend(["--batch-size", str(config["batch_size"])])

            # Execute evaluation
            print(f"Starting evaluation: {' '.join(eval_cmd)}")
            result = subprocess.run(
                eval_cmd,
                capture_output=True,
                text=True,
                check=True
            )

            # Parse results (assuming JSON output)
            results_file = Path(f"/app/evaluations/{experiment_id}/results.json")
            if results_file.exists():
                with open(results_file) as f:
                    results = json.load(f)

                # Create evaluation record
                evaluation = Evaluation(
                    id=f"eval_{experiment_id}_{benchmark}_{datetime.utcnow().timestamp()}",
                    experiment_id=experiment_id,
                    benchmark_name=benchmark,
                    score=results.get("score", 0.0),
                    metrics=results.get("metrics", {}),
                    eval_config=config,
                    num_samples=config.get("num_samples"),
                    completed_at=datetime.utcnow()
                )
                db.add(evaluation)

                # Log to MLflow if run exists
                if experiment.mlflow_run_id:
                    with mlflow.start_run(run_id=experiment.mlflow_run_id):
                        mlflow.log_metrics({
                            f"{benchmark}_score": results.get("score", 0.0),
                            **{f"{benchmark}_{k}": v for k, v in results.get("metrics", {}).items()
                               if isinstance(v, (int, float))}
                        })

                db.commit()

                return {
                    "status": "completed",
                    "experiment_id": experiment_id,
                    "benchmark": benchmark,
                    "score": results.get("score", 0.0),
                    "metrics": results.get("metrics", {})
                }

            else:
                raise FileNotFoundError(f"Results file not found: {results_file}")

        except Exception as e:
            raise


# Signal handlers
@task_prerun.connect
def task_prerun_handler(sender=None, task_id=None, task=None, args=None, kwargs=None, **extra):
    """Handler called before task execution."""
    print(f"Task {task.name} [{task_id}] starting...")


@task_postrun.connect
def task_postrun_handler(sender=None, task_id=None, task=None, args=None, kwargs=None,
                        retval=None, state=None, **extra):
    """Handler called after task execution."""
    print(f"Task {task.name} [{task_id}] completed with state: {state}")


@task_failure.connect
def task_failure_handler(sender=None, task_id=None, exception=None, args=None,
                        kwargs=None, traceback=None, einfo=None, **extra):
    """Handler called on task failure."""
    print(f"Task {sender.name} [{task_id}] failed: {exception}")
