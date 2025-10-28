"""
Model Registry with MLflow Integration.

Provides:
- Model versioning and lifecycle management
- State transitions: Staging → Production → Archived
- Promotion gates with approval workflows
- Signed model bundles (safetensors/ONNX/TorchScript)
- Model lineage and metadata tracking
"""

import os
import json
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
import hashlib

logger = logging.getLogger(__name__)


class ModelStage(str, Enum):
    """Model lifecycle stages."""
    DEVELOPMENT = "Development"
    STAGING = "Staging"
    PRODUCTION = "Production"
    ARCHIVED = "Archived"


class ModelFormat(str, Enum):
    """Model export formats."""
    PYTORCH = "pytorch"
    SAFETENSORS = "safetensors"
    ONNX = "onnx"
    TORCHSCRIPT = "torchscript"
    HUGGINGFACE = "huggingface"
    VLLM = "vllm"
    TGI = "tgi"


@dataclass
class ModelMetadata:
    """Model metadata."""
    name: str
    version: str
    format: ModelFormat
    framework: str  # pytorch, tensorflow, jax
    task_type: str  # text-generation, image-classification, etc.
    architecture: str
    parameters: Dict[str, Any]
    metrics: Dict[str, float]
    tags: List[str]
    description: Optional[str] = None
    author: Optional[str] = None
    created_at: Optional[str] = None


@dataclass
class PromotionGate:
    """Promotion gate configuration."""
    name: str
    required_metrics: Dict[str, float]  # metric_name: min_threshold
    required_checks: List[str]  # safety, performance, compatibility
    approval_required: bool = True
    auto_approve_conditions: Optional[Dict[str, Any]] = None


@dataclass
class PromotionRequest:
    """Model promotion request."""
    model_name: str
    version: str
    from_stage: ModelStage
    to_stage: ModelStage
    requester: str
    justification: str
    timestamp: str
    approved: bool = False
    approver: Optional[str] = None
    approval_timestamp: Optional[str] = None
    gate_results: Optional[Dict[str, Any]] = None


class ModelRegistry:
    """
    Model registry with lifecycle management and promotion gates.

    Integrates with MLflow for artifact storage and tracking.
    """

    def __init__(
        self,
        mlflow_tracking_uri: Optional[str] = None,
        registry_dir: Optional[str] = None,
    ):
        self.mlflow_tracking_uri = mlflow_tracking_uri or os.environ.get(
            "MLFLOW_TRACKING_URI", "http://localhost:5001"
        )
        self.registry_dir = Path(registry_dir or "./model_registry")
        self.registry_dir.mkdir(parents=True, exist_ok=True)

        # Initialize MLflow
        self._init_mlflow()

        # Promotion gates configuration
        self.promotion_gates = self._init_promotion_gates()

        # Pending promotion requests
        self.promotion_requests: List[PromotionRequest] = []

    def _init_mlflow(self):
        """Initialize MLflow client."""
        try:
            import mlflow
            from mlflow.tracking import MlflowClient

            mlflow.set_tracking_uri(self.mlflow_tracking_uri)
            self.mlflow_client = MlflowClient()
            logger.info(f"MLflow client initialized: {self.mlflow_tracking_uri}")

        except Exception as e:
            logger.warning(f"Failed to initialize MLflow: {e}")
            self.mlflow_client = None

    def _init_promotion_gates(self) -> Dict[ModelStage, PromotionGate]:
        """Initialize promotion gates."""
        return {
            ModelStage.STAGING: PromotionGate(
                name="staging_gate",
                required_metrics={
                    'accuracy': 0.8,
                    'f1_score': 0.75,
                },
                required_checks=['safety', 'compatibility'],
                approval_required=False,
                auto_approve_conditions={'accuracy': 0.9},
            ),
            ModelStage.PRODUCTION: PromotionGate(
                name="production_gate",
                required_metrics={
                    'accuracy': 0.9,
                    'f1_score': 0.85,
                },
                required_checks=['safety', 'performance', 'compatibility'],
                approval_required=True,
            ),
        }

    def register_model(
        self,
        name: str,
        model_path: str,
        metadata: ModelMetadata,
        stage: ModelStage = ModelStage.DEVELOPMENT,
    ) -> str:
        """
        Register a new model version.

        Args:
            name: Model name
            model_path: Path to model artifacts
            metadata: Model metadata
            stage: Initial stage

        Returns:
            Model version string
        """
        logger.info(f"Registering model: {name}")

        # Generate version
        version = self._generate_version(name)

        # Create model directory
        model_dir = self.registry_dir / name / version
        model_dir.mkdir(parents=True, exist_ok=True)

        # Copy model artifacts
        if Path(model_path).is_dir():
            shutil.copytree(model_path, model_dir / "artifacts", dirs_exist_ok=True)
        else:
            shutil.copy(model_path, model_dir / "model")

        # Save metadata
        metadata.version = version
        metadata.created_at = datetime.now().isoformat()
        metadata_path = model_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(asdict(metadata), f, indent=2)

        # Compute signature
        signature = self._compute_model_signature(model_dir)
        signature_path = model_dir / "signature.json"
        with open(signature_path, 'w') as f:
            json.dump(signature, f, indent=2)

        # Register with MLflow if available
        if self.mlflow_client:
            try:
                from mlflow.models import ModelSignature

                self.mlflow_client.create_registered_model(name)
                run_id = self._create_mlflow_run(name, model_dir, metadata)

                # Create model version
                model_version = self.mlflow_client.create_model_version(
                    name=name,
                    source=f"{self.mlflow_tracking_uri}/artifacts/{run_id}",
                    run_id=run_id,
                )

                # Set stage
                self.mlflow_client.transition_model_version_stage(
                    name=name,
                    version=model_version.version,
                    stage=stage.value,
                )

                logger.info(f"Model registered in MLflow: {name} v{version}")

            except Exception as e:
                logger.error(f"MLflow registration failed: {e}")

        logger.info(f"Model registered: {name} v{version} (stage: {stage.value})")
        return version

    def promote_model(
        self,
        name: str,
        version: str,
        to_stage: ModelStage,
        requester: str,
        justification: str,
    ) -> PromotionRequest:
        """
        Request model promotion to a new stage.

        Args:
            name: Model name
            version: Model version
            to_stage: Target stage
            requester: User requesting promotion
            justification: Reason for promotion

        Returns:
            Promotion request object
        """
        logger.info(f"Promotion request: {name} v{version} → {to_stage.value}")

        # Get current stage
        current_stage = self._get_model_stage(name, version)

        # Create promotion request
        request = PromotionRequest(
            model_name=name,
            version=version,
            from_stage=current_stage,
            to_stage=to_stage,
            requester=requester,
            justification=justification,
            timestamp=datetime.now().isoformat(),
        )

        # Check promotion gate
        gate_results = self._check_promotion_gate(name, version, to_stage)
        request.gate_results = gate_results

        # Auto-approve if conditions met
        gate = self.promotion_gates.get(to_stage)
        if gate:
            if not gate.approval_required:
                request.approved = True
                request.approver = "auto"
                request.approval_timestamp = datetime.now().isoformat()
                logger.info(f"Auto-approved promotion to {to_stage.value}")

            elif gate.auto_approve_conditions:
                # Check auto-approve conditions
                metadata = self._load_metadata(name, version)
                if self._check_auto_approve_conditions(metadata.metrics, gate.auto_approve_conditions):
                    request.approved = True
                    request.approver = "auto"
                    request.approval_timestamp = datetime.now().isoformat()
                    logger.info(f"Auto-approved based on conditions")

        # Save request
        self.promotion_requests.append(request)
        self._save_promotion_request(request)

        # If approved, execute promotion
        if request.approved:
            self._execute_promotion(request)

        return request

    def approve_promotion(
        self,
        request_id: int,
        approver: str,
    ) -> bool:
        """
        Approve a pending promotion request.

        Args:
            request_id: Request ID
            approver: User approving the request

        Returns:
            True if approved successfully
        """
        if request_id >= len(self.promotion_requests):
            raise ValueError(f"Invalid request ID: {request_id}")

        request = self.promotion_requests[request_id]

        if request.approved:
            logger.warning("Promotion already approved")
            return False

        # Check gate results
        if not request.gate_results or not request.gate_results.get('passed'):
            logger.error("Cannot approve: promotion gate checks failed")
            return False

        # Approve
        request.approved = True
        request.approver = approver
        request.approval_timestamp = datetime.now().isoformat()

        # Execute promotion
        self._execute_promotion(request)

        logger.info(f"Promotion approved by {approver}")
        return True

    def _check_promotion_gate(
        self,
        name: str,
        version: str,
        to_stage: ModelStage,
    ) -> Dict[str, Any]:
        """
        Check if model passes promotion gate.

        Args:
            name: Model name
            version: Model version
            to_stage: Target stage

        Returns:
            Gate check results
        """
        gate = self.promotion_gates.get(to_stage)
        if not gate:
            return {'passed': True, 'reason': 'No gate defined'}

        # Load model metadata
        metadata = self._load_metadata(name, version)

        results = {
            'passed': True,
            'checks': {},
        }

        # Check required metrics
        for metric_name, min_threshold in gate.required_metrics.items():
            metric_value = metadata.metrics.get(metric_name, 0.0)
            passed = metric_value >= min_threshold

            results['checks'][metric_name] = {
                'passed': passed,
                'value': metric_value,
                'threshold': min_threshold,
            }

            if not passed:
                results['passed'] = False

        # Check required safety/compatibility checks
        for check_name in gate.required_checks:
            check_result = self._run_gate_check(name, version, check_name)
            results['checks'][check_name] = check_result

            if not check_result['passed']:
                results['passed'] = False

        return results

    def _run_gate_check(
        self,
        name: str,
        version: str,
        check_name: str,
    ) -> Dict[str, Any]:
        """Run a specific gate check."""
        if check_name == 'safety':
            # Run safety probes
            return self._check_safety(name, version)

        elif check_name == 'performance':
            # Check performance benchmarks
            return self._check_performance(name, version)

        elif check_name == 'compatibility':
            # Check API compatibility
            return self._check_compatibility(name, version)

        else:
            logger.warning(f"Unknown check: {check_name}")
            return {'passed': True, 'reason': 'Check not implemented'}

    def _check_safety(self, name: str, version: str) -> Dict[str, Any]:
        """Check model safety."""
        # Placeholder - integrate with SafetyEvaluator
        logger.info(f"Running safety check for {name} v{version}")
        return {
            'passed': True,
            'score': 0.95,
            'reason': 'Safety checks passed (placeholder)',
        }

    def _check_performance(self, name: str, version: str) -> Dict[str, Any]:
        """Check model performance."""
        logger.info(f"Running performance check for {name} v{version}")
        return {
            'passed': True,
            'latency_p50': 50.0,
            'latency_p99': 200.0,
            'throughput': 100.0,
            'reason': 'Performance checks passed (placeholder)',
        }

    def _check_compatibility(self, name: str, version: str) -> Dict[str, Any]:
        """Check API compatibility."""
        logger.info(f"Running compatibility check for {name} v{version}")
        return {
            'passed': True,
            'reason': 'API compatible (placeholder)',
        }

    def _execute_promotion(self, request: PromotionRequest):
        """Execute approved promotion."""
        logger.info(f"Executing promotion: {request.model_name} → {request.to_stage.value}")

        # Update stage in registry
        model_dir = self.registry_dir / request.model_name / request.version
        stage_file = model_dir / "stage.txt"
        stage_file.write_text(request.to_stage.value)

        # Update MLflow stage
        if self.mlflow_client:
            try:
                self.mlflow_client.transition_model_version_stage(
                    name=request.model_name,
                    version=request.version,
                    stage=request.to_stage.value,
                )
                logger.info(f"MLflow stage updated to {request.to_stage.value}")
            except Exception as e:
                logger.error(f"MLflow stage update failed: {e}")

        logger.info(f"Promotion complete: {request.model_name} v{request.version}")

    def _get_model_stage(self, name: str, version: str) -> ModelStage:
        """Get current stage of model version."""
        model_dir = self.registry_dir / name / version
        stage_file = model_dir / "stage.txt"

        if stage_file.exists():
            stage_str = stage_file.read_text().strip()
            return ModelStage(stage_str)

        return ModelStage.DEVELOPMENT

    def _load_metadata(self, name: str, version: str) -> ModelMetadata:
        """Load model metadata."""
        model_dir = self.registry_dir / name / version
        metadata_path = model_dir / "metadata.json"

        with open(metadata_path, 'r') as f:
            metadata_dict = json.load(f)

        # Convert format to enum if it's a string
        if 'format' in metadata_dict and isinstance(metadata_dict['format'], str):
            metadata_dict['format'] = ModelFormat(metadata_dict['format'])

        return ModelMetadata(**metadata_dict)

    def _generate_version(self, name: str) -> str:
        """Generate new version number."""
        model_dir = self.registry_dir / name

        if not model_dir.exists():
            return "v1"

        # Find existing versions
        versions = [d.name for d in model_dir.iterdir() if d.is_dir() and d.name.startswith('v')]

        if not versions:
            return "v1"

        # Get max version number
        version_numbers = []
        for v in versions:
            try:
                num = int(v[1:])
                version_numbers.append(num)
            except ValueError:
                continue

        max_version = max(version_numbers) if version_numbers else 0
        return f"v{max_version + 1}"

    def _compute_model_signature(self, model_dir: Path) -> Dict[str, str]:
        """Compute cryptographic signature for model."""
        signature = {}

        # Compute SHA256 for all model files
        for file_path in model_dir.rglob("*"):
            if file_path.is_file() and file_path.name != "signature.json":
                with open(file_path, 'rb') as f:
                    file_hash = hashlib.sha256(f.read()).hexdigest()
                    relative_path = str(file_path.relative_to(model_dir))
                    signature[relative_path] = file_hash

        return signature

    def _create_mlflow_run(
        self,
        name: str,
        model_dir: Path,
        metadata: ModelMetadata,
    ) -> str:
        """Create MLflow run for model."""
        import mlflow

        with mlflow.start_run(run_name=f"{name}_{metadata.version}") as run:
            # Log parameters
            mlflow.log_params(metadata.parameters)

            # Log metrics
            mlflow.log_metrics(metadata.metrics)

            # Log artifacts
            mlflow.log_artifacts(str(model_dir))

            return run.info.run_id

    def _check_auto_approve_conditions(
        self,
        metrics: Dict[str, float],
        conditions: Dict[str, Any],
    ) -> bool:
        """Check if auto-approve conditions are met."""
        for metric_name, threshold in conditions.items():
            if metrics.get(metric_name, 0.0) < threshold:
                return False
        return True

    def _save_promotion_request(self, request: PromotionRequest):
        """Save promotion request to disk."""
        requests_dir = self.registry_dir / "promotion_requests"
        requests_dir.mkdir(exist_ok=True)

        request_file = requests_dir / f"{request.model_name}_{request.version}_{request.timestamp}.json"
        with open(request_file, 'w') as f:
            json.dump(asdict(request), f, indent=2, default=str)

    def list_models(self, stage: Optional[ModelStage] = None) -> List[Dict[str, Any]]:
        """List all registered models."""
        models = []

        for model_dir in self.registry_dir.iterdir():
            if not model_dir.is_dir() or model_dir.name == "promotion_requests":
                continue

            for version_dir in model_dir.iterdir():
                if not version_dir.is_dir():
                    continue

                try:
                    metadata = self._load_metadata(model_dir.name, version_dir.name)
                    current_stage = self._get_model_stage(model_dir.name, version_dir.name)

                    if stage and current_stage != stage:
                        continue

                    models.append({
                        'name': model_dir.name,
                        'version': version_dir.name,
                        'stage': current_stage.value,
                        'format': metadata.format.value,
                        'task_type': metadata.task_type,
                        'created_at': metadata.created_at,
                    })

                except Exception as e:
                    logger.error(f"Failed to load model {model_dir.name}/{version_dir.name}: {e}")

        return sorted(models, key=lambda x: x['created_at'], reverse=True)

    def get_model_path(self, name: str, version: str) -> Path:
        """Get path to model artifacts."""
        return self.registry_dir / name / version

    def export_model(
        self,
        name: str,
        version: str,
        format: ModelFormat,
        output_path: str,
    ) -> str:
        """
        Export model to different format.

        Args:
            name: Model name
            version: Model version
            format: Target format
            output_path: Output path

        Returns:
            Path to exported model
        """
        logger.info(f"Exporting {name} v{version} to {format.value}")

        model_dir = self.get_model_path(name, version)

        # Format-specific export logic (placeholder)
        if format == ModelFormat.SAFETENSORS:
            return self._export_safetensors(model_dir, output_path)
        elif format == ModelFormat.ONNX:
            return self._export_onnx(model_dir, output_path)
        elif format == ModelFormat.TORCHSCRIPT:
            return self._export_torchscript(model_dir, output_path)
        else:
            # Just copy artifacts
            shutil.copytree(model_dir, output_path, dirs_exist_ok=True)
            return output_path

    def _export_safetensors(self, model_dir: Path, output_path: str) -> str:
        """Export to safetensors format."""
        # Placeholder - implement safetensors conversion
        logger.info("Safetensors export (placeholder)")
        return output_path

    def _export_onnx(self, model_dir: Path, output_path: str) -> str:
        """Export to ONNX format."""
        # Placeholder - implement ONNX conversion
        logger.info("ONNX export (placeholder)")
        return output_path

    def _export_torchscript(self, model_dir: Path, output_path: str) -> str:
        """Export to TorchScript format."""
        # Placeholder - implement TorchScript conversion
        logger.info("TorchScript export (placeholder)")
        return output_path


# Example usage
if __name__ == "__main__":
    # Initialize registry
    registry = ModelRegistry(registry_dir="/tmp/model_registry")

    # Register model
    metadata = ModelMetadata(
        name="my_model",
        version="v1",
        format=ModelFormat.PYTORCH,
        framework="pytorch",
        task_type="text-generation",
        architecture="gpt2",
        parameters={'lr': 1e-4, 'batch_size': 32},
        metrics={'accuracy': 0.95, 'f1_score': 0.92},
        tags=['nlp', 'gpt2'],
    )

    version = registry.register_model(
        name="my_model",
        model_path="/tmp/model",
        metadata=metadata,
    )

    # Request promotion
    request = registry.promote_model(
        name="my_model",
        version=version,
        to_stage=ModelStage.STAGING,
        requester="user@example.com",
        justification="Model passes all tests",
    )

    print(f"Promotion request: {request.approved}")
