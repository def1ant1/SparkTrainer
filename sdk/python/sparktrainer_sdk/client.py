"""
SparkTrainer Python SDK Client
"""

import requests
from typing import Optional, List, Dict, Any, Iterator
from .models import Job, Experiment, Dataset, Model, GPU, Deployment, HPOStudy
from .exceptions import (
    SparkTrainerException,
    AuthenticationError,
    NotFoundError,
    RateLimitError,
    ValidationError,
    ServerError
)


class SparkTrainerClient:
    """
    SparkTrainer API Client

    Example:
        >>> client = SparkTrainerClient(
        ...     base_url="http://localhost:5001",
        ...     api_key="your-api-key"
        ... )
        >>> job = client.jobs.create(
        ...     name="my-job",
        ...     command="python train.py",
        ...     gpu_count=4
        ... )
    """

    def __init__(
        self,
        base_url: str = "http://localhost:5001",
        api_key: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None
    ):
        """
        Initialize SparkTrainer client

        Args:
            base_url: Base URL of the SparkTrainer API
            api_key: API key for authentication (preferred)
            username: Username for password authentication
            password: Password for password authentication
        """
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()

        # Authenticate
        if api_key:
            self.session.headers['Authorization'] = f'Bearer {api_key}'
        elif username and password:
            self._login(username, password)

        # Initialize resource clients
        self.jobs = JobsClient(self)
        self.experiments = ExperimentsClient(self)
        self.datasets = DatasetsClient(self)
        self.models = ModelsClient(self)
        self.gpus = GPUsClient(self)
        self.deployments = DeploymentsClient(self)
        self.hpo = HPOClient(self)

    def _login(self, username: str, password: str):
        """Authenticate with username/password"""
        response = self.session.post(
            f"{self.base_url}/api/auth/login",
            json={"username": username, "password": password}
        )

        if response.status_code == 401:
            raise AuthenticationError("Invalid credentials")

        response.raise_for_status()
        data = response.json()
        self.session.headers['Authorization'] = f'Bearer {data["access_token"]}'

    def _request(
        self,
        method: str,
        path: str,
        **kwargs
    ) -> requests.Response:
        """Make an HTTP request and handle errors"""
        url = f"{self.base_url}{path}"

        try:
            response = self.session.request(method, url, **kwargs)

            # Handle common errors
            if response.status_code == 401:
                raise AuthenticationError("Authentication failed")
            elif response.status_code == 404:
                raise NotFoundError("Resource not found")
            elif response.status_code == 429:
                raise RateLimitError("Rate limit exceeded")
            elif response.status_code == 400:
                raise ValidationError(response.json().get('message', 'Validation error'))
            elif response.status_code >= 500:
                raise ServerError(f"Server error: {response.status_code}")

            response.raise_for_status()
            return response

        except requests.exceptions.RequestException as e:
            raise SparkTrainerException(f"Request failed: {str(e)}")


class JobsClient:
    """Client for job operations"""

    def __init__(self, client: SparkTrainerClient):
        self.client = client

    def list(
        self,
        status: Optional[str] = None,
        limit: int = 20,
        offset: int = 0
    ) -> List[Job]:
        """List jobs"""
        params = {"limit": limit, "offset": offset}
        if status:
            params["status"] = status

        response = self.client._request("GET", "/api/jobs", params=params)
        return [Job.from_dict(job) for job in response.json()]

    def create(
        self,
        name: str,
        command: str,
        gpu_count: int = 1,
        priority: int = 0,
        environment: Optional[Dict[str, str]] = None
    ) -> Job:
        """Create a new job"""
        data = {
            "name": name,
            "command": command,
            "gpu_count": gpu_count,
            "priority": priority,
        }
        if environment:
            data["environment"] = environment

        response = self.client._request("POST", "/api/jobs", json=data)
        return Job.from_dict(response.json())

    def get(self, job_id: str) -> Job:
        """Get job by ID"""
        response = self.client._request("GET", f"/api/jobs/{job_id}")
        return Job.from_dict(response.json())

    def cancel(self, job_id: str) -> None:
        """Cancel a job"""
        self.client._request("POST", f"/api/jobs/{job_id}/cancel")

    def delete(self, job_id: str) -> None:
        """Delete a job"""
        self.client._request("DELETE", f"/api/jobs/{job_id}")

    def logs(self, job_id: str, stream: str = "stdout") -> str:
        """Get job logs"""
        response = self.client._request(
            "GET",
            f"/api/jobs/{job_id}/logs",
            params={"stream": stream}
        )
        return response.text

    def stream_metrics(self, job_id: str) -> Iterator[Dict[str, Any]]:
        """
        Stream job metrics in real-time using SSE

        Yields:
            Dict containing metrics (step, loss, lr, etc.)
        """
        response = self.client._request(
            "GET",
            f"/api/jobs/{job_id}/stream",
            stream=True
        )

        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    import json
                    yield json.loads(line[6:])


class ExperimentsClient:
    """Client for experiment operations"""

    def __init__(self, client: SparkTrainerClient):
        self.client = client

    def list(self) -> List[Experiment]:
        """List experiments"""
        response = self.client._request("GET", "/api/experiments")
        return [Experiment.from_dict(exp) for exp in response.json()]

    def create(
        self,
        name: str,
        description: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> Experiment:
        """Create a new experiment"""
        data = {"name": name}
        if description:
            data["description"] = description
        if tags:
            data["tags"] = tags

        response = self.client._request("POST", "/api/experiments", json=data)
        return Experiment.from_dict(response.json())

    def get(self, experiment_id: str) -> Experiment:
        """Get experiment by ID"""
        response = self.client._request("GET", f"/api/experiments/{experiment_id}")
        return Experiment.from_dict(response.json())


class DatasetsClient:
    """Client for dataset operations"""

    def __init__(self, client: SparkTrainerClient):
        self.client = client

    def list(self) -> List[Dataset]:
        """List datasets"""
        response = self.client._request("GET", "/api/datasets")
        return [Dataset.from_dict(ds) for ds in response.json()]

    def create(self, name: str, file_path: str) -> Dataset:
        """Upload a new dataset"""
        with open(file_path, 'rb') as f:
            files = {'file': f}
            data = {'name': name}
            response = self.client._request(
                "POST",
                "/api/datasets",
                files=files,
                data=data
            )
        return Dataset.from_dict(response.json())

    def get(self, dataset_id: str) -> Dataset:
        """Get dataset by ID"""
        response = self.client._request("GET", f"/api/datasets/{dataset_id}")
        return Dataset.from_dict(response.json())


class ModelsClient:
    """Client for model operations"""

    def __init__(self, client: SparkTrainerClient):
        self.client = client

    def list(self) -> List[Model]:
        """List models"""
        response = self.client._request("GET", "/api/models")
        return [Model.from_dict(model) for model in response.json()]

    def get(self, model_id: str) -> Model:
        """Get model by ID"""
        response = self.client._request("GET", f"/api/models/{model_id}")
        return Model.from_dict(response.json())


class GPUsClient:
    """Client for GPU operations"""

    def __init__(self, client: SparkTrainerClient):
        self.client = client

    def list(self) -> List[GPU]:
        """List GPU status"""
        response = self.client._request("GET", "/api/gpus")
        return [GPU.from_dict(gpu) for gpu in response.json()]


class DeploymentsClient:
    """Client for deployment operations"""

    def __init__(self, client: SparkTrainerClient):
        self.client = client

    def list(self) -> List[Deployment]:
        """List deployments"""
        response = self.client._request("GET", "/api/deployments")
        return [Deployment.from_dict(dep) for dep in response.json()]

    def create(
        self,
        name: str,
        model_id: str,
        backend: str,
        replicas: int = 1,
        gpu_count: int = 1
    ) -> Deployment:
        """Create a new deployment"""
        data = {
            "name": name,
            "model_id": model_id,
            "backend": backend,
            "replicas": replicas,
            "gpu_count": gpu_count
        }
        response = self.client._request("POST", "/api/deployments", json=data)
        return Deployment.from_dict(response.json())

    def get(self, deployment_id: str) -> Deployment:
        """Get deployment by ID"""
        response = self.client._request("GET", f"/api/deployments/{deployment_id}")
        return Deployment.from_dict(response.json())

    def delete(self, deployment_id: str) -> None:
        """Stop and delete a deployment"""
        self.client._request("DELETE", f"/api/deployments/{deployment_id}")


class HPOClient:
    """Client for hyperparameter optimization"""

    def __init__(self, client: SparkTrainerClient):
        self.client = client

    def list_studies(self) -> List[HPOStudy]:
        """List HPO studies"""
        response = self.client._request("GET", "/api/hpo/studies")
        return [HPOStudy.from_dict(study) for study in response.json()]

    def create_study(
        self,
        name: str,
        objective: str,
        search_space: Dict[str, Any],
        n_trials: int = 100,
        parallelism: int = 1
    ) -> HPOStudy:
        """Create a new HPO study"""
        data = {
            "name": name,
            "objective": objective,
            "search_space": search_space,
            "n_trials": n_trials,
            "parallelism": parallelism
        }
        response = self.client._request("POST", "/api/hpo/studies", json=data)
        return HPOStudy.from_dict(response.json())

    def get_study(self, study_id: str) -> HPOStudy:
        """Get study by ID"""
        response = self.client._request("GET", f"/api/hpo/studies/{study_id}")
        return HPOStudy.from_dict(response.json())
