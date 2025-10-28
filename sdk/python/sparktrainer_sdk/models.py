"""
SparkTrainer SDK Data Models
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from datetime import datetime


@dataclass
class Job:
    """Training job model"""
    id: str
    name: str
    status: str
    command: str
    gpu_count: int
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    metrics: Optional[Dict[str, Any]] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Job':
        return cls(
            id=data['id'],
            name=data['name'],
            status=data['status'],
            command=data['command'],
            gpu_count=data['gpu_count'],
            created_at=datetime.fromisoformat(data['created_at'].replace('Z', '+00:00')),
            started_at=datetime.fromisoformat(data['started_at'].replace('Z', '+00:00')) if data.get('started_at') else None,
            completed_at=datetime.fromisoformat(data['completed_at'].replace('Z', '+00:00')) if data.get('completed_at') else None,
            metrics=data.get('metrics')
        )


@dataclass
class Experiment:
    """Experiment model"""
    id: str
    name: str
    description: Optional[str]
    created_at: datetime
    runs: List[Dict[str, Any]]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Experiment':
        return cls(
            id=data['id'],
            name=data['name'],
            description=data.get('description'),
            created_at=datetime.fromisoformat(data['created_at'].replace('Z', '+00:00')),
            runs=data.get('runs', [])
        )


@dataclass
class Dataset:
    """Dataset model"""
    id: str
    name: str
    size: int
    format: str
    created_at: datetime
    version: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Dataset':
        return cls(
            id=data['id'],
            name=data['name'],
            size=data['size'],
            format=data['format'],
            created_at=datetime.fromisoformat(data['created_at'].replace('Z', '+00:00')),
            version=data.get('version')
        )


@dataclass
class Model:
    """Model model"""
    id: str
    name: str
    framework: str
    size: int
    created_at: datetime
    tags: List[str]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Model':
        return cls(
            id=data['id'],
            name=data['name'],
            framework=data['framework'],
            size=data['size'],
            created_at=datetime.fromisoformat(data['created_at'].replace('Z', '+00:00')),
            tags=data.get('tags', [])
        )


@dataclass
class GPU:
    """GPU model"""
    id: int
    name: str
    memory_total: int
    memory_used: int
    utilization: float
    temperature: float
    power_usage: float

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GPU':
        return cls(
            id=data['id'],
            name=data['name'],
            memory_total=data['memory_total'],
            memory_used=data['memory_used'],
            utilization=data['utilization'],
            temperature=data['temperature'],
            power_usage=data['power_usage']
        )


@dataclass
class Deployment:
    """Deployment model"""
    id: str
    name: str
    model_id: str
    backend: str
    status: str
    endpoint: str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Deployment':
        return cls(
            id=data['id'],
            name=data['name'],
            model_id=data['model_id'],
            backend=data['backend'],
            status=data['status'],
            endpoint=data['endpoint']
        )


@dataclass
class HPOStudy:
    """HPO Study model"""
    id: str
    name: str
    status: str
    n_trials: int
    best_value: Optional[float]
    best_params: Optional[Dict[str, Any]]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'HPOStudy':
        return cls(
            id=data['id'],
            name=data['name'],
            status=data['status'],
            n_trials=data['n_trials'],
            best_value=data.get('best_value'),
            best_params=data.get('best_params')
        )
