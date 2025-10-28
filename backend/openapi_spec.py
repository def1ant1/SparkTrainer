"""
OpenAPI Specification Generator for SparkTrainer API
Generates a comprehensive OpenAPI 3.0 specification from the Flask application
"""

import yaml
from typing import Dict, Any, List


def generate_openapi_spec() -> Dict[str, Any]:
    """Generate the complete OpenAPI 3.0 specification"""

    spec = {
        "openapi": "3.0.3",
        "info": {
            "title": "SparkTrainer API",
            "description": """
# SparkTrainer API Documentation

SparkTrainer is a comprehensive MLOps platform for managing training jobs, experiments, datasets, and models.

## Features
- **Job Management**: Submit, monitor, and manage training jobs
- **Experiment Tracking**: Track experiments with MLflow integration
- **Dataset Management**: Upload, version, and manage datasets
- **Model Registry**: Store and version trained models
- **GPU Scheduling**: Efficient GPU resource allocation
- **Team Collaboration**: Multi-user support with role-based access
- **Real-time Monitoring**: Live metrics streaming via SSE/WebSocket
- **Hyperparameter Optimization**: Optuna-powered HPO sweeps
- **Model Deployment**: Deploy models with vLLM, TGI, and Triton

## Authentication
Most endpoints require authentication using JWT tokens:
1. Login via `/api/auth/login` to receive access and refresh tokens
2. Include the access token in the `Authorization: Bearer <token>` header
3. Refresh tokens using `/api/auth/refresh` when they expire

## Rate Limiting
API requests are rate-limited per user:
- Standard users: 100 requests/minute
- Premium users: 1000 requests/minute
            """,
            "version": "1.0.0",
            "contact": {
                "name": "SparkTrainer Support",
                "url": "https://github.com/def1ant1/SparkTrainer"
            },
            "license": {
                "name": "MIT",
                "url": "https://opensource.org/licenses/MIT"
            }
        },
        "servers": [
            {
                "url": "http://localhost:5001",
                "description": "Development server"
            },
            {
                "url": "http://localhost:5001/api",
                "description": "Development server with /api prefix"
            }
        ],
        "tags": [
            {"name": "auth", "description": "Authentication and authorization"},
            {"name": "jobs", "description": "Training job management"},
            {"name": "experiments", "description": "Experiment tracking"},
            {"name": "datasets", "description": "Dataset management"},
            {"name": "models", "description": "Model registry"},
            {"name": "gpus", "description": "GPU resource management"},
            {"name": "metrics", "description": "System and job metrics"},
            {"name": "teams", "description": "Team and user management"},
            {"name": "billing", "description": "Billing and usage tracking"},
            {"name": "pipelines", "description": "Training pipelines"},
            {"name": "schedules", "description": "Job scheduling"},
            {"name": "health", "description": "Health and readiness checks"},
            {"name": "hpo", "description": "Hyperparameter optimization"},
            {"name": "deployments", "description": "Model deployments"},
            {"name": "evaluation", "description": "Model evaluation and benchmarks"},
        ],
        "paths": _generate_paths(),
        "components": _generate_components()
    }

    return spec


def _generate_paths() -> Dict[str, Any]:
    """Generate all API paths"""
    return {
        # Health checks
        "/healthz": {
            "get": {
                "tags": ["health"],
                "summary": "Health check",
                "description": "Basic health check endpoint",
                "responses": {
                    "200": {
                        "description": "Service is healthy",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "status": {"type": "string", "example": "healthy"}
                                    }
                                }
                            }
                        }
                    }
                }
            }
        },
        "/readyz": {
            "get": {
                "tags": ["health"],
                "summary": "Readiness check",
                "description": "Check if service is ready to accept traffic",
                "responses": {
                    "200": {
                        "description": "Service is ready",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "status": {"type": "string"},
                                        "database": {"type": "boolean"},
                                        "redis": {"type": "boolean"},
                                        "celery": {"type": "boolean"}
                                    }
                                }
                            }
                        }
                    },
                    "503": {
                        "description": "Service not ready"
                    }
                }
            }
        },

        # Authentication
        "/api/auth/login": {
            "post": {
                "tags": ["auth"],
                "summary": "User login",
                "description": "Authenticate user and receive JWT tokens",
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "required": ["username", "password"],
                                "properties": {
                                    "username": {"type": "string"},
                                    "password": {"type": "string", "format": "password"}
                                }
                            }
                        }
                    }
                },
                "responses": {
                    "200": {
                        "description": "Login successful",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/AuthTokens"}
                            }
                        }
                    },
                    "401": {"description": "Invalid credentials"}
                }
            }
        },
        "/api/auth/refresh": {
            "post": {
                "tags": ["auth"],
                "summary": "Refresh access token",
                "security": [{"bearerAuth": []}],
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "required": ["refresh_token"],
                                "properties": {
                                    "refresh_token": {"type": "string"}
                                }
                            }
                        }
                    }
                },
                "responses": {
                    "200": {
                        "description": "New access token",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/AuthTokens"}
                            }
                        }
                    }
                }
            }
        },
        "/api/auth/logout": {
            "post": {
                "tags": ["auth"],
                "summary": "User logout",
                "security": [{"bearerAuth": []}],
                "responses": {
                    "200": {"description": "Logout successful"}
                }
            }
        },

        # Jobs
        "/api/jobs": {
            "get": {
                "tags": ["jobs"],
                "summary": "List all jobs",
                "security": [{"bearerAuth": []}],
                "parameters": [
                    {"name": "status", "in": "query", "schema": {"type": "string"}},
                    {"name": "limit", "in": "query", "schema": {"type": "integer"}},
                    {"name": "offset", "in": "query", "schema": {"type": "integer"}}
                ],
                "responses": {
                    "200": {
                        "description": "List of jobs",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "array",
                                    "items": {"$ref": "#/components/schemas/Job"}
                                }
                            }
                        }
                    }
                }
            },
            "post": {
                "tags": ["jobs"],
                "summary": "Submit a new job",
                "security": [{"bearerAuth": []}],
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/JobSubmit"}
                        }
                    }
                },
                "responses": {
                    "201": {
                        "description": "Job created",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/Job"}
                            }
                        }
                    }
                }
            }
        },
        "/api/jobs/{job_id}": {
            "get": {
                "tags": ["jobs"],
                "summary": "Get job details",
                "security": [{"bearerAuth": []}],
                "parameters": [
                    {"name": "job_id", "in": "path", "required": True, "schema": {"type": "string"}}
                ],
                "responses": {
                    "200": {
                        "description": "Job details",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/Job"}
                            }
                        }
                    },
                    "404": {"description": "Job not found"}
                }
            },
            "delete": {
                "tags": ["jobs"],
                "summary": "Delete a job",
                "security": [{"bearerAuth": []}],
                "parameters": [
                    {"name": "job_id", "in": "path", "required": True, "schema": {"type": "string"}}
                ],
                "responses": {
                    "200": {"description": "Job deleted"},
                    "404": {"description": "Job not found"}
                }
            }
        },
        "/api/jobs/{job_id}/cancel": {
            "post": {
                "tags": ["jobs"],
                "summary": "Cancel a running job",
                "security": [{"bearerAuth": []}],
                "parameters": [
                    {"name": "job_id", "in": "path", "required": True, "schema": {"type": "string"}}
                ],
                "responses": {
                    "200": {"description": "Job cancelled"},
                    "404": {"description": "Job not found"}
                }
            }
        },
        "/api/jobs/{job_id}/logs": {
            "get": {
                "tags": ["jobs"],
                "summary": "Get job logs",
                "security": [{"bearerAuth": []}],
                "parameters": [
                    {"name": "job_id", "in": "path", "required": True, "schema": {"type": "string"}},
                    {"name": "stream", "in": "query", "schema": {"type": "string", "enum": ["stdout", "stderr"]}}
                ],
                "responses": {
                    "200": {
                        "description": "Job logs",
                        "content": {
                            "text/plain": {
                                "schema": {"type": "string"}
                            }
                        }
                    }
                }
            }
        },
        "/api/jobs/{job_id}/stream": {
            "get": {
                "tags": ["jobs"],
                "summary": "Stream job metrics (SSE)",
                "security": [{"bearerAuth": []}],
                "parameters": [
                    {"name": "job_id", "in": "path", "required": True, "schema": {"type": "string"}}
                ],
                "responses": {
                    "200": {
                        "description": "Server-Sent Events stream",
                        "content": {
                            "text/event-stream": {
                                "schema": {"type": "string"}
                            }
                        }
                    }
                }
            }
        },

        # Experiments
        "/api/experiments": {
            "get": {
                "tags": ["experiments"],
                "summary": "List experiments",
                "security": [{"bearerAuth": []}],
                "responses": {
                    "200": {
                        "description": "List of experiments",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "array",
                                    "items": {"$ref": "#/components/schemas/Experiment"}
                                }
                            }
                        }
                    }
                }
            },
            "post": {
                "tags": ["experiments"],
                "summary": "Create experiment",
                "security": [{"bearerAuth": []}],
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/ExperimentCreate"}
                        }
                    }
                },
                "responses": {
                    "201": {"description": "Experiment created"}
                }
            }
        },

        # Datasets
        "/api/datasets": {
            "get": {
                "tags": ["datasets"],
                "summary": "List datasets",
                "security": [{"bearerAuth": []}],
                "responses": {
                    "200": {
                        "description": "List of datasets",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "array",
                                    "items": {"$ref": "#/components/schemas/Dataset"}
                                }
                            }
                        }
                    }
                }
            },
            "post": {
                "tags": ["datasets"],
                "summary": "Create dataset",
                "security": [{"bearerAuth": []}],
                "requestBody": {
                    "required": True,
                    "content": {
                        "multipart/form-data": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "file": {"type": "string", "format": "binary"}
                                }
                            }
                        }
                    }
                },
                "responses": {
                    "201": {"description": "Dataset created"}
                }
            }
        },

        # Models
        "/api/models": {
            "get": {
                "tags": ["models"],
                "summary": "List models",
                "security": [{"bearerAuth": []}],
                "responses": {
                    "200": {
                        "description": "List of models",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "array",
                                    "items": {"$ref": "#/components/schemas/Model"}
                                }
                            }
                        }
                    }
                }
            }
        },

        # GPUs
        "/api/gpus": {
            "get": {
                "tags": ["gpus"],
                "summary": "List GPU status",
                "security": [{"bearerAuth": []}],
                "responses": {
                    "200": {
                        "description": "GPU information",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "array",
                                    "items": {"$ref": "#/components/schemas/GPU"}
                                }
                            }
                        }
                    }
                }
            }
        },

        # Metrics
        "/api/metrics": {
            "get": {
                "tags": ["metrics"],
                "summary": "Get system metrics",
                "security": [{"bearerAuth": []}],
                "responses": {
                    "200": {
                        "description": "System metrics",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/Metrics"}
                            }
                        }
                    }
                }
            }
        },

        # HPO
        "/api/hpo/studies": {
            "get": {
                "tags": ["hpo"],
                "summary": "List HPO studies",
                "security": [{"bearerAuth": []}],
                "responses": {
                    "200": {
                        "description": "List of studies",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "array",
                                    "items": {"$ref": "#/components/schemas/HPOStudy"}
                                }
                            }
                        }
                    }
                }
            },
            "post": {
                "tags": ["hpo"],
                "summary": "Create HPO study",
                "security": [{"bearerAuth": []}],
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/HPOStudyCreate"}
                        }
                    }
                },
                "responses": {
                    "201": {"description": "Study created"}
                }
            }
        },

        # Deployments
        "/api/deployments": {
            "get": {
                "tags": ["deployments"],
                "summary": "List deployments",
                "security": [{"bearerAuth": []}],
                "responses": {
                    "200": {
                        "description": "List of deployments",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "array",
                                    "items": {"$ref": "#/components/schemas/Deployment"}
                                }
                            }
                        }
                    }
                }
            },
            "post": {
                "tags": ["deployments"],
                "summary": "Create deployment",
                "security": [{"bearerAuth": []}],
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/DeploymentCreate"}
                        }
                    }
                },
                "responses": {
                    "201": {"description": "Deployment created"}
                }
            }
        },

        # WebSocket
        "/ws/jobs/{job_id}/stream": {
            "get": {
                "tags": ["jobs"],
                "summary": "WebSocket stream for real-time metrics",
                "description": "WebSocket endpoint for streaming job metrics in real-time",
                "parameters": [
                    {"name": "job_id", "in": "path", "required": True, "schema": {"type": "string"}}
                ],
                "responses": {
                    "101": {"description": "Switching Protocols"}
                }
            }
        }
    }


def _generate_components() -> Dict[str, Any]:
    """Generate reusable component schemas"""
    return {
        "securitySchemes": {
            "bearerAuth": {
                "type": "http",
                "scheme": "bearer",
                "bearerFormat": "JWT"
            }
        },
        "schemas": {
            "AuthTokens": {
                "type": "object",
                "properties": {
                    "access_token": {"type": "string"},
                    "refresh_token": {"type": "string"},
                    "token_type": {"type": "string", "example": "Bearer"},
                    "expires_in": {"type": "integer", "example": 3600}
                }
            },
            "Job": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "name": {"type": "string"},
                    "status": {"type": "string", "enum": ["pending", "running", "completed", "failed", "cancelled"]},
                    "command": {"type": "string"},
                    "gpu_count": {"type": "integer"},
                    "created_at": {"type": "string", "format": "date-time"},
                    "started_at": {"type": "string", "format": "date-time"},
                    "completed_at": {"type": "string", "format": "date-time"},
                    "metrics": {"type": "object"}
                }
            },
            "JobSubmit": {
                "type": "object",
                "required": ["name", "command"],
                "properties": {
                    "name": {"type": "string"},
                    "command": {"type": "string"},
                    "gpu_count": {"type": "integer", "default": 1},
                    "environment": {"type": "object"},
                    "priority": {"type": "integer", "default": 0}
                }
            },
            "Experiment": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                    "created_at": {"type": "string", "format": "date-time"},
                    "runs": {"type": "array", "items": {"type": "object"}}
                }
            },
            "ExperimentCreate": {
                "type": "object",
                "required": ["name"],
                "properties": {
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                    "tags": {"type": "object"}
                }
            },
            "Dataset": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "name": {"type": "string"},
                    "size": {"type": "integer"},
                    "format": {"type": "string"},
                    "created_at": {"type": "string", "format": "date-time"},
                    "version": {"type": "string"}
                }
            },
            "Model": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "name": {"type": "string"},
                    "framework": {"type": "string"},
                    "size": {"type": "integer"},
                    "created_at": {"type": "string", "format": "date-time"},
                    "tags": {"type": "array", "items": {"type": "string"}}
                }
            },
            "GPU": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "name": {"type": "string"},
                    "memory_total": {"type": "integer"},
                    "memory_used": {"type": "integer"},
                    "utilization": {"type": "number"},
                    "temperature": {"type": "number"},
                    "power_usage": {"type": "number"}
                }
            },
            "Metrics": {
                "type": "object",
                "properties": {
                    "timestamp": {"type": "string", "format": "date-time"},
                    "cpu_percent": {"type": "number"},
                    "memory_percent": {"type": "number"},
                    "disk_usage": {"type": "number"},
                    "network_io": {"type": "object"}
                }
            },
            "HPOStudy": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "name": {"type": "string"},
                    "status": {"type": "string"},
                    "n_trials": {"type": "integer"},
                    "best_value": {"type": "number"},
                    "best_params": {"type": "object"}
                }
            },
            "HPOStudyCreate": {
                "type": "object",
                "required": ["name", "objective", "search_space"],
                "properties": {
                    "name": {"type": "string"},
                    "objective": {"type": "string"},
                    "search_space": {"type": "object"},
                    "n_trials": {"type": "integer", "default": 100},
                    "parallelism": {"type": "integer", "default": 1}
                }
            },
            "Deployment": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "name": {"type": "string"},
                    "model_id": {"type": "string"},
                    "backend": {"type": "string", "enum": ["vllm", "tgi", "triton"]},
                    "status": {"type": "string"},
                    "endpoint": {"type": "string"}
                }
            },
            "DeploymentCreate": {
                "type": "object",
                "required": ["name", "model_id", "backend"],
                "properties": {
                    "name": {"type": "string"},
                    "model_id": {"type": "string"},
                    "backend": {"type": "string", "enum": ["vllm", "tgi", "triton"]},
                    "replicas": {"type": "integer", "default": 1},
                    "gpu_count": {"type": "integer", "default": 1}
                }
            }
        }
    }


def save_openapi_spec(filename: str = "openapi.yaml"):
    """Save the OpenAPI spec to a YAML file"""
    spec = generate_openapi_spec()
    with open(filename, 'w') as f:
        yaml.dump(spec, f, default_flow_style=False, sort_keys=False)
    return spec


if __name__ == "__main__":
    spec = save_openapi_spec()
    print("OpenAPI specification generated successfully!")
    print(f"Paths: {len(spec['paths'])}")
    print(f"Schemas: {len(spec['components']['schemas'])}")
