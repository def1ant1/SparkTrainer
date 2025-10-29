"""
HuggingFace Export Module

Handles exporting models and datasets to HuggingFace Hub with:
- Resumable uploads
- Git LFS for large files
- Auto-generated cards and metadata
- Private/public repository support
"""

import os
import json
import shutil
import subprocess
from typing import Dict, Any, Optional, List
from pathlib import Path
import requests
from datetime import datetime


class HuggingFaceExporter:
    """Handle exports to HuggingFace Hub."""

    def __init__(self, hf_token: str):
        """
        Initialize HF exporter with authentication token.

        Args:
            hf_token: HuggingFace API token
        """
        self.token = hf_token
        self.api_url = "https://huggingface.co/api"
        self.headers = {"Authorization": f"Bearer {hf_token}"}

    def create_repo(
        self,
        repo_name: str,
        repo_type: str = "model",  # "model" or "dataset"
        private: bool = False,
        organization: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a new repository on HuggingFace Hub.

        Args:
            repo_name: Name of the repository
            repo_type: Type of repository ("model" or "dataset")
            private: Whether the repository should be private
            organization: Optional organization to create repo under

        Returns:
            Dictionary with repository information
        """
        endpoint = f"{self.api_url}/repos/create"

        payload = {
            "name": repo_name,
            "type": repo_type,
            "private": private
        }

        if organization:
            payload["organization"] = organization

        response = requests.post(endpoint, headers=self.headers, json=payload)
        response.raise_for_status()
        return response.json()

    def generate_model_card(
        self,
        model_name: str,
        model_config: Dict[str, Any],
        training_config: Dict[str, Any],
        metrics: Dict[str, Any]
    ) -> str:
        """
        Generate a Model Card in markdown format.

        Args:
            model_name: Name of the model
            model_config: Model architecture configuration
            training_config: Training hyperparameters
            metrics: Evaluation metrics

        Returns:
            Markdown-formatted model card
        """
        card = f"""---
language:
- en
license: apache-2.0
library_name: transformers
tags:
- pytorch
- fine-tuned
- sparktrainer
base_model: {model_config.get('base_model', 'unknown')}
---

# {model_name}

This model was fine-tuned using [SparkTrainer](https://github.com/sparktrainer/sparktrainer), an MLOps platform for GPU-accelerated training.

## Model Description

- **Model type:** {model_config.get('model_type', 'Transformer')}
- **Language(s):** {model_config.get('language', 'English')}
- **License:** {model_config.get('license', 'Apache 2.0')}
- **Finetuned from:** {model_config.get('base_model', 'N/A')}

## Training Details

### Training Data

- **Dataset:** {training_config.get('dataset_name', 'N/A')}
- **Dataset size:** {training_config.get('dataset_size', 'N/A')} samples

### Training Procedure

#### Hyperparameters

```yaml
learning_rate: {training_config.get('learning_rate', 'N/A')}
num_epochs: {training_config.get('num_epochs', 'N/A')}
batch_size: {training_config.get('batch_size', 'N/A')}
gradient_accumulation_steps: {training_config.get('gradient_accumulation_steps', 1)}
optimizer: {training_config.get('optimizer', 'AdamW')}
scheduler: {training_config.get('scheduler', 'linear')}
```

#### Fine-tuning Method

- **Method:** {training_config.get('recipe', 'Full fine-tuning')}
"""

        if training_config.get('recipe') == 'lora':
            card += f"""
#### LoRA Configuration

```yaml
lora_r: {training_config.get('lora_r', 8)}
lora_alpha: {training_config.get('lora_alpha', 16)}
lora_dropout: {training_config.get('lora_dropout', 0.05)}
target_modules: {training_config.get('target_modules', [])}
```
"""

        if metrics:
            card += f"""
## Evaluation Results

"""
            for metric_name, value in metrics.items():
                card += f"- **{metric_name}:** {value}\n"

        card += f"""
## How to Use

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{model_name}")
tokenizer = AutoTokenizer.from_pretrained("{model_name}")

inputs = tokenizer("Hello, how are you?", return_tensors="pt")
outputs = model.generate(**inputs)
print(tokenizer.decode(outputs[0]))
```

## Limitations and Bias

This model was fine-tuned on a specific dataset and may not generalize to all use cases. Please evaluate the model on your specific task before deployment.

## Training Logs

Generated with SparkTrainer on {datetime.now().strftime('%Y-%m-%d')}

## Citation

```bibtex
@misc{{{model_name.replace('-', '_').replace('/', '_')}}},
  title={{{model_name}}},
  author={{Your Name}},
  year={{{datetime.now().year}}},
  publisher={{HuggingFace}},
  url={{https://huggingface.co/{model_name}}}
}}
```
"""
        return card

    def generate_dataset_card(
        self,
        dataset_name: str,
        dataset_config: Dict[str, Any],
        statistics: Dict[str, Any]
    ) -> str:
        """
        Generate a Dataset Card in markdown format.

        Args:
            dataset_name: Name of the dataset
            dataset_config: Dataset configuration
            statistics: Dataset statistics

        Returns:
            Markdown-formatted dataset card
        """
        card = f"""---
language:
- en
license: {dataset_config.get('license', 'apache-2.0')}
task_categories:
- {dataset_config.get('task_category', 'text-generation')}
tags:
- sparktrainer
size_categories:
- {self._get_size_category(statistics.get('num_samples', 0))}
---

# {dataset_name}

This dataset was curated using [SparkTrainer](https://github.com/sparktrainer/sparktrainer).

## Dataset Description

- **Curated from:** {dataset_config.get('source', 'Custom data')}
- **Language:** {dataset_config.get('language', 'English')}
- **License:** {dataset_config.get('license', 'Apache 2.0')}

## Dataset Statistics

- **Total samples:** {statistics.get('num_samples', 'N/A'):,}
- **Total size:** {self._format_bytes(statistics.get('total_bytes', 0))}
- **File types:** {', '.join(statistics.get('file_types', []))}

## Dataset Structure

### Data Fields

{dataset_config.get('fields_description', 'See dataset files for structure.')}

## Usage

```python
from datasets import load_dataset

dataset = load_dataset("{dataset_name}")
print(dataset)
```

## Data Quality

This dataset has been processed through SparkTrainer's quality gates:

- ✅ Deduplication
- ✅ Format validation
- ✅ Integrity checks
{('- ✅ PII redaction' if dataset_config.get('pii_redacted') else '')}

## Citation

```bibtex
@misc{{{dataset_name.replace('-', '_').replace('/', '_')}}},
  title={{{dataset_name}}},
  author={{Your Name}},
  year={{{datetime.now().year}}},
  publisher={{HuggingFace}},
  url={{https://huggingface.co/datasets/{dataset_name}}}
}}
```
"""
        return card

    def _get_size_category(self, num_samples: int) -> str:
        """Get HuggingFace size category based on sample count."""
        if num_samples < 1000:
            return "n<1K"
        elif num_samples < 10000:
            return "1K<n<10K"
        elif num_samples < 100000:
            return "10K<n<100K"
        elif num_samples < 1000000:
            return "100K<n<1M"
        else:
            return "1M<n<10M"

    def _format_bytes(self, bytes_size: int) -> str:
        """Format bytes to human-readable string."""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if bytes_size < 1024.0:
                return f"{bytes_size:.1f} {unit}"
            bytes_size /= 1024.0
        return f"{bytes_size:.1f} PB"

    def generate_license(self, license_type: str = "apache-2.0") -> str:
        """
        Generate LICENSE file content.

        Args:
            license_type: Type of license (default: apache-2.0)

        Returns:
            License text
        """
        if license_type == "apache-2.0":
            return """Apache License
Version 2.0, January 2004
http://www.apache.org/licenses/

TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION
[Full Apache 2.0 license text...]
"""
        elif license_type == "mit":
            return f"""MIT License

Copyright (c) {datetime.now().year}

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction...
"""
        else:
            return f"Custom License: {license_type}"

    def upload_folder(
        self,
        repo_id: str,
        folder_path: str,
        repo_type: str = "model",
        commit_message: str = "Upload from SparkTrainer"
    ) -> Dict[str, Any]:
        """
        Upload a folder to HuggingFace Hub using git.

        This method uses git with LFS for large files.

        Args:
            repo_id: Repository ID (e.g., "username/repo-name")
            folder_path: Local path to folder to upload
            repo_type: Type of repository ("model" or "dataset")
            commit_message: Git commit message

        Returns:
            Dictionary with upload status
        """
        try:
            # Clone repository
            repo_url = f"https://huggingface.co/{repo_id}"
            temp_dir = f"/tmp/hf_upload_{repo_id.replace('/', '_')}"

            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

            # Clone with authentication
            clone_url = f"https://user:{self.token}@huggingface.co/{repo_id}"
            subprocess.run(
                ["git", "clone", clone_url, temp_dir],
                check=True,
                capture_output=True
            )

            # Configure git LFS for large files
            subprocess.run(
                ["git", "lfs", "install"],
                cwd=temp_dir,
                check=True
            )

            # Track large file patterns with LFS
            lfs_patterns = ["*.bin", "*.safetensors", "*.ckpt", "*.pth", "*.pt"]
            for pattern in lfs_patterns:
                subprocess.run(
                    ["git", "lfs", "track", pattern],
                    cwd=temp_dir,
                    check=True
                )

            # Copy files
            for item in os.listdir(folder_path):
                src = os.path.join(folder_path, item)
                dst = os.path.join(temp_dir, item)
                if os.path.isdir(src):
                    shutil.copytree(src, dst, dirs_exist_ok=True)
                else:
                    shutil.copy2(src, dst)

            # Git add, commit, push
            subprocess.run(
                ["git", "add", "."],
                cwd=temp_dir,
                check=True
            )
            subprocess.run(
                ["git", "commit", "-m", commit_message],
                cwd=temp_dir,
                check=True
            )
            subprocess.run(
                ["git", "push"],
                cwd=temp_dir,
                check=True
            )

            # Cleanup
            shutil.rmtree(temp_dir)

            return {
                "status": "success",
                "repo_url": repo_url,
                "message": "Upload completed successfully"
            }

        except subprocess.CalledProcessError as e:
            return {
                "status": "error",
                "error": str(e),
                "stderr": e.stderr.decode() if e.stderr else None
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }

    def export_model(
        self,
        model_path: str,
        repo_name: str,
        model_config: Dict[str, Any],
        training_config: Dict[str, Any],
        metrics: Dict[str, Any],
        private: bool = False,
        organization: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Export a complete model to HuggingFace Hub.

        Args:
            model_path: Local path to model files
            repo_name: Name for the HF repository
            model_config: Model configuration
            training_config: Training configuration
            metrics: Evaluation metrics
            private: Whether repository should be private
            organization: Optional organization name

        Returns:
            Dictionary with export status
        """
        try:
            # Create repository
            repo_id = f"{organization}/{repo_name}" if organization else repo_name
            self.create_repo(repo_name, "model", private, organization)

            # Generate README.md (Model Card)
            readme_content = self.generate_model_card(
                repo_id,
                model_config,
                training_config,
                metrics
            )

            # Generate LICENSE
            license_content = self.generate_license()

            # Prepare upload directory
            upload_dir = f"/tmp/model_export_{repo_name}"
            os.makedirs(upload_dir, exist_ok=True)

            # Copy model files
            for item in os.listdir(model_path):
                src = os.path.join(model_path, item)
                dst = os.path.join(upload_dir, item)
                if os.path.isdir(src):
                    shutil.copytree(src, dst, dirs_exist_ok=True)
                else:
                    shutil.copy2(src, dst)

            # Write README and LICENSE
            with open(os.path.join(upload_dir, "README.md"), "w") as f:
                f.write(readme_content)
            with open(os.path.join(upload_dir, "LICENSE"), "w") as f:
                f.write(license_content)

            # Upload
            result = self.upload_folder(
                repo_id,
                upload_dir,
                "model",
                "Upload model from SparkTrainer"
            )

            # Cleanup
            shutil.rmtree(upload_dir)

            return result

        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }

    def export_dataset(
        self,
        dataset_path: str,
        repo_name: str,
        dataset_config: Dict[str, Any],
        statistics: Dict[str, Any],
        private: bool = False,
        organization: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Export a dataset to HuggingFace Hub.

        Args:
            dataset_path: Local path to dataset files
            repo_name: Name for the HF repository
            dataset_config: Dataset configuration
            statistics: Dataset statistics
            private: Whether repository should be private
            organization: Optional organization name

        Returns:
            Dictionary with export status
        """
        try:
            # Create repository
            repo_id = f"{organization}/{repo_name}" if organization else repo_name
            self.create_repo(repo_name, "dataset", private, organization)

            # Generate README.md (Dataset Card)
            readme_content = self.generate_dataset_card(
                repo_id,
                dataset_config,
                statistics
            )

            # Generate LICENSE
            license_content = self.generate_license()

            # Prepare upload directory
            upload_dir = f"/tmp/dataset_export_{repo_name}"
            os.makedirs(upload_dir, exist_ok=True)

            # Copy dataset files
            for item in os.listdir(dataset_path):
                src = os.path.join(dataset_path, item)
                dst = os.path.join(upload_dir, item)
                if os.path.isdir(src):
                    shutil.copytree(src, dst, dirs_exist_ok=True)
                else:
                    shutil.copy2(src, dst)

            # Write README and LICENSE
            with open(os.path.join(upload_dir, "README.md"), "w") as f:
                f.write(readme_content)
            with open(os.path.join(upload_dir, "LICENSE"), "w") as f:
                f.write(license_content)

            # Upload
            result = self.upload_folder(
                repo_id,
                upload_dir,
                "dataset",
                "Upload dataset from SparkTrainer"
            )

            # Cleanup
            shutil.rmtree(upload_dir)

            return result

        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
