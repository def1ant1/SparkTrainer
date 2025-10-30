# Dataset Versioning with DVC

This guide explains how to use DVC (Data Version Control) to version and manage datasets in SparkTrainer for reproducible experiments.

## Table of Contents

- [What is DVC?](#what-is-dvc)
- [Setup](#setup)
- [Basic Usage](#basic-usage)
- [Remote Storage](#remote-storage)
- [Workflows](#workflows)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

## What is DVC?

DVC (Data Version Control) is a version control system for machine learning projects that:
- **Versions large files** efficiently (datasets, models)
- **Tracks data pipelines** and dependencies
- **Enables reproducibility** across experiments
- **Integrates with Git** for complete version control
- **Supports remote storage** (S3, GCS, Azure, SSH, etc.)

### Why Use DVC?

- **Reproducibility**: Track exact dataset versions used in experiments
- **Collaboration**: Share datasets without committing large files to Git
- **Storage Efficiency**: Store datasets in cloud storage, not in Git
- **Experiment Tracking**: Link datasets to specific model versions
- **Data Pipelines**: Automate data preprocessing and transformation

## Setup

### Install DVC

```bash
# Install DVC with S3 support
pip install 'dvc[s3]'

# Or with Google Cloud Storage
pip install 'dvc[gs]'

# Or with Azure Blob Storage
pip install 'dvc[azure]'

# Or with all remotes
pip install 'dvc[all]'
```

### Initialize DVC

DVC is already initialized in SparkTrainer. If starting fresh:

```bash
# Initialize DVC in your project
dvc init

# Commit DVC configuration
git add .dvc .dvcignore
git commit -m "Initialize DVC"
```

### Configure Remote Storage

Edit `.dvc/config` or use the CLI:

```bash
# For S3
dvc remote add -d storage s3://my-bucket/sparktrainer-datasets

# For Google Cloud Storage
dvc remote add -d storage gs://my-bucket/sparktrainer-datasets

# For Azure Blob Storage
dvc remote add -d storage azure://my-container/sparktrainer-datasets

# For local storage (testing)
dvc remote add -d storage /path/to/local/storage
```

### Configure AWS Credentials (for S3)

```bash
# Option 1: AWS CLI
aws configure

# Option 2: Environment variables
export AWS_ACCESS_KEY_ID="your_key"
export AWS_SECRET_ACCESS_KEY="your_secret"
export AWS_DEFAULT_REGION="us-east-1"

# Option 3: DVC configuration
dvc remote modify storage access_key_id 'your_key'
dvc remote modify storage secret_access_key 'your_secret'
```

## Basic Usage

### Track a Dataset

```bash
# Add dataset to DVC tracking
dvc add datasets/my-dataset

# This creates: datasets/my-dataset.dvc
# And updates: .gitignore

# Commit the .dvc file to Git
git add datasets/my-dataset.dvc datasets/.gitignore
git commit -m "Add my-dataset v1.0"

# Push dataset to remote storage
dvc push
```

### Download a Dataset

```bash
# Pull dataset from remote storage
dvc pull datasets/my-dataset.dvc

# Or pull all datasets
dvc pull
```

### Update a Dataset

```bash
# Modify dataset
# ... update files in datasets/my-dataset/ ...

# Update DVC tracking
dvc add datasets/my-dataset

# Commit the new version
git add datasets/my-dataset.dvc
git commit -m "Update my-dataset to v1.1 (added 1000 samples)"

# Push new version to remote
dvc push
```

### Checkout Specific Dataset Version

```bash
# Checkout old Git commit
git checkout HEAD~1

# Pull corresponding dataset version
dvc checkout

# Or in one command
dvc checkout datasets/my-dataset.dvc
```

## Remote Storage

### S3 Configuration

```bash
# Add S3 remote
dvc remote add -d s3remote s3://my-bucket/dvc-storage

# Configure region
dvc remote modify s3remote region us-west-2

# Use IAM role (recommended)
dvc remote modify s3remote credentialpath ~/.aws/credentials

# Or use environment variables
export AWS_ACCESS_KEY_ID="..."
export AWS_SECRET_ACCESS_KEY="..."
```

### Google Cloud Storage Configuration

```bash
# Add GCS remote
dvc remote add -d gcsremote gs://my-bucket/dvc-storage

# Configure project
dvc remote modify gcsremote projectname my-gcp-project

# Authenticate
gcloud auth application-default login
```

### Azure Blob Storage Configuration

```bash
# Add Azure remote
dvc remote add -d azureremote azure://my-container/dvc-storage

# Configure connection string
dvc remote modify azureremote connection_string "..."

# Or use account name and key
dvc remote modify azureremote account_name "my_account"
dvc remote modify azureremote account_key "my_key"
```

### SSH/SFTP Configuration

```bash
# Add SSH remote
dvc remote add -d sshremote ssh://user@example.com/path/to/dvc-storage

# Configure SSH key
dvc remote modify sshremote keyfile ~/.ssh/id_rsa
```

## Workflows

### Workflow 1: New Dataset

```bash
# 1. Ingest dataset using SparkTrainer
spark-trainer ingest video \
  --input videos/ \
  --output datasets/my-video-dataset \
  --caption-model blip2

# 2. Track with DVC
dvc add datasets/my-video-dataset

# 3. Commit to Git
git add datasets/my-video-dataset.dvc datasets/.gitignore
git commit -m "Add video dataset v1.0 (1000 videos)"

# 4. Push to remote storage
dvc push

# 5. Push to Git
git push
```

### Workflow 2: Reproduce Experiment

```bash
# 1. Clone repository
git clone https://github.com/user/sparktrainer-experiments.git
cd sparktrainer-experiments

# 2. Checkout specific experiment
git checkout experiment-v1.0

# 3. Pull corresponding datasets
dvc pull

# 4. Run training
spark-trainer train --config configs/experiment-v1.0.yaml

# Results are now reproducible!
```

### Workflow 3: Dataset Splitting

```bash
# 1. Add full dataset
dvc add datasets/full-dataset

# 2. Create pipeline to split dataset
cat > dvc.yaml <<EOF
stages:
  split:
    cmd: python scripts/split_dataset.py
    deps:
      - datasets/full-dataset
      - scripts/split_dataset.py
    outs:
      - datasets/train-split
      - datasets/val-split
      - datasets/test-split
EOF

# 3. Run pipeline
dvc repro

# 4. Commit pipeline and splits
git add dvc.yaml dvc.lock datasets/*.dvc
git commit -m "Add dataset splitting pipeline"
dvc push
git push
```

### Workflow 4: Experiment Tracking

```bash
# 1. Create experiment branch
git checkout -b experiment-lora-r32

# 2. Train model with specific dataset version
spark-trainer train \
  --dataset datasets/my-dataset \
  --recipe lora \
  --config configs/lora-r32.yaml

# 3. Track experiment
git add configs/lora-r32.yaml
git commit -m "Experiment: LoRA with r=32"

# 4. Dataset version is automatically tracked via .dvc files
# No need to copy large datasets!

# 5. Compare experiments
git checkout main
git log --oneline --graph --all
```

## Best Practices

### 1. Dataset Organization

```
datasets/
├── raw/                    # Original, unprocessed data
│   └── videos/            # Track with DVC
├── processed/             # Preprocessed data
│   ├── train/            # Track with DVC
│   ├── val/              # Track with DVC
│   └── test/             # Track with DVC
└── manifests/             # JSONL manifests (track with Git)
    ├── train.jsonl
    ├── val.jsonl
    └── test.jsonl
```

### 2. Version Naming Convention

Use semantic versioning for datasets:

```bash
git commit -m "Add dataset v1.0.0 (initial release)"
git tag dataset-v1.0.0

git commit -m "Update dataset v1.1.0 (added 1000 samples)"
git tag dataset-v1.1.0

git commit -m "Update dataset v2.0.0 (changed format)"
git tag dataset-v2.0.0
```

### 3. Dataset Metadata

Track metadata alongside datasets:

```bash
# Create metadata file
cat > datasets/my-dataset/metadata.json <<EOF
{
  "name": "my-dataset",
  "version": "1.0.0",
  "created_at": "2024-01-01",
  "num_samples": 10000,
  "format": "jsonl",
  "license": "CC-BY-4.0",
  "description": "Dataset of videos with captions",
  "splits": {
    "train": 8000,
    "val": 1000,
    "test": 1000
  },
  "checksum": "sha256:..."
}
EOF

# Track metadata with Git (small file)
git add datasets/my-dataset/metadata.json
git commit -m "Add metadata for my-dataset v1.0.0"
```

### 4. .dvcignore

Exclude temporary files from DVC tracking:

```
# .dvcignore
*.tmp
*.temp
.DS_Store
__pycache__/
*.pyc
.ipynb_checkpoints/
```

### 5. Pipeline Definition

Define data pipelines in `dvc.yaml`:

```yaml
stages:
  download:
    cmd: python scripts/download_videos.py
    outs:
      - datasets/raw/videos

  extract_frames:
    cmd: python scripts/extract_frames.py
    deps:
      - datasets/raw/videos
      - scripts/extract_frames.py
    outs:
      - datasets/processed/frames

  generate_captions:
    cmd: python scripts/generate_captions.py
    deps:
      - datasets/processed/frames
      - scripts/generate_captions.py
    outs:
      - datasets/processed/captions

  create_manifest:
    cmd: python scripts/create_manifest.py
    deps:
      - datasets/processed/frames
      - datasets/processed/captions
    outs:
      - datasets/manifests/dataset.jsonl
```

Run pipeline:

```bash
dvc repro
```

### 6. Sharing Datasets

```bash
# Team member A creates dataset
dvc add datasets/team-dataset
git add datasets/team-dataset.dvc
git commit -m "Add team dataset"
dvc push
git push

# Team member B downloads dataset
git pull
dvc pull datasets/team-dataset.dvc
```

## Integration with SparkTrainer

### Track Ingested Datasets

```python
# In your ingestion script
from spark_trainer.ingestion import VideoIngestionWizard
import subprocess

# Ingest dataset
wizard = VideoIngestionWizard()
wizard.run(input_path='videos/', output_path='datasets/my-dataset')

# Automatically track with DVC
subprocess.run(['dvc', 'add', 'datasets/my-dataset'])
subprocess.run(['git', 'add', 'datasets/my-dataset.dvc'])
subprocess.run(['git', 'commit', '-m', 'Add ingested dataset'])
subprocess.run(['dvc', 'push'])
```

### Link Datasets to Experiments

```python
# In your training script
import mlflow
import subprocess

# Get current dataset version
dataset_hash = subprocess.check_output(
    ['git', 'rev-parse', 'HEAD:datasets/my-dataset.dvc']
).decode().strip()

# Log to MLflow
with mlflow.start_run():
    mlflow.log_param('dataset', 'my-dataset')
    mlflow.log_param('dataset_version', dataset_hash)
    mlflow.log_param('dataset_commit', subprocess.check_output(
        ['git', 'rev-parse', 'HEAD']
    ).decode().strip())

    # Train model...
```

## Troubleshooting

### Problem: dvc push fails with authentication error

**Solution**:
```bash
# Check remote configuration
dvc remote list

# Verify credentials
aws s3 ls s3://my-bucket/  # For S3
gsutil ls gs://my-bucket/  # For GCS

# Reconfigure remote
dvc remote modify storage --unset credentialpath
dvc remote modify storage access_key_id "new_key"
dvc remote modify storage secret_access_key "new_secret"
```

### Problem: dvc pull is slow

**Solution**:
```bash
# Enable caching
dvc config cache.type hardlink

# Use compression
dvc remote modify storage use_ssl true

# Parallelize downloads
dvc pull -j 4  # 4 parallel jobs
```

### Problem: Dataset already tracked by Git

**Solution**:
```bash
# Remove from Git
git rm -r --cached datasets/my-dataset

# Add to DVC
dvc add datasets/my-dataset

# Update .gitignore
git add datasets/my-dataset.dvc datasets/.gitignore
git commit -m "Move dataset to DVC"
```

### Problem: Disk space full

**Solution**:
```bash
# Clear DVC cache
dvc gc --workspace --force

# Or remove specific cached version
dvc remove datasets/old-dataset.dvc --outs
```

## Resources

- [DVC Documentation](https://dvc.org/doc)
- [DVC with S3](https://dvc.org/doc/user-guide/data-management/remote-storage/amazon-s3)
- [DVC Pipelines](https://dvc.org/doc/user-guide/pipelines)
- [DVC Best Practices](https://dvc.org/doc/user-guide/best-practices)

## Support

For DVC-specific questions:
- [DVC Discussions](https://github.com/iterative/dvc/discussions)
- [DVC Discord](https://dvc.org/chat)

For SparkTrainer + DVC integration:
- [SparkTrainer Discussions](https://github.com/def1ant1/SparkTrainer/discussions)

---

**Happy versioning!** 📦
