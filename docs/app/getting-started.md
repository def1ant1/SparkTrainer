# Getting Started

This guide will help you get up and running with SparkTrainer in minutes.

## Prerequisites

- NVIDIA GPU with CUDA 11.0+ support
- Docker and Docker Compose
- At least 50GB of free disk space
- 8GB+ GPU memory recommended

## Quick Start

### 1. Launch SparkTrainer

```bash
docker-compose up -d
```

Access the web interface at `http://localhost:80`

### 2. Create Your First Dataset

1. Navigate to **Datasets** in the sidebar
2. Click **+ New Dataset**
3. Upload your training data (images, videos, text files)
4. SparkTrainer will automatically:
   - Generate manifests
   - Run quality checks
   - Create dataset cards

### 3. Start a Training Job

1. Go to **Dashboard** and click **Create Job**
2. Select your dataset
3. Choose a model template (e.g., `llama-2-7b`)
4. Select a training recipe (e.g., `lora`)
5. Configure hyperparameters
6. Click **Launch**

### 4. Monitor Training

Watch your training progress in real-time:
- GPU utilization and memory
- Training loss and metrics
- Power consumption
- ETA to completion

### 5. Export Your Model

Once training completes:
1. Go to **Models** page
2. Find your trained model
3. Click **Export to HuggingFace** to share publicly
4. Or download locally for inference

## Next Steps

- [Dashboard Guide](#pages/dashboard) - Learn about all dashboard widgets
- [Training Recipes](#recipes) - Explore different training strategies
- [GPU Management](#admin/gpu-management) - Optimize GPU utilization
