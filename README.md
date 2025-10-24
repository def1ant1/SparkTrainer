# DGX AI Trainer

A comprehensive web-based application for training and fine-tuning AI models on NVIDIA DGX Spark systems. This platform provides an intuitive interface for managing machine learning training jobs with support for PyTorch, TensorFlow, and Hugging Face Transformers.

## Features

### Core Capabilities
- **Train from Scratch**: Create custom neural networks with configurable architectures
- **Fine-tune Models**: Fine-tune pre-trained models for your specific tasks
- **Multi-Framework Support**: PyTorch, TensorFlow, and Hugging Face Transformers
- **Real-time Monitoring**: Track training progress, metrics, and logs in real-time
- **GPU Management**: Monitor GPU utilization and manage training job allocation
- **Job Queue**: Efficient job scheduling and management system

### Supported Architectures
- Custom fully-connected networks
- ResNet-style convolutional networks
- Pre-trained models (ResNet, VGG, DenseNet, BERT, GPT-2, etc.)
- Custom transformer architectures

## Architecture

```
dgx-ai-trainer/
├── backend/                 # Flask API server
│   ├── app.py              # Main API application
│   └── requirements.txt    # Python dependencies
├── frontend/               # React web interface
│   ├── src/
│   │   ├── App.jsx        # Main React application
│   │   ├── main.jsx       # Entry point
│   │   └── index.css      # Styles
│   ├── package.json
│   └── index.html
├── training_scripts/       # Training implementations
│   ├── train_pytorch.py
│   ├── finetune_pytorch.py
│   └── train_huggingface.py
├── jobs/                   # Job metadata and state
├── models/                 # Trained model checkpoints
└── logs/                   # Training logs
```

## Installation

### Prerequisites
- Python 3.8+
- Node.js 16+
- NVIDIA GPU with CUDA support (recommended)
- DGX Spark system (or any CUDA-capable system)

### Backend Setup

1. **Create a Python virtual environment**:
```bash
cd /home/claude/dgx-ai-trainer/backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. **Install Python dependencies**:
```bash
pip install -r requirements.txt
```

3. **Start the Flask API server**:
```bash
python app.py
```

The API server will start on `http://localhost:5000`

### Frontend Setup

1. **Install Node.js dependencies**:
```bash
cd /home/claude/dgx-ai-trainer/frontend
npm install
```

2. **Start the development server**:
```bash
npm run dev
```

The web interface will be available at `http://localhost:3000`

## Usage Guide

### 1. Dashboard
Access the dashboard to view:
- Number of available GPUs
- Running and queued jobs
- GPU memory utilization
- Saved models

### 2. Creating a Training Job

#### Training from Scratch
1. Click "Create New Training Job"
2. Select "Train from Scratch"
3. Choose a framework (PyTorch, TensorFlow, or Hugging Face)
4. Configure model parameters:
   - **Epochs**: Number of training iterations
   - **Batch Size**: Samples per training step
   - **Learning Rate**: Optimization step size
   - **Architecture**: Model structure (custom, ResNet, etc.)
   - **Hidden Layers**: Network depth and width
   - **Number of Classes**: Output dimension

5. Click "Create Training Job"

#### Fine-tuning a Model
1. Click "Create New Training Job"
2. Select "Fine-tune"
3. Choose a framework
4. Specify base model:
   - PyTorch: `resnet18`, `resnet50`, `vgg16`, etc.
   - Hugging Face: `bert-base-uncased`, `gpt2`, etc.
5. Configure fine-tuning parameters:
   - Lower learning rate (e.g., 0.0001)
   - Smaller number of epochs
   - Optional layer freezing

6. Click "Create Training Job"

### 3. Monitoring Jobs

View all jobs from the Jobs page:
- **Status tracking**: queued, running, completed, failed, cancelled
- **Real-time logs**: View training output and metrics
- **Job details**: Configuration, timestamps, and results
- **Cancel running jobs**: Stop jobs if needed

### 4. Using Trained Models

After training completes, models are saved in `/home/claude/dgx-ai-trainer/models/{job_id}/`:
- `model.pth` or `best_model.pth`: Model weights
- `config.json`: Model configuration and metadata

Load models for inference:
```python
import torch

# Load PyTorch model
checkpoint = torch.load('models/{job_id}/model.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# Load Hugging Face model
from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained('models/{job_id}')
```

## API Reference

### Endpoints

#### System Information
```
GET /api/health
GET /api/system/info
GET /api/frameworks
```

#### Job Management
```
GET /api/jobs              # List all jobs
GET /api/jobs/{id}         # Get job details
POST /api/jobs             # Create new job
POST /api/jobs/{id}/cancel # Cancel job
```

#### Models
```
GET /api/models            # List saved models
```

### Job Configuration Schema

```json
{
  "name": "My Training Job",
  "type": "train",  // or "finetune"
  "framework": "pytorch",  // pytorch, tensorflow, huggingface
  "config": {
    "epochs": 10,
    "batch_size": 32,
    "learning_rate": 0.001,
    "architecture": "custom",
    "input_size": 784,
    "output_size": 10,
    "hidden_layers": [512, 256, 128],
    "num_classes": 10,
    
    // Fine-tuning specific
    "model_name": "resnet18",
    "freeze_layers": true,
    "use_scheduler": true
  }
}
```

## Advanced Configuration

### Custom Model Architectures

Modify `training_scripts/train_pytorch.py` to add custom architectures:

```python
class MyCustomModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Define your architecture
        pass
    
    def forward(self, x):
        # Define forward pass
        pass
```

### Data Loading

Replace dummy data in training scripts with your actual datasets:

```python
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self, data_path):
        # Load your data
        pass
    
    def __getitem__(self, idx):
        # Return sample
        pass

# In training script
dataset = MyDataset('/path/to/data')
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
```

### Distributed Training

For multi-GPU training on DGX Spark:

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

# Initialize distributed training
dist.init_process_group(backend='nccl')
model = DistributedDataParallel(model)
```

## Troubleshooting

### Common Issues

1. **CUDA out of memory**
   - Reduce batch size
   - Use gradient accumulation
   - Enable mixed precision training

2. **Job stuck in queue**
   - Check backend logs: `tail -f logs/{job_id}.log`
   - Verify GPU availability: `nvidia-smi`
   - Restart backend server

3. **Connection errors**
   - Ensure backend is running on port 5000
   - Check firewall settings
   - Verify API_BASE URL in frontend

## Performance Optimization

### For DGX Spark Systems

1. **Enable TensorFloat-32 (TF32)**:
```python
torch.backends.cuda.matmul.allow_tf32 = True
```

2. **Use mixed precision training**:
```python
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

with autocast():
    output = model(input)
    loss = criterion(output, target)
```

3. **Optimize data loading**:
```python
train_loader = DataLoader(
    dataset,
    batch_size=batch_size,
    num_workers=4,
    pin_memory=True
)
```

## Production Deployment

For production use on DGX Spark:

1. **Use a production WSGI server**:
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 backend.app:app
```

2. **Set up process management**:
```bash
# Using systemd
sudo systemctl enable dgx-ai-trainer
sudo systemctl start dgx-ai-trainer
```

3. **Configure reverse proxy** (nginx):
```nginx
location /api {
    proxy_pass http://localhost:5000;
}

location / {
    proxy_pass http://localhost:3000;
}
```

## Security Considerations

- Enable authentication/authorization for production
- Use HTTPS for all communications
- Implement rate limiting
- Validate all user inputs
- Secure model checkpoints and logs

## Contributing

To extend this system:

1. Add new frameworks: Create training scripts in `training_scripts/`
2. Add new architectures: Extend model classes in training scripts
3. Enhance UI: Add components to `frontend/src/`
4. Add metrics: Update job tracking in `backend/app.py`

## License

This project is provided as-is for use on NVIDIA DGX systems.

## Support

For issues related to:
- **DGX Spark hardware**: Contact NVIDIA support
- **Application bugs**: Check logs in `/home/claude/dgx-ai-trainer/logs/`
- **Training issues**: Review training script output

## Acknowledgments

Built with:
- React + Vite + Tailwind CSS
- Flask + PyTorch + Transformers
- NVIDIA CUDA and cuDNN
