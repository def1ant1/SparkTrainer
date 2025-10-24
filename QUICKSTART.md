# DGX AI Trainer - Quick Start Guide

## 🚀 Getting Started in 5 Minutes

### Step 1: Extract the Application
```bash
tar -xzf dgx-ai-trainer.tar.gz
cd dgx-ai-trainer
```

### Step 2: Run Setup
```bash
chmod +x setup.sh
./setup.sh
```

This will:
- Create Python virtual environment
- Install all dependencies (PyTorch, TensorFlow, Transformers)
- Install React frontend dependencies
- Create startup scripts

### Step 3: Start the Application
```bash
./start.sh
```

Or start components separately:
```bash
# Terminal 1 - Backend
./start_backend.sh

# Terminal 2 - Frontend
./start_frontend.sh
```

### Step 4: Open in Browser
Navigate to: **http://localhost:3000**

---

## 📊 What You Can Do

### 1. Train Models from Scratch
- Custom neural networks
- ResNet architectures
- Transformer models
- Fully configurable layers and hyperparameters

### 2. Fine-tune Pre-trained Models
- **PyTorch**: ResNet18/50, VGG16, DenseNet121
- **Hugging Face**: BERT, GPT-2, T5, LLaMA
- Transfer learning with layer freezing
- Adaptive learning rates

### 3. Monitor Training
- Real-time job status
- Training logs and metrics
- GPU utilization tracking
- Job queue management

---

## 🎯 Quick Training Examples

### Example 1: Train a Simple Classifier
```json
{
  "name": "My First Model",
  "type": "train",
  "framework": "pytorch",
  "config": {
    "epochs": 10,
    "batch_size": 32,
    "learning_rate": 0.001,
    "num_classes": 10
  }
}
```

### Example 2: Fine-tune BERT
```json
{
  "name": "BERT Fine-tune",
  "type": "finetune",
  "framework": "huggingface",
  "config": {
    "model_name": "bert-base-uncased",
    "epochs": 3,
    "batch_size": 16,
    "learning_rate": 2e-5,
    "num_classes": 2
  }
}
```

---

## 🗂️ Project Structure

```
dgx-ai-trainer/
├── backend/              # Flask API
│   ├── app.py           # Main server
│   └── requirements.txt # Python deps
├── frontend/            # React UI
│   ├── src/
│   │   └── App.jsx     # Main component
│   └── package.json
├── training_scripts/    # Training code
│   ├── train_pytorch.py
│   ├── finetune_pytorch.py
│   └── train_huggingface.py
├── jobs/                # Job metadata
├── models/              # Saved models
├── logs/                # Training logs
└── setup.sh            # Setup script
```

---

## 💡 Key Features

✅ **Multi-Framework Support**: PyTorch, TensorFlow, Hugging Face
✅ **Real-time Monitoring**: Track jobs and GPU usage
✅ **Job Queue System**: Automatic scheduling
✅ **Model Management**: Save and organize trained models
✅ **Web Interface**: Easy-to-use React dashboard
✅ **RESTful API**: Programmatic access
✅ **DGX Optimized**: Built for NVIDIA DGX Spark

---

## 🔧 Configuration Options

### Training Parameters
- **epochs**: Number of training iterations
- **batch_size**: Samples per gradient update
- **learning_rate**: Optimization step size
- **optimizer**: adam, sgd, adamw
- **architecture**: Model structure

### Fine-tuning Options
- **model_name**: Base model to fine-tune
- **freeze_layers**: Freeze pre-trained layers
- **use_scheduler**: Learning rate scheduling
- **weight_decay**: L2 regularization

---

## 📈 Monitoring Your Jobs

### Dashboard View
- Total GPUs available
- Running jobs count
- Queued jobs
- Saved models
- GPU memory usage

### Job Details
- Status: queued → running → completed
- Configuration
- Real-time logs
- Training metrics
- Cancel option

---

## 🐳 Docker Deployment (Optional)

For containerized deployment:

```bash
# Build and start containers
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

Access at: **http://localhost**

---

## 🛠️ Customization

### Add Your Own Data
Edit training scripts in `training_scripts/`:
```python
# Replace dummy data with your dataset
dataset = YourCustomDataset('/path/to/data')
train_loader = DataLoader(dataset, batch_size=batch_size)
```

### Add Custom Architectures
Extend model classes in training scripts:
```python
class MyCustomModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Your architecture here
```

### Add New Frameworks
Create new training script:
```bash
cp training_scripts/train_pytorch.py training_scripts/train_myframework.py
# Modify for your framework
```

---

## 🔍 Troubleshooting

### Port Already in Use
```bash
# Kill process on port 5000
sudo lsof -t -i:5000 | xargs kill -9

# Kill process on port 3000
sudo lsof -t -i:3000 | xargs kill -9
```

### CUDA Out of Memory
- Reduce batch_size
- Enable mixed precision
- Use gradient accumulation

### Module Not Found
```bash
cd backend
source venv/bin/activate
pip install -r requirements.txt
```

### Frontend Won't Start
```bash
cd frontend
rm -rf node_modules
npm install
```

---

## 📚 Additional Resources

- **README.md**: Comprehensive documentation
- **EXAMPLES.md**: Training configuration examples
- **API Docs**: Backend API reference
- **logs/**: Check training logs for errors

---

## 🎓 Best Practices

1. **Start Small**: Test with small models first
2. **Monitor GPUs**: Use `nvidia-smi` to check utilization
3. **Save Checkpoints**: Models auto-save after training
4. **Use Validation**: Split your data properly
5. **Track Experiments**: Name jobs clearly

---

## 📞 Support

For issues:
1. Check `logs/` directory for error messages
2. Verify GPU availability: `nvidia-smi`
3. Check backend logs: `tail -f logs/[job-id].log`
4. Review training script output

---

## 🚀 Next Steps

1. ✅ Complete setup
2. ✅ Start the application
3. ✅ Create your first training job
4. ✅ Monitor training progress
5. ✅ Use your trained model

**Happy Training! 🎉**
