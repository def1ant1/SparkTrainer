# SparkTrainer Complete Tutorial

Learn SparkTrainer from start to finish with this comprehensive, hands-on tutorial.

## What You'll Learn

By the end of this tutorial, you'll know how to:
1. Set up and navigate SparkTrainer
2. Create and manage projects
3. Prepare datasets for training
4. Train models with different techniques (LoRA, full fine-tuning)
5. Monitor training progress
6. Evaluate model performance
7. Export and use your trained models

**Time Required:** 30-45 minutes

**Prerequisites:** SparkTrainer installed and running ([see QUICKSTART.md](../QUICKSTART.md))

---

## Part 1: Getting Started (5 minutes)

### Step 1: Launch SparkTrainer

Open your terminal and start SparkTrainer:

```bash
cd SparkTrainer
docker compose up -d
```

Wait about 60 seconds for all services to start.

### Step 2: Access the Interface

Open your browser to **http://localhost:3000**

You should see the SparkTrainer dashboard with:
- System metrics (GPU count, memory)
- Quick access buttons
- Recent jobs (empty for now)

### Step 3: Tour the Interface

Click through each page in the sidebar:

1. **Dashboard** - Overview and system status
2. **Jobs** - All training jobs (running, completed, failed)
3. **Experiments** - Group related training runs
4. **Datasets** - Manage training data
5. **Models** - Browse trained models
6. **Leaderboard** - Compare model performance

**Tip:** Hover over any button or icon to see tooltips!

---

## Part 2: Your First Project (3 minutes)

### Step 1: Create a Project

Projects help organize your work. Let's create one for this tutorial.

1. Click **Projects** in the sidebar
2. Click **+ New Project** button
3. Fill in:
   - **Name:** "Tutorial Project"
   - **Description:** "Learning SparkTrainer basics"
4. Click **Create**

You should see your project in the list!

### Step 2: Understand Projects

Projects are like folders. They can contain:
- Multiple datasets
- Multiple experiments
- All related models

**Best Practice:** Create one project per task (e.g., "Sentiment Analysis", "Image Classification", "Chatbot").

---

## Part 3: Preparing a Dataset (7 minutes)

### Example 1: Text Dataset for NLP

Let's create a simple sentiment analysis dataset.

**Step 1: Create the dataset folder**

```bash
# In your terminal
mkdir -p datasets/sentiment_demo
```

**Step 2: Create training data**

```bash
cat > datasets/sentiment_demo/train.jsonl << 'EOF'
{"text": "I love this product! It's amazing!", "label": "positive"}
{"text": "Terrible experience. Would not recommend.", "label": "negative"}
{"text": "Pretty good, met my expectations.", "label": "positive"}
{"text": "Waste of money. Very disappointed.", "label": "negative"}
{"text": "Excellent quality and fast shipping!", "label": "positive"}
{"text": "Not worth the price. Poor quality.", "label": "negative"}
{"text": "Absolutely fantastic! Best purchase ever!", "label": "positive"}
{"text": "Horrible customer service. Avoid.", "label": "negative"}
{"text": "Works as advertised. Happy with it.", "label": "positive"}
{"text": "Broke after one week. Don't buy.", "label": "negative"}
EOF
```

**Step 3: Create metadata file**

```bash
cat > datasets/sentiment_demo/manifest.json << 'EOF'
{
  "name": "sentiment_demo",
  "description": "Simple sentiment classification dataset",
  "type": "text_classification",
  "num_samples": 10,
  "labels": ["positive", "negative"]
}
EOF
```

**Step 4: Verify in UI**

1. Go to **Datasets** page
2. Click **Refresh** or restart backend:
   ```bash
   docker compose restart backend
   ```
3. You should see "sentiment_demo" in the list!
4. Click on it to see samples

### Example 2: Using Video Data

SparkTrainer has a special wizard for video datasets.

**Step 1: Get sample videos**

```bash
# Download a sample video (or use your own)
mkdir -p datasets/video_demo
# Place your .mp4 files in datasets/video_demo/
```

**Step 2: Use the Video Wizard**

1. Go to **Datasets** → **Dataset Wizard**
2. **Step 1 - Upload:**
   - Drag and drop your video files
   - Or click to select files
3. **Step 2 - Configure:**
   - **FPS:** 1 (extract 1 frame per second)
   - **Resolution:** 224x224
   - **Audio Transcription:** Enable (uses Whisper)
   - **Captioning Backend:** BLIP-2 (fast and good)
   - **Scene Detection:** Optional
4. **Step 3 - Integrity Check:**
   - Wait for validation (shows video info)
5. **Step 4 - Process:**
   - Click **Start Processing**
   - Grab coffee! This takes a few minutes.

The wizard will:
- Extract frames from videos
- Generate captions for each frame
- Transcribe audio to text
- Create a `manifest.jsonl` file
- Validate everything

---

## Part 4: Training Your First Model (10 minutes)

Now the fun part - training a model!

### Using the Web UI (Easiest)

**Step 1: Start Training Wizard**

1. Go to **Dashboard**
2. Click **Create Job** button (big, blue, can't miss it!)
3. The Training Wizard opens

**Step 2: Select Data**

- **Project:** Select "Tutorial Project"
- **Dataset:** Select "sentiment_demo"
- **Preview:** Check that data looks correct
- Click **Next**

**Step 3: Choose Model**

- Select **"From HuggingFace"**
- **Model ID:** Enter `distilbert-base-uncased`
  - This is a small, fast model perfect for learning!
  - Only 66M parameters, trains in minutes
- Click **Next**

**Step 4: Pick Recipe**

- Select **"LoRA"** (efficient fine-tuning)
- Why LoRA? It's:
  - Fast (only trains 0.1% of parameters)
  - Memory efficient (works on 4GB GPUs)
  - Quality (often as good as full fine-tuning)
- Click **Next**

**Step 5: Configure Parameters**

Let's use these settings for fast training:

- **Epochs:** 3
- **Batch Size:** 4 (reduce to 2 if OOM error)
- **Learning Rate:** 2e-4 (default is fine)
- **LoRA Rank:** 8
- **LoRA Alpha:** 16
- **Max Length:** 128 (short sequences = faster)

**Advanced Options** (expand if interested):
- **Gradient Accumulation Steps:** 1
- **Warmup Steps:** 50
- **Save Strategy:** "epoch" (saves after each epoch)

Click **Launch Training**!

**Step 6: Watch It Train**

You'll be redirected to the Job Details page. Watch:

- **Status:** Changes from "pending" → "running"
- **Live Logs:** Scroll to bottom for real-time updates
- **GPU Utilization:** Should be 50-100%
- **Training Loss:** Should decrease over time
- **ETA:** Estimated time remaining

**Expected Timeline:**
- CPU: ~15 minutes
- GPU (8GB): ~2-3 minutes

---

## Part 5: Monitoring Training (3 minutes)

While your model trains, let's explore monitoring tools.

### Dashboard View

1. Go to **Dashboard**
2. See your job in "Active Jobs" widget
3. Metrics update every few seconds

### Jobs Page

1. Go to **Jobs** page
2. Your job appears at the top
3. Click on it for detailed view with tabs:
   - **Overview:** Status, config, progress
   - **Logs:** Full training logs
   - **Metrics:** Charts (loss, accuracy, etc.)
   - **Checkpoints:** Saved model snapshots

### MLflow UI

For deep metrics analysis:

1. Open **http://localhost:5001**
2. Find your experiment
3. See detailed metrics:
   - Loss curves
   - Learning rate schedule
   - Gradient norms
   - Custom metrics

### Real-time Terminal Logs

```bash
# In terminal, watch logs
docker compose logs -f worker

# Or tail job log directly
tail -f logs/job_*.log
```

---

## Part 6: Evaluating Your Model (5 minutes)

Training complete? Let's evaluate the model!

### Quick Evaluation

**Step 1: Go to Models Page**

1. Click **Models** in sidebar
2. Find your trained model (named after the job)
3. Click on it to see details

**Step 2: Test Inference**

Let's test if it works:

```python
# In Python or Jupyter notebook
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# Load your model
model_path = "./models/job_1"  # Replace with actual job ID
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Test it
text = "This is absolutely wonderful! I'm so happy!"
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)
predictions = torch.softmax(outputs.logits, dim=1)

print(f"Positive: {predictions[0][1]:.2%}")
print(f"Negative: {predictions[0][0]:.2%}")
```

Expected output:
```
Positive: 95.3%
Negative: 4.7%
```

### Benchmark Evaluation

For standardized metrics:

**For NLP Models - MMLU:**

```bash
docker compose exec backend python -m spark_trainer.evaluation.mmlu_eval \
    --model-path ./models/job_1 \
    --output-dir ./eval_results \
    --num-fewshot 5 \
    --subjects abstract_algebra anatomy
```

**Via UI:**

1. Go to **Models** → Your model
2. Click **Evaluate**
3. Select **MMLU** benchmark
4. Choose subjects (or "All")
5. Click **Run Evaluation**

Results will appear on the **Leaderboard** page!

---

## Part 7: Exporting Your Model (3 minutes)

Now that you have a trained model, let's export it.

### Option 1: Download Locally

**Via UI:**

1. Go to **Models** → Your model
2. Click **Download**
3. Get a ZIP file with:
   - Model weights
   - Configuration files
   - Tokenizer files

**Via Terminal:**

```bash
# Models are already on disk!
ls models/job_*/

# Copy to your project
cp -r models/job_1 ~/my_project/sentiment_model
```

### Option 2: Export to HuggingFace Hub

Share your model with the world!

**Prerequisites:**
- HuggingFace account
- Access token with write permissions

**Via UI:**

1. Go to **Models** → Your model
2. Click **Export to HuggingFace**
3. Enter:
   - **Username:** your_hf_username
   - **Repository:** sentiment-model-demo
   - **Token:** your_token (from https://huggingface.co/settings/tokens)
4. Click **Export**

**Via CLI:**

```bash
docker compose exec backend python << 'EOF'
from huggingface_hub import HfApi, login

login(token="your_token_here")

api = HfApi()
api.upload_folder(
    folder_path="./models/job_1",
    repo_id="your_username/sentiment-model-demo",
    repo_type="model"
)

print("Model uploaded to HuggingFace!")
EOF
```

Your model will be public at:
`https://huggingface.co/your_username/sentiment-model-demo`

### Option 3: Deploy for Inference

Set up an inference server:

```python
# Save this as serve_model.py
from spark_trainer.inference.serving_adapters import create_serving_adapter

adapter = create_serving_adapter(
    backend="vllm",  # Fast inference engine
    model_path="./models/job_1",
    host="0.0.0.0",
    port=8000
)

adapter.serve()
```

Run it:
```bash
python serve_model.py
```

Test it:
```bash
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "sentiment-model",
    "prompt": "This product is terrible!",
    "max_tokens": 10
  }'
```

---

## Part 8: Advanced Techniques (7 minutes)

Now that you understand the basics, let's explore advanced features.

### Technique 1: Full Fine-tuning vs LoRA

**When to use LoRA:**
- Large models (7B+ parameters)
- Limited GPU memory
- Quick experiments
- Good enough performance

**When to use full fine-tuning:**
- Small models (<1B parameters)
- Maximum performance needed
- Plenty of GPU memory
- More training data

**Try full fine-tuning:**

Same wizard, but in Step 3:
- Select **"Full Fine-tune"** instead of LoRA
- Increase epochs to 5-10
- May need to reduce batch size

### Technique 2: Hyperparameter Optimization

SparkTrainer can automatically find the best hyperparameters!

```python
# Save as hpo_search.py
from spark_trainer.hpo import run_hyperparameter_search

search_space = {
    "learning_rate": [1e-5, 5e-5, 1e-4, 2e-4],
    "batch_size": [4, 8, 16],
    "lora_r": [4, 8, 16],
    "num_epochs": [3, 5, 10]
}

best_params = run_hyperparameter_search(
    base_model="distilbert-base-uncased",
    dataset="sentiment_demo",
    search_space=search_space,
    num_trials=20,
    optimization_metric="eval_accuracy"
)

print(f"Best parameters: {best_params}")
```

### Technique 3: Multi-GPU Training

If you have multiple GPUs:

**Edit .env:**
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3  # Use 4 GPUs
```

**In training config:**
```yaml
distributed: true
world_size: 4  # Number of GPUs
training_mode: "fsdp"  # Fully Sharded Data Parallel
```

**Expected speedup:**
- 2 GPUs: ~1.8x faster
- 4 GPUs: ~3.5x faster
- 8 GPUs: ~7x faster

### Technique 4: Gradient Checkpointing

For large models that don't fit in memory:

```yaml
# In training config
gradient_checkpointing: true

# Trade-off:
# - Uses 30-40% less memory
# - Slows training by ~20%
```

### Technique 5: Mixed Precision Training

Speed up training with fp16:

```yaml
# In training config
fp16: true  # For older GPUs (V100, T4)
# Or:
bf16: true  # For newer GPUs (A100, H100) - better numerical stability
```

**Expected speedup:**
- 1.5-2x faster
- Uses slightly less memory

---

## Part 9: Real-World Example (5 minutes)

Let's train a real model for a practical task.

### Example: Fine-tune LLaMA for Instruction Following

**Step 1: Get Dataset**

```bash
# Download Alpaca dataset (instruction-following)
docker compose exec backend python << 'EOF'
from datasets import load_dataset

dataset = load_dataset("tatsu-lab/alpaca")

# Save as JSONL
with open("/app/datasets/alpaca/train.jsonl", "w") as f:
    for item in dataset["train"]:
        f.write(json.dumps({
            "instruction": item["instruction"],
            "input": item["input"],
            "output": item["output"]
        }) + "\n")

print("Dataset downloaded!")
EOF
```

**Step 2: Train with QLoRA**

Why QLoRA?
- LLaMA-2-7B has 7 billion parameters
- Normal training needs 28GB GPU memory
- QLoRA needs only 6-8GB!

```python
# In Training Wizard or config:
base_model: "meta-llama/Llama-2-7b-hf"
dataset: "alpaca"
recipe: "qlora"  # 4-bit quantization

lora:
  r: 16
  alpha: 32
  target_modules: ["q_proj", "v_proj", "k_proj", "o_proj"]

training:
  num_epochs: 3
  batch_size: 4
  gradient_accumulation_steps: 4  # Effective batch size: 16
  learning_rate: 2e-4
  max_length: 512

optimization:
  fp16: false  # 4-bit already
  gradient_checkpointing: true
```

**Expected Training Time:**
- 1x A100 (40GB): ~6 hours
- 1x RTX 4090 (24GB): ~10 hours
- 1x RTX 3090 (24GB): ~12 hours
- 4x A100: ~1.5 hours

**Step 3: Test Your Model**

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    load_in_4bit=True,
    device_map="auto"
)

# Load LoRA adapters
model = PeftModel.from_pretrained(base_model, "./models/job_123")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# Test instruction following
prompt = """### Instruction:
Write a Python function to calculate fibonacci numbers.

### Response:
"""

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=200)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(response)
```

Expected output:
```python
Here's a Python function to calculate Fibonacci numbers:

def fibonacci(n):
    if n <= 1:
        return n
    else:
        return fibonacci(n-1) + fibonacci(n-2)

# Example usage:
print(fibonacci(10))  # Output: 55
```

---

## Part 10: Best Practices & Tips (3 minutes)

### Tip 1: Start Small

Always test with a small model first:
- Use DistilBERT instead of BERT
- Try 1-2 epochs initially
- Use small batch sizes

Once it works, scale up!

### Tip 2: Monitor Everything

Keep an eye on:
- **GPU utilization** (should be 70-100%)
- **Training loss** (should decrease smoothly)
- **Validation loss** (should decrease, not increase)
- **Memory usage** (should be stable)

### Tip 3: Save Checkpoints

Enable frequent checkpointing:
```yaml
save_strategy: "steps"
save_steps: 500
keep_last_n_checkpoints: 3
```

Why? If training crashes, you don't start over!

### Tip 4: Use Experiment Tracking

Create experiments for related training runs:

1. **Experiments** → **New Experiment**
2. Name it: "Sentiment Model v1"
3. Run multiple jobs with different hyperparameters
4. Compare results easily

### Tip 5: Version Your Datasets

When you improve your dataset:

```bash
# In UI: Datasets → Your dataset → Create Version
# Creates a snapshot: sentiment_demo_v2
```

Now you can compare:
- Model trained on v1
- Model trained on v2
- Which data is better?

### Tip 6: Use the Leaderboard

After training multiple models:

1. Evaluate all with same benchmark
2. Check leaderboard for rankings
3. Identify best model
4. Deploy winner to production

### Tip 7: Clean Up Regularly

Models and datasets use lots of space:

```bash
# Delete old jobs
# Via UI: Jobs → Select old jobs → Bulk Delete

# Clean up disk
docker system prune -a
```

---

## What's Next?

Congratulations! You now know how to use SparkTrainer. 🎉

### Continue Learning:

**Guides:**
- [LoRA Deep Dive](app/recipes/lora.md) - Master efficient fine-tuning
- [Vision-Language Models](VISION_LANGUAGE.md) - Train multimodal models
- [Distributed Training](distributed_training.md) - Use multiple GPUs
- [Production Deployment](DEPLOYMENT.md) - Deploy at scale

**Advanced Topics:**
- [Model Registry](MODEL_REGISTRY.md) - Manage model lifecycle
- [A/B Testing](AB_TESTING.md) - Test models in production
- [Safety Gates](GATING_MECHANISMS.md) - Ensure model safety
- [Custom Recipes](CUSTOM_RECIPES.md) - Create your own training recipes

**API:**
- [REST API Reference](api.md) - Automate with API calls
- [Python SDK](../sdk/python/README.md) - Programmatic access
- [TypeScript SDK](../sdk/typescript/README.md) - For Node.js apps

### Join the Community:

- [GitHub Discussions](https://github.com/def1ant1/SparkTrainer/discussions) - Ask questions
- [Contributing Guide](../CONTRIBUTING.md) - Help improve SparkTrainer
- Discord [Coming Soon] - Chat with other users

### Share Your Success:

Trained something cool? Share it!
- Tweet with #SparkTrainer
- Post on [Show HN](https://news.ycombinator.com/)
- Write a blog post
- Create a YouTube tutorial

---

## Quick Reference

### Common Commands

```bash
# Start SparkTrainer
docker compose up -d

# Stop SparkTrainer
docker compose down

# View logs
docker compose logs -f worker

# Restart service
docker compose restart backend

# Initialize database
docker compose exec backend python init_db.py --sample-data

# Run training via CLI
docker compose exec backend python -m spark_trainer.train --config my_config.yaml

# Export model
docker compose exec backend python -m spark_trainer.export --model-path ./models/job_123 --output ./exports/my_model

# Evaluate model
docker compose exec backend python -m spark_trainer.evaluation.mmlu_eval --model-path ./models/job_123
```

### Keyboard Shortcuts (in UI)

- `Ctrl+K` - Command palette
- `Ctrl+N` - New job
- `Ctrl+P` - New project
- `Ctrl+D` - New dataset
- `Ctrl+/` - Toggle sidebar
- `?` - Show all shortcuts

---

**Happy Training! 🚀**

Questions? Check the [Troubleshooting Guide](TROUBLESHOOTING.md) or [ask for help](https://github.com/def1ant1/SparkTrainer/discussions).
