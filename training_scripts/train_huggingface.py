import argparse
import json
import os
from datetime import datetime
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
    TrainerCallback
)
from datasets import Dataset
import numpy as np

# Resolve project base directory (two levels up from this script)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def create_dummy_dataset(config):
    """Create dummy dataset for demonstration"""
    num_samples = config.get('num_samples', 1000)
    max_length = config.get('max_length', 128)
    num_classes = config.get('num_classes', 2)
    
    # Generate random text-like data
    data = {
        'text': [f"Sample text {i} with some random content for training" for i in range(num_samples)],
        'label': np.random.randint(0, num_classes, num_samples).tolist()
    }
    
    return Dataset.from_dict(data)

def train_transformer(config, job_id):
    """Train or fine-tune a transformer model"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Optional experiment tracking (best-effort)
    tracking = config.get('tracking', {}) if isinstance(config.get('tracking'), dict) else {}
    wandb_run = None
    try:
        if tracking.get('wandb') or os.environ.get('WANDB_API_KEY'):
            import wandb  # type: ignore
            wandb_run = wandb.init(project=tracking.get('project', 'dgx-ai-trainer'), config=config, reinit=True)
    except Exception:
        wandb_run = None

    # Model configuration
    model_name = config.get('model_name', 'bert-base-uncased')
    task_type = config.get('task_type', 'classification')  # 'classification' or 'generation'
    num_classes = config.get('num_classes', 2)
    
    print(f"Loading model: {model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load model based on task
    if task_type == 'classification':
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_classes
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name)
    
    print(f"Model loaded: {model_name}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Prepare dataset
    print("Preparing dataset...")
    dataset = create_dummy_dataset(config)
    
    # Split dataset
    train_test_split = dataset.train_test_split(test_size=0.2)
    train_dataset = train_test_split['train']
    eval_dataset = train_test_split['test']
    
    # Tokenize function
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            padding='max_length',
            truncation=True,
            max_length=config.get('max_length', 128)
        )
    
    # Tokenize datasets
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    eval_dataset = eval_dataset.map(tokenize_function, batched=True)
    
    # Set format
    train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
    eval_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
    
    # Training arguments
    output_dir = os.path.join(BASE_DIR, 'models', job_id)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        learning_rate=config.get('learning_rate', 2e-5),
        per_device_train_batch_size=config.get('batch_size', 16),
        per_device_eval_batch_size=config.get('batch_size', 16),
        num_train_epochs=config.get('epochs', 3),
        weight_decay=config.get('weight_decay', 0.01),
        warmup_steps=config.get('warmup_steps', 500),
        logging_dir=os.path.join(BASE_DIR, 'logs', job_id),
        logging_steps=10,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
    )
    
    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Compute metrics function
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        if task_type == 'classification':
            predictions = np.argmax(predictions, axis=1)
            accuracy = (predictions == labels).mean()
            return {'accuracy': accuracy}
        return {}
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if task_type == 'classification' else None,
    )

    class JsonLoggerCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            try:
                payload = {
                    'kind': 'log',
                    'step': int(state.global_step),
                    'epoch': float(state.epoch) if state.epoch is not None else None,
                    'logs': logs or {},
                    'time': datetime.now().isoformat(),
                }
                print('METRIC:' + json.dumps(payload), flush=True)
            except Exception:
                pass
        def on_epoch_end(self, args, state, control, **kwargs):
            try:
                payload = {
                    'kind': 'epoch',
                    'epoch': float(state.epoch) if state.epoch is not None else None,
                    'step': int(state.global_step),
                    'time': datetime.now().isoformat(),
                }
                print('METRIC:' + json.dumps(payload), flush=True)
            except Exception:
                pass

    trainer.add_callback(JsonLoggerCallback())
    
    # Train
    print(f"\nStarting training for {config.get('epochs', 3)} epochs...")
    train_result = trainer.train()
    
    # Evaluate
    print("\nEvaluating model...")
    metrics = trainer.evaluate()
    
    print("\nTraining Metrics:")
    print(f"  Train Loss: {train_result.training_loss:.4f}")
    print(f"  Eval Loss: {metrics['eval_loss']:.4f}")
    if 'eval_accuracy' in metrics:
        print(f"  Eval Accuracy: {metrics['eval_accuracy']:.4f}")
    
    # Save model
    print(f"\nSaving model to: {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Save config
    config_path = os.path.join(output_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump({
            'name': config.get('name', f'Transformer Model {job_id[:8]}'),
            'framework': 'huggingface',
            'base_model': model_name,
            'task_type': task_type,
            'created': datetime.now().isoformat(),
            'parameters': sum(p.numel() for p in model.parameters()),
            'eval_loss': metrics['eval_loss'],
            'eval_accuracy': metrics.get('eval_accuracy', None)
        }, f, indent=2)
    
    print("\nTraining completed successfully!")
    try:
        if wandb_run is not None:
            wandb_run.finish()
    except Exception:
        pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--job-id', required=True)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    
    config = json.loads(args.config)
    train_transformer(config, args.job_id)
