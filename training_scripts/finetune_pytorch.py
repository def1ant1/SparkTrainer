import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import argparse
import json
import os
from datetime import datetime
import torchvision.models as models
import random
import numpy as np

try:
    import mlflow
except Exception:
    mlflow = None  # type: ignore
try:
    import wandb
except Exception:
    wandb = None  # type: ignore

# Resolve project base directory (two levels up from this script)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_pretrained_model(config):
    """Load a pre-trained model"""
    
    model_source = config.get('model_source', 'torchvision')
    model_name = config.get('model_name', 'resnet18')
    
    if model_source == 'torchvision':
        # Load from torchvision
        if model_name == 'resnet18':
            model = models.resnet18(pretrained=True)
        elif model_name == 'resnet50':
            model = models.resnet50(pretrained=True)
        elif model_name == 'vgg16':
            model = models.vgg16(pretrained=True)
        elif model_name == 'densenet121':
            model = models.densenet121(pretrained=True)
        else:
            model = models.resnet18(pretrained=True)
        
        # Modify final layer for new task
        num_classes = config.get('num_classes', 10)
        
        if 'resnet' in model_name or 'densenet' in model_name:
            num_features = model.fc.in_features
            model.fc = nn.Linear(num_features, num_classes)
        elif 'vgg' in model_name:
            num_features = model.classifier[6].in_features
            model.classifier[6] = nn.Linear(num_features, num_classes)
    
    elif model_source == 'custom':
        # Load custom saved model
        model_path = config.get('model_path')
        checkpoint = torch.load(model_path)
        
        # Rebuild model architecture (simplified)
        from train_pytorch import CustomModel
        model = CustomModel(checkpoint['config'])
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Modify for new task if needed
        num_classes = config.get('num_classes')
        if num_classes:
            # Replace final layer
            input_features = list(model.network.children())[-1].in_features
            model.network[-1] = nn.Linear(input_features, num_classes)
    
    return model

def finetune_model(config, job_id):
    """Fine-tune a pre-trained model"""
    # Seed & determinism
    seed = config.get('seed')
    deterministic = bool(config.get('deterministic', False))
    if seed is not None:
        try:
            seed = int(seed)
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
            if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
            if deterministic:
                try:
                    torch.backends.cudnn.deterministic = True
                    torch.backends.cudnn.benchmark = False
                    torch.use_deterministic_algorithms(True)
                except Exception:
                    pass
        except Exception:
            pass

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load pre-trained model
    print("Loading pre-trained model...")
    model = load_pretrained_model(config).to(device)
    
    print(f"Model loaded: {config.get('model_name', 'custom')}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Freeze layers if specified
    freeze_layers = config.get('freeze_layers', False)
    if freeze_layers:
        # Freeze all layers except the last one
        for param in model.parameters():
            param.requires_grad = False
        
        # Unfreeze final layer
        if hasattr(model, 'fc'):
            for param in model.fc.parameters():
                param.requires_grad = True
        elif hasattr(model, 'classifier'):
            for param in model.classifier.parameters():
                param.requires_grad = True
        
        print("Froze all layers except final classification layer")
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Trainable parameters: {trainable_params:,}")
    
    # Setup optimizer with potentially different learning rate
    learning_rate = config.get('learning_rate', 0.0001)  # Lower LR for fine-tuning
    weight_decay = config.get('weight_decay', 0.0001)
    
    optimizer_name = config.get('optimizer', 'adam')
    if optimizer_name == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                              lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), 
                             lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    else:
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                               lr=learning_rate, weight_decay=weight_decay)
    
    # Learning rate scheduler
    use_scheduler = config.get('use_scheduler', True)
    if use_scheduler:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Create dummy data (replace with actual data loading)
    batch_size = config.get('batch_size', 32)
    num_samples = config.get('num_samples', 1000)
    num_classes = config.get('num_classes', 10)
    
    # For image models
    if config.get('model_source') == 'torchvision':
        # Dummy image data (3x224x224)
        X = torch.randn(num_samples, 3, 224, 224)
    else:
        # Flat features
        input_size = config.get('input_size', 784)
        X = torch.randn(num_samples, input_size)
    
    y = torch.randint(0, num_classes, (num_samples,))
    
    # Split into train/val
    train_size = int(0.8 * num_samples)
    train_dataset = TensorDataset(X[:train_size], y[:train_size])
    val_dataset = TensorDataset(X[train_size:], y[train_size:])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Training loop
    num_epochs = config.get('epochs', 10)
    best_val_loss = float('inf')
    
    print(f"\nStarting fine-tuning for {num_epochs} epochs...")
    
    total_steps = num_epochs * len(train_loader)
    global_step = 0
    start_time = datetime.now().isoformat()
    # Tracking
    tracking = config.get('tracking') or {}
    use_mlflow = bool((tracking.get('mlflow') or {}).get('enabled') and mlflow is not None)
    use_wandb = bool((tracking.get('wandb') or {}).get('enabled') and wandb is not None)
    run_name = tracking.get('name') or f"run_{job_id[:8]}"
    if use_mlflow:
        try:
            mlflow.set_experiment(tracking.get('experiment') or 'default')
            mlflow.start_run(run_name=run_name)
            def _flat(d, p=''):
                out={}
                for k,v in (d or {}).items():
                    kk=f"{p}.{k}" if p else k
                    if isinstance(v, dict): out.update(_flat(v, kk))
                    else: out[kk]=v
                return out
            mlflow.log_params({k:v for k,v in _flat(config).items() if isinstance(v,(int,float,str,bool))})
        except Exception:
            pass
    if use_wandb:
        try:
            wandb.init(project=(tracking.get('project') or 'trainer'), name=run_name, reinit=True, config=config)
        except Exception:
            pass

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}] Batch [{batch_idx}/{len(train_loader)}] "
                      f"Loss: {loss.item():.4f}")
                try:
                    metric = {
                        'kind': 'batch',
                        'epoch': epoch + 1,
                        'num_epochs': num_epochs,
                        'batch_idx': batch_idx,
                        'num_batches': len(train_loader),
                        'loss': float(loss.item()),
                        'accuracy': float(correct / total) if total else None,
                        'lr': float(optimizer.param_groups[0].get('lr', learning_rate)) if hasattr(optimizer, 'param_groups') else None,
                        'step': global_step,
                        'total_steps': total_steps,
                        'time': datetime.now().isoformat(),
                        'start_time': start_time,
                    }
                    print('METRIC:' + json.dumps(metric), flush=True)
                except Exception:
                    pass
                # Trackers
                try:
                    if use_mlflow:
                        mlflow.log_metric('loss', float(loss.item()), step=global_step)
                    if use_wandb:
                        wandb.log({'loss': float(loss.item())}, step=global_step)
                except Exception:
                    pass
            global_step += 1
        
        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = 100. * correct / total
        
        # Validation phase
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                
                val_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100. * correct / total
        
        print(f"\nEpoch {epoch+1}/{num_epochs} Complete:")
        print(f"  Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:.2f}%")
        print(f"  Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.2f}%")
        try:
            epoch_metric = {
                'kind': 'epoch',
                'epoch': epoch + 1,
                'num_epochs': num_epochs,
                'avg_train_loss': float(avg_train_loss),
                'val_loss': float(avg_val_loss),
                'train_acc_pct': float(train_accuracy),
                'val_acc_pct': float(val_accuracy),
                'step': global_step,
                'total_steps': total_steps,
                'time': datetime.now().isoformat(),
                'start_time': start_time,
            }
            print('METRIC:' + json.dumps(epoch_metric), flush=True)
        except Exception:
            pass
        try:
            if use_mlflow:
                mlflow.log_metrics({'val_loss': float(avg_val_loss), 'val_acc_pct': float(val_accuracy)}, step=global_step)
            if use_wandb:
                wandb.log({'val_loss': float(avg_val_loss), 'val_acc_pct': float(val_accuracy)}, step=global_step)
        except Exception:
            pass
        
        # Update learning rate
        if use_scheduler:
            scheduler.step(avg_val_loss)
            print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}\n")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_dir = os.path.join(BASE_DIR, 'models', job_id)
            os.makedirs(save_dir, exist_ok=True)
            
            best_model_path = os.path.join(save_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'val_accuracy': val_accuracy,
                'config': config
            }, best_model_path)
    
    # Save final model
    save_dir = os.path.join(BASE_DIR, 'models', job_id)
    os.makedirs(save_dir, exist_ok=True)
    
    model_path = os.path.join(save_dir, 'model.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config
    }, model_path)
    
    # Save config
    config_path = os.path.join(save_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump({
            'name': config.get('name', f'Finetuned Model {job_id[:8]}'),
            'framework': 'pytorch',
            'base_model': config.get('model_name', 'custom'),
            'created': datetime.now().isoformat(),
            'parameters': sum(p.numel() for p in model.parameters()),
            'best_val_loss': best_val_loss
        }, f, indent=2)
    
    print(f"\nModel saved to: {save_dir}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print("Fine-tuning completed successfully!")
    try:
        if use_mlflow:
            mlflow.end_run()
    except Exception:
        pass
    try:
        if use_wandb:
            wandb.finish()
    except Exception:
        pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--job-id', required=True)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    
    config = json.loads(args.config)
    finetune_model(config, args.job_id)
