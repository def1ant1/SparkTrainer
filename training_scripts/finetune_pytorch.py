import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import argparse
import json
import os
from datetime import datetime
import torchvision.models as models

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
        
        # Update learning rate
        if use_scheduler:
            scheduler.step(avg_val_loss)
            print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}\n")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_dir = f'/home/claude/dgx-ai-trainer/models/{job_id}'
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
    save_dir = f'/home/claude/dgx-ai-trainer/models/{job_id}'
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--job-id', required=True)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    
    config = json.loads(args.config)
    finetune_model(config, args.job_id)
