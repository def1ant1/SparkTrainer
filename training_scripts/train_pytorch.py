import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import argparse
import json
import os
from datetime import datetime

# Resolve project base directory (two levels up from this script)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class CustomModel(nn.Module):
    """Flexible neural network architecture"""
    def __init__(self, config):
        super(CustomModel, self).__init__()
        
        layers = []
        input_size = config.get('input_size', 784)
        hidden_sizes = config.get('hidden_layers', [512, 256, 128])
        output_size = config.get('output_size', 10)
        activation = config.get('activation', 'relu')
        dropout = config.get('dropout', 0.2)
        
        # Build layers
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'leaky_relu':
                layers.append(nn.LeakyReLU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, output_size))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

class ResNetBlock(nn.Module):
    """ResNet-style residual block"""
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out

class CustomResNet(nn.Module):
    """Custom ResNet architecture"""
    def __init__(self, config):
        super(CustomResNet, self).__init__()
        
        num_classes = config.get('num_classes', 10)
        num_blocks = config.get('num_blocks', [2, 2, 2, 2])
        channels = config.get('channels', [64, 128, 256, 512])
        
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.layer1 = self._make_layer(64, channels[0], num_blocks[0], 1)
        self.layer2 = self._make_layer(channels[0], channels[1], num_blocks[1], 2)
        self.layer3 = self._make_layer(channels[1], channels[2], num_blocks[2], 2)
        self.layer4 = self._make_layer(channels[2], channels[3], num_blocks[3], 2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(channels[3], num_classes)
    
    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = [ResNetBlock(in_channels, out_channels, stride)]
        for _ in range(1, num_blocks):
            layers.append(ResNetBlock(out_channels, out_channels, 1))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

def train_model(config, job_id):
    """Main training function"""
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Optional experiment tracking (best-effort)
    tracking = config.get('tracking', {}) if isinstance(config.get('tracking'), dict) else {}
    wandb_run = None
    tb_writer = None
    try:
        if tracking.get('wandb') or os.environ.get('WANDB_API_KEY'):
            import wandb  # type: ignore
            wandb_run = wandb.init(project=tracking.get('project', 'dgx-ai-trainer'), config=config, reinit=True)
    except Exception:
        wandb_run = None
    try:
        if tracking.get('tensorboard'):
            from torch.utils.tensorboard import SummaryWriter  # type: ignore
            tb_writer = SummaryWriter(log_dir=os.path.join(BASE_DIR, 'logs', job_id, 'tb'))
    except Exception:
        tb_writer = None
    
    # Create model
    architecture = config.get('architecture', 'custom')
    if architecture == 'resnet':
        model = CustomResNet(config).to(device)
    else:
        model = CustomModel(config).to(device)
    
    print(f"Model architecture: {architecture}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Setup optimizer
    optimizer_name = config.get('optimizer', 'adam')
    learning_rate = config.get('learning_rate', 0.001)
    
    if optimizer_name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Setup loss
    loss_fn_name = config.get('loss', 'cross_entropy')
    if loss_fn_name == 'cross_entropy':
        criterion = nn.CrossEntropyLoss()
    elif loss_fn_name == 'mse':
        criterion = nn.MSELoss()
    else:
        criterion = nn.CrossEntropyLoss()
    
    # Create dummy data for demonstration (replace with actual data loading)
    batch_size = config.get('batch_size', 32)
    num_samples = config.get('num_samples', 1000)
    input_size = config.get('input_size', 784)
    output_size = config.get('output_size', 10)
    
    # Generate synthetic data (in production, load real data)
    X = torch.randn(num_samples, input_size)
    y = torch.randint(0, output_size, (num_samples,))
    dataset = TensorDataset(X, y)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Training loop
    num_epochs = config.get('epochs', 10)
    
    print(f"\nStarting training for {num_epochs} epochs...")
    
    total_steps = num_epochs * len(train_loader)
    global_step = 0
    start_time = datetime.now().isoformat()
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            global_step += 1
            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}] Batch [{batch_idx}/{len(train_loader)}] "
                      f"Loss: {loss.item():.4f}")
                # Emit JSON metric line for realtime monitoring
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
                # Experiment trackers
                try:
                    if wandb_run is not None:
                        wandb_run.log({'loss': float(loss.item()), 'accuracy': float(correct/total) if total else None, 'lr': float(optimizer.param_groups[0].get('lr', learning_rate)) if hasattr(optimizer, 'param_groups') else None, 'epoch': epoch+1, 'step': global_step})
                except Exception:
                    pass
                try:
                    if tb_writer is not None:
                        tb_writer.add_scalar('loss', float(loss.item()), global_step)
                        if total:
                            tb_writer.add_scalar('accuracy', float(correct/total), global_step)
                        try:
                            tb_writer.add_scalar('lr', float(optimizer.param_groups[0].get('lr', learning_rate)), global_step)
                        except Exception:
                            pass
                except Exception:
                    pass
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        
        print(f"\nEpoch {epoch+1}/{num_epochs} Complete:")
        print(f"  Average Loss: {avg_loss:.4f}")
        print(f"  Training Accuracy: {accuracy:.2f}%\n")
        try:
            epoch_metric = {
                'kind': 'epoch',
                'epoch': epoch + 1,
                'num_epochs': num_epochs,
                'avg_loss': float(avg_loss),
                'accuracy_pct': float(accuracy),
                'step': global_step,
                'total_steps': total_steps,
                'time': datetime.now().isoformat(),
                'start_time': start_time,
            }
            print('METRIC:' + json.dumps(epoch_metric), flush=True)
        except Exception:
            pass
        # Trackers epoch-level
        try:
            if wandb_run is not None:
                wandb_run.log({'epoch_avg_loss': float(avg_loss), 'epoch_accuracy_pct': float(accuracy), 'epoch': epoch+1, 'step': global_step})
        except Exception:
            pass
        try:
            if tb_writer is not None:
                tb_writer.add_scalar('epoch_avg_loss', float(avg_loss), epoch+1)
                tb_writer.add_scalar('epoch_accuracy_pct', float(accuracy), epoch+1)
        except Exception:
            pass
    
    # Save model
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
            'name': config.get('name', f'Model {job_id[:8]}'),
            'framework': 'pytorch',
            'architecture': architecture,
            'created': datetime.now().isoformat(),
            'parameters': sum(p.numel() for p in model.parameters())
        }, f, indent=2)
    
    print(f"\nModel saved to: {save_dir}")
    print("Training completed successfully!")
    # Finish trackers
    try:
        if tb_writer is not None:
            tb_writer.flush(); tb_writer.close()
    except Exception:
        pass
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
    train_model(config, args.job_id)
