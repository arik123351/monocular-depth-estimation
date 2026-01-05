"""
Training script for depth estimation model
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import argparse
import yaml
import time
from tqdm import tqdm

from dataset import get_dataloader
from models import create_model
from utils import compute_depth_metrics, save_checkpoint, get_device


def train_epoch(model, dataloader, criterion, optimizer, device, writer, epoch):
    """
    Train for one epoch.
    
    Args:
        model: Model to train
        dataloader: Training dataloader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        writer: TensorBoard writer
        epoch: Current epoch
        
    Returns:
        float: Average loss for the epoch
    """
    model.train()
    total_loss = 0
    metrics = {'mae': 0, 'rmse': 0}
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(progress_bar):
        images = batch['image'].to(device)
        depths = batch['depth'].to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, depths)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Compute metrics
        with torch.no_grad():
            batch_metrics = compute_depth_metrics(outputs, depths)
        
        total_loss += loss.item()
        for key in batch_metrics:
            if key in metrics:
                metrics[key] += batch_metrics[key]
        
        # Update progress bar
        progress_bar.set_postfix({
            'Loss': loss.item(),
            'MAE': batch_metrics['mae'],
            'RMSE': batch_metrics['rmse']
        })
        
        # Log to TensorBoard
        if writer is not None:
            global_step = epoch * len(dataloader) + batch_idx
            writer.add_scalar('train/loss', loss.item(), global_step)
    
    # Average metrics
    avg_loss = total_loss / len(dataloader)
    for key in metrics:
        metrics[key] /= len(dataloader)
    
    # Log to TensorBoard
    if writer is not None:
        writer.add_scalar('train/epoch_loss', avg_loss, epoch)
        for key, value in metrics.items():
            writer.add_scalar(f'train/{key}', value, epoch)
    
    return avg_loss


def validate(model, dataloader, criterion, device, writer, epoch):
    """
    Validate the model.
    
    Args:
        model: Model to validate
        dataloader: Validation dataloader
        criterion: Loss function
        device: Device to validate on
        writer: TensorBoard writer
        epoch: Current epoch
        
    Returns:
        float: Average loss and metrics dict
    """
    model.eval()
    total_loss = 0
    metrics = {'mae': 0, 'rmse': 0, 'delta1': 0, 'delta2': 0, 'delta3': 0}
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            images = batch['image'].to(device)
            depths = batch['depth'].to(device)
            
            outputs = model(images)
            loss = criterion(outputs, depths)
            
            batch_metrics = compute_depth_metrics(outputs, depths)
            
            total_loss += loss.item()
            for key in batch_metrics:
                if key in metrics:
                    metrics[key] += batch_metrics[key]
    
    # Average metrics
    avg_loss = total_loss / len(dataloader)
    for key in metrics:
        metrics[key] /= len(dataloader)
    
    # Log to TensorBoard
    if writer is not None:
        writer.add_scalar('val/loss', avg_loss, epoch)
        for key, value in metrics.items():
            writer.add_scalar(f'val/{key}', value, epoch)
    
    return avg_loss, metrics


def main(config_path):
    """
    Main training loop.
    
    Args:
        config_path: Path to configuration file
    """
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup
    device = get_device()
    output_dir = Path(config.get('output_dir', './outputs'))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create model
    print("Creating model...")
    model = create_model(
        model_type=config['model'].get('type', 'depth'),
        backbone=config['model'].get('backbone', 'resnet50'),
        pretrained=config['model'].get('pretrained', True),
        device=device
    )
    
    # Loss function and optimizer
    criterion = nn.L1Loss()  # MAE loss
    optimizer = optim.Adam(model.parameters(), lr=config['training']['lr'])
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config['training'].get('step_size', 10),
        gamma=config['training'].get('gamma', 0.1)
    )
    
    # Data loaders
    print("Loading datasets...")
    max_train_samples = config.get('training', {}).get('max_train_samples', None)
    max_val_samples = config.get('training', {}).get('max_val_samples', None)
    
    if max_train_samples is not None:
        print(f"Limiting training to {max_train_samples} samples")
    if max_val_samples is not None:
        print(f"Limiting validation to {max_val_samples} samples")
    
    train_loader = get_dataloader(
        split='train',
        batch_size=config['training']['batch_size'],
        num_workers=config['training'].get('num_workers', 0),
        shuffle=True,
        download_path=config.get('dataset_path'),
        max_samples=max_train_samples
    )
    
    val_loader = get_dataloader(
        split='val',
        batch_size=config['training']['batch_size'],
        num_workers=config['training'].get('num_workers', 0),
        shuffle=False,
        download_path=config.get('dataset_path'),
        max_samples=max_val_samples
    )
    
    # TensorBoard writer (optional)
    use_tensorboard = config.get('use_tensorboard', True)
    if use_tensorboard:
        try:
            writer = SummaryWriter(output_dir / 'runs')
            print("TensorBoard enabled")
        except Exception as e:
            print(f"Warning: Could not initialize TensorBoard: {e}")
            writer = None
    else:
        writer = None
        print("TensorBoard disabled")
    
    # Training loop
    best_loss = float('inf')
    start_epoch = 0
    num_epochs = config['training']['epochs']
    
    print(f"Starting training for {num_epochs} epochs...")
    
    for epoch in range(start_epoch, num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, writer, epoch)
        
        # Validate
        val_loss, val_metrics = validate(model, val_loader, criterion, device, writer, epoch)
        
        # Step scheduler
        scheduler.step()
        
        # Print metrics
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Val MAE: {val_metrics['mae']:.4f}")
        print(f"Val RMSE: {val_metrics['rmse']:.4f}")
        
        # Save checkpoint
        if val_loss < best_loss:
            best_loss = val_loss
            save_checkpoint(
                model, optimizer, epoch,
                output_dir / 'best_model.pth',
                metrics=val_metrics
            )
        
        # Save periodic checkpoint
        if (epoch + 1) % config['training'].get('save_interval', 5) == 0:
            save_checkpoint(
                model, optimizer, epoch,
                output_dir / f'checkpoint_epoch_{epoch}.pth'
            )
    
    if writer is not None:
        writer.close()
    
    # Save final model
    save_checkpoint(
        model, optimizer, num_epochs - 1,
        output_dir / 'final_model.pth',
        metrics={'best_validation_loss': best_loss}
    )
    
    print(f"\nTraining completed!")
    print(f"  Best model: {output_dir / 'best_model.pth'}")
    print(f"  Final model: {output_dir / 'final_model.pth'}")
    print(f"  Best validation loss: {best_loss:.4f}")


if __name__ == "__main__":
    import sys
    import os
    
    # Add src directory to Python path
    src_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, src_dir)
    
    parser = argparse.ArgumentParser(description='Train depth estimation model')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to configuration file')
    args = parser.parse_args()
    
    main(args.config)
