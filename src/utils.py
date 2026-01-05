"""
Utility functions for the MDE project
"""

import torch
import numpy as np
import cv2
from pathlib import Path
import json


def normalize_image(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Normalize image with ImageNet statistics.
    
    Args:
        image (torch.Tensor): Input image (C, H, W) or (B, C, H, W)
        mean (list): Mean values for normalization
        std (list): Standard deviation values for normalization
        
    Returns:
        torch.Tensor: Normalized image
    """
    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)
    return (image - mean) / std


def denormalize_image(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Denormalize image from ImageNet statistics.
    
    Args:
        image (torch.Tensor): Normalized image
        mean (list): Mean values used for normalization
        std (list): Standard deviation values used for normalization
        
    Returns:
        torch.Tensor: Denormalized image
    """
    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)
    return image * std + mean


def compute_depth_metrics(pred, target, mask=None):
    """
    Compute common depth estimation metrics.
    
    Args:
        pred (torch.Tensor): Predicted depth
        target (torch.Tensor): Ground truth depth
        mask (torch.Tensor): Valid mask (optional)
        
    Returns:
        dict: Dictionary containing computed metrics
    """
    if mask is not None:
        pred = pred[mask]
        target = target[mask]
    
    # Remove zero values
    valid = (target > 0) & (pred > 0)
    pred = pred[valid]
    target = target[valid]
    
    # Metrics
    mae = torch.mean(torch.abs(pred - target))
    rmse = torch.sqrt(torch.mean((pred - target) ** 2))
    
    # Threshold accuracy
    delta = torch.max(pred / (target + 1e-8), target / (pred + 1e-8))
    delta1 = (delta < 1.25).float().mean()
    delta2 = (delta < 1.25 ** 2).float().mean()
    delta3 = (delta < 1.25 ** 3).float().mean()
    
    return {
        'mae': mae.item(),
        'rmse': rmse.item(),
        'delta1': delta1.item(),
        'delta2': delta2.item(),
        'delta3': delta3.item()
    }


def save_checkpoint(model, optimizer, epoch, path, metrics=None):
    """
    Save model checkpoint.
    
    Args:
        model (nn.Module): Model to save
        optimizer: Optimizer state
        epoch (int): Current epoch
        path (str): Path to save checkpoint
        metrics (dict): Optional metrics to save
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    
    if metrics:
        checkpoint['metrics'] = metrics
    
    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path}")


def load_checkpoint(model, optimizer, path, device='cuda'):
    """
    Load model checkpoint.
    
    Args:
        model (nn.Module): Model to load into
        optimizer: Optimizer to load state into
        path (str): Path to checkpoint
        device (str): Device to load to
        
    Returns:
        int: Epoch from checkpoint
    """
    checkpoint = torch.load(path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    print(f"Checkpoint loaded from {path} (epoch {epoch})")
    
    return epoch


def visualize_depth(depth, colormap=cv2.COLORMAP_TURBO):
    """
    Visualize depth map with colormap.
    
    Args:
        depth (np.ndarray or torch.Tensor): Depth map
        colormap: OpenCV colormap
        
    Returns:
        np.ndarray: Colored depth visualization
    """
    if isinstance(depth, torch.Tensor):
        depth = depth.cpu().numpy()
    
    # Normalize to 0-255
    depth_min = np.nanmin(depth)
    depth_max = np.nanmax(depth)
    depth_normalized = ((depth - depth_min) / (depth_max - depth_min + 1e-8) * 255).astype(np.uint8)
    
    # Apply colormap
    depth_colored = cv2.applyColorMap(depth_normalized, colormap)
    
    return depth_colored


def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = 'cpu'
        print("Using CPU")
    
    return device
