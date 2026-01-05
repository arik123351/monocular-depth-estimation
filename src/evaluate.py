"""
Evaluation script for depth estimation model
"""

import torch
import argparse
from pathlib import Path
import json
import cv2
import numpy as np

from dataset import get_dataloader
from models import create_model
from utils import compute_depth_metrics, load_checkpoint, visualize_depth, get_device


def evaluate(model, dataloader, device, save_viz=False, viz_dir=None):
    """
    Evaluate model on a dataset.
    
    Args:
        model: Model to evaluate
        dataloader: Evaluation dataloader
        device: Device to evaluate on
        save_viz: Whether to save visualizations
        viz_dir: Directory to save visualizations
        
    Returns:
        dict: Computed metrics
    """
    model.eval()
    
    all_metrics = {
        'mae': [],
        'rmse': [],
        'delta1': [],
        'delta2': [],
        'delta3': []
    }
    
    if save_viz and viz_dir:
        Path(viz_dir).mkdir(parents=True, exist_ok=True)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            images = batch['image'].to(device)
            depths = batch['depth'].to(device)
            
            outputs = model(images)
            
            # Compute metrics
            metrics = compute_depth_metrics(outputs, depths)
            for key in all_metrics:
                all_metrics[key].append(metrics[key])
            
            # Save visualizations
            if save_viz and viz_dir:
                for i in range(outputs.shape[0]):
                    pred_depth = outputs[i, 0].cpu().numpy()
                    gt_depth = depths[i, 0].cpu().numpy()
                    
                    # Visualize
                    pred_viz = visualize_depth(pred_depth)
                    gt_viz = visualize_depth(gt_depth)
                    
                    # Save
                    pred_path = Path(viz_dir) / f'pred_{batch_idx}_{i}.png'
                    gt_path = Path(viz_dir) / f'gt_{batch_idx}_{i}.png'
                    
                    cv2.imwrite(str(pred_path), pred_viz)
                    cv2.imwrite(str(gt_path), gt_viz)
    
    # Average metrics
    final_metrics = {}
    for key, values in all_metrics.items():
        final_metrics[key] = np.mean(values)
    
    return final_metrics


def main(model_path, dataset_split='test', config_path=None, save_viz=False):
    """
    Main evaluation script.
    
    Args:
        model_path: Path to saved model
        dataset_split: Dataset split to evaluate on
        config_path: Path to configuration file
        save_viz: Whether to save visualizations
    """
    # Setup
    device = get_device()
    
    # Create model
    print("Creating model...")
    model = create_model(device=device)
    
    # Load checkpoint
    print(f"Loading model from {model_path}...")
    load_checkpoint(model, None, model_path, device=device)
    
    # Load dataset
    print(f"Loading {dataset_split} dataset...")
    dataloader = get_dataloader(
        split=dataset_split,
        batch_size=4,
        num_workers=0,
        shuffle=False
    )
    
    # Evaluate
    print("Evaluating...")
    viz_dir = Path(model_path).parent / 'visualizations' if save_viz else None
    metrics = evaluate(model, dataloader, device, save_viz=save_viz, viz_dir=viz_dir)
    
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"MAE (Mean Absolute Error): {metrics['mae']:.4f}")
    print(f"RMSE (Root Mean Square Error): {metrics['rmse']:.4f}")
    print(f"Delta 1 (δ < 1.25): {metrics['delta1']:.4f}")
    print(f"Delta 2 (δ < 1.25²): {metrics['delta2']:.4f}")
    print(f"Delta 3 (δ < 1.25³): {metrics['delta3']:.4f}")
    print("="*50)
    
    # Save results
    results_path = Path(model_path).parent / 'evaluation_results.json'
    with open(results_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Results saved to {results_path}")
    
    if save_viz:
        print(f"Visualizations saved to {viz_dir}")


if __name__ == "__main__":
    import sys
    import os
    
    # Add src directory to Python path
    src_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, src_dir)
    
    parser = argparse.ArgumentParser(description='Evaluate depth estimation model')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to saved model')
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'val', 'test'],
                        help='Dataset split to evaluate on')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to configuration file')
    parser.add_argument('--save-viz', action='store_true',
                        help='Save visualizations')
    
    args = parser.parse_args()
    
    main(args.model, args.split, args.config, args.save_viz)
