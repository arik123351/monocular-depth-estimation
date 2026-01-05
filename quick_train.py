#!/usr/bin/env python
"""
Quick training script with reduced dataset for faster iteration.
This uses only 5000 training and 200 validation samples for quick testing.
"""

import sys
from pathlib import Path

# Add src directory to path
src_dir = Path(__file__).parent / 'src'
sys.path.insert(0, str(src_dir))

import yaml
from train import main

if __name__ == '__main__':
    # Load base config
    config_path = Path(__file__).parent / 'configs' / 'config.yaml'
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override for quick testing (use much smaller dataset)
    config['training']['max_train_samples'] = 5000   # Use 5000 train samples instead of 50k
    config['training']['max_val_samples'] = 200      # Use 200 val samples instead of 650
    config['training']['epochs'] = 5                 # Train for just 5 epochs
    
    print("=" * 60)
    print("QUICK TRAINING MODE")
    print("=" * 60)
    print(f"Model: {config['model']['backbone']}")
    print(f"Epochs: {config['training']['epochs']}")
    print(f"Batch Size: {config['training']['batch_size']}")
    print(f"Train Samples: {config['training']['max_train_samples']}")
    print(f"Val Samples: {config['training']['max_val_samples']}")
    print("=" * 60)
    print()
    
    # Save modified config
    output_dir = Path(config.get('output_dir', './outputs'))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    temp_config_path = output_dir / 'config_quick.yaml'
    with open(temp_config_path, 'w') as f:
        yaml.dump(config, f)
    
    # Run training with modified config
    main(str(temp_config_path))
