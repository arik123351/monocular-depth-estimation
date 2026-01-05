# Training Speed Optimization Summary

## Changes Made

### 1. **Model Architecture**
- **Before**: ResNet50 (50 layers, ~25.5M parameters)
- **After**: ResNet18 (18 layers, ~11.7M parameters)
- **Speedup**: ~2-3x faster per batch

### 2. **Training Duration**
- **Before**: 100 epochs
- **After**: 20 epochs (default) or 5 epochs (quick test)
- **Speedup**: 5x fewer iterations

### 3. **Batch Size**
- **Before**: 4
- **After**: 8
- **Benefit**: Higher throughput, better GPU/CPU utilization

### 4. **Validation Frequency**
- **Before**: Every epoch
- **After**: Every epoch (but on 5x fewer epochs)
- **Speedup**: 5x fewer validation runs

## Estimated Training Times

### Full Training (20 epochs on ResNet18)
- Batch size: 8
- Training samples: 50,688 (full dataset)
- Estimated time: **4-6 hours on CPU** (was 40-60 hours with ResNet50)

### Quick Prototyping (5 epochs on ResNet18)
- Batch size: 8  
- Training samples: 5,000 (limited subset)
- Estimated time: **15-30 minutes on CPU**
- Command: `python quick_train.py`

### Custom Configuration
Edit `configs/config.yaml` to adjust:
```yaml
training:
  epochs: 20                    # Change number of epochs
  batch_size: 8                 # Increase for faster training (if GPU memory allows)
  max_train_samples: null       # Limit to N samples for testing (null = full dataset)
  max_val_samples: null         # Limit validation samples
```

## Usage

### Full Training (20 epochs, all data)
```bash
cd c:\Users\ariel\Projects\mde_project
python src\train.py --config configs\config.yaml
```

### Quick Testing (5 epochs, 5000 samples)
```bash
cd c:\Users\ariel\Projects\mde_project
python quick_train.py
```

### Custom Parameters
Edit `configs/config.yaml` before running:
```yaml
training:
  epochs: 10                    # Reduce epochs
  max_train_samples: 1000       # Use only 1000 samples
```

Then run:
```bash
python src\train.py --config configs\config.yaml
```

## Performance Metrics

After training completes, check metrics in `outputs/` directory:
- `best_model.pth` - Best checkpoint based on validation loss
- `runs/` - TensorBoard logs (if enabled)

To evaluate on test set:
```bash
python src\evaluate.py --model outputs/best_model.pth --save-viz
```

## Tips for Faster Development

1. **Use `quick_train.py`** for rapid prototyping (5 epochs, 5000 samples)
2. **Adjust `max_train_samples`** if you want intermediate speeds
3. **Monitor first epoch** - if loss doesn't decrease, stop and check model/data
4. **Use GPU** if available (50x+ speedup) - set `num_workers: 4` in config

## Expected Results

With current optimizations:
- ResNet18 backbone: Lighter, trains faster
- 20 epochs: Good balance between training time and model convergence
- Batch size 8: Efficient utilization of resources
- Overall speedup: **8-10x faster than original configuration**
