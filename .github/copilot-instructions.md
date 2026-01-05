# MDE Project - Copilot Instructions

## Project Overview
Monocular Depth Estimation (MDE) Deep Learning project using NYU Depth v2 dataset from Kaggle Hub.

## Project Setup Status

### Completed Setup
✓ Project structure created
✓ Core modules implemented:
  - src/dataset.py - NYU Depth v2 dataset loader via kagglehub
  - src/models.py - ResNet50 encoder-decoder architecture
  - src/train.py - Training pipeline
  - src/evaluate.py - Evaluation script
  - src/utils.py - Utility functions
✓ Configuration file created (configs/config.yaml)
✓ Jupyter notebook for exploration created
✓ Requirements file with all dependencies

### Dataset Access
- Uses kagglehub to stream NYU Depth v2 dataset (no local download)
- Requires Kaggle API credentials (~/.kaggle/kaggle.json)
- Default cache location: ~/.cache/kagglehub/

## Usage

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set Up Kaggle Credentials
1. Get your Kaggle API key from https://www.kaggle.com/account
2. Create ~/.kaggle/kaggle.json with your credentials
3. Set permissions: chmod 600 ~/.kaggle/kaggle.json (Linux/Mac)

### 3. Training
```bash
python src/train.py --config configs/config.yaml
```

### 4. Evaluation
```bash
python src/evaluate.py --model outputs/best_model.pth --save-viz
```

### 5. Exploration
Open and run `notebooks/exploration.ipynb` for data visualization and understanding.

## Key Features

- **Dataset Streaming**: Uses kagglehub for on-demand data access
- **ResNet50 Backbone**: Pretrained encoder for feature extraction
- **Skip Connections**: Decoder uses skip connections from encoder
- **Evaluation Metrics**: MAE, RMSE, Delta accuracy (δ < 1.25^n)
- **TensorBoard Logging**: Real-time training visualization
- **Modular Design**: Easy to extend with new architectures

## Configuration

Edit `configs/config.yaml` to customize:
- Model architecture (backbone, pretrained weights)
- Training parameters (batch size, learning rate, epochs)
- Dataset path (leave null for kagglehub default)
- Output directory for models and logs

## Project Structure
```
mde_project/
├── src/
│   ├── __init__.py
│   ├── dataset.py          # NYU Depth v2 loader
│   ├── models.py           # Network architectures
│   ├── train.py            # Training script
│   ├── evaluate.py         # Evaluation script
│   └── utils.py            # Helper functions
├── notebooks/
│   └── exploration.ipynb   # Data exploration
├── configs/
│   └── config.yaml         # Configuration
├── requirements.txt        # Dependencies
└── README.md              # Documentation
```

## Next Steps

1. Install dependencies
2. Set up Kaggle credentials
3. Run exploration notebook to understand data
4. Train model with training script
5. Evaluate and visualize results
