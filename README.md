# Monocular Depth Estimation (MDE) Project

A Deep Learning project for monocular depth estimation using the NYU Depth v2 dataset.

## Features

- Uses NYU Depth v2 dataset via Kaggle Hub (no local download required)
- PyTorch-based implementation
- Streamable dataset loading
- Training and evaluation pipeline
- TensorBoard logging support

## Project Structure

```
mde_project/
├── src/
│   ├── __init__.py
│   ├── dataset.py          # NYU Depth v2 dataset loader
│   ├── models.py           # Depth estimation models
│   ├── train.py            # Training script
│   ├── evaluate.py         # Evaluation script
│   └── utils.py            # Utility functions
├── notebooks/
│   └── exploration.ipynb   # Data exploration and visualization
├── configs/
│   └── config.yaml         # Configuration file
├── requirements.txt
└── README.md
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset

The project uses the **NYU Depth v2 dataset** available on Kaggle:
- Dataset: https://www.kaggle.com/datasets/soumikrakshit/nyu-depth-v2

The dataset is accessed through kagglehub without downloading locally.

## Usage

### Training
```bash
python src/train.py --config configs/config.yaml
```

### Evaluation
```bash
python src/evaluate.py --model path/to/model.pth
```

### Exploration
Open and run `notebooks/exploration.ipynb` for data visualization.

## Requirements

- Python 3.8+
- PyTorch 2.1+
- CUDA-enabled GPU (recommended)

## Notes

- The dataset is streamed from Kaggle Hub, no local storage required
- Training uses PyTorch DataLoader for efficient batch processing
- Results are logged to TensorBoard

## License

This project is for educational purposes.
