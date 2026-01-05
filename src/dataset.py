"""
NYU Depth v2 Dataset Module

This module handles loading the NYU Depth v2 dataset from Kaggle Hub
without downloading it locally.
"""

import os
import torch
import numpy as np
import cv2
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm

try:
    import kagglehub
except ImportError:
    raise ImportError("Please install kagglehub: pip install kagglehub")


class NYUDepthV2Dataset(Dataset):
    """
    NYU Depth v2 Dataset loader.
    
    This dataset streams data from Kaggle Hub without requiring local downloads.
    """
    
    def __init__(self, split='train', transform=None, download_path=None):
        """
        Initialize the dataset.
        
        Args:
            split (str): 'train' or 'test' split
            transform: Optional transforms to apply
            download_path (str): Path to cache the dataset
        """
        self.split = split
        self.transform = transform
        
        # Download/access dataset from Kaggle Hub
        try:
            if download_path is None:
                # Use kagglehub's default cache directory
                dataset_path = kagglehub.dataset_download("soumikrakshit/nyu-depth-v2")
            else:
                dataset_path = download_path
                if not os.path.exists(dataset_path):
                    print(f"Downloading dataset to {download_path}...")
                    dataset_path = kagglehub.dataset_download("soumikrakshit/nyu-depth-v2", 
                                                             path=download_path)
        except Exception as e:
            print(f"\nError accessing dataset from kagglehub:")
            print(f"  {type(e).__name__}: {e}")
            print(f"\nMake sure you have:")
            print(f"  1. Installed kagglehub: pip install kagglehub")
            print(f"  2. Set up Kaggle API credentials at ~/.kaggle/kaggle.json")
            print(f"  3. Accepted the dataset terms on Kaggle")
            raise
        
        self.dataset_path = Path(dataset_path)
        self.img_list = []
        self.depth_list = []
        
        # Build file lists based on available structure
        self._build_file_lists()
    
    def _build_file_lists(self):
        """Build lists of image and depth file paths from CSV metadata."""
        # NYU Depth v2 dataset structure (from kagglehub):
        # {dataset_path}/nyu_data/data/
        #   ├── nyu2_train/  (contains train images in subdirectories)
        #   ├── nyu2_test/   (contains test images)
        #   ├── nyu2_train.csv  (metadata: rgb,depth with "data/" prefix)
        #   └── nyu2_test.csv   (metadata: rgb,depth with "data/" prefix)
        
        # Handle versioned path from kagglehub
        if (self.dataset_path / 'versions').exists():
            # Path includes version directory
            nyu_data_path = self.dataset_path / 'versions' / '1' / 'nyu_data'
        elif (self.dataset_path / 'nyu_data').exists():
            # Direct path to nyu_data
            nyu_data_path = self.dataset_path / 'nyu_data'
        else:
            # Assume parent directory is nyu_data
            nyu_data_path = self.dataset_path
        
        # Determine CSV file based on split
        # 'val' is treated as an alias for 'test' (validation set)
        if self.split == 'train':
            csv_file = nyu_data_path / 'data' / 'nyu2_train.csv'
        elif self.split == 'val' or self.split == 'test':
            csv_file = nyu_data_path / 'data' / 'nyu2_test.csv'
        else:
            raise ValueError(f"Unknown split: {self.split}. Use 'train', 'val', or 'test'")
        
        # Load metadata CSV
        if not csv_file.exists():
            print(f"Error: CSV file not found at {csv_file}")
            raise FileNotFoundError(f"Dataset metadata not found at {csv_file}")
        
        try:
            # Read CSV without headers as it contains relative paths
            df_metadata = pd.read_csv(csv_file, header=None, names=["rgb", "depth"])
            
            # Build file lists
            # CSV paths are relative to nyu_data directory (e.g., "data/nyu2_train/...")
            # Skip existence check for speed - it's too slow when checking thousands of files
            # CSV files are trusted to contain valid paths
            for idx, row in df_metadata.iterrows():
                rgb_path = nyu_data_path / row['rgb']
                depth_path = nyu_data_path / row['depth']
                
                # Trust the CSV - skip .exists() check (too slow for 50k+ files)
                self.img_list.append(str(rgb_path))
                self.depth_list.append(str(depth_path))
            
            print(f"Found {len(self.img_list)} samples in {self.split} split")
            
            if len(self.img_list) == 0:
                raise RuntimeError(f"No samples found in dataset metadata.")
            
        except Exception as e:
            print(f"Error reading dataset: {e}")
            raise
    
    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Args:
            idx (int): Index of the sample
            
        Returns:
            tuple: (rgb_image, depth_map) as torch tensors
        """
        # Load RGB image
        rgb_image = cv2.imread(self.img_list[idx])
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        
        # Load depth map
        depth_map = cv2.imread(self.depth_list[idx], cv2.IMREAD_ANYDEPTH)
        
        # Normalize and convert to float
        if depth_map.dtype == np.uint16:
            depth_map = depth_map.astype(np.float32) / 1000.0  # Convert to meters
        else:
            depth_map = depth_map.astype(np.float32)
        
        # Convert to tensors
        rgb_image = torch.from_numpy(rgb_image).permute(2, 0, 1).float() / 255.0
        depth_map = torch.from_numpy(depth_map).unsqueeze(0).float()
        
        if self.transform:
            rgb_image = self.transform(rgb_image)
        
        return {
            'image': rgb_image,
            'depth': depth_map,
            'path': self.img_list[idx]
        }


def get_dataloader(split='train', batch_size=4, num_workers=0, 
                   shuffle=True, download_path=None, max_samples=None):
    """
    Create a DataLoader for the NYU Depth v2 dataset.
    
    Args:
        split (str): 'train' or 'test' split
        batch_size (int): Batch size
        num_workers (int): Number of workers for data loading
        shuffle (bool): Whether to shuffle the data
        download_path (str): Path to cache the dataset
        max_samples (int): Limit dataset to N samples (None for unlimited)
        
    Returns:
        DataLoader: PyTorch DataLoader
    """
    dataset = NYUDepthV2Dataset(split=split, download_path=download_path)
    
    # Limit dataset size if specified
    if max_samples is not None and max_samples > 0:
        dataset.img_list = dataset.img_list[:max_samples]
        dataset.depth_list = dataset.depth_list[:max_samples]
        print(f"Limited {split} dataset to {len(dataset.img_list)} samples")
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=True
    )
    
    return dataloader


if __name__ == "__main__":
    # Test dataset loading
    print("Testing NYU Depth v2 Dataset...")
    dataset = NYUDepthV2Dataset(split='train')
    
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"Sample image shape: {sample['image'].shape}")
        print(f"Sample depth shape: {sample['depth'].shape}")
        print("Dataset test passed!")
    else:
        print("No samples found. Check dataset structure.")
