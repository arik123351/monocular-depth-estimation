"""
Utility script to inspect the NYU Depth v2 dataset structure
"""

import os
from pathlib import Path

try:
    import kagglehub
    dataset_path = kagglehub.dataset_download("soumikrakshit/nyu-depth-v2")
    print(f"Dataset path: {dataset_path}\n")
    
    # List directory structure
    print("Directory structure:")
    print("=" * 60)
    
    for root, dirs, files in os.walk(dataset_path):
        level = root.replace(dataset_path, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f'{indent}{os.path.basename(root)}/')
        subindent = ' ' * 2 * (level + 1)
        
        # Limit files shown
        files_to_show = files[:10]
        for file in files_to_show:
            print(f'{subindent}{file}')
        if len(files) > 10:
            print(f'{subindent}... and {len(files) - 10} more files')
        
        # Limit depth
        if level > 3:
            dirs[:] = []
    
    print("\n" + "=" * 60)
    
    # Count files by extension
    print("\nFile types in dataset:")
    extensions = {}
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            ext = os.path.splitext(file)[1]
            extensions[ext] = extensions.get(ext, 0) + 1
    
    for ext, count in sorted(extensions.items(), key=lambda x: x[1], reverse=True):
        print(f"  {ext if ext else '(no extension)'}: {count} files")
    
except Exception as e:
    print(f"Error: {e}")
    print("Make sure you have kagglehub installed and Kaggle credentials set up")
