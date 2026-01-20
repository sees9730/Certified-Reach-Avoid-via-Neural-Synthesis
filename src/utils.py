"""Utility functions for training and setup."""

import shutil
import torch.nn as nn
from pathlib import Path

# Set up directories
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]   # repo_root
import sys
sys.path.insert(0, str(ROOT))
from src.hyperparameters import Hyperparameters


def cleanup_and_setup_directories(dirs: list) -> None:
    """Remove and recreate directories for fresh training run."""
    for dir_path in dirs:
        dir_path = Path(dir_path)
        if dir_path.exists():
            shutil.rmtree(dir_path)
            print(f"Cleaned up old {dir_path.name} directory")
        dir_path.mkdir(exist_ok=True)

