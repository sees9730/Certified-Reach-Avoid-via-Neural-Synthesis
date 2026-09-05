"""Utility functions for training and setup."""

import shutil
from pathlib import Path

def cleanup_and_setup_directories(dirs: list) -> None:
    """Remove and recreate directories for fresh training run."""
    for dir_path in dirs:
        dir_path = Path(dir_path)
        if dir_path.exists():
            shutil.rmtree(dir_path)
            print(f"Cleaned up old {dir_path.name} directory")
        dir_path.mkdir(parents=True, exist_ok=True)
