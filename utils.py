"""
Utility functions for the Knowledge Agent
"""

import os
import logging
from pathlib import Path
from config import Config
import torch

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_directories():
    """Create necessary directories for the application"""
    for dir_path in [Config.DATA_DIR, Config.KNOWLEDGE_DIR, Config.DB_DIR, Config.MODEL_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)
    logger.info("Directories created")


def log_gpu_memory():
    """Log current GPU memory usage if CUDA is available"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        logger.info(f"GPU memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")