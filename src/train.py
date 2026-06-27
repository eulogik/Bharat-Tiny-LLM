#!/usr/bin/env python3
"""
Bharat-Tiny-LLM Training Script
Mac Mini M4 16GB Optimized
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def check_environment():
    """Verify environment is ready for training."""
    logger.info("Checking environment...")
    
    # Check MLX
    try:
        import mlx.core as mx
        logger.info(f"MLX available: {mx.__version__}")
        logger.info(f"MLX devices: {mx.default_device()}")
    except ImportError:
        logger.error("MLX not installed. Run: pip install mlx mlx-lm")
        sys.exit(1)
    
    # Check memory
    import psutil
    mem = psutil.virtual_memory()
    logger.info(f"Total memory: {mem.total / 1024**3:.1f} GB")
    logger.info(f"Available memory: {mem.available / 1024**3:.1f} GB")
    
    if mem.available / 1024**3 < 8:
        logger.warning("Low available memory. Consider closing other apps.")
    
    return True


def load_config():
    """Load training configuration."""
    config_path = Path("configs/training.yaml")
    
    if config_path.exists():
        import yaml
        with open(config_path) as f:
            return yaml.safe_load(f)
    
    # Default configuration
    return {
        "model": {
            "name": "HuggingFaceTB/SmolLM2-360M",
            "save_path": "./models/smolm2-360m-base"
        },
        "training": {
            "batch_size": 2,
            "gradient_accumulation": 8,
            "learning_rate": 1e-5,
            "epochs": 3,
            "context_length": 2048,
            "warmup_steps": 100
        },
        "data": {
            "train_file": "data/processed/train.jsonl",
            "val_file": "data/processed/val.jsonl"
        }
    }


def prepare_data(data_path: str, max_samples: int = None):
    """Prepare training data."""
    logger.info(f"Loading data from {data_path}")
    
    data = []
    with open(data_path) as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            data.append(json.loads(line))
    
    logger.info(f"Loaded {len(data)} samples")
    return data


def train(config: dict):
    """Main training loop."""
    from mlx_lm import lora, train as mlx_train
    
    logger.info("Starting training...")
    logger.info(f"Config: {json.dumps(config, indent=2)}")
    
    # Load data
    train_data = prepare_data(config["data"]["train_file"])
    
    # Train
    mlx_train(
        model=config["model"]["save_path"],
        train=True,
        data=config["data"]["train_file"],
        batch_size=config["training"]["batch_size"],
        lora_layers=8,
        epochs=config["training"]["epochs"],
        lr=config["training"]["learning_rate"]
    )
    
    logger.info("Training complete!")


def main():
    """Main entry point."""
    logger.info("=" * 50)
    logger.info("Bharat-Tiny-LLM Training")
    logger.info(f"Started at: {datetime.now()}")
    logger.info("=" * 50)
    
    # Check environment
    if not check_environment():
        logger.error("Environment check failed")
        sys.exit(1)
    
    # Load config
    config = load_config()
    
    # Start training
    train(config)
    
    logger.info("=" * 50)
    logger.info(f"Finished at: {datetime.now()}")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
