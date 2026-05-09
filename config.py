"""
Global configuration for the CLIP reproduction project.
All hyperparameters and hardware settings are centralized here.
"""

import os
import torch

# Hardware configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
BATCH_SIZE = 128          # Tuned for 8 GB VRAM (RTX 5060 Laptop)
NUM_WORKERS = 4           # Set to 0 on Windows to avoid shared-memory errors

# Paths
SAVE_DIR = "./clip_reproduction_results"
os.makedirs(SAVE_DIR, exist_ok=True)

# Model names
CLIP_MODEL = "ViT-B/32"
RESNET_EPOCHS = 30
RESNET_LR = 0.001