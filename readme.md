# CLIP Reproduction Experiments

Reproduction of core CLIP paper experiments on consumer hardware (RTX 5060 Laptop, 8 GB VRAM).

## Experiments

1. **Prompt Engineering** (Figure 4) — CIFAR-10 / CIFAR-100 / STL-10
2. **Robustness** — CIFAR-10-C synthetic corruptions (Gaussian, Blur, Pixelate, Contrast)
3. **Fine-grained Classification** — Oxford-IIIT Pets & Flowers-102 with domain-specific prompts

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the full pipeline
python main.py

# Or run modules independently
python train_resnet.py --dataset CIFAR10 --epochs 30
python eval_prompt.py
python eval_robustness.py
python eval_finegrained.py