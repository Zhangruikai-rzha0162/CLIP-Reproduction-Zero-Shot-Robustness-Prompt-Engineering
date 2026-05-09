"""
Main orchestration script.
Runs the full reproduction pipeline end-to-end:
1. ResNet baseline training (if missing)
2. Prompt engineering benchmark (Figure 4)
3. Robustness evaluation (CIFAR-10-C)
4. Fine-grained classification (Pets & Flowers102)
5. Visualization of all results
"""

import os
import json

import config
from utils import set_seed
from train_resnet import train_resnet
from eval_prompt import evaluate_prompt_engineering
from eval_robustness import RobustnessBenchmark
from eval_finegrained import FineGrainedPromptEngineering
from visualization import plot_figure4_reproduction, plot_robustness_curves, plot_finegrained_comparison
from models import ResNetClassifier


def ensure_resnet_checkpoints():
    """
    Train ResNet baselines if their checkpoints do not yet exist.
    """
    datasets = ["CIFAR10", "CIFAR100", "STL10"]
    for ds in datasets:
        path = f"{config.SAVE_DIR}/resnet_{ds.lower()}.pth"
        if not os.path.exists(path):
            print(f"\nCheckpoint missing for {ds}. Starting training...")
            train_resnet(ds, epochs=config.RESNET_EPOCHS, lr=config.RESNET_LR)
        else:
            print(f"Found checkpoint for {ds}: {path}")


def main():
    set_seed(config.SEED)
    os.makedirs(config.SAVE_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # Stage 1: Ensure supervised baselines exist
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STAGE 1: ResNet Baseline Checkpoints")
    print("=" * 70)
    ensure_resnet_checkpoints()

    # ------------------------------------------------------------------
    # Stage 2: Prompt Engineering (Figure 4 reproduction)
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STAGE 2: Prompt Engineering Benchmark")
    print("=" * 70)
    prompt_results = evaluate_prompt_engineering()
    plot_figure4_reproduction(prompt_results)

    with open(f"{config.SAVE_DIR}/prompt_results.json", "w") as f:
        json.dump(prompt_results, f, indent=2)

    # ------------------------------------------------------------------
    # Stage 3: Robustness on CIFAR-10-C
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STAGE 3: Robustness Evaluation")
    print("=" * 70)
    benchmark = RobustnessBenchmark()

    print("\n--- CLIP Zero-Shot ---")
    clip_rob = benchmark.evaluate_robustness("clip")

    print("\n--- ResNet Supervised ---")
    resnet_model = ResNetClassifier(num_classes=10).to(config.DEVICE)
    resnet_model.load_state_dict(
        torch.load(
            f"{config.SAVE_DIR}/resnet_cifar10.pth",
            map_location=config.DEVICE,
        )
    )
    resnet_model.eval()
    resnet_rob = benchmark.evaluate_robustness("resnet", resnet_model)

    plot_robustness_curves(clip_rob, resnet_rob)

    with open(f"{config.SAVE_DIR}/robustness_clip.json", "w") as f:
        json.dump(clip_rob, f, indent=2)
    with open(f"{config.SAVE_DIR}/robustness_resnet.json", "w") as f:
        json.dump(resnet_rob, f, indent=2)

    # ------------------------------------------------------------------
    # Stage 4: Fine-grained Classification
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STAGE 4: Fine-grained Prompt Engineering")
    print("=" * 70)
    fg_engine = FineGrainedPromptEngineering()

    print("\n--- Oxford-IIIT Pets ---")
    pets_res, _ = fg_engine.evaluate("Pets")

    print("\n--- Flowers102 ---")
    flowers_res, _ = fg_engine.evaluate("Flowers102")

    plot_finegrained_comparison(pets_res, flowers_res)

    with open(f"{config.SAVE_DIR}/finegrained_results.json", "w") as f:
        json.dump({"Pets": pets_res, "Flowers102": flowers_res}, f, indent=2)

    # ------------------------------------------------------------------
    # Final Report
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print(f"All artifacts saved to: {config.SAVE_DIR}")
    print("Generated files:")
    for fname in sorted(os.listdir(config.SAVE_DIR)):
        print(f"  - {fname}")


if __name__ == "__main__":
    import torch  # local import for checkpoint loading

    main()