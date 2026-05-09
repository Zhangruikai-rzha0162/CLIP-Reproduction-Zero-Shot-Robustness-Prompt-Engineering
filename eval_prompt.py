"""
Reproduce Figure 4 from the CLIP paper:
Compare ResNet (supervised) vs CLIP Contextless vs CLIP Prompted
across CIFAR-10, CIFAR-100, and STL-10.
"""

import os
import platform
import numpy as np
import torch
from torch.utils.data import DataLoader

import config
from utils import set_seed, compute_accuracy_with_ci, wilson_interval
from datasets import load_dataset, DATASET_CLASSNAMES
from models import ResNetClassifier, CLIPZeroShotClassifier


# Windows compatibility: disable multiprocessing to avoid shared-memory errors
IS_WINDOWS = platform.system() == "Windows"
NUM_WORKERS_FIX = 0 if IS_WINDOWS else config.NUM_WORKERS


def load_or_train_resnet(dataset_name: str, epochs: int = 30):
    """
    Load a pre-trained ResNet if available; otherwise train from scratch.
    """
    save_path = f"{config.SAVE_DIR}/resnet_{dataset_name.lower()}.pth"
    num_classes_map = {"CIFAR10": 10, "CIFAR100": 100, "STL10": 10}
    num_classes = num_classes_map[dataset_name]

    model = ResNetClassifier(num_classes=num_classes).to(config.DEVICE)

    if os.path.exists(save_path):
        print(f"[System] Loading checkpoint: {save_path}")
        model.load_state_dict(
            torch.load(save_path, map_location=config.DEVICE)
        )
        model.eval()
        return model
    else:
        print(f"[System] Checkpoint not found. Training ResNet ({epochs} epochs)...")
        from train_resnet import train_resnet

        model, _ = train_resnet(dataset_name, epochs=epochs)
        return model


def evaluate_prompt_engineering():
    """
    Run the full prompt-engineering benchmark and return structured results.
    """
    set_seed(config.SEED)
    clip_classifier = CLIPZeroShotClassifier(config.CLIP_MODEL)

    datasets_to_test = ["CIFAR10", "CIFAR100", "STL10"]
    results = {}

    for dataset_name in datasets_to_test:
        print(f"\n{'=' * 70}")
        print(f"Evaluating dataset: {dataset_name}")
        print(f"{'=' * 70}")
        results[dataset_name] = {}

        # Load test split
        test_dataset = load_dataset(dataset_name, train=False, model_type="clip")
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS_FIX,
            pin_memory=False if IS_WINDOWS else True,
        )
        true_labels = np.array([label for _, label in test_dataset])

        # 1) ResNet supervised baseline
        print(f"\n[1/3] ResNet Supervised ({dataset_name})")
        resnet_model = load_or_train_resnet(dataset_name)
        resnet_model.eval()

        resnet_preds, resnet_top5_preds = [], []
        with torch.no_grad():
            for images, _ in test_loader:
                images = images.to(config.DEVICE)
                outputs = resnet_model(images)
                _, predicted = outputs.max(1)
                resnet_preds.extend(predicted.cpu().numpy())

                _, top5_idx = outputs.topk(5, dim=1)
                resnet_top5_preds.append(top5_idx.cpu().numpy())

        resnet_preds = np.array(resnet_preds)
        resnet_top5 = np.vstack(resnet_top5_preds)

        acc, low, up, _, _ = compute_accuracy_with_ci(resnet_preds, true_labels)
        top5_correct = sum(
            true_labels[i] in resnet_top5[i] for i in range(len(true_labels))
        )
        top5_acc = top5_correct / len(true_labels)
        top5_low, top5_up = wilson_interval(top5_acc, len(true_labels))

        results[dataset_name]["ResNet (Supervised)"] = {
            "top1": acc,
            "top1_lower": low,
            "top1_upper": up,
            "top5": top5_acc,
            "top5_lower": top5_low,
            "top5_upper": top5_up,
        }
        print(f"Top-1: {acc * 100:.2f}% [95% CI: {low * 100:.2f}, {up * 100:.2f}]")
        print(f"Top-5: {top5_acc * 100:.2f}% [95% CI: {top5_low * 100:.2f}, {top5_up * 100:.2f}]")

        # 2) CLIP with two prompt strategies
        classnames = DATASET_CLASSNAMES[dataset_name]
        prompt_strategies = {
            "Contextless": ["{label}"],
            "Prompted": ["A photo of a {label}."],
        }

        for strategy_name, templates in prompt_strategies.items():
            print(f"\n[{'2' if strategy_name == 'Contextless' else '3'}/3] CLIP + {strategy_name}")
            clip_classifier.set_classes(classnames, templates)
            top1_preds, top5_preds = clip_classifier.predict(test_loader, top_k=5)

            acc, low, up, _, _ = compute_accuracy_with_ci(top1_preds, true_labels)
            top5_correct = sum(
                true_labels[i] in top5_preds[i] for i in range(len(true_labels))
            )
            top5_acc = top5_correct / len(true_labels)
            top5_low, top5_up = wilson_interval(top5_acc, len(true_labels))

            delta = acc - results[dataset_name]["ResNet (Supervised)"]["top1"]

            results[dataset_name][f"CLIP + {strategy_name}"] = {
                "top1": acc,
                "top1_lower": low,
                "top1_upper": up,
                "top5": top5_acc,
                "top5_lower": top5_low,
                "top5_upper": top5_up,
                "vs_resnet": delta,
            }
            print(f"Top-1: {acc * 100:.2f}% [95% CI: {low * 100:.2f}, {up * 100:.2f}]")
            print(f"Top-5: {top5_acc * 100:.2f}% [95% CI: {top5_low * 100:.2f}, {top5_up * 100:.2f}]")
            print(f"Delta vs ResNet: {delta * 100:+.2f}%")

    return results


if __name__ == "__main__":
    prompt_results = evaluate_prompt_engineering()
    # Optionally save raw results for plotting
    import json

    with open(f"{config.SAVE_DIR}/prompt_results.json", "w") as f:
        json.dump(
            {k: {m: {n: float(v) if isinstance(v, (np.floating, float)) else v for n, v in d.items()}
                 for m, d in v.items()}
             for k, v in prompt_results.items()},
            f,
            indent=2,
        )
    print(f"Results saved to {config.SAVE_DIR}/prompt_results.json")