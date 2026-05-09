"""
Fine-grained classification with domain-specific prompt engineering.
Evaluates Oxford-IIIT Pets and Flowers102 using three strategies:
1. Single Generic template
2. Generic Ensemble (multi-template averaging)
3. Domain-Specific templates
"""

import numpy as np
from torch.utils.data import DataLoader

import config
from utils import set_seed, compute_accuracy_with_ci
from datasets import load_dataset
from models import CLIPZeroShotClassifier


class FineGrainedPromptEngineering:
    """
    Benchmark fine-grained tasks with tailored prompt strategies.
    """

    def __init__(self):
        self.clip_classifier = CLIPZeroShotClassifier(config.CLIP_MODEL)

    def get_prompt_strategies(self, dataset_name: str):
        """
        Return three prompt strategies for a given fine-grained dataset.
        """
        if dataset_name == "Pets":
            return {
                "Single_Generic": ["A photo of a {label}."],
                "Generic_Ensemble": [
                    "A photo of a {label}.",
                    "A blurry photo of a {label}.",
                    "A photo of the large {label}.",
                    "A photo of the small {label}.",
                    "A photo of the {label} in the wild.",
                    "A photo of a {label} on a couch.",
                    "A bright photo of a {label}.",
                    "A dark photo of a {label}.",
                ],
                "Domain_Specific": [
                    "A photo of a {label}, a type of pet.",
                    "A photo of a {label}, a breed of domestic animal.",
                    "A close-up photo of a {label}.",
                    "A photo of the {label}, a popular pet breed.",
                ],
            }
        elif dataset_name == "Flowers102":
            return {
                "Single_Generic": ["A photo of a {label}."],
                "Generic_Ensemble": [
                    "A photo of a {label}.",
                    "A photo of the {label} in full bloom.",
                    "A close-up photo of a {label}.",
                    "A blurry photo of a {label}.",
                    "A bright photo of a {label}.",
                    "A photo of a {label} with leaves.",
                    "A photo of the beautiful {label}.",
                    "A photo of a {label} in a garden.",
                ],
                "Domain_Specific": [
                    "A photo of a {label}, a type of flower.",
                    "A photo of a {label} with colorful petals.",
                    "A botanical photo of {label}.",
                    "A photo of the {label} plant in bloom.",
                    "A macro photo of a {label} flower.",
                ],
            }
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

    def evaluate(self, dataset_name: str):
        """
        Run evaluation and return per-strategy metrics.
        """
        set_seed(config.SEED)
        test_dataset = load_dataset(dataset_name, train=False, model_type="clip")
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=0,
        )

        # Resolve class names
        if dataset_name == "Pets":
            classnames = test_dataset.classes
        else:
            classnames = [f"flower_{i}" for i in range(102)]

        true_labels = np.array([label for _, label in test_dataset])
        results = {}

        for strategy_name, templates in self.get_prompt_strategies(dataset_name).items():
            print(f"\nEvaluating strategy: {strategy_name} ({len(templates)} templates)")
            self.clip_classifier.set_classes(classnames, templates)
            top1_preds, top5_preds = self.clip_classifier.predict(test_loader, top_k=5)

            acc, low, up, _, _ = compute_accuracy_with_ci(top1_preds, true_labels)
            top5_correct = sum(
                true_labels[i] in top5_preds[i] for i in range(len(true_labels))
            )
            top5_acc = top5_correct / len(true_labels)

            # Per-class accuracy
            per_class_acc = {}
            for cls_idx in range(len(classnames)):
                mask = true_labels == cls_idx
                if mask.sum() > 0:
                    cls_acc = (top1_preds[mask] == cls_idx).mean()
                    per_class_acc[classnames[cls_idx]] = float(cls_acc)

            results[strategy_name] = {
                "top1": float(acc),
                "top1_ci": (float(low), float(up)),
                "top5": float(top5_acc),
                "per_class_acc": per_class_acc,
                "mean_per_class_acc": float(np.mean(list(per_class_acc.values()))),
            }
            print(f"Top-1: {acc * 100:.2f}% [{low * 100:.2f}, {up * 100:.2f}]")
            print(f"Top-5: {top5_acc * 100:.2f}%")
            print(f"Mean Per-Class Acc: {results[strategy_name]['mean_per_class_acc'] * 100:.2f}%")

        return results, classnames


if __name__ == "__main__":
    engine = FineGrainedPromptEngineering()

    print("=" * 60)
    print("Fine-grained Analysis: Oxford-IIIT Pets")
    print("=" * 60)
    pets_results, _ = engine.evaluate("Pets")

    print("\n" + "=" * 60)
    print("Fine-grained Analysis: Flowers102")
    print("=" * 60)
    flowers_results, _ = engine.evaluate("Flowers102")

    import json

    with open(f"{config.SAVE_DIR}/finegrained_results.json", "w") as f:
        json.dump({"Pets": pets_results, "Flowers102": flowers_results}, f, indent=2)
    print("\nFine-grained results saved.")