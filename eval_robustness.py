"""
Evaluate effective robustness on CIFAR-10-C synthetic corruptions.
Compares CLIP zero-shot against the supervised ResNet-18 baseline.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader
from collections import defaultdict
from tqdm import tqdm

import config
from utils import set_seed, wilson_interval
from datasets import load_dataset, DATASET_CLASSNAMES
from corruptions import CorruptionGenerator
from models import ResNetClassifier, CLIPZeroShotClassifier


class RobustnessBenchmark:
    """
    Effective Robustness benchmark.
    Measures accuracy degradation under gaussian noise, blur, pixelation, and contrast changes.
    """

    def __init__(self):
        self.corruption_gen = CorruptionGenerator()
        self.corruption_types = ["gaussian", "blur", "pixelate", "contrast"]

    def _denormalize(self, img: torch.Tensor) -> torch.Tensor:
        """Approximate inverse normalization to [0, 1] for corruption injection."""
        return torch.clamp(img * 0.5 + 0.5, 0, 1)

    def _normalize(self, img: torch.Tensor) -> torch.Tensor:
        """Apply CLIP-specific normalization after corruption."""
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
        return (img - mean) / std

    def generate_corrupted_dataset(
        self, clean_dataset, corruption_type: str, severity: int
    ):
        """
        Generate a fully corrupted tensor dataset.
        """
        corrupted_images, labels = [], []
        print(f"Generating {corruption_type} (severity={severity}) ...")
        for idx in tqdm(range(len(clean_dataset))):
            img, label = clean_dataset[idx]
            img_denorm = self._denormalize(img)
            corrupted = self.corruption_gen.apply_corruption(
                img_denorm, corruption_type, severity
            )
            corrupted_norm = self._normalize(corrupted)
            corrupted_images.append(corrupted_norm)
            labels.append(label)

        return torch.stack(corrupted_images), torch.tensor(labels)

    def evaluate_robustness(self, model_type: str = "clip", model=None):
        """
        Run the full robustness sweep and return structured metrics.
        """
        set_seed(config.SEED)

        clean_dataset = load_dataset("CIFAR10", train=False, model_type="clip")
        true_labels = np.array([label for _, label in clean_dataset])

        results = {
            "clean": {},
            "corrupted": defaultdict(lambda: defaultdict(dict)),
        }

        # 1) Clean accuracy baseline
        print(f"\nEvaluating {model_type} on clean data ...")
        if model_type == "clip":
            clip_clf = CLIPZeroShotClassifier(config.CLIP_MODEL)
            clip_clf.set_classes(
                DATASET_CLASSNAMES["CIFAR10"], ["A photo of a {label}."]
            )
            loader = DataLoader(
                clean_dataset,
                batch_size=config.BATCH_SIZE,
                shuffle=False,
                num_workers=0,
            )
            preds, _ = clip_clf.predict(loader)
            acc = (preds == true_labels).mean()
        else:
            loader = DataLoader(
                clean_dataset,
                batch_size=config.BATCH_SIZE,
                shuffle=False,
                num_workers=0,
            )
            preds = []
            with torch.no_grad():
                for images, _ in loader:
                    images = images.to(config.DEVICE)
                    outputs = model(images)
                    _, predicted = outputs.max(1)
                    preds.extend(predicted.cpu().numpy())
            acc = (np.array(preds) == true_labels).mean()

        results["clean"]["accuracy"] = float(acc)
        results["clean"]["confidence_interval"] = wilson_interval(acc, len(true_labels))
        print(f"Clean accuracy: {acc * 100:.2f}%")

        # 2) Corrupted evaluations
        for corruption in self.corruption_types:
            print(f"\nEvaluating corruption: {corruption}")
            for severity in range(1, 6):
                corrupted_images, _ = self.generate_corrupted_dataset(
                    clean_dataset, corruption, severity
                )
                corrupted_dataset = torch.utils.data.TensorDataset(
                    corrupted_images, torch.tensor(true_labels)
                )
                corrupted_loader = DataLoader(
                    corrupted_dataset,
                    batch_size=config.BATCH_SIZE,
                    shuffle=False,
                    num_workers=0,
                )

                if model_type == "clip":
                    preds, _ = clip_clf.predict(corrupted_loader)
                    acc = (preds == true_labels).mean()
                else:
                    preds = []
                    with torch.no_grad():
                        for images, _ in corrupted_loader:
                            images = images.to(config.DEVICE)
                            outputs = model(images)
                            _, predicted = outputs.max(1)
                            preds.extend(predicted.cpu().numpy())
                    acc = (np.array(preds) == true_labels).mean()

                effective_rob = acc / results["clean"]["accuracy"]
                results["corrupted"][corruption][severity] = {
                    "accuracy": float(acc),
                    "ci": wilson_interval(acc, len(true_labels)),
                    "effective_robustness": float(effective_rob),
                    "relative_drop": float(
                        (results["clean"]["accuracy"] - acc)
                        / results["clean"]["accuracy"]
                    ),
                }
                print(
                    f"  Severity {severity}: {acc * 100:.2f}% "
                    f"(Effective Robustness: {effective_rob:.2f})"
                )

        return results


if __name__ == "__main__":
    # Evaluate CLIP
    print("=" * 60)
    print("CLIP Robustness Evaluation")
    print("=" * 60)
    benchmark = RobustnessBenchmark()
    clip_results = benchmark.evaluate_robustness("clip")

    # Evaluate ResNet
    print("\n" + "=" * 60)
    print("ResNet Robustness Evaluation")
    print("=" * 60)
    resnet_model = ResNetClassifier(num_classes=10).to(config.DEVICE)
    resnet_model.load_state_dict(
        torch.load(f"{config.SAVE_DIR}/resnet_cifar10.pth", map_location=config.DEVICE)
    )
    resnet_model.eval()
    resnet_results = benchmark.evaluate_robustness("resnet", resnet_model)

    # Save
    import json

    with open(f"{config.SAVE_DIR}/robustness_clip.json", "w") as f:
        json.dump(clip_results, f, indent=2)
    with open(f"{config.SAVE_DIR}/robustness_resnet.json", "w") as f:
        json.dump(resnet_results, f, indent=2)
    print("\nRobustness results saved.")