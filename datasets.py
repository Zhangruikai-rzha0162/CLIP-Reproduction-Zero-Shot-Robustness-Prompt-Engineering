"""
Dataset loading and unified preprocessing for both CLIP and ResNet.
All images are resized to 224x224 to ensure a fair comparison.
"""

import torchvision.transforms as T
from torchvision.datasets import CIFAR10, CIFAR100, STL10, OxfordIIITPet, Flowers102
from typing import Optional


def get_preprocess(model_type: str = "clip"):
    """
    Build preprocessing pipeline. CLIP and ResNet both use 224x224 inputs.
    """
    if model_type == "clip":
        # Use CLIP's official preprocess (resize, center crop, normalize)
        import clip
        _, preprocess = clip.load("ViT-B/32", device="cpu")
        return preprocess
    else:
        # ResNet standard ImageNet normalization
        return T.Compose(
            [
                T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )


def load_dataset(
    dataset_name: str, train: bool = True, model_type: str = "clip"
):
    """
    Load standard benchmarks with automatic download.
    """
    preprocess = get_preprocess(model_type)

    if dataset_name == "CIFAR10":
        dataset = CIFAR10(
            root="./data", train=train, download=True, transform=preprocess
        )
    elif dataset_name == "CIFAR100":
        dataset = CIFAR100(
            root="./data", train=train, download=True, transform=preprocess
        )
    elif dataset_name == "STL10":
        split = "train" if train else "test"
        dataset = STL10(
            root="./data", split=split, download=True, transform=preprocess
        )
    elif dataset_name == "Pets":
        split = "trainval" if train else "test"
        dataset = OxfordIIITPet(
            root="./data", split=split, download=True, transform=preprocess
        )
    elif dataset_name == "Flowers102":
        split = "train" if train else "test"
        dataset = Flowers102(
            root="./data", split=split, download=True, transform=preprocess
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return dataset


# Class name mappings for zero-shot text prompts
DATASET_CLASSNAMES = {
    "CIFAR10": [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ],
    "CIFAR100": CIFAR100(root="./data", download=True).classes,
    "STL10": [
        "airplane",
        "bird",
        "car",
        "cat",
        "deer",
        "dog",
        "horse",
        "monkey",
        "ship",
        "truck",
    ],
}