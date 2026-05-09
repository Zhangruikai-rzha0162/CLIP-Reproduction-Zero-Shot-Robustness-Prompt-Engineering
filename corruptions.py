"""
Pure-PyTorch implementation of CIFAR-10-C style corruptions.
No external dependencies (e.g., imgaug or albumentations) are required.
"""

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from typing import List


class CorruptionGenerator:
    """
    Generate synthetic corruptions with 5 severity levels.
    Supports gaussian noise, gaussian blur, pixelation, and contrast reduction.
    """

    def __init__(self, severity_levels: List[int] = None):
        if severity_levels is None:
            severity_levels = [1, 2, 3, 4, 5]
        self.severity_levels = severity_levels

        # Severity-dependent intensity parameters
        self.gaussian_std = [0.08, 0.12, 0.18, 0.26, 0.38]
        self.blur_sigma = [0.5, 1.0, 1.5, 2.0, 2.5]
        self.pixelate_factor = [0.6, 0.5, 0.4, 0.3, 0.25]  # Keep ratio
        self.contrast_factor = [0.75, 0.6, 0.45, 0.3, 0.15]

    def apply_corruption(
        self, img: torch.Tensor, corruption_type: str, severity: int
    ) -> torch.Tensor:
        """
        Apply a specific corruption to a single image tensor.

        Args:
            img: Tensor of shape [C, H, W], values in [0, 1].
            corruption_type: One of {'gaussian', 'blur', 'pixelate', 'contrast'}.
            severity: Integer in [1, 5].

        Returns:
            Corrupted image tensor, clamped to [0, 1].
        """
        idx = severity - 1
        assert 0 <= idx < 5, "Severity must be in the range 1-5"

        if corruption_type == "gaussian":
            std = self.gaussian_std[idx]
            noise = torch.randn_like(img) * std
            return torch.clamp(img + noise, 0, 1)

        elif corruption_type == "blur":
            sigma = self.blur_sigma[idx]
            kernel_size = 5 if severity <= 3 else 7
            return TF.gaussian_blur(img, kernel_size=kernel_size, sigma=sigma)

        elif corruption_type == "pixelate":
            ratio = self.pixelate_factor[idx]
            h, w = img.shape[1:]
            small_h, small_w = int(h * ratio), int(w * ratio)
            # Downsample then upsample to create blocky pixelation
            small = F.interpolate(
                img.unsqueeze(0), size=(small_h, small_w), mode="area"
            )
            return F.interpolate(small, size=(h, w), mode="nearest").squeeze(0)

        elif corruption_type == "contrast":
            factor = self.contrast_factor[idx]
            return TF.adjust_contrast(img, contrast_factor=factor)

        else:
            raise ValueError(f"Unknown corruption type: {corruption_type}")