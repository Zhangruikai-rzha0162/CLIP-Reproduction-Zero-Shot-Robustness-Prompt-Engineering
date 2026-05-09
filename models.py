"""
Model wrappers: supervised ResNet-18 and CLIP zero-shot classifier.
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import List, Dict, Tuple, Optional
import numpy as np
from tqdm import tqdm

import config


class ResNetClassifier(nn.Module):
    """
    ResNet-18 adapted for CIFAR-10/100.
    Only the final FC layer is modified; input 224x224 is natively supported.
    """

    def __init__(self, num_classes: int = 10, pretrained: bool = False):
        super().__init__()
        self.model = models.resnet18(pretrained=pretrained)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class CLIPZeroShotClassifier:
    """
    CLIP zero-shot classifier supporting multiple prompt strategies:
    contextless, prompted template, and ensemble averaging.
    """

    def __init__(self, model_name: str = "ViT-B/32"):
        import clip

        self.model, self.preprocess = clip.load(model_name, device=config.DEVICE)
        self.model.eval()
        self.classnames: Optional[List[str]] = None
        self.templates: Optional[List[str]] = None

    def set_classes(
        self, classnames: List[str], templates: Optional[List[str]] = None
    ) -> None:
        """
        Configure class names and prompt templates.
        If templates is None, defaults to contextless (raw class names).
        """
        self.classnames = classnames
        self.templates = templates if templates else ["{label}"]

    def compute_text_embeddings(self) -> torch.Tensor:
        """
        Encode all class prompts and average multiple templates per class.
        """
        import clip

        if self.classnames is None:
            raise ValueError("set_classes() must be called before encoding.")

        all_embeddings = []
        with torch.no_grad():
            for classname in self.classnames:
                texts = [
                    template.format(label=classname) for template in self.templates
                ]
                tokens = clip.tokenize(texts).to(config.DEVICE)
                class_embeddings = self.model.encode_text(tokens)
                class_embeddings = class_embeddings / class_embeddings.norm(
                    dim=-1, keepdim=True
                )
                # Average over templates
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding = class_embedding / class_embedding.norm()
                all_embeddings.append(class_embedding)

        return torch.stack(all_embeddings, dim=0)

    @torch.no_grad()
    def predict(
        self, dataloader: torch.utils.data.DataLoader, top_k: int = 5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run inference over the entire dataset.

        Returns:
            top1_preds: [N] array of top-1 class indices.
            top5_preds: [N, top_k] array of top-k class indices.
        """
        text_embeddings = self.compute_text_embeddings()

        all_top1, all_top5 = [], []
        for images, _ in tqdm(dataloader, desc="CLIP inference"):
            images = images.to(config.DEVICE)

            image_features = self.model.encode_image(images)
            image_features = image_features / image_features.norm(
                dim=-1, keepdim=True
            )

            # Cosine similarity scaled by 100 (CLIP temperature)
            similarity = 100.0 * image_features @ text_embeddings.T

            indices = similarity.topk(top_k, dim=-1)[1]
            all_top1.append(indices[:, 0].cpu().numpy())
            all_top5.append(indices.cpu().numpy())

        return np.concatenate(all_top1), np.vstack(all_top5)