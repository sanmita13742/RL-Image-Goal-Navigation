"""
encoder.py (Version 2)
Clean perception module for MINav.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

import torch
import torch.nn.functional as F
import timm
from PIL import Image
from torchvision import transforms


@dataclass
class PerceptionResult:
    image_path: str
    visual_representation: torch.Tensor
    ssd: float
    valid_goal: bool


class DINOEncoder:
    """
    Frozen DINOv3 encoder with SSD-based goal filtering.

    Pipeline:
        Image
          -> Resize (448x784)
          -> Frozen DINOv3
          -> Visual representation φ(o)
          -> Patch embeddings
          -> 14x25 center crop
          -> SSD (Method 2)
          -> Goal validity
    """

    PATCH_SIZE = 16
    IMAGE_SIZE = (448, 784)
    CROP_H = 14
    CROP_W = 25

    def __init__(
        self,
        model_name: str = "vit_small_patch16_dinov3",
        ssd_threshold: float = 0.02,
        device: Optional[str] = None,
        return_patch_grid: bool = False,
    ):
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.return_patch_grid = return_patch_grid
        self.ssd_threshold = ssd_threshold

        self.model = timm.create_model(model_name, pretrained=True).to(self.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        self.transform = transforms.Compose([
            transforms.Resize(self.IMAGE_SIZE),
            transforms.ToTensor()
        ])

    def _prepare(self, image):
        if isinstance(image, (str, Path)):
            path = str(image)
            image = Image.open(path).convert("RGB")
        elif isinstance(image, Image.Image):
            path = "<PIL>"
        elif torch.is_tensor(image):
            path = "<Tensor>"
            x = image if image.ndim == 4 else image.unsqueeze(0)
            return x.to(self.device), path
        else:
            raise TypeError("Input must be a file path, PIL image, or tensor.")

        x = self.transform(image).unsqueeze(0).to(self.device)
        return x, path

    def _forward(self, x):
        with torch.no_grad():
            visual_representation = self.model(x)
            features = self.model.forward_features(x)
        return visual_representation, features

    @staticmethod
    def _patch_grid(features):
        patches = features[:, 5:]          # remove CLS + 4 register tokens
        b, n, d = patches.shape
        h = 28
        w = n // h
        return patches.reshape(b, h, w, d)

    @classmethod
    def _crop(cls, grid):
        _, h, w, _ = grid.shape
        sh = (h - cls.CROP_H) // 2
        sw = (w - cls.CROP_W) // 2
        return grid[:, sh:sh+cls.CROP_H, sw:sw+cls.CROP_W, :]

    @staticmethod
    def _ssd(crop):
        tokens = crop.reshape(crop.shape[0], -1, crop.shape[-1])
        tokens = F.normalize(tokens, p=2, dim=-1)
        spatial_std = tokens.std(dim=1)
        return spatial_std.mean(dim=-1)

    def process_image(self, image):
        x, path = self._prepare(image)

        visual_representation, features = self._forward(x)

        grid = self._patch_grid(features)
        crop = self._crop(grid)
        ssd = self._ssd(crop)

        result = PerceptionResult(
            image_path=path,
            visual_representation=visual_representation.squeeze(0).cpu(),
            ssd=float(ssd.item()),
            valid_goal=bool(ssd.item() > self.ssd_threshold),
        )

        if self.return_patch_grid:
            return result, grid.cpu()

        return result

    def process_batch(self, images: List[Union[str, Path, Image.Image]]):
        return [self.process_image(img) for img in images]
