"""Frozen MedSAM2 slice encoder used by both APE and registration."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAdapter(nn.Module):
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 5 or image.shape[1] != 1:
            raise ValueError(f"Expected [B,1,D,H,W], got {tuple(image.shape)}")
        depth = image.shape[2]
        padded = F.pad(image, (0, 0, 0, 0, 1, 1), mode="replicate")
        return torch.cat(
            [
                padded[:, :, 0:depth],
                padded[:, :, 1 : depth + 1],
                padded[:, :, 2 : depth + 2],
            ],
            dim=1,
        )


def _import_build_sam2(root: Path):
    for candidate in (root.resolve(), root.resolve().parent):
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
    errors = []
    for module_name in ("sam2main.sam2.build_sam", "sam2.build_sam"):
        try:
            module = __import__(module_name, fromlist=["build_sam2"])
            return module.build_sam2
        except (ImportError, AttributeError) as error:
            errors.append(f"{module_name}: {error}")
    raise ImportError("Could not import build_sam2. " + " | ".join(errors))


def _package_roots(root: Path) -> List[Path]:
    root = root.expanduser().resolve()
    candidates = [root / "sam2", root / "sam2main" / "sam2"]
    if root.name == "sam2":
        candidates.insert(0, root)
    return list(dict.fromkeys(path for path in candidates if path.is_dir()))


def resolve_config_name(root: Path, config: Path) -> str:
    if not config.is_absolute():
        value = str(config).replace("\\", "/").lstrip("./")
        if not value:
            raise ValueError("SAM2 config name is empty")
        return value
    absolute = config.expanduser().resolve()
    roots = _package_roots(root)
    for package_root in roots:
        try:
            return absolute.relative_to(package_root).as_posix()
        except ValueError:
            pass
    matches: List[Tuple[Path, Path]] = []
    for package_root in roots:
        matches.extend((package_root, item) for item in package_root.rglob(absolute.name))
    if len(matches) == 1:
        return matches[0][1].relative_to(matches[0][0]).as_posix()
    raise FileNotFoundError(
        f"Cannot map {absolute} into SAM2 Hydra package roots: {roots}"
    )


class MedSAM2Encoder3D(nn.Module):
    def __init__(
        self,
        sam2_root: Path,
        config: Path,
        checkpoint: Path,
        device: str,
        chunk_size: int = 16,
    ) -> None:
        super().__init__()
        if not checkpoint.expanduser().is_file():
            raise FileNotFoundError(checkpoint)
        build_sam2 = _import_build_sam2(sam2_root)
        model = build_sam2(
            resolve_config_name(sam2_root, config),
            str(checkpoint.expanduser().resolve()),
            device=device,
            apply_postprocessing=False,
        )
        self.backbone = model.image_encoder
        self.backbone.eval()
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        self.adapter = ChannelAdapter()
        self.chunk_size = int(chunk_size)

    @torch.no_grad()
    def forward(self, image: torch.Tensor):
        adapted = self.adapter(image)
        batch, channels, depth, height, width = adapted.shape
        slices = adapted.permute(0, 2, 1, 3, 4).reshape(
            batch * depth, channels, height, width
        )
        collected = [[], [], []]
        for start in range(0, slices.shape[0], self.chunk_size):
            fpn = self.backbone(slices[start : start + self.chunk_size])[
                "backbone_fpn"
            ]
            if len(fpn) != 3:
                raise ValueError(f"Expected three FPN levels, got {len(fpn)}")
            for stage, feature in enumerate(fpn):
                collected[stage].append(feature)
        result = []
        for stage_features in collected:
            feature = torch.cat(stage_features, dim=0)
            _, feature_channels, feature_height, feature_width = feature.shape
            feature = feature.view(
                batch, depth, feature_channels, feature_height, feature_width
            )
            result.append(feature.permute(0, 2, 1, 3, 4).contiguous())
        return result
