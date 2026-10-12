"""Strict AbdomenCTCT readers, masks, metrics, and registration losses."""

from __future__ import annotations

import json
import pickle
import random
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from torch.utils.data import Dataset

EXPECTED_HWD = (160, 192, 224)
EXPECTED_DHW = (224, 160, 192)
ORGAN_LABELS = tuple(range(1, 14))


def make_body_mask_np(
    image_dhw: np.ndarray,
    threshold: float = 0.25,
    closing_iterations: int = 2,
    dilation_iterations: int = 5,
) -> np.ndarray:
    """Exact CT-mask fallback used by the established abdomen pipeline."""

    image = np.asarray(image_dhw, dtype=np.float32)
    if image.ndim != 3 or not np.isfinite(image).all():
        raise ValueError("Body-mask input must be a finite DHW volume")
    structure = ndimage.generate_binary_structure(3, 1)
    mask = image > float(threshold)
    if closing_iterations > 0:
        mask = ndimage.binary_closing(
            mask, structure=structure, iterations=int(closing_iterations)
        )
    components, count = ndimage.label(mask, structure=structure)
    if count == 0:
        raise ValueError("CT body-mask extraction found no foreground component")
    sizes = np.bincount(components.ravel())
    sizes[0] = 0
    mask = components == int(np.argmax(sizes))
    mask = np.stack(
        [ndimage.binary_fill_holes(section) for section in mask], axis=0
    )
    if dilation_iterations > 0:
        mask = ndimage.binary_dilation(
            mask, structure=structure, iterations=int(dilation_iterations)
        )
    return np.ascontiguousarray(mask, dtype=np.bool_)


def resize_mask(mask: torch.Tensor, size: Sequence[int]) -> torch.Tensor:
    if mask.ndim != 5 or mask.shape[1] != 1:
        raise ValueError(f"Expected mask [B,1,D,H,W], got {tuple(mask.shape)}")
    return F.interpolate(mask.float(), size=tuple(map(int, size)), mode="nearest")


def masked_mean(
    value: torch.Tensor, mask: torch.Tensor, epsilon: float = 1e-8
) -> torch.Tensor:
    denominator = mask.expand_as(value).sum().clamp_min(epsilon)
    return (value * mask).sum() / denominator


def _read_payload(
    path: Path, mask_cache: Optional[Dict[Path, np.ndarray]] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    with path.open("rb") as stream, warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r"numpy\.core\.numeric is deprecated.*")
        payload = pickle.load(stream)
    if not isinstance(payload, (tuple, list)) or len(payload) < 2:
        raise ValueError(f"{path}: expected (image, label[, body_mask])")
    image_hwd = np.asarray(payload[0], dtype=np.float32)
    label_hwd = np.asarray(payload[1], dtype=np.int64)
    if image_hwd.shape != EXPECTED_HWD or label_hwd.shape != EXPECTED_HWD:
        raise ValueError(
            f"{path}: expected HWD={EXPECTED_HWD}, got "
            f"image={image_hwd.shape}, label={label_hwd.shape}"
        )
    if not np.isfinite(image_hwd).all() or image_hwd.min() < 0 or image_hwd.max() > 1:
        raise ValueError(f"{path}: image must be finite and normalized to [0,1]")
    image = np.ascontiguousarray(image_hwd.transpose(2, 0, 1))
    label = np.ascontiguousarray(label_hwd.transpose(2, 0, 1))
    if len(payload) >= 3:
        mask_hwd = np.asarray(payload[2], dtype=np.bool_)
        if mask_hwd.shape != EXPECTED_HWD:
            raise ValueError(f"{path}: body-mask shape mismatch")
        mask = np.ascontiguousarray(mask_hwd.transpose(2, 0, 1))
    else:
        mask = None if mask_cache is None else mask_cache.get(path)
        if mask is None:
            mask = make_body_mask_np(image)
            if mask_cache is not None:
                mask_cache[path] = mask
    missed = int(np.count_nonzero((label > 0) & ~mask))
    if missed:
        raise ValueError(f"{path}: body mask excludes {missed} labeled voxels")
    return image, label, mask


class AbdomenCaseDataset(Dataset):
    def __init__(self, root: Path) -> None:
        self.root = Path(root).resolve()
        self.files = sorted(self.root.glob("*.pkl"))
        if not self.files:
            raise FileNotFoundError(f"No pkl cases in {self.root}")
        self._mask_cache: Dict[Path, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.files)

    @staticmethod
    def case_id(path: Path) -> str:
        parts = path.stem.split("_")
        return parts[1] if len(parts) >= 3 else path.stem

    def load_case(self, index: int):
        path = self.files[index]
        image, label, mask = _read_payload(path, self._mask_cache)
        return (
            torch.from_numpy(image).unsqueeze(0),
            torch.from_numpy(label),
            torch.from_numpy(mask).unsqueeze(0),
            self.case_id(path),
        )

    def __getitem__(self, index: int):
        return self.load_case(index)


class AbdomenPairDataset(AbdomenCaseDataset):
    def __getitem__(self, index: int):
        moving_index = random.randrange(len(self))
        while moving_index == index and len(self) > 1:
            moving_index = random.randrange(len(self))
        fixed, fixed_label, fixed_mask, _ = self.load_case(index)
        moving, moving_label, moving_mask, _ = self.load_case(moving_index)
        return fixed, moving, fixed_label, moving_label, fixed_mask, moving_mask


class AbdomenRegistrationPairs(Dataset):
    def __init__(self, case_dir: Path, manifest: Path) -> None:
        self.case_dir = Path(case_dir).resolve()
        self.pairs = json.loads(Path(manifest).read_text(encoding="utf-8"))
        if not isinstance(self.pairs, list) or not self.pairs:
            raise ValueError(f"{manifest}: empty pair manifest")
        self._mask_cache: Dict[Path, np.ndarray] = {}
        for pair in self.pairs:
            for key in ("moving", "fixed"):
                path = self.case_dir / pair[key]
                if not path.is_file():
                    raise FileNotFoundError(path)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        pair = self.pairs[index]
        moving = _read_payload(self.case_dir / pair["moving"], self._mask_cache)
        fixed = _read_payload(self.case_dir / pair["fixed"], self._mask_cache)
        return {
            "moving": torch.from_numpy(moving[0]).unsqueeze(0),
            "moving_label": torch.from_numpy(moving[1]),
            "moving_mask": torch.from_numpy(moving[2]).unsqueeze(0),
            "fixed": torch.from_numpy(fixed[0]).unsqueeze(0),
            "fixed_label": torch.from_numpy(fixed[1]),
            "fixed_mask": torch.from_numpy(fixed[2]).unsqueeze(0),
            "moving_id": pair.get("moving_id", pair["moving"]),
            "fixed_id": pair.get("fixed_id", pair["fixed"]),
        }


def masked_local_ncc_loss(
    moving: torch.Tensor,
    fixed: torch.Tensor,
    fixed_mask: torch.Tensor,
    window: int = 9,
    epsilon: float = 1e-5,
) -> torch.Tensor:
    if moving.shape != fixed.shape or moving.ndim != 5 or moving.shape[1] != 1:
        raise ValueError("NCC expects matching scalar BCDHW images")
    if int(window) <= 0 or int(window) % 2 == 0:
        raise ValueError("NCC window must be a positive odd integer")
    mask = resize_mask(fixed_mask, moving.shape[2:]).to(moving.dtype)
    kernel = torch.ones(
        (1, 1, window, window, window), dtype=moving.dtype, device=moving.device
    )
    padding = window // 2

    def local_sum(value: torch.Tensor) -> torch.Tensor:
        return F.conv3d(value * mask, kernel, padding=padding)

    count = F.conv3d(mask, kernel, padding=padding).clamp_min(1.0)
    moving_sum, fixed_sum = local_sum(moving), local_sum(fixed)
    moving_mean, fixed_mean = moving_sum / count, fixed_sum / count
    cross = (
        local_sum(moving * fixed)
        - moving_mean * fixed_sum
        - fixed_mean * moving_sum
        + moving_mean * fixed_mean * count
    )
    moving_var = (
        local_sum(moving.square())
        - 2 * moving_mean * moving_sum
        + moving_mean.square() * count
    )
    fixed_var = (
        local_sum(fixed.square())
        - 2 * fixed_mean * fixed_sum
        + fixed_mean.square() * count
    )
    ncc = cross.square() / (
        moving_var.clamp_min(0) * fixed_var.clamp_min(0) + epsilon
    )
    return -masked_mean(ncc, mask * (count > 1).to(mask.dtype))


def body_mask_dice_loss(
    warped_moving_mask: torch.Tensor, fixed_mask: torch.Tensor
) -> torch.Tensor:
    fixed = resize_mask(fixed_mask, warped_moving_mask.shape[2:]).to(
        warped_moving_mask.dtype
    )
    dims = (1, 2, 3, 4)
    intersection = (warped_moving_mask * fixed).sum(dim=dims)
    denominator = warped_moving_mask.sum(dim=dims) + fixed.sum(dim=dims)
    return (1 - (2 * intersection + 1e-6) / (denominator + 1e-6)).mean()


def organ_dice_full_grid(
    warped: np.ndarray,
    fixed: np.ndarray,
    labels: Sequence[int] = ORGAN_LABELS,
) -> Dict[int, float]:
    if np.asarray(warped).shape != np.asarray(fixed).shape:
        raise ValueError("Dice inputs must have matching shapes")
    result: Dict[int, float] = {}
    for label in labels:
        a, b = warped == int(label), fixed == int(label)
        denominator = int(a.sum() + b.sum())
        if denominator:
            result[int(label)] = 2.0 * float(np.logical_and(a, b).sum()) / denominator
    return result


def binary_surface_distance_metrics(
    warped_mask: np.ndarray,
    fixed_mask: np.ndarray,
    spacing_mm: Sequence[float],
) -> Dict[str, Any]:
    """Compute symmetric HD95 and ASSD in physical units for one binary ROI.

    Surfaces use 6-connectivity. If exactly one mask is empty, both distances
    receive the physical full-grid diagonal as a conservative finite penalty.
    If both masks are empty, the ROI is marked not applicable.
    """

    warped = np.asarray(warped_mask, dtype=np.bool_)
    fixed = np.asarray(fixed_mask, dtype=np.bool_)
    if warped.ndim != 3 or fixed.ndim != 3 or warped.shape != fixed.shape:
        raise ValueError("Surface-distance inputs must be matching 3-D masks")
    spacing = np.asarray(tuple(float(value) for value in spacing_mm), dtype=np.float64)
    if spacing.shape != (3,) or not np.isfinite(spacing).all() or np.any(spacing <= 0):
        raise ValueError("Surface-distance spacing must contain three positive values")

    warped_present = bool(warped.any())
    fixed_present = bool(fixed.any())
    if not warped_present and not fixed_present:
        return {
            "hd95_mm": None,
            "assd_mm": None,
            "empty_case": "both_empty",
            "penalized": False,
        }
    if not warped_present or not fixed_present:
        diagonal = float(
            np.linalg.norm((np.asarray(warped.shape, dtype=np.float64) - 1) * spacing)
        )
        return {
            "hd95_mm": diagonal,
            "assd_mm": diagonal,
            "empty_case": (
                "warped_empty" if not warped_present else "fixed_empty"
            ),
            "penalized": True,
        }

    union_coordinates = np.where(np.logical_or(warped, fixed))
    slices = tuple(
        slice(
            max(0, int(axis_values.min()) - 1),
            min(warped.shape[axis], int(axis_values.max()) + 2),
        )
        for axis, axis_values in enumerate(union_coordinates)
    )
    warped = warped[slices]
    fixed = fixed[slices]
    structure = ndimage.generate_binary_structure(3, 1)
    warped_surface = np.logical_xor(
        warped,
        ndimage.binary_erosion(
            warped, structure=structure, border_value=0
        ),
    )
    fixed_surface = np.logical_xor(
        fixed,
        ndimage.binary_erosion(
            fixed, structure=structure, border_value=0
        ),
    )
    distance_to_fixed = ndimage.distance_transform_edt(
        np.logical_not(fixed_surface), sampling=spacing
    )
    distance_to_warped = ndimage.distance_transform_edt(
        np.logical_not(warped_surface), sampling=spacing
    )
    warped_to_fixed = distance_to_fixed[warped_surface]
    fixed_to_warped = distance_to_warped[fixed_surface]
    symmetric = np.concatenate([warped_to_fixed, fixed_to_warped])
    return {
        "hd95_mm": float(np.percentile(symmetric, 95)),
        "assd_mm": float(
            0.5 * (warped_to_fixed.mean() + fixed_to_warped.mean())
        ),
        "empty_case": None,
        "penalized": False,
    }
