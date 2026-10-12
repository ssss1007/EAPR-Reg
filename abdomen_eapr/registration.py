"""Checkpoint-compatible registration model with explicit APE contracts."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .ape import (
    IXI_APE_VARIANT,
    SEMANTIC_APE_VARIANT,
    AssignmentAwareDetailFusion,
    EntropyAwareDetailFusion,
    load_ape_checkpoint,
    model_mode_from_checkpoint,
)
from .config import CATEGORICAL_ASSIGNMENT_MODE, SEMANTIC_ASSIGNMENT_MODE
from .data import resize_mask
from .medsam import MedSAM2Encoder3D

if "bool" not in np.__dict__:
    np.bool = np.bool_
if "int" not in np.__dict__:
    np.int = int
if "float" not in np.__dict__:
    np.float = float

WORKSPACE = Path(__file__).resolve().parent.parent
IIRP_ROOT = WORKSPACE / "ixi__eapr"
EXTERNAL_ROOT = IIRP_ROOT / "voxelmorph" / "torch" / "external"
for item in (IIRP_ROOT, EXTERNAL_ROOT):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))
os.environ.setdefault("VXM_BACKEND", "pytorch")

from voxelmorph.torch.iirp import (  # noqa: E402
    DecoderBlock,
    FineRefineEncoder,
    RPNet,
    SpatialTransformer,
    _get_or_build_stn,
    _resize_flow_with_scale,
)

THREE_LEVEL_ARCHITECTURE = "ixi_three_level_ape_plus_full_refine_v1"
FIVE_SCALE_ARCHITECTURE = "five_scale_isotropic_ape_v1"
REGISTRATION_ARCHITECTURE = FIVE_SCALE_ARCHITECTURE


def registration_pyramid_shapes(
    size: Sequence[int],
) -> Dict[str, Tuple[int, int, int]]:
    full = tuple(int(value) for value in size)
    if len(full) != 3 or any(value <= 0 or value % 16 for value in full):
        raise ValueError(
            f"Five-scale registration requires positive DHW dimensions "
            f"divisible by 16, got {full}"
        )
    return {
        "l4": tuple(value // 16 for value in full),
        "l3": tuple(value // 8 for value in full),
        "l2": tuple(value // 4 for value in full),
        "l1": tuple(value // 2 for value in full),
        "l0": full,
    }


class DepthPyramidAdapter(nn.Module):
    """Convert slice-wise MedSAM features into isotropic 3-D levels."""

    def __init__(self, channels: int, num_downsamples: int) -> None:
        super().__init__()
        self.channels = int(channels)
        blocks: List[nn.Module] = []
        for _ in range(int(num_downsamples)):
            depth_filter = nn.Conv3d(
                self.channels,
                self.channels,
                kernel_size=(3, 1, 1),
                stride=(2, 1, 1),
                padding=(1, 0, 0),
                groups=self.channels,
                bias=False,
                padding_mode="replicate",
            )
            channel_mix = nn.Conv3d(
                self.channels, self.channels, kernel_size=1, bias=False
            )
            with torch.no_grad():
                depth_filter.weight.zero_()
                depth_filter.weight[:, 0, :, 0, 0] = depth_filter.weight.new_tensor(
                    [0.25, 0.5, 0.25]
                )
                channel_mix.weight.zero_()
                indices = torch.arange(self.channels)
                channel_mix.weight[indices, indices, 0, 0, 0] = 1.0
            blocks.extend([depth_filter, channel_mix])
        self.net = nn.Sequential(*blocks)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        if feature.ndim != 5 or feature.shape[1] != self.channels:
            raise ValueError(
                f"Depth adapter expected [B,{self.channels},D,H,W], "
                f"got {tuple(feature.shape)}"
            )
        return self.net(feature)


class AnatomyGuidance(nn.Module):
    """Frozen MedSAM2 plus a checkpoint-selected APE implementation."""

    def __init__(
        self,
        ape_checkpoint: Path,
        sam2_root: Path,
        sam2_config: Path,
        sam2_checkpoint: Path,
        device: str,
        chunk_size: int = 16,
    ) -> None:
        super().__init__()
        self.ape, metadata = load_ape_checkpoint(ape_checkpoint)
        self.metadata = metadata
        self.model_mode = model_mode_from_checkpoint(metadata)
        self.ape_variant = (
            SEMANTIC_APE_VARIANT
            if self.model_mode == SEMANTIC_ASSIGNMENT_MODE
            else IXI_APE_VARIANT
        )
        self.ape_channels = int(metadata["ape_channels"])
        self.ape_registration_status = str(
            metadata.get("registration_status", "unspecified")
        )
        self.encoder = MedSAM2Encoder3D(
            sam2_root,
            sam2_config,
            sam2_checkpoint,
            device=device,
            chunk_size=chunk_size,
        )
        for parameter in self.parameters():
            parameter.requires_grad = False
        self.eval()

    @torch.no_grad()
    def forward(self, volume: torch.Tensor, body_mask: torch.Tensor):
        encoded = self.encoder(volume)
        if self.model_mode == SEMANTIC_ASSIGNMENT_MODE:
            semantic, assignments, projected, _, masks = self.ape(
                encoded, body_mask
            )
            return semantic, assignments, projected, masks
        reconstructed, assignments, projected, masks = self.ape(encoded, body_mask)
        return reconstructed, assignments, projected, masks


class CleanRegistrationModel(RPNet):
    """Three-level APE registration path matching the established checkpoint.

    The semantic-assignment mode reproduces v3.1 module-for-module. The
    categorical-assignment mode changes only APE/EADF; deformation decoders and the
    full-resolution refinement path retain the same parameter names.
    """

    def __init__(
        self,
        size: Sequence[int],
        first_channel: int,
        guidance: nn.Module,
        *,
        use_fusion: bool = True,
        strict_coarse_composition: bool = False,
        registration_architecture: str = FIVE_SCALE_ARCHITECTURE,
    ) -> None:
        super().__init__(size=tuple(size), first_channel=first_channel, use_ape=False)
        self.use_ape = True
        self.anatomy_guide = guidance
        self.model_mode = str(guidance.model_mode)
        self.ape_variant = str(guidance.ape_variant)
        self.ape_channels = int(guidance.ape_channels)
        self.ape_registration_status = str(guidance.ape_registration_status)
        self.use_fusion = bool(use_fusion)
        self.strict_coarse_composition = bool(strict_coarse_composition)
        if registration_architecture not in (
            THREE_LEVEL_ARCHITECTURE,
            FIVE_SCALE_ARCHITECTURE,
        ):
            raise ValueError(
                f"Unsupported registration architecture: "
                f"{registration_architecture}"
            )
        self.registration_architecture = str(registration_architecture)
        self.pyramid_shapes = (
            registration_pyramid_shapes(size)
            if self.registration_architecture == FIVE_SCALE_ARCHITECTURE
            else None
        )
        if self.use_fusion:
            self.ape_eadf = (
                AssignmentAwareDetailFusion(self.ape_channels)
                if self.model_mode == SEMANTIC_ASSIGNMENT_MODE
                else EntropyAwareDetailFusion(self.ape_channels)
            )
        self.ape_decoder3 = DecoderBlock(
            self.ape_channels, self.ape_channels, self.ape_channels
        )
        self.ape_decoder2 = DecoderBlock(
            self.ape_channels, self.ape_channels, self.ape_channels
        )
        self.ape_decoder1 = DecoderBlock(
            self.ape_channels, self.ape_channels, self.ape_channels
        )
        if self.registration_architecture == FIVE_SCALE_ARCHITECTURE:
            self.ape_depth_l2 = DepthPyramidAdapter(
                self.ape_channels, num_downsamples=2
            )
            self.ape_depth_l3 = DepthPyramidAdapter(
                self.ape_channels, num_downsamples=3
            )
            self.ape_depth_l4 = DepthPyramidAdapter(
                self.ape_channels, num_downsamples=4
            )
            self.half_refine_encoder = FineRefineEncoder(
                in_channel=1, out_channel=32
            )
            self.half_refine_decoder = DecoderBlock(
                x_channel=32, y_channel=32, out_channel=32
            )
        self.fine_refine_encoder = FineRefineEncoder(in_channel=1, out_channel=32)
        self.fine_refine_decoder = DecoderBlock(
            x_channel=32, y_channel=32, out_channel=32
        )
        self.ape_stn_cache = nn.ModuleDict()

    def _isotropic_pyramid(
        self, pyramid: Sequence[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        levels = (
            self.ape_depth_l2(pyramid[0]),
            self.ape_depth_l3(pyramid[1]),
            self.ape_depth_l4(pyramid[2]),
        )
        for name, feature in zip(("l2", "l3", "l4"), levels):
            actual = tuple(int(value) for value in feature.shape[2:])
            expected = self.pyramid_shapes[name]
            if actual != expected:
                raise ValueError(
                    f"{name.upper()} shape mismatch: expected {expected}, got {actual}"
                )
        return levels

    def forward(
        self,
        moving: torch.Tensor,
        fixed: torch.Tensor,
        moving_mask: torch.Tensor,
        fixed_mask: torch.Tensor,
        return_aux: bool = False,
    ):
        with torch.no_grad():
            self.anatomy_guide.eval()
            moving_rec, moving_assign, moving_proj, moving_masks = self.anatomy_guide(
                moving, moving_mask
            )
            fixed_rec, fixed_assign, fixed_proj, fixed_masks = self.anatomy_guide(
                fixed, fixed_mask
            )
        moving_proj = [
            feature * mask for feature, mask in zip(moving_proj, moving_masks)
        ]
        fixed_proj = [
            feature * mask for feature, mask in zip(fixed_proj, fixed_masks)
        ]
        if self.use_fusion:
            moving_pyramid = self.ape_eadf(
                moving_rec, moving_assign, moving_proj
            )
            fixed_pyramid = self.ape_eadf(fixed_rec, fixed_assign, fixed_proj)
        elif self.model_mode == SEMANTIC_ASSIGNMENT_MODE:
            moving_pyramid = [
                projected + semantic
                for projected, semantic in zip(moving_proj, moving_rec)
            ]
            fixed_pyramid = [
                projected + semantic
                for projected, semantic in zip(fixed_proj, fixed_rec)
            ]
        else:
            moving_pyramid, fixed_pyramid = moving_rec, fixed_rec
        moving_pyramid = [
            feature * mask for feature, mask in zip(moving_pyramid, moving_masks)
        ]
        fixed_pyramid = [
            feature * mask for feature, mask in zip(fixed_pyramid, fixed_masks)
        ]
        if self.registration_architecture == FIVE_SCALE_ARCHITECTURE:
            moving_l1, moving_l2, moving_l3 = self._isotropic_pyramid(
                moving_pyramid
            )
            fixed_l1, fixed_l2, fixed_l3 = self._isotropic_pyramid(fixed_pyramid)
        else:
            moving_l1, moving_l2, moving_l3 = moving_pyramid
            fixed_l1, fixed_l2, fixed_l3 = fixed_pyramid
        stn3 = _get_or_build_stn(
            self.ape_stn_cache, moving_l3.shape[2:], moving_l3.device
        )
        stn2 = _get_or_build_stn(
            self.ape_stn_cache, moving_l2.shape[2:], moving_l2.device
        )
        stn1 = _get_or_build_stn(
            self.ape_stn_cache, moving_l1.shape[2:], moving_l1.device
        )

        coarse_u = self.ape_decoder3(moving_l3, fixed_l3)
        coarse_v = self.ape_decoder3(stn3(moving_l3, coarse_u), fixed_l3)
        flow3 = (
            stn3(coarse_u, coarse_v) + coarse_v
            if self.strict_coarse_composition
            else coarse_u + coarse_v
        )
        flow2_up = _resize_flow_with_scale(flow3, moving_l2.shape[2:])
        delta2 = self.ape_decoder2(stn2(moving_l2, flow2_up), fixed_l2)
        flow2 = stn2(flow2_up, delta2) + delta2
        flow1_up = _resize_flow_with_scale(flow2, moving_l1.shape[2:])
        delta1 = self.ape_decoder1(stn1(moving_l1, flow1_up), fixed_l1)
        flow1 = stn1(flow1_up, delta1) + delta1

        moving_foreground = moving * moving_mask.to(moving.dtype)
        fixed_foreground = fixed * fixed_mask.to(fixed.dtype)
        full_size = moving.shape[2:]
        if self.registration_architecture == FIVE_SCALE_ARCHITECTURE:
            half_size = self.pyramid_shapes["l1"]
            moving_half = F.interpolate(
                moving_foreground,
                size=half_size,
                mode="trilinear",
                align_corners=True,
            )
            fixed_half = F.interpolate(
                fixed_foreground,
                size=half_size,
                mode="trilinear",
                align_corners=True,
            )
            moving_half_ref = self.half_refine_encoder(moving_half)
            fixed_half_ref = self.half_refine_encoder(fixed_half)
            half_stn = _get_or_build_stn(
                self.ape_stn_cache, half_size, moving.device
            )
            flow_half_up = _resize_flow_with_scale(flow1, half_size)
            delta_half = self.half_refine_decoder(
                half_stn(moving_half_ref, flow_half_up), fixed_half_ref
            )
            flow_half = half_stn(flow_half_up, delta_half) + delta_half
            flow_up = _resize_flow_with_scale(flow_half, full_size)
        else:
            flow_up = _resize_flow_with_scale(flow1, full_size)
        moving_ref = self.fine_refine_encoder(moving_foreground)
        fixed_ref = self.fine_refine_encoder(fixed_foreground)
        full_stn = _get_or_build_stn(self.ape_stn_cache, full_size, moving.device)
        delta_full = self.fine_refine_decoder(
            full_stn(moving_ref, flow_up), fixed_ref
        )
        flow = full_stn(flow_up, delta_full) + delta_full
        warped = full_stn(moving, flow)
        if return_aux:
            return warped, flow, {
                "moving_assignments": moving_assign,
                "fixed_assignments": fixed_assign,
                "fixed_stage_masks": fixed_masks,
                "model_mode": self.model_mode,
            }
        return warped, flow


def flow_gradient_loss(flow: torch.Tensor) -> torch.Tensor:
    dz = (flow[:, :, 1:] - flow[:, :, :-1]).square().mean()
    dy = (flow[:, :, :, 1:] - flow[:, :, :, :-1]).square().mean()
    dx = (flow[:, :, :, :, 1:] - flow[:, :, :, :, :-1]).square().mean()
    return (dz + dy + dx) / 3


def negative_jacobian_percentage(flow: torch.Tensor) -> float:
    if flow.ndim != 5 or flow.shape[0] != 1 or flow.shape[1] != 3:
        raise ValueError(f"Expected [1,3,D,H,W], got {tuple(flow.shape)}")
    displacement = flow.detach()[0].float()
    dz = torch.gradient(displacement, dim=1)[0]
    dy = torch.gradient(displacement, dim=2)[0]
    dx = torch.gradient(displacement, dim=3)[0]
    j00, j01, j02 = 1 + dz[0], dy[0], dx[0]
    j10, j11, j12 = dz[1], 1 + dy[1], dx[1]
    j20, j21, j22 = dz[2], dy[2], 1 + dx[2]
    determinant = (
        j00 * (j11 * j22 - j12 * j21)
        - j01 * (j10 * j22 - j12 * j20)
        + j02 * (j10 * j21 - j11 * j20)
    )
    return 100 * float((determinant < 0).float().mean())


def ape_assignment_consistency_loss(
    moving_assignments: Sequence[torch.Tensor],
    fixed_assignments: Sequence[torch.Tensor],
    fixed_masks: Sequence[torch.Tensor],
    full_flow: torch.Tensor,
    stn_cache: nn.ModuleDict,
    stage_weights: Sequence[float],
    epsilon: float = 1e-6,
) -> torch.Tensor:
    if not (
        len(moving_assignments)
        == len(fixed_assignments)
        == len(fixed_masks)
        == len(stage_weights)
    ):
        raise ValueError("APE consistency stages must have matching lengths")
    total = full_flow.new_zeros([])
    weight_total = 0.0
    for moving, fixed, mask, weight in zip(
        moving_assignments, fixed_assignments, fixed_masks, stage_weights
    ):
        weight = float(weight)
        if weight <= 0:
            continue
        if moving.shape != fixed.shape:
            raise ValueError("APE consistency assignment shapes differ")
        stage_flow = _resize_flow_with_scale(full_flow, moving.shape[2:])
        stn = _get_or_build_stn(
            stn_cache, moving.shape[2:], full_flow.device
        )
        warped = stn(moving, stage_flow) * mask
        target = fixed * mask
        dims = tuple(range(2, warped.ndim))
        intersection = (warped * target).sum(dim=dims)
        denominator = warped.square().sum(dim=dims) + target.square().sum(dim=dims)
        dice = (2 * intersection + epsilon) / (denominator + epsilon)
        total = total + weight * (1 - dice).mean()
        weight_total += weight
    if weight_total <= 0:
        raise ValueError("APE consistency needs a positive stage weight")
    return total / weight_total


def registration_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {
        key: value
        for key, value in model.state_dict().items()
        if not key.startswith(("anatomy_guide.", "ape_stn_cache."))
    }


def infer_registration_architecture(
    checkpoint: Dict[str, Any],
) -> str:
    recorded = checkpoint.get("registration_architecture")
    if recorded in (THREE_LEVEL_ARCHITECTURE, FIVE_SCALE_ARCHITECTURE):
        return str(recorded)
    state = checkpoint.get("model_state_dict", {})
    if any(
        key.startswith(("ape_depth_l2.", "half_refine_encoder."))
        for key in state
    ):
        return FIVE_SCALE_ARCHITECTURE
    return THREE_LEVEL_ARCHITECTURE


def load_registration_state(
    model: nn.Module,
    state: Dict[str, torch.Tensor],
    *,
    warm_start: bool = False,
) -> Dict[str, List[str]]:
    if warm_start:
        target = model.state_dict()
        compatible = {
            key: value
            for key, value in state.items()
            if key in target
            and target[key].shape == value.shape
            and not key.startswith("ape_eadf.")
        }
        missing, unexpected = model.load_state_dict(compatible, strict=False)
        return {
            "loaded": sorted(compatible),
            "missing": sorted(missing),
            "unexpected": sorted(unexpected),
        }
    missing, unexpected = model.load_state_dict(state, strict=False)
    invalid_missing = [
        key
        for key in missing
        if not key.startswith(("anatomy_guide.", "ape_stn_cache."))
    ]
    if invalid_missing or unexpected:
        raise ValueError(
            f"Registration checkpoint mismatch: missing={invalid_missing}, "
            f"unexpected={unexpected}"
        )
    return {"loaded": sorted(state), "missing": sorted(missing), "unexpected": []}
