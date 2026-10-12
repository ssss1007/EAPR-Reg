"""APE implementations, fusion modules, and checkpoint compatibility utilities."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import (
    DEFAULT_APE_CHANNELS,
    DEFAULT_K_LIST,
    DEFAULT_LABELS_PER_STAGE,
    DEFAULT_STAGE_WEIGHTS,
    DEFAULT_TEMPERATURE,
    CATEGORICAL_ASSIGNMENT_MODE,
    HISTORICAL_SEMANTIC_MODE,
    SEMANTIC_ASSIGNMENT_MODE,
    validate_stage_configuration,
)
from .data import masked_mean, resize_mask

SEMANTIC_APE_VARIANT = "hierarchical_semantic_assignment_v31"
IXI_APE_VARIANT = "abdomen_masked_categorical_softmax_ixi_style_v1"
HISTORICAL_SEMANTIC_APE_VARIANT = (
    "hierarchical_calibrated_sigmoid_no_residual_v31"
)


class SemanticAssignmentAPE(nn.Module):
    """Exact v3.1 semantic-assignment layout and state-dict contract."""

    def __init__(
        self,
        k_list: Sequence[int] = DEFAULT_K_LIST,
        labels_per_stage: Sequence[Sequence[int]] = DEFAULT_LABELS_PER_STAGE,
        temperature: float = DEFAULT_TEMPERATURE,
        channels: int = DEFAULT_APE_CHANNELS,
        initial_slot_bias: float = -2.0,
    ) -> None:
        super().__init__()
        validate_stage_configuration(k_list, labels_per_stage)
        self.K = [int(value) for value in k_list]
        self.labels_per_stage = [list(map(int, labels)) for labels in labels_per_stage]
        self.semantic_counts = [len(labels) for labels in self.labels_per_stage]
        self.discovery_counts = [
            k - count for k, count in zip(self.K, self.semantic_counts)
        ]
        self.channels = [int(channels)] * 3
        self.register_buffer("current_temp", torch.tensor(float(temperature)))
        self.projectors = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv3d(256, channels, 3, padding=1, bias=False),
                    nn.InstanceNorm3d(channels, affine=True),
                    nn.LeakyReLU(0.2, inplace=True),
                )
                for _ in range(3)
            ]
        )
        self.semantic_protos = nn.ParameterList(
            [
                nn.Parameter(torch.randn(count, channels) * 0.02)
                for count in self.semantic_counts
            ]
        )
        self.semantic_biases = nn.ParameterList(
            [
                nn.Parameter(torch.full((count,), float(initial_slot_bias)))
                for count in self.semantic_counts
            ]
        )
        self.discovery_protos = nn.ParameterList()
        for count in self.discovery_counts:
            parameter = nn.Parameter(torch.randn(count, channels) * 0.02)
            parameter.requires_grad_(False)
            self.discovery_protos.append(parameter)

    def _forward_impl(self, features, body_mask):
        projected = [
            projector(feature) for projector, feature in zip(self.projectors, features)
        ]
        semantic_features, assignments, discovery_assignments = [], [], []
        stage_masks, logits_per_stage = [], []
        for feature, prototypes, biases, discovery in zip(
            projected,
            self.semantic_protos,
            self.semantic_biases,
            self.discovery_protos,
        ):
            batch, channels, depth, height, width = feature.shape
            stage_mask = resize_mask(body_mask, (depth, height, width)).to(feature.dtype)
            flattened = feature.flatten(2).transpose(1, 2)
            cosine = torch.einsum(
                "bnc,kc->bnk",
                F.normalize(flattened, dim=-1),
                F.normalize(prototypes, dim=-1),
            )
            logits_flat = cosine / self.current_temp.clamp_min(1e-4)
            logits_flat = logits_flat + biases.view(1, 1, -1)
            assignment_flat = torch.sigmoid(logits_flat)
            assignment = assignment_flat.view(
                batch, depth, height, width, prototypes.shape[0]
            ).permute(0, 4, 1, 2, 3)
            logits = logits_flat.view(
                batch, depth, height, width, prototypes.shape[0]
            ).permute(0, 4, 1, 2, 3)
            semantic = torch.einsum(
                "bnk,kc->bnc", assignment_flat, F.normalize(prototypes, dim=-1)
            )
            semantic = semantic.transpose(1, 2).view(
                batch, channels, depth, height, width
            )
            semantic_features.append(semantic * stage_mask)
            assignments.append(assignment * stage_mask)
            discovery_assignments.append(
                feature.new_zeros(
                    (batch, discovery.shape[0], depth, height, width)
                )
            )
            stage_masks.append(stage_mask)
            logits_per_stage.append(logits)
        return (
            semantic_features,
            assignments,
            projected,
            discovery_assignments,
            stage_masks,
            logits_per_stage,
        )

    def forward(self, features, body_mask):
        semantic, assignments, projected, discovery, masks, _ = self._forward_impl(
            features, body_mask
        )
        return semantic, assignments, projected, discovery, masks

    def forward_with_logits(self, features, body_mask):
        return self._forward_impl(features, body_mask)


class IXIStyleAPE(nn.Module):
    """Exterior-masked categorical APE following the original IXI contract."""

    def __init__(
        self,
        k_list: Sequence[int] = DEFAULT_K_LIST,
        labels_per_stage: Sequence[Sequence[int]] = DEFAULT_LABELS_PER_STAGE,
        temperature: float = DEFAULT_TEMPERATURE,
        channels: int = DEFAULT_APE_CHANNELS,
    ) -> None:
        super().__init__()
        validate_stage_configuration(k_list, labels_per_stage)
        self.K = [int(value) for value in k_list]
        self.labels_per_stage = [list(map(int, labels)) for labels in labels_per_stage]
        self.channels = [int(channels)] * 3
        self.register_buffer("current_temp", torch.tensor(float(temperature)))
        self.projectors = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv3d(256, channels, 3, padding=1, bias=False),
                    nn.InstanceNorm3d(channels, affine=True),
                    nn.LeakyReLU(0.2, inplace=True),
                )
                for _ in range(3)
            ]
        )
        self.protos = nn.ParameterList(
            [
                nn.Parameter(torch.randn(prototype_count, channels) * 0.02)
                for prototype_count in self.K
            ]
        )
        self.cs_convs = nn.ModuleList()
        for fine_count, coarse_count in zip(self.K[:-1], self.K[1:]):
            layer = nn.Conv3d(coarse_count, fine_count, kernel_size=1, bias=True)
            nn.init.constant_(layer.weight, 1.0 / coarse_count)
            nn.init.zeros_(layer.bias)
            self.cs_convs.append(layer)

    def forward(
        self, features: Sequence[torch.Tensor], body_mask: torch.Tensor
    ) -> Tuple[
        List[torch.Tensor],
        List[torch.Tensor],
        List[torch.Tensor],
        List[torch.Tensor],
    ]:
        if len(features) != 3:
            raise ValueError(f"APE expects three stages, got {len(features)}")
        projected = [
            projector(feature) for projector, feature in zip(self.projectors, features)
        ]
        reconstructed, assignments, masks = [], [], []
        for stage, (feature, prototypes) in enumerate(zip(projected, self.protos)):
            batch, channels, depth, height, width = feature.shape
            mask = resize_mask(body_mask, (depth, height, width)).to(feature.dtype)
            flattened = feature.flatten(2).transpose(1, 2)
            similarity = torch.einsum(
                "bnc,kc->bnk",
                F.normalize(flattened, dim=-1),
                F.normalize(prototypes, dim=-1),
            ) / self.current_temp.clamp_min(1e-4)
            assignment = F.softmax(similarity, dim=-1)
            assignment = assignment.view(
                batch, depth, height, width, prototypes.shape[0]
            ).permute(0, 4, 1, 2, 3)
            smooth_ratio = 0.2 if stage == 0 else 0.3
            blurred = F.avg_pool3d(assignment, 3, stride=1, padding=1)
            assignment = (1 - smooth_ratio) * assignment + smooth_ratio * blurred
            assignment = assignment * mask
            assignment = assignment / assignment.sum(
                dim=1, keepdim=True
            ).clamp_min(1e-8)
            assignment = assignment * mask
            assignment_flat = assignment.permute(0, 2, 3, 4, 1).reshape(
                batch, -1, prototypes.shape[0]
            )
            reconstruction = torch.einsum(
                "bnk,kc->bnc", assignment_flat, prototypes
            )
            reconstruction = reconstruction.transpose(1, 2).view(
                batch, channels, depth, height, width
            )
            reconstructed.append(reconstruction * mask)
            assignments.append(assignment)
            masks.append(mask)
        return reconstructed, assignments, projected, masks


def normalized_assignment_entropy(
    assignments: torch.Tensor, epsilon: float = 1e-8
) -> torch.Tensor:
    entropy = -(
        assignments * torch.log(assignments.clamp_min(epsilon))
    ).sum(dim=1, keepdim=True)
    return entropy / math.log(max(2, int(assignments.shape[1])))


class EntropyAwareDetailFusion(nn.Module):
    """Original IXI EADF: R + G(R,Z,H) * D(Z,Z-R)."""

    def __init__(self, channels: int = DEFAULT_APE_CHANNELS) -> None:
        super().__init__()
        self.detail_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv3d(channels * 2, channels, 3, padding=1, bias=False),
                    nn.InstanceNorm3d(channels, affine=True),
                    nn.LeakyReLU(0.2, inplace=True),
                )
                for _ in range(3)
            ]
        )
        self.gates = nn.ModuleList(
            [
                nn.Conv3d(channels * 2 + 1, channels, 3, padding=1, bias=True)
                for _ in range(3)
            ]
        )

    def forward(self, reconstructed, assignments, projected):
        result = []
        for detail_block, gate, rec, assignment, proj in zip(
            self.detail_blocks,
            self.gates,
            reconstructed,
            assignments,
            projected,
        ):
            entropy = normalized_assignment_entropy(assignment).to(rec.dtype)
            detail = detail_block(torch.cat([proj, proj - rec], dim=1))
            gate_map = torch.sigmoid(gate(torch.cat([rec, proj, entropy], dim=1)))
            result.append(rec + gate_map * detail)
        return result


class AssignmentAwareDetailFusion(nn.Module):
    """Assignment-confidence-guided fusion used by the semantic APE."""

    def __init__(self, channels: int = DEFAULT_APE_CHANNELS) -> None:
        super().__init__()
        self.gates = nn.ModuleList(
            [
                nn.Conv3d(channels * 2 + 1, channels, 3, padding=1, bias=True)
                for _ in range(3)
            ]
        )

    def forward(self, semantic, assignments, projected):
        result = []
        for gate, semantic_stage, assignment, projected_stage in zip(
            self.gates, semantic, assignments, projected
        ):
            ambiguity = 1 - assignment.max(dim=1, keepdim=True).values
            gate_map = torch.sigmoid(
                gate(
                    torch.cat(
                        [projected_stage, semantic_stage, ambiguity], dim=1
                    )
                )
            )
            result.append(projected_stage + gate_map * semantic_stage)
        return result


def reconstruction_loss(reconstructed, projected, masks, stage_weights):
    total = projected[0].new_zeros([])
    weight_total = 0.0
    for rec, proj, mask, weight in zip(
        reconstructed, projected, masks, stage_weights
    ):
        total = total + float(weight) * masked_mean((rec - proj).square(), mask)
        weight_total += float(weight)
    return total / max(weight_total, 1e-8)


def cross_scale_consistency_loss(
    assignments: Sequence[torch.Tensor],
    mappings: nn.ModuleList,
    masks: Sequence[torch.Tensor],
) -> torch.Tensor:
    total = assignments[0].new_zeros([])
    for stage in range(len(assignments) - 1):
        coarse = F.interpolate(
            assignments[stage + 1],
            size=assignments[stage].shape[2:],
            mode="trilinear",
            align_corners=True,
        )
        mapped = F.softmax(mappings[stage](coarse), dim=1)
        total = total + masked_mean(
            (mapped - assignments[stage]).square(), masks[stage]
        )
    return total / max(1, len(assignments) - 1)


def prototype_diversity_loss(prototypes: Sequence[torch.Tensor]) -> torch.Tensor:
    total = prototypes[0].new_zeros([])
    for prototype in prototypes:
        normalized = F.normalize(prototype, dim=1)
        similarity = normalized @ normalized.t()
        off_diagonal = similarity - torch.eye(
            similarity.shape[0], device=similarity.device
        )
        total = total + off_diagonal.square().mean()
    return total / len(prototypes)


def entropy_regularization(assignments, masks):
    total = assignments[0].new_zeros([])
    for assignment, mask in zip(assignments, masks):
        total = total + masked_mean(
            normalized_assignment_entropy(assignment), mask
        )
    return total / len(assignments)


def spatial_smoothness_loss(assignments, masks):
    total = assignments[0].new_zeros([])
    for assignment, mask in zip(assignments, masks):
        dz = masked_mean(
            (assignment[:, :, 1:] - assignment[:, :, :-1]).abs(),
            mask[:, :, 1:] * mask[:, :, :-1],
        )
        dy = masked_mean(
            (assignment[:, :, :, 1:] - assignment[:, :, :, :-1]).abs(),
            mask[:, :, :, 1:] * mask[:, :, :, :-1],
        )
        dx = masked_mean(
            (assignment[:, :, :, :, 1:] - assignment[:, :, :, :, :-1]).abs(),
            mask[:, :, :, :, 1:] * mask[:, :, :, :, :-1],
        )
        total = total + dz + dy + dx
    return total / len(assignments)


def usage_balance_loss(assignments, masks, minimum_usage: float = 0.01):
    total = assignments[0].new_zeros([])
    for assignment, mask in zip(assignments, masks):
        usage = (assignment * mask).sum(dim=(0, 2, 3, 4)) / mask.sum().clamp_min(1)
        total = total + F.relu(float(minimum_usage) - usage).mean()
    return total / len(assignments)


def fixed_slot_anatomical_loss(
    assignments: Sequence[torch.Tensor],
    segmentation: torch.Tensor,
    masks: Sequence[torch.Tensor],
    labels_per_stage: Sequence[Sequence[int]],
    stage_weights: Sequence[float],
) -> torch.Tensor:
    """Weak supervision for fixed semantic slots within categorical APE.

    Semantic label ``labels_per_stage[l][k]`` supervises prototype slot ``k``.
    Non-selected tissue and unused discovery slots remain unsupervised.
    """

    total = assignments[0].new_zeros([])
    total_weight = 0.0
    epsilon = 1e-8
    for assignment, mask, labels, weight in zip(
        assignments, masks, labels_per_stage, stage_weights
    ):
        resized = F.interpolate(
            segmentation.unsqueeze(1).float(),
            size=assignment.shape[2:],
            mode="nearest",
        ).squeeze(1).long()
        organ_losses = []
        for batch_index in range(assignment.shape[0]):
            for slot, label in enumerate(labels):
                target = (resized[batch_index] == int(label)).float()
                if not target.any():
                    continue
                prediction = assignment[batch_index, slot] * mask[batch_index, 0]
                intersection = (prediction * target).sum()
                dice = (2 * intersection + epsilon) / (
                    prediction.square().sum() + target.square().sum() + epsilon
                )
                positive_nll = -torch.log(
                    prediction[target > 0].clamp_min(epsilon)
                ).mean()
                organ_losses.append((1 - dice) + 0.1 * positive_nll)
        if organ_losses:
            total = total + float(weight) * torch.stack(organ_losses).mean()
            total_weight += float(weight)
    if total_weight <= 0:
        raise ValueError("APE batch contains no supervised labels")
    return total / total_weight


def build_soft_targets(
    segmentation: torch.Tensor,
    body_mask: torch.Tensor,
    size: Sequence[int],
    labels: Sequence[int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    targets = torch.cat(
        [
            F.adaptive_avg_pool3d(
                (segmentation == int(label)).float().unsqueeze(1), tuple(size)
            )
            for label in labels
        ],
        dim=1,
    )
    soft_body = F.adaptive_avg_pool3d(body_mask.float(), tuple(size))
    return targets * soft_body, soft_body


def _zero_bce(
    logits: torch.Tensor,
    candidate: torch.Tensor,
    count: int,
    *,
    hard: bool,
) -> torch.Tensor:
    values = logits[candidate]
    if values.numel() == 0 or int(count) <= 0:
        return logits.new_zeros([])
    take = min(int(count), values.numel())
    if hard:
        selected = torch.topk(values, k=take, largest=True).values
    else:
        indices = torch.randint(values.numel(), (take,), device=values.device)
        selected = values[indices]
    return F.binary_cross_entropy_with_logits(
        selected, torch.zeros_like(selected)
    )


def semantic_assignment_loss(
    logits_per_stage: Sequence[torch.Tensor],
    assignments: Sequence[torch.Tensor],
    masks: Sequence[torch.Tensor],
    segmentation: torch.Tensor,
    body_mask: torch.Tensor,
    labels_per_stage: Sequence[Sequence[int]] = DEFAULT_LABELS_PER_STAGE,
    stage_weights: Sequence[float] = DEFAULT_STAGE_WEIGHTS,
    positive_weight: float = 0.5,
    local_negative_weight: float = 0.25,
    global_hard_negative_weight: float = 0.25,
    global_random_negative_weight: float = 0.10,
    dilation_radius: int = 2,
    local_negative_ratio: float = 3.0,
    global_hard_negative_ratio: float = 3.0,
    global_random_negative_ratio: float = 3.0,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Exact corrected v3.1 organ-equal semantic objective."""

    component_names = (
        "dice",
        "positive",
        "local_negative",
        "global_hard_negative",
        "global_random_negative",
    )
    totals = {name: assignments[0].new_zeros([]) for name in component_names}
    total = assignments[0].new_zeros([])
    weight_total = 0.0
    epsilon = 1e-6
    radius = int(dilation_radius)
    kernel = 2 * radius + 1
    for logits, assignment, stage_mask, labels, stage_weight in zip(
        logits_per_stage, assignments, masks, labels_per_stage, stage_weights
    ):
        label_is_supervised = torch.stack(
            [
                (segmentation == int(label)).flatten(start_dim=1).any(dim=1)
                for label in labels
            ],
            dim=1,
        )
        targets, _ = build_soft_targets(
            segmentation, body_mask, assignment.shape[2:], labels
        )
        per_component = {name: [] for name in component_names}
        organ_losses = []
        for batch_index in range(assignment.shape[0]):
            body = stage_mask[batch_index, 0] > 0.5
            for organ_index in range(assignment.shape[1]):
                if not bool(label_is_supervised[batch_index, organ_index]):
                    continue
                prediction = assignment[batch_index, organ_index]
                organ_logits = logits[batch_index, organ_index]
                target = targets[batch_index, organ_index]
                positive = target > 0
                positive_count = int(positive.sum().item())
                if positive_count <= 0:
                    raise RuntimeError(
                        f"Label {int(labels[organ_index])} is present at full "
                        "resolution but vanished after target pooling"
                    )
                intersection = (prediction * target).sum()
                denominator = prediction.square().sum() + target.square().sum()
                dice_loss = 1 - (2 * intersection + epsilon) / (
                    denominator + epsilon
                )
                positive_loss = F.binary_cross_entropy_with_logits(
                    organ_logits[positive], target[positive]
                )
                roi = F.max_pool3d(
                    positive.float().unsqueeze(0).unsqueeze(0),
                    kernel,
                    1,
                    radius,
                )[0, 0] > 0
                local_candidate = body & roi & ~positive
                far_candidate = body & ~roi & ~positive
                local_negative = _zero_bce(
                    organ_logits,
                    local_candidate,
                    math.ceil(positive_count * float(local_negative_ratio)),
                    hard=True,
                )
                global_hard_negative = _zero_bce(
                    organ_logits,
                    far_candidate,
                    math.ceil(positive_count * float(global_hard_negative_ratio)),
                    hard=True,
                )
                global_random_negative = _zero_bce(
                    organ_logits,
                    far_candidate,
                    math.ceil(
                        positive_count * float(global_random_negative_ratio)
                    ),
                    hard=False,
                )
                values = {
                    "dice": dice_loss,
                    "positive": positive_loss,
                    "local_negative": local_negative,
                    "global_hard_negative": global_hard_negative,
                    "global_random_negative": global_random_negative,
                }
                organ_losses.append(
                    dice_loss
                    + float(positive_weight) * positive_loss
                    + float(local_negative_weight) * local_negative
                    + float(global_hard_negative_weight) * global_hard_negative
                    + float(global_random_negative_weight)
                    * global_random_negative
                )
                for name, value in values.items():
                    per_component[name].append(value)
        if not organ_losses:
            continue
        total = total + float(stage_weight) * torch.stack(organ_losses).mean()
        for name in component_names:
            totals[name] = totals[name] + float(stage_weight) * torch.stack(
                per_component[name]
            ).mean()
        weight_total += float(stage_weight)
    if weight_total <= 0:
        raise ValueError("APE batch contains no supervised semantic labels")
    return total / weight_total, {
        name: value / weight_total for name, value in totals.items()
    }


def model_mode_from_checkpoint(checkpoint: Dict[str, Any]) -> str:
    variant = str(checkpoint.get("ape_variant", ""))
    if variant in (
        SEMANTIC_APE_VARIANT,
        HISTORICAL_SEMANTIC_APE_VARIANT,
    ):
        return SEMANTIC_ASSIGNMENT_MODE
    if variant == IXI_APE_VARIANT:
        return CATEGORICAL_ASSIGNMENT_MODE
    recorded_mode = str(checkpoint.get("model_mode", ""))
    if recorded_mode in (
        SEMANTIC_ASSIGNMENT_MODE,
        HISTORICAL_SEMANTIC_MODE,
    ):
        return SEMANTIC_ASSIGNMENT_MODE
    raise ValueError(f"Unsupported APE checkpoint variant: {variant!r}")


def build_ape_from_checkpoint(checkpoint: Dict[str, Any]) -> nn.Module:
    mode = model_mode_from_checkpoint(checkpoint)
    arguments = dict(
        k_list=checkpoint["k_list"],
        labels_per_stage=checkpoint["labels_per_stage"],
        temperature=float(checkpoint["temperature"]),
        channels=int(checkpoint["ape_channels"]),
    )
    if mode == SEMANTIC_ASSIGNMENT_MODE:
        model = SemanticAssignmentAPE(
            **arguments,
            initial_slot_bias=float(checkpoint.get("initial_slot_bias", -2.0)),
        )
    else:
        model = IXIStyleAPE(**arguments)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model


def load_ape_checkpoint(path: Path, map_location: str = "cpu"):
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    return build_ape_from_checkpoint(checkpoint), checkpoint


def migrate_v31_checkpoint(
    source_checkpoint: Dict[str, Any],
) -> Tuple[IXIStyleAPE, Dict[str, Any]]:
    """Warm-start categorical APE from the current v3.1 checkpoint."""

    if model_mode_from_checkpoint(source_checkpoint) != SEMANTIC_ASSIGNMENT_MODE:
        raise ValueError("Migration requires a semantic-assignment v3.1 checkpoint")
    source = build_ape_from_checkpoint(source_checkpoint)
    categorical = IXIStyleAPE(
        k_list=source_checkpoint["k_list"],
        labels_per_stage=source_checkpoint["labels_per_stage"],
        temperature=float(source_checkpoint["temperature"]),
        channels=int(source_checkpoint["ape_channels"]),
    )
    categorical.projectors.load_state_dict(source.projectors.state_dict(), strict=True)
    with torch.no_grad():
        for target, semantic, discovery in zip(
            categorical.protos, source.semantic_protos, source.discovery_protos
        ):
            combined = torch.cat([semantic, discovery], dim=0)
            if combined.shape != target.shape:
                raise ValueError(
                    f"Prototype migration shape mismatch: {combined.shape} vs {target.shape}"
                )
            target.copy_(combined)
    metadata = {
        "format_version": 40,
        "task": "AbdomenCTCT_IXI_style_categorical_APE_pretraining",
        "ape_variant": IXI_APE_VARIANT,
        "model_mode": CATEGORICAL_ASSIGNMENT_MODE,
        "migration_source_variant": SEMANTIC_APE_VARIANT,
        "migration_source_epoch": source_checkpoint.get("epoch"),
        "k_list": list(categorical.K),
        "labels_per_stage": categorical.labels_per_stage,
        "ape_channels": categorical.channels[0],
        "temperature": float(categorical.current_temp.item()),
        "assignment_mode": "categorical_softmax",
        "ambiguity_mode": "normalized_categorical_entropy",
        "eadf_contract": "R_plus_gate_times_detail_of_Z_and_Z_minus_R",
        "background_handling": {
            "exterior": "CT-derived body mask",
            "interior_non_target": "unused categorical discovery slots",
            "residual_output_channels": [
                k - len(labels)
                for k, labels in zip(categorical.K, categorical.labels_per_stage)
            ],
        },
        "registration_status": "requires post-migration APE training and validation",
    }
    return categorical, metadata
