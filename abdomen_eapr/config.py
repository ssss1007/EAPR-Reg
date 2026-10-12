"""Shared configuration and checkpoint-policy helpers."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

SEMANTIC_ASSIGNMENT_MODE = "semantic_assignment_v31"
CATEGORICAL_ASSIGNMENT_MODE = "categorical_assignment_v1"
MODEL_MODES = (SEMANTIC_ASSIGNMENT_MODE, CATEGORICAL_ASSIGNMENT_MODE)

# Historical values are accepted only when loading checkpoints created before
# the neutral naming refactor. Newly written checkpoints use the names above.
HISTORICAL_SEMANTIC_MODE = "legacy_v31"

DEFAULT_LABELS_PER_STAGE = (
    (4, 5, 10, 11, 12, 13),
    (1, 2, 3, 6, 11),
    (1, 2, 3, 6),
)
DEFAULT_K_LIST = (8, 6, 4)
DEFAULT_APE_CHANNELS = 128
DEFAULT_TEMPERATURE = 0.2
DEFAULT_STAGE_WEIGHTS = (1.5, 1.0, 0.5)
DEFAULT_CONSISTENCY_STAGE_WEIGHTS = (1.0, 0.5, 0.25)


def serializable_arguments(args: argparse.Namespace) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            result[key] = str(value)
        elif isinstance(value, (list, tuple)):
            result[key] = list(value)
        else:
            result[key] = value
    return result


def add_consistency_arguments(parser: argparse.ArgumentParser) -> None:
    """Register an opt-out consistency policy.

    New training runs enable APE consistency without requiring a CLI flag.
    ``None`` is retained until checkpoint resolution so existing checkpoints
    automatically reproduce their recorded behavior.
    """

    parser.add_argument(
        "--ape-consistency",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "APE assignment consistency. New runs default to enabled. "
            "Resume/evaluation defaults to the value recorded in the checkpoint."
        ),
    )
    parser.add_argument("--w-ape-consistency", type=float, default=0.1)
    parser.add_argument(
        "--ape-consistency-stage-weights",
        type=float,
        nargs=3,
        default=list(DEFAULT_CONSISTENCY_STAGE_WEIGHTS),
        metavar=("FINE", "MIDDLE", "COARSE"),
    )
    parser.add_argument("--ape-consistency-warmup-epochs", type=int, default=10)


def resolve_ape_consistency(
    requested: Optional[bool],
    checkpoint: Optional[Dict[str, Any]],
    *,
    resume_training: bool,
) -> bool:
    """Resolve default-on behavior without breaking legacy reproducibility."""

    if checkpoint is None:
        return True if requested is None else bool(requested)
    recorded = bool(checkpoint.get("use_ape_consistency", False))
    if requested is None:
        return recorded
    requested_bool = bool(requested)
    if resume_training and requested_bool != recorded:
        raise ValueError(
            "APE-consistency mismatch while resuming: "
            f"checkpoint={recorded}, requested={requested_bool}. "
            "Start a new run to change the training objective."
        )
    return recorded if not resume_training else requested_bool


def validate_stage_configuration(
    k_list: Sequence[int], labels_per_stage: Sequence[Sequence[int]]
) -> None:
    if len(k_list) != 3 or len(labels_per_stage) != 3:
        raise ValueError("APE requires exactly three feature stages")
    for stage, (prototype_count, labels) in enumerate(
        zip(k_list, labels_per_stage)
    ):
        if int(prototype_count) <= 0:
            raise ValueError(f"Stage {stage}: prototype count must be positive")
        if len(labels) > int(prototype_count):
            raise ValueError(
                f"Stage {stage}: {len(labels)} semantic labels exceed "
                f"K={prototype_count}"
            )
        if len(set(int(label) for label in labels)) != len(labels):
            raise ValueError(f"Stage {stage}: labels contain duplicates")
        if any(int(label) < 1 or int(label) > 13 for label in labels):
            raise ValueError(f"Stage {stage}: labels must be in [1,13]")
