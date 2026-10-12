#!/usr/bin/env python3
"""Train the fixed-slot semantic-assignment APE v3.1."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from abdomen_eapr.ape import (
        SEMANTIC_APE_VARIANT,
        SemanticAssignmentAPE,
        build_soft_targets,
        model_mode_from_checkpoint,
        semantic_assignment_loss,
    )
    from abdomen_eapr.config import (
        DEFAULT_APE_CHANNELS,
        DEFAULT_K_LIST,
        DEFAULT_LABELS_PER_STAGE,
        DEFAULT_STAGE_WEIGHTS,
        DEFAULT_TEMPERATURE,
        SEMANTIC_ASSIGNMENT_MODE,
        serializable_arguments,
    )
    from abdomen_eapr.data import AbdomenCaseDataset, AbdomenPairDataset
    from abdomen_eapr.medsam import MedSAM2Encoder3D
else:
    from .ape import (
        SEMANTIC_APE_VARIANT,
        SemanticAssignmentAPE,
        build_soft_targets,
        model_mode_from_checkpoint,
        semantic_assignment_loss,
    )
    from .config import (
        DEFAULT_APE_CHANNELS,
        DEFAULT_K_LIST,
        DEFAULT_LABELS_PER_STAGE,
        DEFAULT_STAGE_WEIGHTS,
        DEFAULT_TEMPERATURE,
        SEMANTIC_ASSIGNMENT_MODE,
        serializable_arguments,
    )
    from .data import AbdomenCaseDataset, AbdomenPairDataset
    from .medsam import MedSAM2Encoder3D

PACKAGE_DIR = Path(__file__).resolve().parent
WORKSPACE = PACKAGE_DIR.parent


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    data_root = (
        WORKSPACE
        / "data"
        / "AbdomenCTCT_processed_sacb_center_prospective_split_v3"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", type=Path, default=data_root / "Train")
    parser.add_argument("--val-dir", type=Path, default=data_root / "Val")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=WORKSPACE / "abdomen_eapr_runs" / "semantic_ape_v31",
    )
    parser.add_argument("--sam2-root", type=Path, required=True)
    parser.add_argument(
        "--sam2-config",
        type=Path,
        default=Path("configs/sam2.1/sam2.1_hiera_t.yaml"),
    )
    parser.add_argument(
        "--sam2-checkpoint",
        type=Path,
        required=True,
    )
    parser.add_argument("--ape-channels", type=int, default=DEFAULT_APE_CHANNELS)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--initial-slot-bias", type=float, default=-2.0)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--w-positive", type=float, default=0.5)
    parser.add_argument("--w-local-negative", type=float, default=0.25)
    parser.add_argument("--w-global-hard-negative", type=float, default=0.25)
    parser.add_argument("--w-global-random-negative", type=float, default=0.10)
    parser.add_argument("--hard-negative-radius", type=int, default=2)
    parser.add_argument("--local-negative-ratio", type=float, default=3.0)
    parser.add_argument("--global-hard-negative-ratio", type=float, default=3.0)
    parser.add_argument("--global-random-negative-ratio", type=float, default=3.0)
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--save-every", type=int, default=20)
    parser.add_argument("--vis-every", type=int, default=20)
    parser.add_argument("--early-stop-patience", type=int, default=10)
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--skip-centroid-init", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)
    for name in (
        "w_positive",
        "w_local_negative",
        "w_global_hard_negative",
        "w_global_random_negative",
    ):
        if getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} must be non-negative")
    return args


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def dataset_metadata(train_dir: Path) -> Dict[str, Any]:
    manifests = train_dir.resolve().parent / "manifests"
    result: Dict[str, Any] = {"dataset_root": str(train_dir.resolve().parent)}
    for filename, key in (
        ("dataset_fingerprint.sha256", "dataset_fingerprint"),
        ("preprocessing.json", "preprocessing"),
        ("split.json", "split"),
    ):
        path = manifests / filename
        if path.exists():
            result[key] = (
                json.loads(path.read_text(encoding="utf-8"))
                if path.suffix == ".json"
                else path.read_text(encoding="ascii").strip()
            )
    return result


@torch.no_grad()
def initialize_semantic_prototypes(
    encoder: MedSAM2Encoder3D,
    ape: SemanticAssignmentAPE,
    dataset: AbdomenCaseDataset,
    device: torch.device,
) -> None:
    per_stage = [[[] for _ in labels] for labels in ape.labels_per_stage]
    encoder.eval()
    ape.eval()
    for index in tqdm(range(len(dataset)), desc="v3.1 semantic centroid init"):
        image, segmentation, body, _ = dataset.load_case(index)
        image = image.unsqueeze(0).to(device)
        segmentation = segmentation.unsqueeze(0).to(device)
        body = body.unsqueeze(0).to(device)
        features = encoder(image)
        projected = [
            projector(feature) for projector, feature in zip(ape.projectors, features)
        ]
        for stage, (feature, labels) in enumerate(
            zip(projected, ape.labels_per_stage)
        ):
            targets, _ = build_soft_targets(
                segmentation, body, feature.shape[2:], labels
            )
            for slot in range(len(labels)):
                target = targets[:, slot : slot + 1]
                mass = target.sum()
                if mass <= 0:
                    continue
                center = (feature * target).sum(dim=(0, 2, 3, 4)) / mass
                per_stage[stage][slot].append(F.normalize(center, dim=0))
    for stage, organ_centers in enumerate(per_stage):
        centers = []
        for slot, case_centers in enumerate(organ_centers):
            if not case_centers:
                raise ValueError(
                    f"No train voxels for stage {stage}, "
                    f"label {ape.labels_per_stage[stage][slot]}"
                )
            centers.append(
                F.normalize(torch.stack(case_centers).mean(dim=0), dim=0)
            )
        ape.semantic_protos[stage].copy_(torch.stack(centers))


@torch.no_grad()
def fixed_channel_evaluation(
    encoder: MedSAM2Encoder3D,
    ape: SemanticAssignmentAPE,
    dataset: AbdomenCaseDataset,
    device: torch.device,
) -> Dict[str, Any]:
    values = [{int(label): [] for label in labels} for labels in ape.labels_per_stage]
    per_case: Dict[str, Any] = {}
    encoder.eval()
    ape.eval()
    for index in range(len(dataset)):
        image, segmentation, body, case_id = dataset.load_case(index)
        image = image.unsqueeze(0).to(device)
        segmentation = segmentation.unsqueeze(0).to(device)
        body = body.unsqueeze(0).to(device)
        _, assignments, _, _, masks = ape(encoder(image), body)
        case_values = []
        for stage, (assignment, mask, labels) in enumerate(
            zip(assignments, masks, ape.labels_per_stage)
        ):
            targets, _ = build_soft_targets(
                segmentation, body, assignment.shape[2:], labels
            )
            for slot, label in enumerate(labels):
                target = targets[:, slot]
                if target.sum() <= 0:
                    continue
                prediction = assignment[:, slot] * mask[:, 0]
                intersection = (prediction * target).sum()
                dice = (2 * intersection + 1e-6) / (
                    prediction.square().sum() + target.square().sum() + 1e-6
                )
                value = float(dice)
                values[stage][int(label)].append(value)
                if stage == 0:
                    case_values.append(value)
        per_case[case_id] = (
            float(np.mean(case_values)) if case_values else None
        )
    per_stage_per_organ = []
    per_stage = []
    for stage in values:
        organ_means = {
            str(label): float(np.mean(scores)) if scores else None
            for label, scores in stage.items()
        }
        valid = [value for value in organ_means.values() if value is not None]
        per_stage_per_organ.append(organ_means)
        per_stage.append(float(np.mean(valid)) if valid else None)
    return {
        "metric": "fixed_slot_soft_occupancy_dice",
        "per_stage": per_stage,
        "per_stage_per_organ": per_stage_per_organ,
        "fine_target_fixed_mean": per_stage[0],
        "mean": float(np.mean([value for value in per_stage if value is not None])),
        "per_case": per_case,
    }


def checkpoint_payload(
    ape,
    optimizer,
    scheduler,
    scaler,
    epoch,
    metrics,
    best_fine,
    args,
) -> Dict[str, Any]:
    return {
        "format_version": 31,
        "task": "AbdomenCTCT_semantic_assignment_APE_pretraining",
        "ape_variant": SEMANTIC_APE_VARIANT,
        "model_mode": SEMANTIC_ASSIGNMENT_MODE,
        "epoch": int(epoch),
        "model_state_dict": ape.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "amp_scaler_state_dict": scaler.state_dict(),
        "metrics": metrics,
        "best_fine_target_fixed_mean": float(best_fine),
        "k_list": list(ape.K),
        "K_list": list(ape.K),
        "ape_channels": int(ape.channels[0]),
        "temperature": float(ape.current_temp),
        "initial_slot_bias": float(args.initial_slot_bias),
        "labels_per_stage": ape.labels_per_stage,
        "semantic_slot_counts": ape.semantic_counts,
        "discovery_slot_counts": ape.discovery_counts,
        "assignment_mode": "independent_semantic_assignment",
        "slot_mode": "fixed_anatomical_prototype_assignment",
        "semantic_logit_mode": (
            "cosine_similarity_with_learnable_slot_calibration"
        ),
        "negative_sampling": {
            "local_hard": True,
            "global_hard": True,
            "global_random": True,
            "count_basis": "nonzero_soft_target_cells",
        },
        "missing_label_policy": (
            "skip_all_slot_losses_when_label_absent_at_full_resolution_v1"
        ),
        "background_handling": {
            "exterior": "CT-derived body mask",
            "interior_non_target": "explicit local/global sampled negatives",
            "residual_output_channels": 0,
            "discovery_slots_active": False,
        },
        "registration_status": "offline semantic audit required before integration",
        "dataset_metadata": dataset_metadata(args.train_dir),
        "seed": int(args.seed),
        "training_arguments": serializable_arguments(args),
    }


def save_checkpoint(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device(args.device)
    train_cases = AbdomenCaseDataset(args.train_dir)
    train_pairs = AbdomenPairDataset(args.train_dir)
    val_cases = AbdomenCaseDataset(args.val_dir)
    encoder = MedSAM2Encoder3D(
        args.sam2_root,
        args.sam2_config,
        args.sam2_checkpoint,
        str(device),
        args.chunk_size,
    ).to(device)
    ape = SemanticAssignmentAPE(
        DEFAULT_K_LIST,
        DEFAULT_LABELS_PER_STAGE,
        args.temperature,
        args.ape_channels,
        args.initial_slot_bias,
    ).to(device)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in ape.parameters() if parameter.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    amp_enabled = bool(args.amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    start_epoch = 1
    best_fine = -float("inf")
    if args.resume is not None:
        checkpoint = torch.load(
            args.resume, map_location=device, weights_only=False
        )
        if model_mode_from_checkpoint(checkpoint) != SEMANTIC_ASSIGNMENT_MODE:
            raise ValueError("Resume checkpoint is not APE v3.1")
        ape.load_state_dict(checkpoint["model_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        scaler.load_state_dict(checkpoint.get("amp_scaler_state_dict", {}))
        start_epoch = int(checkpoint["epoch"]) + 1
        best_fine = float(
            checkpoint.get("best_fine_target_fixed_mean", best_fine)
        )
    elif not args.skip_centroid_init:
        initialize_semantic_prototypes(encoder, ape, train_cases, device)
    loader = DataLoader(
        train_pairs,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(args.seed),
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_config.json").write_text(
        json.dumps(serializable_arguments(args), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    no_improvement = 0
    for epoch in range(start_epoch, args.epochs + 1):
        ape.train()
        totals = {
            name: 0.0
            for name in (
                "loss",
                "dice",
                "positive",
                "local_negative",
                "global_hard_negative",
                "global_random_negative",
            )
        }
        progress = tqdm(loader, desc=f"Clean APE v3.1 {epoch}/{args.epochs}")
        for fixed, moving, fixed_seg, moving_seg, fixed_mask, moving_mask in progress:
            optimizer.zero_grad(set_to_none=True)
            batch_loss = torch.zeros([], device=device)
            batch_parts = {
                name: torch.zeros([], device=device)
                for name in totals
                if name != "loss"
            }
            for image, segmentation, body in (
                (fixed, fixed_seg, fixed_mask),
                (moving, moving_seg, moving_mask),
            ):
                image = image.to(device, non_blocking=True).float()
                segmentation = segmentation.to(device, non_blocking=True).long()
                body = body.to(device, non_blocking=True).float()
                with torch.no_grad():
                    features = encoder(image)
                with torch.cuda.amp.autocast(enabled=amp_enabled):
                    _, assignments, _, _, masks, logits = ape.forward_with_logits(
                        features, body
                    )
                    loss, parts = semantic_assignment_loss(
                        logits,
                        assignments,
                        masks,
                        segmentation,
                        body,
                        positive_weight=args.w_positive,
                        local_negative_weight=args.w_local_negative,
                        global_hard_negative_weight=args.w_global_hard_negative,
                        global_random_negative_weight=args.w_global_random_negative,
                        dilation_radius=args.hard_negative_radius,
                        local_negative_ratio=args.local_negative_ratio,
                        global_hard_negative_ratio=args.global_hard_negative_ratio,
                        global_random_negative_ratio=args.global_random_negative_ratio,
                    )
                batch_loss = batch_loss + 0.5 * loss
                for name in batch_parts:
                    batch_parts[name] = batch_parts[name] + 0.5 * parts[name]
            scaler.scale(batch_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            totals["loss"] += float(batch_loss.detach())
            for name, value in batch_parts.items():
                totals[name] += float(value.detach())
            progress.set_postfix(loss=f"{float(batch_loss):.4f}")
        scheduler.step()
        averages = {name: value / len(loader) for name, value in totals.items()}
        metrics: Dict[str, Any] = {"train": averages}
        improved = False
        if epoch == 1 or epoch % max(1, args.eval_every) == 0:
            evaluation = fixed_channel_evaluation(
                encoder, ape, val_cases, device
            )
            metrics["validation"] = evaluation
            fine = float(evaluation["fine_target_fixed_mean"])
            if fine > best_fine:
                best_fine = fine
                improved = True
                no_improvement = 0
                save_checkpoint(
                    output_dir / "checkpoints" / "ape_v31_best_fine_fixed.pth",
                    checkpoint_payload(
                        ape,
                        optimizer,
                        scheduler,
                        scaler,
                        epoch,
                        metrics,
                        best_fine,
                        args,
                    ),
                )
            else:
                no_improvement += 1
        if epoch % max(1, args.save_every) == 0:
            save_checkpoint(
                output_dir / "checkpoints" / f"ape_v31_epoch{epoch:04d}.pth",
                checkpoint_payload(
                    ape,
                    optimizer,
                    scheduler,
                    scaler,
                    epoch,
                    metrics,
                    best_fine,
                    args,
                ),
            )
        if args.vis_every > 0 and epoch % args.vis_every == 0:
            (output_dir / "visualization_metrics").mkdir(
                parents=True, exist_ok=True
            )
            (output_dir / "visualization_metrics" / f"epoch_{epoch:04d}.json").write_text(
                json.dumps(
                    metrics.get("validation", {}),
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "best_fine": best_fine,
                    "improved": improved,
                    **metrics,
                },
                ensure_ascii=False,
            )
        )
        if (
            args.early_stop_patience > 0
            and no_improvement >= args.early_stop_patience
        ):
            print(
                f"Early stopping after {no_improvement} validation checks "
                "without improvement"
            )
            break


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
