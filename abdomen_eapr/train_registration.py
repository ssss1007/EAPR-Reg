#!/usr/bin/env python3
"""Train or evaluate the clean AbdomenCTCT EAPR-Reg registration model."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from abdomen_eapr.config import (
        SEMANTIC_ASSIGNMENT_MODE,
        add_consistency_arguments,
        resolve_ape_consistency,
        serializable_arguments,
    )
    from abdomen_eapr.data import (
        ORGAN_LABELS,
        AbdomenRegistrationPairs,
        body_mask_dice_loss,
        binary_surface_distance_metrics,
        masked_local_ncc_loss,
        organ_dice_full_grid,
    )
    from abdomen_eapr.registration import (
        FIVE_SCALE_ARCHITECTURE,
        REGISTRATION_ARCHITECTURE,
        THREE_LEVEL_ARCHITECTURE,
        AnatomyGuidance,
        CleanRegistrationModel,
        SpatialTransformer,
        ape_assignment_consistency_loss,
        flow_gradient_loss,
        infer_registration_architecture,
        load_registration_state,
        negative_jacobian_percentage,
        registration_state_dict,
    )
    from abdomen_eapr.ape import model_mode_from_checkpoint
else:
    from .config import (
        SEMANTIC_ASSIGNMENT_MODE,
        add_consistency_arguments,
        resolve_ape_consistency,
        serializable_arguments,
    )
    from .data import (
        ORGAN_LABELS,
        AbdomenRegistrationPairs,
        body_mask_dice_loss,
        binary_surface_distance_metrics,
        masked_local_ncc_loss,
        organ_dice_full_grid,
    )
    from .registration import (
        FIVE_SCALE_ARCHITECTURE,
        REGISTRATION_ARCHITECTURE,
        THREE_LEVEL_ARCHITECTURE,
        AnatomyGuidance,
        CleanRegistrationModel,
        SpatialTransformer,
        ape_assignment_consistency_loss,
        flow_gradient_loss,
        infer_registration_architecture,
        load_registration_state,
        negative_jacobian_percentage,
        registration_state_dict,
    )
    from .ape import model_mode_from_checkpoint

PACKAGE_DIR = Path(__file__).resolve().parent
WORKSPACE = PACKAGE_DIR.parent


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    dataset_root = (
        WORKSPACE
        / "data"
        / "AbdomenCTCT_processed_sacb_center_prospective_split_v3"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", type=Path, default=dataset_root / "Train")
    parser.add_argument("--val-dir", type=Path, default=dataset_root / "Val")
    parser.add_argument(
        "--train-pairs",
        type=Path,
        default=dataset_root / "manifests" / "train_pairs.json",
    )
    parser.add_argument(
        "--val-pairs",
        type=Path,
        default=dataset_root / "manifests" / "val_pairs.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=WORKSPACE / "abdomen_eapr_runs" / "clean_registration_v1",
    )
    parser.add_argument("--ape-checkpoint", type=Path, required=True)
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
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--first-channel", type=int, default=8)
    parser.add_argument(
        "--registration-architecture",
        choices=(FIVE_SCALE_ARCHITECTURE, THREE_LEVEL_ARCHITECTURE),
        default=FIVE_SCALE_ARCHITECTURE,
        help=(
            "New runs default to the five-scale architecture used by the "
            "provided training command. Resume/evaluation is inferred from checkpoint."
        ),
    )
    parser.add_argument(
        "--fusion",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable the fusion associated with the APE checkpoint contract.",
    )
    parser.add_argument(
        "--strict-coarse-composition",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    add_consistency_arguments(parser)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--ncc-window", type=int, default=9)
    parser.add_argument(
        "--voxel-spacing-mm",
        type=float,
        nargs=3,
        default=[2.0, 2.0, 2.0],
        metavar=("Z", "Y", "X"),
        help="Physical DHW voxel spacing used by HD95 and ASSD evaluation.",
    )
    parser.add_argument("--w-ncc", type=float, default=1.0)
    parser.add_argument("--w-grad", type=float, default=0.5)
    parser.add_argument("--w-body", type=float, default=0.05)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument(
        "--warm-start-registration",
        type=Path,
        default=None,
        help=(
            "Load compatible deformation/refinement weights from an earlier "
            "registration checkpoint; incompatible fusion weights are excluded."
        ),
    )
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument(
        "--figure-pair",
        nargs=2,
        metavar=("MOVING_ID", "FIXED_ID"),
        default=None,
        help="pair IDs to export while evaluating for the qualitative figure",
    )
    parser.add_argument(
        "--export-figure-npz",
        type=Path,
        default=None,
        help="save --figure-pair in the common qualitative-figure NPZ format",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)
    if args.evaluate_only and args.resume is None:
        parser.error("--evaluate-only requires --resume")
    if (args.figure_pair is None) != (args.export_figure_npz is None):
        parser.error("--figure-pair and --export-figure-npz must be used together")
    if args.figure_pair is not None and not args.evaluate_only:
        parser.error("--figure-pair is currently supported with --evaluate-only")
    if args.resume is not None and args.warm_start_registration is not None:
        parser.error("--resume and --warm-start-registration are mutually exclusive")
    if args.w_ape_consistency < 0:
        parser.error("--w-ape-consistency must be non-negative")
    if any(value < 0 for value in args.ape_consistency_stage_weights):
        parser.error("APE consistency stage weights must be non-negative")
    if sum(args.ape_consistency_stage_weights) <= 0:
        parser.error("At least one APE consistency stage weight must be positive")
    if any(value <= 0 for value in args.voxel_spacing_mm):
        parser.error("--voxel-spacing-mm values must be positive")
    return args


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@torch.no_grad()
def evaluate(
    model: CleanRegistrationModel,
    dataset: AbdomenRegistrationPairs,
    device: torch.device,
    *,
    include_surface_metrics: bool = False,
    voxel_spacing_mm: Sequence[float] = (2.0, 2.0, 2.0),
    figure_pair: Optional[Sequence[str]] = None,
    export_figure_npz: Optional[Path] = None,
) -> Dict[str, Any]:
    model.eval()
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    full_size = dataset[0]["fixed"].shape[1:]
    nearest = SpatialTransformer(full_size, mode="nearest").to(device)
    per_organ = {label: [] for label in ORGAN_LABELS}
    per_organ_hd95 = {label: [] for label in ORGAN_LABELS}
    per_organ_assd = {label: [] for label in ORGAN_LABELS}
    cases = []
    jacobians = []
    figure_exported = False
    for batch in tqdm(loader, desc="Clean abdomen validation", leave=False):
        moving = batch["moving"].to(device).float()
        fixed = batch["fixed"].to(device).float()
        moving_mask = batch["moving_mask"].to(device).float()
        fixed_mask = batch["fixed_mask"].to(device).float()
        warped_image, flow = model(moving, fixed, moving_mask, fixed_mask)
        jacobian = negative_jacobian_percentage(flow.cpu())
        jacobians.append(jacobian)
        moving_label = batch["moving_label"].unsqueeze(1).to(device).float()
        warped_label = (
            nearest(moving_label, flow).round().cpu().numpy().squeeze().astype(np.int16)
        )
        fixed_label = batch["fixed_label"].numpy().squeeze().astype(np.int16)
        moving_id = str(batch["moving_id"][0])
        fixed_id = str(batch["fixed_id"][0])
        if (
            figure_pair is not None
            and moving_id == str(figure_pair[0])
            and fixed_id == str(figure_pair[1])
        ):
            if figure_exported:
                raise RuntimeError(
                    f"Duplicate requested figure pair {moving_id}->{fixed_id}"
                )
            export_figure_npz.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                export_figure_npz,
                moving_image=moving.detach().cpu().numpy().squeeze().astype(np.float32),
                fixed_image=fixed.detach().cpu().numpy().squeeze().astype(np.float32),
                warped_image=(
                    warped_image.detach().cpu().numpy().squeeze().astype(np.float32)
                ),
                moving_seg=(
                    batch["moving_label"].numpy().squeeze().astype(np.int16)
                ),
                fixed_seg=fixed_label,
                warped_seg=warped_label,
                flow=flow.detach().cpu().numpy().squeeze().astype(np.float32),
            )
            figure_exported = True
            print(
                f"Saved qualitative figure data: {export_figure_npz}",
                flush=True,
            )
        scores = organ_dice_full_grid(warped_label, fixed_label)
        for label, value in scores.items():
            per_organ[label].append(value)
        case: Dict[str, Any] = {
            "moving_id": moving_id,
            "fixed_id": fixed_id,
            "mean_dice": float(np.mean(list(scores.values()))),
            "negative_jacobian_percent": jacobian,
            "valid_organ_count": len(scores),
            "per_organ_dice": {
                str(label): float(value) for label, value in scores.items()
            },
        }
        if include_surface_metrics:
            organ_surface: Dict[str, Dict[str, Any]] = {}
            pair_hd95, pair_assd = [], []
            penalized_count = 0
            for label in ORGAN_LABELS:
                surface = binary_surface_distance_metrics(
                    warped_label == label,
                    fixed_label == label,
                    voxel_spacing_mm,
                )
                organ_surface[str(label)] = surface
                if surface["hd95_mm"] is not None:
                    hd95 = float(surface["hd95_mm"])
                    assd = float(surface["assd_mm"])
                    per_organ_hd95[label].append(hd95)
                    per_organ_assd[label].append(assd)
                    pair_hd95.append(hd95)
                    pair_assd.append(assd)
                penalized_count += int(bool(surface["penalized"]))
            case.update(
                {
                    "hd95_mm": float(np.mean(pair_hd95)),
                    "assd_mm": float(np.mean(pair_assd)),
                    "surface_valid_organ_count": len(pair_hd95),
                    "surface_penalized_organ_count": penalized_count,
                    "per_organ_surface": organ_surface,
                }
            )
        cases.append(case)
    if figure_pair is not None and not figure_exported:
        raise RuntimeError(
            f"Requested figure pair {figure_pair[0]}->{figure_pair[1]} "
            "was not found in the evaluation manifest"
        )
    organ_means = {
        str(label): float(np.mean(values)) if values else None
        for label, values in per_organ.items()
    }
    valid = [value for value in organ_means.values() if value is not None]
    pair_dice = [float(case["mean_dice"]) for case in cases]
    result: Dict[str, Any] = {
        "equal_weighted_organ_mean_dice": float(np.mean(valid)) if valid else None,
        "dice_mean": float(np.mean(pair_dice)),
        "dice_std": float(np.std(pair_dice)),
        "per_organ_dice": organ_means,
        "negative_jacobian_percent_mean": float(np.mean(jacobians)),
        "negative_jacobian_percent_std": float(np.std(jacobians)),
        "valid_pair_count": len(cases),
        "cases": cases,
        "voxel_spacing_mm_dhw": [float(value) for value in voxel_spacing_mm],
        "rules": (
            "full-grid labels 1-13; label 0 excluded; population std (ddof=0); "
            "pair-level metrics average valid ROIs; empty-empty is N/A"
        ),
        "aggregation": (
            "SACB-style mean/std across pair-level 13-ROI means. "
            "equal_weighted_organ_mean_dice is retained for checkpoint selection."
        ),
    }
    if include_surface_metrics:
        pair_hd95 = [float(case["hd95_mm"]) for case in cases]
        pair_assd = [float(case["assd_mm"]) for case in cases]
        per_organ_metrics = {}
        for label in ORGAN_LABELS:
            dice_values = per_organ[label]
            hd95_values = per_organ_hd95[label]
            assd_values = per_organ_assd[label]
            per_organ_metrics[str(label)] = {
                "dice_mean": float(np.mean(dice_values)) if dice_values else None,
                "dice_std": float(np.std(dice_values)) if dice_values else None,
                "hd95_mm_mean": (
                    float(np.mean(hd95_values)) if hd95_values else None
                ),
                "hd95_mm_std": (
                    float(np.std(hd95_values)) if hd95_values else None
                ),
                "assd_mm_mean": (
                    float(np.mean(assd_values)) if assd_values else None
                ),
                "assd_mm_std": (
                    float(np.std(assd_values)) if assd_values else None
                ),
                "valid_pair_count": len(dice_values),
            }
        result.update(
            {
                "hd95_mm_mean": float(np.mean(pair_hd95)),
                "hd95_mm_std": float(np.std(pair_hd95)),
                "assd_mm_mean": float(np.mean(pair_assd)),
                "assd_mm_std": float(np.std(pair_assd)),
                "per_organ_metrics": per_organ_metrics,
                "paper_table_metrics": {
                    "dice": {
                        "mean": float(np.mean(pair_dice)),
                        "std": float(np.std(pair_dice)),
                    },
                    "hd95_mm": {
                        "mean": float(np.mean(pair_hd95)),
                        "std": float(np.std(pair_hd95)),
                    },
                    "assd_mm": {
                        "mean": float(np.mean(pair_assd)),
                        "std": float(np.std(pair_assd)),
                    },
                    "negative_jacobian_percent": {
                        "mean": float(np.mean(jacobians)),
                        "std": float(np.std(jacobians)),
                    },
                },
                "surface_distance_rules": (
                    "6-connected binary surfaces; symmetric HD95; ASSD is the "
                    "mean of both directional surface-distance means; one-sided "
                    "empty masks receive the physical full-grid diagonal penalty"
                ),
            }
        )
    return result


def write_evaluation_csv(output_dir: Path, metrics: Dict[str, Any]) -> None:
    """Write pair, pair-organ, and organ-summary tables for analysis."""

    output_dir.mkdir(parents=True, exist_ok=True)
    pair_fields = [
        "moving_id",
        "fixed_id",
        "mean_dice",
        "hd95_mm",
        "assd_mm",
        "negative_jacobian_percent",
        "valid_organ_count",
        "surface_valid_organ_count",
        "surface_penalized_organ_count",
    ]
    with (output_dir / "evaluation_pairs.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=pair_fields)
        writer.writeheader()
        for case in metrics["cases"]:
            writer.writerow({key: case.get(key) for key in pair_fields})

    with (output_dir / "evaluation_pair_organs.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        fields = [
            "moving_id",
            "fixed_id",
            "organ_label",
            "dice",
            "hd95_mm",
            "assd_mm",
            "empty_case",
            "penalized",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for case in metrics["cases"]:
            for label in ORGAN_LABELS:
                surface = case.get("per_organ_surface", {}).get(str(label), {})
                writer.writerow(
                    {
                        "moving_id": case["moving_id"],
                        "fixed_id": case["fixed_id"],
                        "organ_label": label,
                        "dice": case["per_organ_dice"].get(str(label)),
                        "hd95_mm": surface.get("hd95_mm"),
                        "assd_mm": surface.get("assd_mm"),
                        "empty_case": surface.get("empty_case"),
                        "penalized": surface.get("penalized"),
                    }
                )

    if "per_organ_metrics" in metrics:
        with (output_dir / "evaluation_organs.csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            fields = [
                "organ_label",
                "dice_mean",
                "dice_std",
                "hd95_mm_mean",
                "hd95_mm_std",
                "assd_mm_mean",
                "assd_mm_std",
                "valid_pair_count",
            ]
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for label in ORGAN_LABELS:
                writer.writerow(
                    {
                        "organ_label": label,
                        **metrics["per_organ_metrics"][str(label)],
                    }
                )


def checkpoint_payload(
    model: CleanRegistrationModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_dice: float,
    args: argparse.Namespace,
    use_ape_consistency: bool,
) -> Dict[str, Any]:
    arguments = serializable_arguments(args)
    arguments["effective_ape_consistency"] = bool(use_ape_consistency)
    return {
        "format_version": 50,
        "task": "AbdomenCTCT_clean_EAPR_registration",
        "registration_architecture": model.registration_architecture,
        "epoch": int(epoch),
        "best_dice": float(best_dice),
        "model_state_dict": registration_state_dict(model),
        "optimizer_state_dict": optimizer.state_dict(),
        "ape_checkpoint": str(args.ape_checkpoint),
        "ape_variant": model.ape_variant,
        "model_mode": model.model_mode,
        "ape_registration_status": model.ape_registration_status,
        "use_eadf": bool(model.use_fusion),
        "fusion_contract": (
            "assignment_confidence_guided_detail_fusion"
            if model.model_mode == SEMANTIC_ASSIGNMENT_MODE
            else "normalized_entropy_aware_detail_fusion"
        ),
        "use_ape_consistency": bool(use_ape_consistency),
        "strict_coarse_composition": bool(model.strict_coarse_composition),
        "training_arguments": arguments,
    }


def save_checkpoint(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device(args.device)
    resume_checkpoint = (
        torch.load(args.resume, map_location="cpu", weights_only=False)
        if args.resume is not None
        else None
    )
    use_ape_consistency = resolve_ape_consistency(
        args.ape_consistency,
        resume_checkpoint,
        resume_training=not args.evaluate_only,
    )
    if resume_checkpoint is not None:
        args.fusion = bool(resume_checkpoint.get("use_eadf", True))
        args.strict_coarse_composition = bool(
            resume_checkpoint.get("strict_coarse_composition", False)
        )
        args.registration_architecture = infer_registration_architecture(
            resume_checkpoint
        )
    train_dataset = AbdomenRegistrationPairs(args.train_dir, args.train_pairs)
    val_dataset = AbdomenRegistrationPairs(args.val_dir, args.val_pairs)
    full_size = tuple(int(value) for value in train_dataset[0]["moving"].shape[1:])
    guidance = AnatomyGuidance(
        args.ape_checkpoint,
        args.sam2_root,
        args.sam2_config,
        args.sam2_checkpoint,
        device=str(device),
        chunk_size=args.chunk_size,
    ).to(device)
    model = CleanRegistrationModel(
        full_size,
        args.first_channel,
        guidance,
        use_fusion=args.fusion,
        strict_coarse_composition=args.strict_coarse_composition,
        registration_architecture=args.registration_architecture,
    ).to(device)
    optimizer = torch.optim.Adam(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.lr,
    )
    start_epoch = 1
    best_dice = -float("inf")
    if resume_checkpoint is not None:
        if model_mode_from_checkpoint(resume_checkpoint) != model.model_mode:
            raise ValueError(
                "Registration checkpoint and supplied APE checkpoint variants differ"
            )
        load_registration_state(model, resume_checkpoint["model_state_dict"])
        if not args.evaluate_only:
            optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
        start_epoch = int(resume_checkpoint["epoch"]) + 1
        best_dice = float(resume_checkpoint.get("best_dice", best_dice))
    elif args.warm_start_registration is not None:
        warm = torch.load(
            args.warm_start_registration, map_location="cpu", weights_only=False
        )
        report = load_registration_state(
            model, warm["model_state_dict"], warm_start=True
        )
        print(
            f"Warm start loaded {len(report['loaded'])} tensors; "
            f"new/missing tensors={len(report['missing'])}"
        )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_config = {
        **serializable_arguments(args),
        "effective_ape_consistency": use_ape_consistency,
        "ape_variant": model.ape_variant,
        "model_mode": model.model_mode,
        "registration_architecture": model.registration_architecture,
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"Model mode={model.model_mode}; fusion={model.use_fusion}; "
        f"APE consistency={use_ape_consistency}"
    )
    if args.evaluate_only:
        metrics = evaluate(
            model,
            val_dataset,
            device,
            include_surface_metrics=True,
            voxel_spacing_mm=args.voxel_spacing_mm,
            figure_pair=args.figure_pair,
            export_figure_npz=args.export_figure_npz,
        )
        (output_dir / "evaluation.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        write_evaluation_csv(output_dir, metrics)
        print(
            json.dumps(
                {
                    "paper_table_metrics": metrics["paper_table_metrics"],
                    "valid_pair_count": metrics["valid_pair_count"],
                    "voxel_spacing_mm_dhw": metrics["voxel_spacing_mm_dhw"],
                    "output_dir": str(output_dir),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    bilinear = SpatialTransformer(full_size, mode="bilinear").to(device)
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        totals = {"loss": 0.0, "ncc": 0.0, "grad": 0.0, "body": 0.0}
        if use_ape_consistency:
            warmup = max(0, int(args.ape_consistency_warmup_epochs))
            ramp = 1.0 if warmup == 0 else min(1.0, epoch / float(warmup))
            ape_weight = float(args.w_ape_consistency) * ramp
        else:
            ape_weight = 0.0
        progress = tqdm(loader, desc=f"Clean abdomen registration {epoch}/{args.epochs}")
        for batch in progress:
            moving = batch["moving"].to(device).float()
            fixed = batch["fixed"].to(device).float()
            moving_mask = batch["moving_mask"].to(device).float()
            fixed_mask = batch["fixed_mask"].to(device).float()
            if use_ape_consistency:
                warped, flow, aux = model(
                    moving, fixed, moving_mask, fixed_mask, return_aux=True
                )
                ape_loss = ape_assignment_consistency_loss(
                    aux["moving_assignments"],
                    aux["fixed_assignments"],
                    aux["fixed_stage_masks"],
                    flow,
                    model.ape_stn_cache,
                    args.ape_consistency_stage_weights,
                )
            else:
                warped, flow = model(moving, fixed, moving_mask, fixed_mask)
                ape_loss = flow.new_zeros([])
            ncc = masked_local_ncc_loss(
                warped, fixed, fixed_mask, window=args.ncc_window
            )
            gradient = flow_gradient_loss(flow)
            warped_mask = bilinear(moving_mask, flow).clamp(0, 1)
            body = body_mask_dice_loss(warped_mask, fixed_mask)
            loss = (
                args.w_ncc * ncc
                + args.w_grad * gradient
                + args.w_body * body
                + ape_weight * ape_loss
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            values = {
                "loss": loss,
                "ncc": ncc,
                "grad": gradient,
                "body": body,
            }
            for name, value in values.items():
                totals[name] += float(value)
            progress.set_postfix(loss=f"{float(loss):.4f}")
        averages = {name: value / len(loader) for name, value in totals.items()}
        if epoch == 1 or epoch % args.eval_every == 0:
            metrics = evaluate(model, val_dataset, device)
            metrics["epoch"] = epoch
            metrics["train"] = averages
            (output_dir / f"validation_epoch_{epoch:04d}.json").write_text(
                json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            score = metrics["equal_weighted_organ_mean_dice"]
            if score is not None and score > best_dice:
                best_dice = float(score)
                save_checkpoint(
                    output_dir / "checkpoints" / "registration_best.pth",
                    checkpoint_payload(
                        model,
                        optimizer,
                        epoch,
                        best_dice,
                        args,
                        use_ape_consistency,
                    ),
                )
        if epoch % args.save_every == 0:
            save_checkpoint(
                output_dir / "checkpoints" / f"registration_epoch{epoch:04d}.pth",
                checkpoint_payload(
                    model,
                    optimizer,
                    epoch,
                    best_dice,
                    args,
                    use_ape_consistency,
                ),
            )
        print(
            f"Epoch {epoch}: loss={averages['loss']:.5f}; "
            f"best validation Dice={best_dice:.5f}"
        )


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
