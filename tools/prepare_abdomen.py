#!/usr/bin/env python3
"""Prepare the labeled Learn2Reg AbdomenCTCT subjects for EAPR-Reg.

The source NIfTI files are never modified. Each processed subject is stored as
``(image, label, body_mask)``. Existing loaders that read the first two entries
remain compatible; abdomen-specific loaders also consume the CT-derived mask.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import os
import pickle
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
try:
    import nibabel as nib
except ImportError:
    nib = None

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from abdomen_eapr.data import make_body_mask_np


LABELS = {
    0: "background",
    1: "spleen",
    2: "right_kidney",
    3: "left_kidney",
    4: "gall_bladder",
    5: "esophagus",
    6: "liver",
    7: "stomach",
    8: "aorta",
    9: "inferior_vena_cava",
    10: "portal_and_splenic_vein",
    11: "pancreas",
    12: "left_adrenal_gland",
    13: "right_adrenal_gland",
}

EXPECTED_SOURCE_SHAPE = (192, 160, 256)  # nibabel X, Y, Z
EXPECTED_SPACING_MM = (2.0, 2.0, 2.0)
EXPECTED_STORED_SHAPE = (160, 192, 224)  # H, W, D for existing loaders
CROP_PROFILES = {
    "label_preserving": {
        "start_xyz": (0, 0, 32),
        "stop_xyz": (192, 160, 256),
        "allow_label_loss": False,
        "version": "abdomen_ctct_processed_v1",
        "rationale": "Fixed 224-slice label-preserving window; removes no labeled voxels in cases 0001-0030.",
    },
    "sacb_center": {
        "start_xyz": (0, 0, 16),
        "stop_xyz": (192, 160, 240),
        "allow_label_loss": True,
        "version": "abdomen_ctct_sacb_center_v1",
        "rationale": "Center crop stated by SACB-Net: 192x160x256 to 192x160x224, removing 16 slices from each z boundary.",
    },
}


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Original Learn2Reg AbdomenCTCT directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=here.parent / "data" / "AbdomenCTCT_processed_sacb_center_prospective_split_v3",
        help="Destination for processed pkl files and manifests.",
    )
    parser.add_argument(
        "--split",
        type=Path,
        default=here.parent / "protocols" / "prospective_split_v3.json",
        help="Subject-level split JSON.",
    )
    parser.add_argument(
        "--crop-profile",
        choices=sorted(CROP_PROFILES),
        default="sacb_center",
        help="Crop protocol. Use sacb_center for the center crop reported by SACB-Net.",
    )
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    path.write_text(text, encoding="utf-8")


def atomic_pickle(path: Path, payload: Tuple[np.ndarray, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def load_split(path: Path) -> Dict[str, Any]:
    split = json.loads(path.read_text(encoding="utf-8"))
    expected_counts = {"train": 19, "val": 3, "test": 8}
    all_ids: List[str] = []
    for split_name, count in expected_counts.items():
        case_ids = split.get(split_name)
        if not isinstance(case_ids, list) or len(case_ids) != count:
            raise ValueError(f"{split_name} must contain {count} case IDs")
        if case_ids != sorted(case_ids):
            raise ValueError(f"{split_name} case IDs must be sorted")
        all_ids.extend(case_ids)
    expected_ids = [f"{index:04d}" for index in range(1, 31)]
    if sorted(all_ids) != expected_ids:
        raise ValueError("The split must contain each labeled case 0001-0030 exactly once")
    return split


def case_paths(source: Path, case_id: str) -> Tuple[Path, Path]:
    filename = f"AbdomenCTCT_{case_id}_0000.nii.gz"
    return source / "imagesTr" / filename, source / "labelsTr" / filename


def validate_header(image_nii: nib.Nifti1Image, label_nii: nib.Nifti1Image, case_id: str) -> None:
    if tuple(image_nii.shape) != EXPECTED_SOURCE_SHAPE:
        raise ValueError(f"{case_id}: unexpected image shape {image_nii.shape}")
    if tuple(label_nii.shape) != EXPECTED_SOURCE_SHAPE:
        raise ValueError(f"{case_id}: unexpected label shape {label_nii.shape}")
    spacing = tuple(float(value) for value in image_nii.header.get_zooms()[:3])
    if not np.allclose(spacing, EXPECTED_SPACING_MM, atol=1e-5):
        raise ValueError(f"{case_id}: unexpected spacing {spacing}")
    if not np.allclose(image_nii.affine, label_nii.affine, atol=1e-5):
        raise ValueError(f"{case_id}: image and label affines do not match")
    orientation = "".join(nib.aff2axcodes(image_nii.affine))
    if orientation != "RAS":
        raise ValueError(f"{case_id}: expected RAS orientation, got {orientation}")


def crop_xyz(
    array: np.ndarray,
    crop_start_xyz: Sequence[int],
    crop_stop_xyz: Sequence[int],
) -> np.ndarray:
    x0, y0, z0 = crop_start_xyz
    x1, y1, z1 = crop_stop_xyz
    return array[x0:x1, y0:y1, z0:z1]


def process_case(
    source: Path,
    output_path: Path,
    case_id: str,
    split_name: str,
    crop_profile: Dict[str, Any],
) -> Dict[str, Any]:
    image_path, label_path = case_paths(source, case_id)
    if not image_path.is_file() or not label_path.is_file():
        raise FileNotFoundError(f"{case_id}: missing source image or label")

    image_nii = nib.load(str(image_path))
    label_nii = nib.load(str(label_path))
    validate_header(image_nii, label_nii, case_id)

    image_xyz = image_nii.get_fdata(dtype=np.float32)
    label_raw = np.asanyarray(label_nii.dataobj)
    if not np.all(label_raw == np.rint(label_raw)):
        raise ValueError(f"{case_id}: label contains non-integer values")
    label_xyz = np.asarray(label_raw, dtype=np.uint8)

    unique_labels = sorted(int(value) for value in np.unique(label_xyz))
    invalid_labels = sorted(set(unique_labels) - set(LABELS))
    if invalid_labels:
        raise ValueError(f"{case_id}: invalid labels {invalid_labels}")

    retained_mask = np.zeros(EXPECTED_SOURCE_SHAPE, dtype=bool)
    crop_start_xyz = crop_profile["start_xyz"]
    crop_stop_xyz = crop_profile["stop_xyz"]
    x0, y0, z0 = crop_start_xyz
    x1, y1, z1 = crop_stop_xyz
    retained_mask[x0:x1, y0:y1, z0:z1] = True
    cropped_region = label_xyz[~retained_mask]
    cropped_label_voxels_by_label = {
        label_id: int(np.count_nonzero(cropped_region == label_id)) for label_id in range(1, 14)
    }
    cropped_label_voxels_by_label = {
        label_id: count for label_id, count in cropped_label_voxels_by_label.items() if count > 0
    }
    cropped_label_voxels = int(sum(cropped_label_voxels_by_label.values()))
    if cropped_label_voxels and not crop_profile["allow_label_loss"]:
        raise ValueError(f"{case_id}: crop would remove {cropped_label_voxels} labeled voxels")

    hu_min = float(np.min(image_xyz))
    hu_max = float(np.max(image_xyz))
    image_xyz = np.clip(image_xyz, -1000.0, 1000.0)
    image_xyz = (image_xyz + 1000.0) / 2000.0

    image_xyz = crop_xyz(image_xyz, crop_start_xyz, crop_stop_xyz)
    label_xyz = crop_xyz(label_xyz, crop_start_xyz, crop_stop_xyz)

    # Store H, W, D so the existing loader's (2, 0, 1) transpose produces D, H, W.
    image_hwd = np.ascontiguousarray(np.transpose(image_xyz, (1, 0, 2)), dtype=np.float32)
    label_hwd = np.ascontiguousarray(np.transpose(label_xyz, (1, 0, 2)), dtype=np.uint8)
    if image_hwd.shape != EXPECTED_STORED_SHAPE or label_hwd.shape != EXPECTED_STORED_SHAPE:
        raise AssertionError(f"{case_id}: unexpected stored shape")
    if not np.isfinite(image_hwd).all() or image_hwd.min() < 0.0 or image_hwd.max() > 1.0:
        raise ValueError(f"{case_id}: normalized image is outside [0, 1]")

    image_dhw = np.ascontiguousarray(np.transpose(image_hwd, (2, 0, 1)), dtype=np.float32)
    body_mask_dhw = make_body_mask_np(image_dhw)
    label_dhw = np.ascontiguousarray(np.transpose(label_hwd, (2, 0, 1)))
    missed_organ_voxels = int(np.count_nonzero((label_dhw > 0) & ~body_mask_dhw))
    if missed_organ_voxels:
        raise ValueError(f"{case_id}: CT body mask excludes {missed_organ_voxels} labeled voxels")
    body_mask_hwd = np.ascontiguousarray(np.transpose(body_mask_dhw, (1, 2, 0)), dtype=np.bool_)

    atomic_pickle(output_path, (image_hwd, label_hwd, body_mask_hwd))

    counts = {label_id: int(np.count_nonzero(label_hwd == label_id)) for label_id in LABELS}
    missing = [label_id for label_id in range(1, 14) if counts[label_id] == 0]
    voxel_volume_ml = float(np.prod(EXPECTED_SPACING_MM) / 1000.0)
    return {
        "case_id": case_id,
        "split": split_name,
        "source_image": str(image_path.resolve()),
        "source_label": str(label_path.resolve()),
        "processed_file": str(output_path.resolve()),
        "source_image_sha256": sha256_file(image_path),
        "source_label_sha256": sha256_file(label_path),
        "processed_sha256": sha256_file(output_path),
        "source_hu_min": hu_min,
        "source_hu_max": hu_max,
        "cropped_label_voxels_total": cropped_label_voxels,
        "cropped_label_voxels_by_label": {
            str(label_id): count for label_id, count in cropped_label_voxels_by_label.items()
        },
        "body_mask_voxels": int(body_mask_hwd.sum()),
        "body_mask_fraction": round(float(body_mask_hwd.mean()), 8),
        "present_labels": [label_id for label_id in range(1, 14) if counts[label_id] > 0],
        "missing_labels": missing,
        "label_voxels": {str(label_id): counts[label_id] for label_id in range(1, 14)},
        "label_volume_ml": {
            str(label_id): round(counts[label_id] * voxel_volume_ml, 6) for label_id in range(1, 14)
        },
    }


def build_pairs(case_ids: Sequence[str]) -> List[Dict[str, str]]:
    return [
        {
            "moving": f"AbdomenCTCT_{moving}_0000.pkl",
            "fixed": f"AbdomenCTCT_{fixed}_0000.pkl",
            "moving_id": moving,
            "fixed_id": fixed,
        }
        for moving, fixed in itertools.permutations(case_ids, 2)
    ]


def write_qc_csv(path: Path, records: Sequence[Dict[str, Any]]) -> None:
    fieldnames = [
        "case_id",
        "split",
        "source_hu_min",
        "source_hu_max",
        "missing_labels",
        "cropped_label_voxels_total",
        "cropped_label_voxels_by_label",
    ] + [f"label_{label_id:02d}_volume_ml" for label_id in range(1, 14)]
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = {
                "case_id": record["case_id"],
                "split": record["split"],
                "source_hu_min": record["source_hu_min"],
                "source_hu_max": record["source_hu_max"],
                "missing_labels": ",".join(str(value) for value in record["missing_labels"]),
                "cropped_label_voxels_total": record["cropped_label_voxels_total"],
                "cropped_label_voxels_by_label": json.dumps(
                    record["cropped_label_voxels_by_label"], sort_keys=True
                ),
            }
            for label_id in range(1, 14):
                row[f"label_{label_id:02d}_volume_ml"] = record["label_volume_ml"][str(label_id)]
            writer.writerow(row)


def qc_summary(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    missing = {
        record["case_id"]: record["missing_labels"]
        for record in records
        if record["missing_labels"]
    }
    volume_outliers: Dict[str, List[Dict[str, Any]]] = {}
    for label_id in range(1, 14):
        values = np.asarray(
            [record["label_volume_ml"][str(label_id)] for record in records], dtype=np.float64
        )
        q1, q3 = np.percentile(values, [25.0, 75.0])
        iqr = q3 - q1
        lower = max(0.0, q1 - 3.0 * iqr)
        upper = q3 + 3.0 * iqr
        flagged = []
        for record, value in zip(records, values):
            if value == 0.0 or value < lower or value > upper:
                flagged.append({"case_id": record["case_id"], "volume_ml": round(float(value), 6)})
        if flagged:
            volume_outliers[str(label_id)] = flagged
    crop_loss_by_case = {
        record["case_id"]: {
            "total": record["cropped_label_voxels_total"],
            "by_label": record["cropped_label_voxels_by_label"],
        }
        for record in records
        if record["cropped_label_voxels_total"] > 0
    }
    crop_loss_by_label = {
        str(label_id): int(
            sum(
                record["cropped_label_voxels_by_label"].get(str(label_id), 0)
                for record in records
            )
        )
        for label_id in range(1, 14)
    }
    crop_loss_by_label = {key: value for key, value in crop_loss_by_label.items() if value > 0}
    return {
        "missing_labels_by_case": missing,
        "cropped_label_voxels_total": int(
            sum(record["cropped_label_voxels_total"] for record in records)
        ),
        "cropped_label_voxels_by_case": crop_loss_by_case,
        "cropped_label_voxels_by_label": crop_loss_by_label,
        "volume_outliers_3x_iqr": volume_outliers,
        "manual_review_required": [
            "Cases 0013 and 0025 have no label 4 (gallbladder).",
            "Case 0009 has an unusually large label 1 (spleen) volume and should be visually reviewed.",
        ],
    }


def validate_output(output: Path, split: Dict[str, Any]) -> None:
    expected = {"Train": 19, "Val": 3, "Test": 8}
    for directory, count in expected.items():
        files = sorted((output / directory).glob("*.pkl"))
        if len(files) != count:
            raise AssertionError(f"{directory}: expected {count} pkl files, got {len(files)}")
        for path in files:
            with path.open("rb") as stream:
                payload = pickle.load(stream)
            if not isinstance(payload, tuple) or len(payload) != 3:
                raise AssertionError(f"{path}: expected (image, label, body_mask)")
            if np.asarray(payload[2]).shape != EXPECTED_STORED_SHAPE:
                raise AssertionError(f"{path}: unexpected body-mask shape")
            image, label = payload[:2]
            if image.shape != EXPECTED_STORED_SHAPE or image.dtype != np.float32:
                raise AssertionError(f"{path}: invalid image shape or dtype")
            if label.shape != EXPECTED_STORED_SHAPE or label.dtype != np.uint8:
                raise AssertionError(f"{path}: invalid label shape or dtype")
    pair_counts = {"train": 342, "val": 6, "test": 56}
    for split_name, expected_count in pair_counts.items():
        pairs = json.loads((output / "manifests" / f"{split_name}_pairs.json").read_text(encoding="utf-8"))
        if len(pairs) != expected_count:
            raise AssertionError(f"{split_name}: expected {expected_count} pairs, got {len(pairs)}")


def prepare(args: argparse.Namespace) -> None:
    if nib is None:
        raise ImportError("NIfTI preprocessing requires nibabel: pip install nibabel")
    source = args.source.resolve()
    output = args.output.resolve()
    split_path = args.split.resolve()
    crop_profile = CROP_PROFILES[args.crop_profile]
    if not source.is_dir():
        raise FileNotFoundError(f"Source directory not found: {source}")
    if output == source or source in output.parents:
        raise ValueError("Output must not be the source directory or a child of the source directory")
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}. Choose a new output directory.")

    split = load_split(split_path)
    output.mkdir(parents=True)
    records: List[Dict[str, Any]] = []
    directory_names = {"train": "Train", "val": "Val", "test": "Test"}
    try:
        for split_name in ("train", "val", "test"):
            target_directory = output / directory_names[split_name]
            for case_id in split[split_name]:
                filename = f"AbdomenCTCT_{case_id}_0000.pkl"
                print(f"[{split_name}] processing {case_id}", flush=True)
                records.append(
                    process_case(
                        source,
                        target_directory / filename,
                        case_id,
                        split_name,
                        crop_profile,
                    )
                )

        manifests = output / "manifests"
        write_json(manifests / "split.json", split)
        for split_name in ("train", "val", "test"):
            write_json(manifests / f"{split_name}_pairs.json", build_pairs(split[split_name]))
        write_json(
            manifests / "preprocessing.json",
            {
                "version": crop_profile["version"],
                "crop_profile": args.crop_profile,
                "source_shape_xyz": EXPECTED_SOURCE_SHAPE,
                "source_spacing_mm": EXPECTED_SPACING_MM,
                "source_orientation": "RAS",
                "crop_start_xyz_inclusive": crop_profile["start_xyz"],
                "crop_stop_xyz_exclusive": crop_profile["stop_xyz"],
                "crop_allows_label_loss": crop_profile["allow_label_loss"],
                "crop_rationale": crop_profile["rationale"],
                "hu_clip": [-1000.0, 1000.0],
                "normalization": "(clip(HU, -1000, 1000) + 1000) / 2000",
                "stored_axis_order": "HWD=YXZ",
                "stored_shape_hwd": EXPECTED_STORED_SHAPE,
                "network_axis_order": "DHW=ZYX",
                "network_shape_dhw": (224, 160, 192),
                "image_dtype": "float32",
                "label_dtype": "uint8",
                "labels": {str(key): value for key, value in LABELS.items()},
                "pkl_payload": "tuple(image_hwd, label_hwd, body_mask_hwd)",
                "body_mask": {
                    "source": "CT intensity only (no organ-label input)",
                    "threshold_normalized": 0.25,
                    "threshold_hu": -500,
                    "largest_component_3d": True,
                    "fill_holes_axial": True,
                    "dilation_iterations": 5,
                },
            },
        )
        write_json(manifests / "cases.json", records)
        write_json(manifests / "qc_summary.json", qc_summary(records))
        write_qc_csv(manifests / "qc.csv", records)
        validate_output(output, split)

        fingerprint_inputs = [
            manifests / "split.json",
            manifests / "preprocessing.json",
            manifests / "cases.json",
            manifests / "train_pairs.json",
            manifests / "val_pairs.json",
            manifests / "test_pairs.json",
        ]
        fingerprint = hashlib.sha256()
        for path in fingerprint_inputs:
            fingerprint.update(path.name.encode("utf-8"))
            fingerprint.update(path.read_bytes())
        (manifests / "dataset_fingerprint.sha256").write_text(
            fingerprint.hexdigest() + "\n", encoding="ascii"
        )
    except BaseException:
        print(f"Preparation failed; partial output remains for inspection: {output}", file=sys.stderr)
        raise

    print(f"Prepared dataset: {output}")
    print("Cases: Train=19, Val=3, Test=8")
    print("Ordered pairs: Train=342, Val=6, Test=56")
    print(f"Dataset fingerprint: {fingerprint.hexdigest()}")


if __name__ == "__main__":
    prepare(parse_args())
