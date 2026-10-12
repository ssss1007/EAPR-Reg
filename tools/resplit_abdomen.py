"""Create a new AbdomenCTCT split without reprocessing image voxels."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List


EXPECTED_COUNTS = {"train": 19, "val": 3, "test": 8}
DIRECTORY_NAMES = {"train": "Train", "val": "Val", "test": "Test"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy PKL files instead of creating same-volume hard links.",
    )
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def validate_split(payload: Dict[str, Any]) -> Dict[str, str]:
    membership: Dict[str, str] = {}
    for split_name, expected_count in EXPECTED_COUNTS.items():
        case_ids = payload.get(split_name)
        if not isinstance(case_ids, list) or len(case_ids) != expected_count:
            raise ValueError(
                f"{split_name} must contain {expected_count} case IDs, got {case_ids!r}"
            )
        if case_ids != sorted(case_ids):
            raise ValueError(f"{split_name} case IDs must be sorted")
        for case_id in case_ids:
            if case_id in membership:
                raise ValueError(f"Case {case_id} appears in multiple splits")
            membership[str(case_id)] = split_name
    expected_ids = {f"{index:04d}" for index in range(1, 31)}
    if set(membership) != expected_ids:
        raise ValueError(
            f"Split must contain cases 0001-0030 exactly once; "
            f"missing={sorted(expected_ids - set(membership))}, "
            f"unexpected={sorted(set(membership) - expected_ids)}"
        )
    return membership


def discover_cases(source: Path) -> Dict[str, Path]:
    cases: Dict[str, Path] = {}
    for directory_name in DIRECTORY_NAMES.values():
        for path in sorted((source / directory_name).glob("AbdomenCTCT_*_0000.pkl")):
            parts = path.stem.split("_")
            if len(parts) < 3:
                continue
            case_id = parts[1]
            if case_id in cases:
                raise ValueError(f"Duplicate source case {case_id}: {cases[case_id]} and {path}")
            cases[case_id] = path
    expected_ids = {f"{index:04d}" for index in range(1, 31)}
    if set(cases) != expected_ids:
        raise ValueError(
            f"Source must contain cases 0001-0030 exactly once; "
            f"missing={sorted(expected_ids - set(cases))}"
        )
    return cases


def build_pairs(case_ids: Iterable[str]) -> List[Dict[str, str]]:
    return [
        {
            "moving": f"AbdomenCTCT_{moving}_0000.pkl",
            "moving_id": moving,
            "fixed": f"AbdomenCTCT_{fixed}_0000.pkl",
            "fixed_id": fixed,
        }
        for moving, fixed in itertools.permutations(case_ids, 2)
    ]


def update_case_records(
    records: List[Dict[str, Any]], membership: Dict[str, str], output: Path
) -> List[Dict[str, Any]]:
    by_id = {str(record["case_id"]): dict(record) for record in records}
    if set(by_id) != set(membership):
        raise ValueError("cases.json does not contain exactly the split case IDs")
    updated: List[Dict[str, Any]] = []
    for case_id in sorted(membership):
        record = by_id[case_id]
        split_name = membership[case_id]
        record["split"] = split_name
        record["processed_file"] = str(
            (output / DIRECTORY_NAMES[split_name] / f"AbdomenCTCT_{case_id}_0000.pkl").resolve()
        )
        updated.append(record)
    return updated


def update_qc_csv(source: Path, output: Path, membership: Dict[str, str]) -> None:
    with source.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    if "case_id" not in fieldnames or "split" not in fieldnames:
        raise ValueError("qc.csv must contain case_id and split columns")
    for row in rows:
        row["split"] = membership[str(row["case_id"])]
    rows.sort(key=lambda row: str(row["case_id"]))
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def dataset_fingerprint(manifests: Path) -> str:
    names = (
        "split.json",
        "preprocessing.json",
        "cases.json",
        "train_pairs.json",
        "val_pairs.json",
        "test_pairs.json",
    )
    digest = hashlib.sha256()
    for name in names:
        path = manifests / name
        digest.update(name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    split_path = args.split.resolve()
    if not source.is_dir():
        raise FileNotFoundError(source)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    if source == output or source in output.parents:
        raise ValueError("Output must not be the source or a child of the source")

    split_payload = read_json(split_path)
    if not isinstance(split_payload, dict):
        raise ValueError("Split JSON must be an object")
    membership = validate_split(split_payload)
    source_cases = discover_cases(source)
    source_manifests = source / "manifests"

    output.mkdir(parents=True)
    try:
        for case_id in sorted(membership):
            split_name = membership[case_id]
            target_dir = output / DIRECTORY_NAMES[split_name]
            target_dir.mkdir(parents=True, exist_ok=True)
            target = target_dir / source_cases[case_id].name
            if args.copy:
                shutil.copy2(source_cases[case_id], target)
            else:
                os.link(source_cases[case_id], target)

        manifests = output / "manifests"
        manifests.mkdir()
        write_json(manifests / "split.json", split_payload)
        for split_name in EXPECTED_COUNTS:
            write_json(
                manifests / f"{split_name}_pairs.json",
                build_pairs(split_payload[split_name]),
            )
        shutil.copy2(source_manifests / "preprocessing.json", manifests / "preprocessing.json")
        shutil.copy2(source_manifests / "qc_summary.json", manifests / "qc_summary.json")
        records = read_json(source_manifests / "cases.json")
        write_json(manifests / "cases.json", update_case_records(records, membership, output))
        update_qc_csv(source_manifests / "qc.csv", manifests / "qc.csv", membership)
        fingerprint = dataset_fingerprint(manifests)
        (manifests / "dataset_fingerprint.sha256").write_text(
            fingerprint + "\n", encoding="ascii"
        )

        for split_name, expected_count in EXPECTED_COUNTS.items():
            directory = output / DIRECTORY_NAMES[split_name]
            actual_count = len(list(directory.glob("*.pkl")))
            if actual_count != expected_count:
                raise AssertionError(
                    f"{split_name}: expected {expected_count} PKLs, got {actual_count}"
                )
            pair_count = len(read_json(manifests / f"{split_name}_pairs.json"))
            expected_pairs = expected_count * (expected_count - 1)
            if pair_count != expected_pairs:
                raise AssertionError(
                    f"{split_name}: expected {expected_pairs} pairs, got {pair_count}"
                )
    except BaseException:
        print(f"Resplit failed; partial output remains for inspection: {output}")
        raise

    mode = "copies" if args.copy else "hard links"
    print(f"Created resplit dataset using {mode}: {output}")
    print("Cases: Train=19, Val=3, Test=8")
    print("Ordered pairs: Train=342, Val=6, Test=56")
    print(f"Dataset fingerprint: {fingerprint}")


if __name__ == "__main__":
    main()
