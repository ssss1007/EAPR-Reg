"""Build a source-only archive using an explicit allowlist; no Git required."""

import argparse
import hashlib
import json
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[1]
ROOT_FILES = (
    "README.md", "THIRD_PARTY_NOTICES.md", "requirements.txt", ".gitignore",
    "ape_plan_d_supervised_record_copy.py",
)
IXI_FILES = (
    "__init__.py", "README.md", "UPSTREAM_README.md", "train.py", "eval_pkl.py",
    "evaluation_metrics.py", "test_evaluation_metrics.py", "sam2_runtime.py",
    "checkpoints.py",
)
ABD_FILES = (
    "__init__.py", "README.md", "ape.py", "config.py", "data.py", "medsam.py",
    "registration.py", "train_ape_v31.py", "train_registration.py", "test_clean_pipeline.py",
)
TOOL_FILES = (
    "__init__.py", "prepare_abdomen.py", "resplit_abdomen.py",
    "build_release.py", "test_release.py",
)


def release_files(root=ROOT):
    paths = [root / name for name in ROOT_FILES]
    for folder, names in (("ixi__eapr", IXI_FILES), ("abdomen_eapr", ABD_FILES), ("tools", TOOL_FILES)):
        paths.extend(root / folder / name for name in names)
    paths.extend(root / "protocols" / name for name in
                 ("prospective_split_v3.json", "identity_easy_split_v1.json"))
    vxm = root / "ixi__eapr" / "voxelmorph"
    paths.extend(vxm.glob("*.py"))
    paths.extend((vxm / "py").glob("*.py"))
    paths.extend((vxm / "torch").glob("*.py"))
    for vendor in ("pystrum", "neuron"):
        vendor_root = vxm / "torch" / "external" / vendor
        paths.extend(vendor_root.rglob("*.py"))
        paths.extend(vendor_root.rglob("LICENSE*"))
    result = sorted(set(paths), key=lambda path: path.relative_to(root).as_posix())
    for path in result:
        if not path.is_file():
            raise FileNotFoundError(f"Missing release source: {path}")
        if root.resolve() not in path.resolve().parents:
            raise ValueError(f"Source resolves outside repository: {path}")
    return result


def build(output, root=ROOT):
    files = release_files(root)
    output.mkdir(parents=True, exist_ok=True)
    manifest = []
    archive = output / "eapr-reg-source.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as stream:
        for path in files:
            name = path.relative_to(root).as_posix()
            content = path.read_bytes()
            # Fixed metadata keeps identical source archives reproducible.
            info = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            stream.writestr(info, content)
            manifest.append({"path": name, "bytes": len(content),
                             "sha256": hashlib.sha256(content).hexdigest()})
    report = {"archive": archive.name, "file_count": len(manifest),
              "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
              "files": manifest}
    (output / "release_manifest.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Built {archive}: {len(files)} source files, {archive.stat().st_size} bytes")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "dist")
    build(parser.parse_args().output.resolve())
