"""Locate an external SAM2 installation without machine-specific paths."""

import importlib
import sys
from pathlib import Path


def build_sam2_from_paths(config, checkpoint, repo=None, device="cuda"):
    root = Path(repo).expanduser().resolve() if repo else None
    if root:
        for candidate in (root, root.parent):
            if str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
    module = None
    errors = []
    for name in ("sam2main.sam2.build_sam", "sam2.build_sam"):
        try:
            module = importlib.import_module(name)
            break
        except ImportError as exc:
            errors.append(str(exc))
    if module is None:
        raise ImportError("Install SAM2 and pass --sam2-repo/--sam2-root. " + "; ".join(errors))

    config_path = Path(config).expanduser()
    if config_path.is_absolute():
        package_root = Path(module.__file__).resolve().parent
        try:
            config_name = config_path.resolve().relative_to(package_root).as_posix()
        except ValueError:
            # Hydra resolves configs within the SAM2 package, not arbitrary paths.
            raise ValueError(
                "Use a SAM2 package-relative config, e.g. "
                "configs/sam2.1/sam2.1_hiera_t.yaml, or an absolute path "
                f"inside {package_root}; got {config_path}"
            )
    else:
        config_name = config_path.as_posix()
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    return module.build_sam2(config_name, str(checkpoint_path), device=device,
                             apply_postprocessing=False)
