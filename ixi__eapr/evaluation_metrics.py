"""Physical-space overlap and surface metrics for registration evaluation."""

from __future__ import annotations

from typing import Any, Dict, Sequence

import numpy as np
from scipy import ndimage


def _surface_distance_variant(
    warped: np.ndarray,
    fixed: np.ndarray,
    spacing: np.ndarray,
    connectivity: int,
):
    """
    connectivity:
        1 -> 6-connected surface
        3 -> 26-connected surface
    """

    structure = ndimage.generate_binary_structure(3, connectivity)

    warped_surface = np.logical_xor(
        warped,
        ndimage.binary_erosion(
            warped,
            structure=structure,
            border_value=0,
        ),
    )

    fixed_surface = np.logical_xor(
        fixed,
        ndimage.binary_erosion(
            fixed,
            structure=structure,
            border_value=0,
        ),
    )

    distance_to_fixed = ndimage.distance_transform_edt(
        np.logical_not(fixed_surface),
        sampling=spacing,
    )

    distance_to_warped = ndimage.distance_transform_edt(
        np.logical_not(warped_surface),
        sampling=spacing,
    )

    warped_to_fixed = distance_to_fixed[warped_surface]
    fixed_to_warped = distance_to_warped[fixed_surface]

    symmetric = np.concatenate([
        warped_to_fixed,
        fixed_to_warped,
    ])

    # -----------------------------
    # ASSD definition 1:
    # two directions contribute 50% each
    # this is your CURRENT definition
    # -----------------------------
    assd_directional_mean = 0.5 * (
        warped_to_fixed.mean()
        + fixed_to_warped.mean()
    )

    # -----------------------------
    # ASSD definition 2:
    # pool all surface distances
    # surface with more points gets larger weight
    # -----------------------------
    assd_pooled = symmetric.mean()

    # -----------------------------
    # HD95 definition 1:
    # pool both directions, then P95
    # this is your CURRENT definition
    # -----------------------------
    hd95_pooled = np.percentile(symmetric, 95)

    # -----------------------------
    # HD95 definition 2:
    # P95 each direction separately,
    # then take the worse direction
    # -----------------------------
    hd95_directional_max = max(
        np.percentile(warped_to_fixed, 95),
        np.percentile(fixed_to_warped, 95),
    )

    return {
        "assd_directional_mean_mm": float(assd_directional_mean),
        "assd_pooled_mm": float(assd_pooled),
        "hd95_pooled_mm": float(hd95_pooled),
        "hd95_directional_max_mm": float(hd95_directional_max),
    }


def binary_surface_distance_metrics(
    warped_mask: np.ndarray,
    fixed_mask: np.ndarray,
    spacing_mm: Sequence[float],
) -> Dict[str, Any]:
    """
    Calculate multiple HD95 / ASSD definitions.

    Returns:
        Existing compatibility metrics:
            hd95_mm
            assd_mm

        Additional audit metrics:
            6C:
                hd95_6c_pooled_mm
                hd95_6c_directional_max_mm
                assd_6c_directional_mean_mm
                assd_6c_pooled_mm

            26C:
                hd95_26c_pooled_mm
                hd95_26c_directional_max_mm
                assd_26c_directional_mean_mm
                assd_26c_pooled_mm
    """

    warped = np.asarray(warped_mask, dtype=np.bool_)
    fixed = np.asarray(fixed_mask, dtype=np.bool_)

    if (
        warped.ndim != 3
        or fixed.ndim != 3
        or warped.shape != fixed.shape
    ):
        raise ValueError(
            "Surface-distance inputs must be matching 3-D masks"
        )

    spacing = np.asarray(
        tuple(float(value) for value in spacing_mm),
        dtype=np.float64,
    )

    if (
        spacing.shape != (3,)
        or not np.isfinite(spacing).all()
        or np.any(spacing <= 0)
    ):
        raise ValueError(
            "spacing_mm must contain three positive finite values"
        )

    warped_present = bool(warped.any())
    fixed_present = bool(fixed.any())

    if not warped_present and not fixed_present:
        return {
            "hd95_mm": None,
            "assd_mm": None,

            "hd95_6c_pooled_mm": None,
            "hd95_6c_directional_max_mm": None,
            "assd_6c_directional_mean_mm": None,
            "assd_6c_pooled_mm": None,

            "hd95_26c_pooled_mm": None,
            "hd95_26c_directional_max_mm": None,
            "assd_26c_directional_mean_mm": None,
            "assd_26c_pooled_mm": None,

            "empty_case": "both_empty",
            "penalized": False,
        }

    if not warped_present or not fixed_present:
        diagonal = float(
            np.linalg.norm(
                (
                    np.asarray(warped.shape, dtype=np.float64)
                    - 1
                )
                * spacing
            )
        )

        return {
            "hd95_mm": diagonal,
            "assd_mm": diagonal,

            "hd95_6c_pooled_mm": diagonal,
            "hd95_6c_directional_max_mm": diagonal,
            "assd_6c_directional_mean_mm": diagonal,
            "assd_6c_pooled_mm": diagonal,

            "hd95_26c_pooled_mm": diagonal,
            "hd95_26c_directional_max_mm": diagonal,
            "assd_26c_directional_mean_mm": diagonal,
            "assd_26c_pooled_mm": diagonal,

            "empty_case": (
                "warped_empty"
                if not warped_present
                else "fixed_empty"
            ),
            "penalized": True,
        }

    # crop to relevant ROI region
    union_coordinates = np.where(
        np.logical_or(warped, fixed)
    )

    slices = tuple(
        slice(
            max(0, int(axis_values.min()) - 1),
            min(
                warped.shape[axis],
                int(axis_values.max()) + 2,
            ),
        )
        for axis, axis_values
        in enumerate(union_coordinates)
    )

    warped = warped[slices]
    fixed = fixed[slices]

    metrics_6c = _surface_distance_variant(
        warped,
        fixed,
        spacing,
        connectivity=1,
    )

    metrics_26c = _surface_distance_variant(
        warped,
        fixed,
        spacing,
        connectivity=3,
    )

    return {
        # -----------------------------------
        # Keep old names EXACTLY compatible.
        #
        # Old evaluator:
        # HD95 = 6C pooled percentile
        # ASSD = 6C directional-average
        # -----------------------------------
        "hd95_mm":
            metrics_6c["hd95_pooled_mm"],

        "assd_mm":
            metrics_6c["assd_directional_mean_mm"],

        # 6-connectivity
        "hd95_6c_pooled_mm":
            metrics_6c["hd95_pooled_mm"],

        "hd95_6c_directional_max_mm":
            metrics_6c["hd95_directional_max_mm"],

        "assd_6c_directional_mean_mm":
            metrics_6c["assd_directional_mean_mm"],

        "assd_6c_pooled_mm":
            metrics_6c["assd_pooled_mm"],

        # 26-connectivity
        "hd95_26c_pooled_mm":
            metrics_26c["hd95_pooled_mm"],

        "hd95_26c_directional_max_mm":
            metrics_26c["hd95_directional_max_mm"],

        "assd_26c_directional_mean_mm":
            metrics_26c["assd_directional_mean_mm"],

        "assd_26c_pooled_mm":
            metrics_26c["assd_pooled_mm"],

        "empty_case": None,
        "penalized": False,
    }