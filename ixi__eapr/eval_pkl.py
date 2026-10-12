import os
import glob
import pickle
import argparse
import csv
import json
import sys
from checkpoints import load_registration_weights

import numpy as np
import torch
from scipy import ndimage

if 'bool' not in np.__dict__:
    np.bool = np.bool_
if 'int' not in np.__dict__:
    np.int = int
if 'float' not in np.__dict__:
    np.float = float

os.environ['VXM_BACKEND'] = 'pytorch'

import voxelmorph as vxm


VOI_LABELS = np.array([
    2, 3, 4, 7, 8, 10, 11, 12, 13, 14,
    15, 16, 17, 18, 24, 28, 31, 41,
    42, 43, 46, 47, 49, 50, 51, 52,
    53, 54, 60, 63,
], dtype=np.int32)


SURFACE_METRIC_KEYS = [
    'hd95_6c_pooled_mm',
    'hd95_6c_directional_max_mm',
    'assd_6c_directional_mean_mm',
    'assd_6c_pooled_mm',
    'hd95_26c_pooled_mm',
    'hd95_26c_directional_max_mm',
    'assd_26c_directional_mean_mm',
    'assd_26c_pooled_mm',
]


def _to_dhw(arr):
    vol = np.asarray(arr)
    vol = np.squeeze(vol)
    if vol.ndim != 3:
        raise RuntimeError(f'Expected 3D volume, got shape={vol.shape}')
    if vol.shape[2] >= vol.shape[0] and vol.shape[2] >= vol.shape[1]:
        vol = np.transpose(vol, (2, 0, 1))
    return vol


def _load_pkl_img_seg(path):
    with open(path, 'rb') as f:
        obj = pickle.load(f)

    if isinstance(obj, (list, tuple)) and len(obj) >= 2:
        img, seg = obj[0], obj[1]
    elif isinstance(obj, dict):
        img = obj.get('image', obj.get('img', obj.get('vol', obj.get('data', None))))
        seg = obj.get('seg', obj.get('label', obj.get('mask', None)))
        if img is None or seg is None:
            raise RuntimeError(f'Cannot parse image/seg from dict pkl: {path}')
    else:
        raise RuntimeError(f'Unsupported pkl payload type: {type(obj)}')

    return _to_dhw(img), _to_dhw(seg)


def _load_labels(path):
    if path is None:
        return VOI_LABELS
    labels = np.load(path)
    if isinstance(labels, np.lib.npyio.NpzFile):
        return labels['labels'].astype(np.int32)
    return np.asarray(labels, dtype=np.int32)


def _select_figure_case(test_files, selector):
    """Resolve a 1-based test index or a unique filename substring."""
    if selector is None:
        return None

    selector = str(selector).strip()

    if selector.isdigit():
        index = int(selector)
        if index < 1 or index > len(test_files):
            raise RuntimeError(
                f'--figure-case index {index} is outside 1..{len(test_files)}'
            )
        return test_files[index - 1]

    exact = [
        path for path in test_files
        if os.path.basename(path) == selector
        or os.path.splitext(os.path.basename(path))[0] == selector
    ]

    if len(exact) == 1:
        return exact[0]

    matches = [
        path for path in test_files
        if selector in os.path.basename(path)
    ]

    if len(matches) != 1:
        names = ', '.join(os.path.basename(path) for path in matches[:8])
        raise RuntimeError(
            f'--figure-case {selector!r} matched {len(matches)} files'
            + (f': {names}' if names else '')
        )

    return matches[0]


def _save_figure_npz(
    output_path,
    moving_image,
    fixed_image,
    warped_image,
    moving_seg,
    fixed_seg,
    warped_seg,
    flow,
):
    output_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    np.savez_compressed(
        output_path,
        moving_image=np.asarray(moving_image, dtype=np.float32),
        fixed_image=np.asarray(fixed_image, dtype=np.float32),
        warped_image=np.asarray(warped_image, dtype=np.float32),
        moving_seg=np.asarray(moving_seg, dtype=np.int16),
        fixed_seg=np.asarray(fixed_seg, dtype=np.int16),
        warped_seg=np.asarray(warped_seg, dtype=np.int16),
        flow=np.asarray(flow, dtype=np.float32),
    )

    print(f'Saved qualitative figure data: {output_path}', flush=True)


def _build_model(args, inshape, device):
    ape_k_values = tuple(
        int(v.strip())
        for v in str(args.ape_k_list).split(',')
        if v.strip()
    )

    ape_channels_value = (
        None if int(args.ape_channels) <= 0
        else int(args.ape_channels)
    )

    model = vxm.iirp.RPNet(
        inshape,
        first_channel=int(args.first_channel),
        use_ape=args.use_ape,
        ape_ckpt=args.ape_ckpt,
        sam2_cfg=args.sam2_cfg,
        sam2_ckpt=args.sam2_ckpt,
        sam2_repo=args.sam2_repo,
        ape_k_list=ape_k_values,
        ape_temp=args.ape_temp,
        ape_channels=ape_channels_value,
        use_raw_medsam=args.use_raw_medsam,
        ape_use_raw_fusion=args.ape_use_raw_fusion,
        ape_use_gated_raw_fusion=args.ape_use_gated_raw_fusion,
        ape_use_eadf=args.ape_use_eadf,
        ape_use_half_refine=args.ape_use_half_refine,
        ape_use_context_refine=args.ape_use_context_refine,
        ape_use_proto_attn=args.ape_use_proto_attn,
        raw_medsam_use_refine=args.raw_medsam_use_refine,
    )

    checkpoint = torch.load(
        args.model,
        map_location=torch.device(device),
    )

    state_dict = (
        checkpoint['model_state']
        if isinstance(checkpoint, dict) and 'model_state' in checkpoint
        else checkpoint
    )

    load_registration_weights(model, state_dict)
    model.to(device)
    model.eval()

    return model


def _surface_distance_variant(
    warped,
    fixed,
    spacing,
    connectivity,
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

    assd_directional_mean = 0.5 * (
        warped_to_fixed.mean()
        + fixed_to_warped.mean()
    )

    assd_pooled = symmetric.mean()

    hd95_pooled = np.percentile(symmetric, 95)

    hd95_directional_max = max(
        np.percentile(warped_to_fixed, 95),
        np.percentile(fixed_to_warped, 95),
    )

    return {
        'hd95_pooled_mm': float(hd95_pooled),
        'hd95_directional_max_mm': float(hd95_directional_max),
        'assd_directional_mean_mm': float(assd_directional_mean),
        'assd_pooled_mm': float(assd_pooled),
    }


def binary_surface_distance_metrics(
    warped_mask,
    fixed_mask,
    spacing_mm,
):
    """
    Compute multiple surface-distance definitions for one binary ROI.

    Compatibility metrics:
        hd95_mm:
            6-connected surfaces + pooled symmetric 95th percentile.
        assd_mm:
            6-connected surfaces + equal-weight directional means.

    Audit metrics:
        6C / 26C
        pooled HD95 / directional-max HD95
        directional-mean ASSD / pooled ASSD
    """

    warped = np.asarray(warped_mask, dtype=np.bool_)
    fixed = np.asarray(fixed_mask, dtype=np.bool_)

    if (
        warped.ndim != 3
        or fixed.ndim != 3
        or warped.shape != fixed.shape
    ):
        raise ValueError(
            'Surface-distance inputs must be matching 3-D masks'
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
            'spacing_mm must contain three positive finite values'
        )

    warped_present = bool(warped.any())
    fixed_present = bool(fixed.any())

    if not warped_present and not fixed_present:
        result = {
            'hd95_mm': None,
            'assd_mm': None,
            'empty_case': 'both_empty',
            'penalized': False,
        }
        for key in SURFACE_METRIC_KEYS:
            result[key] = None
        return result

    if not warped_present or not fixed_present:
        diagonal = float(
            np.linalg.norm(
                (
                    np.asarray(warped.shape, dtype=np.float64)
                    - 1
                ) * spacing
            )
        )

        result = {
            'hd95_mm': diagonal,
            'assd_mm': diagonal,
            'empty_case': (
                'warped_empty'
                if not warped_present
                else 'fixed_empty'
            ),
            'penalized': True,
        }

        for key in SURFACE_METRIC_KEYS:
            result[key] = diagonal

        return result

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
        # Original/compatibility metrics.
        'hd95_mm':
            metrics_6c['hd95_pooled_mm'],
        'assd_mm':
            metrics_6c['assd_directional_mean_mm'],

        # 6-connected.
        'hd95_6c_pooled_mm':
            metrics_6c['hd95_pooled_mm'],
        'hd95_6c_directional_max_mm':
            metrics_6c['hd95_directional_max_mm'],
        'assd_6c_directional_mean_mm':
            metrics_6c['assd_directional_mean_mm'],
        'assd_6c_pooled_mm':
            metrics_6c['assd_pooled_mm'],

        # 26-connected.
        'hd95_26c_pooled_mm':
            metrics_26c['hd95_pooled_mm'],
        'hd95_26c_directional_max_mm':
            metrics_26c['hd95_directional_max_mm'],
        'assd_26c_directional_mean_mm':
            metrics_26c['assd_directional_mean_mm'],
        'assd_26c_pooled_mm':
            metrics_26c['assd_pooled_mm'],

        'empty_case': None,
        'penalized': False,
    }


def _safe_mean_std(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return None, None
    return float(values.mean()), float(values.std())


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--test-dir',
        required=True,
        help='directory containing test .pkl files',
    )

    parser.add_argument(
        '--atlas-file',
        required=True,
        help='fixed atlas .pkl with image and segmentation',
    )

    parser.add_argument(
        '--model',
        required=True,
        help='registration checkpoint, e.g. 0010.pt',
    )

    parser.add_argument(
        '--labels',
        default=None,
        help='optional npz/npy label list',
    )

    parser.add_argument('--gpu', default='0')
    parser.add_argument('--first-channel', type=int, default=8)

    parser.add_argument('--use-ape', action='store_true')
    parser.add_argument('--ape-ckpt', default=None)
    parser.add_argument('--sam2-cfg', default=None)
    parser.add_argument('--sam2-ckpt', default=None)
    parser.add_argument('--sam2-repo', default=None)
    parser.add_argument('--ape-k-list', default='24,10,6')
    parser.add_argument('--ape-temp', type=float, default=0.3)
    parser.add_argument('--ape-channels', type=int, default=0)
    parser.add_argument('--use-raw-medsam', action='store_true')
    parser.add_argument('--ape-use-raw-fusion', action='store_true')
    parser.add_argument('--ape-use-gated-raw-fusion', action='store_true')
    parser.add_argument('--ape-use-eadf', action='store_true')
    parser.add_argument('--ape-use-half-refine', action='store_true')
    parser.add_argument('--ape-use-context-refine', action='store_true')
    parser.add_argument('--ape-use-proto-attn', action='store_true')
    parser.add_argument('--raw-medsam-use-refine', action='store_true')

    parser.add_argument('--save-csv', default=None)

    parser.add_argument(
        '--figure-case',
        default=None,
        help=(
            'test case to export, specified as a 1-based sorted test index, '
            'an exact filename/stem, or a unique filename substring'
        ),
    )

    parser.add_argument(
        '--export-figure-npz',
        default=None,
        help='save the selected case in the common qualitative-figure NPZ format',
    )

    parser.add_argument(
        '--figure-only',
        action='store_true',
        help='evaluate only --figure-case; do not use this mode for paper metrics',
    )

    parser.add_argument(
        '--voxel-spacing-mm',
        type=float,
        nargs=3,
        default=[1.0, 1.0, 1.0],
        metavar=('D0', 'D1', 'D2'),
        help='physical spacing for HD95/ASSD in the evaluated array axis order',
    )

    args = parser.parse_args()

    if any(value <= 0 for value in args.voxel_spacing_mm):
        parser.error(
            '--voxel-spacing-mm values must be positive'
        )

    if (
        (args.figure_case is None)
        != (args.export_figure_npz is None)
    ):
        parser.error(
            '--figure-case and --export-figure-npz must be used together'
        )

    if args.figure_only and args.figure_case is None:
        parser.error(
            '--figure-only requires --figure-case and --export-figure-npz'
        )

    if args.gpu and args.gpu != '-1':
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        device = 'cuda'
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        device = 'cpu'

    test_files = sorted(
        glob.glob(
            os.path.join(args.test_dir, '*.pkl')
        )
    )

    test_files += sorted(
        glob.glob(
            os.path.join(
                args.test_dir,
                '**',
                '*.pkl',
            ),
            recursive=True,
        )
    )

    test_files = sorted(set(test_files))

    if not test_files:
        raise RuntimeError(
            f'No .pkl files found under {args.test_dir}'
        )

    selected_figure_path = _select_figure_case(
        test_files,
        args.figure_case,
    )

    if args.figure_only:
        test_files = [selected_figure_path]

    atlas_img, atlas_seg = _load_pkl_img_seg(
        args.atlas_file
    )

    labels = _load_labels(args.labels)

    inshape = tuple(
        int(v) for v in atlas_img.shape
    )

    model = _build_model(
        args,
        inshape,
        device,
    )

    trf = vxm.layers.SpatialTransformer(
        inshape,
        mode='nearest',
    ).to(device)

    fixed = (
        torch.from_numpy(atlas_img)
        .float()
        .unsqueeze(0)
        .unsqueeze(0)
        .to(device)
    )

    atlas_seg_i = atlas_seg.astype(np.int32)

    case_means = []
    roi_rows = []

    jac_nonpos = []
    jac_nonpos_percent = []
    jac_negative_percent = []

    # Original/compatibility surface metrics.
    case_hd95 = []
    case_assd = []

    # Audit variants.
    surface_case_values = {
        key: []
        for key in SURFACE_METRIC_KEYS
    }

    surface_roi_values = {
        key: []
        for key in SURFACE_METRIC_KEYS
    }

    case_records = []
    roi_records = []

    with torch.no_grad():
        for idx, path in enumerate(
            test_files,
            start=1,
        ):
            moving_img, moving_seg = _load_pkl_img_seg(
                path
            )

            moving = (
                torch.from_numpy(moving_img)
                .float()
                .unsqueeze(0)
                .unsqueeze(0)
                .to(device)
            )

            moving_seg_t = (
                torch.from_numpy(moving_seg)
                .float()
                .unsqueeze(0)
                .unsqueeze(0)
                .to(device)
            )

            pred = model(moving, fixed)

            flow = pred[1]

            warped_image = (
                pred[0]
                .detach()
                .cpu()
                .numpy()
                .squeeze()
            )

            warped_seg = (
                trf(
                    moving_seg_t,
                    flow,
                )
                .detach()
                .cpu()
                .numpy()
                .squeeze()
            )

            warped_seg = np.rint(
                warped_seg
            ).astype(np.int32)

            if (
                selected_figure_path is not None
                and path == selected_figure_path
            ):
                _save_figure_npz(
                    args.export_figure_npz,
                    moving_img,
                    atlas_img,
                    warped_image,
                    moving_seg,
                    atlas_seg_i,
                    warped_seg,
                    flow.detach().cpu().numpy().squeeze(),
                )

            roi_dice = vxm.py.utils.dice(
                warped_seg,
                atlas_seg_i,
                labels=labels,
            )

            case_mean = float(
                np.mean(roi_dice)
            )

            case_means.append(case_mean)
            roi_rows.extend(
                [float(v) for v in roi_dice]
            )

            flow_np = (
                flow.detach()
                .cpu()
                .numpy()
                .squeeze()
                .transpose(1, 2, 3, 0)
            )

            jac = vxm.py.utils.jacobian_determinant(
                flow_np
            )

            if not np.isfinite(jac).all():
                raise RuntimeError(
                    f'Non-finite Jacobian determinant for {path}'
                )

            nonpos = int(
                np.sum(jac <= 0)
            )

            nonpos_percent = (
                100.0
                * float(np.mean(jac <= 0))
            )

            negative_percent = (
                100.0
                * float(np.mean(jac < 0))
            )

            jac_nonpos.append(nonpos)
            jac_nonpos_percent.append(
                nonpos_percent
            )
            jac_negative_percent.append(
                negative_percent
            )

            hd95_values = []
            assd_values = []

            case_surface_values = {
                key: []
                for key in SURFACE_METRIC_KEYS
            }

            penalized_rois = 0

            for label, dice_value in zip(
                labels,
                roi_dice,
            ):
                surface = binary_surface_distance_metrics(
                    warped_seg == int(label),
                    atlas_seg_i == int(label),
                    args.voxel_spacing_mm,
                )

                if surface['hd95_mm'] is not None:
                    hd95_values.append(
                        float(surface['hd95_mm'])
                    )

                    assd_values.append(
                        float(surface['assd_mm'])
                    )

                for key in SURFACE_METRIC_KEYS:
                    value = surface[key]

                    if value is not None:
                        value = float(value)

                        case_surface_values[key].append(
                            value
                        )

                        surface_roi_values[key].append(
                            value
                        )

                penalized_rois += int(
                    bool(surface['penalized'])
                )

                roi_record = {
                    'case': os.path.basename(path),
                    'label': int(label),
                    'dice': float(dice_value),

                    # Original metrics.
                    'hd95_mm':
                        surface['hd95_mm'],
                    'assd_mm':
                        surface['assd_mm'],

                    'empty_case':
                        surface['empty_case'],
                    'penalized':
                        bool(surface['penalized']),
                }

                for key in SURFACE_METRIC_KEYS:
                    roi_record[key] = surface[key]

                roi_records.append(
                    roi_record
                )

            if not hd95_values:
                raise RuntimeError(
                    f'No valid surface metrics for {path}'
                )

            hd95_mean = float(
                np.mean(hd95_values)
            )

            assd_mean = float(
                np.mean(assd_values)
            )

            case_hd95.append(
                hd95_mean
            )

            case_assd.append(
                assd_mean
            )

            case_record = {
                'case':
                    os.path.basename(path),
                'dice_mean':
                    case_mean,
                'hd95_mm':
                    hd95_mean,
                'assd_mm':
                    assd_mean,
                'jac_negative_percent':
                    negative_percent,
                'jac_nonpositive_percent':
                    nonpos_percent,
                'jac_nonpositive_voxels':
                    nonpos,
                'valid_roi_count':
                    len(hd95_values),
                'penalized_roi_count':
                    penalized_rois,
            }

            for key in SURFACE_METRIC_KEYS:
                values = np.asarray(
                    case_surface_values[key],
                    dtype=np.float64,
                )

                if values.size == 0:
                    raise RuntimeError(
                        f'No valid {key} for {path}'
                    )

                value_mean = float(
                    values.mean()
                )

                surface_case_values[key].append(
                    value_mean
                )

                case_record[key] = value_mean

            case_records.append(
                case_record
            )

            print(
                f'[{idx:03d}/{len(test_files):03d}] '
                f'{os.path.basename(path)} '
                f'Dice={case_mean:.4f} '
                f'HD95={hd95_mean:.4f}mm '
                f'ASSD={assd_mean:.4f}mm '
                f'J<0={negative_percent:.4f}%',
                flush=True,
            )

    case_means = np.asarray(
        case_means,
        dtype=np.float64,
    )

    roi_rows = np.asarray(
        roi_rows,
        dtype=np.float64,
    )

    jac_nonpos = np.asarray(
        jac_nonpos,
        dtype=np.float64,
    )

    jac_nonpos_percent = np.asarray(
        jac_nonpos_percent,
        dtype=np.float64,
    )

    jac_negative_percent = np.asarray(
        jac_negative_percent,
        dtype=np.float64,
    )

    case_hd95 = np.asarray(
        case_hd95,
        dtype=np.float64,
    )

    case_assd = np.asarray(
        case_assd,
        dtype=np.float64,
    )

    for key in SURFACE_METRIC_KEYS:
        surface_case_values[key] = np.asarray(
            surface_case_values[key],
            dtype=np.float64,
        )

        surface_roi_values[key] = np.asarray(
            surface_roi_values[key],
            dtype=np.float64,
        )

    per_label = {}

    for label in labels:
        rows = [
            row for row in roi_records
            if row['label'] == int(label)
        ]

        dice_values = np.asarray(
            [row['dice'] for row in rows],
            dtype=np.float64,
        )

        hd95_values = np.asarray(
            [
                row['hd95_mm']
                for row in rows
                if row['hd95_mm'] is not None
            ],
            dtype=np.float64,
        )

        assd_values = np.asarray(
            [
                row['assd_mm']
                for row in rows
                if row['assd_mm'] is not None
            ],
            dtype=np.float64,
        )

        per_label[str(int(label))] = {
            'dice_mean':
                float(dice_values.mean()),
            'dice_std':
                float(dice_values.std()),

            'hd95_mm_mean':
                float(hd95_values.mean())
                if hd95_values.size
                else None,

            'hd95_mm_std':
                float(hd95_values.std())
                if hd95_values.size
                else None,

            'assd_mm_mean':
                float(assd_values.mean())
                if assd_values.size
                else None,

            'assd_mm_std':
                float(assd_values.std())
                if assd_values.size
                else None,

            'case_count':
                len(rows),

            'penalized_case_count':
                int(
                    sum(
                        bool(row['penalized'])
                        for row in rows
                    )
                ),
        }

    surface_metric_audit = {}

    for key in SURFACE_METRIC_KEYS:
        case_values = surface_case_values[key]
        roi_values = surface_roi_values[key]

        surface_metric_audit[key] = {
            'case_level': {
                'mean':
                    float(case_values.mean()),
                'std':
                    float(case_values.std()),
            },
            'roi_level': {
                'mean':
                    float(roi_values.mean()),
                'std':
                    float(roi_values.std()),
            },
        }

    summary = {
        'evaluation_inputs': {
            'test_dir':
                os.path.abspath(args.test_dir),

            'atlas_file':
                os.path.abspath(args.atlas_file),

            'model':
                os.path.abspath(args.model),

            'ape_checkpoint': (
                os.path.abspath(args.ape_ckpt)
                if args.ape_ckpt
                else None
            ),
        },

        # Preserves the behavior of your previous evaluator.
        'paper_table_metrics_original_case_level': {
            'dice': {
                'mean':
                    float(case_means.mean()),
                'std':
                    float(case_means.std()),
            },

            'hd95_mm': {
                'mean':
                    float(case_hd95.mean()),
                'std':
                    float(case_hd95.std()),
            },

            'assd_mm': {
                'mean':
                    float(case_assd.mean()),
                'std':
                    float(case_assd.std()),
            },

            'jac_negative_percent': {
                'mean':
                    float(jac_negative_percent.mean()),
                'std':
                    float(jac_negative_percent.std()),
            },
        },

        # Dice ROI-level statistics are useful for matching published
        # IXI table conventions where std is often much larger than
        # case-level std.
        'roi_level_dice': {
            'mean':
                float(roi_rows.mean()),
            'std':
                float(roi_rows.std()),
        },

        'surface_metric_audit':
            surface_metric_audit,

        'jac_nonpositive_percent': {
            'mean':
                float(jac_nonpos_percent.mean()),
            'std':
                float(jac_nonpos_percent.std()),
        },

        'jac_nonpositive_voxels': {
            'mean':
                float(jac_nonpos.mean()),
            'std':
                float(jac_nonpos.std()),
        },

        'case_count':
            len(case_records),

        'roi_labels': [
            int(label)
            for label in labels
        ],

        'voxel_spacing_mm': [
            float(value)
            for value in args.voxel_spacing_mm
        ],

        'per_label_metrics':
            per_label,

        'rules': {
            'original_hd95':
                (
                    '6-connected surfaces; concatenate both '
                    'directional surface distances; pooled 95th percentile'
                ),

            'original_assd':
                (
                    '6-connected surfaces; mean of the two '
                    'directional surface-distance means'
                ),

            'audit_connectivity':
                (
                    'both 6-connected and 26-connected surfaces '
                    'are reported'
                ),

            'audit_hd95':
                (
                    'both pooled symmetric P95 and max of the '
                    'two directional P95 values are reported'
                ),

            'audit_assd':
                (
                    'both equal-weight directional means and '
                    'pooled symmetric surface-distance mean are reported'
                ),

            'empty_roi':
                (
                    'one-sided empty receives the physical '
                    'full-grid diagonal; empty-empty is not applicable'
                ),

            'jacobian':
                (
                    'J<0 percentage is reported for paper comparison; '
                    'J<=0 is retained separately'
                ),

            'std':
                'population standard deviation (ddof=0)',
        },
    }

    print('')
    print('--- Summary ---')

    print(
        f'Cases: {len(case_means)}'
    )

    print(
        f'Case Dice mean +- std: '
        f'{case_means.mean():.4f} '
        f'+- {case_means.std():.4f}'
    )

    print(
        f'ROI Dice mean +- std:  '
        f'{roi_rows.mean():.4f} '
        f'+- {roi_rows.std():.4f}'
    )

    print(
        f'Original HD95 mean +- std: '
        f'{case_hd95.mean():.4f} '
        f'+- {case_hd95.std():.4f} mm'
    )

    print(
        f'Original ASSD mean +- std: '
        f'{case_assd.mean():.4f} '
        f'+- {case_assd.std():.4f} mm'
    )

    print(
        f'J<0 % mean +- std: '
        f'{jac_negative_percent.mean():.4f} '
        f'+- {jac_negative_percent.std():.4f}'
    )

    print(
        f'J<=0 % mean +- std: '
        f'{jac_nonpos_percent.mean():.4f} '
        f'+- {jac_nonpos_percent.std():.4f}'
    )

    print('')
    print('--- Surface Metric Audit ---')

    for key in SURFACE_METRIC_KEYS:
        case_values = surface_case_values[key]
        roi_values = surface_roi_values[key]

        print(
            f'{key}:'
        )

        print(
            f'    case-level = '
            f'{case_values.mean():.4f} '
            f'+- {case_values.std():.4f}'
        )

        print(
            f'    ROI-level  = '
            f'{roi_values.mean():.4f} '
            f'+- {roi_values.std():.4f}'
        )

    if args.save_csv:
        output_parent = os.path.dirname(
            os.path.abspath(args.save_csv)
        )

        os.makedirs(
            output_parent,
            exist_ok=True,
        )

        with open(
            args.save_csv,
            'w',
            newline='',
            encoding='utf-8',
        ) as f:
            fields = list(
                case_records[0].keys()
            )

            writer = csv.DictWriter(
                f,
                fieldnames=fields,
            )

            writer.writeheader()
            writer.writerows(
                case_records
            )

        output_stem, _ = os.path.splitext(
            args.save_csv
        )

        roi_csv = (
            output_stem
            + '_roi.csv'
        )

        with open(
            roi_csv,
            'w',
            newline='',
            encoding='utf-8',
        ) as f:
            fields = list(
                roi_records[0].keys()
            )

            writer = csv.DictWriter(
                f,
                fieldnames=fields,
            )

            writer.writeheader()
            writer.writerows(
                roi_records
            )

        label_csv = (
            output_stem
            + '_labels.csv'
        )

        with open(
            label_csv,
            'w',
            newline='',
            encoding='utf-8',
        ) as f:
            fields = [
                'label',
                'dice_mean',
                'dice_std',
                'hd95_mm_mean',
                'hd95_mm_std',
                'assd_mm_mean',
                'assd_mm_std',
                'case_count',
                'penalized_case_count',
            ]

            writer = csv.DictWriter(
                f,
                fieldnames=fields,
            )

            writer.writeheader()

            for label in labels:
                writer.writerow({
                    'label':
                        int(label),
                    **per_label[str(int(label))],
                })

        summary_json = (
            output_stem
            + '_summary.json'
        )

        with open(
            summary_json,
            'w',
            encoding='utf-8',
        ) as f:
            json.dump(
                summary,
                f,
                ensure_ascii=False,
                indent=2,
            )

        print(
            f'Saved CSV: {args.save_csv}'
        )

        print(
            f'Saved ROI CSV: {roi_csv}'
        )

        print(
            f'Saved label CSV: {label_csv}'
        )

        print(
            f'Saved summary JSON: {summary_json}'
        )


if __name__ == '__main__':
    main()
