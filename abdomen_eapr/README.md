# AbdomenCTCT training and evaluation

The two public training entries are `train_ape_v31.py` and
`train_registration.py`. Run them as modules from the repository root. Core
APE, encoder, registration and data modules are required dependencies.
Registration reuses components from `../ixi__eapr/voxelmorph`; distribute both folders.

## Training configuration

The v3.1 APE prototype and supervision settings are:

| Stage | Total slots | Active semantic slots | Inactive discovery slots | Supervision label IDs |
| --- | --- | --- | --- | --- |
| Fine | 8 | 6 | 2 | 4, 5, 10, 11, 12, 13 |
| Middle | 6 | 5 | 1 | 1, 2, 3, 6, 11 |
| Coarse | 4 | 4 | 0 | 1, 2, 3, 6 |

There are 10 distinct supervised labels; registration evaluates all 13 organs.
Defaults are 128 channels and temperature 0.2.

## 1. Data preparation

Start from the 30 labeled Learn2Reg AbdomenCTCT cases, shape `(192,160,256)` XYZ
and spacing `(2,2,2)` mm. The historical center crop removes 16 slices from each
end of Z. Intensities are clipped to `[-1000,1000]` HU and mapped to `[0,1]`.
Output is `(image, label, body_mask)` in HWD `(160,192,224)`; loaders convert it
to DHW `(224,160,192)`.

## 2. APE v3.1

```bash
SAM2=/path/to/sam2
MEDSAM=/path/to/MedSAM2_latest.pt
CFG=configs/sam2.1/sam2.1_hiera_t.yaml
APE_OUT=runs/abdomen_ape_v31
python -m abdomen_eapr.train_ape_v31 \
  --train-dir "$DATA/Train" --val-dir "$DATA/Val" --output-dir "$APE_OUT" \
  --sam2-root "$SAM2" --sam2-config "$CFG" --sam2-checkpoint "$MEDSAM" \
  --ape-channels 128 --temperature 0.2 --initial-slot-bias -2.0 \
  --epochs 400 --batch-size 1 --lr 1e-4 --weight-decay 1e-5 \
  --w-positive 0.5 --w-local-negative 0.25 \
  --w-global-hard-negative 0.25 --w-global-random-negative 0.10 \
  --hard-negative-radius 2 --local-negative-ratio 3.0 \
  --global-hard-negative-ratio 3.0 --global-random-negative-ratio 3.0 \
  --eval-every 20 --save-every 20 --vis-every 20 --early-stop-patience 10 \
  --chunk-size 16 --num-workers 0 --seed 42 --device cuda
```

The selected checkpoint is `$APE_OUT/checkpoints/ape_v31_best_fine_fixed.pth`,
chosen by the fine-stage fixed-target validation metric. 
## 3. registration

```bash
APE="$APE_OUT/checkpoints/ape_v31_best_fine_fixed.pth"
REG_OUT=runs/abdomen_registration
python -m abdomen_eapr.train_registration \
  --ape-checkpoint "$APE" \
  --train-dir "$DATA/Train" --val-dir "$DATA/Val" \
  --train-pairs "$DATA/manifests/train_pairs.json" \
  --val-pairs "$DATA/manifests/val_pairs.json" --output-dir "$REG_OUT" \
  --sam2-root "$SAM2" --sam2-config "$CFG" --sam2-checkpoint "$MEDSAM" \
  --epochs 20 --batch-size 1 --lr 1e-4 --ncc-window 9 \
  --w-ncc 1.0 --w-grad 0.5 --w-body 0.05 \
  --eval-every 1 --save-every 1 --chunk-size 16 --num-workers 0 --seed 42 --device cuda
```

 Settings are
recorded in checkpoints and `run_config.json`. `--resume` restores compatible
registration weights and optimizer state. Evaluation infers architecture from
the checkpoint.

## 4. Evaluation

After selecting a checkpoint using Val, evaluate the entire fixed Test manifest:

```bash
REG="$REG_OUT/checkpoints/registration_best.pth"
python -m abdomen_eapr.train_registration --evaluate-only \
  --ape-checkpoint "$APE" --resume "$REG" \
  --val-dir "$DATA/Test" --val-pairs "$DATA/manifests/test_pairs.json" \
  --output-dir runs/abdomen_test \
  --sam2-root "$SAM2" --sam2-config "$CFG" --sam2-checkpoint "$MEDSAM" \
  --voxel-spacing-mm 2 2 2 --device cuda
```

The evaluator reuses `--val-*` names for any evaluation split. It exports
`evaluation.json`, `evaluation_pairs.csv`, `evaluation_pair_organs.csv` and
`evaluation_organs.csv`: Dice, HD95, ASSD and negative-Jacobian percentage,
including population mean/std across pair-level 13-ROI means. 

## Historical 0.592193 run

Recorded Test Dice **0.5921930687217105** belongs to `registration_epoch0070.pth`
in `abdomen_registration_runs/identity_easy_v3_wgrad05_consistency_baseline/checkpoints/`.


