# IXI training and evaluation
Run commands from the repository
root. The APE entry remains `ape_ixi.py` at the root.

## Data and external encoder

Provide the processed IXI data as `Train/*.pkl`, `Val/*.pkl`, `Test/*.pkl` and
`atlas.pkl`. Each pickle contains `(image, segmentation)` with intensities
already normalized as in the training data. Stored arrays are HWD
`(160,192,224)`; the APE loader transposes them to DHW `(224,160,192)`.
Use the same atlas, subject split, label IDs and preprocessing for comparisons.


Replace these example paths with your own:

```bash
DATA=/path/to/IXI_data
SAM2=/path/to/sam2
MEDSAM=/path/to/MedSAM2_latest.pt
CFG=configs/sam2.1/sam2.1_hiera_t.yaml
```

## 1. Plan-D APE pretraining

```bash
python ape_ixi.py \
  --train-dir "$DATA/Train" \
  --sam2-root "$SAM2" --sam2-config "$CFG" --sam2-checkpoint "$MEDSAM" \
  --output-dir runs/ixi_ape --epochs 100 --device cuda
```

Historical settings: 128 channels, prototype counts `[24,10,6]`, temperature
`0.3`, batch size 1, learning rate `5e-6`, AdamW with weight decay `1e-5`, cosine
scheduling and seed 42. Anatomical supervision uses 11/5/3 label IDs at the three
levels.

Weights, logs and `run_config.json` are written under `--output-dir`. The best
filename is `ape_pretrained_plan_d_c128_best.pth`. 
## 2. Registration

```bash
APE=runs/ixi_ape/ape_pretrained_plan_d_c128_best.pth
python ixi__eapr/train.py "$DATA/Train" \
  --model-dir runs/ixi_registration --gpu 0 \
  --use-ape --ape-ckpt "$APE" --ape-k-list 24,10,6 --ape-channels 128 \
  --ape-use-eadf --sam2-repo "$SAM2" --sam2-cfg "$CFG" --sam2-ckpt "$MEDSAM" \
  --batch-size 1 --epochs 300 --steps-per-epoch 400 --lr 5e-6 --grad-weight 1.5 \
  --atlas-file "$DATA/atlas.pkl" \
  --val-dir "$DATA/Val" --val-atlas-file "$DATA/atlas.pkl" --val-every 1
```

This is a new-training example, not the final historical fine-tuning run. 




## 3. Test against the fixed atlas

```bash
python ixi__eapr/eval_pkl.py \
  --test-dir "$DATA/Test" --atlas-file "$DATA/atlas.pkl" \
  --model runs/models_ape_k24_c128_eadf_finetune2_lr5e6_g15/best_val.pt \
  --gpu 0 --use-ape --ape-ckpt "$APE" --ape-k-list 24,10,6 \
  --ape-channels 128 --ape-use-eadf \
  --sam2-repo "$SAM2" --sam2-cfg "$CFG" --sam2-ckpt "$MEDSAM" \
  --voxel-spacing-mm 1 1 1 --save-csv runs/ixi_test.csv
```

Use the actual processed voxel spacing for HD95/ASSD. Evaluation uses 30
predefined label IDs unless overridden by `--labels`; multiple surface metric
definitions are named explicitly in the output. Keep the same APE and
fusion/refinement flags as training. Parameter mismatches now raise an error;
deterministic spatial-grid buffers are rebuilt. Additional half/context
refinement and prototype attention were not enabled in the historical command.

