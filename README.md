# EAPR-Reg: IXI and AbdomenCTCT

Research code for the IXI and AbdomenCTCT experiments. Dataset-specific training
instructions and historical checkpoint provenance are documented separately.

| Dataset | APE pretraining | Registration and evaluation |
| --- | --- | --- |
| IXI | `ape_ixi.py` | `ixi__eapr/train.py`, `ixi__eapr/eval_pkl.py` |
| AbdomenCTCT | `abdomen_eapr/train_ape_v31.py` | `abdomen_eapr/train_registration.py` |

Start with the [IXI instructions](ixi__eapr/README.md) or the
[AbdomenCTCT instructions](abdomen_eapr/README.md). Run commands from this repository
root; 

## Setup

Use a CUDA-capable PyTorch environment compatible with your external SAM2/MedSAM2
installation. Install PyTorch for your CUDA version first, then:

```bash
python -m pip install -r requirements.txt
```

Install SAM2 separately and provide its repository, Hydra configuration and the
MedSAM2 checkpoint through the CLI. Weights and medical images are not bundled.
The Python code uses Python 3.9+ syntax; the installed SAM2 version can require a
newer Python/PyTorch version. `requirements.txt` lists direct dependencies, not a
lock file of the historical GPU environment.

## Reproduction status

The release provides training and evaluation entry points for both datasets,
with configuration and checkpoint instructions in each dataset README.

The recorded abdomen Test Dice is **0.5921930687 **. 
This source release
does not include the final server checkpoints. New training commands do not
guarantee the recorded score. The final IXI run was a warm-started fine-tune;
see its README for the required preceding checkpoint.

## Checks and source package

```bash
python -m unittest ixi__eapr.test_evaluation_metrics abdomen_eapr.test_clean_pipeline tools.test_release
python tools/build_release.py
```

The packager uses an explicit source allowlist and writes `dist/eapr-reg-source.zip`
and `dist/release_manifest.json` with SHA-256 hashes. Datasets, weights, logs, old
experiment trees and paper figures remain local.
