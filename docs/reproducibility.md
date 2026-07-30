# Reproducibility Guide

## Clone the code repository

```bash
git clone https://github.com/tianluyao1024/PlantEssentialGenePredictor.git
cd PlantEssentialGenePredictor
```

## Obtain the large model/data artifact

Download the large model/data archive and independent-validation supplement
from Zenodo version v1.2.3:

```text
Version DOI: 10.5281/zenodo.21387076
```

Extract the large artifact into the repository root. After extraction, these paths should exist:

```text
models/
data/processed_features/
data/labels/
predictions/
```

## Predict from released features

```bash
python scripts/prediction/predict_from_processed_features.py \
  --features data/processed_features/rice_common6751_all_genes.npz \
  --model rice_single \
  --out predictions/rice_single_rerun.tsv
```

```bash
python scripts/prediction/predict_from_processed_features.py \
  --features data/processed_features/arabidopsis_unknown20460_common6751_sequence_plm_imputed_input.npz \
  --model joint \
  --out predictions/arabidopsis_unknown_joint_rerun.tsv
```

## Run the web app

```bash
pip install -r requirements.txt
streamlit run webapp/app.py
```

## Re-train models

Training scripts are under `scripts/training/`. They preserve the original project logic and may require local paths to raw or intermediate feature sources to be edited before rerunning.

Primary scripts:

- `train_ath_three_labelsets_common6751_fixed_split.py`
- `train_rice_strict399_N4_OOF_threshold_bootstrap.py`
- `train_joint_ath2601_rice_strict399_common6751.py`

