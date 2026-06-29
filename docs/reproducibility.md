# Reproducibility Guide

## Clone With Large Files

```bash
git lfs install
git clone https://github.com/tianluyao1024/PlantEssentialGenePredictor.git
cd PlantEssentialGenePredictor
git lfs pull
```

## Predict From Released Features

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

## Run Web App

```bash
streamlit run webapp/app.py
```

## Re-train Models

Training scripts are under `scripts/training/`. They preserve the original project logic and may require local paths to raw or intermediate feature sources to be edited before rerunning.

Primary scripts:

- `train_ath_three_labelsets_common6751_fixed_split.py`
- `train_rice_strict399_N4_OOF_threshold_bootstrap.py`
- `train_joint_ath2601_rice_strict399_common6751.py`
