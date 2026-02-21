# Data Directory: Train-Ready Splits

This directory holds the **preprocessed, patient-stratified CSV splits** required by the training pipeline.

## Required Files

| File | Description |
|------|-------------|
| `train_patient_stratified.csv` | Training set |
| `val_patient_stratified.csv` | Validation set (threshold selection & early stopping) |
| `test_patient_stratified.csv` | Held-out test set (final evaluation) |

## Format Specification

- **File type**: CSV (comma-separated)
- **First column**: `patient_id`
- **Target column**: `mortality` (binary: 0 = survived, 1 = deceased)
- **Preprocessing**: Data must be preprocessed before placement here. Use the companion data preprocessing repository (see main [README](../../README.md#data-preprocessing)) to generate these files from raw eICU data.

## Important Notes

1. **Do NOT commit data files to Git.** All `.csv`, `.parquet`, and `.pkl` files in this directory are excluded by `.gitignore` in compliance with the Data Use Agreement (DUA).
2. **Consistent preprocessing is required.** If you are running multiple experiments or comparing across projects, ensure the same preprocessing pipeline is applied to all splits.
