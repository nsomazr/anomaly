"""
artifacts.py
============
Save and load all deployment artifacts.

Notebook mapping
----------------
Step 12 — Save artifacts → save_all()
Step 13 — Load artifacts → load_all()
"""

import os
import joblib
import pandas as pd
from config import MODELS_DIR, ARTIFACT_NAMES


def save_all(
    models_dir: str = MODELS_DIR,
    xgb_model=None,
    rf_model=None,
    feature_cols: list = None,
    encoders: dict = None,
    thresholds: dict = None,
    icd10_medians: pd.Series = None,
    hospital_medians: pd.Series = None,
    provider_counts: pd.Series = None,
    hospital_counts: pd.Series = None,
) -> None:
    """
    Save all artifacts required for production inference.

    All nine artifacts must be present for the prediction pipeline
    to function correctly. Missing any one will cause inference to fail.
    """
    os.makedirs(models_dir, exist_ok=True)

    artifacts = {
        'xgb_model'       : xgb_model,
        'rf_model'        : rf_model,
        'feature_cols'    : feature_cols,
        'encoders'        : encoders,
        'thresholds'      : thresholds,
        'icd10_medians'   : icd10_medians,
        'hospital_medians': hospital_medians,
        'provider_counts' : provider_counts,
        'hospital_counts' : hospital_counts,
    }

    for key, obj in artifacts.items():
        if obj is None:
            raise ValueError(f'Artifact "{key}" is None — cannot save.')
        path = os.path.join(models_dir, ARTIFACT_NAMES[key])
        joblib.dump(obj, path)

    print(f'=== ARTIFACTS SAVED TO {models_dir} ===')
    for key in artifacts:
        fname = ARTIFACT_NAMES[key]
        size  = os.path.getsize(os.path.join(models_dir, fname))
        print(f'  {fname:<35s}  {size/1024:>7.1f} KB')

    print('\n=== DEPLOYMENT THRESHOLDS ===')
    for model_name, thr in thresholds.items():
        print(f'  {model_name:<20s}: {thr:.4f}')


def load_all(models_dir: str = MODELS_DIR, verbose: bool = False) -> dict:
    """
    Load all artifacts required for production inference.

    Returns
    -------
    dict with keys matching ARTIFACT_NAMES in config.py
    """
    loaded = {}
    for key, fname in ARTIFACT_NAMES.items():
        path = os.path.join(models_dir, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(f'Artifact not found: {path}')
        loaded[key] = joblib.load(path)

    if verbose:
        print(f'All {len(loaded)} artifacts loaded from {models_dir}')
        print(f'  XGBoost threshold    : {loaded["thresholds"]["xgboost"]:.4f}')
        print(f'  RandomForest threshold: {loaded["thresholds"]["random_forest"]:.4f}')
    return loaded
