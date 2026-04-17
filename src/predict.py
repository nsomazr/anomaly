"""
predict.py
==========
Production inference: convert a raw claim dict into a model decision.

Notebook mapping
----------------
Step 13 — Smoke Test → prepare_sample(), predict_claim()

This module is designed to be importable by a REST API handler or any
production scoring service. It loads artifacts once at import time and
exposes a simple predict_claim() function.

Usage
-----
    from src.predict import predict_claim

    result = predict_claim({
        'service_date'      : '2023-11-05',
        'submission_date'   : '2023-11-12',
        'claimed_amount_tzs': 35000,
        'facility_type'     : 'Public',
        'plan_type'         : 'Standard',
        'service_type'      : 'Outpatient',
        'patient_gender'    : 'Female',
        'patient_district'  : 'Urban West',
        'icd10_code'        : 'I10',
        'hospital_id'       : 'HOS011',
        'provider_id'       : 'PRV0036',
    })
    # → {'xgb_score': 0.012, 'rf_score': 0.008, 'xgb_flag': False, 'rf_flag': False}
"""

import numpy as np
import pandas as pd
from artifacts import load_all
from config import CATEGORICAL_COLS, MALARIA_MONTHS, MODELS_DIR
# ── Load artifacts once at import time ─────────────────────────────────────
_artifacts = load_all(MODELS_DIR)

_xgb_model    = _artifacts['xgb_model']
_rf_model     = _artifacts['rf_model']
_feature_cols = _artifacts['feature_cols']
_encoders     = _artifacts['encoders']
_thresholds   = _artifacts['thresholds']
_icd10_ref    = _artifacts['icd10_medians']
_hospital_ref = _artifacts['hospital_medians']
_provider_ref = _artifacts['provider_counts']
_hospital_cnt = _artifacts['hospital_counts']


def prepare_sample(raw: dict) -> pd.DataFrame:
    """
    Convert a raw claim dict into a model-ready feature row.

    This function mirrors exactly what the training notebook does for each
    row in the val/test sets. Any change here must also be reflected in
    features.py → apply_train_refs() to keep training and inference in sync.

    Expected keys in raw
    --------------------
    service_date, submission_date, claimed_amount_tzs,
    facility_type, plan_type, service_type,
    patient_gender, patient_district,
    icd10_code, hospital_id, provider_id

    Returns
    -------
    pd.DataFrame with exactly the columns in _feature_cols (same order)
    """
    r      = raw.copy()
    svc    = pd.Timestamp(r['service_date'])
    sub    = pd.Timestamp(r['submission_date'])
    lag    = max(0, (sub - svc).days)
    amount = float(r['claimed_amount_tzs'])

    # ── Temporal ────────────────────────────────────────────────────────────
    row = {
        'claim_age_days'       : lag,
        'submission_month'     : sub.month,
        'submission_dayofweek' : sub.dayofweek,
        'submission_quarter'   : sub.quarter,
        'is_weekend_submission': int(sub.dayofweek >= 5),
        'is_malaria_season'    : int(sub.month in MALARIA_MONTHS),
        'log_claim_amount'     : np.log1p(amount),
    }

    # ── Frequency lookups ────────────────────────────────────────────────────
    # hospital uses _hospital_cnt, not _provider_ref (v1 bug)
    row['provider_claim_count'] = int(_provider_ref.get(r['provider_id'], 1))
    row['hospital_claim_count'] = int(_hospital_cnt.get(r['hospital_id'], 1))
    row['rolling_30d_provider'] = 0   # unknown for a single new claim

    # ── Cost-deviation lookups ───────────────────────────────────────────────
    icd      = str(r.get('icd10_code', 'UNKNOWN') or 'UNKNOWN')
    icd_med  = float(_icd10_ref.get(icd,           float(_icd10_ref.median())))
    hosp_med = float(_hospital_ref.get(r['hospital_id'], float(_hospital_ref.median())))

    row['icd10_median_cost']         = icd_med
    row['cost_deviation_pct']        = round((amount - icd_med) / (icd_med + 1e-6) * 100, 2)
    row['amount_vs_hospital_median'] = round(amount / (hosp_med + 1e-6), 4)

    # ── Categorical encoding ─────────────────────────────────────────────────
    for raw_col, enc_col in CATEGORICAL_COLS.items():
        le  = _encoders[raw_col]
        val = str(r.get(raw_col, ''))
        row[enc_col] = int(le.transform([val])[0]) if val in le.classes_ else 0

    return pd.DataFrame([row])[_feature_cols]


def predict_claim(raw: dict, verbose: bool = True) -> dict:
    """
    Score a single raw claim dict with both models.

    Parameters
    ----------
    raw     : dict with raw claim fields (see prepare_sample docstring)
    verbose : print result to stdout if True

    Returns
    -------
    dict with keys: xgb_score, rf_score, xgb_flag, rf_flag
    """
    X = prepare_sample(raw).values

    xgb_score = float(_xgb_model.predict_proba(X)[0, 1])
    rf_score  = float(_rf_model.predict_proba(X)[0, 1])
    xgb_flag  = xgb_score >= _thresholds['xgboost']
    rf_flag   = rf_score  >= _thresholds['random_forest']

    if verbose:
        print(f'XGBoost      score: {xgb_score:.4f}  '
              f'threshold: {_thresholds["xgboost"]:.4f}  '
              f'→ {"ANOMALY" if xgb_flag else "Normal"}')
        print(f'RandomForest score: {rf_score:.4f}  '
              f'threshold: {_thresholds["random_forest"]:.4f}  '
              f'→ {"ANOMALY" if rf_flag else "Normal"}')

    return dict(xgb_score=xgb_score, rf_score=rf_score,
                xgb_flag=xgb_flag,   rf_flag=rf_flag)
