"""
config.py
=========
Central configuration for the ZHSF anomaly-detection pipeline.

All paths, constants, and feature definitions live here.
Every other module imports from this file — nothing is hardcoded elsewhere.
"""

import os

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(BASE_DIR, 'data', 'raw')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
LOGS_DIR   = os.path.join(BASE_DIR, 'logs')
FIGURES_DIR= os.path.join(BASE_DIR, 'figures')

for d in [DATA_DIR, MODELS_DIR, LOGS_DIR, FIGURES_DIR]:
    os.makedirs(d, exist_ok=True)

# ── Feature columns (single source of truth) ───────────────────────────────
# Order matters: models are sensitive to column order.
FEATURE_COLS = [
    # Group 1: Temporal signals
    'claim_age_days',           # days from service to submission
    'submission_month',         # 1–12
    'submission_dayofweek',     # 0=Mon … 6=Sun
    'submission_quarter',       # 1–4
    'is_weekend_submission',    # 1 if submitted Sat/Sun
    'is_malaria_season',        # 1 if April, May, Oct, Nov
    'log_claim_amount',         # log1p(claimed_amount_tzs)
    # Group 2: Provider / hospital frequency
    'provider_claim_count',     # total claims by this provider (train reference)
    'hospital_claim_count',     # total claims at this hospital (train reference)
    'rolling_30d_provider',     # claims by provider in 30-day window before this claim
    # Group 3: Cost-deviation signals
    'icd10_median_cost',        # reference median cost for this ICD-10 code
    'cost_deviation_pct',       # % above/below ICD-10 median
    'amount_vs_hospital_median',# ratio: claimed / hospital median
    # Group 4: Encoded categoricals
    'facility_type_enc',
    'plan_type_enc',
    'service_type_enc',
    'patient_gender_enc',
    'patient_district_enc',
]

# ── Categorical columns → encoded column name ──────────────────────────────
CATEGORICAL_COLS = {
    'facility_type'   : 'facility_type_enc',
    'plan_type'       : 'plan_type_enc',
    'service_type'    : 'service_type_enc',
    'patient_gender'  : 'patient_gender_enc',
    'patient_district': 'patient_district_enc',
}

# ── Domain constants ────────────────────────────────────────────────────────
MALARIA_MONTHS   = [4, 5, 10, 11]   # Zanzibar malaria seasons
ROLLING_WINDOW_DAYS = 30            # window for rolling_30d_provider

# ── Train / Val / Test split ratios ────────────────────────────────────────
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15                  # remaining 15% = test

# ── Reproducibility ────────────────────────────────────────────────────────
RANDOM_STATE = 42

# ── Model artifact filenames ────────────────────────────────────────────────
ARTIFACT_NAMES = {
    'xgb_model'       : 'xgb_anomaly_detector.pkl',
    'rf_model'        : 'rf_anomaly_detector.pkl',
    'feature_cols'    : 'feature_cols.pkl',
    'encoders'        : 'encoders.pkl',
    'thresholds'      : 'thresholds.pkl',
    'icd10_medians'   : 'icd10_medians.pkl',
    'hospital_medians': 'hospital_medians.pkl',
    'provider_counts' : 'provider_counts.pkl',
    'hospital_counts' : 'hospital_counts.pkl',
}
