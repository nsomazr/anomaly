"""
features.py
===========
Feature engineering and categorical encoding.

Notebook mapping
----------------
Step 4  — Feature Engineering  → build_temporal_features(), build_frequency_features(),
                                  build_rolling_provider_feature(), build_cost_features()
Step 5  — Encode Categoricals  → encode_categoricals()
Step 6b — Leakage-safe refs    → build_train_references(), apply_train_refs()

Design rule: ALL aggregation-based features (medians, counts) are computed
from the TRAINING set only and looked up for val/test. This mirrors exactly
what the production inference function does for each new incoming claim.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from config import CATEGORICAL_COLS, MALARIA_MONTHS, ROLLING_WINDOW_DAYS


# ── Temporal features ────────────────────────────────────────────────────────

def build_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add time-based features derived from each row's own dates.
    No aggregation → no leakage risk.
    """
    df = df.copy()
    df['claim_age_days']        = df['lag_days']
    df['submission_month']      = df['submission_date'].dt.month
    df['submission_dayofweek']  = df['submission_date'].dt.dayofweek   # 0=Mon
    df['submission_quarter']    = df['submission_date'].dt.quarter
    df['is_weekend_submission'] = (df['submission_dayofweek'] >= 5).astype(int)
    df['is_malaria_season']     = df['submission_month'].isin(MALARIA_MONTHS).astype(int)
    df['log_claim_amount']      = np.log1p(df['claimed_amount_tzs'])
    return df


# ── Frequency features ────────────────────────────────────────────────────────

def build_frequency_features(
    df: pd.DataFrame,
    provider_ref: pd.Series = None,
    hospital_ref: pd.Series = None
) -> tuple[pd.DataFrame, dict]:
    """
    Add provider and hospital total-claim-count features.

    If reference Series are provided (val/test/production), values are
    looked up from those references. Unknown IDs default to 1.
    If not provided (training), counts are computed from df itself.

    Returns
    -------
    (df_with_features, reference_dict)
    """
    df = df.copy()

    if provider_ref is None:
        provider_ref = df.groupby('provider_id')['claim_id'].count()
    if hospital_ref is None:
        hospital_ref = df.groupby('hospital_id')['claim_id'].count()

    df['provider_claim_count'] = (
        df['provider_id'].map(provider_ref).fillna(1).astype(int)
    )
    df['hospital_claim_count'] = (
        df['hospital_id'].map(hospital_ref).fillna(1).astype(int)
    )

    ref = {'provider_counts': provider_ref, 'hospital_counts': hospital_ref}
    return df, ref


def build_rolling_provider_feature(df: pd.DataFrame,
                                    window_days: int = ROLLING_WINDOW_DAYS
                                    ) -> pd.DataFrame:
    """
    True rolling window: count how many claims this provider submitted
    in the `window_days` days BEFORE each claim's submission_date.

    FIX v2: Previous implementation used cumcount() — a cumulative total,
    not a time window. This implementation computes a genuine 30-day window.

    Parameters
    ----------
    df          : DataFrame with 'provider_id' and 'submission_date'
    window_days : lookback window in days (default 30)
    """
    df = df.sort_values('submission_date').copy()
    rolling_counts = []

    for _, group in df.groupby('provider_id'):
        dates  = group['submission_date'].values
        window = np.timedelta64(window_days, 'D')
        counts = []
        for i, d in enumerate(dates):
            count = int(np.sum((dates[:i] >= d - window) & (dates[:i] < d)))
            counts.append(count)
        rolling_counts.extend(zip(group.index, counts))

    df['rolling_30d_provider'] = pd.Series(dict(rolling_counts))
    return df


# ── Cost-deviation features ───────────────────────────────────────────────────

def build_cost_features(
    df: pd.DataFrame,
    icd10_ref: pd.Series = None,
    hospital_ref: pd.Series = None
) -> tuple[pd.DataFrame, dict]:
    """
    Add cost-deviation features comparing each claim's amount to reference medians.

    If icd10_ref / hospital_ref are provided, they are used as-is (training stats).
    Otherwise they are computed from df (only correct for training set).

    Returns
    -------
    (df_with_features, reference_dict)
    """
    df = df.copy()

    if icd10_ref is None:
        icd10_ref = df.groupby('icd10_code')['claimed_amount_tzs'].median()
    if hospital_ref is None:
        hospital_ref = df.groupby('hospital_id')['claimed_amount_tzs'].median()

    icd_fallback  = float(icd10_ref.median())
    hosp_fallback = float(hospital_ref.median())

    df['icd10_median_cost']    = df['icd10_code'].map(icd10_ref).fillna(icd_fallback)
    df['hospital_median_cost'] = df['hospital_id'].map(hospital_ref).fillna(hosp_fallback)

    df['cost_deviation_pct'] = (
        (df['claimed_amount_tzs'] - df['icd10_median_cost'])
        / (df['icd10_median_cost'] + 1e-6) * 100
    ).round(2)

    df['amount_vs_hospital_median'] = (
        df['claimed_amount_tzs'] / (df['hospital_median_cost'] + 1e-6)
    ).round(4)

    ref = {'icd10_medians': icd10_ref, 'hospital_medians': hospital_ref}
    return df, ref


# ── Leakage-safe reference stats ──────────────────────────────────────────────

def build_train_references(train_df: pd.DataFrame) -> dict:
    """
    Compute all reference statistics from the training split only.
    These are then applied to val/test via apply_train_refs().

    Returns a dict with keys matching ARTIFACT_NAMES in config.py.
    """
    return {
        'provider_counts' : train_df.groupby('provider_id')['claim_id'].count(),
        'hospital_counts' : train_df.groupby('hospital_id')['claim_id'].count(),
        'icd10_medians'   : train_df.groupby('icd10_code')['claimed_amount_tzs'].median(),
        'hospital_medians': train_df.groupby('hospital_id')['claimed_amount_tzs'].median(),
    }


def apply_train_refs(df: pd.DataFrame, refs: dict) -> pd.DataFrame:
    """
    Apply training-set reference statistics to any split (val, test, or new data).

    Unknown IDs/codes fall back to training-set medians.
    This function mirrors exactly what prepare_sample() does for a single claim.
    """
    df = df.copy()

    # Frequency lookups
    df['provider_claim_count'] = (
        df['provider_id'].map(refs['provider_counts']).fillna(1).astype(int)
    )
    df['hospital_claim_count'] = (
        df['hospital_id'].map(refs['hospital_counts']).fillna(1).astype(int)
    )

    # Cost lookups
    icd_fallback  = float(refs['icd10_medians'].median())
    hosp_fallback = float(refs['hospital_medians'].median())

    df['icd10_median_cost']    = df['icd10_code'].map(refs['icd10_medians']).fillna(icd_fallback)
    df['hospital_median_cost'] = df['hospital_id'].map(refs['hospital_medians']).fillna(hosp_fallback)

    df['cost_deviation_pct'] = (
        (df['claimed_amount_tzs'] - df['icd10_median_cost'])
        / (df['icd10_median_cost'] + 1e-6) * 100
    ).round(2)

    df['amount_vs_hospital_median'] = (
        df['claimed_amount_tzs'] / (df['hospital_median_cost'] + 1e-6)
    ).round(4)

    return df


# ── Categorical encoding ──────────────────────────────────────────────────────

def encode_categoricals(
    df: pd.DataFrame,
    encoders: dict = None,
    fit: bool = True
) -> tuple[pd.DataFrame, dict]:
    """
    Encode categorical columns with LabelEncoder.

    Parameters
    ----------
    df       : input DataFrame
    encoders : pre-fitted encoder dict (None → fit new encoders from df)
    fit      : if True, fit new encoders; if False, transform using provided encoders

    Returns
    -------
    (df_encoded, encoders_dict)
    """
    df = df.copy()
    if encoders is None:
        encoders = {}

    for raw_col, enc_col in CATEGORICAL_COLS.items():
        if fit or raw_col not in encoders:
            le = LabelEncoder()
            df[enc_col] = le.fit_transform(df[raw_col].astype(str))
            encoders[raw_col] = le
            print(f'  {raw_col:20s} → {enc_col}  | classes: {list(le.classes_)}')
        else:
            le = encoders[raw_col]
            values = df[raw_col].astype(str)
            df[enc_col] = values.apply(
                lambda v: int(le.transform([v])[0]) if v in le.classes_ else 0
            )

    return df, encoders


# ── Full feature pipeline ─────────────────────────────────────────────────────

def build_all_features(
    df: pd.DataFrame,
    refs: dict = None,
    encoders: dict = None,
    fit_encoders: bool = True
) -> tuple[pd.DataFrame, dict, dict]:
    """
    Run the complete feature engineering pipeline on a DataFrame.

    Parameters
    ----------
    df            : cleaned claims DataFrame (must have lag_days column)
    refs          : reference stats dict (None → computed from df)
    encoders      : pre-fitted encoder dict (None → fit from df)
    fit_encoders  : fit new encoders if True

    Returns
    -------
    (df_features, refs, encoders)
    """
    df = build_temporal_features(df)
    df, freq_ref  = build_frequency_features(
        df,
        provider_ref=refs['provider_counts']  if refs else None,
        hospital_ref=refs['hospital_counts']  if refs else None,
    )
    df, cost_ref  = build_cost_features(
        df,
        icd10_ref   =refs['icd10_medians']    if refs else None,
        hospital_ref=refs['hospital_medians'] if refs else None,
    )
    df = build_rolling_provider_feature(df)

    if refs is None:
        refs = {**freq_ref, **{
            'icd10_medians'   : cost_ref['icd10_medians'],
            'hospital_medians': cost_ref['hospital_medians'],
        }}

    df, encoders = encode_categoricals(df, encoders=encoders, fit=fit_encoders)
    return df, refs, encoders
