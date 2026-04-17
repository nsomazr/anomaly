"""
data.py
=======
Data loading, cleaning, and splitting functions.

Notebook mapping
----------------
Step 2 — Load Data          → load_raw_data()
Step 3 — Data Quality/Clean → clean_claims()
Step 6 — Split              → chronological_split()
"""

import pandas as pd
import numpy as np
from config import DATA_DIR, TRAIN_RATIO, VAL_RATIO


# ── Load ────────────────────────────────────────────────────────────────────

def load_raw_data(data_dir: str = DATA_DIR) -> dict[str, pd.DataFrame]:
    """
    Load all four ZHSF CSV tables.

    Returns
    -------
    dict with keys: 'patients', 'members', 'claims', 'payments'
    """
    patients = pd.read_csv(
        f'{data_dir}/zhsf_patients.csv',
        parse_dates=['date_of_birth', 'registration_date']
    )
    members = pd.read_csv(
        f'{data_dir}/zhsf_members.csv',
        parse_dates=['enrollment_date', 'expiry_date']
    )
    claims = pd.read_csv(
        f'{data_dir}/zhsf_claims.csv',
        parse_dates=['service_date', 'submission_date']
    )
    payments = pd.read_csv(
        f'{data_dir}/zhsf_payments.csv',
        parse_dates=['payment_date']
    )

    tables = {'patients': patients, 'members': members,
              'claims': claims, 'payments': payments}

    for name, df in tables.items():
        print(f'{name:10s}: {len(df):>5,} rows × {df.shape[1]} columns')

    print()
    print(f'Claims date range : {claims["service_date"].min().date()} → '
          f'{claims["service_date"].max().date()}')
    print(f'Anomaly rate      : {claims["is_anomaly"].mean() * 100:.1f}%  '
          f'({claims["is_anomaly"].sum()} / {len(claims)})')

    return tables


# ── Clean ───────────────────────────────────────────────────────────────────

def clean_claims(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Apply all data quality fixes to the claims table.

    Fixes applied
    -------------
    1. Remove duplicate claim_id rows (keep first)
    2. Fill missing claimed_amount_tzs with per-hospital median
    3. Fill missing icd10_code with 'UNKNOWN'
    4. Compute lag_days (submission - service), clipped at 0

    Parameters
    ----------
    df      : raw claims DataFrame
    verbose : print audit report if True

    Returns
    -------
    Cleaned DataFrame
    """
    df = df.copy()

    # Audit
    if verbose:
        lag_check = (df['submission_date'] - df['service_date']).dt.days
        null_report = (
            df.isnull().sum()
            .to_frame('null_count')
            .assign(null_pct=lambda x: (x['null_count'] / len(df) * 100).round(2))
            .query('null_count > 0')
            .sort_values('null_pct', ascending=False)
        )
        print('=== MISSING VALUES ===')
        print(null_report.to_string())
        print(f'\nDuplicate claim_id        : {df.duplicated(["claim_id"]).sum()}')
        print(f'Negative lag (bad dates)  : {(lag_check < 0).sum()}')
        print(f'Median submission lag     : {lag_check.median():.0f} days')
        print(f'Claims submitted > 90 days: {(lag_check > 90).sum()}')

    # 1. Remove duplicates
    n_before = len(df)
    df = df.drop_duplicates(subset=['claim_id'], keep='first')
    if verbose:
        print(f'\nRemoved {n_before - len(df)} duplicates → {len(df):,} rows remain')

    # 2. Fill missing amounts with per-hospital median
    hosp_fill = df.groupby('hospital_id')['claimed_amount_tzs'].transform('median')
    df['claimed_amount_tzs'] = df['claimed_amount_tzs'].fillna(hosp_fill)

    # 3. Fill missing ICD-10 codes
    df['icd10_code'] = df['icd10_code'].fillna('UNKNOWN')

    # 4. Submission lag
    df['lag_days'] = (
        (df['submission_date'] - df['service_date']).dt.days.clip(lower=0)
    )

    remaining = df[['claimed_amount_tzs', 'icd10_code']].isnull().sum().sum()
    if verbose:
        print(f'Remaining nulls in key columns: {remaining}')
        print('Data cleaning complete ✓')

    return df


# ── Split ────────────────────────────────────────────────────────────────────

def chronological_split(
    df: pd.DataFrame,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    date_col: str = 'service_date'
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split a DataFrame chronologically into train / validation / test.

    Parameters
    ----------
    df          : cleaned claims DataFrame
    train_ratio : fraction for training (default 0.70)
    val_ratio   : fraction for validation (default 0.15)
    date_col    : column to sort by

    Returns
    -------
    (train_df, val_df, test_df)
    """
    df = df.sort_values(date_col).reset_index(drop=True)
    n         = len(df)
    train_end = int(n * train_ratio)
    val_end   = int(n * (train_ratio + val_ratio))

    train_df = df.iloc[:train_end].copy()
    val_df   = df.iloc[train_end:val_end].copy()
    test_df  = df.iloc[val_end:].copy()

    for name, split in [('Train', train_df), ('Validation', val_df), ('Test', test_df)]:
        anom = split['is_anomaly'].mean() * 100
        print(f'{name:12s}: {len(split):>5,} rows | '
              f'{split[date_col].min().date()} → {split[date_col].max().date()} | '
              f'anomaly rate: {anom:.1f}% ({split["is_anomaly"].sum()})')

    return train_df, val_df, test_df
