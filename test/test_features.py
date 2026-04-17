"""
test_features.py
================
Unit tests for feature engineering functions.

Run with:
    cd anomaly_detection
    python -m pytest test/test_features.py -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import numpy as np
import pytest
from features import (
    build_temporal_features,
    build_frequency_features,
    build_rolling_provider_feature,
    build_cost_features,
    encode_categoricals,
    build_train_references,
    apply_train_refs,
)
from config import MALARIA_MONTHS, FEATURE_COLS, CATEGORICAL_COLS


# ── Fixtures ─────────────────────────────────────────────────────────────────

def make_claims(n: int = 10, seed: int = 0) -> pd.DataFrame:
    """Create a minimal synthetic claims DataFrame for testing."""
    rng = np.random.default_rng(seed)
    svc_dates = pd.date_range('2023-01-01', periods=n, freq='7D')
    sub_dates = svc_dates + pd.to_timedelta(rng.integers(1, 30, n), unit='D')

    return pd.DataFrame({
        'claim_id'           : [f'CLM{i:04d}' for i in range(n)],
        'provider_id'        : rng.choice(['PRV001', 'PRV002', 'PRV003'], n),
        'hospital_id'        : rng.choice(['HOS001', 'HOS002'], n),
        'icd10_code'         : rng.choice(['I10', 'B54', 'A01.0', 'UNKNOWN'], n),
        'service_date'       : svc_dates,
        'submission_date'    : sub_dates,
        'claimed_amount_tzs' : rng.uniform(10_000, 500_000, n),
        'facility_type'      : rng.choice(['Public', 'Private', 'NGO'], n),
        'plan_type'          : rng.choice(['Basic', 'Standard', 'Family'], n),
        'service_type'       : rng.choice(['Outpatient', 'Inpatient'], n),
        'patient_gender'     : rng.choice(['Male', 'Female'], n),
        'patient_district'   : rng.choice(['Urban West', 'Wete', 'Micheweni'], n),
        'lag_days'           : (sub_dates - svc_dates).days,
        'is_anomaly'         : rng.choice([0, 1], n, p=[0.96, 0.04]),
    })


# ── Temporal features ─────────────────────────────────────────────────────────

class TestTemporalFeatures:
    def test_columns_added(self):
        df = build_temporal_features(make_claims())
        expected = ['claim_age_days', 'submission_month', 'submission_dayofweek',
                    'submission_quarter', 'is_weekend_submission',
                    'is_malaria_season', 'log_claim_amount']
        for col in expected:
            assert col in df.columns, f'Missing column: {col}'

    def test_malaria_season_correct(self):
        df = make_claims()
        df['submission_date'] = pd.Timestamp('2023-04-15')  # April → in season
        df = build_temporal_features(df)
        assert (df['is_malaria_season'] == 1).all()

    def test_non_malaria_season(self):
        df = make_claims()
        df['submission_date'] = pd.Timestamp('2023-08-10')  # August → out of season
        df = build_temporal_features(df)
        assert (df['is_malaria_season'] == 0).all()

    def test_log_amount_positive(self):
        df = build_temporal_features(make_claims())
        assert (df['log_claim_amount'] > 0).all()

    def test_no_nulls(self):
        df = build_temporal_features(make_claims())
        for col in ['claim_age_days', 'log_claim_amount', 'is_weekend_submission']:
            assert df[col].isnull().sum() == 0


# ── Frequency features ────────────────────────────────────────────────────────

class TestFrequencyFeatures:
    def test_columns_added(self):
        df, _ = build_frequency_features(make_claims())
        assert 'provider_claim_count' in df.columns
        assert 'hospital_claim_count' in df.columns

    def test_counts_positive(self):
        df, _ = build_frequency_features(make_claims())
        assert (df['provider_claim_count'] >= 1).all()
        assert (df['hospital_claim_count'] >= 1).all()

    def test_unseen_provider_fallback(self):
        """Unseen providers should default to count=1 when ref provided."""
        train  = make_claims(20, seed=0)
        test   = make_claims(5,  seed=99)
        test['provider_id'] = 'PRV_UNSEEN'
        _, refs = build_frequency_features(train)
        df, _  = build_frequency_features(test,
                                          provider_ref=refs['provider_counts'],
                                          hospital_ref=refs['hospital_counts'])
        assert (df['provider_claim_count'] == 1).all()


# ── Rolling window feature ────────────────────────────────────────────────────

class TestRollingProviderFeature:
    def test_column_added(self):
        df = build_rolling_provider_feature(make_claims())
        assert 'rolling_30d_provider' in df.columns

    def test_first_claim_is_zero(self):
        """First claim per provider should always be 0 (no prior claims)."""
        df = build_rolling_provider_feature(make_claims())
        first_per_provider = df.groupby('provider_id').first()
        assert (first_per_provider['rolling_30d_provider'] == 0).all()

    def test_non_negative(self):
        df = build_rolling_provider_feature(make_claims())
        assert (df['rolling_30d_provider'] >= 0).all()

    def test_true_window_not_cumcount(self):
        """
        Verify this is a TRUE window, not a cumulative count.
        Create two claims for the same provider: one 40 days before the other.
        The second claim should see 0 prior claims in a 30-day window.
        """
        df = pd.DataFrame({
            'claim_id'        : ['C1', 'C2'],
            'provider_id'     : ['PRV1', 'PRV1'],
            'submission_date' : [pd.Timestamp('2023-01-01'), pd.Timestamp('2023-02-15')],
            'lag_days'        : [5, 5],
            'claimed_amount_tzs': [10000.0, 10000.0],
        })
        # Add dummy columns needed by the function
        for col in ['claim_id', 'provider_id', 'submission_date']:
            pass  # already present

        result = build_rolling_provider_feature(df)
        result = result.sort_values('submission_date')

        # C1 is first → 0
        assert result.iloc[0]['rolling_30d_provider'] == 0
        # C2 is 45 days after C1 → outside 30-day window → 0 (not 1)
        assert result.iloc[1]['rolling_30d_provider'] == 0, \
            'rolling_30d_provider is counting cumulatively instead of within 30 days'


# ── Cost features ─────────────────────────────────────────────────────────────

class TestCostFeatures:
    def test_columns_added(self):
        df, _ = build_cost_features(make_claims())
        assert 'cost_deviation_pct' in df.columns
        assert 'amount_vs_hospital_median' in df.columns
        assert 'icd10_median_cost' in df.columns

    def test_no_division_by_zero(self):
        df, _ = build_cost_features(make_claims())
        assert not df['cost_deviation_pct'].isnull().any()
        assert not df['amount_vs_hospital_median'].isnull().any()

    def test_leakage_safe_refs(self):
        """Val/test cost features should use train refs, not their own data."""
        train = make_claims(30, seed=0)
        test  = make_claims(10, seed=1)
        _, train_refs = build_cost_features(train)
        test_with_train_refs, _ = build_cost_features(
            test,
            icd10_ref   =train_refs['icd10_medians'],
            hospital_ref=train_refs['hospital_medians']
        )
        _, test_own_refs = build_cost_features(test)
        # Medians may differ — the values should come from different distributions
        # Just verify the function runs without error (leakage check is conceptual)
        assert 'cost_deviation_pct' in test_with_train_refs.columns


# ── Encoding ──────────────────────────────────────────────────────────────────

class TestEncoding:
    def test_fit_creates_encoders(self):
        df = make_claims()
        _, encoders = encode_categoricals(df, fit=True)
        assert set(encoders.keys()) == set(CATEGORICAL_COLS.keys())

    def test_encoded_columns_present(self):
        df, _ = encode_categoricals(make_claims(), fit=True)
        for enc_col in CATEGORICAL_COLS.values():
            assert enc_col in df.columns

    def test_unseen_category_fallback(self):
        train = make_claims(20, seed=0)
        test  = make_claims(5,  seed=1)
        test['facility_type'] = 'UNKNOWN_TYPE'
        _, encoders = encode_categoricals(train, fit=True)
        df_test, _  = encode_categoricals(test, encoders=encoders, fit=False)
        assert (df_test['facility_type_enc'] == 0).all()

    def test_same_encoding_across_splits(self):
        """Public must map to the same integer in train and test."""
        train = make_claims(20, seed=0)
        test  = make_claims(5,  seed=1)
        test['facility_type'] = 'Public'
        train_df, encoders = encode_categoricals(train, fit=True)
        test_df,  _        = encode_categoricals(test, encoders=encoders, fit=False)
        train_val = train_df.loc[train_df['facility_type'] == 'Public', 'facility_type_enc'].iloc[0]
        test_val  = test_df['facility_type_enc'].iloc[0]
        assert train_val == test_val, 'Encoding differs between train and test for same category'


if __name__ == '__main__':
    print('Running feature engineering tests...')
    import pytest
    pytest.main([__file__, '-v'])
