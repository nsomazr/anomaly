"""
test_predict.py
===============
Unit tests for the ZHSF anomaly detection inference pipeline.

Run with:
    cd anomaly_detection
    python -m pytest test/test_predict.py -v

These tests verify:
1. prepare_sample() produces a DataFrame with the correct shape and columns
2. predict_claim() returns a dict with the expected keys and value types
3. Scores are in [0, 1] range
4. Sample A scores as Normal; Sample B scores as Anomaly
5. Unseen categories (Sample C) do not raise exceptions
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pytest
from sample_claims import SAMPLE_A, SAMPLE_B, SAMPLE_C, clean_sample

# These imports will load artifacts at import time — requires models to be saved first
ARTIFACTS_AVAILABLE = False
SKIP_REASON = 'Artifacts not loaded yet'

try:
    from predict import prepare_sample, predict_claim, _feature_cols, _thresholds
    ARTIFACTS_AVAILABLE = True
    SKIP_REASON = ''
except Exception as e:
    SKIP_REASON = f'Artifacts not found (run train.py first): {e}'


# ── Helpers ─────────────────────────────────────────────────────────────────

def _run_prediction(sample: dict) -> tuple:
    """Return (feature_df, result_dict) for a sample."""
    raw  = clean_sample(sample)
    feat = prepare_sample(raw)
    res  = predict_claim(raw, verbose=False)
    return feat, res


# ── Tests ────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not ARTIFACTS_AVAILABLE, reason=SKIP_REASON)
class TestPrepareFeatures:
    def test_output_shape(self):
        feat, _ = _run_prediction(SAMPLE_A)
        assert feat.shape == (1, len(_feature_cols)), \
            f'Expected (1, {len(_feature_cols)}), got {feat.shape}'

    def test_column_order(self):
        feat, _ = _run_prediction(SAMPLE_A)
        assert list(feat.columns) == _feature_cols, \
            'Column order does not match FEATURE_COLS'

    def test_no_nulls(self):
        for sample in [SAMPLE_A, SAMPLE_B, SAMPLE_C]:
            feat, _ = _run_prediction(sample)
            assert feat.isnull().sum().sum() == 0, \
                f'Null values found in prepared features for {sample["_note"]}'

    def test_log_amount_positive(self):
        feat, _ = _run_prediction(SAMPLE_A)
        assert feat['log_claim_amount'].values[0] > 0

    def test_malaria_season_flag(self):
        # Sample B submitted in August → outside season → 0
        feat_b, _ = _run_prediction(SAMPLE_B)
        assert feat_b['is_malaria_season'].values[0] == 0, \
            'Sample B submitted in August should have is_malaria_season=0'

    def test_weekend_flag(self):
        # Sample A submitted on 2023-11-12 (Sunday) → is_weekend=1
        feat_a, _ = _run_prediction(SAMPLE_A)
        import pandas as pd
        day = pd.Timestamp('2023-11-12').dayofweek
        expected = int(day >= 5)
        assert feat_a['is_weekend_submission'].values[0] == expected

    def test_lag_days(self):
        feat_b, _ = _run_prediction(SAMPLE_B)
        assert feat_b['claim_age_days'].values[0] == 89, \
            'Sample B should have 89-day lag'

    def test_unknown_categories_fallback(self):
        # Sample C has unseen hospital/provider/ICD10 — should not raise
        feat_c, _ = _run_prediction(SAMPLE_C)
        assert feat_c.shape == (1, len(_feature_cols))


@pytest.mark.skipif(not ARTIFACTS_AVAILABLE, reason=SKIP_REASON)
class TestPredictClaim:
    def test_output_keys(self):
        _, result = _run_prediction(SAMPLE_A)
        expected_keys = {'xgb_score', 'rf_score', 'xgb_flag', 'rf_flag'}
        assert expected_keys == set(result.keys())

    def test_scores_in_range(self):
        for sample in [SAMPLE_A, SAMPLE_B]:
            _, result = _run_prediction(sample)
            assert 0.0 <= result['xgb_score'] <= 1.0
            assert 0.0 <= result['rf_score']  <= 1.0

    def test_flags_are_bool(self):
        _, result = _run_prediction(SAMPLE_A)
        assert isinstance(result['xgb_flag'], bool)
        assert isinstance(result['rf_flag'],  bool)

    def test_sample_a_normal(self):
        """Sample A should be classified as Normal by both models."""
        _, result = _run_prediction(SAMPLE_A)
        assert result['xgb_flag'] is False, \
            f'Sample A (routine) incorrectly flagged by XGBoost (score={result["xgb_score"]:.4f})'
        assert result['rf_flag'] is False, \
            f'Sample A (routine) incorrectly flagged by RF (score={result["rf_score"]:.4f})'

    def test_sample_b_anomaly(self):
        """Sample B (suspicious) should be flagged by both models."""
        _, result = _run_prediction(SAMPLE_B)
        assert result['xgb_flag'] is True, \
            f'Sample B (suspicious) NOT flagged by XGBoost (score={result["xgb_score"]:.4f})'
        assert result['rf_flag'] is True, \
            f'Sample B (suspicious) NOT flagged by RF (score={result["rf_score"]:.4f})'

    def test_sample_c_no_exception(self):
        """Sample C (unseen IDs) should return valid results without raising."""
        _, result = _run_prediction(SAMPLE_C)
        assert 'xgb_score' in result

    def test_thresholds_reasonable(self):
        """After the scale_pos_weight fix, XGB threshold should be > 0.20."""
        xgb_thr = _thresholds['xgboost']
        assert xgb_thr > 0.20, \
            (f'XGBoost threshold {xgb_thr:.4f} is still below 0.20 — '
             f'double-penalty bug (SMOTE + scale_pos_weight) may still be active. '
             f'Ensure scale_pos_weight=1 in train_xgboost().')

    def test_b_scores_higher_than_a(self):
        """The suspicious claim must score higher than the routine claim."""
        _, res_a = _run_prediction(SAMPLE_A)
        _, res_b = _run_prediction(SAMPLE_B)
        assert res_b['xgb_score'] > res_a['xgb_score'], \
            'XGBoost: suspicious claim scored lower than routine claim'
        assert res_b['rf_score'] > res_a['rf_score'], \
            'RF: suspicious claim scored lower than routine claim'


if __name__ == "__main__":
    if not ARTIFACTS_AVAILABLE:
        print("Cannot run tests:", SKIP_REASON)
        print("Please run 'python src/train.py' first to train and save the models.")
        exit(1)
    else:
        print("Artifacts found. Run with pytest: python -m pytest test/test_predict.py -v")


# ── Run standalone ───────────────────────────────────────────────────────────

if __name__ == '__main__':
    if not ARTIFACTS_AVAILABLE:
        print(f'Cannot run tests: {SKIP_REASON}')
        sys.exit(1)

    print('Running smoke tests on sample claims...\n')

    for sample in [SAMPLE_A, SAMPLE_B, SAMPLE_C]:
        raw  = clean_sample(sample)
        feat = prepare_sample(raw)
        res  = predict_claim(raw, verbose=False)
        flag_xgb = '🚨 ANOMALY' if res['xgb_flag'] else '✅ Normal'
        flag_rf  = '🚨 ANOMALY' if res['rf_flag']  else '✅ Normal'
        print(f'[{sample["_expected"].upper():8s}] {sample["_note"]}')
        print(f'  XGB  score={res["xgb_score"]:.4f} → {flag_xgb}')
        print(f'  RF   score={res["rf_score"]:.4f} → {flag_rf}')
        print()
