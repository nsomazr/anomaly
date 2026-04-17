"""
sample_claims.py
================
Canonical test cases for the ZHSF anomaly detection pipeline.

These samples are used in:
  - Notebook Step 13 (smoke test)
  - test/test_predict.py (unit tests)
  - Any integration or regression tests

Adding a new sample
-------------------
Add an entry to SAMPLE_CLAIMS and document which flags it tests.
Each sample should isolate one or two specific fraud signals where possible.
"""

# ── Sample A — Routine outpatient claim ────────────────────────────────────
# All signals within normal ranges.
# Expected: NORMAL by both models.
SAMPLE_A = {
    'service_date'       : '2023-11-05',
    'submission_date'    : '2023-11-12',   # 7-day lag (population median = 11)
    'claimed_amount_tzs' : 35_000,         # near typical outpatient median
    'facility_type'      : 'Public',
    'plan_type'          : 'Standard',
    'service_type'       : 'Outpatient',
    'patient_gender'     : 'Female',
    'patient_district'   : 'Urban West',
    'icd10_code'         : 'I10',           # Essential hypertension — common code
    'hospital_id'        : 'HOS011',
    'provider_id'        : 'PRV0036',
    '_expected'          : 'normal',
    '_note'              : 'All features within normal range'
}

# ── Sample B — Suspicious: multiple red flags ──────────────────────────────
# Red flags:
#   (1) 89-day submission lag — far above median of 11 days
#   (2) Amount 950,000 TZS — approx 16× HOS008 median
#   (3) Malaria code B54 submitted in August — outside season (Apr/May/Oct/Nov)
# Expected: ANOMALY by both models.
SAMPLE_B = {
    'service_date'       : '2023-06-01',
    'submission_date'    : '2023-08-29',   # 89-day lag
    'claimed_amount_tzs' : 950_000,        # ~16× hospital median
    'facility_type'      : 'Private',
    'plan_type'          : 'Family',
    'service_type'       : 'Inpatient',
    'patient_gender'     : 'Male',
    'patient_district'   : 'Chake-Chake',
    'icd10_code'         : 'B54',           # Unspecified malaria — out of season
    'hospital_id'        : 'HOS008',
    'provider_id'        : 'PRV0118',
    '_expected'          : 'anomaly',
    '_note'              : 'Late submission + massive upcoding + off-season malaria'
}

# ── Sample C — Edge case: unknown provider & ICD-10 ────────────────────────
# Tests graceful fallback for unseen categories at inference time.
# Expected: model should not crash; result depends on other features.
SAMPLE_C = {
    'service_date'       : '2023-08-10',
    'submission_date'    : '2023-08-20',
    'claimed_amount_tzs' : 60_000,
    'facility_type'      : 'NGO',
    'plan_type'          : 'Basic',
    'service_type'       : 'Outpatient',
    'patient_gender'     : 'Female',
    'patient_district'   : 'Micheweni',
    'icd10_code'         : 'Z99.99',        # unseen ICD-10 code → fallback
    'hospital_id'        : 'HOS_NEW',       # unseen hospital → fallback
    'provider_id'        : 'PRV_NEW',       # unseen provider → fallback
    '_expected'          : 'unknown',
    '_note'              : 'All lookups should fall back gracefully'
}

# ── Convenience list for batch testing ─────────────────────────────────────
SAMPLE_CLAIMS = [SAMPLE_A, SAMPLE_B, SAMPLE_C]

# Strip private _keys before passing to predict_claim()
INFERENCE_KEYS = [
    'service_date', 'submission_date', 'claimed_amount_tzs',
    'facility_type', 'plan_type', 'service_type',
    'patient_gender', 'patient_district',
    'icd10_code', 'hospital_id', 'provider_id',
]


def clean_sample(sample: dict) -> dict:
    """Return a copy of sample with only inference keys (no _meta keys)."""
    return {k: v for k, v in sample.items() if k in INFERENCE_KEYS}


if __name__ == "__main__":
    """
    Run sample predictions for demonstration.
    Requires trained models in models/ directory.
    """
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
    
    from predict import predict_claim
    
    print("Running sample predictions...")
    for i, sample in enumerate(SAMPLE_CLAIMS, 1):
        name = f"Sample {chr(64+i)}"  # A, B, C
        expected = sample.get('_expected', 'unknown')
        note = sample.get('_note', '')
        
        print(f"\n{name} — Expected: {expected.upper()}")
        if note:
            print(f"  Note: {note}")
        
        raw = clean_sample(sample)
        result = predict_claim(raw)
        print(f"  Result: {result}")
