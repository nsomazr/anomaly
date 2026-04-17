# ZHSF Anomaly Detection

**AI / ML for ZHSF System Optimization**  
Zephania Reuben & Philipp Ramjoué — April 2026

Detects suspicious claims in the ZHSF health insurance system using two models
trained for A/B testing: XGBoost (primary) and Random Forest (challenger).

---

## Quick Start

### Prerequisites
- Python 3.8+
- Virtual environment (recommended)

### Installation
```bash
# Clone or navigate to the project directory
cd anomaly_detection

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Data Setup
Place the following CSV files in `data/raw/`:
- `zhsf_patients.csv`
- `zhsf_members.csv`
- `zhsf_claims.csv`
- `zhsf_payments.csv`

### Training
```bash
# Train models and save artifacts
python src/train.py
```

### Evaluation
```bash
# Evaluate trained models on test set
python src/evaluate.py
```

### Testing
```bash
# Run unit tests
python -m pytest test/ -v

# Run sample predictions
python test/sample_claims.py
```

---

## Project Structure

```
anomaly_detection/
├── data/
│   └── raw/                     # Raw CSV data files
├── figures/                     # Generated plots and visualizations
├── logs/                        # Training and evaluation logs
├── models/                      # Saved model artifacts (*.pkl files)
├── notebooks/
│   └── end_to_end_deployment.ipynb   # Legacy notebook (use scripts instead)
├── src/
│   ├── __init__.py
│   ├── config.py                # Configuration, paths, constants
│   ├── data.py                  # Data loading and preprocessing
│   ├── features.py              # Feature engineering and encoding
│   ├── train.py                 # Model training pipeline
│   ├── evaluate.py              # Model evaluation and plotting
│   ├── artifacts.py             # Model serialization utilities
│   └── predict.py               # Production inference functions
└── test/
    ├── sample_claims.py         # Test data samples and demo
    ├── test_features.py         # Feature engineering tests
    └── test_predict.py          # Inference pipeline tests
```

---

## Script Usage

### Training Pipeline (`src/train.py`)
Trains XGBoost and Random Forest models with proper validation.
- Loads and cleans data
- Engineers features
- Applies SMOTE for class balancing
- Finds optimal thresholds on validation set
- Saves all artifacts to `models/`

### Evaluation (`src/evaluate.py`)
Evaluates trained models on test data.
- Loads saved models and artifacts
- Computes detailed metrics and reports
- Generates plots (threshold curves, feature importance, confusion matrices)
- Saves visualizations to `figures/`

### Prediction (`src/predict.py`)
Production inference module.
- `prepare_sample()`: Converts raw claim dict to model features
- `predict_claim()`: Scores a claim and returns anomaly decision

### Testing
- `test/test_predict.py`: Unit tests for inference pipeline
- `test/sample_claims.py`: Demo with canonical test cases
- `test/test_features.py`: Feature engineering validation

---

## Key Design Decisions

### Threshold Selection
- Thresholds optimized on **validation set** (not test set) to prevent overfitting
- F1-score maximization for balanced precision/recall
- Separate thresholds for XGBoost and Random Forest

### Feature Engineering
- Leakage-safe: Reference statistics computed from training set only
- Applied consistently to val/test/production data
- Includes temporal, frequency, and cost-deviation features

### Model Architecture
- XGBoost: Primary model with early stopping
- Random Forest: Challenger model for comparison
- SMOTE used for class imbalance (no scale_pos_weight to avoid double-penalty)

---

## Troubleshooting

### Common Issues
1. **"Artifacts not found"**: Run `python src/train.py` first
2. **Import errors**: Ensure virtual environment is activated
3. **Memory issues**: Reduce batch sizes in config.py if needed
4. **Plot display**: Matplotlib backend may need configuration for headless servers

### Logs and Debugging
- Training logs: `logs/train_*.log`
- Check `figures/` for generated plots
- Use `python test/sample_claims.py` for quick validation

---

## API Usage

For production deployment:

```python
from src.predict import predict_claim

result = predict_claim({
    'service_date': '2023-11-05',
    'submission_date': '2023-11-12',
    'claimed_amount_tzs': 35000,
    'facility_type': 'Public',
    # ... other fields
})

print(result)
# {'xgb_score': 0.012, 'rf_score': 0.008, 'xgb_flag': False, 'rf_flag': False}
```

---

## Contributing

1. Follow the existing code structure
2. Add tests for new features
3. Update this README for significant changes
4. Ensure models can be reproduced from saved artifacts
| Step 7 — SMOTE | `train.py → smote_balance()` |
| Step 8 — Train XGBoost | `train.py → train_xgboost()` |
| Step 9 — Train Random Forest | `train.py → train_random_forest()` |
| Step 10 — Threshold selection | `evaluate.py → find_f1_threshold()` |
| Step 10b — Feature importance | `evaluate.py → plot_feature_importance()` |
| Step 11 — Test evaluation | `evaluate.py → full_report()` |
| Step 12 — Save artifacts | `artifacts.py → save_all()` |
| Step 13 — Smoke test | `predict.py → predict_claim()` |

---

## Bugs Fixed in v2

| # | Bug | Root cause | Fix |
|---|-----|-----------|-----|
| 1 | XGBoost threshold ≈ 0.09 | `scale_pos_weight=23` applied on top of SMOTE | Set `scale_pos_weight=1` when training on SMOTE-balanced data |
| 2 | `hospital_claim_count` wrong | `prepare_sample()` used `_provider_ref` for hospital lookup | Use `_hospital_cnt` for hospitals |
| 3 | Reference stat leakage | Medians computed on full dataset (incl. val/test) | Compute on train only; look up for val/test |
| 4 | `rolling_30d_provider` wrong | `cumcount()` is cumulative total, not a 30-day window | Implemented true date-range window |

---

## Deployment Artifacts

Nine files saved to `models/`:

| File | Purpose |
|------|---------|
| `xgb_anomaly_detector.pkl` | XGBoost primary model |
| `rf_anomaly_detector.pkl` | Random Forest A/B challenger |
| `feature_cols.pkl` | Column names and order |
| `encoders.pkl` | One LabelEncoder per categorical column |
| `thresholds.pkl` | Per-model F1-optimal decision thresholds |
| `icd10_medians.pkl` | Training reference for cost_deviation_pct |
| `hospital_medians.pkl` | Training reference for amount_vs_hospital_median |
| `provider_counts.pkl` | Training reference for provider_claim_count |
| `hospital_counts.pkl` | Training reference for hospital_claim_count |

---

## Quick Start

### Option 1: Run Training Script (Recommended)
```bash
cd anomaly_detection
./run_training.sh
```

### Option 2: Run Training Directly
```bash
cd anomaly_detection
source ../folder/dp-env/bin/activate
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
python src/train.py
```

### Option 3: Run Tests (requires trained models)
```bash
cd anomaly_detection
source ../folder/dp-env/bin/activate
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
python -m pytest test/ -v
```

### Option 4: Score a New Claim (requires trained models)
```bash
cd anomaly_detection
source ../folder/dp-env/bin/activate
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
python -c "
from predict import predict_claim
result = predict_claim({
    'service_date': '2023-11-05', 'submission_date': '2023-11-12',
    'claimed_amount_tzs': 35000, 'facility_type': 'Public',
    'plan_type': 'Standard', 'service_type': 'Outpatient',
    'patient_gender': 'Female', 'patient_district': 'Urban West',
    'icd10_code': 'I10', 'hospital_id': 'HOS011', 'provider_id': 'PRV0036'
})
print(result)
"
```

---

## Logging

Training logs are saved to `logs/` directory with timestamps. The system provides:
- **Console output**: Real-time verbose logging during training
- **File logging**: Persistent logs saved to `logs/training_YYYYMMDD_HHMMSS.log`
- **Progress tracking**: Step-by-step pipeline progress with metrics

---

---

## Key Design Principles

1. **Single source of truth** — `config.py` owns all paths, constants, and feature lists
2. **No leakage** — reference statistics (medians, counts) are computed from training data only
3. **Train/inference parity** — `prepare_sample()` mirrors `apply_train_refs()` exactly
4. **One imbalance strategy** — SMOTE is used; `scale_pos_weight=1` and `class_weight=None`
5. **Threshold from data** — F1-optimal threshold computed on validation set; never hardcoded
