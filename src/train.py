"""
train.py
========
SMOTE balancing and model training functions.

Notebook mapping
----------------
Step 7 — SMOTE          → smote_balance()
Step 8 — Train XGBoost  → train_xgboost()
Step 9 — Train RF       → train_random_forest()

Key design decision
-------------------
SMOTE and scale_pos_weight are ALTERNATIVE strategies for handling class imbalance.
Using both simultaneously double-penalises the majority class, inflates anomaly
probability scores, and pushes the F1-optimal threshold to an operationally
meaningless low value (e.g. ~0.09 in v1).

Rule applied here: use SMOTE → set scale_pos_weight=1 (neutral).
"""

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from config import RANDOM_STATE
from logs import setup_logging, get_logger
from data import load_raw_data, clean_claims, chronological_split
from features import (
    build_temporal_features, build_frequency_features,
    build_rolling_provider_feature, build_cost_features,
    build_train_references, encode_categoricals
)
from evaluate import find_f1_threshold, full_report
from artifacts import save_all


def smote_balance(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int = RANDOM_STATE
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply SMOTE to balance the training set.

    Parameters
    ----------
    X_train      : feature matrix
    y_train      : binary labels (0=normal, 1=anomaly)
    random_state : for reproducibility

    Returns
    -------
    (X_resampled, y_resampled)
    """
    print(f'Before SMOTE — Normal: {(y_train==0).sum():,}  '
          f'Anomaly: {(y_train==1).sum():,}')

    smote = SMOTE(random_state=random_state)
    X_sm, y_sm = smote.fit_resample(X_train, y_train)

    print(f'After  SMOTE — Normal: {(y_sm==0).sum():,}  '
          f'Anomaly: {(y_sm==1).sum():,}')
    print('Data is now balanced 50/50 for training.')

    return X_sm, y_sm


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    random_state: int = RANDOM_STATE
) -> XGBClassifier:
    """
    Train an XGBoost classifier.

    IMPORTANT: scale_pos_weight=1 because training data is already balanced
    by SMOTE. If SMOTE is not used upstream, change scale_pos_weight to
    (n_negatives / n_positives) and remove the SMOTE call.

    Parameters
    ----------
    X_train, y_train : SMOTE-balanced training arrays
    X_val, y_val     : validation arrays (used for early stopping only)
    random_state     : for reproducibility

    Returns
    -------
    Fitted XGBClassifier
    """
    model = XGBClassifier(
        n_estimators          = 300,
        max_depth             = 6,
        learning_rate         = 0.05,
        subsample             = 0.8,
        colsample_bytree      = 0.8,
        min_child_weight      = 3,
        # FIX v2: scale_pos_weight=1 — SMOTE already balanced the training data.
        # In v1 this was set to 23 (original class ratio), which combined with
        # SMOTE caused double-penalisation and a pathologically low threshold.
        scale_pos_weight      = 1,
        eval_metric           = 'logloss',
        early_stopping_rounds = 20,
        random_state          = random_state,
        n_jobs                = -1
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=50
    )
    print(f'\nBest number of trees: {model.best_iteration}')
    return model


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int = RANDOM_STATE
) -> RandomForestClassifier:
    """
    Train a Random Forest classifier.

    class_weight=None because training data is already balanced by SMOTE.
    Using class_weight='balanced' on top of SMOTE would also double-penalise.

    Parameters
    ----------
    X_train, y_train : SMOTE-balanced training arrays
    random_state     : for reproducibility

    Returns
    -------
    Fitted RandomForestClassifier
    """
    model = RandomForestClassifier(
        n_estimators     = 300,
        max_depth        = 10,       # limit depth → reduce overfitting
        min_samples_leaf = 3,        # at least 3 samples per leaf
        class_weight     = None,     # SMOTE already balanced — no extra weighting
        random_state     = random_state,
        n_jobs           = -1
    )
    model.fit(X_train, y_train)
    print('Random Forest training complete.')
    return model


def main():
    """
    Main training pipeline for anomaly detection models.
    """
    # Setup logging
    logger = setup_logging(log_level='INFO', log_to_file=True)
    logger.info("Starting anomaly detection training pipeline")

    try:
        # Step 1: Load raw data
        logger.info("Step 1: Loading raw data...")
        tables = load_raw_data()
        claims_df = tables['claims']
        logger.info(f"Loaded {len(claims_df):,} claims")

        # Step 2: Clean data
        logger.info("Step 2: Cleaning claims data...")
        claims_clean = clean_claims(claims_df, verbose=True)
        logger.info(f"Cleaned data has {len(claims_clean):,} claims")

        # Step 3: Build temporal features
        logger.info("Step 3: Building temporal features...")
        claims_features = build_temporal_features(claims_clean)
        logger.info(f"Built temporal features for {len(claims_features):,} claims")

        # Step 4: Chronological split
        logger.info("Step 4: Splitting data chronologically...")
        train_df, val_df, test_df = chronological_split(claims_features)
        logger.info(f"Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")

        # Step 5: Build training references from training split only
        logger.info("Step 5: Building training references...")
        train_refs = build_train_references(train_df)
        logger.info("Built training references (medians, counts)")

        # Step 6: Apply training references to all splits
        logger.info("Step 6: Applying training references...")
        from features import apply_train_refs
        train_df = apply_train_refs(train_df, train_refs)
        val_df = apply_train_refs(val_df, train_refs)
        test_df = apply_train_refs(test_df, train_refs)
        logger.info("Applied training references to all splits")

        # Step 7: Build rolling provider features
        logger.info("Step 7: Building rolling provider features...")
        train_df = build_rolling_provider_feature(train_df)
        val_df = build_rolling_provider_feature(val_df)
        test_df = build_rolling_provider_feature(test_df)
        logger.info("Built rolling provider features")

        # Step 8: Encode categoricals
        logger.info("Step 8: Encoding categorical features...")
        # Rebuild encoders from training data
        from features import encode_categoricals
        train_encoded, encoders = encode_categoricals(train_df, {}, fit=True)
        val_encoded, _ = encode_categoricals(val_df, encoders, fit=False)
        test_encoded, _ = encode_categoricals(test_df, encoders, fit=False)
        logger.info("Encoded categorical features")

        # Step 9: Prepare feature matrices
        logger.info("Step 9: Preparing feature matrices...")
        from config import FEATURE_COLS

        X_train = train_encoded[FEATURE_COLS].values
        y_train = train_encoded['is_anomaly'].values
        X_val = val_encoded[FEATURE_COLS].values
        y_val = val_encoded['is_anomaly'].values
        X_test = test_encoded[FEATURE_COLS].values
        y_test = test_encoded['is_anomaly'].values

        logger.info(f"Training set: {X_train.shape[0]:,} samples, {X_train.shape[1]} features")
        logger.info(f"Class distribution - Normal: {(y_train==0).sum():,}, Anomalies: {(y_train==1).sum():,}")

        # Step 10: Apply SMOTE
        logger.info("Step 10: Applying SMOTE for class balancing...")
        X_train_sm, y_train_sm = smote_balance(X_train, y_train)

        # Step 11: Train XGBoost
        logger.info("Step 11: Training XGBoost model...")
        xgb_model = train_xgboost(X_train_sm, y_train_sm, X_val, y_val)

        # Step 12: Train Random Forest
        logger.info("Step 12: Training Random Forest model...")
        rf_model = train_random_forest(X_train_sm, y_train_sm)

        # Step 13: Find optimal thresholds on validation set
        logger.info("Step 13: Finding optimal thresholds on validation set...")
        from evaluate import find_f1_threshold
        xgb_proba_val = xgb_model.predict_proba(X_val)[:, 1]
        xgb_threshold, _, _, _ = find_f1_threshold(y_val, xgb_proba_val, 'XGBoost')
        rf_proba_val = rf_model.predict_proba(X_val)[:, 1]
        rf_threshold, _, _, _ = find_f1_threshold(y_val, rf_proba_val, 'Random Forest')
        logger.info(f"XGBoost threshold: {xgb_threshold:.3f}")
        logger.info(f"Random Forest threshold: {rf_threshold:.3f}")

        # Step 14: Save artifacts
        logger.info("Step 13: Saving model artifacts...")
        save_all(
            xgb_model=xgb_model,
            rf_model=rf_model,
            feature_cols=FEATURE_COLS,
            encoders=encoders,
            thresholds={'xgboost': xgb_threshold, 'random_forest': rf_threshold},
            icd10_medians=train_refs['icd10_medians'],
            hospital_medians=train_refs['hospital_medians'],
            provider_counts=train_refs['provider_counts'],
            hospital_counts=train_refs['hospital_counts']
        )
        logger.info("All artifacts saved successfully")

        # Final summary
        logger.info("Training pipeline completed successfully!")
        logger.info(f"XGBoost trained with threshold: {xgb_threshold:.3f}")
        logger.info(f"Random Forest trained with threshold: {rf_threshold:.3f}")

    except Exception as e:
        logger.error(f"Training pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
