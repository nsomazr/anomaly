"""
evaluate.py
===========
Model evaluation, threshold selection, and visualisation.

Notebook mapping
----------------
Step 10  — Threshold Selection   → find_f1_threshold()
Step 10b — Feature Importance    → plot_feature_importance()
Step 11  — Final Test Evaluation → full_report(), plot_confusion_matrices()
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, roc_auc_score, precision_recall_curve,
    average_precision_score, f1_score, precision_score, recall_score
)
from config import FIGURES_DIR


def find_f1_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    model_name: str = 'Model',
    n_steps: int = 200
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """
    Find the decision threshold that maximises F1-score.

    Sweeps thresholds from 0.01 to 0.99 and evaluates F1 at each point.
    Must be run on the VALIDATION set — never the test set.

    Parameters
    ----------
    y_true     : true binary labels
    y_proba    : predicted probabilities for the positive class
    model_name : label for printing
    n_steps    : number of threshold values to test

    Returns
    -------
    (best_threshold, thresholds_array, f1_scores_array)
    """
    thresholds = np.linspace(0.01, 0.99, n_steps)
    f1_scores  = [
        f1_score(y_true, (y_proba >= t).astype(int), zero_division=0)
        for t in thresholds
    ]
    best_idx = int(np.argmax(f1_scores))
    best_thr = float(thresholds[best_idx])
    best_f1  = float(f1_scores[best_idx])
    print(f'{model_name:15s} — F1-optimal threshold: {best_thr:.3f}  '
          f'(F1 = {best_f1:.3f})')
    return best_thr, best_f1, thresholds, np.array(f1_scores)


def full_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    label: str,
    threshold: float
) -> dict:
    """
    Print and return a comprehensive metrics dict.

    Parameters
    ----------
    y_true    : true binary labels
    y_pred    : binary predictions (already thresholded)
    y_proba   : raw predicted probabilities
    label     : display name
    threshold : decision threshold used

    Returns
    -------
    dict with all metric values
    """
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred)
    f1   = f1_score(y_true, y_pred)
    auc  = roc_auc_score(y_true, y_proba)
    ap   = average_precision_score(y_true, y_proba)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    print(f'\n══════ {label} (threshold={threshold:.3f}) ══════')
    print(f'  Precision  : {prec*100:>5.1f}%')
    print(f'  Recall     : {rec*100:>5.1f}%')
    print(f'  F1-Score   : {f1*100:>5.1f}%')
    print(f'  AUC-ROC    : {auc:.4f}')
    print(f'  Avg Prec   : {ap:.4f}')
    print(f'  Flags      : {y_pred.sum()} / {len(y_pred)} '
          f'({y_pred.mean()*100:.1f}% flag rate)')
    print(f'  TN={tn}  FP={fp}  FN={fn}  TP={tp}')

    return dict(label=label, precision=prec, recall=rec, f1=f1,
                auc_roc=auc, avg_prec=ap, threshold=threshold,
                tn=tn, fp=fp, fn=fn, tp=tp, n_flags=int(y_pred.sum()))


def plot_threshold_curves(
    models: list[tuple],
    save_path: str = f'{FIGURES_DIR}/threshold_curves.png'
) -> None:
    """
    Plot F1-score vs threshold curves for one or more models.

    Parameters
    ----------
    models    : list of (model_name, thr_array, f1_array, opt_threshold, color)
    save_path : file path to save figure
    """
    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 4))
    if len(models) == 1:
        axes = [axes]

    for ax, (name, thr_range, f1_curve, opt, color) in zip(axes, models):
        ax.plot(thr_range, f1_curve, color=color, linewidth=2)
        ax.axvline(opt, color='crimson', linestyle='--', linewidth=1.5,
                   label=f'Optimal = {opt:.3f}')
        ax.set_xlabel('Decision Threshold')
        ax.set_ylabel('F1-Score')
        ax.set_title(f'{name} — Threshold vs F1', fontweight='bold')
        ax.legend()
        ax.set_xlim([0, 1])

    plt.suptitle('Threshold selection on Validation Set', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f'Saved: {save_path}')


def plot_pr_curves(
    models: list[tuple],
    y_true: np.ndarray,
    save_path: str = f'{FIGURES_DIR}/pr_curves.png'
) -> None:
    """
    Plot Precision-Recall curves for one or more models.

    Parameters
    ----------
    models    : list of (model_name, y_proba, threshold, color)
    y_true    : true binary labels
    save_path : file path to save figure
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, y_proba, threshold, color in models:
        ap   = average_precision_score(y_true, y_proba)
        prec, rec, _ = precision_recall_curve(y_true, y_proba)
        ax.plot(rec, prec, color=color, linewidth=2,
                label=f'{name}  (AP={ap:.3f}, thr={threshold:.3f})')

    ax.axhline(0.85, color='grey', linestyle='--', linewidth=1.2,
               label='85% precision target')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve — Validation Set', fontweight='bold')
    ax.legend(loc='lower left')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f'Saved: {save_path}')


def plot_feature_importance(
    models: list[tuple],
    feature_cols: list[str],
    save_path: str = f'{FIGURES_DIR}/feature_importance.png'
) -> None:
    """
    Plot feature importances side-by-side for multiple models.

    Parameters
    ----------
    models       : list of (model_name, fitted_model, color)
    feature_cols : list of feature names matching model's input
    save_path    : file path to save figure
    """
    fig, axes = plt.subplots(1, len(models), figsize=(7 * len(models), 6))
    if len(models) == 1:
        axes = [axes]

    for ax, (name, model, color) in zip(axes, models):
        imp = pd.Series(model.feature_importances_,
                        index=feature_cols).sort_values(ascending=True)
        imp.plot(kind='barh', ax=ax, color=color, edgecolor='white')
        ax.set_title(f'{name} — Feature Importance', fontweight='bold')
        ax.set_xlabel('Importance Score')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f'Saved: {save_path}')


def plot_confusion_matrices(
    models: list[tuple],
    y_true: np.ndarray,
    save_path: str = f'{FIGURES_DIR}/confusion_matrices.png'
) -> None:
    """
    Plot confusion matrices side-by-side.

    Parameters
    ----------
    models    : list of (model_name, y_pred, threshold)
    y_true    : true binary labels
    save_path : file path to save figure
    """
    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 4))
    if len(models) == 1:
        axes = [axes]

    for ax, (name, y_pred, threshold) in zip(axes, models):
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Pred: Normal', 'Pred: Anomaly'],
                    yticklabels=['Act: Normal',  'Act: Anomaly'],
                    annot_kws={'size': 13})
        ax.set_title(f'{name} (thr={threshold:.3f})', fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f'Saved: {save_path}')


def evaluate_models(
    xgb_model, rf_model,
    X_val: np.ndarray, y_val: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    feature_cols: list[str]
) -> dict:
    """
    Complete evaluation pipeline: threshold selection on val, test evaluation, plots.

    Parameters
    ----------
    xgb_model   : trained XGBoost model
    rf_model    : trained Random Forest model
    X_val       : validation features
    y_val       : validation labels
    X_test      : test features
    y_test      : test labels
    feature_cols: list of feature names

    Returns
    -------
    dict with thresholds and test F1 scores
    """
    print("\n" + "="*60)
    print("MODEL EVALUATION PIPELINE")
    print("="*60)

    # Step 1: Find optimal thresholds on validation set
    print("\n1. Finding optimal thresholds on validation set...")
    xgb_proba_val = xgb_model.predict_proba(X_val)[:, 1]
    xgb_threshold, _, xgb_thresholds, xgb_f1_scores = find_f1_threshold(y_val, xgb_proba_val, 'XGBoost')
    rf_proba_val = rf_model.predict_proba(X_val)[:, 1]
    rf_threshold, _, rf_thresholds, rf_f1_scores = find_f1_threshold(y_val, rf_proba_val, 'Random Forest')
    print(f"XGBoost threshold: {xgb_threshold:.3f}")
    print(f"Random Forest threshold: {rf_threshold:.3f}")

    # Step 2: Evaluate on test set
    print("\n2. Evaluating on test set...")
    xgb_proba_test = xgb_model.predict_proba(X_test)[:, 1]
    xgb_pred_test = (xgb_proba_test >= xgb_threshold).astype(int)
    rf_proba_test = rf_model.predict_proba(X_test)[:, 1]
    rf_pred_test = (rf_proba_test >= rf_threshold).astype(int)

    # Detailed reports
    full_report(y_test, xgb_pred_test, xgb_proba_test, 'XGBoost Test', xgb_threshold)
    full_report(y_test, rf_pred_test, rf_proba_test, 'Random Forest Test', rf_threshold)

    # Step 3: Generate plots
    print("\n3. Generating evaluation plots...")
    plot_threshold_curves([
        ('XGBoost', xgb_thresholds, xgb_f1_scores, xgb_threshold, 'darkblue'),
        ('Random Forest', rf_thresholds, rf_f1_scores, rf_threshold, 'darkgreen')
    ])
    plot_feature_importance([
        ('XGBoost', xgb_model, 'darkblue'),
        ('Random Forest', rf_model, 'darkgreen')
    ], feature_cols)
    plot_confusion_matrices([
        ('XGBoost', xgb_pred_test, xgb_threshold),
        ('Random Forest', rf_pred_test, rf_threshold)
    ], y_test)
    print("Plots saved to figures/ directory.")

    # Return results
    xgb_f1_test = f1_score(y_test, xgb_pred_test)
    rf_f1_test = f1_score(y_test, rf_pred_test)
    return {
        'xgb_threshold': xgb_threshold,
        'rf_threshold': rf_threshold,
        'xgb_f1_test': xgb_f1_test,
        'rf_f1_test': rf_f1_test
    }


if __name__ == "__main__":
    """
    Standalone evaluation script.
    Loads saved models and evaluates on test data using saved thresholds.
    """
    import os
    import joblib
    import numpy as np
    import pandas as pd
    from sklearn.metrics import f1_score
    from config import MODELS_DIR, DATA_DIR, FEATURE_COLS, ARTIFACT_NAMES
    from data import load_raw_data, clean_claims, chronological_split
    from features import build_temporal_features, apply_train_refs, build_rolling_provider_feature, encode_categoricals

    # Check if models directory exists and has files
    if not os.path.exists(MODELS_DIR):
        print(f"Error: Models directory '{MODELS_DIR}' does not exist. Run train.py first to train and save models.")
        exit(1)

    required_files = [ARTIFACT_NAMES[key] for key in ['xgb_model', 'rf_model', 'encoders', 'thresholds', 'icd10_medians', 'hospital_medians', 'provider_counts', 'hospital_counts']]
    missing_files = [f for f in required_files if not os.path.exists(os.path.join(MODELS_DIR, f))]
    if missing_files:
        print(f"Error: Missing saved artifacts: {missing_files}. Run train.py first to train and save models.")
        exit(1)

    print("Loading saved models and artifacts...")
    xgb_model = joblib.load(os.path.join(MODELS_DIR, ARTIFACT_NAMES['xgb_model']))
    rf_model = joblib.load(os.path.join(MODELS_DIR, ARTIFACT_NAMES['rf_model']))
    encoders = joblib.load(os.path.join(MODELS_DIR, ARTIFACT_NAMES['encoders']))
    thresholds = joblib.load(os.path.join(MODELS_DIR, ARTIFACT_NAMES['thresholds']))
    train_refs = {
        'icd10_medians': joblib.load(os.path.join(MODELS_DIR, ARTIFACT_NAMES['icd10_medians'])),
        'hospital_medians': joblib.load(os.path.join(MODELS_DIR, ARTIFACT_NAMES['hospital_medians'])),
        'provider_counts': joblib.load(os.path.join(MODELS_DIR, ARTIFACT_NAMES['provider_counts'])),
        'hospital_counts': joblib.load(os.path.join(MODELS_DIR, ARTIFACT_NAMES['hospital_counts'])),
    }

    print("Loading and preparing test data...")
    tables = load_raw_data()
    claims_df = tables['claims']
    claims_clean = clean_claims(claims_df, verbose=False)
    claims_features = build_temporal_features(claims_clean)
    train_df, val_df, test_df = chronological_split(claims_features)

    # Apply refs
    test_df = apply_train_refs(test_df, train_refs)
    test_df = build_rolling_provider_feature(test_df)

    # Encode
    test_encoded, _ = encode_categoricals(test_df, encoders, fit=False)

    X_test = test_encoded[FEATURE_COLS].values
    y_test = test_encoded['is_anomaly'].values

    print("Running evaluation with saved thresholds...")
    xgb_threshold = thresholds['xgboost']
    rf_threshold = thresholds['random_forest']

    xgb_proba_test = xgb_model.predict_proba(X_test)[:, 1]
    xgb_pred_test = (xgb_proba_test >= xgb_threshold).astype(int)
    rf_proba_test = rf_model.predict_proba(X_test)[:, 1]
    rf_pred_test = (rf_proba_test >= rf_threshold).astype(int)

    # Detailed reports
    full_report(y_test, xgb_pred_test, xgb_proba_test, 'XGBoost Test', xgb_threshold)
    full_report(y_test, rf_pred_test, rf_proba_test, 'Random Forest Test', rf_threshold)

    # Generate plots (using dummy val data for threshold curves, but since we have saved, perhaps skip or use saved)
    # For simplicity, skip threshold curves since thresholds are fixed
    print("\nGenerating plots...")
    plot_feature_importance([
        ('XGBoost', xgb_model, 'darkblue'),
        ('Random Forest', rf_model, 'darkgreen')
    ], FEATURE_COLS)
    plot_confusion_matrices([
        ('XGBoost', xgb_pred_test, xgb_threshold),
        ('Random Forest', rf_pred_test, rf_threshold)
    ], y_test)
    print("Plots saved to figures/ directory.")

    xgb_f1_test = f1_score(y_test, xgb_pred_test)
    rf_f1_test = f1_score(y_test, rf_pred_test)
    print(f"\nFinal Results:")
    print(f"XGBoost Test F1: {xgb_f1_test:.3f} (threshold: {xgb_threshold:.3f})")
    print(f"Random Forest Test F1: {rf_f1_test:.3f} (threshold: {rf_threshold:.3f})")
    print("Evaluation complete!")
