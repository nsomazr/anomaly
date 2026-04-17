"""
ZHSF Anomaly Detection — src package

Modules
-------
config    : paths, constants, feature definitions
data      : load_raw_data, clean_claims, chronological_split
features  : build_temporal_features, build_frequency_features,
            build_rolling_provider_feature, build_cost_features,
            build_train_references, apply_train_refs, encode_categoricals
train     : smote_balance, train_xgboost, train_random_forest
evaluate  : find_f1_threshold, full_report, plot_* functions
artifacts : save_all, load_all
predict   : prepare_sample, predict_claim
"""
