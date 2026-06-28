"""Central paths and reproducibility settings for the project.

Keeping shared locations and random seeds in one module prevents analytical scripts
from silently using different files or stochastic settings.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = PROJECT_ROOT / "src" / "bankruptcy_risk"

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
METADATA_DIR = OUTPUT_DIR / "metadata"

RAW_DATA_PATH = RAW_DATA_DIR / "american_bankruptcy.csv"
INTERIM_PANEL_PATH = INTERIM_DATA_DIR / "bankruptcy_panel.parquet"
MODEL_FEATURES_PATH = PROCESSED_DATA_DIR / "model_features.parquet"
MODEL_DATASET_PATH = PROCESSED_DATA_DIR / "model_dataset.parquet"
RAW_VALIDATION_REPORT_PATH = METADATA_DIR / "raw_data_validation.json"
FEATURE_DICTIONARY_PATH = OUTPUT_DIR / "tables" / "feature_dictionary.csv"
TEMPORAL_SPLIT_SUMMARY_PATH = OUTPUT_DIR / "tables" / "temporal_split_summary.csv"
EXPANDING_FOLD_SUMMARY_PATH = OUTPUT_DIR / "tables" / "expanding_fold_summary.csv"
ANNUAL_OVERVIEW_TABLE_PATH = OUTPUT_DIR / "tables" / "annual_bankruptcy_overview.csv"
TRAINING_RATIO_SUMMARY_PATH = OUTPUT_DIR / "tables" / "training_ratio_summary.csv"
VALIDATION_BASELINE_PREDICTIONS_PATH = (
    OUTPUT_DIR / "tables" / "validation_baseline_predictions.csv"
)
INTERPRETABLE_LOGIT_COEFFICIENTS_PATH = (
    OUTPUT_DIR / "tables" / "interpretable_logit_coefficients.csv"
)
INTERPRETABLE_LOGIT_MARGINAL_EFFECTS_PATH = (
    OUTPUT_DIR / "tables" / "interpretable_logit_marginal_effects.csv"
)
INTERPRETABLE_LOGIT_VALIDATION_PREDICTIONS_PATH = (
    OUTPUT_DIR / "tables" / "interpretable_logit_validation_predictions.csv"
)
INTERPRETABLE_LOGIT_MODEL_PATH = OUTPUT_DIR / "models" / "interpretable_logit.joblib"
REGULARIZED_LOGIT_CV_RESULTS_PATH = (
    OUTPUT_DIR / "tables" / "regularized_logit_cv_results.csv"
)
REGULARIZED_LOGIT_SELECTION_PATH = (
    OUTPUT_DIR / "tables" / "regularized_logit_selection.csv"
)
REGULARIZED_LOGIT_COEFFICIENTS_PATH = (
    OUTPUT_DIR / "tables" / "regularized_logit_coefficients.csv"
)
REGULARIZED_LOGIT_VALIDATION_PREDICTIONS_PATH = (
    OUTPUT_DIR / "tables" / "regularized_logit_validation_predictions.csv"
)
REGULARIZED_LOGIT_MODELS_PATH = OUTPUT_DIR / "models" / "regularized_logit_models.joblib"
DECISION_TREE_CV_RESULTS_PATH = OUTPUT_DIR / "tables" / "decision_tree_cv_results.csv"
DECISION_TREE_SELECTION_PATH = OUTPUT_DIR / "tables" / "decision_tree_selection.csv"
DECISION_TREE_FEATURE_IMPORTANCE_PATH = (
    OUTPUT_DIR / "tables" / "decision_tree_feature_importance.csv"
)
DECISION_TREE_VALIDATION_PREDICTIONS_PATH = (
    OUTPUT_DIR / "tables" / "decision_tree_validation_predictions.csv"
)
DECISION_TREE_MODEL_PATH = OUTPUT_DIR / "models" / "pruned_decision_tree.joblib"
RANDOM_FOREST_CV_RESULTS_PATH = OUTPUT_DIR / "tables" / "random_forest_cv_results.csv"
RANDOM_FOREST_SELECTION_PATH = OUTPUT_DIR / "tables" / "random_forest_selection.csv"
RANDOM_FOREST_FEATURE_IMPORTANCE_PATH = (
    OUTPUT_DIR / "tables" / "random_forest_feature_importance.csv"
)
RANDOM_FOREST_OOB_DIAGNOSTICS_PATH = (
    OUTPUT_DIR / "tables" / "random_forest_oob_diagnostics.csv"
)
RANDOM_FOREST_VALIDATION_PREDICTIONS_PATH = (
    OUTPUT_DIR / "tables" / "random_forest_validation_predictions.csv"
)
RANDOM_FOREST_MODEL_PATH = OUTPUT_DIR / "models" / "random_forest.joblib"

ANNUAL_OVERVIEW_FIGURE_PATH = OUTPUT_DIR / "figures" / "annual_bankruptcy_overview.png"
PERIOD_BALANCE_FIGURE_PATH = OUTPUT_DIR / "figures" / "class_balance_by_period.png"
RATIO_DISTRIBUTION_FIGURE_PATH = (
    OUTPUT_DIR / "figures" / "training_ratio_distributions.png"
)
RATIO_CORRELATION_FIGURE_PATH = OUTPUT_DIR / "figures" / "training_ratio_correlation.png"
DECISION_TREE_FIGURE_PATH = OUTPUT_DIR / "figures" / "pruned_decision_tree.png"
RANDOM_FOREST_IMPORTANCE_FIGURE_PATH = (
    OUTPUT_DIR / "figures" / "random_forest_feature_importance.png"
)

CONFIG_MODULE_PATH = PACKAGE_DIR / "config.py"
DATA_VALIDATION_MODULE_PATH = PACKAGE_DIR / "data_validation.py"
TARGET_MODULE_PATH = PACKAGE_DIR / "target.py"
FEATURES_MODULE_PATH = PACKAGE_DIR / "features.py"
SPLITTING_MODULE_PATH = PACKAGE_DIR / "splitting.py"
PREPROCESSING_MODULE_PATH = PACKAGE_DIR / "preprocessing.py"
EXPLORATION_MODULE_PATH = PACKAGE_DIR / "exploration.py"
VISUALIZATION_MODULE_PATH = PACKAGE_DIR / "visualization.py"
BASELINES_MODULE_PATH = PACKAGE_DIR / "baselines.py"
INTERPRETABLE_LOGIT_MODULE_PATH = PACKAGE_DIR / "interpretable_logit.py"
REGULARIZED_LOGIT_MODULE_PATH = PACKAGE_DIR / "regularized_logit.py"
DECISION_TREE_MODULE_PATH = PACKAGE_DIR / "decision_tree.py"
RANDOM_FOREST_MODULE_PATH = PACKAGE_DIR / "random_forest.py"

RANDOM_SEED = 42
