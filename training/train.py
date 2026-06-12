"""
Model Training + MLflow Experiment Tracking
---------------------------------------------
Trains two models and tracks every experiment in MLflow:
  1. Logistic Regression  — simple baseline 
  2. XGBoost              — main model

MLflow tracks for every experiment run:
  - Parameters  : hyperparameters used (learning rate, max_depth, etc.)
  - Metrics     : AUC, precision, recall, F1, Gini coefficient
  - Artifacts   : the trained model file, feature importance plot
  - Tags        : who ran it, what data version, what git commit
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import shap
import matplotlib.pyplot as plt
from pathlib import Path
from loguru import logger
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).parent.parent
PREP_DIR    = PROJECT_ROOT / "training" / "prepared"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
MLFLOW_DIR  = PROJECT_ROOT / "mlruns"

# Point MLflow to local folder
mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DIR / 'mlflow.db'}")
mlflow.set_experiment("digital_lending_delinquency")


def load_prepared_data():
    X_train = pd.read_parquet(PREP_DIR / "X_train.parquet")
    X_test  = pd.read_parquet(PREP_DIR / "X_test.parquet")
    y_train = pd.read_parquet(PREP_DIR / "y_train.parquet").squeeze()
    y_test  = pd.read_parquet(PREP_DIR / "y_test.parquet").squeeze()
    X_train_scaled = pd.read_parquet(PREP_DIR / "X_train_scaled.parquet")
    X_test_scaled  = pd.read_parquet(PREP_DIR / "X_test_scaled.parquet")
    features = pd.read_csv(PREP_DIR / "feature_names.csv").squeeze().tolist()
    return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, features


def compute_metrics(y_true, y_pred, y_prob) -> dict:
    """Standard credit risk model evaluation metrics."""
    auc   = roc_auc_score(y_true, y_prob)
    gini  = 2 * auc - 1          # Gini = 2*AUC - 1, standard in credit scoring
    prec  = precision_score(y_true, y_pred, zero_division=0)
    rec   = recall_score(y_true, y_pred, zero_division=0)
    f1    = f1_score(y_true, y_pred, zero_division=0)
    return {
        "auc":       round(auc,  4),
        "gini":      round(gini, 4),
        "precision": round(prec, 4),
        "recall":    round(rec,  4),
        "f1":        round(f1,   4),
    }


def plot_feature_importance(model, feature_names: list, model_name: str) -> str:
    """Saves feature importance bar chart as PNG artifact."""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:
        importances = np.abs(model.coef_[0])

    idx = np.argsort(importances)[::-1][:15]   # top 15
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh([feature_names[i] for i in idx[::-1]],
            importances[idx[::-1]], color="#2563eb")
    ax.set_title(f"Top 15 Feature Importances — {model_name}")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    path = str(ARTIFACT_DIR / f"{model_name}_feature_importance.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_shap_summary(model, X_test: pd.DataFrame, model_name: str) -> str:
    """
    SHAP summary plot — explains WHY the model made each prediction.
    Each dot = one customer. Red = high feature value, Blue = low.
    X-axis = how much that feature pushed the score up or down.
    This is what you show to the CRO to explain the model.
    """
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test.iloc[:200])  # sample for speed

    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test.iloc[:200],
                      show=False, plot_size=(10, 8))
    path = str(ARTIFACT_DIR / f"{model_name}_shap_summary.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def train_baseline(X_train_scaled, X_test_scaled, y_train, y_test, features):
    """
    Logistic Regression baseline.
    Rule in ML: always build the simplest possible model first.
    If XGBoost doesn't beat this, something is wrong.
    """
    logger.info("Training baseline: Logistic Regression...")

    params = {"C": 1.0, "max_iter": 1000, "random_state": 42}

    with mlflow.start_run(run_name="logistic_regression_baseline"):
        mlflow.set_tag("model_type", "baseline")
        mlflow.set_tag("data_version", "v1.0")
        mlflow.log_params(params)

        model = LogisticRegression(**params)
        model.fit(X_train_scaled, y_train)

        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
        metrics = compute_metrics(y_test, y_pred, y_prob)

        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "logistic_regression")

        imp_path = plot_feature_importance(model, features, "logistic_regression")
        mlflow.log_artifact(imp_path)

        logger.info(f"  Baseline — AUC: {metrics['auc']}, "
                    f"Gini: {metrics['gini']}, F1: {metrics['f1']}")

    return model, metrics


def train_xgboost(X_train, X_test, y_train, y_test, features):
    """
    XGBoost — main production model.

    Key hyperparameters explained:
      n_estimators    : number of trees (more = better but slower)
      max_depth       : how deep each tree can go (deeper = more complex)
      learning_rate   : how much each tree corrects the previous one
      scale_pos_weight: handles imbalance (sum negatives / sum positives)
      subsample       : fraction of rows used per tree (prevents overfitting)
      colsample_bytree: fraction of features used per tree (prevents overfitting)
    """
    logger.info("Training XGBoost...")

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    params = {
        "n_estimators":     300,
        "max_depth":        5,
        "learning_rate":    0.05,
        "scale_pos_weight": round(float(scale_pos_weight), 2),
        "subsample":        0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "gamma":            0.1,
        "random_state":     42,
        "eval_metric":      "auc",
        "early_stopping_rounds": 20,
    }

    with mlflow.start_run(run_name="xgboost_v1"):
        mlflow.set_tag("model_type", "xgboost")
        mlflow.set_tag("data_version", "v1.0")
        mlflow.log_params(params)

        fit_params = {k: v for k, v in params.items()
                      if k not in ["early_stopping_rounds"]}
        model = XGBClassifier(**fit_params, early_stopping_rounds=20,
                              verbosity=0)
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        metrics = compute_metrics(y_test, y_pred, y_prob)

        mlflow.log_metrics(metrics)
        mlflow.xgboost.log_model(model, "xgboost_model")

        imp_path  = plot_feature_importance(model, features, "xgboost")
        shap_path = plot_shap_summary(model, X_test, "xgboost")
        mlflow.log_artifact(imp_path)
        mlflow.log_artifact(shap_path)

        logger.info(f"  XGBoost   — AUC: {metrics['auc']}, "
                    f"Gini: {metrics['gini']}, F1: {metrics['f1']}")

        logger.info(f"\n{classification_report(y_test, y_pred, target_names=['Good','Bad'])}")

    return model, metrics


def run_training():
    logger.info("Loading prepared data...")
    X_train, X_test, y_train, y_test, \
        X_train_sc, X_test_sc, features = load_prepared_data()

    baseline_model, baseline_metrics = train_baseline(
        X_train_sc, X_test_sc, y_train, y_test, features)

    xgb_model, xgb_metrics = train_xgboost(
        X_train, X_test, y_train, y_test, features)

    logger.info("\n── Model comparison ──")
    logger.info(f"  Baseline AUC : {baseline_metrics['auc']}")
    logger.info(f"  XGBoost AUC  : {xgb_metrics['auc']}")
    logger.info(f"  Gini (XGB)   : {xgb_metrics['gini']} "
                f"({'Good' if xgb_metrics['gini'] > 0.4 else 'Needs improvement'})")

    return xgb_model, xgb_metrics, features


if __name__ == "__main__":
    run_training()