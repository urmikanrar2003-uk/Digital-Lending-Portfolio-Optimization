"""
run_layer3.py — Layer 3: ML Training & Experiment Tracking
------------------------------------------------------------
Runs:
  1. Data preparation  → loads features, handles nulls, SMOTE, train/test split
  2. Model training    → baseline (LogReg) + XGBoost, tracked in MLflow
  3. Model registry    → saves best model, promotes to champion

Usage:
    python run_layer3.py

After running, view MLflow UI (Windows fix — use --workers 1):
    mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db --workers 1
    Then open: http://127.0.0.1:5000
"""

import sys
from pathlib import Path
from loguru import logger

logger.remove()
logger.add(sys.stdout, format="<green>{time:HH:mm:ss}</green> | {level} | {message}")

sys.path.insert(0, str(Path(__file__).parent / "training"))
sys.path.insert(0, str(Path(__file__).parent / "registry"))


def main():
    logger.info("=" * 55)
    logger.info("LAYER 3 — ML Training & Experiment Tracking")
    logger.info("=" * 55)

    # Step 1: Data preparation
    logger.info("\nSTEP 1/3 — Preparing training data...")
    from training.data_prep import prepare_data
    X_train, X_test, y_train, y_test, features = prepare_data()

    # Step 2: Train models + track in MLflow
    logger.info("\nSTEP 2/3 — Training models (MLflow tracking)...")
    from training.train import run_training
    xgb_model, xgb_metrics, features = run_training()

    # Step 3: Register best model
    logger.info("\nSTEP 3/3 — Registering model...")
    from registry.model_registry import register_model, promote_to_champion, list_models

    card = register_model(
        model=xgb_model,
        model_name="delinquency_scorer",
        metrics=xgb_metrics,
        features=features,
        notes="XGBoost v1 — trained on synthetic lending data, Layer 3"
    )

    # Auto-promote if AUC > 0.75
    if xgb_metrics["auc"] >= 0.75:
        promote_to_champion(card["model_id"])
    else:
        logger.warning(f"AUC {xgb_metrics['auc']} below 0.75 — not auto-promoted. "
                       f"Review and manually promote.")

    list_models()

    logger.info("\n" + "=" * 55)
    logger.success("Layer 3 complete. Summary:")
    logger.info(f"  Best model   : XGBoost")
    logger.info(f"  AUC          : {xgb_metrics['auc']}")
    logger.info(f"  Gini         : {xgb_metrics['gini']}")
    logger.info(f"  F1           : {xgb_metrics['f1']}")
    logger.info(f"  Registry     : registry/")
    logger.info(f"  MLflow runs  : mlruns/mlflow.db")
    logger.info(f"\n  View experiments (Windows — WinError 10022 fix):")
    logger.info(f"  mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db --workers 1")
    logger.info(f"  Then open: http://127.0.0.1:5000")
    logger.info("=" * 55)
    logger.info("Ready to build Layer 4 — Model Serving (FastAPI + Docker)")


if __name__ == "__main__":
    main()