"""
run_layer2.py — Layer 2: Feature Store Pipeline
-------------------------------------------------
Runs:
  1. Feature engineering  → computes 3 feature groups from mart_portfolio
  2. Data validation      → Great Expectations checks on all feature files
  3. Feature store        → registers features, demos offline + online serving

"""

import sys
from pathlib import Path
from loguru import logger

logger.remove()
logger.add(sys.stdout, format="<green>{time:HH:mm:ss}</green> | {level} | {message}")

sys.path.insert(0, str(Path(__file__).parent / "feature_storage"))
sys.path.insert(0, str(Path(__file__).parent / "validation"))
sys.path.insert(0, str(Path(__file__).parent / "feature_repo"))


def main():
    logger.info("=" * 55)
    logger.info("LAYER 2 — Feature Store Pipeline")
    logger.info("=" * 55)

    # Step 1: Feature engineering
    logger.info("\nSTEP 1/3 — Engineering features from mart_portfolio...")
    from feature_storage.feature_engineering import run_feature_engineering
    master_df = run_feature_engineering()

    # Step 2: Data quality validation
    logger.info("\nSTEP 2/3 — Running Great Expectations validation...")
    from validation.data_validation import run_validation
    passed = run_validation()
    if not passed:
        logger.error("Validation failed — fix data issues before proceeding")
        sys.exit(1)

    # Step 3: Feature store demo
    logger.info("\nSTEP 3/3 — Feature store: offline + online serving demo...")
    from feature_repo.feature_store import FeatureStore
    store = FeatureStore()
    store.list_feature_groups()

    # Offline: full training matrix
    train_features = store.get_offline_features(
        feature_groups=["risk_features", "behavioral_features", "acquisition_features"]
    )
    delinquency_rate = master_df["is_delinquent"].mean() * 100

    # Online: single customer real-time score
    sample_id = train_features["customer_id"].iloc[0]
    vec = store.get_online_features(
        feature_groups=["risk_features", "behavioral_features"],
        customer_id=sample_id
    )

    logger.info("\n" + "=" * 55)
    logger.success("Layer 2 complete. Summary:")
    logger.info(f"  Feature groups : 3 (risk, behavioral, acquisition)")
    logger.info(f"  Total features : {len(train_features.columns) - 1}")
    logger.info(f"  Training rows  : {len(train_features):,}")
    logger.info(f"  Delinquency rate: {delinquency_rate:.1f}%")
    logger.info(f"  Offline fetch  : working")
    logger.info(f"  Online fetch   : working (sample: {sample_id[:12]}...)")
    logger.info("=" * 55)
    logger.info("Ready to build Layer 3 — ML Training & Experiment Tracking")


if __name__ == "__main__":
    main()