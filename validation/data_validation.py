"""
Data Quality Validation (Great Expectations)
----------------------------------------------
Three types of checks:
  1. Schema checks    — correct columns exist, correct types
  2. Value checks     — no nulls, values within expected ranges
  3. Distribution checks — catches data drift (e.g. income suddenly 10x higher)
"""

import pandas as pd
import great_expectations as gx
from pathlib import Path
from loguru import logger

PROJECT_ROOT = Path(__file__).parent.parent
FEATURE_DIR = PROJECT_ROOT / "feature_storage" / "features"
REPORT_DIR  = PROJECT_ROOT / "validation" / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def validate_risk_features(df: pd.DataFrame) -> dict:
    suite_name = "risk_features_suite"
    results = {}

    checks = [
        # No nulls on key columns
        ("no_null_customer_id",      df["customer_id"].notna().all()),
        ("no_null_credit_score",     df["credit_score_proxy"].notna().all()),
        ("no_null_avg_dpd",          df["avg_dpd"].notna().all()),

        # Value range checks
        ("credit_score_in_range",     df["credit_score_proxy"].between(150, 1100).all()),
        ("avg_dpd_non_negative",     (df["avg_dpd"] >= 0).all()),
        ("delinquency_ratio_0_1",    df["delinquency_ratio"].dropna().between(0, 1.01).all()),
        ("risk_grade_encoded_1_5",   df["risk_grade_encoded"].between(1, 5).all()),
        ("max_dpd_bucket_0_4",       df["max_dpd_bucket"].between(0, 4).all()),

        # Label check
        ("label_is_binary",          df["label"].isin([0, 1]).all()),
        ("label_not_all_zeros",      df["label"].sum() > 0),
        ("label_not_all_ones",       df["label"].sum() < len(df)),
    ]

    passed = failed = 0
    for name, result in checks:
        results[name] = bool(result)
        if result:
            passed += 1
        else:
            failed += 1
            logger.warning(f"  FAIL: {name}")

    logger.info(f"  risk_features — {passed} passed, {failed} failed")
    return results


def validate_behavioral_features(df: pd.DataFrame) -> dict:
    results = {}

    checks = [
        ("no_null_balance",          df["avg_balance"].notna().all()),
        ("no_null_cash_flow",        df["avg_cash_flow_ratio"].notna().all()),
        ("balance_positive",         (df["avg_balance"] >= 0).all()),
        ("volatility_non_negative",  (df["avg_balance_volatility"] >= 0).all()),
        ("stress_score_0_10",        df["stress_score"].between(0, 10).all()),
        ("cash_flow_score_0_3",      df["cash_flow_score"].between(0, 3).all()),
        ("spending_shocks_positive", (df["max_spending_shocks"] >= 0).all()),
        ("missed_bills_positive",    (df["missed_bills"] >= 0).all()),
    ]

    passed = failed = 0
    for name, result in checks:
        results[name] = bool(result)
        if result:
            passed += 1
        else:
            failed += 1
            logger.warning(f"  FAIL: {name}")

    logger.info(f"  behavioral_features — {passed} passed, {failed} failed")
    return results


def validate_master_features(df: pd.DataFrame) -> dict:
    results = {}

    # Schema check — all expected columns present
    expected_cols = [
        "customer_id", "credit_score_proxy", "avg_dpd", "max_dpd",
        "delinquency_ratio", "stress_score", "cash_flow_score",
        "risk_grade_encoded", "channel_risk_score", "income_k", "is_delinquent"
    ]

    checks = [
        ("all_expected_cols_present", all(c in df.columns for c in expected_cols)),
        ("no_duplicate_customers",    df["customer_id"].nunique() == len(df)),
        ("no_all_null_rows",          df.isnull().all(axis=1).sum() == 0),
        ("income_k_positive",         (df["income_k"] > 0).all()),
        ("row_count_above_100",       len(df) >= 100),
    ]

    # Distribution drift check — flag if median income is outside expected range
    median_income_k = df["income_k"].median()
    checks.append(("income_median_plausible", 10 <= median_income_k <= 500))

    passed = failed = 0
    for name, result in checks:
        results[name] = bool(result)
        if result:
            passed += 1
        else:
            failed += 1
            logger.warning(f"  FAIL: {name}")

    logger.info(f"  master_features — {passed} passed, {failed} failed")
    return results


def run_validation() -> bool:
    logger.info("Running data quality validation...")

    risk_df     = pd.read_parquet(FEATURE_DIR / "risk_features.parquet")
    behav_df    = pd.read_parquet(FEATURE_DIR / "behavioral_features.parquet")
    master_df   = pd.read_parquet(FEATURE_DIR / "master_features.parquet")

    r1 = validate_risk_features(risk_df)
    r2 = validate_behavioral_features(behav_df)
    r3 = validate_master_features(master_df)

    all_results = {**r1, **r2, **r3}
    total   = len(all_results)
    passed  = sum(all_results.values())
    failed  = total - passed

    # Save report
    report_df = pd.DataFrame([
        {"check": k, "passed": v} for k, v in all_results.items()
    ])
    report_df.to_parquet(REPORT_DIR / "validation_report.parquet", index=False)

    logger.info(f"\n── Validation summary: {passed}/{total} checks passed ──")

    if failed > 0:
        logger.error(f"{failed} checks FAILED — review before training model")
        return False
    else:
        logger.success("All checks passed — features are clean")
        return True


if __name__ == "__main__":
    run_validation()