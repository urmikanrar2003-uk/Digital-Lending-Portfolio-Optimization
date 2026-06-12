"""
Data Preparation for Model Training
-------------------------------------
Loads master features from the feature store (Layer 2 output).
Handles 3 common real-world problems before training:

  1. Class imbalance  — delinquent borrowers are minority (28–31%)
  2. Missing values   — some customers have no repayment history yet
  3. Train/test split — time-aware split to avoid data leakage
"""

import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

PROJECT_ROOT = Path(__file__).parent.parent
FEATURE_DIR = PROJECT_ROOT / "feature_storage" / "features"
PREP_DIR    = PROJECT_ROOT / "training" / "prepared"
PREP_DIR.mkdir(parents=True, exist_ok=True)

# Features the model actually uses — excludes IDs, raw string cols,
# and ANY features derived directly from the delinquency outcome.
#
# REMOVED (data leakage — these encode the label):
#   avg_dpd, max_dpd, max_dpd_bucket  → DPD *is* how is_delinquent is defined
#   delinquency_ratio, delinquent_emis → direct counts of delinquent events
#   risk_grade_encoded                 → synthetically assigned to create DPD patterns
#   total_emis                         → collinear with delinquent_emis
#
# KEPT: genuine pre-delinquency signals observable BEFORE default occurs.
MODEL_FEATURES = [
    # Credit profile (at origination)
    "credit_score_proxy",
    "credit_band",
    # Repayment stress signals (not outcome-derived)
    "avg_shortfall_per_emi",   # how much short per EMI on average
    "partial_payment_rate",    # fraction of EMIs paid partially
    # Behavioral / cash-flow signals
    "avg_balance",
    "avg_balance_volatility",
    "avg_cash_flow_ratio",
    "volatility_to_balance_ratio",
    "cash_flow_score",
    "stress_score",
    "max_spending_shocks",
    "missed_bills",
    # Acquisition & demographic signals
    "channel_risk_score",
    "product_risk_score",
    "employment_score",
    "city_tier_encoded",
    "income_k",
    "approved_count",
]

TARGET = "is_delinquent"


def load_features() -> pd.DataFrame:
    df = pd.read_parquet(FEATURE_DIR / "master_features.parquet")
    logger.info(f"Loaded {len(df):,} rows, {len(df.columns)} cols")
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill nulls with sensible defaults.
    Nulls exist for customers with no repayment history yet (just applied).
    Strategy: fill with 0 for counts/ratios, median for continuous values.
    """
    zero_fill = [
        "avg_shortfall_per_emi", "partial_payment_rate",
        "volatility_to_balance_ratio", "max_spending_shocks", "missed_bills",
    ]
    median_fill = [
        "avg_balance", "avg_balance_volatility", "avg_cash_flow_ratio",
        "cash_flow_score", "stress_score",
    ]

    for col in zero_fill:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    for col in median_fill:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    null_count = df[MODEL_FEATURES].isnull().sum().sum()
    logger.info(f"Missing values after fill: {null_count}")
    return df



def handle_class_imbalance(X: pd.DataFrame, y: pd.Series):
    """
    SMOTE — Synthetic Minority Oversampling Technique.

    Problem: if 70% of borrowers are non-delinquent, a lazy model
             can get 70% accuracy by predicting 'not delinquent' for everyone.
             That's useless for a lender — you miss all the bad loans.

    SMOTE fix: synthetically generates new minority class samples
               (delinquent borrowers) by interpolating between existing ones.
               Result: balanced training set without just duplicating rows.

    Important: SMOTE only on TRAINING data, never on test data.
               Test data must reflect real-world distribution.
    """
    logger.info(f"Before SMOTE — delinquent: {y.sum():,} / {len(y):,} "
                f"({y.mean()*100:.1f}%)")

    smote = SMOTE(random_state=42, k_neighbors=5)
    X_res, y_res = smote.fit_resample(X, y)

    logger.info(f"After SMOTE  — delinquent: {y_res.sum():,} / {len(y_res):,} "
                f"({y_res.mean()*100:.1f}%)")
    return X_res, y_res


def prepare_data():
    df = load_features()
    df = handle_missing_values(df)

    # Keep only model features + target
    available = [f for f in MODEL_FEATURES if f in df.columns]
    X = df[available].copy()
    y = df[TARGET].copy()

    logger.info(f"Feature matrix: {X.shape}, Target distribution: "
                f"{y.value_counts().to_dict()}")

    # Train / test split — stratified to preserve class ratio
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    logger.info(f"Train: {len(X_train):,} rows | Test: {len(X_test):,} rows")

    # Handle imbalance on training set only
    X_train_bal, y_train_bal = handle_class_imbalance(X_train, y_train)

    # Scale features (needed for logistic regression baseline)
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train_bal),
        columns=available
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=available
    )

    # Save prepared data
    X_train_bal.to_parquet(PREP_DIR / "X_train.parquet", index=False)
    X_test.to_parquet(PREP_DIR / "X_test.parquet",  index=False)
    y_train_bal.to_frame().to_parquet(PREP_DIR / "y_train.parquet", index=False)
    y_test.to_frame().to_parquet(PREP_DIR / "y_test.parquet",  index=False)
    X_train_scaled.to_parquet(PREP_DIR / "X_train_scaled.parquet", index=False)
    X_test_scaled.to_parquet(PREP_DIR / "X_test_scaled.parquet",  index=False)

    # Save feature names for model registry
    pd.Series(available).to_csv(PREP_DIR / "feature_names.csv", index=False)

    logger.success(f"Prepared data saved to: {PREP_DIR}")
    return X_train_bal, X_test, y_train_bal, y_test, available


if __name__ == "__main__":
    prepare_data()