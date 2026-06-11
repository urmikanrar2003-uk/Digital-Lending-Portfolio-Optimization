"""
Feature Engineering
--------------------
Reads mart_portfolio from DuckDB (Layer 1 output).
Computes 3 feature groups:
  - risk_features       : credit and repayment risk signals
  - behavioral_features : spending and cash flow patterns
  - acquisition_features: channel and product attributes (encoded)
"""

import duckdb
import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger

DB_PATH        = Path("D:\portfolio optimization\warehouse\lending.duckdb")
FEATURE_DIR    = Path("D:\portfolio optimization\feature_store\features")
FEATURE_DIR.mkdir(parents=True, exist_ok=True)


def compute_risk_features(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """
    Risk features — derived from repayment behavior and credit profile.
    These are the strongest predictors of delinquency.
    """
    df = con.execute("""
        SELECT
            customer_id,

            -- Raw signals
            credit_score_proxy,
            avg_dpd,
            max_dpd,
            delinquent_emis,
            total_shortfall,
            partial_payment_rate,
            total_emis,

            -- Engineered features
            CASE
                WHEN credit_score_proxy >= 750 THEN 5
                WHEN credit_score_proxy >= 700 THEN 4
                WHEN credit_score_proxy >= 650 THEN 3
                WHEN credit_score_proxy >= 600 THEN 2
                ELSE 1
            END                                                 AS credit_band,

            ROUND(delinquent_emis * 1.0 /
                  NULLIF(total_emis, 0), 4)                     AS delinquency_ratio,

            CASE
                WHEN max_dpd = 0   THEN 0
                WHEN max_dpd <= 7  THEN 1
                WHEN max_dpd <= 30 THEN 2
                WHEN max_dpd <= 60 THEN 3
                ELSE 4
            END                                                 AS max_dpd_bucket,

            ROUND(total_shortfall /
                  NULLIF(total_emis, 0), 2)                     AS avg_shortfall_per_emi,

            -- Risk grade encoded (ordinal)
            CASE risk_grade
                WHEN 'A' THEN 1
                WHEN 'B' THEN 2
                WHEN 'C' THEN 3
                WHEN 'D' THEN 4
                WHEN 'E' THEN 5
            END                                                 AS risk_grade_encoded,

            is_delinquent                                       AS label

        FROM mart_portfolio
    """).df()

    logger.info(f"  risk_features: {len(df)} rows, {len(df.columns)} cols")
    return df


def compute_behavioral_features(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """
    Behavioral features — cash flow, balance, and spending pattern signals.
    Capture early stress signals BEFORE missed payments appear.
    """
    df = con.execute("""
        SELECT
            customer_id,

            -- Raw signals
            avg_balance,
            avg_balance_volatility,
            avg_cash_flow_ratio,
            max_spending_shocks,
            missed_bills,

            -- Engineered features
            ROUND(avg_balance_volatility /
                  NULLIF(avg_balance, 0), 4)                    AS volatility_to_balance_ratio,

            CASE
                WHEN avg_cash_flow_ratio >= 1.3  THEN 'healthy'
                WHEN avg_cash_flow_ratio >= 1.0  THEN 'neutral'
                WHEN avg_cash_flow_ratio >= 0.8  THEN 'stressed'
                ELSE                                  'distressed'
            END                                                 AS cash_flow_segment,

            CASE
                WHEN avg_cash_flow_ratio >= 1.3  THEN 3
                WHEN avg_cash_flow_ratio >= 1.0  THEN 2
                WHEN avg_cash_flow_ratio >= 0.8  THEN 1
                ELSE                                  0
            END                                                 AS cash_flow_score,

            -- Composite stress score (0-10)
            LEAST(10, ROUND(
                (max_spending_shocks * 1.5) +
                (missed_bills * 2.0) +
                CASE WHEN avg_cash_flow_ratio < 1.0
                     THEN (1.0 - avg_cash_flow_ratio) * 5 ELSE 0 END +
                CASE WHEN avg_balance_volatility > 10000
                     THEN 1.5 ELSE 0 END
            , 2))                                               AS stress_score,

            is_delinquent                                       AS label

        FROM mart_portfolio
    """).df()

    logger.info(f"  behavioral_features: {len(df)} rows, {len(df.columns)} cols")
    return df


def compute_acquisition_features(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """
    Acquisition features — channel, product, and demographic signals.
    Answers: does how a borrower came in predict their behavior?
    """
    df = con.execute("""
        SELECT
            customer_id,

            -- Raw
            acquisition_channel,
            primary_product,
            employment_type,
            city_tier,
            age,
            income_monthly,
            approved_count,

            -- Engineered
            CASE acquisition_channel
                WHEN 'app'     THEN 4
                WHEN 'web'     THEN 3
                WHEN 'partner' THEN 2
                WHEN 'agent'   THEN 1
            END                                                 AS channel_risk_score,

            CASE primary_product
                WHEN 'personal'           THEN 2
                WHEN 'bnpl'               THEN 1
                WHEN 'sme_working_capital' THEN 3
            END                                                 AS product_risk_score,

            CASE employment_type
                WHEN 'salaried'      THEN 4
                WHEN 'self_employed' THEN 3
                WHEN 'gig'           THEN 2
                WHEN 'unemployed'    THEN 1
            END                                                 AS employment_score,

            CASE city_tier
                WHEN 'tier1' THEN 3
                WHEN 'tier2' THEN 2
                WHEN 'tier3' THEN 1
            END                                                 AS city_tier_encoded,

            ROUND(income_monthly / 1000.0, 2)                  AS income_k,

            CASE
                WHEN age < 25 THEN 'young'
                WHEN age < 35 THEN 'prime'
                WHEN age < 50 THEN 'mature'
                ELSE               'senior'
            END                                                 AS age_segment,

            is_delinquent                                       AS label

        FROM mart_portfolio
    """).df()

    logger.info(f"  acquisition_features: {len(df)} rows, {len(df.columns)} cols")
    return df


def build_master_feature_table(risk_df, behavioral_df, acquisition_df) -> pd.DataFrame:
    """
    Joins all 3 feature groups into one master feature table.
    This is what gets served to the ML model.
    In production: Feast serves this from Redis (online) or S3 (offline).
    """
    drop_label = lambda df: df.drop(columns=["label"])

    master = (
        drop_label(risk_df)
        .merge(drop_label(behavioral_df), on="customer_id", how="inner")
        .merge(drop_label(acquisition_df),  on="customer_id", how="inner")
    )
    # Add label back once from risk_df
    master["is_delinquent"] = risk_df.set_index("customer_id").loc[
        master["customer_id"], "label"
    ].values

    # Drop string cols not needed for ML (kept in acquisition for reference)
    master = master.drop(columns=[
        "acquisition_channel", "primary_product",
        "employment_type", "city_tier", "cash_flow_segment", "age_segment"
    ])

    logger.info(f"  master_features: {len(master)} rows, {len(master.columns)} cols")
    return master


def run_feature_engineering():
    con = duckdb.connect(str(DB_PATH))
    logger.info("Computing feature groups...")

    risk_df       = compute_risk_features(con)
    behavioral_df = compute_behavioral_features(con)
    acq_df        = compute_acquisition_features(con)
    master_df     = build_master_feature_table(risk_df, behavioral_df, acq_df)

    # Save each feature group as Parquet (Feast offline store format)
    risk_df.to_parquet(FEATURE_DIR / "risk_features.parquet",       index=False)
    behavioral_df.to_parquet(FEATURE_DIR / "behavioral_features.parquet", index=False)
    acq_df.to_parquet(FEATURE_DIR / "acquisition_features.parquet", index=False)
    master_df.to_parquet(FEATURE_DIR / "master_features.parquet",   index=False)

    logger.success(f"Feature files saved to: {FEATURE_DIR}")
    con.close()
    return master_df


if __name__ == "__main__":
    df = run_feature_engineering()
    print(df.describe().round(2).to_string())