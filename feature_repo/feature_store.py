"""
Feature Store Registry
----------------------
we simulate the feature store as a simple registry class that:
  - tracks which features belong to which group
  - serves feature vectors by customer_id from Parquet files
  - logs feature retrieval exactly like Feast would

This gives us the same CODE PATTERNS as real Feast, without needing Redis.
"""

import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional
from loguru import logger

PROJECT_ROOT = Path(__file__).parent.parent
FEATURE_DIR = PROJECT_ROOT / "feature_storage" / "features"

# ── Feature group definitions (mirrors Feast FeD:\portfolioatureView) ────────────────────

@dataclass
class FeatureGroup:
    name: str
    features: List[str]
    description: str
    parquet_path: Path

    def load(self) -> pd.DataFrame:
        return pd.read_parquet(self.parquet_path).set_index("customer_id")


FEATURE_REGISTRY = {

    "risk_features": FeatureGroup(
        name="risk_features",
        description="Credit score, DPD, delinquency ratio, risk grade signals",
        features=[
            "credit_score_proxy", "avg_dpd", "max_dpd", "delinquency_ratio",
            "max_dpd_bucket", "avg_shortfall_per_emi", "risk_grade_encoded",
            "credit_band", "partial_payment_rate", "delinquent_emis", "total_emis"
        ],
        parquet_path=FEATURE_DIR / "risk_features.parquet",
    ),

    "behavioral_features": FeatureGroup(
        name="behavioral_features",
        description="Balance volatility, cash flow, spending shock signals",
        features=[
            "avg_balance", "avg_balance_volatility", "avg_cash_flow_ratio",
            "volatility_to_balance_ratio", "cash_flow_score", "stress_score",
            "max_spending_shocks", "missed_bills"
        ],
        parquet_path=FEATURE_DIR / "behavioral_features.parquet",
    ),

    "acquisition_features": FeatureGroup(
        name="acquisition_features",
        description="Channel, product, employment, and demographic signals",
        features=[
            "channel_risk_score", "product_risk_score", "employment_score",
            "city_tier_encoded", "income_k", "approved_count"
        ],
        parquet_path=FEATURE_DIR / "acquisition_features.parquet",
    ),
}


# ── Feature store client (mirrors Feast API) ──────────────────────────────────

class FeatureStore:
    """
    Local feature store client.
    Mirrors the Feast API so switching to real Feast is a one-line change.
    """

    def __init__(self):
        self._cache = {}
        logger.info("Feature store initialized")

    def _load_group(self, group_name: str) -> pd.DataFrame:
        if group_name not in self._cache:
            group = FEATURE_REGISTRY[group_name]
            self._cache[group_name] = group.load()
            logger.debug(f"  Loaded feature group: {group_name}")
        return self._cache[group_name]

    def get_offline_features(
        self,
        feature_groups: List[str],
        customer_ids: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Returns a feature dataframe for model training (offline / batch).
        In production: Feast fetches from S3 / BigQuery.
        """
        frames = []
        for group_name in feature_groups:
            df = self._load_group(group_name)
            cols = FEATURE_REGISTRY[group_name].features
            available = [c for c in cols if c in df.columns]
            frames.append(df[available])

        result = pd.concat(frames, axis=1).reset_index()

        if customer_ids:
            result = result[result["customer_id"].isin(customer_ids)]

        logger.info(f"  Offline fetch: {len(result)} rows, "
                    f"{len(result.columns)-1} features, "
                    f"groups={feature_groups}")
        return result

    def get_online_features(
        self,
        feature_groups: List[str],
        customer_id: str,
    ) -> dict:
        """
        Returns a single customer's feature vector for real-time scoring.
        In production: Feast fetches from Redis in <10ms.
        """
        df = self.get_offline_features(feature_groups, customer_ids=[customer_id])

        if df.empty:
            logger.warning(f"  No features found for customer: {customer_id}")
            return {}

        record = df.iloc[0].to_dict()
        logger.info(f"  Online fetch: customer={customer_id[:8]}..., "
                    f"features={list(record.keys())[:5]}...")
        return record

    def list_feature_groups(self):
        logger.info("\n── Registered feature groups ──")
        for name, group in FEATURE_REGISTRY.items():
            logger.info(f"  {name}: {len(group.features)} features — {group.description}")

    def get_feature_stats(self, group_name: str) -> pd.DataFrame:
        df = self._load_group(group_name)
        cols = FEATURE_REGISTRY[group_name].features
        available = [c for c in cols if c in df.columns]
        return df[available].describe().round(3)


if __name__ == "__main__":
    store = FeatureStore()
    store.list_feature_groups()

    # Offline batch fetch (for model training)
    train_df = store.get_offline_features(
        feature_groups=["risk_features", "behavioral_features", "acquisition_features"]
    )
    print(f"\nTraining feature matrix: {train_df.shape}")
    print(train_df.head(3).to_string())

    # Online fetch (for real-time scoring)
    sample_id = train_df["customer_id"].iloc[0]
    vec = store.get_online_features(
        feature_groups=["risk_features", "behavioral_features"],
        customer_id=sample_id,
    )
    print(f"\nOnline feature vector for {sample_id[:12]}...:")
    for k, v in list(vec.items())[:8]:
        print(f"  {k}: {v}")