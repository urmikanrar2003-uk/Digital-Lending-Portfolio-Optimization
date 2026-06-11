"""
Stream Consumer → Delta Lake Writer
-----------------------------------
Delta Lake gives us:
  - ACID transactions (safe concurrent writes)
  - Time travel (query any historical snapshot)
  - Schema enforcement (bad records are rejected, not silently dropped)
  - Upsert / merge support
"""

import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from loguru import logger
from deltalake import DeltaTable, write_deltalake

RAW_STREAM_DIR = Path(__file__).parent.parent / "lakehouse" / "raw_stream"
RAW_STREAM_DIR.mkdir(parents=True, exist_ok=True)
DELTA_BASE = Path(__file__).parent.parent / "lakehouse" / "delta"
DELTA_BASE.mkdir(parents=True, exist_ok=True)

TOPIC_SCHEMA = {
    "loan_applications": {
        "timestamp":           "datetime64[us]",
        "income_monthly":      "float64",
        "credit_score_proxy":  "float64",
        "loan_amount":         "float64",
        "interest_rate":       "float64",
    },
    "repayments": {
        "timestamp":    "datetime64[us]",
        "emi_due_date": "datetime64[us]",
        "emi_amount":   "float64",
        "amount_paid":  "float64",
        "days_past_due":"int64",
    },
    "behavioral_signals": {
        "timestamp":           "datetime64[us]",
        "avg_monthly_balance": "float64",
        "balance_volatility":  "float64",
        "cash_flow_ratio":     "float64",
    },
}


def _cast_types(df: pd.DataFrame, topic: str) -> pd.DataFrame:
    casts = TOPIC_SCHEMA.get(topic, {})
    for col, dtype in casts.items():
        if col in df.columns:
            df[col] = pd.to_datetime(df[col]) if "datetime" in dtype else df[col].astype(dtype)
    return df


def _add_partition_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Add year/month partition columns for efficient lakehouse queries."""
    df["year"]  = df["timestamp"].dt.year.astype(str)
    df["month"] = df["timestamp"].dt.month.astype(str).str.zfill(2)
    return df


def consume_topic(topic: str, batch_size: int = 200):
    topic_file = RAW_STREAM_DIR / f"{topic}.jsonl"
    if not topic_file.exists():
        logger.warning(f"No data found for topic: {topic}")
        return

    records = []
    with open(topic_file) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records:
        logger.warning(f"Topic {topic} is empty.")
        return

    logger.info(f"Consuming {len(records)} records from '{topic}'...")

    # Process in batches (mirrors micro-batch streaming)
    for i in range(0, len(records), batch_size):
        batch = records[i : i + batch_size]
        df = pd.DataFrame(batch)
        df = _cast_types(df, topic)
        df = _add_partition_cols(df)

        delta_path = str(DELTA_BASE / topic)

        write_deltalake(
            delta_path,
            df,
            mode="append",              # append each batch
            partition_by=["year", "month"]
        )

    logger.success(f"Written {len(records)} records → {DELTA_BASE / topic}")


def show_delta_stats(topic: str):
    delta_path = str(DELTA_BASE / topic)
    try:
        dt = DeltaTable(delta_path)
        df = dt.to_pandas()
        logger.info(f"\n--- {topic} Delta table ---")
        logger.info(f"  Rows      : {len(df):,}")
        logger.info(f"  Columns   : {list(df.columns)}")

        logger.info(f"  Version   : {dt.version()}")
    except Exception as e:
        logger.error(f"Could not read {topic}: {e}")


if __name__ == "__main__":
    for topic in ["loan_applications", "repayments", "behavioral_signals"]:
        consume_topic(topic)
        show_delta_stats(topic)