"""
run_layer1.py — Master pipeline runner for Layer 1
----------------------------------------------------
Runs:
  1. Producer  → simulates Kafka event stream
  2. Consumer  → writes events to Delta Lake (lakehouse)
  3. Warehouse → builds analytical tables in DuckDB

Usage:
    python run_layer1.py
    python run_layer1.py --customers 5000
"""

import argparse
import sys
from pathlib import Path
from loguru import logger

logger.remove()
logger.add(sys.stdout, format="<green>{time:HH:mm:ss}</green> | {level} | {message}")

sys.path.insert(0, str(Path(__file__).parent / "ingestion"))
sys.path.insert(0, str(Path(__file__).parent / "warehouse"))


def main(n_customers: int):
    logger.info("=" * 55)
    logger.info("LAYER 1 — Data Ingestion Pipeline")
    logger.info("=" * 55)

    # Step 1: Stream simulation
    logger.info("\nSTEP 1/3 — Producing events to stream...")
    from ingestion.Producer import simulate_stream
    simulate_stream(n_customers=n_customers)

    # Step 2: Delta Lake ingestion
    logger.info("\nSTEP 2/3 — Consuming stream → Delta Lake...")
    from ingestion.Consumer import consume_topic, show_delta_stats
    for topic in ["loan_applications", "repayments", "behavioral_signals"]:
        consume_topic(topic)
        show_delta_stats(topic)

    # Step 3: Warehouse build
    logger.info("\nSTEP 3/3 — Building warehouse tables (DuckDB)...")
    sys.path.insert(0, str(Path(__file__).parent / 'warehouse'))
    from warehouse.build_warehouse import build_warehouse
    build_warehouse()

    logger.info("\n" + "=" * 55)
    logger.success("Layer 1 complete. Outputs:")
    logger.info("  lakehouse/raw_stream/   → raw JSONL event files")
    logger.info("  lakehouse/delta/        → Delta Lake parquet tables")
    logger.info("  warehouse/lending.duckdb → analytical mart (mart_portfolio)")
    logger.info("=" * 55)
    logger.info("Ready to build Layer 2 — Feature Store")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--customers", type=int, default=1000,
                        help="Number of customer journeys to simulate")
    args = parser.parse_args()
    main(args.customers)