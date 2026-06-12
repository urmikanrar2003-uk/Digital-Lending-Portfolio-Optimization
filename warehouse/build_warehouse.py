import json
import duckdb
from pathlib import Path
from loguru import logger
from pathlib import Path

DELTA_BASE = Path(__file__).parent.parent / "lakehouse" / "delta"
WAREHOUSE_DIR = Path(__file__).parent
WAREHOUSE_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DB_PATH = "warehouse/lending.duckdb"

def build_warehouse():
    con = duckdb.connect(DB_PATH)
    logger.info("Building warehouse tables...")

    print("DELTA_BASE =", DELTA_BASE)
    print("Contents =", list(DELTA_BASE.iterdir()))

    for topic in ["loan_applications", "repayments", "behavioral_signals"]:
        parquet_glob = str(DELTA_BASE / topic / 'year=*' / 'month=*' / '*.parquet')
        con.execute(f"""
            CREATE OR REPLACE VIEW raw_{topic} AS
            SELECT * FROM read_parquet('{parquet_glob}', hive_partitioning=true)
        """)
        count = con.execute(f"SELECT COUNT(*) FROM raw_{topic}").fetchone()[0]
        logger.info(f"  raw_{topic}: {count:,} rows")

    con.execute("""
        CREATE OR REPLACE TABLE dim_customers AS
        SELECT
            customer_id,
            MAX(age)                    AS age,
            MAX(income_monthly)         AS income_monthly,
            MAX(employment_type)        AS employment_type,
            MAX(city_tier)              AS city_tier,
            MAX(credit_score_proxy)     AS credit_score_proxy,
            MAX(origination_channel)    AS acquisition_channel,
            MAX(risk_grade)             AS risk_grade,
            MAX(loan_product)           AS primary_product,
            MAX(loan_amount)            AS loan_amount,
            MAX(loan_tenure_months)     AS loan_tenure_months,
            MAX(interest_rate)          AS interest_rate,
            COUNT(*)                    AS total_applications,
            SUM(CASE WHEN approval_status='approved' THEN 1 ELSE 0 END) AS approved_count,
            MAX(timestamp)::DATE        AS last_seen_date,
            -- Gap fields
            MIN(origination_date)       AS origination_date,
            MAX(cost_of_acquisition_inr)    AS cost_of_acquisition_inr,
            AVG(approval_turnaround_days)   AS avg_approval_turnaround_days
        FROM raw_loan_applications
        GROUP BY customer_id
    """)
    logger.success("  dim_customers built")

    con.execute("""
        CREATE OR REPLACE TABLE fact_repayments AS
        SELECT
            customer_id, loan_id,
            timestamp::DATE AS repayment_date,
            emi_amount, amount_paid, days_past_due, is_partial, payment_mode,
            CASE
                WHEN days_past_due = 0   THEN 'on_time'
                WHEN days_past_due <= 30  THEN 'dpd_1_30'
                WHEN days_past_due <= 60  THEN 'dpd_31_60'
                ELSE 'dpd_60_plus'
            END AS dpd_bucket,
            (emi_amount - amount_paid) AS shortfall
        FROM raw_repayments
    """)
    logger.success("  fact_repayments built")

    con.execute("""
        CREATE OR REPLACE TABLE mart_portfolio AS
        SELECT
            c.customer_id, c.age, c.income_monthly, c.employment_type,
            c.city_tier, c.credit_score_proxy, c.acquisition_channel,
            c.risk_grade, c.primary_product, c.approved_count,
            c.loan_amount, c.loan_tenure_months, c.interest_rate,
            c.cost_of_acquisition_inr, c.avg_approval_turnaround_days,
            c.origination_date,

            -- Cohort quarter for vintage analysis (e.g. '2025-Q1')
            CONCAT(
                YEAR(CAST(c.origination_date AS DATE)), '-Q',
                QUARTER(CAST(c.origination_date AS DATE))
            ) AS cohort_quarter,

            COUNT(r.loan_id)                                                AS total_emis,
            COALESCE(AVG(r.days_past_due), 0)                               AS avg_dpd,
            COALESCE(MAX(r.days_past_due), 0)                               AS max_dpd,
            COALESCE(SUM(CASE WHEN r.dpd_bucket != 'on_time' THEN 1 ELSE 0 END), 0) AS delinquent_emis,
            COALESCE(SUM(r.shortfall), 0)                                   AS total_shortfall,
            COALESCE(AVG(CASE WHEN r.is_partial THEN 1.0 ELSE 0 END), 0)   AS partial_payment_rate,
            COALESCE(AVG(b.avg_monthly_balance), 0)                         AS avg_balance,
            COALESCE(AVG(b.balance_volatility), 0)                          AS avg_balance_volatility,
            COALESCE(AVG(b.cash_flow_ratio), 1)                             AS avg_cash_flow_ratio,
            COALESCE(MAX(b.num_spending_shocks), 0)                         AS max_spending_shocks,
            COALESCE(MAX(b.missed_bill_payments), 0)                        AS missed_bills,

            CASE WHEN COALESCE(MAX(r.days_past_due), 0) > 30
                      OR c.risk_grade IN ('D','E')
                 THEN 1 ELSE 0 END                                          AS is_delinquent,

            -- Value proxy: expected interest revenue minus estimated credit loss
            -- Revenue  = loan_amount * (interest_rate/100) * (tenure/12)
            -- Loss est = loan_amount * loss_given_default (40%) * prob_default
            -- Prob of default proxied by delinquency flag
            ROUND(
                (c.loan_amount * (c.interest_rate / 100.0) * (c.loan_tenure_months / 12.0))
                - (c.loan_amount * 0.40 *
                   CASE WHEN COALESCE(MAX(r.days_past_due), 0) > 30
                             OR c.risk_grade IN ('D','E') THEN 1.0 ELSE 0.0 END)
                - c.cost_of_acquisition_inr,
            2) AS risk_adjusted_return,

            -- Simple CLV proxy: income_monthly * expected_active_months / 1000
            -- Active months = tenure if not delinquent, else half tenure
            ROUND(
                (c.income_monthly *
                 CASE WHEN COALESCE(MAX(r.days_past_due), 0) > 30
                           OR c.risk_grade IN ('D','E')
                      THEN c.loan_tenure_months * 0.5
                      ELSE c.loan_tenure_months
                 END) / 1000.0,
            2) AS customer_ltv_proxy

        FROM dim_customers c
        LEFT JOIN fact_repayments r ON c.customer_id = r.customer_id
        LEFT JOIN raw_behavioral_signals b ON c.customer_id = b.customer_id
        GROUP BY
            c.customer_id, c.age, c.income_monthly, c.employment_type,
            c.city_tier, c.credit_score_proxy, c.acquisition_channel,
            c.risk_grade, c.primary_product, c.approved_count,
            c.loan_amount, c.loan_tenure_months, c.interest_rate,
            c.cost_of_acquisition_inr, c.avg_approval_turnaround_days,
            c.origination_date
    """)
    logger.success("  mart_portfolio built")

    summary = con.execute("""
        SELECT
            COUNT(*)                                        AS total_customers,
            ROUND(AVG(income_monthly), 0)                  AS avg_income,
            ROUND(AVG(credit_score_proxy), 1)              AS avg_credit_score,
            SUM(is_delinquent)                             AS delinquent_count,
            ROUND(100.0 * SUM(is_delinquent) / COUNT(*), 2) AS delinquency_rate_pct,
            ROUND(AVG(cost_of_acquisition_inr), 0)         AS avg_cac_inr,
            ROUND(AVG(avg_approval_turnaround_days), 1)    AS avg_turnaround_days,
            ROUND(AVG(risk_adjusted_return), 0)            AS avg_risk_adj_return,
            ROUND(AVG(customer_ltv_proxy), 0)              AS avg_clv_proxy,
            COUNT(DISTINCT cohort_quarter)                 AS cohort_count
        FROM mart_portfolio
    """).df()

    logger.info("\n-- Portfolio summary --")
    print(summary.to_string(index=False))
    con.close()
    logger.success(f"\nWarehouse ready: {DB_PATH}")

if __name__ == "__main__":
    build_warehouse()