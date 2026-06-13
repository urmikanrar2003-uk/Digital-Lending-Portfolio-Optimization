import duckdb
import pandas as pd

# Connect to the DuckDB warehouse
con = duckdb.connect("warehouse/lending.duckdb")

# Query to replicate the findings table
query = """
SELECT 
    risk_grade AS Grade,
    COUNT(*) AS Customers,
    ROUND(100.0 * AVG(is_delinquent), 1) AS "Delinquency (%)",
    ROUND(AVG(risk_adjusted_return), 0) AS "Avg Return",
    ROUND(AVG(customer_ltv_proxy), 0) AS "Avg CLV",
    CASE risk_grade
        WHEN 'A' THEN 'Prime — Expand'
        WHEN 'B' THEN 'Near-Prime — Grow'
        WHEN 'C' THEN 'Subprime — Price Higher'
        WHEN 'D' THEN 'High Risk — Restrict'
        WHEN 'E' THEN 'Very High — Suspend'
    END AS Assessment
FROM mart_portfolio
GROUP BY risk_grade
ORDER BY risk_grade
"""

# Execute query and fetch as a pandas DataFrame
df = con.execute(query).df()

# Print the result nicely to the console
print("\n--- Portfolio Segmentation Findings ---")
print(df.to_string(index=False))
print("---------------------------------------\n")

# Query for Acquisition Channel Unit Economics
channel_query = """
SELECT 
    CASE LOWER(acquisition_channel)
        WHEN 'app' THEN 'App (Direct)' 
        WHEN 'web' THEN 'Web'
        WHEN 'agent' THEN 'Agent'
        WHEN 'partner' THEN 'Partner'
        ELSE acquisition_channel 
    END AS Channel,
    ROUND(AVG(cost_of_acquisition_inr), 0) AS "Avg CAC",
    ROUND(AVG(avg_approval_turnaround_days), 1)::VARCHAR || ' days' AS Turnaround,
    ROUND(100.0 * AVG(is_delinquent), 1) AS "Delinquency (%)",
    ROUND(AVG(risk_adjusted_return), 0) AS "Avg Return",
    CASE LOWER(acquisition_channel)
        WHEN 'app' THEN 'Best'
        WHEN 'web' THEN 'Good'
        WHEN 'agent' THEN 'Marginal'
        WHEN 'partner' THEN 'Surprising'
    END AS "Unit Economics"
FROM mart_portfolio
GROUP BY acquisition_channel
ORDER BY 
    CASE LOWER(acquisition_channel)
        WHEN 'app' THEN 1
        WHEN 'web' THEN 2
        WHEN 'agent' THEN 3
        WHEN 'partner' THEN 4
        ELSE 5
    END
"""

df_channel = con.execute(channel_query).df()

print("\n--- 4.1 Acquisition Channel — Unit Economics ---")
print(df_channel.to_string(index=False))
print("------------------------------------------------\n")

# Query for Cohort Vintage Analysis
cohort_query = """
SELECT 
    cohort_quarter AS Cohort,
    COUNT(*) AS Customers,
    ROUND(100.0 * AVG(is_delinquent), 1) AS "Delinquency Rate (%)",
    ROUND(AVG(risk_adjusted_return), 0) AS "Avg Return",
    CASE cohort_quarter
        WHEN '2025-Q1' THEN 'Baseline'
        WHEN '2025-Q4' THEN 'Spike'
        WHEN '2026-Q1' THEN 'Recovery'
        WHEN '2026-Q2' THEN 'Elevated'
        ELSE ''
    END AS Trend
FROM mart_portfolio
WHERE cohort_quarter IN ('2025-Q1', '2025-Q4', '2026-Q1', '2026-Q2')
GROUP BY cohort_quarter
ORDER BY cohort_quarter
"""

df_cohort = con.execute(cohort_query).df()

print("\n--- 5 Cohort Vintage Analysis ---")
print(df_cohort.to_string(index=False))
print("---------------------------------\n")

con.close()
