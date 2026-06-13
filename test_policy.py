import duckdb

con = duckdb.connect("warehouse/lending.duckdb")

query = """
SELECT 
    ROUND(100.0 * AVG(is_delinquent), 1) AS delinq_rate,
    ROUND(AVG(risk_adjusted_return), 0) AS avg_return,
    ROUND(100.0 * SUM(CASE WHEN is_delinquent=1 THEN loan_amount * 0.40 ELSE 0 END) / SUM(loan_amount), 1) AS net_credit_loss_rate,
    ROUND(AVG(customer_ltv_proxy), 0) AS avg_clv
FROM mart_portfolio
"""

print(con.execute(query).df().to_string(index=False))

query2 = """
SELECT 
    ROUND(100.0 * AVG(is_delinquent), 1) AS delinq_rate,
    ROUND(AVG(risk_adjusted_return), 0) AS avg_return,
    ROUND(100.0 * SUM(CASE WHEN is_delinquent=1 THEN loan_amount * 0.40 ELSE 0 END) / SUM(loan_amount), 1) AS net_credit_loss_rate,
    ROUND(AVG(customer_ltv_proxy), 0) AS avg_clv
FROM mart_portfolio
WHERE risk_grade NOT IN ('D', 'E') AND loan_tenure_months >= 12
"""

print(con.execute(query2).df().to_string(index=False))
con.close()
