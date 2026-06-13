# Digital Lending Portfolio Optimization

This repository contains an end-to-end data analytics and machine learning pipeline designed to optimize a digital lending portfolio. The project simulates a real-world financial data environment to identify value-destroying lending segments, evaluate acquisition channels, and build a proactive Early Warning System (EWS) for default prevention.

---

## 🏗️ Methodologies & Architecture

The project is built on a modern **4-Layer Data Architecture**:

### 1. Synthetic Event Generation & Ingestion (Layer 1)
- **Methodology:** We simulate 2,000 realistic customer journeys spanning an 18-month period using `Faker` and `Pydantic`. The data includes 7 distinct dimensions: Customer Profile, Loan Details, Repayment Behavior, Behavioral Signals, Acquisition Metrics, Time Dimensions, and Outcomes.
- **Technology:** Simulated Kafka streaming written directly into **Delta Lake** using Hive-style partitioning (Year/Month).

### 2. Analytical Warehouse (Layer 1)
- **Methodology:** Raw Delta Lake tables are aggregated into a dimensional model (`dim_customers`, `fact_repayments`, `mart_portfolio`). This layer computes critical business KPIs, such as **Risk-Adjusted Return** (Interest Revenue - Expected Credit Loss - CAC) and **Customer Lifetime Value (CLV)**.
- **Technology:** **DuckDB** for fast, embedded OLAP queries.

### 3. Feature Store & Validation (Layer 2)
- **Methodology:** Features are engineered into three groups (Risk, Behavioral, Acquisition). Crucially, target-leaking features (like explicit Days Past Due) are dropped. The pipeline enforces strict data quality checks before features are committed to the offline store.
- **Technology:** **Great Expectations** for declarative data validation.

### 4. ML Training & Experiment Tracking (Layer 3)
- **Methodology:** A classification model is trained to predict loan delinquency. We handle heavy class imbalance using **SMOTE** (Synthetic Minority Over-sampling Technique). The champion model is evaluated using AUC, Gini, and F1 scores. Model explainability (SHAP values) is used to extract behavioral Early Warning Signals.
- **Technology:** **XGBoost** for gradient-boosted classification; **MLflow** for experiment tracking, model registry, and lifecycle management; **SHAP** for interpretability.

---

## 📁 Project Structure

```text
📦 portfolio optimization
 ┣ 📂 ingestion/           # Layer 1: Data generators (Producer) & Delta Lake writers
 ┣ 📂 lakehouse/           # Storage: Raw Parquet/Delta files partitioned by date
 ┣ 📂 warehouse/           # Layer 1: DuckDB data mart build scripts
 ┣ 📂 feature_storage/     # Layer 2: Engineered feature files
 ┣ 📂 validation/          # Layer 2: Great Expectations quality checks
 ┣ 📂 training/            # Layer 3: XGBoost model training and SMOTE balancing
 ┣ 📂 registry/            # Layer 3: MLflow Champion/Challenger model registry
 ┣ 📂 mlruns/              # MLflow local SQLite backend and artifact store
 ┣ 📜 run_layer1.py        # Executable: Generate data & build warehouse
 ┣ 📜 run_layer2.py        # Executable: Build feature store & run data validation
 ┣ 📜 run_layer3.py        # Executable: Train models & log to MLflow
 ┣ 📜 launch_mlflow.py     # Executable: Windows-safe MLflow UI launcher
 ┣ 📜 view_findings.py     # Executable: Query and display portfolio analytics findings
 ┣ 📜 test_policy.py       # Executable: Test portfolio impact of new risk policies
 ┗ 📜 requirements.txt     # Python dependencies
```

---

## 🚀 How to Run the Pipeline

Ensure your virtual environment is active and dependencies are installed (`pip install -r requirements.txt`).

1. **Generate Data & Build Warehouse:**
   ```bash
   python run_layer1.py
   ```
   *Generates 2,000 customers, writes to Delta Lake, and builds the DuckDB analytical mart.*

2. **Compute Features & Validate Data:**
   ```bash
   python run_layer2.py
   ```
   *Engineers pre-delinquency features and runs 25/25 Great Expectations data quality checks.*

3. **Train Models & Register Champion:**
   ```bash
   python run_layer3.py
   ```
   *Trains Logistic Regression (baseline) and XGBoost models, balances data via SMOTE, logs metrics to MLflow, and promotes the best model to Champion.*

4. **View MLflow Tracking UI (Windows Fix):**
   ```bash
   python launch_mlflow.py
   ```
   *Launches the MLflow UI in single-process mode (bypassing a known Windows multiprocessing socket bug). Open `http://127.0.0.1:5000` in your browser.*

5. **View Portfolio Insights:**
   ```bash
   python view_findings.py
   ```
   *Displays portfolio segmentation, acquisition channel economics, and cohort analysis directly from the data warehouse.*

6. **Test Risk Policy Impact:**
   ```bash
   python test_policy.py
   ```
   *Compares current portfolio KPIs against a simulated policy that excludes high-risk grades and short tenures.*

---

## 📊 Key Findings

The analytical output of this pipeline yields several critical insights for portfolio strategy. Key strategic findings include:

* **Risk Segmentation (Value Destruction):** Prime segments (Grade A/B) generate massive positive risk-adjusted returns (₹97K+ per customer). Conversely, Grade D and E borrowers carry a 100% delinquency rate and actively destroy value (-₹28K per loan). Suspending them saves significant capital.
* **Acquisition Channel Efficiency:** The Direct App channel is by far the most efficient acquisition method (CAC ₹550). The Partner channel has a massive CAC (₹5,590) but still yields positive net returns due to larger average loan sizes. Agent channels require immediate audit.
* **Product & Tenure Optimization:** Short-tenure loans (<12 months) are universally loss-making across all products (BNPL, Personal, SME) because accumulated interest cannot cover the CAC and expected credit losses. 24-36 month SME loans are the most profitable.
* **Cohort Vintage Analysis:** While early 2025 cohorts showed stable delinquency (~38%), recent originations (Q4-2025 and Q2-2026) experienced concerning spikes to 46%+, indicating a potential loosening of underwriting standards or macro-driven stress.
* **Early Warning System (EWS):** Utilizing SHAP values from the XGBoost model, we identified 5 behavioral signals (e.g., Cash Flow Ratio drops below 1.0, Balance Volatility spikes) that predict default 60-90 days before a missed payment occurs.
* **Projected Policy Impact:** By enforcing data-driven policies (suspending Grade D/E, restricting short tenures, scaling App acquisition), the portfolio's delinquency rate is projected to drop by **17.4 percentage points**, while average risk-adjusted returns increase by **73%**.
