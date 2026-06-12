"""
Event Producer
--------------
 writes JSON events to a queue directory simulating the stream.
"""

import json
import random
import uuid
from datetime import datetime, timedelta, date
from pathlib import Path
from faker import Faker
from loguru import logger
from Schemas import LoanApplicationEvent, RepaymentEvent, BehavioralEvent

fake = Faker("en_IN")

RAW_STREAM_DIR = Path(__file__).parent.parent / "lakehouse" / "raw_stream"
RAW_STREAM_DIR.mkdir(parents=True, exist_ok=True)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _weighted_choice(choices: dict) -> str:
    keys, weights = zip(*choices.items())
    return random.choices(keys, weights=weights, k=1)[0]


# Portfolio spans 18 months: Jan-2025 through Jun-2026
# Allows cohort/vintage analysis across at least 3 quarterly cohorts
_COHORT_START = date(2025, 1, 1)
_COHORT_END   = date(2026, 6, 30)
_COHORT_DAYS  = (_COHORT_END - _COHORT_START).days


def _random_origination_date() -> str:
    """Random business day within the portfolio window."""
    d = _COHORT_START + timedelta(days=random.randint(0, _COHORT_DAYS))
    return d.isoformat()


# Channel CAC benchmarks (INR) — app cheapest, agent/partner most expensive
_CAC_RANGES = {
    "app":     (300,   800),
    "web":     (600,  1500),
    "agent":  (2000,  5000),
    "partner":(3000,  8000),
}

# Approval turnaround by risk grade (days) — riskier = longer review
_TURNAROUND_GRADE = {
    "A": (0.5, 1.0),
    "B": (0.5, 2.0),
    "C": (1.0, 5.0),
    "D": (5.0, 10.0),
    "E": (10.0, 21.0),
}


# ── Generators ───────────────────────────────────────────────────────────────

def generate_loan_application(customer_id: str | None = None) -> LoanApplicationEvent:
    product     = _weighted_choice({"personal": 50, "bnpl": 30, "sme_working_capital": 20})
    risk_grade  = _weighted_choice({"A": 15, "B": 25, "C": 30, "D": 20, "E": 10})
    channel     = _weighted_choice({"app": 45, "web": 25, "agent": 20, "partner": 10})
    approval_map = {
        "A": "approved",
        "B": "approved",
        "C": _weighted_choice({"approved": 60, "manual_review": 30, "rejected": 10}),
        "D": _weighted_choice({"manual_review": 50, "rejected": 40, "approved": 10}),
        "E": "rejected",
    }
    approval_status = approval_map[risk_grade]

    loan_ranges = {
        "personal": (10_000, 500_000),
        "bnpl": (2_000, 50_000),
        "sme_working_capital": (100_000, 2_000_000),
    }
    lo, hi = loan_ranges[product]

    # Turnaround: rejected/manual_review take longer than clean approvals
    base_lo, base_hi = _TURNAROUND_GRADE[risk_grade]
    if approval_status == "rejected":
        turnaround = round(random.uniform(base_hi, base_hi * 1.5), 1)
    elif approval_status == "manual_review":
        turnaround = round(random.uniform(base_lo * 1.5, base_hi * 1.3), 1)
    else:
        turnaround = round(random.uniform(base_lo, base_hi), 1)

    return LoanApplicationEvent(
        customer_id=customer_id or str(uuid.uuid4()),
        age=random.randint(21, 58),
        income_monthly=round(random.uniform(15_000, 250_000), 2),
        employment_type=_weighted_choice(
            {"salaried": 50, "self_employed": 25, "gig": 15, "unemployed": 10}),
        city_tier=_weighted_choice({"tier1": 30, "tier2": 45, "tier3": 25}),
        credit_score_proxy=round(random.gauss(620, 120), 1),
        loan_amount=round(random.uniform(lo, hi), 2),
        loan_tenure_months=random.choice([3, 6, 12, 18, 24, 36]),
        loan_product=product,
        interest_rate=round(random.uniform(10.5, 36.0), 2),
        origination_channel=channel,
        approval_status=approval_status,
        risk_grade=risk_grade,
        # --- New fields ---
        origination_date=_random_origination_date(),
        cost_of_acquisition_inr=round(random.uniform(*_CAC_RANGES[channel]), 2),
        approval_turnaround_days=turnaround,
    )


def generate_repayment(customer_id: str, loan_id: str,
                       risk_grade: str) -> RepaymentEvent:
    # Realistic DPD distributions WITH overlap between grades.
    # A/B customers occasionally miss payments; D/E customers sometimes pay on time.
    dpd_dist = {
        "A": lambda: random.choices(
            [0, random.randint(1, 10), random.randint(11, 30)],
            [88, 9, 3])[0],
        "B": lambda: random.choices(
            [0, random.randint(1, 15), random.randint(16, 45)],
            [75, 17, 8])[0],
        "C": lambda: random.choices(
            [0, random.randint(1, 30), random.randint(31, 60)],
            [55, 28, 17])[0],
        "D": lambda: random.choices(
            [0, random.randint(1, 45), random.randint(46, 90)],
            [35, 35, 30])[0],
        "E": lambda: random.choices(
            [0, random.randint(1, 30), random.randint(31, 90)],
            [22, 28, 50])[0],
    }
    dpd = dpd_dist[risk_grade]()
    emi = round(random.uniform(2_000, 25_000), 2)
    # Partial payment rate varies by grade but with noise
    partial_prob = {"A": 0.05, "B": 0.12, "C": 0.22, "D": 0.35, "E": 0.48}
    is_partial = random.random() < partial_prob[risk_grade]
    paid = round(emi * random.uniform(0.4, 0.9), 2) if is_partial else emi

    return RepaymentEvent(
        customer_id=customer_id,
        loan_id=loan_id,
        emi_due_date=datetime.utcnow() - timedelta(days=dpd),
        emi_amount=emi,
        amount_paid=paid,
        days_past_due=dpd,
        payment_mode=_weighted_choice(
            {"upi": 50, "netbanking": 25, "auto_debit": 20, "cash": 5}),
        is_partial=is_partial,
    )


def generate_behavioral_signal(customer_id: str,
                                risk_grade: str) -> BehavioralEvent:
    # Behavioral signals overlap across grades — financial stress is noisy.
    # D/E can have stable months; A/B can have shock events.
    grade_params = {
        #            bal_lo  bal_hi   vol_lo vol_hi  shocks_lo shocks_hi  cfr_lo cfr_hi  missed_lo missed_hi
        "A": dict(bal=(5_000, 50_000),  vol=(200,  5_000),  shk=(0, 1), cfr=(1.1, 1.8), msd=(0, 1)),
        "B": dict(bal=(3_000, 40_000),  vol=(500,  8_000),  shk=(0, 2), cfr=(0.95,1.5), msd=(0, 1)),
        "C": dict(bal=(2_000, 30_000),  vol=(1_000,14_000), shk=(0, 4), cfr=(0.80,1.3), msd=(0, 2)),
        "D": dict(bal=(1_000, 20_000),  vol=(2_000,20_000), shk=(1, 6), cfr=(0.60,1.1), msd=(0, 4)),
        "E": dict(bal=(500,  15_000),   vol=(3_000,25_000), shk=(2, 8), cfr=(0.50,1.0), msd=(1, 5)),
    }
    p = grade_params[risk_grade]
    # Add 15% random noise — borrow parameters from adjacent grade occasionally
    if random.random() < 0.15:
        adj = {"A": "B", "B": "C", "C": "D", "D": "E", "E": "D"}
        p = grade_params[adj[risk_grade]]

    return BehavioralEvent(
        customer_id=customer_id,
        avg_monthly_balance=round(random.uniform(*p["bal"]), 2),
        balance_volatility=round(random.uniform(*p["vol"]), 2),
        num_spending_shocks=random.randint(*p["shk"]),
        cash_flow_ratio=round(random.uniform(*p["cfr"]), 3),
        app_login_frequency=random.randint(1, 30),
        missed_bill_payments=random.randint(*p["msd"]),
    )



# ── Local producer (no Kafka needed) ─────────────────────────────────────────

class LocalProducer:
    """Writes events as newline-delimited JSON to topic files — mirrors Kafka API."""

    def __init__(self, queue_dir: Path = RAW_STREAM_DIR):
        self.queue_dir = queue_dir

    def send(self, topic: str, event: dict):
        topic_file = self.queue_dir / f"{topic}.jsonl"
        with open(topic_file, "a") as f:
            f.write(json.dumps(event) + "\n")

    def flush(self):
        pass  

# ── Main simulation ───────────────────────────────────────────────────────────

def simulate_stream(n_customers: int = 500, producer=None):
    if producer is None:
        producer = LocalProducer()

    logger.info(f"Simulating {n_customers} customer journeys...")

    for i in range(n_customers):
        cid = str(uuid.uuid4())

        # 1. Loan application event
        app = generate_loan_application(customer_id=cid)
        producer.send("loan_applications", app.model_dump(mode="json"))

        if app.approval_status == "approved":
            loan_id = str(uuid.uuid4())
            # 2. Repayment events (simulate 3 EMI cycles)
            for _ in range(random.randint(1, 6)):
                repay = generate_repayment(cid, loan_id, app.risk_grade)
                producer.send("repayments", repay.model_dump(mode="json"))

        # 3. Behavioral signal
        beh = generate_behavioral_signal(cid, app.risk_grade)
        producer.send("behavioral_signals", beh.model_dump(mode="json"))

        if (i + 1) % 100 == 0:
            logger.info(f"  {i+1}/{n_customers} customers streamed")

    producer.flush()
    logger.success(f"Stream simulation complete. Files in: {RAW_STREAM_DIR}")


if __name__ == "__main__":
    simulate_stream(n_customers=1000)