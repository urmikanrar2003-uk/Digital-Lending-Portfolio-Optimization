"""
Event Producer
--------------
 writes JSON events to a queue directory simulating the stream.
"""

import json
import random
import uuid
from datetime import datetime, timedelta
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


# ── Generators ───────────────────────────────────────────────────────────────

def generate_loan_application(customer_id: str | None = None) -> LoanApplicationEvent:
    product = _weighted_choice({"personal": 50, "bnpl": 30, "sme_working_capital": 20})
    risk_grade = _weighted_choice({"A": 15, "B": 25, "C": 30, "D": 20, "E": 10})
    approval_map = {"A": "approved", "B": "approved", "C": _weighted_choice(
        {"approved": 60, "manual_review": 30, "rejected": 10}),
        "D": _weighted_choice({"manual_review": 50, "rejected": 40, "approved": 10}),
        "E": "rejected"}

    loan_ranges = {
        "personal": (10_000, 500_000),
        "bnpl": (2_000, 50_000),
        "sme_working_capital": (100_000, 2_000_000),
    }
    lo, hi = loan_ranges[product]

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
        origination_channel=_weighted_choice(
            {"app": 45, "web": 25, "agent": 20, "partner": 10}),
        approval_status=approval_map[risk_grade],
        risk_grade=risk_grade,
    )


def generate_repayment(customer_id: str, loan_id: str,
                       risk_grade: str) -> RepaymentEvent:
    dpd_dist = {
        "A": lambda: 0,
        "B": lambda: random.choices([0, random.randint(1, 5)], [90, 10])[0],
        "C": lambda: random.choices([0, random.randint(1, 30)], [70, 30])[0],
        "D": lambda: random.choices([0, random.randint(1, 60)], [50, 50])[0],
        "E": lambda: random.choices([0, random.randint(30, 90)], [30, 70])[0],
    }
    dpd = dpd_dist[risk_grade]()
    emi = round(random.uniform(2_000, 25_000), 2)
    is_partial = random.random() < (0.05 if risk_grade in "AB" else 0.25)
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
    high_risk = risk_grade in ("D", "E")
    return BehavioralEvent(
        customer_id=customer_id,
        avg_monthly_balance=round(random.uniform(500, 50_000), 2),
        balance_volatility=round(random.uniform(5_000 if high_risk else 500,
                                                 30_000 if high_risk else 8_000), 2),
        num_spending_shocks=random.randint(2 if high_risk else 0, 8 if high_risk else 2),
        cash_flow_ratio=round(random.uniform(0.5 if high_risk else 0.9, 1.5), 3),
        app_login_frequency=random.randint(1, 30),
        missed_bill_payments=random.randint(1 if high_risk else 0, 5 if high_risk else 1),
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