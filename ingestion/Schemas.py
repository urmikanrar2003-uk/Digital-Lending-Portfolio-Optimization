from pydantic import BaseModel, Field
from typing import Optional, Literal
from datetime import datetime
import uuid


class LoanApplicationEvent(BaseModel):
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    event_type: Literal["loan_application"] = "loan_application"
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    customer_id: str
    age: int
    income_monthly: float
    employment_type: Literal["salaried", "self_employed", "gig", "unemployed"]
    city_tier: Literal["tier1", "tier2", "tier3"]
    credit_score_proxy: float        # 300–900

    loan_amount: float
    loan_tenure_months: int
    loan_product: Literal["personal", "bnpl", "sme_working_capital"]
    interest_rate: float
    origination_channel: Literal["app", "web", "agent", "partner"]

    approval_status: Literal["approved", "rejected", "manual_review"]
    risk_grade: Literal["A", "B", "C", "D", "E"]


class RepaymentEvent(BaseModel):
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    event_type: Literal["repayment"] = "repayment"
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    customer_id: str
    loan_id: str
    emi_due_date: datetime
    emi_amount: float
    amount_paid: float
    days_past_due: int                # 0 = on-time
    payment_mode: Literal["upi", "netbanking", "auto_debit", "cash"]
    is_partial: bool


class BehavioralEvent(BaseModel):
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    event_type: Literal["behavioral_signal"] = "behavioral_signal"
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    customer_id: str
    avg_monthly_balance: float
    balance_volatility: float         # std dev of daily balance
    num_spending_shocks: int          # large unexpected debits in last 30d
    cash_flow_ratio: float            # inflow / outflow
    app_login_frequency: int          # logins in last 30d
    missed_bill_payments: int