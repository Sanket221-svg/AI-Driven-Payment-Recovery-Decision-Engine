from __future__ import annotations

import random
from typing import Any, Dict

from app.schemas import PaymentFailureEvent


def simulate_recovery(payment: PaymentFailureEvent, action: str, seed: int | None = None) -> Dict[str, Any]:
    rng = random.Random(seed if seed is not None else (hash(f"{payment.payment_id}:{action}") % (2 ** 32)))
    amount = float(payment.amount_inr)
    failure = payment.failure_code.upper()
    action_key = action.upper()

    base_probability = {
        "RETRY_IMMEDIATE": 0.62,
        "RETRY_6H": 0.68,
        "PAYMENT_REMINDER": 0.32,
        "ALTERNATE_PAYMENT": 0.55,
        "DO_NOTHING": 0.02,
        "HUMAN_REVIEW": 0.12,
    }

    if failure == "NETWORK_TIMEOUT":
        base_probability.update({"RETRY_IMMEDIATE": 0.66, "RETRY_6H": 0.55, "ALTERNATE_PAYMENT": 0.48, "PAYMENT_REMINDER": 0.28})
    elif failure == "INSUFFICIENT_BALANCE":
        base_probability.update({"RETRY_IMMEDIATE": 0.28, "RETRY_6H": 0.64, "ALTERNATE_PAYMENT": 0.36, "PAYMENT_REMINDER": 0.22})
    elif failure == "EXPIRED_CARD":
        base_probability.update({"RETRY_IMMEDIATE": 0.08, "RETRY_6H": 0.16, "ALTERNATE_PAYMENT": 0.78, "PAYMENT_REMINDER": 0.25})
    elif failure == "RISK_FLAGGED":
        base_probability.update({"RETRY_IMMEDIATE": 0.08, "RETRY_6H": 0.08, "ALTERNATE_PAYMENT": 0.16, "PAYMENT_REMINDER": 0.10, "HUMAN_REVIEW": 0.20})
    elif failure == "BANK_DECLINE":
        base_probability.update({"RETRY_IMMEDIATE": 0.18, "RETRY_6H": 0.22, "ALTERNATE_PAYMENT": 0.52, "PAYMENT_REMINDER": 0.18})

    if payment.simulate_bank_outage and payment.payment_method.upper() == "UPI":
        base_probability["RETRY_IMMEDIATE"] = max(0.05, base_probability["RETRY_IMMEDIATE"] - 0.22)
        base_probability["RETRY_6H"] = max(0.05, base_probability["RETRY_6H"] - 0.18)
        base_probability["ALTERNATE_PAYMENT"] = min(0.9, base_probability["ALTERNATE_PAYMENT"] + 0.18)

    if payment.customer_success_rate < 0.5:
        base_probability["RETRY_IMMEDIATE"] *= 0.8
        base_probability["RETRY_6H"] *= 0.8

    if payment.customer_value.upper() == "HIGH":
        base_probability["PAYMENT_REMINDER"] += 0.05

    probability = max(0.0, min(0.96, base_probability.get(action_key, 0.2)))

    if action_key in {"DO_NOTHING", "HUMAN_REVIEW"}:
        probability = max(0.0, min(probability, 0.15))

    recovered = rng.random() < probability
    recovered_amount = amount if recovered else 0.0
    recovery_time_minutes = int(rng.randint(8, 180)) if recovered else 0
    reason = "Recovered under the selected intervention" if recovered else "Recovery condition not met; no monetary recovery recorded"

    return {
        "recovered": recovered,
        "recovered_amount": round(recovered_amount, 2),
        "recovery_time_minutes": recovery_time_minutes,
        "reason": reason,
        "simulation_seed": seed,
        "actual_probability": round(probability, 4),
    }
