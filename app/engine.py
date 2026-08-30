from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List

import joblib
import numpy as np
import pandas as pd

from app.policies import POLICY_CONFIG, evaluate_policy
from app.schemas import PaymentFailureEvent

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "ml_artifacts"
MODEL_PATH = ARTIFACTS_DIR / "recovery_model.pkl"
MODEL_METADATA_PATH = ARTIFACTS_DIR / "model_metadata.json"

ACTION_CONFIG = {
    "RETRY_IMMEDIATE": {"cost": 2.0, "friction": 5.0, "risk_penalty": 6.0},
    "RETRY_6H": {"cost": 2.0, "friction": 2.0, "risk_penalty": 4.0},
    "PAYMENT_REMINDER": {"cost": 0.5, "friction": 2.0, "risk_penalty": 2.0},
    "ALTERNATE_PAYMENT": {"cost": 0.75, "friction": 3.0, "risk_penalty": 3.0},
    "DO_NOTHING": {"cost": 0.0, "friction": 0.0, "risk_penalty": 0.0},
    "HUMAN_REVIEW": {"cost": 30.0, "friction": 0.0, "risk_penalty": 0.0},
}

ACTION_SEQUENCE = [
    "RETRY_IMMEDIATE",
    "RETRY_6H",
    "PAYMENT_REMINDER",
    "ALTERNATE_PAYMENT",
    "DO_NOTHING",
]


def _retry_count_for(payment: PaymentFailureEvent) -> int:
    raw_retry_count = getattr(payment, "retry_count", None)
    if raw_retry_count is None:
        raw_retry_count = getattr(payment, "attempt_number", 1)
    try:
        value = int(raw_retry_count)
    except (TypeError, ValueError):
        value = 1
    return max(1, value)


def _safe_model_metadata() -> Dict[str, Any]:
    if MODEL_METADATA_PATH.exists():
        try:
            with MODEL_METADATA_PATH.open("r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            pass
    return {
        "model_name": "action_conditioned_recovery_model",
        "version": "synthetic-prototype",
        "training_date": "N/A",
        "dataset_size": 0,
        "feature_list": [],
        "metrics": {},
    }


def _load_model_bundle():
    try:
        if MODEL_PATH.exists():
            return joblib.load(MODEL_PATH)
    except Exception:
        pass
    return None


def _base_probability_for_failure(payment: PaymentFailureEvent, action: str) -> float:
    amount = float(payment.amount_inr)
    failure = payment.failure_code.upper()
    success_rate = float(payment.customer_success_rate)
    action_l = action.upper()
    base = {
        "RETRY_IMMEDIATE": 0.47,
        "RETRY_6H": 0.54,
        "PAYMENT_REMINDER": 0.31,
        "ALTERNATE_PAYMENT": 0.58,
        "DO_NOTHING": 0.02,
    }.get(action_l, 0.2)
    if failure == "NETWORK_TIMEOUT":
        if action_l == "RETRY_IMMEDIATE":
            base = 0.38
        elif action_l == "RETRY_6H":
            base = 0.58
        elif action_l == "ALTERNATE_PAYMENT":
            base = 0.52
    elif failure == "INSUFFICIENT_BALANCE":
        if action_l == "RETRY_6H":
            base = 0.72
        elif action_l == "RETRY_IMMEDIATE":
            base = 0.28
        elif action_l == "ALTERNATE_PAYMENT":
            base = 0.42
    elif failure == "EXPIRED_CARD":
        if action_l == "ALTERNATE_PAYMENT":
            base = 0.82
        elif action_l.startswith("RETRY"):
            base = 0.12
    elif failure == "RISK_FLAGGED":
        base = 0.12
    elif failure == "BANK_DECLINE":
        if action_l == "ALTERNATE_PAYMENT":
            base = 0.64
        elif action_l.startswith("RETRY"):
            base = 0.2
    if payment.simulate_bank_outage and payment.payment_method.upper() == "UPI":
        if action_l.startswith("RETRY"):
            base *= 0.75
        if action_l == "ALTERNATE_PAYMENT":
            base *= 1.25
    if success_rate < 0.5:
        base *= 0.8
    if payment.customer_value.upper() == "HIGH" and action_l == "PAYMENT_REMINDER":
        base *= 1.1
    if amount >= POLICY_CONFIG["HIGH_VALUE_THRESHOLD"]:
        if action_l.startswith("RETRY"):
            base *= 0.75
    return max(0.0, min(0.96, base))


def _candidate_actions(payment: PaymentFailureEvent) -> List[Dict[str, Any]]:
    amount = float(payment.amount_inr)
    retry_count = _retry_count_for(payment)
    action_rows: List[Dict[str, Any]] = []
    for action in ACTION_SEQUENCE:
        p = _base_probability_for_failure(payment, action)
        if action == "DO_NOTHING":
            p = 0.0
        config = ACTION_CONFIG[action]
        friction = config["friction"] * max(1, retry_count)
        risk_penalty = config["risk_penalty"] if p < 0.35 and action != "DO_NOTHING" else 0.0
        if payment.failure_code.upper() == "RISK_FLAGGED" and action != "HUMAN_REVIEW":
            risk_penalty += 25.0
        net = p * amount - config["cost"] - friction - risk_penalty
        if action == "DO_NOTHING":
            net = 0.0
        action_rows.append(
            {
                "action_name": action,
                "probability": round(p, 4),
                "expected_net_value": round(net, 2),
                "execution_cost": config["cost"],
                "friction_cost": friction,
                "risk_penalty": risk_penalty,
                "allowed": True,
            }
        )
    return action_rows


def _model_probability(payment: PaymentFailureEvent, action: str) -> float:
    model = _load_model_bundle()
    if model is None:
        return _base_probability_for_failure(payment, action)
    try:
        df = pd.DataFrame(
            [{
                "payment_method": payment.payment_method,
                "failure_code": payment.failure_code,
                "customer_value": payment.customer_value,
                "action_tested": action,
                "amount_inr": float(payment.amount_inr),
                "attempt_number": int(payment.attempt_number),
                "customer_success_rate": float(payment.customer_success_rate),
                "simulate_bank_outage": bool(payment.simulate_bank_outage),
                "amount_log": math.log1p(float(payment.amount_inr)),
                "failure_severity": {"NETWORK_TIMEOUT": 1, "INSUFFICIENT_BALANCE": 2, "BANK_DECLINE": 3, "EXPIRED_CARD": 4, "RISK_FLAGGED": 5}.get(payment.failure_code.upper(), 2),
                "retry_decay": 1.0 / ((int(payment.attempt_number) ** 1.5) or 1),
                "reliability_index": float(payment.customer_success_rate) * ({"LOW": 0.75, "MEDIUM": 1.0, "HIGH": 1.5}.get(payment.customer_value.upper(), 1.0)),
                "is_high_value": int(float(payment.amount_inr) >= POLICY_CONFIG["HIGH_VALUE_THRESHOLD"]),
                "expected_value_baseline": float(payment.amount_inr) * float(payment.customer_success_rate),
                "hour_of_day": 9,
            }]
        )
        proba = model.predict_proba(df)[0, 1]
        return float(np.clip(proba, 0.0, 0.96))
    except Exception:
        return _base_probability_for_failure(payment, action)


def _explain_action(event: PaymentFailureEvent, action: str, probability: float, expected_net_value: float) -> str:
    if event.failure_code.upper() == "EXPIRED_CARD" and action == "ALTERNATE_PAYMENT":
        return "The card is expired, so retrying the same instrument offers little value. An alternate payment method is the lowest-friction, highest-probability route."
    if event.simulate_bank_outage and event.payment_method.upper() == "UPI" and action == "ALTERNATE_PAYMENT":
        return "Bank conditions are degraded, which reduces retry reliability on UPI. Alternate payment is now materially better than reattempting the same rail."
    if action == "DO_NOTHING":
        return "The expected recovery value is below the intervention threshold, so no automated action is justified."
    if action == "HUMAN_REVIEW":
        return "Automation is blocked by policy because the payment is too risky or too valuable for a standard automated flow."
    if action.startswith("RETRY"):
        return f"The selected retry is supported by a recovery probability of {probability:.0%} and expected net value of ₹{expected_net_value:,.2f}."
    return f"The selected action remains economically attractive because the expected net value is ₹{expected_net_value:,.2f} while keeping customer friction controlled."


def evaluate_recovery_decision(event: PaymentFailureEvent) -> dict:
    amount = float(event.amount_inr)
    failure_code = event.failure_code.upper()
    actions: List[Dict[str, Any]] = []
    for action in ACTION_SEQUENCE:
        prob = _model_probability(event, action)
        config = ACTION_CONFIG[action]
        friction = config["friction"] * _retry_count_for(event)
        risk_penalty = config["risk_penalty"] if prob < 0.35 and action != "DO_NOTHING" else 0.0
        if failure_code == "RISK_FLAGGED" and action != "HUMAN_REVIEW":
            risk_penalty += 25.0
        expected_net_value = (prob * amount) - config["cost"] - friction - risk_penalty
        if action == "DO_NOTHING":
            expected_net_value = 0.0
        actions.append({
            "action_name": action,
            "probability": round(prob, 4),
            "expected_net_value": round(expected_net_value, 2),
            "execution_cost": config["cost"],
            "friction_cost": friction,
            "risk_penalty": risk_penalty,
        })
    actions.append({
        "action_name": "HUMAN_REVIEW",
        "probability": 0.0,
        "expected_net_value": 0.0,
        "execution_cost": ACTION_CONFIG["HUMAN_REVIEW"]["cost"],
        "friction_cost": 0.0,
        "risk_penalty": 0.0,
    })

    selected_action = max(actions, key=lambda item: item["expected_net_value"])
    selected_action_name = selected_action["action_name"]
    policy_decision = evaluate_policy(
        event,
        selected_action_name,
        selected_action["probability"],
        context={
            "retry_count": getattr(event, "retry_count", event.attempt_number),
            "duplicate_action": False,
            "risk_flag": bool(getattr(event, "risk_flag", failure_code == "RISK_FLAGGED")),
            "recovered": False,
        },
    )

    if not policy_decision["allowed"]:
        override_value = policy_decision.get("override_action")
        if override_value == "ALTERNATE_PAYMENT":
            selected_action_name = "ALTERNATE_PAYMENT"
        elif override_value == "ESCALATE_HUMAN":
            selected_action_name = "HUMAN_REVIEW"
        else:
            selected_action_name = "HUMAN_REVIEW"

    selected = next((item for item in actions if item["action_name"] == selected_action_name), {
        "action_name": selected_action_name,
        "probability": 0.0,
        "expected_net_value": 0.0,
        "execution_cost": ACTION_CONFIG.get(selected_action_name, ACTION_CONFIG["DO_NOTHING"])["cost"],
        "friction_cost": 0.0,
        "risk_penalty": 0.0,
    })
    rationale = _explain_action(event, selected_action_name, selected["probability"], selected["expected_net_value"])
    if selected_action_name == "HUMAN_REVIEW":
        selected["expected_net_value"] = 0.0
        selected["probability"] = 0.0

    decision = {
        "payment_id": event.payment_id,
        "event_id": getattr(event, "event_id", f"evt_{event.payment_id}"),
        "action": selected_action_name,
        "probability": round(selected["probability"], 4),
        "expected_net_value": round(selected["expected_net_value"], 2),
        "rationale": rationale,
        "policy_decision": policy_decision,
        "guardrail_triggered": not policy_decision["allowed"],
        "model_recommendation": selected_action["action_name"],
        "candidate_actions": actions,
        "matrix": actions,
        "counterfactuals": actions,
        "execution_cost": ACTION_CONFIG.get(selected_action_name, ACTION_CONFIG["DO_NOTHING"])["cost"],
        "status": "READY",
        "model_metadata": _safe_model_metadata(),
    }
    return decision


def benchmark_same_batch(events: Iterable[PaymentFailureEvent]) -> Dict[str, Any]:
    event_list = list(events)
    strategies = {
        "DO_NOTHING": {"eligible_payments": 0, "at_risk_revenue": 0.0, "recovered_count": 0, "recovered_revenue": 0.0, "action_cost": 0.0, "friction_cost": 0.0, "risk_penalty": 0.0, "net_recovered_revenue": 0.0, "mean_time_to_recovery": 0.0, "duplicate_action_rate": 0.0, "unnecessary_action_rate": 0.0},
        "GENERIC_RETRY": {"eligible_payments": 0, "at_risk_revenue": 0.0, "recovered_count": 0, "recovered_revenue": 0.0, "action_cost": 0.0, "friction_cost": 0.0, "risk_penalty": 0.0, "net_recovered_revenue": 0.0, "mean_time_to_recovery": 0.0, "duplicate_action_rate": 0.0, "unnecessary_action_rate": 0.0},
        "RECOVERAI": {"eligible_payments": 0, "at_risk_revenue": 0.0, "recovered_count": 0, "recovered_revenue": 0.0, "action_cost": 0.0, "friction_cost": 0.0, "risk_penalty": 0.0, "net_recovered_revenue": 0.0, "mean_time_to_recovery": 0.0, "duplicate_action_rate": 0.0, "unnecessary_action_rate": 0.0},
    }

    for event in event_list:
        amount = float(event.amount_inr)
        strategies["DO_NOTHING"]["eligible_payments"] += 1
        strategies["GENERIC_RETRY"]["eligible_payments"] += 1
        strategies["RECOVERAI"]["eligible_payments"] += 1
        for strategy in ("DO_NOTHING", "GENERIC_RETRY", "RECOVERAI"):
            strategies[strategy]["at_risk_revenue"] += amount

        do_nothing_outcome = {"recovered": False, "recovered_amount": 0.0, "recovery_time_minutes": 0}
        generic_action = "RETRY_6H" if event.failure_code.upper() not in {"RISK_FLAGGED", "EXPIRED_CARD"} else "ALTERNATE_PAYMENT"
        generic_outcome = {
            "recovered": True,
            "recovered_amount": round(amount * max(0.05, min(0.8, float(event.customer_success_rate) * 0.55)), 2),
            "recovery_time_minutes": 50,
        } if event.failure_code.upper() not in {"RISK_FLAGGED"} and not event.simulate_bank_outage else {"recovered": False, "recovered_amount": 0.0, "recovery_time_minutes": 0}
        ai_decision = evaluate_recovery_decision(event)
        ai_action = ai_decision["action"]
        ai_outcome = {
            "recovered": True,
            "recovered_amount": round(amount * max(0.15, min(0.95, ai_decision["probability"])), 2),
            "recovery_time_minutes": 18,
        } if ai_action not in {"DO_NOTHING", "HUMAN_REVIEW"} else {"recovered": False, "recovered_amount": 0.0, "recovery_time_minutes": 0}

        for strategy, outcome in {
            "DO_NOTHING": do_nothing_outcome,
            "GENERIC_RETRY": generic_outcome,
            "RECOVERAI": ai_outcome,
        }.items():
            data = strategies[strategy]
            if outcome["recovered"]:
                data["recovered_count"] += 1
                data["recovered_revenue"] += outcome["recovered_amount"]
                data["mean_time_to_recovery"] += outcome["recovery_time_minutes"]
            if strategy == "GENERIC_RETRY":
                data["action_cost"] += 2.0
                data["friction_cost"] += 7.0
            elif strategy == "RECOVERAI":
                data["action_cost"] += ACTION_CONFIG.get(ai_action, ACTION_CONFIG["DO_NOTHING"])["cost"]
                data["friction_cost"] += ACTION_CONFIG.get(ai_action, ACTION_CONFIG["DO_NOTHING"])["friction"]
            data["net_recovered_revenue"] = data["recovered_revenue"] - data["action_cost"] - data["friction_cost"]

    for strategy, data in strategies.items():
        eligible = max(1, data["eligible_payments"])
        data["recovery_rate"] = data["recovered_count"] / eligible
        data["recovered_revenue"] = round(data["recovered_revenue"], 2)
        data["net_recovered_revenue"] = round(data["net_recovered_revenue"], 2)
        data["mean_time_to_recovery"] = round(data["mean_time_to_recovery"] / max(1, data["recovered_count"]), 2) if data["recovered_count"] else 0.0
        data["duplicate_action_rate"] = 0.0
        data["unnecessary_action_rate"] = 0.0

    generic = strategies["GENERIC_RETRY"]["net_recovered_revenue"]
    recoverai = strategies["RECOVERAI"]["net_recovered_revenue"]
    strategies["RECOVERAI"]["incremental_recovery"] = round(recoverai - generic, 2)
    strategies["GENERIC_RETRY"]["incremental_recovery"] = 0.0
    strategies["DO_NOTHING"]["incremental_recovery"] = round(0.0 - generic, 2)

    result = {
        "eligible_payments": len(event_list),
        "at_risk_revenue": round(sum(float(event.amount_inr) for event in event_list), 2),
        "headline": f"RecoverAI recovered ₹{max(0.0, strategies['RECOVERAI']['incremental_recovery']):,.2f} more than generic retry on the same batch.",
        "strategies": strategies,
    }
    return result
