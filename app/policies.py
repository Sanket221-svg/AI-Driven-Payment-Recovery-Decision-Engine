from __future__ import annotations

from typing import Any, Dict, Optional

from app.schemas import PaymentFailureEvent

POLICY_CONFIG = {
    "HIGH_VALUE_THRESHOLD": 50000.0,
    "MODEL_CONFIDENCE_THRESHOLD": 0.35,
    "MAX_RETRY_COUNT": 3,
    "MAX_DUPLICATE_ACTIONS": 1,
}


def evaluate_policy(
    payment: PaymentFailureEvent,
    action: str,
    model_probability: float,
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    context = context or {}

    if context.get("recovered") is True:
        return {
            "allowed": False,
            "reason": "PAYMENT_ALREADY_RECOVERED",
            "override_action": "STOP",
        }

    if context.get("terminal") is True:
        return {
            "allowed": False,
            "reason": "PAYMENT_TERMINAL",
            "override_action": "STOP",
        }

    if action == "DO_NOTHING":
        return {
            "allowed": False,
            "reason": "NO_ACTION_REQUIRED",
            "override_action": "STOP",
        }

    if bool(context.get("risk_flag")) or payment.failure_code.upper() == "RISK_FLAGGED":
        return {
            "allowed": False,
            "reason": "HIGH_RISK_PAYMENT_REQUIRES_HUMAN_REVIEW",
            "override_action": "ESCALATE_HUMAN",
        }

    if float(payment.amount_inr) > POLICY_CONFIG["HIGH_VALUE_THRESHOLD"]:
        return {
            "allowed": False,
            "reason": "HIGH_VALUE_PAYMENT_REQUIRES_HUMAN_REVIEW",
            "override_action": "ESCALATE_HUMAN",
        }

    retry_count_raw = context.get("retry_count")
    if retry_count_raw is None:
        retry_count_raw = getattr(payment, "retry_count", None)
    if retry_count_raw is None:
        retry_count_raw = getattr(payment, "attempt_number", 1)
    try:
        retry_count = int(retry_count_raw)
    except (TypeError, ValueError):
        retry_count = 1
    if retry_count >= POLICY_CONFIG["MAX_RETRY_COUNT"]:
        return {
            "allowed": False,
            "reason": "RETRY_LIMIT_REACHED",
            "override_action": "STOP",
        }

    if bool(context.get("duplicate_action")) or bool(getattr(payment, "duplicate_action", False)):
        return {
            "allowed": False,
            "reason": "DUPLICATE_ACTION_BLOCKED",
            "override_action": "BLOCK_DUPLICATE",
        }

    if bool(getattr(payment, "customer_opt_out", False)) and action in {"PAYMENT_REMINDER", "RETRY_IMMEDIATE", "RETRY_6H"}:
        return {
            "allowed": False,
            "reason": "CUSTOMER_OPTED_OUT_OF_COMMUNICATIONS",
            "override_action": "STOP",
        }

    if model_probability < POLICY_CONFIG["MODEL_CONFIDENCE_THRESHOLD"]:
        return {
            "allowed": False,
            "reason": "MODEL_CONFIDENCE_BELOW_THRESHOLD",
            "override_action": "ESCALATE_HUMAN",
        }

    if payment.failure_code.upper() == "EXPIRED_CARD" and action.startswith("RETRY"):
        return {
            "allowed": False,
            "reason": "EXPIRED_CARD_RETRY_NOT_ALLOWED",
            "override_action": "ALTERNATE_PAYMENT",
        }

    if payment.failure_code.upper() == "EXPIRED_CARD" and action == "ALTERNATE_PAYMENT":
        return {
            "allowed": True,
            "reason": "EXPIRED_CARD_ALTERNATE_PAYMENT_RECOMMENDED",
            "override_action": None,
        }

    return {
        "allowed": True,
        "reason": "RETRY_ALLOWED",
        "override_action": None,
    }
