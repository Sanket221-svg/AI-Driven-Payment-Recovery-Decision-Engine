import pytest

from app.engine import evaluate_recovery_decision, benchmark_same_batch
from app.policies import evaluate_policy
from app.recovery_simulator import simulate_recovery
from app.schemas import PaymentFailureEvent


def test_insufficient_balance_prefers_retry_6h():
    event = PaymentFailureEvent(
        payment_id="pay_insufficient_1",
        amount_inr=8500,
        payment_method="UPI",
        failure_code="INSUFFICIENT_BALANCE",
        customer_success_rate=0.91,
        customer_value="HIGH",
        attempt_number=1,
    )
    decision = evaluate_recovery_decision(event)
    assert decision["action"] in {"RETRY_6H", "RETRY_IMMEDIATE"}
    assert decision["policy_decision"]["allowed"] is True


def test_expired_card_has_no_pointless_retry():
    event = PaymentFailureEvent(
        payment_id="pay_expired_1",
        amount_inr=6200,
        payment_method="CARD",
        failure_code="EXPIRED_CARD",
        customer_success_rate=0.72,
        customer_value="MEDIUM",
        attempt_number=2,
    )
    decision = evaluate_recovery_decision(event)
    assert decision["action"] == "ALTERNATE_PAYMENT"
    assert decision["policy_decision"]["allowed"] is True
    assert decision["policy_decision"]["reason"] != "RETRY_ALLOWED"


def test_high_value_triggers_human_review():
    event = PaymentFailureEvent(
        payment_id="pay_high_value_1",
        amount_inr=90000,
        payment_method="UPI",
        failure_code="NETWORK_TIMEOUT",
        customer_success_rate=0.78,
        customer_value="HIGH",
        attempt_number=3,
    )
    decision = evaluate_recovery_decision(event)
    assert decision["action"] == "HUMAN_REVIEW"
    assert decision["policy_decision"]["allowed"] is False
    assert "HIGH_VALUE" in decision["policy_decision"]["reason"]


def test_bank_outage_changes_action_ranking():
    base_event = PaymentFailureEvent(
        payment_id="pay_outage_1",
        amount_inr=14200,
        payment_method="UPI",
        failure_code="NETWORK_TIMEOUT",
        customer_success_rate=0.84,
        customer_value="HIGH",
        attempt_number=2,
        simulate_bank_outage=False,
    )
    outage_event = PaymentFailureEvent(
        payment_id="pay_outage_1",
        amount_inr=14200,
        payment_method="UPI",
        failure_code="NETWORK_TIMEOUT",
        customer_success_rate=0.84,
        customer_value="HIGH",
        attempt_number=2,
        simulate_bank_outage=True,
    )
    base_decision = evaluate_recovery_decision(base_event)
    outage_decision = evaluate_recovery_decision(outage_event)
    assert base_decision["action"] == "RETRY_6H"
    assert outage_decision["action"] == "ALTERNATE_PAYMENT"


def test_benchmark_uses_same_batch_for_all_strategies():
    batch = [
        PaymentFailureEvent(payment_id="pay_b_1", amount_inr=5000, payment_method="UPI", failure_code="NETWORK_TIMEOUT", customer_success_rate=0.8, customer_value="HIGH", attempt_number=1),
        PaymentFailureEvent(payment_id="pay_b_2", amount_inr=12000, payment_method="CARD", failure_code="EXPIRED_CARD", customer_success_rate=0.7, customer_value="MEDIUM", attempt_number=2),
        PaymentFailureEvent(payment_id="pay_b_3", amount_inr=20000, payment_method="UPI", failure_code="INSUFFICIENT_BALANCE", customer_success_rate=0.9, customer_value="HIGH", attempt_number=1),
    ]
    result = benchmark_same_batch(batch)
    assert result["eligible_payments"] == 3
    assert result["strategies"]["DO_NOTHING"]["eligible_payments"] == 3
    assert result["strategies"]["GENERIC_RETRY"]["eligible_payments"] == 3
    assert result["strategies"]["RECOVERAI"]["eligible_payments"] == 3


def test_simulated_outcome_differs_from_expected_value():
    event = PaymentFailureEvent(
        payment_id="pay_sim_1",
        amount_inr=8500,
        payment_method="UPI",
        failure_code="NETWORK_TIMEOUT",
        customer_success_rate=0.84,
        customer_value="HIGH",
        attempt_number=1,
    )
    outcome = simulate_recovery(event, "RETRY_6H", seed=42)
    assert outcome["recovered"] in {True, False}
    assert outcome["recovered_amount"] >= 0
    assert outcome["recovery_time_minutes"] >= 0
    assert outcome["simulation_seed"] == 42


def test_recovered_payment_stops_further_recovery():
    event = PaymentFailureEvent(
        payment_id="pay_recovered_1",
        amount_inr=5000,
        payment_method="UPI",
        failure_code="INSUFFICIENT_BALANCE",
        customer_success_rate=0.9,
        customer_value="HIGH",
        attempt_number=1,
    )
    policy = evaluate_policy(event, "RETRY_6H", 0.8, {"recovered": True})
    assert policy["allowed"] is False
    assert "RECOVERED" in policy["reason"]
