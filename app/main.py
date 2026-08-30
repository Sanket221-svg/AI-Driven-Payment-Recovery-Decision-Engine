from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware

from app.audit import append_audit_entry, get_audit_records, has_processed_event, record_webhook_event
from app.benchmark import benchmark_same_batch as _engine_benchmark
from app.engine import evaluate_recovery_decision
from app.executor import executor
from app.recovery_simulator import simulate_recovery
from app.schemas import PaymentFailureEvent, RecoveryDecisionResponse

app = FastAPI(
    title="RecoverAI - Autonomous Payment Revenue Recovery Agent",
    description="AI revenue recovery decision engine with policy guardrails and bounded execution.",
    version="2.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health_check() -> Dict[str, Any]:
    return {"status": "healthy", "service": "RecoverAI", "mode": "prototype"}


@app.post("/api/analyze")
def analyze_payment(event: PaymentFailureEvent) -> Dict[str, Any]:
    decision = evaluate_recovery_decision(event)
    return decision


@app.post("/api/execute")
def execute_payment(event: PaymentFailureEvent) -> Dict[str, Any]:
    decision = evaluate_recovery_decision(event)
    action = decision["action"]
    execution = executor.execute_action(event, action)
    outcome = simulate_recovery(event, action, seed=42)
    return {
        "payment_id": event.payment_id,
        "decision": decision,
        "execution": execution,
        "outcome": outcome,
        "status": "EXECUTED",
    }


@app.post("/api/simulate-recovery")
def simulate_action(payload: Dict[str, Any]) -> Dict[str, Any]:
    payment = PaymentFailureEvent(**payload.get("payment", {}))
    action = payload.get("action", "RETRY_6H")
    seed = payload.get("seed", 42)
    return simulate_recovery(payment, action, seed=int(seed))


@app.post("/api/webhook")
def process_webhook(event: PaymentFailureEvent) -> Dict[str, Any]:
    event_id = event.event_id or f"evt_{event.payment_id}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
    if has_processed_event(event_id):
        raise HTTPException(status_code=409, detail={"status": "duplicate", "message": "Duplicate webhook event ignored."})
    record_webhook_event(event_id, "payment_failed", event.model_dump(), "processed", {"payment_id": event.payment_id})

    decision = evaluate_recovery_decision(event)
    action = decision["action"]
    execution = executor.execute_action(event, action)
    outcome = simulate_recovery(event, action, seed=42)
    record = {
        "event_id": event_id,
        "payment_id": event.payment_id,
        "timestamp": datetime.utcnow().isoformat(),
        "failure_reason": event.failure_code,
        "model_probabilities": {k["action_name"]: k["probability"] for k in decision["candidate_actions"]},
        "candidate_actions": [item["action_name"] for item in decision["candidate_actions"]],
        "selected_action": action,
        "policy_decision": decision["policy_decision"],
        "execution_status": execution.get("status"),
        "outcome": "RECOVERED" if outcome["recovered"] else "FAILED",
        "recovered_amount": outcome["recovered_amount"],
        "recovery_time_minutes": outcome["recovery_time_minutes"],
        "human_override": "ESCALATE_HUMAN" if decision["guardrail_triggered"] else None,
    }
    append_audit_entry(record)

    return {
        "event_id": event_id,
        "payment_id": event.payment_id,
        "decision": decision,
        "execution": execution,
        "outcome": outcome,
        "audit_record": record,
        "status": "PROCESSED",
    }


@app.get("/api/audit")
def fetch_audit(payment_id: Optional[str] = Query(default=None), action: Optional[str] = None, status: Optional[str] = None, recovered: Optional[bool] = None):
    return get_audit_records({
        "payment_id": payment_id,
        "action": action,
        "status": status,
        "recovered": recovered,
    })


@app.get("/api/metrics")
def metrics_summary() -> Dict[str, Any]:
    records = get_audit_records()
    total_recovered = sum(item.get("recovered_amount", 0) for item in records)
    return {
        "total_audit_entries": len(records),
        "total_recovered_revenue": round(total_recovered, 2),
        "recovery_rate": round((sum(1 for item in records if item.get("outcome") == "RECOVERED") / max(1, len(records))), 4),
        "human_review_count": sum(1 for item in records if item.get("execution_status") == "ESCALATED"),
    }


@app.get("/api/model/info")
def model_info() -> Dict[str, Any]:
    metadata = evaluate_recovery_decision(PaymentFailureEvent(
        payment_id="meta_probe",
        amount_inr=1000,
        payment_method="UPI",
        failure_code="NETWORK_TIMEOUT",
        customer_success_rate=0.8,
        customer_value="HIGH",
        attempt_number=1,
    ))["model_metadata"]
    return {
        "model_name": metadata.get("model_name", "action_conditioned_recovery_model"),
        "version": metadata.get("version", "1.0.0"),
        "training_date": metadata.get("training_date", "N/A"),
        "dataset_size": metadata.get("dataset_size", 0),
        "feature_list": metadata.get("feature_list", []),
        "metrics": metadata.get("metrics", {}),
    }


@app.get("/api/model/metrics")
def model_metrics() -> Dict[str, Any]:
    return evaluate_recovery_decision(PaymentFailureEvent(
        payment_id="meta_probe",
        amount_inr=1000,
        payment_method="UPI",
        failure_code="NETWORK_TIMEOUT",
        customer_success_rate=0.8,
        customer_value="HIGH",
        attempt_number=1,
    ))["model_metadata"].get("metrics", {})


@app.get("/api/benchmark")
def benchmark() -> Dict[str, Any]:
    batch = [
        PaymentFailureEvent(payment_id=f"bench_{i:03d}", amount_inr=4500 + i * 200, payment_method="UPI", failure_code="NETWORK_TIMEOUT", customer_success_rate=0.85, customer_value="HIGH", attempt_number=1) for i in range(6)
    ]
    return _engine_benchmark(batch)


@app.post("/webhook/payment_failed", response_model=RecoveryDecisionResponse, status_code=status.HTTP_200_OK)
async def handle_payment_failed_webhook(event: PaymentFailureEvent):
    response = process_webhook(event)
    return RecoveryDecisionResponse(
        payment_id=event.payment_id,
        status=response["status"],
        recommended_action=response["decision"]["action"],
        calibrated_recovery_probability=response["decision"]["probability"],
        expected_net_recovery_inr=response["decision"]["expected_net_value"],
        costs_saved_inr=0.0,
        human_escalation_flag=response["decision"]["guardrail_triggered"],
        decision_rationale=response["decision"]["rationale"],
        guardrail_triggered=response["decision"]["guardrail_triggered"],
        model_recommendation=response["decision"]["model_recommendation"],
        policy_decision=response["decision"]["policy_decision"],
        policy_reason=response["decision"]["policy_decision"].get("reason"),
        counterfactual_matrix=[{**item} for item in response["decision"]["candidate_actions"]],
        counterfactuals=[{**item} for item in response["decision"]["candidate_actions"]],
        execution_result=response["execution"],
    )


@app.get("/audit_logs")
def fetch_audit_logs():
    rows = get_audit_records()
    return {
        "total_processed": len(rows),
        "total_projected_net_recovery": sum(float(r.get("recovered_amount", 0)) for r in rows),
        "total_costs_saved": 0.0,
        "escalations_count": sum(1 for r in rows if r.get("human_override")),
        "recent_records": rows[:20],
    }


@app.get("/health")
def legacy_health_check():
    return health_check()
