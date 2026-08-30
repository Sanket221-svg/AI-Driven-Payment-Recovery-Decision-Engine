from datetime import datetime
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from app.schemas import PaymentFailureEvent, RecoveryDecisionResponse
from app.engine import evaluate_recovery_decision
from app.executor import executor

app = FastAPI(
    title="RecoverAI - Production Decision & Action API",
    description="Track 3 Microservice Architecture for Autonomous Revenue Recovery",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

idempotency_store = set()
audit_log_store = []

@app.post("/webhook/payment_failed", response_model=RecoveryDecisionResponse, status_code=status.HTTP_200_OK)
async def handle_payment_failed_webhook(event: PaymentFailureEvent):
    # 1. Idempotency Gate
    if event.payment_id in idempotency_store:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Duplicate event detected: Payment ID {event.payment_id} is already processed."
        )
    idempotency_store.add(event.payment_id)

    # 2. Decision Engine Execution
    decision = evaluate_recovery_decision(event)
    action = decision["action"]

    # 3. Dispatch to Bounded Executor
    if action in ["RETRY_6H", "RETRY_IMMEDIATE"]:
        delay = 0 if action == "RETRY_IMMEDIATE" else 6
        exec_res = executor.schedule_retry(event.payment_id, delay)
    elif action == "PAYMENT_REMINDER":
        exec_res = executor.send_payment_reminder(event.payment_id, event.customer_id, event.amount_inr)
    elif action == "ALTERNATE_METHOD":
        exec_res = executor.suggest_alternate_method(event.payment_id, event.customer_id)
    elif action == "HUMAN_REVIEW":
        exec_res = executor.escalate_human(event.payment_id, decision["rationale"], event.amount_inr)
    else:
        exec_res = {"status": "SKIPPED", "action_executed": "NONE", "message": "Intervention suppressed by policy."}

    # 4. Immutable Audit Log
    record = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "payment_id": event.payment_id,
        "customer_id": event.customer_id,
        "amount_inr": event.amount_inr,
        "failure_code": event.failure_code,
        "action": action,
        "recovery_probability": decision["prob"],
        "expected_net_recovery_inr": decision["net_val"],
        "costs_saved_inr": decision.get("costs_saved", 0.0),
        "human_escalation": decision["escalate"],
        "execution_status": exec_res.get("status")
    }
    audit_log_store.insert(0, record)

    counterfactuals = decision.get("matrix", [])
    return RecoveryDecisionResponse(
        payment_id=event.payment_id,
        status="PROCESSED",
        recommended_action=action,
        calibrated_recovery_probability=decision["prob"],
        expected_net_recovery_inr=decision["net_val"],
        costs_saved_inr=decision.get("costs_saved", 0.0),
        human_escalation_flag=decision["escalate"],
        decision_rationale=decision["rationale"],
        guardrail_triggered=decision.get("guardrail_triggered", False),
        model_recommendation=decision.get("model_recommendation"),
        policy_decision=decision.get("policy_decision", action),
        policy_reason=decision.get("policy_reason"),
        counterfactual_matrix=[{
            "action_name": sim.get("action_name"),
            "probability": sim.get("probability"),
            "expected_net_value": sim.get("expected_net_value"),
            "expected_regret": sim.get("expected_regret"),
            "execution_cost": sim.get("execution_cost")
        } for sim in counterfactuals],
        counterfactuals=[{
            "action_name": sim.get("action_name"),
            "probability": sim.get("probability"),
            "expected_net_value": sim.get("expected_net_value"),
            "expected_regret": sim.get("expected_regret"),
            "execution_cost": sim.get("execution_cost")
        } for sim in counterfactuals],
        execution_result=exec_res
    )

@app.get("/audit_logs")
def fetch_audit_logs():
    return {
        "total_processed": len(audit_log_store),
        "total_projected_net_recovery": sum(r.get("expected_net_recovery_inr", 0) for r in audit_log_store),
        "total_costs_saved": sum(r.get("costs_saved_inr", 0) for r in audit_log_store),
        "escalations_count": sum(1 for r in audit_log_store if r.get("human_escalation")),
        "recent_records": audit_log_store[:20]
    }

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "RecoverAI Microservice v2.0"}