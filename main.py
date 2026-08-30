import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional, List
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from executor import executor

app = FastAPI(
    title="RecoverAI - Autonomous Payment Revenue Recovery Engine",
    description="Razorpay Track 3 Decision & Execution API",
    version="1.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

processed_event_store = set()
audit_log = []

# Load ML artifacts if present
try:
    model = joblib.load("recovery_model.pkl")
    preprocessor = joblib.load("preprocessor.pkl")
    imputer = joblib.load("imputer.pkl")
    print("All ML artifacts loaded successfully.")
except Exception as e:
    print(f"Warning: Artifact loading bypassed: {e}")
    model, preprocessor, imputer = None, None, None

class PaymentFailureEvent(BaseModel):
    payment_id: str = Field(..., example="pay_demo_100231")
    customer_id: str = Field(default="cust_981", example="cust_981")
    merchant_id: str = Field(default="mch_12", example="mch_12")
    amount_inr: float = Field(..., example=3499.0)
    payment_method: str = Field(default="UPI", example="UPI")
    failure_code: str = Field(..., example="INSUFFICIENT_BALANCE")
    failure_source: str = Field(default="bank", example="bank")
    attempt_number: int = Field(default=1, example=1)
    customer_success_rate: float = Field(default=0.85, example=0.85)
    customer_value: str = Field(default="HIGH", example="HIGH")
    created_at: Optional[str] = Field(default_factory=lambda: datetime.now().strftime("%d-%m-%Y %H:%M"))

class RecoveryDecisionResponse(BaseModel):
    payment_id: str
    status: str
    recommended_action: str
    calibrated_recovery_probability: float
    expected_net_recovery_inr: float
    human_escalation_flag: bool
    decision_rationale: str
    execution_result: dict

def execute_decision_engine(event: PaymentFailureEvent) -> dict:
    amount = float(event.amount_inr)
    failure_code = event.failure_code.upper()
    
    # 1. Deterministic Guardrails
    if failure_code == "RISK_FLAGGED" or amount >= 50000:
        return {
            "action": "HUMAN_REVIEW",
            "prob": 0.15,
            "net_val": max(0.0, amount * 0.15 - 30.0),
            "escalate": True,
            "rationale": "Transaction risk or high-value threshold (>= INR 50,000) exceeded."
        }
    
    if failure_code == "EXPIRED_CARD":
        return {
            "action": "ALTERNATE_METHOD",
            "prob": 0.65,
            "net_val": max(0.0, amount * 0.65 - 0.50),
            "escalate": False,
            "rationale": "Card instrument expired. Bypassed retry; triggered alternate payment method."
        }
    
    # 2. Probability Computation
    prob = 0.50
    if model and preprocessor and imputer:
        try:
            severity_map = {'NETWORK_TIMEOUT': 1, 'INSUFFICIENT_BALANCE': 2, 'BANK_DECLINE': 3, 'EXPIRED_CARD': 4, 'RISK_FLAGGED': 4}
            tier_map = {'LOW': 0.5, 'MEDIUM': 1.0, 'HIGH': 1.5}
            
            input_df = pd.DataFrame([{
                'payment_method': event.payment_method,
                'failure_code': failure_code,
                'failure_source': event.failure_source,
                'customer_value': event.customer_value.upper(),
                'action_tested': 'RETRY_6H',
                'amount_inr': amount,
                'amount_log': np.log1p(amount),
                'attempt_number': event.attempt_number,
                'customer_success_rate': event.customer_success_rate,
                'failure_severity': severity_map.get(failure_code, 2),
                'retry_decay': 1.0 / (event.attempt_number ** 1.5),
                'reliability_index': event.customer_success_rate * tier_map.get(event.customer_value.upper(), 1.0),
                'is_high_value': int(amount >= 50000),
                'expected_value_baseline': amount * event.customer_success_rate,
                'hour_of_day': datetime.now().hour
            }])
            
            processed = preprocessor.transform(input_df)
            imputed = imputer.transform(processed)
            prob = float(model.predict_proba(imputed)[:, 1][0])
        except Exception:
            prob = 0.50

    # 3. Action Selection & Cost Optimization
    RETRY_COST = 2.0
    nuisance_penalty = 50.0 if prob < 0.35 else 0.0
    expected_net = (prob * amount) - RETRY_COST - nuisance_penalty
    
    if failure_code == "INSUFFICIENT_BALANCE" and prob >= 0.35:
        action = "RETRY_6H"
        rationale = "Temporary balance deficit with high customer reliability. Scheduled 6h smart retry."
    elif failure_code == "NETWORK_TIMEOUT" and prob >= 0.40:
        action = "RETRY_IMMEDIATE"
        rationale = "Transient gateway/network failure. Triggered immediate automated retry."
    elif expected_net > 0 and prob >= 0.30:
        action = "PAYMENT_REMINDER"
        rationale = "Positive expected net value. Dispatched reminder link."
    else:
        action = "DO_NOTHING"
        expected_net = 0.0
        rationale = "Expected recovery is below cost/risk threshold. Intervention suppressed to protect customer experience."
        
    return {
        "action": action,
        "prob": round(prob, 4),
        "net_val": round(max(0.0, expected_net), 2),
        "escalate": False,
        "rationale": rationale
    }

@app.post("/webhook/payment_failed", response_model=RecoveryDecisionResponse, status_code=status.HTTP_200_OK)
async def handle_payment_failure_webhook(event: PaymentFailureEvent):
    if event.payment_id in processed_event_store:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Duplicate event: Payment ID {event.payment_id} has already been processed."
        )
    
    processed_event_store.add(event.payment_id)
    decision = execute_decision_engine(event)
    
    # 4. Trigger Autonomous Execution
    action = decision["action"]
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
        exec_res = {"status": "SKIPPED", "action_executed": "NONE", "message": "No action required."}
    
    record = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "payment_id": event.payment_id,
        "amount_inr": event.amount_inr,
        "failure_code": event.failure_code,
        "recommended_action": action,
        "recovery_probability": decision["prob"],
        "expected_net_recovery_inr": decision["net_val"],
        "human_escalation": decision["escalate"],
        "execution_status": exec_res["status"]
    }
    audit_log.insert(0, record)
    
    return RecoveryDecisionResponse(
        payment_id=event.payment_id,
        status="PROCESSED",
        recommended_action=action,
        calibrated_recovery_probability=decision["prob"],
        expected_net_recovery_inr=decision["net_val"],
        human_escalation_flag=decision["escalate"],
        decision_rationale=decision["rationale"],
        execution_result=exec_res
    )

@app.get("/audit_logs")
def get_audit_logs():
    return {
        "total_processed": len(audit_log),
        "total_recovered_projected": sum(r["expected_net_recovery_inr"] for r in audit_log),
        "escalations_count": sum(1 for r in audit_log if r["human_escalation"]),
        "recent_records": audit_log[:20]
    }

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "RecoverAI Decision Engine"}