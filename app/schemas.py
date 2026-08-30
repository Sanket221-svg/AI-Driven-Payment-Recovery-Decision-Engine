from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime

class PaymentFailureEvent(BaseModel):
    payment_id: str = Field(..., example="pay_demo_100231")
    customer_id: str = Field(default="cust_default")
    merchant_id: str = Field(default="mch_default")
    amount_inr: float = Field(..., example=3499.0)
    payment_method: str = Field(default="UPI")
    failure_code: str = Field(..., example="INSUFFICIENT_BALANCE")
    failure_source: str = Field(default="bank")
    attempt_number: int = Field(default=1, ge=1)
    customer_success_rate: float = Field(default=0.85, ge=0.0, le=1.0)
    customer_value: str = Field(default="HIGH")
    simulate_bank_outage: bool = Field(default=False, description="Stress test toggle")
    created_at: Optional[str] = Field(default_factory=lambda: datetime.now().strftime("%d-%m-%Y %H:%M"))

class ActionSimulation(BaseModel):
    action_name: str
    probability: float
    expected_net_value: float
    expected_regret: float
    execution_cost: float

class RecoveryDecisionResponse(BaseModel):
    payment_id: str
    status: str
    recommended_action: str
    calibrated_recovery_probability: float
    expected_net_recovery_inr: float
    costs_saved_inr: float
    human_escalation_flag: bool
    decision_rationale: str
    guardrail_triggered: bool = False
    model_recommendation: Optional[str] = None
    policy_decision: Optional[str] = None
    policy_reason: Optional[str] = None
    counterfactual_matrix: List[ActionSimulation] = Field(default_factory=list)
    counterfactuals: List[Dict[str, Any]] = Field(default_factory=list)
    execution_result: dict