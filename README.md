# **RecoverAI: Intelligent Payment Recovery Decision Engine**
**From Smart Retry to Smart Recovery.**

Not every failed payment should be retried. RecoverAI is a cost-aware, risk-aware payment recovery decision engine built for the FinSpark'26 Hackathon by Team Byte Battle. Instead of relying on naive "blind retry" bots that waste API gateway fees and ignore business context, RecoverAI evaluates counterfactuals to select the recovery action that maximizes expected financial value.

🧠 Core Philosophy
The system operates on a strict four-stage architectural principle:

The Model Predicts: An ML model calculates the base probability of recovery.

The Policy Governs: Deterministic guardrails block unsafe automated actions.

The Engine Optimizes: The system calculates the Expected Net Value (EV) of all alternative actions.

The Executor Acts: The optimal action is routed for automated execution or human escalation.

✨ Key Features
Counterfactual Regret Analysis: Dynamically calculates the "expected regret" (financial loss) of choosing suboptimal actions like immediate retry versus suggesting an alternate payment method.

Deterministic Policy Guardrails: Explicitly overrides the ML model to block automated execution for high-value transactions (≥ ₹50,000) and RISK_FLAGGED events, forcing a HUMAN_REVIEW escalation.

Live Bank Outage Simulation: A hero demo feature that instantly recalculates probabilities and pivots from retries to alternate payment methods when degraded gateway rails are detected.

Strict Idempotency: Built-in duplicate webhook protection (HTTP 409) ensures a failed payment event is never processed or charged twice.

Executive Dashboard: A premium dark-mode fintech UI that provides total explainability, showing exactly why a specific action was chosen and why a retry was rejected.

🏗️ Technical Architecture
Backend: FastAPI (Python) driving the core decision engine and API contracts (/webhook/payment_failed).

Machine Learning: XGBoost Classifier with CalibratedClassifierCV for accurate probability scoring based on failure severity, customer LTV, and retry decay.

Frontend: Vanilla HTML/CSS/JS with zero heavy frameworks, rendering responsive, data-bound KPI cards and decision matrices.

Orchestration: Bounded executor ready for integration with n8n webhooks to orchestrate the final winning action (e.g., sending an SMS payment link).
