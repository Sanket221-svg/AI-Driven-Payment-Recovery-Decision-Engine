import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

try:
    import google.generativeai as genai
except ModuleNotFoundError:
    genai = None

from dotenv import load_dotenv
from app.schemas import PaymentFailureEvent

# --- 1. Setup and Configurations ---
project_root = Path(__file__).resolve().parent.parent
env_candidates = [
    project_root / ".env",
    Path.cwd() / ".env",
]

for env_path in env_candidates:
    if env_path.exists():
        load_dotenv(env_path)
        break

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not api_key:
    print("SECURITY WARNING: GEMINI_API_KEY or GOOGLE_API_KEY not found in .env file.")
elif genai is not None:
    genai.configure(api_key=api_key)
else:
    print("SECURITY WARNING: google-generativeai package is not installed; skipping Gemini config.")

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "..", "ml_artifacts")
try:
    model = joblib.load(os.path.join(ARTIFACTS_DIR, "recovery_model.pkl"))
    preprocessor = joblib.load(os.path.join(ARTIFACTS_DIR, "preprocessor.pkl"))
    imputer = joblib.load(os.path.join(ARTIFACTS_DIR, "imputer.pkl"))
except Exception:
    model, preprocessor, imputer = None, None, None

ACTION_CONFIG = {
    'RETRY_IMMEDIATE': {'cost': 2.0, 'friction': 5.0},
    'RETRY_6H': {'cost': 2.0, 'friction': 2.0},
    'PAYMENT_REMINDER': {'cost': 0.50, 'friction': 15.0},
    'ALTERNATE_METHOD': {'cost': 0.75, 'friction': 8.0},
    'DO_NOTHING': {'cost': 0.0, 'friction': 0.0}
}

customer_fatigue_store = {}

# --- 2. LLM Integration ---
def generate_dynamic_rationale(event, best_action_name, net_ev, costs_saved, guardrail_triggered=False, model_recommendation=None):
    if guardrail_triggered:
        return (
            f"Guardrail override activated for {event.failure_code}. The ML model favored "
            f"{model_recommendation or 'an alternative action'}, but policy enforced "
            f"{best_action_name} to maintain risk and customer-experience constraints."
        )
    if best_action_name == "DO_NOTHING":
        return (
            f"Recovery probability below economic threshold for {event.failure_code}. "
            f"Suppressed action to prevent customer fatigue, saving INR {costs_saved:.2f} in retry fees."
        )
    return (
        f"Counterfactual evaluation selected {best_action_name} based on customer success rate "
        f"({int(event.customer_success_rate * 100)}%), projected to recover net INR {net_ev:.2f}."
    )

# --- 3. Core Engine Logic ---
def evaluate_recovery_decision(event: PaymentFailureEvent) -> dict:
    amount = float(event.amount_inr)
    failure_code = event.failure_code.upper()
    customer_id = event.customer_id
    fatigue_count = customer_fatigue_store.get(customer_id, 0)
    outage_penalty = 0.50 if event.simulate_bank_outage and event.payment_method == "UPI" else 0.0

    def estimate_probability(action_name: str, fallback_value: float) -> float:
        if action_name == 'DO_NOTHING':
            return 0.0
        if action_name == 'ALTERNATE_METHOD':
            return max(0.0, min(0.95, fallback_value + 0.10))
        if 'RETRY' in action_name:
            return max(0.0, fallback_value - (outage_penalty if event.simulate_bank_outage and event.payment_method == "UPI" else 0.0))
        return max(0.0, min(0.90, fallback_value))

    simulated_actions = []
    actions = ['RETRY_IMMEDIATE', 'RETRY_6H', 'PAYMENT_REMINDER', 'ALTERNATE_METHOD', 'DO_NOTHING']

    if model and preprocessor and imputer:
        sim_records = []
        for act in actions:
            sim_records.append({
                'payment_method': event.payment_method,
                'failure_code': failure_code,
                'failure_source': event.failure_source,
                'customer_value': event.customer_value.upper(),
                'action_tested': act,
                'amount_inr': amount,
                'amount_log': np.log1p(amount),
                'attempt_number': event.attempt_number,
                'customer_success_rate': event.customer_success_rate,
                'failure_severity': 2,
                'retry_decay': 1.0 / (event.attempt_number ** 1.5),
                'reliability_index': event.customer_success_rate * 1.5,
                'is_high_value': int(amount >= 50000),
                'expected_value_baseline': amount * event.customer_success_rate,
                'hour_of_day': datetime.now().hour
            })

        df_sim = pd.DataFrame(sim_records)
        processed = preprocessor.transform(df_sim)
        imputed = imputer.transform(processed)
        probs = model.predict_proba(imputed)[:, 1]

        for idx, act in enumerate(actions):
            raw_prob = float(probs[idx])
            p_rec = estimate_probability(act, raw_prob)
            cost = ACTION_CONFIG[act]['cost']
            friction = ACTION_CONFIG[act]['friction'] * (fatigue_count + 1)
            nuisance = 50.0 if (p_rec < 0.35 and act != 'DO_NOTHING') else 0.0
            net_ev = (p_rec * amount) - cost - friction - nuisance
            if act == 'DO_NOTHING':
                net_ev = 0.0
            simulated_actions.append({
                'action_name': act,
                'probability': p_rec,
                'expected_net_value': net_ev,
                'execution_cost': cost
            })
    else:
        base_probs = {
            'RETRY_IMMEDIATE': 0.45,
            'RETRY_6H': 0.52,
            'PAYMENT_REMINDER': 0.33,
            'ALTERNATE_METHOD': 0.68,
            'DO_NOTHING': 0.0,
        }
        if failure_code == 'RISK_FLAGGED':
            base_probs.update({'RETRY_IMMEDIATE': 0.12, 'RETRY_6H': 0.15, 'PAYMENT_REMINDER': 0.10, 'ALTERNATE_METHOD': 0.18})
        elif failure_code == 'EXPIRED_CARD':
            base_probs.update({'RETRY_IMMEDIATE': 0.12, 'RETRY_6H': 0.20, 'PAYMENT_REMINDER': 0.28, 'ALTERNATE_METHOD': 0.76})
        elif failure_code == 'NETWORK_TIMEOUT':
            base_probs.update({'RETRY_IMMEDIATE': 0.62, 'RETRY_6H': 0.45, 'PAYMENT_REMINDER': 0.32, 'ALTERNATE_METHOD': 0.50})

        for act in actions:
            p_rec = estimate_probability(act, base_probs.get(act, 0.25))
            cost = ACTION_CONFIG[act]['cost']
            friction = ACTION_CONFIG[act]['friction'] * (fatigue_count + 1)
            nuisance = 50.0 if (p_rec < 0.35 and act != 'DO_NOTHING') else 0.0
            net_ev = (p_rec * amount) - cost - friction - nuisance
            if act == 'DO_NOTHING':
                net_ev = 0.0
            simulated_actions.append({
                'action_name': act,
                'probability': p_rec,
                'expected_net_value': net_ev,
                'execution_cost': cost
            })

    best_action_data = max(simulated_actions, key=lambda x: x['expected_net_value'])
    max_ev = best_action_data['expected_net_value']

    for sim in simulated_actions:
        sim['expected_regret'] = max(0.0, max_ev - sim['expected_net_value'])

    model_recommendation = best_action_data['action_name']
    guardrail_triggered = amount >= 50000 or failure_code == 'RISK_FLAGGED'
    policy_reason = None
    if amount >= 50000 and failure_code == 'RISK_FLAGGED':
        policy_reason = "Transaction exceeds the automated recovery threshold and is flagged as risky."
    elif amount >= 50000:
        policy_reason = "Transaction exceeds the automated recovery threshold."
    elif failure_code == 'RISK_FLAGGED':
        policy_reason = "Risk flag requires manual review before automated recovery."

    policy_decision = model_recommendation
    action = model_recommendation
    costs_saved = 0.0

    if guardrail_triggered:
        policy_decision = 'HUMAN_REVIEW'
        action = 'HUMAN_REVIEW'
        costs_saved = 0.0
        rationale = generate_dynamic_rationale(
            event=event,
            best_action_name='HUMAN_REVIEW',
            net_ev=0.0,
            costs_saved=0.0,
            guardrail_triggered=True,
            model_recommendation=model_recommendation,
        )
    else:
        action = model_recommendation
        if action == 'DO_NOTHING':
            costs_saved = ACTION_CONFIG['RETRY_IMMEDIATE']['cost'] + ACTION_CONFIG['RETRY_IMMEDIATE']['friction']
        rationale = generate_dynamic_rationale(
            event=event,
            best_action_name=action,
            net_ev=best_action_data['expected_net_value'],
            costs_saved=costs_saved,
            guardrail_triggered=False,
            model_recommendation=model_recommendation,
        )

    if action != 'DO_NOTHING':
        customer_fatigue_store[customer_id] = fatigue_count + 1

    return {
        'action': action,
        'prob': best_action_data['probability'] if not guardrail_triggered else 0.0,
        'net_val': best_action_data['expected_net_value'] if not guardrail_triggered else 0.0,
        'costs_saved': costs_saved,
        'escalate': guardrail_triggered,
        'rationale': rationale,
        'guardrail_triggered': guardrail_triggered,
        'model_recommendation': model_recommendation,
        'policy_decision': policy_decision,
        'policy_reason': policy_reason,
        'matrix': [{
            **sim,
            'expected_regret': sim['expected_regret'],
            'regret_value': sim['expected_regret']
        } for sim in simulated_actions],
        'counterfactuals': [
            {
                'action_name': sim['action_name'],
                'probability': sim['probability'],
                'expected_net_value': sim['expected_net_value'],
                'expected_regret': sim['expected_regret'],
                'execution_cost': sim['execution_cost'],
            }
            for sim in simulated_actions
        ]
    }
