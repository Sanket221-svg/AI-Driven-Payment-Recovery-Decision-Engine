import random
import pandas as pd
import numpy as np
from app.schemas import PaymentFailureEvent
from app.engine import evaluate_recovery_decision, customer_fatigue_store

def generate_synthetic_batch(n=1000):
    """Generates synthetic failed transaction events with varying risk profiles."""
    random.seed(42)
    np.random.seed(42)
    
    failure_codes = ["INSUFFICIENT_BALANCE", "NETWORK_TIMEOUT", "EXPIRED_CARD", "RISK_FLAGGED", "BANK_DECLINE"]
    payment_methods = ["UPI", "Credit Card", "Debit Card", "NetBanking"]
    customer_tiers = ["LOW", "MEDIUM", "HIGH"]
    
    events = []
    for i in range(n):
        # Generate realistic distribution
        amount = round(float(np.random.exponential(scale=3500) + 200), 2)
        code = random.choices(failure_codes, weights=[0.45, 0.25, 0.15, 0.05, 0.10])[0]
        method = "UPI" if random.random() < 0.65 else random.choice(payment_methods)
        tier = random.choices(customer_tiers, weights=[0.5, 0.35, 0.15])[0]
        success_rate = round(float(np.random.beta(a=7, b=3)), 2)
        
        # 10% chance of an active gateway degradation event
        outage = random.random() < 0.10 and method == "UPI"
        
        events.append(PaymentFailureEvent(
            payment_id=f"pay_bench_{i:04d}",
            customer_id=f"cust_{random.randint(1, 250)}",
            amount_inr=amount,
            payment_method=method,
            failure_code=code,
            customer_success_rate=success_rate,
            customer_value=tier,
            simulate_bank_outage=outage
        ))
    return events

def run_benchmark():
    events = generate_synthetic_batch(1000)
    customer_fatigue_store.clear()
    
    # Financial metrics accumulators
    results = {
        "Do Nothing": {"gross_recovered": 0.0, "fees_friction": 0.0, "net_revenue": 0.0, "actions_taken": 0},
        "Blind Retry Bot": {"gross_recovered": 0.0, "fees_friction": 0.0, "net_revenue": 0.0, "actions_taken": 0},
        "RecoverAI (Ours)": {"gross_recovered": 0.0, "fees_friction": 0.0, "net_revenue": 0.0, "actions_taken": 0}
    }
    
    for event in events:
        amt = event.amount_inr
        
        # 1. Strategy: Do Nothing
        # Zero recovery, zero cost.
        
        # 2. Strategy: Blind Retry Bot
        # Retries everything naively (cost: ₹2.00 API + ₹5.00 friction = ₹7.00 per attempt)
        results["Blind Retry Bot"]["actions_taken"] += 1
        results["Blind Retry Bot"]["fees_friction"] += 7.00
        
        # Hard declines (EXPIRED_CARD, RISK_FLAGGED) or Bank Outages always yield 0% success on blind retry
        if event.failure_code in ["EXPIRED_CARD", "RISK_FLAGGED"] or event.simulate_bank_outage:
            p_blind = 0.0
        else:
            p_blind = max(0.05, event.customer_success_rate * 0.50)
            
        results["Blind Retry Bot"]["gross_recovered"] += amt * p_blind
        
        # 3. Strategy: RecoverAI
        decision = evaluate_recovery_decision(event)
        act = decision["action"]
        prob = decision["prob"]
        
        if act != "DO_NOTHING":
            results["RecoverAI (Ours)"]["actions_taken"] += 1
            cost = 2.0 if "RETRY" in act else 0.50
            results["RecoverAI (Ours)"]["fees_friction"] += cost
            results["RecoverAI (Ours)"]["gross_recovered"] += amt * prob
            
    # Calculate Net Values
    for k in results:
        results[k]["net_revenue"] = results[k]["gross_recovered"] - results[k]["fees_friction"]
        
    df_summary = pd.DataFrame([
        {
            "Strategy": k,
            "Interventions Attempted": f"{v['actions_taken']:,}",
            "Gross Recovered (INR)": f"₹{v['gross_recovered']:,.2f}",
            "Execution & Friction Cost": f"₹{v['fees_friction']:,.2f}",
            "Net Recovered Value (INR)": f"₹{v['net_revenue']:,.2f}",
            "Net ROI vs Baseline": f"{((v['net_revenue'] / (results['Blind Retry Bot']['net_revenue'] + 1e-5)) - 1) * 100:+.1f}%" if k == "RecoverAI (Ours)" else "0.0%"
        }
        for k, v in results.items()
    ])
    
    print("\n" + "="*80)
    print("      RECOVERAI THREE-WAY STRATEGY BENCHMARK (1,000 SYNTHETIC PAYMENTS)")
    print("="*80)
    print(df_summary.to_string(index=False))
    print("="*80 + "\n")

if __name__ == "__main__":
    run_benchmark()