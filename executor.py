import time
import uuid
import requests
from datetime import datetime

# Configure your local n8n webhook URL here
N8N_WEBHOOK_URL = "http://localhost:5678/webhook/recoverai-action"

class RecoveryActionExecutor:
    """
    Executes bounded recovery actions by offloading them to an n8n automation workflow.
    """
    
    def _dispatch_to_n8n(self, payload: dict) -> dict:
        """Helper method to push events to n8n with a fallback."""
        try:
            # Fire-and-forget to n8n webhook (timeout set low so API doesn't hang)
            response = requests.post(N8N_WEBHOOK_URL, json=payload, timeout=1.5)
            if response.status_code == 200:
                payload["n8n_status"] = "Dispatched successfully"
            else:
                payload["n8n_status"] = f"Failed with status {response.status_code}"
        except requests.exceptions.RequestException:
            payload["n8n_status"] = "n8n unreachable - running in simulation mode"
        
        return payload

    def schedule_retry(self, payment_id: str, delay_hours: int = 6) -> dict:
        time.sleep(0.3) 
        payload = {
            "status": "SUCCESS",
            "action_executed": "SCHEDULED_RETRY",
            "payment_id": payment_id,
            "scheduled_time": f"+{delay_hours} hours",
            "retry_job_id": f"job_retry_{uuid.uuid4().hex[:6]}",
            "message": f"Payment {payment_id} queued for smart retry in {delay_hours}h."
        }
        return self._dispatch_to_n8n(payload)
        
    def send_payment_reminder(self, payment_id: str, customer_id: str, amount_inr: float) -> dict:
        time.sleep(0.3)
        mock_payment_link = f"https://rzp.io/i/{uuid.uuid4().hex[:8]}"
        payload = {
            "status": "SUCCESS",
            "action_executed": "PAYMENT_REMINDER_SENT",
            "payment_id": payment_id,
            "customer_id": customer_id,
            "amount": amount_inr,
            "payment_link": mock_payment_link,
            "message": f"Recovery link {mock_payment_link} dispatched for INR {amount_inr}."
        }
        return self._dispatch_to_n8n(payload)

    def suggest_alternate_method(self, payment_id: str, customer_id: str) -> dict:
        time.sleep(0.3)
        payload = {
            "status": "SUCCESS",
            "action_executed": "ALTERNATE_METHOD_NUDGED",
            "payment_id": payment_id,
            "customer_id": customer_id,
            "suggested_rails": ["UPI", "NetBanking"],
            "message": "Instrument update request triggered to user app."
        }
        return self._dispatch_to_n8n(payload)

    def escalate_human(self, payment_id: str, reason: str, amount_inr: float) -> dict:
        time.sleep(0.2)
        ticket_id = f"TICK_{uuid.uuid4().hex[:6].upper()}"
        payload = {
            "status": "ESCALATED",
            "action_executed": "HUMAN_ESCALATION",
            "payment_id": payment_id,
            "amount": amount_inr,
            "ticket_id": ticket_id,
            "priority": "HIGH" if amount_inr >= 50000 else "MEDIUM",
            "reason": reason
        }
        return self._dispatch_to_n8n(payload)

# Singleton instance
executor = RecoveryActionExecutor()