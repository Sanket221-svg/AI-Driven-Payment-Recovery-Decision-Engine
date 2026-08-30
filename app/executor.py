import time
import uuid

import requests

N8N_WEBHOOK_URL = "http://localhost:5678/webhook/recoverai-action"


class RecoveryActionExecutor:
    """Dispatches approved recovery actions to n8n or a local simulation. """

    def _dispatch_to_n8n(self, payload: dict) -> dict:
        try:
            response = requests.post(N8N_WEBHOOK_URL, json=payload, timeout=1.5)
            payload["n8n_status"] = "Dispatched" if response.status_code == 200 else f"Failed ({response.status_code})"
        except requests.exceptions.RequestException:
            payload["n8n_status"] = "SIMULATION_MODE"
        return payload

    def schedule_retry(self, payment_id: str, delay_hours: int = 6) -> dict:
        time.sleep(0.1)
        return self._dispatch_to_n8n({
            "status": "SUCCESS",
            "action_executed": "SCHEDULED_RETRY",
            "payment_id": payment_id,
            "delay_hours": delay_hours,
            "retry_job_id": f"job_{uuid.uuid4().hex[:6]}",
            "message": f"Payment {payment_id} queued for retry in {delay_hours}h.",
        })

    def send_payment_reminder(self, payment_id: str, customer_id: str, amount_inr: float) -> dict:
        time.sleep(0.1)
        mock_link = f"https://rzp.io/i/{uuid.uuid4().hex[:8]}"
        return self._dispatch_to_n8n({
            "status": "SUCCESS",
            "action_executed": "PAYMENT_REMINDER_SENT",
            "payment_id": payment_id,
            "customer_id": customer_id,
            "amount_inr": amount_inr,
            "payment_link": mock_link,
            "message": f"Payment reminder link dispatched via WhatsApp for INR {amount_inr}.",
        })

    def suggest_alternate_method(self, payment_id: str, customer_id: str) -> dict:
        time.sleep(0.1)
        return self._dispatch_to_n8n({
            "status": "SUCCESS",
            "action_executed": "ALTERNATE_PAYMENT_NUDGED",
            "payment_id": payment_id,
            "customer_id": customer_id,
            "suggested_rails": ["UPI", "NetBanking"],
            "message": "Nudge sent to update payment method.",
        })

    def escalate_human(self, payment_id: str, reason: str, amount_inr: float) -> dict:
        time.sleep(0.1)
        ticket_id = f"TICK_{uuid.uuid4().hex[:6].upper()}"
        return self._dispatch_to_n8n({
            "status": "ESCALATED",
            "action_executed": "HUMAN_ESCALATION",
            "payment_id": payment_id,
            "amount_inr": amount_inr,
            "ticket_id": ticket_id,
            "priority": "HIGH" if amount_inr >= 50000 else "MEDIUM",
            "reason": reason,
        })

    def execute_action(self, payment, action: str) -> dict:
        if action in ["RETRY_6H", "RETRY_IMMEDIATE"]:
            delay = 0 if action == "RETRY_IMMEDIATE" else 6
            return self.schedule_retry(payment.payment_id, delay)
        if action == "PAYMENT_REMINDER":
            return self.send_payment_reminder(payment.payment_id, payment.customer_id, payment.amount_inr)
        if action == "ALTERNATE_PAYMENT":
            return self.suggest_alternate_method(payment.payment_id, payment.customer_id)
        if action == "HUMAN_REVIEW":
            return self.escalate_human(payment.payment_id, "Human review required", payment.amount_inr)
        return {"status": "SKIPPED", "action_executed": "NONE", "message": "Intervention suppressed by policy."}


executor = RecoveryActionExecutor()