from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import brier_score_loss, classification_report, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = ROOT / "ml_artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)


def make_synthetic_dataset(n_rows: int = 2200) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rows = []
    for i in range(n_rows):
        payment_method = rng.choice(["UPI", "CARD", "NETBANKING", "WALLET"])
        failure_code = rng.choice(["NETWORK_TIMEOUT", "INSUFFICIENT_BALANCE", "EXPIRED_CARD", "BANK_DECLINE", "RISK_FLAGGED"])
        customer_value = rng.choice(["LOW", "MEDIUM", "HIGH"])
        amount = float(rng.lognormal(mean=8.4, sigma=0.8))
        attempt_number = int(rng.integers(1, 5))
        success_rate = float(rng.beta(6, 3))
        action_tested = rng.choice(["RETRY_IMMEDIATE", "RETRY_6H", "PAYMENT_REMINDER", "ALTERNATE_PAYMENT", "DO_NOTHING"])
        bank_outage = bool(rng.random() < 0.12 and payment_method == "UPI")
        recovery_score = (
            0.15
            + (0.25 if payment_method == "UPI" else 0.10)
            + (0.18 if failure_code == "INSUFFICIENT_BALANCE" else 0.0)
            + (0.10 if failure_code == "NETWORK_TIMEOUT" else 0.0)
            + (0.20 if action_tested == "RETRY_6H" and failure_code == "INSUFFICIENT_BALANCE" else 0.0)
            + (0.18 if action_tested == "ALTERNATE_PAYMENT" and failure_code == "EXPIRED_CARD" else 0.0)
            + (0.12 if bank_outage and action_tested == "ALTERNATE_PAYMENT" else 0.0)
            + (0.08 if customer_value == "HIGH" else 0.0)
            - (0.08 if failure_code == "RISK_FLAGGED" else 0.0)
            - (0.05 if attempt_number > 2 else 0.0)
        )
        recovery_score = max(0.05, min(recovery_score, 0.98))
        recovered = int(rng.random() < recovery_score)
        rows.append({
            "payment_method": payment_method,
            "failure_code": failure_code,
            "customer_value": customer_value,
            "amount_inr": round(amount, 2),
            "attempt_number": attempt_number,
            "customer_success_rate": round(success_rate, 4),
            "action_tested": action_tested,
            "simulate_bank_outage": bank_outage,
            "recovered": recovered,
            "amount_log": np.log1p(amount),
            "failure_severity": {"NETWORK_TIMEOUT": 1, "INSUFFICIENT_BALANCE": 2, "BANK_DECLINE": 3, "EXPIRED_CARD": 4, "RISK_FLAGGED": 5}.get(failure_code, 2),
            "retry_decay": 1.0 / (attempt_number ** 1.5),
            "reliability_index": success_rate * ({"LOW": 0.7, "MEDIUM": 1.0, "HIGH": 1.4}.get(customer_value, 1.0)),
            "is_high_value": int(amount >= 50000),
            "expected_value_baseline": amount * success_rate,
            "hour_of_day": int(rng.integers(0, 24)),
        })
    return pd.DataFrame(rows)


def train_model() -> dict:
    df = make_synthetic_dataset()
    target = "recovered"
    categorical = ["payment_method", "failure_code", "customer_value", "action_tested"]
    numeric = [
        "amount_inr", "attempt_number", "customer_success_rate", "amount_log", "failure_severity",
        "retry_decay", "reliability_index", "is_high_value", "expected_value_baseline", "hour_of_day",
    ]
    features = categorical + numeric

    X = df[features]
    y = df[target].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ],
        remainder="passthrough",
    )
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", RandomForestClassifier(n_estimators=250, random_state=42, class_weight="balanced_subsample")),
        ]
    )
    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    metrics = {
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "brier_score": float(brier_score_loss(y_test, y_prob)),
        "confusion_matrix": [
            [int(x) for x in row] for row in pd.crosstab(y_test, y_pred).to_numpy()
        ],
    }

    joblib.dump(model, ARTIFACTS_DIR / "recovery_model.pkl")
    with open(ARTIFACTS_DIR / "model_metadata.json", "w", encoding="utf-8") as fh:
        json.dump({
            "model_name": "action_conditioned_recovery_model",
            "version": "1.0.0",
            "training_date": "2026-08-30",
            "dataset_size": int(len(df)),
            "feature_list": features,
            "metrics": metrics,
        }, fh, indent=2)

    return {
        "dataset_size": int(len(df)),
        "features": features,
        "metrics": metrics,
    }


if __name__ == "__main__":
    print(train_model())
