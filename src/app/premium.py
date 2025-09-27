# src/app/premium.py
def compute_premium(risk_prob, expected_claim, expense_ratio=0.30, profit_margin=0.10, admin_fee=10.0, min_premium=20.0):
    expected_loss = risk_prob * expected_claim
    denom = 1 - (expense_ratio + profit_margin)
    if denom <= 0:
        raise ValueError("Expense ratio + profit margin too large")
    premium = expected_loss / denom + admin_fee
    if premium < min_premium:
        premium = min_premium
    return round(premium,2)
