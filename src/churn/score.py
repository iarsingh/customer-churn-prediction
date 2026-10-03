import math

def score(customer):
    linear = (
        -2
        + -0.08 * customer["tenure_months"]
        + 0.03 * customer["monthly_charges"]
        + 0.5 * customer["support_tickets"]
    )
    probability = 1 / (1 + math.exp(-linear))
    return {"probability": round(probability, 4), "label": "high" if probability >= 0.5 else "low"}
