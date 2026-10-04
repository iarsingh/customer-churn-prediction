import csv
import math
from functools import lru_cache
from pathlib import Path

DATA = Path(__file__).resolve().parents[2] / "data" / "customers.csv"
FEATURES = ("tenure_months", "monthly_charges", "support_tickets")
LIMITS = {"tenure_months": (0, 600), "monthly_charges": (0, 10000), "support_tickets": (0, 1000)}


class InputError(ValueError):
    pass


def sigmoid(value):
    if value >= 0:
        return 1 / (1 + math.exp(-value))
    exp = math.exp(value)
    return exp / (1 + exp)


def load(path=DATA):
    with open(path, encoding="utf-8") as handle:
        return [{key: float(value) for key, value in row.items()} for row in csv.DictReader(handle)]


def split(rows):
    return [r for i, r in enumerate(rows) if i % 4 != 3], [r for i, r in enumerate(rows) if i % 4 == 3]


def train(rows, epochs=3000, rate=0.1, l2=0.01):
    stats = {}
    for name in FEATURES:
        values = [row[name] for row in rows]
        mean = sum(values) / len(values)
        std = math.sqrt(sum((v - mean) ** 2 for v in values) / len(values)) or 1.0
        stats[name] = (mean, std)
    xs = [[(row[name] - stats[name][0]) / stats[name][1] for name in FEATURES] for row in rows]
    ys = [row["churned"] for row in rows]
    weights = [0.0] * len(FEATURES)
    bias = 0.0
    count = len(rows)
    for _ in range(epochs):
        grad_w = [l2 * w for w in weights]
        grad_b = 0.0
        for x, y in zip(xs, ys):
            error = sigmoid(bias + sum(w * v for w, v in zip(weights, x))) - y
            grad_b += error / count
            for i, value in enumerate(x):
                grad_w[i] += error * value / count
        weights = [w - rate * g for w, g in zip(weights, grad_w)]
        bias -= rate * grad_b
    return {"bias": bias, "weights": dict(zip(FEATURES, weights)), "stats": stats}


def explain(model, customer):
    contributions = {}
    for name in FEATURES:
        mean, std = model["stats"][name]
        contributions[name] = model["weights"][name] * (customer[name] - mean) / std
    logit = model["bias"] + sum(contributions.values())
    return logit, contributions


def evaluate(model, rows, threshold=0.5):
    tp = fp = tn = fn = 0
    for row in rows:
        predicted = sigmoid(explain(model, row)[0]) >= threshold
        actual = row["churned"] == 1
        tp += predicted and actual
        fp += predicted and not actual
        tn += not predicted and not actual
        fn += not predicted and actual
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "accuracy": round((tp + tn) / len(rows), 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "rows": len(rows),
    }


@lru_cache(maxsize=1)
def model():
    training, testing = split(load())
    fitted = train(training)
    fitted["test"] = evaluate(fitted, testing)
    return fitted


def validate(customer):
    for name in FEATURES:
        value = customer.get(name)
        low, high = LIMITS[name]
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not low <= value <= high:
            raise InputError(f"{name} must be a number from {low} to {high}")


def score(customer, threshold=0.5):
    if not isinstance(threshold, (int, float)) or not 0 <= threshold <= 1:
        raise InputError("threshold must be from 0 to 1")
    validate(customer)
    logit, contributions = explain(model(), customer)
    probability = sigmoid(logit)
    factors = sorted(contributions.items(), key=lambda item: abs(item[1]), reverse=True)
    return {
        "probability": round(probability, 4),
        "label": "high" if probability >= threshold else "low",
        "threshold": threshold,
        "factors": [{"feature": name, "contribution": round(value, 4), "direction": "raises" if value > 0 else "lowers"} for name, value in factors],
    }
