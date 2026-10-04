# Customer Churn Prediction

Level: 2 — Data science

Skills: Python, logistic regression from scratch, standardization, precision and recall, explainable scores

`data/customers.csv` has 24 customers with tenure, monthly charges, support tickets, and whether they churned. Every fourth row is held out. The model is logistic regression trained by gradient descent on standardized features with a small L2 penalty, so the weights are learned from data rather than typed in.

```bash
pip install -r requirements.txt
pytest -q
PYTHONPATH=src uvicorn churn.main:app --reload
```

| Method and path | Returns |
| --- | --- |
| `GET /model` | Bias, weights, and held-out accuracy, precision, recall, and confusion counts |
| `POST /score` | Probability, a `high` or `low` label, and each feature's contribution, largest first |
| `POST /score/batch` | Up to 1000 customers with ids, ranked by probability as a retention queue |

```bash
curl -s -X POST localhost:8000/score -H 'content-type: application/json' \
  -d '{"tenure_months":1,"monthly_charges":90,"support_tickets":6,"threshold":0.5}'
```

`threshold` moves the label without changing the probability. Lower it when missing a churner costs more than a wasted retention call.

Refused: a missing feature, a boolean or text value, a value out of range, a threshold outside 0 to 1, and a batch customer without a string `id`.

The service scores only. It does not send retention emails or change any account.
