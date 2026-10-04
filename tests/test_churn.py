import math

import pytest
from fastapi.testclient import TestClient

from churn.main import app
from churn.score import explain, model, sigmoid

client = TestClient(app)
RISKY = {"tenure_months": 1, "monthly_charges": 90, "support_tickets": 6}
LOYAL = {"tenure_months": 36, "monthly_charges": 40, "support_tickets": 0}


def test_high_and_low():
    assert client.post("/score", json=RISKY).json()["label"] == "high"
    assert client.post("/score", json=LOYAL).json()["label"] == "low"


def test_learned_weights_have_the_expected_signs():
    weights = client.get("/model").json()["weights"]
    assert weights["tenure_months"] < 0
    assert weights["support_tickets"] > 0


def test_held_out_accuracy():
    test = client.get("/model").json()["test"]
    assert test["rows"] == 6
    assert test["accuracy"] >= 0.66
    assert sum(test["confusion"].values()) == 6


def test_factors_explain_the_score():
    payload = client.post("/score", json=RISKY).json()
    logit, _ = explain(model(), RISKY)
    assert payload["probability"] == pytest.approx(round(sigmoid(logit), 4))
    tickets = next(f for f in payload["factors"] if f["feature"] == "support_tickets")
    assert tickets["direction"] == "raises"
    magnitudes = [abs(f["contribution"]) for f in payload["factors"]]
    assert magnitudes == sorted(magnitudes, reverse=True)


def test_threshold_changes_the_label_not_the_probability():
    low = client.post("/score", json={**LOYAL, "threshold": 0.0}).json()
    default = client.post("/score", json=LOYAL).json()
    assert low["label"] == "high"
    assert low["probability"] == default["probability"]
    assert client.post("/score", json={**LOYAL, "threshold": 1.5}).status_code == 422


def test_invalid_customers_are_refused():
    assert client.post("/score", json={"tenure_months": 1}).status_code == 422
    assert client.post("/score", json={**RISKY, "support_tickets": -1}).status_code == 422
    assert client.post("/score", json={**RISKY, "tenure_months": True}).status_code == 422


def test_batch_returns_a_ranked_retention_queue():
    customers = [{"id": "c-loyal", **LOYAL}, {"id": "c-risky", **RISKY}]
    payload = client.post("/score/batch", json={"customers": customers}).json()
    assert [row["id"] for row in payload["queue"]] == ["c-risky", "c-loyal"]
    assert payload["high"] == 1
    assert client.post("/score/batch", json={"customers": [RISKY]}).status_code == 422


def test_sigmoid_is_stable_for_large_inputs():
    assert sigmoid(-1000) == 0.0
    assert sigmoid(1000) == 1.0
    assert math.isclose(sigmoid(0), 0.5)
