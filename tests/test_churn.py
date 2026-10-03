from fastapi.testclient import TestClient
from churn.main import app

def test_high_and_low():
    client = TestClient(app)
    risky = {"tenure_months": 1, "monthly_charges": 90, "support_tickets": 6}
    loyal = {"tenure_months": 36, "monthly_charges": 40, "support_tickets": 0}
    assert client.post("/score", json=risky).json()["label"] == "high"
    assert client.post("/score", json=loyal).json()["label"] == "low"
