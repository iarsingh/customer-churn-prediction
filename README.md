# Customer Churn Prediction

<!-- project-guide:start -->
## Project guide

[Project architecture](PROJECT_ARCHITECTURE.md) · [Interview questions and answers](INTERVIEW_QA.md)

Use the architecture document for the component diagram, implementation boundaries, and verification entry points. The interview guide includes source-backed answers and project walkthroughs.

### Implementation map

| Component | Responsibility |
| --- | --- |
| [`src/churn/main.py`](src/churn/main.py) | HTTP handlers: `GET /healthz`, `GET /model`, `POST /score`, `POST /score/batch` |
| [`src/churn/ops.py`](src/churn/ops.py) | HTTP handlers: `GET /readyz`, `POST /workspaces`, `GET /workspaces`, `POST /workspaces/{workspace_id}/jobs`, `GET /jobs/{job_id}` |
| [`src/churn/score.py`](src/churn/score.py) | Functions: `sigmoid`, `load`, `split`, `train`, `explain`, `evaluate`, `model` |
| [`requirements.txt`](requirements.txt) | Implementation or supporting configuration |
| [`src/churn/__init__.py`](src/churn/__init__.py) | Implementation or supporting configuration |
| [`Dockerfile`](Dockerfile) | Container build/service configuration |
| [`Makefile`](Makefile) | Implementation or supporting configuration |
| [`docker-compose.yml`](docker-compose.yml) | Container build/service configuration |
| [`tests/test_churn.py`](tests/test_churn.py) | Executable checks and regression examples |
| [`tests/test_ops.py`](tests/test_ops.py) | Executable checks and regression examples |
| [`.github/workflows/ci.yml`](.github/workflows/ci.yml) | GitHub Actions job definitions |
| [`README.md`](README.md) | Project explanations or operating notes |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Project explanations or operating notes |

### Local setup and verification

From the repository root (the commands follow the checked-in manifests):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
```

To serve the FastAPI application locally, install the server separately if it is not already available:

```bash
python -m pip install uvicorn
PYTHONPATH=src python -m uvicorn churn.main:app --reload
```

<!-- project-guide:end -->

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

## Ops plane

Workspaces, tenant isolation, job approval, and audit live under `/v1`. Production apply is refused. See `docs/ARCHITECTURE.md`.

## Documentation checks

Project architecture, interview guides, and local source links are checked automatically on pushes and pull requests. Run the same check locally:

```bash
python3 .github/scripts/validate_project_docs.py
```

See [service improvements and local run instructions](docs/UPGRADES.md).
