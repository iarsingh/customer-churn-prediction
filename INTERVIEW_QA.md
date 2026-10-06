# customer-churn-prediction — interview questions and answers

[README](README.md) · [Project architecture](PROJECT_ARCHITECTURE.md)

Answers below use this repository’s files and implementation. They distinguish existing behavior from suggested extensions; source links let you verify each walkthrough.

## 1. What problem does customer-churn-prediction address, and what can you demonstrate?

`data/customers.csv` has 24 customers with tenure, monthly charges, support tickets, and whether they churned. Every fourth row is held out. The model is logistic regression trained by gradient descent on standardized features with a small L2 penalty, so the weights are learned from data rather than typed in.

I would demonstrate the linked implementation or examples and distinguish that evidence from any planned production features. Start with [`README.md`](README.md).

## 2. How is this repository organized?

- [`src/churn/main.py`](src/churn/main.py): Implementation or supporting configuration.
- [`src/churn/ops.py`](src/churn/ops.py): Implementation or supporting configuration.
- [`src/churn/score.py`](src/churn/score.py): Implementation or supporting configuration.
- [`requirements.txt`](requirements.txt): Implementation or supporting configuration.
- [`src/churn/__init__.py`](src/churn/__init__.py): Implementation or supporting configuration.
- [`Dockerfile`](Dockerfile): Container build/service configuration.
- [`Makefile`](Makefile): Implementation or supporting configuration.
- [`docker-compose.yml`](docker-compose.yml): Container build/service configuration.

[PROJECT_ARCHITECTURE.md](PROJECT_ARCHITECTURE.md) contains the component diagram and the implementation walkthrough.

## 3. Can you walk through `train` and explain the decision it makes?

The main walkthrough here is `train(rows, epochs=3000, rate=0.1, l2=0.01)` in [`src/churn/score.py`](src/churn/score.py#L31).

```python
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
```

This is an excerpt; follow the source link for the rest of the branches.

The implementation calls `dict`, `enumerate`, `len`, `math.sqrt`, `range`, `sigmoid`, `sum`, `zip`. In an interview, trace those calls in execution order using a fixture input.

## 4. What responsibility does `evaluate` have?

`evaluate(model, rows, threshold=0.5)` is defined in [`src/churn/score.py`](src/churn/score.py#L65).

Its return expressions include:

- `{'accuracy': round((tp + tn) / len(rows), 4), 'precision': round(precision, 4), 'recall': round(recall, 4), 'confusion': {'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn}, 'rows': len(rows)}`

It uses `explain`, `len`, `round`, `sigmoid`. This is the code path I would compare against the caller to explain responsibility boundaries.

## 5. What input validation and failure behavior are implemented?

Explicit failure paths include:

- `HTTPException(status_code=422, detail='customers must be a list of 1 to 1000 items')` in [`src/churn/main.py`](src/churn/main.py#L39).
- `HTTPException(status_code=422, detail=str(exc))` in [`src/churn/main.py`](src/churn/main.py#L31).
- `HTTPException(status_code=422, detail='each customer needs a string id')` in [`src/churn/main.py`](src/churn/main.py#L42).
- `HTTPException(status_code=422, detail=str(exc))` in [`src/churn/main.py`](src/churn/main.py#L46).
- `HTTPException(status_code=404, detail='workspace not found')` in [`src/churn/ops.py`](src/churn/ops.py#L77).
- `HTTPException(status_code=404, detail='job not found')` in [`src/churn/ops.py`](src/churn/ops.py#L100).
- `HTTPException(status_code=404, detail='job not found')` in [`src/churn/ops.py`](src/churn/ops.py#L109).

I would test both the condition that reaches each exception and the caller that translates it. An explicit raise does not mean every malformed input or dependency failure is handled.

## 6. Which test would you use to demonstrate correctness?

[`tests/test_churn.py`](tests/test_churn.py#L14) contains `test_high_and_low`:

```python
def test_high_and_low():
    assert client.post("/score", json=RISKY).json()["label"] == "high"
    assert client.post("/score", json=LOYAL).json()["label"] == "low"
```

This is a concrete regression example from the repository. Its assertions establish that case; they do not establish behavior for every input or under production load.

## 7. What HTTP interface does the code expose?

- `GET /healthz` → `healthz` in [`src/churn/main.py`](src/churn/main.py#L11).
- `GET /model` → `get_model` in [`src/churn/main.py`](src/churn/main.py#L16).
- `POST /score` → `post_score` in [`src/churn/main.py`](src/churn/main.py#L26).
- `POST /score/batch` → `post_batch` in [`src/churn/main.py`](src/churn/main.py#L35).
- `GET /readyz` → `readyz` in [`src/churn/ops.py`](src/churn/ops.py#L74).
- `POST /workspaces` → `create_workspace` in [`src/churn/ops.py`](src/churn/ops.py#L80).
- `GET /workspaces` → `list_workspaces` in [`src/churn/ops.py`](src/churn/ops.py#L98).
- `POST /workspaces/{workspace_id}/jobs` → `create_job` in [`src/churn/ops.py`](src/churn/ops.py#L106).

These are literal decorators. Application/router prefixes, authentication, and middleware must be checked in the corresponding setup code.

## 8. Where does state live, and what happens with multiple workers?

Module-level containers include `_WORKSPACES`, `_JOBS`, `_AUDIT`, `_METRICS` in [`src/churn/ops.py`](src/churn/ops.py); `LIMITS` in [`src/churn/score.py`](src/churn/score.py).

These containers belong to a Python process. Inspect which are constant fixtures and which are mutated. Mutable process state needs an explicit shared-storage or synchronization strategy before multiple workers can provide consistent behavior.

## 9. How would another engineer reproduce your walkthrough?

Start from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
```

These commands follow repository manifests; environment setup and command results still need to be checked on the target machine.

## 10. What does automation verify, and what does it not prove?

Inspect [`.github/workflows/ci.yml`](.github/workflows/ci.yml) for triggers, permissions, and job commands. I would name the checks that those definitions run and show the latest run separately. A workflow definition alone does not establish a successful deployment, security review, or production SLO.

## 11. How would you present this project in a Forward Deployed Engineer interview?

Start with the user and operational problem described in [`README.md`](README.md). Explain one constraint that changes the implementation, show the linked code or example, and walk through a success case and a failure case. Agree on a measurable acceptance criterion before expanding the solution, and leave a handoff with data boundaries and rollback ownership. Any proposed production or business metric should be identified as a target until measured.

## 12. What is the input-to-output contract of `train`?

In [`src/churn/score.py`](src/churn/score.py#L31), `train(rows, epochs=3000, rate=0.1, l2=0.01)` receives the inputs. The function computes these intermediate values:

- `stats = {}`
- `xs = [[(row[name] - stats[name][0]) / stats[name][1] for name in FEATURES] for row in rows]`
- `ys = [row['churned'] for row in rows]`
- `weights = [0.0] * len(FEATURES)`
- `bias = 0.0`
- `count = len(rows)`

Its result is defined by:

- `{'bias': bias, 'weights': dict(zip(FEATURES, weights)), 'stats': stats}`

## 13. What does the operations plane add, and where is its limit?

[`src/churn/ops.py`](src/churn/ops.py) declares `GET /readyz`, `POST /workspaces`, `GET /workspaces`, `POST /workspaces/{workspace_id}/jobs`, `GET /jobs/{job_id}`, `POST /jobs/{job_id}/approve`, `GET /audit`, `GET /metrics`. Inspect the application’s `include_router` call for its URL prefix.

Its state containers are `_WORKSPACES`, `_JOBS`, `_AUDIT`, `_METRICS`. The job-approval handler defines whether a target is accepted or refused; check that branch and the associated tests instead of treating a recorded job as a successful infrastructure apply.
