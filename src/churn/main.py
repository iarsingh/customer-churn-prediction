from fastapi import FastAPI, HTTPException

from churn.score import InputError, model, score

app = FastAPI()


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.get("/model")
def get_model():
    fitted = model()
    return {
        "bias": round(fitted["bias"], 4),
        "weights": {key: round(value, 4) for key, value in fitted["weights"].items()},
        "test": fitted["test"],
    }


@app.post("/score")
def post_score(body: dict):
    threshold = body.pop("threshold", 0.5)
    try:
        return score(body, threshold)
    except InputError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/score/batch")
def post_batch(body: dict):
    customers = body.get("customers")
    threshold = body.get("threshold", 0.5)
    if not isinstance(customers, list) or not 1 <= len(customers) <= 1000:
        raise HTTPException(status_code=422, detail="customers must be a list of 1 to 1000 items")
    for customer in customers:
        if not isinstance(customer, dict) or not isinstance(customer.get("id"), str):
            raise HTTPException(status_code=422, detail="each customer needs a string id")
    try:
        scored = [{"id": c["id"], **score(c, threshold)} for c in customers]
    except InputError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    scored.sort(key=lambda row: row["probability"], reverse=True)
    return {"queue": scored, "high": sum(row["label"] == "high" for row in scored)}
