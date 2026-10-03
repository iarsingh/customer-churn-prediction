from fastapi import FastAPI
from churn.score import score

app = FastAPI()

@app.post("/score")
def post_score(body: dict):
    return score(body)
