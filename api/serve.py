from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Optional
import joblib, json, numpy as np, uvicorn, os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART  = ROOT / "artifacts"

meta     = json.load(open(ART / "metadata.json"))
FEATURES = meta["feature_cols"]
THRESH   = meta["threshold"]
scaler   = joblib.load(ART / "scaler.joblib")
model    = joblib.load(ART / "iforest.joblib")

class ScoreRequest(BaseModel):
    ts: Optional[float] = None
    features: Dict[str, float]

app = FastAPI(title="Anomaly Scoring API", version=meta.get("version","1.0.0"))

def score_vector(feat_dict: Dict[str, float]) -> float:
    missing = [f for f in FEATURES if f not in feat_dict]
    if missing:
        raise HTTPException(status_code=400, detail={"error":"Missing features","missing":missing})
    x = np.array([feat_dict[f] for f in FEATURES], dtype=float).reshape(1, -1)
    x = scaler.transform(x)
    return float(-model.score_samples(x)[0])  # higher = more anomalous

@app.get("/healthz")
def healthz(): return {"ok": True}

@app.post("/score")
def score(req: ScoreRequest):
    s = score_vector(req.features)
    return {"score": s, "is_anomaly": bool(s >= THRESH)}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=int(os.getenv("PORT", 8000)))

