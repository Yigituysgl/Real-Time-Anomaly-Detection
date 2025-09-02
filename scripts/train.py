import json, joblib, numpy as np, pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
ART.mkdir(parents=True, exist_ok=True)

def train(csv_path: str, val_frac: float = 0.2, n_estimators: int = 300, seed: int = 42):
    df = pd.read_csv(csv_path).drop_duplicates().sort_values("Time").reset_index(drop=True)
    feature_cols = [c for c in df.columns if c.startswith("V")] + ["Amount"]

    
    split = int((1 - val_frac) * len(df))
    train_df, val_df = df.iloc[:split], df.iloc[split:]
    X_train = train_df.loc[train_df["Class"] == 0, feature_cols].values  # unsupervised on normal
    X_val   = val_df[feature_cols].values
    y_val   = val_df["Class"].values

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)

    iforest = IsolationForest(
        n_estimators=n_estimators, max_samples=256, random_state=seed, n_jobs=-1
    ).fit(X_train_s)

   
    val_scores = -iforest.score_samples(X_val_s)

   
    percentiles = [99.0, 99.3, 99.5, 99.7, 99.9, 99.95, 99.99]
    best = None
    for p in percentiles:
        thr = float(np.percentile(val_scores, p))
        pred = (val_scores >= thr).astype(int)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_val, pred, average="binary", zero_division=0
        )
        best = max(best or (None,)*5, (f1, p, thr, prec, rec))
    f1, p, thr, prec, rec = best
    auc = roc_auc_score(y_val, val_scores)

    joblib.dump(scaler, ART / "scaler.joblib")
    joblib.dump(iforest, ART / "iforest.joblib")
    json.dump(
        {
            "version": "1.0.0",
            "feature_cols": feature_cols,
            "threshold": thr,
            "percentile": p,
            "metrics": {"val_auc": float(auc), "val_f1": float(f1), "val_precision": float(prec), "val_recall": float(rec)},
        },
        open(ART / "metadata.json", "w"),
        indent=2,
    )
    print("Saved artifacts →", ART)
    print(f"Chosen thr={thr:.4f} (p={p}%)  |  AUC={auc:.3f}  F1={f1:.3f}  P={prec:.3f}  R={rec:.3f}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    args = ap.parse_args()
    train(args.csv)
