import time, sqlite3, requests, argparse
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "storage" / "anomaly_demo.sqlite"

def stream(csv_path, api_url, limit=3000, speedup=3000, max_sleep=0.03):
    df = pd.read_csv(csv_path).drop_duplicates().sort_values("Time").reset_index(drop=True)
    feats = [c for c in df.columns if c.startswith("V")] + ["Amount"]

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS events_clean (event_id INTEGER, ts REAL, amount REAL);""")
    cur.execute("""CREATE TABLE IF NOT EXISTS predictions (event_id INTEGER, ts REAL, score REAL, is_anomaly INTEGER, label INTEGER);""")
    conn.commit()

    prev_ts = float(df["Time"].iloc[0])
    sent = 0
    for idx, row in df.iterrows():
        payload = {"ts": float(row["Time"]), "features": {k: float(row[k]) for k in feats}}
        r = requests.post(api_url, json=payload, timeout=5); r.raise_for_status()
        res = r.json()

        cur.execute("INSERT INTO events_clean VALUES (?, ?, ?)", (int(idx), float(row["Time"]), float(row["Amount"])))
        cur.execute("INSERT INTO predictions VALUES (?, ?, ?, ?, ?)",
                    (int(idx), float(row["Time"]), float(res["score"]), int(bool(res["is_anomaly"])), int(row.get("Class", 0))))

        if sent % 250 == 0:
            conn.commit()
            print(f"sent={sent}, ts={row['Time']:.1f}, score={res['score']:.3f}, anomaly={res['is_anomaly']}")

        delay = max((float(row["Time"]) - prev_ts)/speedup, 0.0)
        time.sleep(min(delay, max_sleep))
        prev_ts = float(row["Time"])

        sent += 1
        if limit and sent >= limit: break

    conn.commit(); conn.close()
    print("Done. Rows sent:", sent)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--api", default="http://127.0.0.1:8000/score")
    ap.add_argument("--limit", type=int, default=3000)
    ap.add_argument("--speedup", type=int, default=3000)
    ap.add_argument("--max_sleep", type=float, default=0.03)
    args = ap.parse_args()
    stream(args.csv, args.api, args.limit, args.speedup, args.max_sleep)

