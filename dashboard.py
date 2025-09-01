
import os, time, threading, json, sqlite3
import pandas as pd
import requests
import streamlit as st

# optional auto-refresh plugin
try:
    from streamlit_autorefresh import st_autorefresh
    AUTO = True
except Exception:
    AUTO = False

DB_PATH = r"C:\Users\yigit\storage\anomaly_demo.sqlite"          # <-- your DB
ART_META = r"C:\Users\yigit\artifacts\metadata.json"             # <-- to read saved threshold (if exists)

st.set_page_config(page_title="Real-Time Anomaly Dashboard", layout="wide")
st.title("Real-Time Anomaly Detection — Live Dashboard")
if AUTO: st_autorefresh(interval=2000, key="refresh")
else:    st.caption("Tip: auto-refresh plugin not installed. Install later:  pip install streamlit-autorefresh")

@st.cache_resource
def get_conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def load_predictions(limit=30000):
    conn = get_conn()
    df = pd.read_sql_query(
        "SELECT event_id, ts, score, is_anomaly, label FROM predictions ORDER BY ts DESC LIMIT ?;",
        conn, params=[limit]
    )
    if not df.empty:
        df["is_anomaly"] = df["is_anomaly"].astype(int)
        df["label"]      = df["label"].astype(int)
    return df

# ---------------- Sidebar controls ----------------
st.sidebar.header("Controls")

API_URL = st.sidebar.text_input("Scoring API URL", "http://127.0.0.1:8000/score")
CSV_PATH = st.sidebar.text_input("CSV path to stream",
    r"C:\Users\yigit\Downloads\archive(33)\creditcard.csv")

limit    = int(st.sidebar.number_input("Rows to stream per click", 100, 50000, 3000, step=100))
speedup  = int(st.sidebar.number_input("Time speedup (higher=faster)", 100, 20000, 3000, step=100))
max_sleep= float(st.sidebar.number_input("Max sleep per row (sec)", 0.0, 1.0, 0.03, step=0.01, format="%.2f"))

if "streaming" not in st.session_state: st.session_state.streaming = False
if "stream_thread" not in st.session_state: st.session_state.stream_thread = None

def stream_worker(csv_path, api_url, limit, speedup, max_sleep):
    # load & normalize CSV exactly like training did
    df = pd.read_csv(csv_path).drop_duplicates().sort_values("Time").reset_index(drop=True)
    feats = [c for c in df.columns if c.startswith("V")] + ["Amount"]

    conn = get_conn()
    cur  = conn.cursor()
    # resume from last event_id if present
    try:
        last = pd.read_sql_query("SELECT MAX(event_id) AS m FROM predictions;", conn)["m"].iloc[0]
        start_idx = 0 if pd.isna(last) else int(last) + 1
    except Exception:
        start_idx = 0

    if start_idx >= len(df): return
    prev_ts = float(df.at[start_idx, "Time"])
    sent = 0

    for idx in range(start_idx, min(start_idx + limit, len(df))):
        if not st.session_state.streaming:
            break
        row = df.iloc[idx]
        payload = {"ts": float(row["Time"]),
                   "features": {k: float(row[k]) for k in feats}}
        try:
            r = requests.post(api_url, json=payload, timeout=5)
            r.raise_for_status()
            res = r.json()  # {'score':..., 'is_anomaly':...}
        except Exception as e:
            st.session_state.streaming = False
            st.toast(f"Streaming stopped: {e}", icon="⚠️")
            break

        cur.execute("INSERT INTO events_clean VALUES (?, ?, ?)",
                    (int(idx), float(row["Time"]), float(row["Amount"])))
        cur.execute("INSERT INTO predictions VALUES (?, ?, ?, ?, ?)",
                    (int(idx), float(row["Time"]), float(res["score"]),
                     int(bool(res["is_anomaly"])), int(row.get("Class", 0))))
        if sent % 200 == 0: conn.commit()

        delay = max((float(row["Time"]) - prev_ts) / speedup, 0.0)
        time.sleep(min(delay, max_sleep))
        prev_ts = float(row["Time"])
        sent += 1

    conn.commit()
    st.toast(f"Streamed {sent} rows.", icon="✅")

cols = st.sidebar.columns(2)
if cols[0].button("Start streaming", disabled=st.session_state.streaming):
    if not os.path.exists(CSV_PATH):
        st.sidebar.error(f"CSV not found: {CSV_PATH}")
    else:
        # quick health check
        try:
            requests.get(API_URL.replace("/score","/healthz"), timeout=3)
        except Exception as e:
            st.sidebar.error(f"API not reachable at {API_URL}: {e}")
        else:
            st.session_state.streaming = True
            st.session_state.stream_thread = threading.Thread(
                target=stream_worker,
                args=(CSV_PATH, API_URL, limit, speedup, max_sleep),
                daemon=True
            )
            st.session_state.stream_thread.start()
            st.sidebar.success("Streaming started…")

if cols[1].button("Stop", type="secondary", disabled=not st.session_state.streaming):
    st.session_state.streaming = False
    st.sidebar.info("Stop requested.")

# ---- Load data & UI threshold ----
df = load_predictions()

# default threshold = saved from training if available, else 99th percentile of scores
def default_threshold(scores: pd.Series) -> float:
    if os.path.exists(ART_META):
        try:
            return float(json.load(open(ART_META))["threshold"])
        except Exception:
            pass
    return float(scores.quantile(0.99)) if len(scores) else 0.5

if len(df):
    thr_default = default_threshold(df["score"])
    min_thr = float(df["score"].quantile(0.95))
    max_thr = float(df["score"].quantile(0.999))
    thr_ui = st.sidebar.slider("Display threshold (re-flag by score)",
                               min_value=min_thr, max_value=max_thr, value=thr_default)
    df["flag_ui"] = (df["score"] >= thr_ui).astype(int)
else:
    thr_ui = 0.5
    df["flag_ui"] = 0

# ---- KPIs (using UI threshold) ----
total = len(df)
pred_pos = int(df["flag_ui"].sum()) if total else 0
tp = int(((df["flag_ui"]==1) & (df["label"]==1)).sum()) if total else 0
precision = (tp / pred_pos * 100) if pred_pos else 0.0
rate = (pred_pos / total * 100) if total else 0.0

c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows scored (loaded)", f"{total:,}")
c2.metric("Anomalies flagged (UI)", f"{pred_pos:,}")
c3.metric("Anomaly rate", f"{rate:.2f}%")
c4.metric("Precision (labels in stream)", f"{precision:.1f}%")

# ---- Trend chart ----
if total:
    df_m = df.copy()
    df_m["minute"] = (df_m["ts"] // 60).astype(int)
    base = int(df_m["minute"].min())
    df_m["t"] = df_m["minute"] - base

    agg = df_m.groupby("t", as_index=False).agg(
        rows=("event_id","count"),
        anomalies=("flag_ui","sum"),
        frauds=("label","sum")
    ).sort_values("t")

    st.subheader("Anomalies per minute (UI threshold)")
    st.line_chart(agg.set_index("t")[["anomalies", "frauds"]])

    st.subheader("Most recent anomalies")
    k = st.slider("Rows to show", 10, 200, 50)
    recent = df[df["flag_ui"]==1].head(k)
    st.dataframe(recent[["event_id","ts","score","flag_ui","label"]], use_container_width=True)

    st.subheader("Top 50 by anomaly score")
    top = df.sort_values("score", ascending=False).head(50)
    st.dataframe(top[["event_id","ts","score","flag_ui","label"]], use_container_width=True)
else:
    st.info("No predictions yet. Use **Start streaming** in the sidebar.")
