# Real-Time Anomaly Detection (IsolationForest → API → Stream → SQLite → Streamlit)

Detects anomalous credit card transactions in real time with:
- Offline model training (IsolationForest) + threshold tuning
- FastAPI scoring service (`/score`, `/healthz`)
- Streamer that feeds a CSV like a live feed & logs predictions to SQLite
- Live Streamlit dashboard with KPIs, trend chart, recent anomalies, top scores
- Reproducible artifacts (`StandardScaler`, model, threshold)

 Dataset
[Kaggle: Credit Card Fraud Detection] – 284,807 transactions with `V1..V28` PCA features, `Amount`, `Class` (fraud=1).  
>  Not included in the repo. Download `creditcard.csv` and pass the path to the streamer or the dashboard sidebar.

 Quick start

 0) Clone + install
```bash
git clone https://github.com/Yigituysgl/real-time-anomaly-detection.git
cd real-time-anomaly-detection
python -m pip install -r requirements.txt
