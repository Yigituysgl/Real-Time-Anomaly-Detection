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
bash 
0) install python -m pip install -r requirements.txt

1) start API
python api/serve.py  # -> http://127.0.0.1:8000/healthz

2) stream data (set your CSV path)
python stream/stream_sqlite.py --csv "C:\path\to\creditcard.csv" --limit 3000

3) open dashboard
streamlit run dashboard/dashboard.py


<img width="1872" height="867" alt="image" src="https://github.com/user-attachments/assets/ca1d22af-bd00-4078-93cb-377835a9a5b2" />

<img width="1858" height="868" alt="image" src="https://github.com/user-attachments/assets/5a4fb8ff-5c28-4ac3-a219-2ec90c0583d4" />

