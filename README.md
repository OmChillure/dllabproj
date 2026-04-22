# RoadSense

Real-time road-video intelligence. Detects vehicles, lanes, and near-misses, stores incident clips on IPFS (Pinata), and writes HIGH/CRITICAL incidents on-chain to Solana Devnet.

## Live

- **Frontend (Streamlit):** https://roadsense-app.streamlit.app/
- **Backend API (Flask on HF Spaces):** https://omchillure-dlproj.hf.space

## Stack

- **Frontend** — Streamlit
- **Backend** — Flask
- **CV/ML** — Ultralytics YOLOv8n, OpenCV
- **Storage** — IPFS via Pinata
- **Chain** — Solana Devnet (via AnchorPy)

## Repo layout

```
dlprojs/
├── dlbackend/          # Flask API + YOLO + Solana + Pinata
│   ├── api/
│   │   ├── app.py
│   │   └── solana_client.py
│   ├── src/
│   │   ├── detection.py
│   │   ├── lane.py
│   │   ├── pinata_client.py
│   │   ├── video_processor.py
│   │   └── weights/yolov8n_custom.pt
│   ├── Dockerfile
│   └── requirements.txt
└── dlfrontend/         # Streamlit UI
    ├── frontend/streamlit_app.py
    └── requirements.txt
```

## Run locally

### 1. Backend

```bash
cd dlbackend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Create a .env in dlbackend/
cat > .env <<EOF
PINATA_JWT=your_pinata_jwt
PROGRAM_ID=your_solana_program_id
REPORTER_PRIVATE_KEY=your_solana_private_key
CAMERA_ID=dashcam-001
PINATA_GATEWAY=https://<your-gateway>.mypinata.cloud/ipfs
EOF

python api/app.py
# Listens on http://localhost:5000
```

### 2. Frontend

In a second terminal:

```bash
cd dlfrontend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Create a .env in dlfrontend/
echo "BACKEND_URL=http://localhost:5000" > .env

streamlit run frontend/streamlit_app.py
# Opens http://localhost:8501
```

### 3. Use it

1. Drop a road video on the UI
2. Press **Start Detection**
3. Watch YOLO + lane overlays stream back
4. Near-misses → clip auto-saved to IPFS (Pinata)
5. HIGH / CRITICAL severity → transaction on Solana Devnet

## API endpoints

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/upload` | Upload video, returns `video_id` |
| `POST` | `/start/<video_id>` | Start processing job |
| `GET`  | `/stream/<video_id>` | MJPEG live stream of processed frames |
| `POST` | `/stop/<video_id>` | Stop running job |
| `GET`  | `/status/<video_id>` | `queued / running / completed / failed` |
| `GET`  | `/clips/<video_id>` | Clips uploaded for this job |
| `GET`  | `/incidents` | All recorded incidents |

## Deployment

- **Backend** — Docker on Hugging Face Spaces (CPU Basic). Weights file stored via HF's Xet storage.
- **Frontend** — Streamlit Community Cloud, `BACKEND_URL` set in Secrets.

## Environment variables

**Backend** (`dlbackend/.env` locally, HF Space Secrets in prod):
- `PINATA_JWT`
- `PROGRAM_ID`
- `REPORTER_PRIVATE_KEY`
- `CAMERA_ID`
- `PINATA_GATEWAY`

**Frontend** (`dlfrontend/.env` locally, Streamlit Secrets in prod):
- `BACKEND_URL`

## Contributors
| Name | Roll No |
|------|---------|
| Om Chillure | A3-45 |
| Adnan Dalal | A3-42 |
| Sakshi Mude | A3-48 |

**Guide:** Prof. V. R. Gupta  
**Course:** ECSP6003-1 Deep Learning Lab, Sem VI B.Tech ECS  
**College:** RCOEM, Nagpur
