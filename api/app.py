import os
import sys
import uuid

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
import logging
import concurrent.futures
import threading
import time

import cv2
from flask import Flask, Response, jsonify, request
from solana_client import submit_to_solana
from ultralytics import YOLO

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from detection import NearMissMonitor, draw_car_detections
from lane import detect_lanes
from pinata_client import PinataClient, PINATA_GATEWAY

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "..", "uploads")
CLIP_FOLDER   = os.path.join(os.path.dirname(__file__), "..", "clips")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CLIP_FOLDER, exist_ok=True)

MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "src", "weights", "yolov8n.pt"
)
FRAME_WIDTH                  = 1280
FRAME_HEIGHT                 = 720
TARGET_FPS                   = 30
CAR_CONFIDENCE_THRESHOLD     = 0.5
TARGET_VEHICLE_CLASSES       = {"car", "truck", "bus"}
FOCAL_LENGTH                 = 1000.0
KNOWN_CAR_WIDTH_M            = 2.0
NEAR_MISS_DISTANCE_THRESHOLD_M = 30.0
NEAR_MISS_DECREASE_STREAK    = 3
CLIP_PRE_ROLL_FRAMES  = 150  # 5 s before alert  (150 / 30 fps = 5 s)
CLIP_POST_ROLL_FRAMES = 150  # 5 s after  alert  → total clip ≈ 10 s

# ── Config ────────────────────────────────────────────────────────────────────
PINATA_JWT = os.getenv("PINATA_JWT", "")
CAMERA_ID  = os.getenv("CAMERA_ID", "cam-01")

# Thread pool for parallel IPFS uploads (up to 4 concurrent)
_upload_executor = concurrent.futures.ThreadPoolExecutor(max_workers=4, thread_name_prefix="ipfs-upload")

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

app   = Flask(__name__)
jobs: dict[str, dict] = {}
recorded_incidents: list = []
model = None
model_lock = threading.Lock()


def get_model():
    global model
    with model_lock:
        if model is None:
            model = YOLO(MODEL_PATH)
    return model


def _make_pinata_group(video_id: str) -> str | None:
    """Create a Pinata group named '<camera_id>/<video_id>' and return its ID."""
    if not PINATA_JWT:
        return None
    try:
        pinata     = PinataClient(jwt_token=PINATA_JWT)
        group_name = f"{CAMERA_ID}/{video_id}"
        group_id   = pinata.create_group(group_name)
        LOGGER.info("Pinata group created: %s  id=%s", group_name, group_id)
        return group_id
    except Exception as e:
        LOGGER.error("Failed to create Pinata group: %s", e)
        return None


def _upload_and_record(clip_path: str, incident_meta: dict, group_id: str | None) -> None:
    """Upload clip to IPFS (in the job's Pinata group); submit to Solana for HIGH/CRITICAL."""
    try:
        severity          = incident_meta.get("severity_label", "").upper()
        onchain_severities = {"HIGH", "CRITICAL"}
        clip_cid = ""
        tx       = ""
        clip_id  = incident_meta.get("clip_id", uuid.uuid4().hex)

        if not PINATA_JWT:
            LOGGER.warning("PINATA_JWT not set — skipping IPFS upload.")
        else:
            pinata = PinataClient(jwt_token=PINATA_JWT)
            result = pinata.upload_clip(
                clip_path,
                name=f"incident_{clip_id}",
                group_id=group_id,          # ← placed inside camid/videoid folder
                keyvalues={
                    "clip_id":    clip_id,
                    "camera_id":  CAMERA_ID,
                    "severity":   severity,
                    "vehicle":    incident_meta["vehicle_class"],
                    "occurred_at": str(incident_meta["occurred_at"]),
                },
            )
            clip_cid = result.ipfs_cid
            LOGGER.info(
                "Clip uploaded to IPFS group=%s  cid=%s  severity=%s",
                group_id, clip_cid, severity,
            )

            if severity in onchain_severities:
                tx = submit_to_solana(incident_meta, clip_cid) or ""
                LOGGER.info("Submitted to Solana: %s", tx)
            else:
                LOGGER.info("Severity %s — IPFS only, not on-chain.", severity)

        recorded_incidents.append(
            {
                "clip_id":        clip_id,
                "occurred_at":    incident_meta["occurred_at"],
                "vehicle_class":  incident_meta["vehicle_class"],
                "severity_label": incident_meta["severity_label"],
                "severity_score": incident_meta["severity_score"],
                "distance_m":     round(incident_meta["distance_m"], 2),
                "ttc_s":          round(incident_meta["ttc_s"], 2),
                "clip_cid":       clip_cid,
                "onchain":        severity in onchain_severities,
                "tx":             tx,
            }
        )

    except Exception as e:
        LOGGER.error("Upload/record failed: %s", e)
    finally:
        try:
            os.remove(clip_path)
        except Exception:
            pass


def _save_clip(frames: list, fps: int, path: str) -> None:
    if not frames:
        return
    h, w = frames[0].shape[:2]
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        out.write(f)
    out.release()


def generate_frames(video_path: str, job_id: str):
    jobs[job_id]["status"] = "running"
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        jobs[job_id]["status"] = "failed"
        return

    group_id = jobs[job_id].get("group_id")          # Pinata group for this cam/video

    yolo = get_model()
    frame_budget = 1.0 / TARGET_FPS
    near_miss_monitor = NearMissMonitor(
        distance_alert_threshold_m=NEAR_MISS_DISTANCE_THRESHOLD_M,
        required_decrease_streak=NEAR_MISS_DECREASE_STREAK,
    )
    previous_frame_start = None

    from collections import deque
    raw_buffer: deque        = deque(maxlen=CLIP_PRE_ROLL_FRAMES)
    post_roll_frames: list   = []
    post_roll_remaining      = 0
    active_incident: dict | None = None

    while cap.isOpened():
        if jobs.get(job_id, {}).get("stop"):
            break

        start = time.perf_counter()
        frame_dt_s = (
            max(start - previous_frame_start, 1e-3)
            if previous_frame_start is not None
            else frame_budget
        )
        previous_frame_start = start

        has_frame, frame = cap.read()
        if not has_frame:
            break

        resized = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        raw_buffer.append(resized.copy())

        results      = yolo(resized, verbose=False)
        vehicle_boxes = [
            tuple(map(int, box.xyxy[0]))
            for result in results
            for box in result.boxes
            if yolo.names[int(box.cls[0])].lower() in TARGET_VEHICLE_CLASSES
            and float(box.conf[0]) >= CAR_CONFIDENCE_THRESHOLD
        ]
        lane_frame  = detect_lanes(resized, vehicle_boxes=vehicle_boxes)
        frame_alerts = []
        output = draw_car_detections(
            lane_frame,
            results,
            yolo.names,
            CAR_CONFIDENCE_THRESHOLD,
            FOCAL_LENGTH,
            KNOWN_CAR_WIDTH_M,
            frame_dt_s,
            near_miss_monitor,
            TARGET_VEHICLE_CLASSES,
            alert_sink=frame_alerts,
        )

        # Post-roll accumulation — fire upload immediately once complete
        if post_roll_remaining > 0:
            post_roll_frames.append(resized.copy())
            post_roll_remaining -= 1
            if post_roll_remaining == 0 and active_incident:
                clip_path   = os.path.join(
                    CLIP_FOLDER, f"incident_{active_incident['clip_id']}.mp4"
                )
                clip_frames = list(raw_buffer) + post_roll_frames
                _save_clip(clip_frames, TARGET_FPS, clip_path)
                # Fire-and-forget upload in the thread pool — does NOT block the stream
                _upload_executor.submit(
                    _upload_and_record, clip_path, active_incident, group_id
                )
                post_roll_frames = []
                active_incident  = None

        for alert in frame_alerts:
            if active_incident is None and post_roll_remaining == 0:
                if alert["severity_label"] in ("LOW", "MEDIUM", "HIGH", "CRITICAL"):
                    active_incident            = dict(alert)
                    active_incident["clip_id"] = uuid.uuid4().hex
                    post_roll_remaining        = CLIP_POST_ROLL_FRAMES
                    LOGGER.info(
                        "Alert triggered — capturing clip for %s (clip_id=%s)",
                        alert["severity_label"],
                        active_incident["clip_id"],
                    )

        _, buffer = cv2.imencode(".jpg", output, [cv2.IMWRITE_JPEG_QUALITY, 80])
        yield (
            b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
        )

        elapsed    = time.perf_counter() - start
        sleep_time = frame_budget - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

    cap.release()
    jobs[job_id]["status"] = "completed"


@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    f = request.files["file"]
    if f.filename == "":
        return jsonify({"error": "Empty filename"}), 400
    ext = os.path.splitext(f.filename)[1].lower()
    if ext not in {".mp4", ".avi", ".mov", ".mkv"}:
        return jsonify({"error": "Invalid file type"}), 400

    video_id  = uuid.uuid4().hex[:8]
    save_path = os.path.join(UPLOAD_FOLDER, f"{video_id}{ext}")
    f.save(save_path)

    # Create Pinata group: <camera_id>/<video_id> — done in background so upload is instant
    group_id = None
    if PINATA_JWT:
        try:
            pinata   = PinataClient(jwt_token=PINATA_JWT)
            group_id = pinata.create_group(f"{CAMERA_ID}/{video_id}")
            LOGGER.info("Pinata group ready: %s/%s  id=%s", CAMERA_ID, video_id, group_id)
        except Exception as e:
            LOGGER.error("Could not create Pinata group: %s", e)

    jobs[video_id] = {
        "status":    "queued",
        "path":      save_path,
        "stop":      False,
        "camera_id": CAMERA_ID,
        "group_id":  group_id,
    }
    return jsonify({
        "video_id":  video_id,
        "camera_id": CAMERA_ID,
        "group_id":  group_id,
        "message":   "upload successful",
    })


@app.route("/start/<video_id>", methods=["POST"])
def start(video_id):
    job = jobs.get(video_id)
    if not job:
        return jsonify({"error": "Unknown video_id"}), 404
    job["status"] = "running"
    job["stop"] = False
    return jsonify({"video_id": video_id, "status": "running"})


@app.route("/stream/<video_id>")
def stream(video_id):
    job = jobs.get(video_id)
    if not job:
        return jsonify({"error": "Unknown video_id"}), 404
    return Response(
        generate_frames(job["path"], video_id),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/stop/<video_id>", methods=["POST"])
def stop(video_id):
    job = jobs.get(video_id)
    if not job:
        return jsonify({"error": "Unknown video_id"}), 404
    job["stop"] = True
    return jsonify({"video_id": video_id, "status": "stopped"})


@app.route("/status/<video_id>")
def status(video_id):
    job = jobs.get(video_id)
    if not job:
        return jsonify({"error": "Unknown video_id"}), 404
    return jsonify({"video_id": video_id, "status": job["status"]})


@app.route("/clips/<video_id>")
def get_clips(video_id):
    """List all clips stored in the Pinata group for this video."""
    job = jobs.get(video_id)
    if not job:
        return jsonify({"error": "Unknown video_id"}), 404

    group_id = job.get("group_id")
    if not group_id or not PINATA_JWT:
        return jsonify({"clips": [], "group_id": None, "folder": None})

    try:
        pinata = PinataClient(jwt_token=PINATA_JWT)
        files  = pinata.list_group_files(group_id)
        clips  = [
            {
                "name":       f.get("name", ""),
                "cid":        f.get("cid", ""),
                "ipfs_url":   f"{PINATA_GATEWAY}/{f['cid']}",
                "size":       f.get("size", 0),
                "created_at": f.get("created_at", ""),
                "keyvalues":  f.get("keyvalues", {}),
            }
            for f in files
        ]
        return jsonify({
            "folder":   f"{CAMERA_ID}/{video_id}",
            "group_id": group_id,
            "clips":    clips,
        })
    except Exception as e:
        LOGGER.error("Failed to list Pinata group files: %s", e)
        return jsonify({"error": str(e)}), 500


@app.route("/incidents")
def get_incidents():
    return jsonify({"incidents": recorded_incidents})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
