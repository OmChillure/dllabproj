# DL Project: Streamlit Frontend + Flask API for Live Road Video Detection

This project already has the core video intelligence in `src/`:
- Lane detection (`src/lane.py`)
- YOLO vehicle detection (`src/video_processor.py`)
- Distance, TTC, risk scoring, near-miss alerts (`src/detection.py`)

The next task is to connect this pipeline to a frontend where users upload a road video and watch live processed output.

## Goal
Build a UI where:
1. User opens a Streamlit app.
2. User uploads a road video.
3. User clicks **Start**.
4. Flask API starts running YOLO + lane + distance/TTC logic on that video.
5. Streamlit frontend plays the uploaded video in real time with live detections (lanes, YOLO boxes, distance, TTC, risk labels) overlaid on the video.

## Tech Stack
- Frontend: `Streamlit`
- Backend API: `Flask`
- Inference/processing: existing OpenCV + Ultralytics YOLO code in `src/`

## Folder Structure
Create these files:

```text
dlproj/
  src/
    main.py
    video_processor.py
    detection.py
    lane.py
  api/
    app.py
    service.py
  frontend/
    streamlit_app.py
  uploads/
  outputs/
```

## Implementation Steps

### 1) Refactor processing into a video stream generator
Current `process_video()` displays via `cv2.imshow`.  
For API/Streamlit, expose a generator that yields processed video output:
- Input: uploaded video path
- Output: processed video chunks/frames for live playback in frontend

Suggested function:
- `run_video_stream(video_path, config) -> Iterator[bytes]`

Inside that function, reuse:
- `detect_lanes(...)`
- `draw_car_detections(...)`
- YOLO model inference from `video_processor.py`

### 2) Build Flask APIs
In `api/app.py`, add:

1. `POST /upload`
- Accept video file
- Save to `uploads/`
- Return `video_id` and stored path metadata

2. `POST /start/<video_id>`
- Mark a job as started
- Initialize processing state/job entry

3. `GET /stream/<video_id>`
- Stream processed video with detections using Flask `Response` with:
  - `mimetype="multipart/x-mixed-replace; boundary=frame"`
- This endpoint should yield processed video data produced by your generator

4. `GET /status/<video_id>` (optional but useful)
- Return job state: `queued/running/completed/failed`

### 3) Build Streamlit UI
In `frontend/streamlit_app.py`:
- Show file uploader: `st.file_uploader(..., type=["mp4", "avi", "mov"])`
- Button: `Start Detection`
- On click:
  - Upload file to Flask `/upload`
  - Call `/start/<video_id>`
  - Play processed video from `/stream/<video_id>` in real time

Display options:
- Embed stream URL directly (recommended), or
- Use `st.image(...)` loop only if direct video embedding is not feasible.

### 4) Replace desktop display calls
In backend flow, do not use:
- `cv2.imshow(...)`
- `cv2.waitKey(...)`

Instead:
- Encode each processed frame:
  - `_, buffer = cv2.imencode(".jpg", frame)`
  - `frame_bytes = buffer.tobytes()`
- Yield frame bytes to stream response.

### 5) Keep config centralized
Move constants (model path, fps, thresholds, classes) into one config module so both Flask and local run can share settings.

### 6) Error handling
Add checks for:
- Invalid upload type
- Video cannot open
- Missing job/video id
- Runtime inference errors

Return clean JSON errors from Flask APIs.

## Minimal API Contract

### Upload
`POST /upload`
- Form-data: `file=<video>`
- Response:

```json
{
  "video_id": "abc123",
  "message": "upload successful"
}
```

### Start
`POST /start/abc123`
- Response:

```json
{
  "video_id": "abc123",
  "status": "running"
}
```

### Stream
`GET /stream/abc123`
- Response: live processed video stream (video playback with detections)

## Local Run Plan
Run backend:

```bash
python -m api.app
```

Run frontend:

```bash
streamlit run frontend/streamlit_app.py
```

## Done Criteria
The task is complete when all are true:
1. User can upload a road video in Streamlit.
2. User clicks Start and processing begins.
3. Video plays in UI with lane + YOLO + distance/TTC/risk overlays in real time.
4. Backend processing is served through Flask APIs.
