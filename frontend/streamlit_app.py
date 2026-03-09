import time

import requests
import streamlit as st

FLASK_URL = "http://localhost:5000"

st.set_page_config(
    page_title="RoadSense — Live Detection",
    page_icon="🚗",
    layout="wide",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background-color: #0a0a0f;
    color: #e8e8f0;
}
.stApp { background: #0a0a0f; }

.main-title {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 3.2rem;
    letter-spacing: -0.03em;
    background: linear-gradient(135deg, #00ff9d 0%, #00cfff 50%, #a78bfa 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1;
    margin-bottom: 0.2rem;
}
.sub-title {
    font-family: 'Space Mono', monospace;
    font-size: 0.78rem;
    color: #555570;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    margin-bottom: 2rem;
}
.status-badge {
    display: inline-block;
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.12em;
    padding: 0.3rem 0.8rem;
    border-radius: 20px;
    margin-bottom: 1rem;
}
.status-running { background:#0d2e1a; color:#00ff9d; border:1px solid #00ff9d40; }
.status-idle    { background:#1e1e30; color:#555570; border:1px solid #2a2a40; }
.status-failed  { background:#2e0d0d; color:#ff4444; border:1px solid #ff444440; }

.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, #1e1e30, transparent);
    margin: 1.2rem 0;
}
.stat-card {
    background: #13131f;
    border: 1px solid #1e1e30;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.6rem;
}
.stat-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    color: #555570;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-bottom: 0.2rem;
}
.stat-value {
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 1.3rem;
    color: #00ff9d;
}

.incident-card {
    background: #0d0d18;
    border: 1px solid #1e1e30;
    border-radius: 10px;
    padding: 0.9rem 1.1rem;
    margin-bottom: 0.6rem;
    border-left: 3px solid #555570;
}
.incident-card.CRITICAL { border-left-color: #ff2244; }
.incident-card.HIGH     { border-left-color: #ff8800; }
.incident-card.MEDIUM   { border-left-color: #ffcc00; }

.incident-top {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.4rem;
}
.severity-pill {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    font-weight: 700;
    padding: 0.15rem 0.6rem;
    border-radius: 20px;
    letter-spacing: 0.1em;
}
.pill-CRITICAL { background:#ff224422; color:#ff2244; border:1px solid #ff224440; }
.pill-HIGH     { background:#ff880022; color:#ff8800; border:1px solid #ff880040; }
.pill-MEDIUM   { background:#ffcc0022; color:#ffcc00; border:1px solid #ffcc0040; }
.pill-LOW      { background:#44ff8822; color:#44ff88; border:1px solid #44ff8840; }

.incident-meta {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    color: #555570;
    margin-bottom: 0.5rem;
}
.incident-cid {
    font-family: 'Space Mono', monospace;
    font-size: 0.6rem;
    color: #333350;
    word-break: break-all;
}
.incident-cid a { color: #00cfff; text-decoration: none; }
.incident-cid a:hover { text-decoration: underline; }

.stButton > button {
    background: linear-gradient(135deg, #00ff9d, #00cfff) !important;
    color: #0a0a0f !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    border: none !important;
    border-radius: 8px !important;
    width: 100% !important;
}
.stButton > button:hover { opacity: 0.85 !important; }
</style>
""",
    unsafe_allow_html=True,
)

if "status" not in st.session_state:
    st.session_state.status = "idle"
if "clips" not in st.session_state:
    st.session_state.clips = []
if "folder" not in st.session_state:
    st.session_state.folder = None

col_left, col_mid, col_right = st.columns([1, 2.2, 1.1], gap="large")

with col_left:
    st.markdown('<div class="main-title">Road<br>Sense</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Live Road Intelligence</div>', unsafe_allow_html=True
    )
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Drop a road video",
        type=["mp4", "avi", "mov", "mkv"],
    )
    st.markdown("<br>", unsafe_allow_html=True)

    start_btn = st.button("▶  Start Detection", disabled=uploaded_file is None)
    stop_btn = st.button("■  Stop", disabled="video_id" not in st.session_state)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    status_placeholder = st.empty()
    info_placeholder = st.empty()

    status_class = {
        "idle": "status-idle",
        "running": "status-running",
        "failed": "status-failed",
        "completed": "status-idle",
    }.get(st.session_state.status, "status-idle")

    status_placeholder.markdown(
        f'<span class="status-badge {status_class}">⬤ &nbsp;{st.session_state.status.upper()}</span>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown(
        """
    <div class="stat-card">
        <div class="stat-label">Model</div>
        <div class="stat-value" style="font-size:0.95rem;color:#a78bfa;">YOLOv8n</div>
    </div>
    <div class="stat-card">
        <div class="stat-label">Targets</div>
        <div class="stat-value" style="font-size:0.9rem;color:#00cfff;">Car · Truck · Bus</div>
    </div>
    <div class="stat-card">
        <div class="stat-label">Chain</div>
        <div class="stat-value" style="font-size:0.9rem;color:#00ff9d;">Solana Devnet</div>
    </div>
    <div class="stat-card">
        <div class="stat-label">Storage</div>
        <div class="stat-value" style="font-size:0.9rem;color:#e8e8f0;">IPFS · Pinata</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

with col_mid:
    stream_placeholder = st.empty()
    stream_placeholder.markdown(
        """
    <div style="background:#0d0d18;border:1px solid #1e1e30;border-radius:14px;height:500px;
        display:flex;flex-direction:column;align-items:center;justify-content:center;gap:1rem;">
        <div style="font-size:3rem;opacity:0.15;">🎥</div>
        <div style="font-family:'Space Mono',monospace;font-size:0.72rem;color:#333350;
            letter-spacing:0.15em;text-transform:uppercase;">Upload a video and press Start</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

with col_right:
    st.markdown(
        '<div class="stat-label" style="margin-bottom:0.8rem;">📁 &nbsp;IPFS CLIPS</div>',
        unsafe_allow_html=True,
    )
    clips_placeholder  = st.empty()
    folder_placeholder = st.empty()

    def render_clips(clips: list, folder: str | None = None):
        if folder:
            folder_placeholder.markdown(
                f'<div class="incident-cid" style="color:#555570;margin-bottom:0.6rem;">'
                f'📂 {folder}</div>',
                unsafe_allow_html=True,
            )

        if not clips:
            clips_placeholder.markdown(
                """
            <div style="font-family:'Space Mono',monospace;font-size:0.65rem;color:#333350;
                text-align:center;padding:2rem 0;">No clips yet — waiting…</div>
            """,
                unsafe_allow_html=True,
            )
            return

        SEV_COLORS = {
            "CRITICAL": "#ff2244",
            "HIGH":     "#ff8800",
            "MEDIUM":   "#ffcc00",
            "LOW":      "#44ff88",
        }

        html = ""
        for clip in reversed(clips):
            kv       = clip.get("keyvalues") or {}
            sev      = (kv.get("severity") or "LOW").upper()
            color    = SEV_COLORS.get(sev, "#555570")
            cid      = clip.get("cid", "")
            name     = clip.get("name", cid[:12])
            url      = clip.get("ipfs_url", f"https://gateway.pinata.cloud/ipfs/{cid}")
            ts_raw   = clip.get("created_at", "")
            vehicle  = kv.get("vehicle", "")
            occurred = kv.get("occurred_at", "")

            # Format timestamp
            try:
                from datetime import datetime as _dt
                ts = _dt.fromisoformat(ts_raw.replace("Z", "+00:00")).strftime("%H:%M:%S")
            except Exception:
                ts = ts_raw[:19] if ts_raw else "—"

            html += f"""
            <div style="background:#0d0d18;border:1px solid #1e1e30;border-radius:10px;
                        padding:0.8rem 1rem;margin-bottom:0.55rem;
                        border-left:3px solid {color};">
                <div style="display:flex;justify-content:space-between;align-items:center;
                             margin-bottom:0.35rem;">
                    <span style="font-family:'Syne',sans-serif;font-weight:700;
                                 font-size:0.82rem;color:#e8e8f0;">
                        {vehicle.capitalize() or "Vehicle"}
                    </span>
                    <span style="font-family:'Space Mono',monospace;font-size:0.62rem;
                                 font-weight:700;padding:0.12rem 0.5rem;border-radius:20px;
                                 background:{color}22;color:{color};border:1px solid {color}40;">
                        {sev}
                    </span>
                </div>
                <div style="font-family:'Space Mono',monospace;font-size:0.62rem;
                             color:#555570;margin-bottom:0.35rem;">
                    {ts}
                </div>
                <div style="font-family:'Space Mono',monospace;font-size:0.6rem;
                             color:#333350;word-break:break-all;">
                    🎬 <a href="{url}" target="_blank"
                          style="color:#00cfff;text-decoration:none;">
                        {cid[:20]}…
                    </a>
                </div>
            </div>
            """
        clips_placeholder.markdown(html, unsafe_allow_html=True)

    render_clips(st.session_state.get("clips", []), st.session_state.get("folder"))


# ── Button handlers ───────────────────────────────────────────────────────────

if start_btn and uploaded_file is not None:
    with st.spinner("Uploading video..."):
        try:
            res = requests.post(
                f"{FLASK_URL}/upload",
                files={
                    "file": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        uploaded_file.type,
                    )
                },
                timeout=60,
            )
            res.raise_for_status()
            data     = res.json()
            video_id = data["video_id"]
            st.session_state.video_id = video_id
            st.session_state.folder   = data.get("folder") or f"{data.get('camera_id','cam')}/{video_id}"
            requests.post(f"{FLASK_URL}/start/{video_id}", timeout=10)
            st.session_state.status = "running"
            st.session_state.clips  = []
            st.rerun()
        except Exception as e:
            st.session_state.status = "failed"
            info_placeholder.error(f"Error: {e}")

if stop_btn and "video_id" in st.session_state:
    try:
        requests.post(f"{FLASK_URL}/stop/{st.session_state.video_id}", timeout=5)
    except Exception:
        pass
    st.session_state.status = "idle"
    del st.session_state["video_id"]
    st.rerun()


# ── Live streaming loop ───────────────────────────────────────────────────────

if st.session_state.get("status") == "running" and "video_id" in st.session_state:
    video_id   = st.session_state.video_id
    stream_url = f"{FLASK_URL}/stream/{video_id}"

    with col_mid:
        stream_placeholder.markdown(
            f"""<div style="border-radius:14px;overflow:hidden;border:1px solid #1e1e30;">
                <img src="{stream_url}" style="width:100%;display:block;border-radius:14px;" />
            </div>""",
            unsafe_allow_html=True,
        )

    status_placeholder.markdown(
        '<span class="status-badge status-running">⬤ &nbsp;RUNNING</span>',
        unsafe_allow_html=True,
    )

    # Check job completion
    try:
        status_res = requests.get(f"{FLASK_URL}/status/{video_id}", timeout=3)
        job_status = status_res.json().get("status", "running")
        if job_status in ("completed", "failed"):
            st.session_state.status = job_status
            st.rerun()
    except Exception:
        pass

    # Poll Pinata group for new clips every ~30 s
    # Streamlit reruns every 3 s; we fetch clips only on every 10th rerun ≈ 30 s
    poll_counter = st.session_state.get("poll_counter", 0) + 1
    st.session_state.poll_counter = poll_counter

    if poll_counter % 10 == 1:   # fetch on 1st run then every 30 s
        try:
            clip_res = requests.get(f"{FLASK_URL}/clips/{video_id}", timeout=10)
            payload  = clip_res.json()
            st.session_state.clips  = payload.get("clips", [])
            st.session_state.folder = payload.get("folder", st.session_state.folder)
        except Exception:
            pass

    render_clips(st.session_state.clips, st.session_state.folder)

    time.sleep(3)
    st.rerun()

