import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
import time
from collections import deque

# ─────────────────────────────────────────────
# Konfiqurasiya
# ─────────────────────────────────────────────
VIDEO_PATHS = {
    "Vaqon 1": r"C:\Users\Lenovo\Downloads\В2-КАМ2_16-04-26_13-00-00.avi",
    "Vaqon 2": r"C:\Users\Lenovo\Downloads\В4-КАМ6_60042_16-04-26_18-00-00.avi",
    "Vaqon 3": r"C:\Users\Lenovo\Downloads\В2-КАМ4_16-04-26_13-00-00.avi",
    "Vaqon 4": r"C:\Users\Lenovo\Downloads\В2-КАМ6_16-04-26_13-00-00.avi",
    "Vaqon 5": r"C:\Users\Lenovo\Downloads\В3-КАМ6_60053_16-04-26_08-20-00.avi"
}

VAGON_POLY = np.array([
    [160,   20],
    [1120,  20],
    [1280, 720],
    [0,    720],
])

FPS                = 30
CALIBRATION_SEC    = 10
CALIBRATION_FRAMES = (FPS * CALIBRATION_SEC) // 3  # hər 3 frame-dən 1-i
CHANGE_THRESHOLD   = 3
SMOOTH_BUF_SIZE    = 15

def sixliq(n):
    if n <= 12:
        return "NORMAL", "#2ecc40", 0
    elif n <= 20:
        return "SIX", "#ff851b", 1
    else:
        return "COX SIX", "#ff4136", 2

# ─────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────
def init_state():
    defaults = {
        "running": False,
        "results": {name: {"count": 0, "level": 0, "color": "#2ecc40", "label": "NORMAL"} for name in VIDEO_PATHS},
        "yolo_model": None,
        "smooth_bufs": {name: deque(maxlen=SMOOTH_BUF_SIZE) for name in VIDEO_PATHS},
        "calib_bufs": {name: [] for name in VIDEO_PATHS},
        "peaks": {name: 0 for name in VIDEO_PATHS},
        "display_vals": {name: 0 for name in VIDEO_PATHS},
        "calibrated": {name: False for name in VIDEO_PATHS},
        "frame_counters": {name: 0 for name in VIDEO_PATHS},
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

def reset_state():
    st.session_state.smooth_bufs    = {name: deque(maxlen=SMOOTH_BUF_SIZE) for name in VIDEO_PATHS}
    st.session_state.calib_bufs     = {name: [] for name in VIDEO_PATHS}
    st.session_state.peaks          = {name: 0 for name in VIDEO_PATHS}
    st.session_state.display_vals   = {name: 0 for name in VIDEO_PATHS}
    st.session_state.calibrated     = {name: False for name in VIDEO_PATHS}
    st.session_state.frame_counters = {name: 0 for name in VIDEO_PATHS}

# ─────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Share+Tech+Mono&display=swap');

.stApp { background: #0a0e1a; font-family: 'Share Tech Mono', monospace; }
.main-title {
    text-align: center; font-family: 'Orbitron', sans-serif;
    font-size: 26px; font-weight: 900; color: #00d4ff;
    letter-spacing: 4px; text-shadow: 0 0 20px rgba(0,212,255,0.5);
    margin-bottom: 4px;
}
.sub-title {
    text-align: center; color: #4a6fa5;
    font-size: 11px; letter-spacing: 5px; margin-bottom: 20px;
}
.legend-bar {
    display: flex; justify-content: center; gap: 24px;
    margin-bottom: 16px; padding: 10px;
    background: #0d1422; border: 1px solid #1a2a4a; border-radius: 8px;
}
.legend-item { display: flex; align-items: center; gap: 8px; font-size: 11px; color: #8899aa; }
.legend-dot  { width: 13px; height: 13px; border-radius: 3px; flex-shrink: 0; }
.calib-bar {
    text-align: center; padding: 8px; margin-bottom: 12px;
    background: #0d1a2a; border: 1px solid #1a3a5a;
    border-radius: 8px; color: #00d4ff; font-size: 11px; letter-spacing: 2px;
}
.wagon-grid {
    display: flex; justify-content: center;
    gap: 16px; flex-wrap: nowrap; padding: 10px 0 20px 0;
}
.wagon-card {
    display: flex; flex-direction: column;
    align-items: center; width: 140px; flex-shrink: 0;
}
.wagon-label {
    font-family: 'Orbitron', sans-serif; font-size: 10px;
    color: #4a6fa5; letter-spacing: 2px; margin-bottom: 8px; text-align: center;
}
.wagon-body {
    width: 130px; height: 200px; border-radius: 16px 16px 8px 8px;
    padding: 16px 10px 12px 10px; display: flex; flex-direction: column;
    align-items: center; justify-content: space-between;
    border: 2px solid rgba(255,255,255,0.06); box-sizing: border-box;
}
.wagon-body.green  { background:linear-gradient(160deg,#1a3a1a,#0d1f0d); border-color:#2ecc40; box-shadow:0 0 20px rgba(46,204,64,.3); }
.wagon-body.orange { background:linear-gradient(160deg,#3a2a0a,#1f1505); border-color:#ff851b; box-shadow:0 0 20px rgba(255,133,27,.3); }
.wagon-body.red    { background:linear-gradient(160deg,#3a0a0a,#1f0505); border-color:#ff4136; box-shadow:0 0 20px rgba(255,65,54,.4); }
.wagon-body.calib  { background:linear-gradient(160deg,#0a1a2a,#050d15); border-color:#00d4ff; box-shadow:0 0 20px rgba(0,212,255,.2); }
.person-icons {
    display: flex; flex-wrap: wrap; justify-content: center;
    align-content: flex-start; gap: 3px; width: 100%; min-height: 80px;
}
.person-icon { font-size: 17px; line-height: 1; }
.wagon-count { font-family: 'Orbitron', sans-serif; font-size: 32px; font-weight: 900; line-height: 1; }
.wagon-count.green  { color:#2ecc40; text-shadow:0 0 12px rgba(46,204,64,.7); }
.wagon-count.orange { color:#ff851b; text-shadow:0 0 12px rgba(255,133,27,.7); }
.wagon-count.red    { color:#ff4136; text-shadow:0 0 12px rgba(255,65,54,.7); }
.wagon-count.calib  { color:#00d4ff; text-shadow:0 0 12px rgba(0,212,255,.7); }
.wagon-status {
    font-size: 9px; letter-spacing: 1.5px; text-transform: uppercase;
    padding: 3px 10px; border-radius: 4px; font-weight: bold; white-space: nowrap;
}
.wagon-status.green  { color:#2ecc40; background:rgba(46,204,64,.15); }
.wagon-status.orange { color:#ff851b; background:rgba(255,133,27,.15); }
.wagon-status.red    { color:#ff4136; background:rgba(255,65,54,.15); }
.wagon-status.calib  { color:#00d4ff; background:rgba(0,212,255,.15); }
.wagon-bottom { width:118px; height:10px; background:#1a2a4a; border-radius:0 0 8px 8px; margin-top:-2px; }
.wagon-number { font-family:'Orbitron',sans-serif; font-size:20px; font-weight:700; color:#4a6fa5; margin-top:10px; }
.status-bar {
    text-align:center; margin-top:16px; padding:10px;
    background:#0d1422; border:1px solid #1a2a4a;
    border-radius:8px; color:#4a6fa5; font-size:11px; letter-spacing:2px;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
st.markdown('<p class="main-title"> BAKI METROSU</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Vaqon Yükləmə Monitorinqi · Real-time</p>', unsafe_allow_html=True)
st.markdown("""
<div class="legend-bar">
  <div class="legend-item"><div class="legend-dot" style="background:#2ecc40"></div>0–10 · NORMAL</div>
  <div class="legend-item"><div class="legend-dot" style="background:#ff851b"></div>11–20 · SIX</div>
  <div class="legend-item"><div class="legend-dot" style="background:#ff4136"></div>&gt;20 · COX SIX</div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Buttons
# ─────────────────────────────────────────────
c1, c2, c3 = st.columns([1, 2, 1])
with c2:
    if st.button("▶  Monitorinqi Başlat", use_container_width=True, type="primary"):
        reset_state()
        st.session_state.running = True
    if st.button("⏹  Dayandır", use_container_width=True):
        st.session_state.running = False

calib_ph  = st.empty()
wagon_ph  = st.empty()
status_ph = st.empty()

# ─────────────────────────────────────────────
# Render — KALİBRASİYA yazısı yoxdur
# ─────────────────────────────────────────────
def render_wagons(results, calibrated_map):
    cc_map = ["green", "orange", "red"]

    def make_icons(count, cc):
        c = {"green":"#2ecc40","orange":"#ff851b","red":"#ff4136","calib":"#00d4ff"}[cc]
        n = min(count, 12)
        html = "".join(f'<span class="person-icon" style="color:{c}">👤</span>' for _ in range(n))
        if count > 12:
            html += f'<span style="color:{c};font-size:11px;font-weight:bold">+{count-12}</span>'
        return html

    html = '<div class="wagon-grid">'
    for i, (name, d) in enumerate(results.items(), 1):
        is_calib = not calibrated_map.get(name, False)
        cc       = "calib" if is_calib else cc_map[d["level"]]

        # FIX 2: KALİBRASİYA yazısı yoxdur — həmişə normal label
        status_txt = d["label"]

        html += f"""
        <div class="wagon-card">
          <div class="wagon-label">{name.upper()}</div>
          <div class="wagon-body {cc}">
            <div class="person-icons">{make_icons(d["count"], cc)}</div>
            <div class="wagon-count {cc}">{d["count"]}</div>
            <div class="wagon-status {cc}">{status_txt}</div>
          </div>
          <div class="wagon-bottom"></div>
          <div class="wagon-number">{i}</div>
        </div>"""
    html += "</div>"
    return html

wagon_ph.markdown(render_wagons(st.session_state.results, st.session_state.calibrated), unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Smart Peak-Hold — frame_counters burada artmır
# ─────────────────────────────────────────────
def smart_count(name, raw):
    fc = st.session_state.frame_counters[name]

    st.session_state.smooth_bufs[name].append(raw)
    smoothed = round(sum(st.session_state.smooth_bufs[name]) /
                     len(st.session_state.smooth_bufs[name]))

    if fc < CALIBRATION_FRAMES:
        st.session_state.calib_bufs[name].append(smoothed)
        # FIX 3: kalibrasiya zamanı ədədi dəyişdirmə — 0 göstər
        result = 0
    else:
        if not st.session_state.calibrated[name]:
            peak = max(st.session_state.calib_bufs[name]) if st.session_state.calib_bufs[name] else smoothed
            st.session_state.peaks[name]        = peak
            st.session_state.display_vals[name] = peak
            st.session_state.calibrated[name]   = True

        current = st.session_state.display_vals[name]
        delta   = smoothed - current

        if delta >= CHANGE_THRESHOLD:
            st.session_state.display_vals[name] = smoothed
            st.session_state.peaks[name]        = smoothed
        elif delta <= -CHANGE_THRESHOLD:
            st.session_state.display_vals[name] = max(0, current - 1)

        result = st.session_state.display_vals[name]

    return result

# ─────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────
if st.session_state.running:

    if st.session_state.yolo_model is None:
        with st.spinner("YOLOv8 modeli yüklənir..."):
            st.session_state.yolo_model = YOLO("yolov8m.pt")
    model = st.session_state.yolo_model

    caps     = {}
    trackers = {}
    zones    = {}

    for name, path in VIDEO_PATHS.items():
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            st.warning(f"⚠ Aça bilmədim: {path}")
            continue
        caps[name]     = cap
        trackers[name] = sv.ByteTrack(lost_track_buffer=60, frame_rate=30)
        zones[name]    = sv.PolygonZone(polygon=VAGON_POLY)

    global_frame = 0

    while st.session_state.running:
        for name, cap in caps.items():
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
            if not ret:
                continue

            # FIX 1: frame_counters ƏVVƏLCƏ artır
            st.session_state.frame_counters[name] += 1

            # Sonra yoxla — bu frame-i işləyəkmi?
            if st.session_state.frame_counters[name] % 3 != 0:
                continue

            res  = model(frame, classes=[0], conf=0.25, iou=0.45, imgsz=640, verbose=False)[0]
            dets = sv.Detections.from_ultralytics(res)
            dets = trackers[name].update_with_detections(dets)
            mask = zones[name].trigger(detections=dets)
            raw  = len(dets[mask])

            count = smart_count(name, raw)
            label, color, level = sixliq(count)

            st.session_state.results[name] = {
                "count": count, "level": level,
                "color": color, "label": label,
            }

        # Kalibrasiya status barı
        done_count = sum(1 for v in st.session_state.calibrated.values() if v)
        total_v    = len(VIDEO_PATHS)
        if done_count < total_v:
            calib_ph.markdown(
                f'<div class="calib-bar"> KALİBRASİYA: {done_count}/{total_v} · {CALIBRATION_SEC} saniyə gözləyin...</div>',
                unsafe_allow_html=True
            )
        else:
            calib_ph.empty()  # FIX 4: kalibrasiya bitəndə bar tamamilə yox olur

        wagon_ph.markdown(
            render_wagons(st.session_state.results, st.session_state.calibrated),
            unsafe_allow_html=True
        )

        total = sum(v["count"] for v in st.session_state.results.values())
        status_ph.markdown(
            f'<div class="status-bar">● Frame #{global_frame} · Ümumi: {total} sərnişin aktiv</div>',
            unsafe_allow_html=True
        )

        global_frame += 1
        time.sleep(0.05)

    for cap in caps.values():
        cap.release()

    status_ph.markdown('<div class="status-bar">⏹ Monitorinq dayandırıldı</div>', unsafe_allow_html=True)