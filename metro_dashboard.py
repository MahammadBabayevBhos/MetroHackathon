from collections import deque
import os
import time
from typing import Dict, Tuple

import cv2
import numpy as np
import streamlit as st
import supervision as sv
from ultralytics import YOLO

from config import (
    CALIBRATION_SEC,
    CHANGE_THRESHOLD,
    DEFAULT_CONFIDENCE,
    DEFAULT_IMGSZ,
    DEFAULT_IOU,
    DEFAULT_MODEL_NANO,
    DEFAULT_WAGON_VIDEOS,
    FPS_DEFAULT,
    SMOOTH_BUF_SIZE,
    THRESHOLD_CROWDED_MAX,
    THRESHOLD_NORMAL_MAX,
    VAGON_POLY,
)

st.set_page_config(page_title="Baki Metrosu: Vaqon Sixligi Monitorinqi", layout="wide")

def evaluate_density(n: int) -> Tuple[str, str, int]:
    if n <= THRESHOLD_NORMAL_MAX:
        return "NORMAL", "#2ecc40", 0
    elif n <= THRESHOLD_CROWDED_MAX:
        return "SIX", "#ff851b", 1
    else:
        return "COX SIX", "#ff4136", 2


def init_state(video_sources: Dict[str, str]):
    defaults = {
        "running": False,
        "results": {name: {"count": 0, "level": 0, "color": "#2ecc40", "label": "NORMAL"} for name in video_sources},
        "yolo_model": None,
        "smooth_bufs": {name: deque(maxlen=SMOOTH_BUF_SIZE) for name in video_sources},
        "calib_bufs": {name: [] for name in video_sources},
        "peaks": {name: 0 for name in video_sources},
        "display_vals": {name: 0 for name in video_sources},
        "calibrated": {name: False for name in video_sources},
        "frame_counters": {name: 0 for name in video_sources},
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def reset_state(video_sources: Dict[str, str]):
    st.session_state.smooth_bufs = {name: deque(maxlen=SMOOTH_BUF_SIZE) for name in video_sources}
    st.session_state.calib_bufs = {name: [] for name in video_sources}
    st.session_state.peaks = {name: 0 for name in video_sources}
    st.session_state.display_vals = {name: 0 for name in video_sources}
    st.session_state.calibrated = {name: False for name in video_sources}
    st.session_state.frame_counters = {name: 0 for name in video_sources}


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
.person-indicator { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin: 1px; }
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

st.sidebar.header("Sistem Parametrleri")
model_choice = st.sidebar.selectbox("Model:", [DEFAULT_MODEL_NANO, "yolov8s.pt", "yolov8m.pt"], index=0)
conf_threshold = st.sidebar.slider("Confidence Heddi:", 0.1, 0.9, DEFAULT_CONFIDENCE, 0.05)

video_sources = {}
st.sidebar.subheader("Video Menbeleri:")
for name, default_path in DEFAULT_WAGON_VIDEOS.items():
    custom_path = st.sidebar.text_input(f"{name} Yolu:", value=default_path)
    video_sources[name] = custom_path

init_state(video_sources)

st.markdown('<p class="main-title">BAKI METROSU</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Vaqon Yukleme Monitorinqi : Real-Time Analiz</p>', unsafe_allow_html=True)
st.markdown(f"""
<div class="legend-bar">
  <div class="legend-item"><div class="legend-dot" style="background:#2ecc40"></div>0:{THRESHOLD_NORMAL_MAX} : NORMAL</div>
  <div class="legend-item"><div class="legend-dot" style="background:#ff851b"></div>{THRESHOLD_NORMAL_MAX + 1}:{THRESHOLD_CROWDED_MAX} : SIX</div>
  <div class="legend-item"><div class="legend-dot" style="background:#ff4136"></div>>{THRESHOLD_CROWDED_MAX} : COX SIX</div>
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if st.button("Monitorinqi Baslat", use_container_width=True, type="primary"):
        reset_state(video_sources)
        st.session_state.running = True
    if st.button("Dayandir", use_container_width=True):
        st.session_state.running = False

calib_ph = st.empty()
wagon_ph = st.empty()
status_ph = st.empty()


def render_wagons(results, calibrated_map) -> str:
    cc_map = ["green", "orange", "red"]

    def make_indicators(count: int, color_hex: str) -> str:
        num = min(count, 12)
        html = "".join(f'<span class="person-indicator" style="background:{color_hex}"></span>' for _ in range(num))
        if count > 12:
            html += f'<span style="color:{color_hex};font-size:11px;font-weight:bold">+{count-12}</span>'
        return html

    html = '<div class="wagon-grid">'
    for i, (name, d) in enumerate(results.items(), 1):
        is_calib = not calibrated_map.get(name, False)
        cc = "calib" if is_calib else cc_map[d["level"]]
        color_hex = {"green": "#2ecc40", "orange": "#ff851b", "red": "#ff4136", "calib": "#00d4ff"}[cc]
        status_txt = d["label"]

        html += f"""
        <div class="wagon-card">
          <div class="wagon-label">{name.upper()}</div>
          <div class="wagon-body {cc}">
            <div class="person-icons">{make_indicators(d['count'], color_hex)}</div>
            <div class="wagon-count {cc}">{d['count']}</div>
            <div class="wagon-status {cc}">{status_txt}</div>
          </div>
          <div class="wagon-bottom"></div>
          <div class="wagon-number">{i}</div>
        </div>"""
    html += "</div>"
    return html


wagon_ph.markdown(render_wagons(st.session_state.results, st.session_state.calibrated), unsafe_allow_html=True)


def smart_count(name: str, raw: int, calib_frames: int) -> int:
    fc = st.session_state.frame_counters[name]
    st.session_state.smooth_bufs[name].append(raw)
    smoothed = round(sum(st.session_state.smooth_bufs[name]) / len(st.session_state.smooth_bufs[name]))

    if fc < calib_frames:
        st.session_state.calib_bufs[name].append(smoothed)
        return 0
    else:
        if not st.session_state.calibrated[name]:
            peak = max(st.session_state.calib_bufs[name]) if st.session_state.calib_bufs[name] else smoothed
            st.session_state.peaks[name] = peak
            st.session_state.display_vals[name] = peak
            st.session_state.calibrated[name] = True

        current = st.session_state.display_vals[name]
        delta = smoothed - current

        if delta >= CHANGE_THRESHOLD:
            st.session_state.display_vals[name] = smoothed
            st.session_state.peaks[name] = smoothed
        elif delta <= -CHANGE_THRESHOLD:
            st.session_state.display_vals[name] = max(0, current - 1)

        return st.session_state.display_vals[name]


if st.session_state.running:
    if st.session_state.yolo_model is None:
        with st.spinner("YOLO modeli yuklenir..."):
            st.session_state.yolo_model = YOLO(model_choice)
    model = st.session_state.yolo_model

    caps = {}
    trackers = {}
    zones = {}

    for name, path in video_sources.items():
        if os.path.exists(path):
            cap = cv2.VideoCapture(path)
            if cap.isOpened():
                caps[name] = cap
                trackers[name] = sv.ByteTrack(lost_track_buffer=60, frame_rate=FPS_DEFAULT)
                zones[name] = sv.PolygonZone(polygon=VAGON_POLY)

    if not caps:
        st.warning("Secilmis video fayllari tapilmadi. Zehmet olmasa yan panelden duzgun video yollarini qeyd edin.")
        st.session_state.running = False
    else:
        calib_frames_total = (FPS_DEFAULT * CALIBRATION_SEC) // 3
        global_frame = 0

        try:
            while st.session_state.running:
                for name, cap in caps.items():
                    ret, frame = cap.read()
                    if not ret:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ret, frame = cap.read()
                    if not ret:
                        continue

                    st.session_state.frame_counters[name] += 1
                    if st.session_state.frame_counters[name] % 3 != 0:
                        continue

                    res = model(
                        frame,
                        classes=[0],
                        conf=conf_threshold,
                        iou=DEFAULT_IOU,
                        imgsz=DEFAULT_IMGSZ,
                        verbose=False
                    )[0]
                    dets = sv.Detections.from_ultralytics(res)
                    dets = trackers[name].update_with_detections(dets)
                    mask = zones[name].trigger(detections=dets)
                    raw = len(dets[mask])

                    count = smart_count(name, raw, calib_frames_total)
                    label, color, level = evaluate_density(count)

                    st.session_state.results[name] = {
                        "count": count,
                        "level": level,
                        "color": color,
                        "label": label,
                    }

                done_count = sum(1 for v in st.session_state.calibrated.values() if v)
                total_v = len(caps)
                if done_count < total_v:
                    calib_ph.markdown(
                        f'<div class="calib-bar">KALIBRASIYA: {done_count}/{total_v} : {CALIBRATION_SEC} saniye gozleyin...</div>',
                        unsafe_allow_html=True
                    )
                else:
                    calib_ph.empty()

                wagon_ph.markdown(
                    render_wagons(st.session_state.results, st.session_state.calibrated),
                    unsafe_allow_html=True
                )

                total_p = sum(v["count"] for v in st.session_state.results.values())
                status_ph.markdown(
                    f'<div class="status-bar">Kadr #{global_frame} : Umumi: {total_p} sernisin aktiv</div>',
                    unsafe_allow_html=True
                )

                global_frame += 1
                time.sleep(0.04)
        finally:
            for cap in caps.values():
                cap.release()
            status_ph.markdown('<div class="status-bar">Monitorinq dayandirildi</div>', unsafe_allow_html=True)
