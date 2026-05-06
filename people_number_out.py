import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
from collections import defaultdict, deque

VIDEO_PATH = r"C:\Users\Lenovo\Downloads\Cixanlar_3.mp4"

video_info = sv.VideoInfo.from_video_path(VIDEO_PATH)
W, H = video_info.resolution_wh

model   = YOLO("yolov8n.pt")
tracker = sv.ByteTrack(lost_track_buffer=90, frame_rate=30)

box_ann   = sv.BoxAnnotator(thickness=2)
label_ann = sv.LabelAnnotator(text_scale=0.45)

#  Şaquli xətt
LINE_START = sv.Point(393, 50)
LINE_END   = sv.Point(415, 475)

line     = sv.LineZone(start=LINE_START, end=LINE_END)
line_ann = sv.LineZoneAnnotator(thickness=3, text_thickness=2, text_scale=0.7)

# ── Blacklist məntiqi
# Soldan sağa keçən ID → ödəniş üçün girdi → blacklist
# Sağdan sola keçən ID → əgər blacklist-dədirsə SAYMA, yoxdursa SAY

blacklist_ids = set()  # ödəniş üçün girənlər
cixan_sayi   = 0

# Hər ID-nin əvvəlki X mərkəzi
cx_history = defaultdict(lambda: deque(maxlen=5))

cap = cv2.VideoCapture(VIDEO_PATH)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results    = model(frame, classes=[0], conf=0.3,
                       iou=0.45, imgsz=640, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results)
    detections = tracker.update_with_detections(detections)

    # LineZone keçidləri izlə
    crossed_in, crossed_out = line.trigger(detections=detections)

    if detections.tracker_id is not None:
        for i, tid in enumerate(detections.tracker_id):
            x1, y1, x2, y2 = detections.xyxy[i]
            cx = int((x1 + x2) / 2)
            cx_history[tid].append(cx)

        # crossed_in → soldan sağa (ödəniş üçün girdi) → blacklist
        for i, is_crossed in enumerate(crossed_in):
            if is_crossed and detections.tracker_id is not None:
                tid = detections.tracker_id[i]
                blacklist_ids.add(tid)

        # crossed_out → sağdan sola (çıxır)
        for i, is_crossed in enumerate(crossed_out):
            if is_crossed and detections.tracker_id is not None:
                tid = detections.tracker_id[i]
                if tid in blacklist_ids:
                    # Ödəniş edib qayıdan → SAYMA
                    blacklist_ids.discard(tid)
                else:
                    # Metro çıxan → SAY
                    cixan_sayi += 1

    # ── Vizuallaşdırma ────────────────────────────────
    labels = []
    if detections.tracker_id is not None:
        for tid in detections.tracker_id:
            tag = "[OD]" if tid in blacklist_ids else "[CX]"
            labels.append(f"{tag}#{tid}")

    annotated = box_ann.annotate(frame.copy(), detections)
    annotated = label_ann.annotate(annotated, detections, labels=labels)
    annotated = line_ann.annotate(annotated, line_counter=line)

    # Dashboard
    overlay = annotated.copy()
    cv2.rectangle(overlay, (10, 10), (420, 110), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, annotated, 0.45, 0, annotated)

    cv2.putText(annotated, f"Metro cixan: {cixan_sayi}",
                (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 100), 3)
    cv2.putText(annotated, f"Blacklist: {len(blacklist_ids)} nefer",
                (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 255), 2)

    cv2.imshow("Baki Metrosu - Metro Cixis Sayaci", annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"\nYekun — Metro cixan: {cixan_sayi}")