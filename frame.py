import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv

VIDEO_PATH = r"C:\Users\Lenovo\Downloads\В4-КАМ6_60042_16-04-26_18-00-00.avi"

video_info = sv.VideoInfo.from_video_path(VIDEO_PATH)
W, H = video_info.resolution_wh

model   = YOLO("yolov8m.pt")
tracker = sv.ByteTrack(lost_track_buffer=60, frame_rate=30)

# ── Yalnız vaqon zone ──────────────────────────────────
VAGON_POLY = np.array([
    [160,   20],
    [1120,  20],
    [1280, 720],
    [0,    720],
])

zone      = sv.PolygonZone(polygon=VAGON_POLY)
zone_ann  = sv.PolygonZoneAnnotator(zone=zone, color=sv.Color.YELLOW, thickness=2)
box_ann   = sv.BoxAnnotator(thickness=2)
label_ann = sv.LabelAnnotator(text_scale=0.45)

def sixliq(n):
    if n <= 15:  return "NORMAL",    (0, 200, 0)
    if n <= 20:  return "SIX",     (0, 165, 255)
    return           "COX SIX",    (0, 0, 220)

cap = cv2.VideoCapture(VIDEO_PATH)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results    = model(frame, classes=[0], conf=0.25,
                       iou=0.45, imgsz=640, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results)
    detections = tracker.update_with_detections(detections)

    # Yalnız zone içindəkilər
    mask       = zone.trigger(detections=detections)
    det_zone   = detections[mask]

    anliq      = len(det_zone)
    signal, color = sixliq(anliq)

    labels     = [f"#{tid}" for tid in det_zone.tracker_id] \
                 if det_zone.tracker_id is not None else []

    annotated  = box_ann.annotate(frame.copy(), det_zone)
    annotated  = label_ann.annotate(annotated, det_zone, labels=labels)
    annotated  = zone_ann.annotate(annotated)

    # Dashboard
    overlay = annotated.copy()
    cv2.rectangle(overlay, (10, 10), (350, 130), (0,0,0), -1)
    cv2.addWeighted(overlay, 0.5, annotated, 0.5, 0, annotated)

    cv2.putText(annotated, f"Vaqonda adam: {anliq}",
                (20, 60),  cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0,255,0), 3)
    cv2.putText(annotated, signal,
                (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)

    cv2.imshow("Baki Metrosu - PlatformAI", annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()