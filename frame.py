import argparse
import os
import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

from config import (
    DEFAULT_CONFIDENCE,
    DEFAULT_IMGSZ,
    DEFAULT_IOU,
    DEFAULT_MODEL_MEDIUM,
    THRESHOLD_CROWDED_MAX,
    THRESHOLD_NORMAL_MAX,
    VAGON_POLY,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Baki Metrosu: Tek Vaqon Analizi")
    parser.add_argument("--video", type=str, default="data/vagon_sample.avi", help="Vaqon video faylinin yolu")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_MEDIUM, help="YOLO model cekisi")
    return parser.parse_args()


def evaluate_density(n: int):
    if n <= THRESHOLD_NORMAL_MAX:
        return "NORMAL", (0, 200, 0)
    if n <= THRESHOLD_CROWDED_MAX:
        return "SIX", (0, 165, 255)
    return "COX SIX", (0, 0, 220)


def main():
    args = parse_arguments()
    if not os.path.exists(args.video):
        print(f"Xeta: Video fayli tapilmadi: {args.video}")
        print("Gosteris: --video parametri ile video faylin yolunu qeyd edin.")
        return

    model = YOLO(args.model)
    tracker = sv.ByteTrack(lost_track_buffer=60, frame_rate=30)
    zone = sv.PolygonZone(polygon=VAGON_POLY)

    zone_ann = sv.PolygonZoneAnnotator(zone=zone, color=sv.Color.YELLOW, thickness=2)
    box_ann = sv.BoxAnnotator(thickness=2)
    label_ann = sv.LabelAnnotator(text_scale=0.45)

    cap = cv2.VideoCapture(args.video)

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(
                frame,
                classes=[0],
                conf=DEFAULT_CONFIDENCE,
                iou=DEFAULT_IOU,
                imgsz=DEFAULT_IMGSZ,
                verbose=False
            )[0]

            detections = sv.Detections.from_ultralytics(results)
            detections = tracker.update_with_detections(detections)

            mask = zone.trigger(detections=detections)
            det_zone = detections[mask]
            count = len(det_zone)
            signal, color = evaluate_density(count)

            labels = [f"#{tid}" for tid in det_zone.tracker_id] if det_zone.tracker_id is not None else []

            annotated = box_ann.annotate(frame.copy(), det_zone)
            annotated = label_ann.annotate(annotated, det_zone, labels=labels)
            annotated = zone_ann.annotate(annotated)

            overlay = annotated.copy()
            cv2.rectangle(overlay, (10, 10), (350, 130), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.5, annotated, 0.5, 0, annotated)

            cv2.putText(annotated, f"Vaqonda adam: {count}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 3)
            cv2.putText(annotated, signal, (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)

            cv2.imshow("Baki Metrosu: Vaqon Analizi", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
