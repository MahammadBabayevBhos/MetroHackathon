import argparse
import os
import sys
from collections import defaultdict, deque
from typing import Set

import cv2
import supervision as sv
from ultralytics import YOLO

from config import (
    DEFAULT_CONFIDENCE,
    DEFAULT_EXIT_VIDEO,
    DEFAULT_IMGSZ,
    DEFAULT_IOU,
    DEFAULT_MODEL_NANO,
    LINE_END_POINT,
    LINE_START_POINT,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Baki Metrosu: Agilli Cixis Saygaci")
    parser.add_argument("--video", type=str, default=DEFAULT_EXIT_VIDEO, help="Giris video faylinin yolu")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_NANO, help="YOLO model cekisi")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONFIDENCE, help="Confidence heddi")
    parser.add_argument("--iou", type=float, default=DEFAULT_IOU, help="IoU heddi")
    parser.add_argument("--hide-view", action="store_true", help="OpenCV goruntusunu gizletmek ucun")
    return parser.parse_args()


def run_exit_counter(video_path: str, model_path: str, conf: float, iou: float, show_view: bool = True) -> int:
    if not os.path.exists(video_path):
        print(f"Xeta: Video fayli tapilmadi: {video_path}")
        print("Gosteris: --video parametri ile movcud video faylin yolunu qeyd edin.")
        return 0

    video_info = sv.VideoInfo.from_video_path(video_path)
    model = YOLO(model_path)
    tracker = sv.ByteTrack(lost_track_buffer=90, frame_rate=int(video_info.fps) if video_info.fps > 0 else 30)

    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_scale=0.45)

    line_start = sv.Point(LINE_START_POINT[0], LINE_START_POINT[1])
    line_end = sv.Point(LINE_END_POINT[0], LINE_END_POINT[1])
    line_zone = sv.LineZone(start=line_start, end=line_end)
    line_annotator = sv.LineZoneAnnotator(thickness=3, text_thickness=2, text_scale=0.7)

    blacklist_ids: Set[int] = set()
    cixan_sayi = 0
    cx_history = defaultdict(lambda: deque(maxlen=10))

    cap = cv2.VideoCapture(video_path)

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(
                frame,
                classes=[0],
                conf=conf,
                iou=iou,
                imgsz=DEFAULT_IMGSZ,
                verbose=False
            )[0]

            detections = sv.Detections.from_ultralytics(results)
            detections = tracker.update_with_detections(detections)

            crossed_in, crossed_out = line_zone.trigger(detections=detections)

            if detections.tracker_id is not None:
                for i, tid in enumerate(detections.tracker_id):
                    x1, _, x2, _ = detections.xyxy[i]
                    cx = int((x1 + x2) / 2)
                    cx_history[tid].append(cx)

                # Soldan saga kecid (Odenis zonasina daxil olma : Blacklist)
                for i, is_crossed in enumerate(crossed_in):
                    if is_crossed and detections.tracker_id is not None:
                        tid = int(detections.tracker_id[i])
                        blacklist_ids.add(tid)

                # Sagdan sola kecid (Metrodan cixis)
                for i, is_crossed in enumerate(crossed_out):
                    if is_crossed and detections.tracker_id is not None:
                        tid = int(detections.tracker_id[i])
                        if tid in blacklist_ids:
                            blacklist_ids.discard(tid)
                        else:
                            cixan_sayi += 1

            if show_view:
                labels = []
                if detections.tracker_id is not None:
                    for tid in detections.tracker_id:
                        tag = "[OD]" if int(tid) in blacklist_ids else "[CX]"
                        labels.append(f"{tag}#{tid}")

                annotated = box_annotator.annotate(frame.copy(), detections)
                annotated = label_annotator.annotate(annotated, detections, labels=labels)
                annotated = line_annotator.annotate(annotated, line_counter=line_zone)

                overlay = annotated.copy()
                cv2.rectangle(overlay, (10, 10), (420, 110), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.55, annotated, 0.45, 0, annotated)

                cv2.putText(
                    annotated,
                    f"Metro cixan: {cixan_sayi}",
                    (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.3,
                    (0, 255, 100),
                    3
                )
                cv2.putText(
                    annotated,
                    f"Blacklist: {len(blacklist_ids)} nefer",
                    (20, 95),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 100, 255),
                    2
                )

                cv2.imshow("Baki Metrosu: Cixis Saygaci", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        if show_view:
            cv2.destroyAllWindows()

    print(f"Netice: Metro cixan sernisin sayi: {cixan_sayi}")
    return cixan_sayi


if __name__ == "__main__":
    args = parse_arguments()
    run_exit_counter(
        video_path=args.video,
        model_path=args.model,
        conf=args.conf,
        iou=args.iou,
        show_view=not args.hide_view
    )
