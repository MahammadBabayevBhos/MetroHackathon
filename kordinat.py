import argparse
import os
import cv2
import numpy as np


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Polygon ve ROI Koordinat Secici")
    parser.add_argument("--video", type=str, required=True, help="Video faylinin yolu")
    return parser.parse_args()


points = []


def click_event(event, x, y, flags, param):
    global points, img_show
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        cv2.circle(img_show, (x, y), 5, (0, 255, 0), -1)
        cv2.putText(img_show, f"{len(points)}: ({x},{y})", (x + 8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        if len(points) > 1:
            cv2.line(img_show, points[-2], points[-1], (0, 255, 0), 2)
        cv2.imshow("Koordinat Secici (Sol klik: noqte, Q: cix)", img_show)
        print(f"Noqte {len(points)}: [{x}, {y}]")


def main():
    global img_show
    args = parse_arguments()
    if not os.path.exists(args.video):
        print(f"Xeta: Video fayli tapilmadi: {args.video}")
        return

    cap = cv2.VideoCapture(args.video)
    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        print("Xeta: Video faylindan kadr oxuna bilmedi.")
        return

    img_show = frame.copy()
    cv2.imshow("Koordinat Secici (Sol klik: noqte, Q: cix)", img_show)
    cv2.setMouseCallback("Koordinat Secici (Sol klik: noqte, Q: cix)", click_event)

    print("Gosteris: Sol klik ile noqteleri secin, bitirdikde Q duymesini sixin.")

    while True:
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()

    print("\nSecilmis koordinatlar:")
    print("np.array([")
    for p in points:
        print(f"    [{p[0]}, {p[1]}],")
    print("], dtype=np.int32)")


if __name__ == "__main__":
    main()
