import cv2
import numpy as np

VIDEO_PATH = r"C:\Users\Lenovo\Downloads\В4-КАМ6_60042_16-04-26_18-00-00.avi"

points = []

def click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        cv2.circle(img_show, (x, y), 5, (0, 255, 0), -1)
        cv2.putText(img_show, f"{len(points)}: ({x},{y})",
                    (x+8, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 1)
        if len(points) > 1:
            cv2.line(img_show, points[-2], points[-1], (0, 255, 0), 2)
        cv2.imshow("Koordinat sec - Sol klik, Q ile cix", img_show)
        print(f"Nokta {len(points)}: [{x}, {y}]")

# Videonun ilk frame-ini götür
cap = cv2.VideoCapture(VIDEO_PATH)
ret, frame = cap.read()
cap.release()

img_show = frame.copy()
cv2.imshow("Koordinat sec - Sol klik, Q ile cix", img_show)
cv2.setMouseCallback("Koordinat sec - Sol klik, Q ile cix", click)

print("Sol klik ile polygon kunclerini sec, Q ile cix")
print("─" * 40)

while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

print("─" * 40)
print("VAGON_POLY = np.array([")
for p in points:
    print(f"    {list(p)},")
print("])")