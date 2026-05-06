import cv2
import supervision as sv

VIDEO_PATH = r"C:\Users\Lenovo\Downloads\Cixanlar_3.mp4"

points = []

def click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        print(f"Noqte {len(points)}: [{x}, {y}]")
        cv2.circle(frame_show, (x, y), 6, (0, 255, 0), -1)
        cv2.putText(frame_show, f"{len(points)}:({x},{y})", (x+8, y-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.imshow("Koordinat sec - Sol klik, Q ile cix", frame_show)

cap = cv2.VideoCapture(VIDEO_PATH)
ret, frame = cap.read()
cap.release()

frame_show = frame.copy()
cv2.imshow("Koordinat sec - Sol klik, Q ile cix", frame_show)
cv2.setMouseCallback("Koordinat sec - Sol klik, Q ile cix", click)

print("Sol klik ile xettin iki noqtesini sec (yuxari ve asagi)")
print("─" * 40)

while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
print("─" * 40)
print("Secilen noqteler:")
for i, p in enumerate(points):
    print(f"  Noqte {i+1}: {p}")