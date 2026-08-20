import os
import numpy as np

# Model konfiqurasiyasi
DEFAULT_MODEL_NANO = "yolov8n.pt"
DEFAULT_MODEL_MEDIUM = "yolov8m.pt"
DEFAULT_CONFIDENCE = 0.25
DEFAULT_IOU = 0.45
DEFAULT_IMGSZ = 640

# Vaqon ucun perspektiv zonasi (ROI)
VAGON_POLY = np.array([
    [160, 20],
    [1120, 20],
    [1280, 720],
    [0, 720],
], dtype=np.int32)

# Cixis xetti koordinatlari (LineZone)
LINE_START_POINT = (393, 50)
LINE_END_POINT = (415, 475)

# Sixliq hedleri
THRESHOLD_NORMAL_MAX = 12
THRESHOLD_CROWDED_MAX = 20

# Kalibrasiya ve hamarlama
CALIBRATION_SEC = 10
FPS_DEFAULT = 30
SMOOTH_BUF_SIZE = 15
CHANGE_THRESHOLD = 3

# Default video yollari
DEFAULT_EXIT_VIDEO = os.getenv("METRO_EXIT_VIDEO", "data/exit_sample.mp4")
DEFAULT_WAGON_VIDEOS = {
    f"Vaqon {i}": os.getenv(f"METRO_WAGON_{i}_VIDEO", f"data/vagon_{i}.avi")
    for i in range(1, 6)
}
