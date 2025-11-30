#
# pothole_detector.py
#
# YOLO-based pothole detection + MiDaS depth-based severity estimation.
#

import cv2
import math
import cvzone
from ultralytics import YOLO

from depth_model import predict_depth
from depth_utils import estimate_pothole_depth_relative

# --------- Configuration ---------

# Input video
video_path = "Media/Potholes.mp4"
cap = cv2.VideoCapture(video_path)

# YOLO model with custom weights
model = YOLO(r"c:/Users/amand/Desktop/CMPT 742/Project Potholes/CMPT742-Final-Project---Pothole-Detection/runs/detect/train12/weights/best.pt")

# Single class for this project
classNames = ["Pothole"]

CONF_THRESH = 0.4       # confidence threshold for detections
ROAD_MARGIN = 15        # pixels around bbox to estimate road plane


# --------- Main loop ---------

while True:
    success, img = cap.read()
    if not success:
        break

    # Compute depth map for current frame (MiDaS)
    depth_map = predict_depth(img)  # (H, W)

    # Run YOLO on the same frame
    results = model(img, stream=True)

    for r in results:
        boxes = r.boxes

        for box in boxes:
            # Bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            w, h = x2 - x1, y2 - y1

            # Confidence and class
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])

            if conf > CONF_THRESH:
                # ----- Depth-based severity estimation -----
                depth_metrics = estimate_pothole_depth_relative(
                    depth_map,
                    (x1, y1, x2, y2),
                    road_margin=ROAD_MARGIN,
                )

                max_rel = depth_metrics["max_rel"]
                mean_rel = depth_metrics["mean_rel"]
                road_depth = depth_metrics["road_depth"]

                #debug print
                print(f"max_rel = {max_rel:.6f}, mean_rel = {mean_rel:.6f}")

                # NEW: metric depths (cm) from depth_utils.py
                max_depth_cm = depth_metrics.get("max_depth_cm", 0.0)
                mean_depth_cm = depth_metrics.get("mean_depth_cm", 0.0)

                # Build label text
                label = f"{classNames[cls]} {conf:.2f}"

                # Only append depth if we successfully estimated road_depth
                if road_depth is not None:
                    # Show approx. max depth in centimeters
                    label += f" | ~{max_depth_cm:.1f} cm"

                # Draw bbox and label
                cvzone.cornerRect(img, (x1, y1, w, h), t=2)
                cvzone.putTextRect(
                    img,
                    label,
                    (max(0, x1), max(35, y1)),
                    scale=1,
                    thickness=1,
                )

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
