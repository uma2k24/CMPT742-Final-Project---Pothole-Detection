#
# main.py 
#
# YOLO-based pothole detection + MiDaS depth-based severity estimation
# + side-by-side depth visualization.
#

import cv2
import math
import cvzone
from ultralytics import YOLO

from depth_model import predict_depth
from depth_utils import estimate_pothole_depth_relative

# NEW: import the visualization helper
from visualize import draw_results   # make sure visualize.py is in the same folder

# --------- Configuration ---------

# Input video
video_path = "CMPT742-Final-Project---Pothole-Detection/Media/Potholes.mp4"
cap = cv2.VideoCapture(video_path)

# YOLO model with custom weights
model = YOLO("Weights/best.pt")

# Single class for this project
classNames = ["Pothole"]

CONF_THRESH = 0.4       # confidence threshold for detections
ROAD_MARGIN = 15        # pixels around bbox to estimate road plane

# Toggle whether to show the depth image next to the RGB
SHOW_DEPTH = True


def classify_severity(max_depth_cm: float) -> str:
    """
    Simple heuristic to map estimated max depth (cm) to severity label.
    You can tweak these thresholds however you like.
    """
    if max_depth_cm < 2.0:
        return "shallow"
    elif max_depth_cm < 5.0:
        return "moderate"
    else:
        return "severe"


# --------- Main loop ---------

while True:
    success, img = cap.read()
    if not success:
        break

    # Compute depth map for current frame (MiDaS)
    depth_map = predict_depth(img)  # (H, W) float32

    # Run YOLO on the same frame
    # NOTE: stream=True returns a generator, but we'll just iterate
    # and accumulate detections in a list for this frame.
    yolo_results = model(img, stream=True)

    # This will hold all pothole detections + depth stats for this frame
    frame_results = []

    for r in yolo_results:
        boxes = r.boxes

        for box in boxes:
            # Bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Width and height (not strictly needed for draw_results)
            w, h = x2 - x1, y2 - y1

            # Confidence and class
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])

            if conf > CONF_THRESH:
                #Depth-based severity estimation
                depth_metrics = estimate_pothole_depth_relative(
                    depth_map,
                    (x1, y1, x2, y2),
                    road_margin=ROAD_MARGIN,
                )

                max_rel = depth_metrics["max_rel"]
                mean_rel = depth_metrics["mean_rel"]
                road_depth = depth_metrics["road_depth"]

                # Build label text
                label = f"{classNames[cls]} {conf:.2f}"

                # Only append depth if we successfully estimated road_depth
                if road_depth is not None:
                    # Note: units are relative MiDaS units, not meters
                    label += f" d={max_rel:.2f}"

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
