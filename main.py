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
video_path = "Media/Potholes.mp4"
cap = cv2.VideoCapture(video_path)

# YOLO model with custom weights
model = YOLO(
    r"c:/Users/amand/Desktop/CMPT 742/Project Potholes/CMPT742-Final-Project---Pothole-Detection/runs/detect/train14/weights/best.pt"
)

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

                # Debug print if you want to monitor relative values
                print(f"max_rel = {max_rel:.6f}, mean_rel = {mean_rel:.6f}")

                # NEW: metric depths (cm) from depth_utils.py
                max_depth_cm = depth_metrics.get("max_depth_cm", 0.0)
                mean_depth_cm = depth_metrics.get("mean_depth_cm", 0.0)

                # Decide severity based on max_depth_cm
                severity = classify_severity(max_depth_cm)

                # Build detection record for visualization helper
                # visualize.draw_results expects:
                #   {
                #     "bbox": (xmin, ymin, xmax, ymax),
                #     "score": float,
                #     "max_rel": float,
                #     "mean_rel": float,
                #     "severity": str
                #   }
                det = {
                    "bbox": (x1, y1, x2, y2),
                    "score": float(conf),
                    "max_rel": float(max_rel),
                    "mean_rel": float(mean_rel),
                    "severity": severity,
                    # optional extra fields if you want later:
                    "max_depth_cm": float(max_depth_cm),
                    "mean_depth_cm": float(mean_depth_cm),
                    "class_name": classNames[cls],
                }

                frame_results.append(det)

    # ---- Visualization step: draw boxes + depth map ----

    # draw_results will:
    #  - draw the bounding boxes and labels on a copy of img
    #  - if show_depth=True and depth_map is provided,
    #    create a side-by-side image: [RGB | depth]
    vis_img = draw_results(
        frame_bgr=img,
        results=frame_results,
        depth_map=depth_map,
        show_depth=SHOW_DEPTH,
    )

    cv2.imshow("Pothole Detection + Depth", vis_img)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
