import cv2
import math
import cvzone
from ultralytics import YOLO

# Initialize video capture
video_path = r"c:/Users/amand/Desktop/CMPT 742/Project Potholes/CMPT742-Final-Project---Pothole-Detection/Media/Potholes.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("❌ Could not open video:", video_path)
    exit()

# Load YOLO model with custom weights
model = YOLO(r"c:/Users/amand/Desktop/CMPT 742/Project Potholes/CMPT742-Final-Project---Pothole-Detection/runs/detect/train12/weights/best.pt")

# Define class names
classNames = ['Pothole']

while True:
    success, img = cap.read()

    # 🔴 If we didn't get a frame, stop the loop (end of video or error)
    if not success or img is None:
        print("🔵 No more frames or failed to read frame. Exiting.")
        break

    # Run YOLO on this frame
    results = model(img, stream=True)

    for r in results:
        boxes = r.boxes
        if boxes is None:
            continue

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            w, h = x2 - x1, y2 - y1

            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])

            if conf > 0.4:
                cvzone.cornerRect(img, (x1, y1, w, h), t=2)
                cvzone.putTextRect(
                    img,
                    f'{classNames[cls]} {conf}',
                    (max(0, x1), max(35, y1)),
                    scale=1,
                    thickness=1
                )

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
