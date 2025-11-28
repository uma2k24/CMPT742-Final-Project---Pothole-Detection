#
# This program aims to convert RGB frame -> depth map 
# Use for calculating depth of pothole
#

import cv2
import torch
import numpy as np

# ------------------------------
# Load MiDaS model from torch.hub
#    *MiDaS:
#    *need internet connection first time to download weights, but cached afterwards for reuse.
# ------------------------------


#Choose a model type: "DPT_Large", "DPT_Hybrid", "MiDaS_small"
#Chose "DPT_Hybrid" for a good balance of speed and accuracy
#________________________________________________________
# "DPT_Hybrid" :
#
# *Monocular depth estimation model which blends CNN and Transformers to produice depth maps fro ma single image
#
# - real-time or near real-time speed
# - smoother surfaces (useful for estimating road)
# - sharp enough edges (useful for pothole boundary)
# - less noise (important for depth difference calculations)
# - This is essential because noise in depth = incorrect pothole measurements.
#

MODEL_TYPE = "DPT_Hybrid"

midas = torch.hub.load("intel-isl/MiDaS", MODEL_TYPE)
midas.eval()  # set to evaluation mode (no training)

# Load the appropriate transforms for the model
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

if MODEL_TYPE in ["DPT_Large", "DPT_Hybrid"]:
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform


def predict_depth(frame):
    """
    Compute a depth map for one BGR frame using MiDaS.

    Input:
        frame: np.ndarray (H, W, 3), BGR from OpenCV.

    Output:
        depth_map: np.ndarray (H, W), float32
                   Larger values ≈ farther from camera.
    """

    # Convert BGR (OpenCV format) to RGB (what MiDaS expects)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Apply the model's transform (resize, normalize, etc.)
    input_batch = transform(img_rgb).unsqueeze(0)  # add batch dimension

    with torch.no_grad():
        # Forward pass through MiDaS
        prediction = midas(input_batch)

        # Resize prediction back to original image size
        prediction_resized = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),          # add channel dimension
            size=img_rgb.shape[:2],          # (H, W)
            mode="bicubic",
            align_corners=False,
        ).squeeze(1)  # remove channel dimension

    # Convert to numpy float32 array
    depth_map = prediction_resized[0].cpu().numpy().astype(np.float32)

    # At this point, depth_map[y, x] is a relative depth value
    return depth_map
