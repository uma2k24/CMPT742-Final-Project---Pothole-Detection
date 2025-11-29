#
# This program aims to convert RGB frame -> depth map 
# Use for calculating depth of pothole
#

import cv2
import torch
import numpy as np



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# Load MiDaS model from torch.hub
#    *MiDaS:
#    *need internet connection first time to download weights, but cached afterwards for reuse.
#
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
# - essential because noise in depth = incorrect pothole measurements.
#
# Architecture:
# - A ResNet-50 CNN backbone (good at edges + local details)
# - A Transformer decoder (good at global structure + depth reasoning)
#




MODEL_TYPE = "DPT_Hybrid"


#Checks if the MiDaS model is already stored in PyTorch cache, otherwise download model weight sand architecture.
midas = torch.hub.load("intel-isl/MiDaS", MODEL_TYPE) #load the model

#Move model to GPU if possible
midas.to(device)


midas.eval()  # set to evaluation mode - no training

#Load the transforms for the model for preprocessing
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

if MODEL_TYPE in ["DPT_Large", "DPT_Hybrid"]:
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform


def predict_depth(frame):
    """
    Compute a depth map for one BGR frame using MiDaS

    Input:
        frame: np.ndarray (H, W, 3), BGR from OpenCV

    Output:
        depth_map: np.ndarray (H, W), float32
                MiDaS will output relative depth -inverse depth values
                   Larger values = closer to camera
                   Smaller values = farther from camera
    """

    # Convert BGR to RGB (what MiDaS expects)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Apply the model's transform (resize, normalize, etc.)
    input_batch = transform(img_rgb).unsqueeze(0)  # add batch dimension
    
    # Move input to GPU
    input_batch = input_batch.to(device) #GPU if available


    #faster inference  by disable gradient calculation
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

    # Move result to CPU and convert to numpy float
    depth_map = prediction_resized[0].cpu().numpy().astype(np.float32)
    
    #depth_map[y, x] is a relative depth value 
    return depth_map
