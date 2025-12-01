
# This program visualizes raw detections and depth info on frames.
# - draw_results draws pothole boxes + text
# - optionally shows RGB + depth side-by-side


import cv2
import numpy as np


def normalize_depth_for_display(depth_map):
    """
    Convert a raw depth_map (float32) into a nice 8-bit image
    that we can add colour and dispaly

    Input:
        depth_map: 2D float32 array, shape (H, W)

    Output:
        depth_normalized: 2D uint8 array, shape (H, W), values in [0, 255]
    """

    #make a copy of the depth map so don't modify the original array
    depth_map_copy = depth_map.copy()

    # Handle edge case where depth_map is all one value
    min_val = np.min(depth_map_copy)
    max_val = np.max(depth_map_copy)

    if max_val - min_val < 1e-6:
        #if all the pixels look similar, return zeros
        return np.zeros_like(depth_map_copy, dtype=np.uint8)

    #normalize depth to [0, 1]
    d_norm = (depth_map_copy - min_val) / (max_val - min_val)

    #Suggestion: Can invert if prefer "closer = brighter", and flip 
    # d_norm = 1.0 - d_norm

    #convert [0, 1] float to [0, 255] uint8
    depth_normalized=  (d_norm * 255.0).astype(np.uint8)

    return depth_normalized


def draw_results(frame_bgr, results, depth_map=None, show_depth=False):
    """
    Draw pothole detection results on a frame.

    Inputs:
        frame_bgr : original BGR frame from OpenCV, shape (H, W, 3)
        results   : list of dicts, each with:
                    {
                      "bbox": (xmin, ymin, xmax, ymax),
                      "score": float,
                      "max_rel": float,
                      "mean_rel": float,
                      "severity": str
                    }
        depth_map : (optional) 2D float32 array (H, W) from MiDaS
        show_depth: if True and depth_map is provided, it will
                    create a side-by-side image: [RGB | depth]

    Output:
        vis_bgr   : BGR image with drawings (either just RGB or RGB+depth)
    """

    # Make a copy so we don't draw directly on the original frame
    vis = frame_bgr.copy()

    # Loop over each detected pothole result
    for res in results:
        #unpack bounding box and make sure it is in integer pixel coords
        xmin, ymin, xmax, ymax = map(int, res["bbox"])

        #draw rectangle around the pothole (green box)
        cv2.rectangle(vis, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

        # Build the label text for this pothole
        # Example: "moderate | max_rel=0.023"
        severity = res.get("severity", "unknown")
        max_rel = res.get("max_rel", 0.0)
        text = f"{severity} | max_rel={max_rel:.3f}"

        # Choose where to put the text  - current orientation is slightly above the box
        text_x = xmin
        text_y = max(0, ymin - 5)

        #Add text to the image
        cv2.putText(
            vis,                    # image to draw on
            text,                   # text string
            (text_x, text_y),       # bottom-left corner of text
            cv2.FONT_HERSHEY_SIMPLEX,  # font
            0.5,                    # font scale
            (0, 255, 0),            # text color (green)
            1,                      # thickness
            cv2.LINE_AA             # anti-aliased lines
        )

    # If we do NOT want to show depth, just return vis
    if not show_depth or depth_map is None:
        return vis

    # If show_depth = True and depth_map is provided:
    #   Create a colored depth visualization and stack it
    #   side-by-side with the RGB frame.

    #1. Convert depth_map to [0, 255] uint8 for display
    depth_8u = normalize_depth_for_display(depth_map)

    #2. Apply colormap so depth differences are easier to see
    depth_color = cv2.applyColorMap(depth_8u, cv2.COLORMAP_MAGMA)

    #3. check depth_color has same size as vis 
    h1, w1 = vis.shape[:2]
    h2, w2 = depth_color.shape[:2]

    if (h1 != h2) or (w1 != w2):
        depth_color = cv2.resize(depth_color, (w1, h1))

    # 4. Put a label on the depth image
    # ----- COLOR CODE BY SEVERITY -----
    severity = res.get("severity", "unknown")
    conf = res.get("score", 0.0)
    depth_score = res.get("depth_score", 0.0)
    max_depth_cm = res.get("max_depth_cm", 0.0)

    # Choose color: (B, G, R)
    if severity == "shallow":
        color = (0, 255, 0)       # green
    elif severity == "moderate":
        color = (0, 255, 255)     # yellow
    elif severity == "severe":
        color = (0, 0, 255)       # red
    else:
        color = (255, 255, 255)   # white fallback

    # Build the label string
    text = (
        f"{severity} | "
        f"Conf:{conf:.2f} | "
        f"Depth:{depth_score:.2f} | "
        f"~{max_depth_cm:.1f}cm"
    )

    # Text placement
    text_x = xmin
    text_y = max(0, ymin - 5)

    # Draw shaded rectangle for readability
    (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(
        vis,
        (text_x, text_y - text_h - 4),
        (text_x + text_w + 4, text_y + 4),
        color,
        -1
    )

    # Draw the text
    cv2.putText(
        vis,
        text,
        (text_x + 2, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 0),       # black text
        1,
        cv2.LINE_AA
    )

    # 5. Concatenate horizontally: [RGB with boxes | Depth map]
    vis_combined = np.hstack([vis, depth_color])

    return vis_combined
