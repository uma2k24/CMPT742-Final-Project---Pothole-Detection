
# This program compute pothole depth
# from a depth map + a YOLO bounding box
#
#   Given:
#       - a depth map (from MiDaS)
#       - a pothole bounding box (from YOLO)
#   Compute: the relative depth of the pothole compared to surrounding road
#       1. depth of road surface around the pothole
#       2. depth inside the pothole pixels
#       3. relative depth = pothole_depth - road_depth (for MiDaS)
#
#
#     Because MiDaS outputs RELATIVE INVERSE DEPTH:
#       - Larger values = closer to camera
#       - Smaller values = farther from camera
#
# So logic should be that a pothole is lower than road surface, so farther from camera
#      - Thus, pothole depth values should be SMALLER than road depth values
#      - Therefore:
#           road_depth - pothole_depth > 0 is  depression, meaning pothole  
#
#   Result shows how  deep the pothole is compared to the road.


import numpy as np



# Helper Function: extract pothole region + road context
def extract_regions(depth_map, bbox, road_margin=15):
    """
    Extract depth values in:
        (a) pothole bounding box (inside the pothole)
        (b) road context region around the pothole (flat road)

    Inputs:
        depth_map   : 2D numpy array (H, W), float32
        bbox        : (xmin, ymin, xmax, ymax)
        road_margin : how many pixels around the box to include as "road"

    Returns:
        pothole_region : 2D array of depths inside pothole
        road_context   : 2D array of depths around pothole (pothole area removed)
    """

    xmin, ymin, xmax, ymax = bbox
    h, w = depth_map.shape

    #Ensure bounding box coordinates are inside image
    xmin = max(0, int(xmin))
    ymin = max(0, int(ymin))
    xmax = min(w - 1, int(xmax))
    ymax = min(h - 1, int(ymax))

 
    #1. Extract pothole depth region
    pothole_region = depth_map[ymin:ymax, xmin:xmax]

    #2. Extract road context region with margin
    xmin_r = max(0, xmin - road_margin)
    xmax_r = min(w - 1, xmax + road_margin)
    ymin_r = max(0, ymin - road_margin)
    ymax_r = min(h - 1, ymax + road_margin)

    road_context = depth_map[ymin_r:ymax_r, xmin_r:xmax_r].copy()


    #Mask out the pothole region inside the road context
    # We set it to NaN so it will not affect median road depth calculation 
    pothole_h = ymax - ymin
    pothole_w = xmax - xmin

    road_context[ymin - ymin_r : ymin - ymin_r + pothole_h,
                 xmin - xmin_r : xmin - xmin_r + pothole_w] = np.nan

    return pothole_region, road_context




#Estimate flat road depth around pothole by median
def estimate_road_reference_depth(road_context):
    """
    Estimate the depth of the flat road around the pothole.

    We use the median because:
    - robust to noise
    - ignores outliers
    - gives stable road level

    Input:
        road_context : depth region around pothole (with NaNs removed)

    Output:
        road_depth : float or None
    """

    valid_pixels = np.isfinite(road_context)
    values = road_context[valid_pixels]

    if values.size == 0:
        return None

    # Median depth = estimated road plane
    road_depth = float(np.median(values))
    return road_depth




#Calulate pothole depth relative to road
def estimate_pothole_depth_relative(depth_map, bbox, road_margin=15):
    """
    Compute relative pothole depth.

    Inputs:
        depth_map : 2D numpy array (H, W),  MiDaS inverse-depth values
        bbox      : (xmin, ymin, xmax, ymax)
        road_margin : margin around pothole for estimating road depth

    Outputs:
        {
          "max_rel" : deepest point in pothole relative to road    (road_depth - pothole_depth) ,
          "mean_rel": average depression depth,
          "road_depth": depth of surrounding road     (median),
          "pothole_depth_values": raw depth values inside pothole (for debugging)
        }
        
    """

    # Extract depth inside pothole + depth around pothole
    pothole_region, road_context = extract_regions(depth_map, bbox, road_margin)

    # Estimate road depth (reference level)
    road_depth = estimate_road_reference_depth(road_context)

    if road_depth is None:
        # No valid road pixels -> return safe defaults
        return {
            "max_rel": 0.0,
            "mean_rel": 0.0,
            "road_depth": None,
            "pothole_depth_values": None
        }

    # MiDaS output inverse depth:
    #   - Larger value = closer to camera perspective
    #   - Smaller value = farther from camera perspective
    #
    # Flat road (closer) -> larger MiDaS val
    # Pothole bottom (farther) -> smaller MiDaS val
    #
    # Depression (pothole) is defined as how much deeper than road surface:
    #   diff = road_depth - pothole_depth
    # If diff > 0, current pixel is deeper than the surrounding road
    diff = road_depth - pothole_region

    # Keep only positive values (deeper than road)
    depression = diff[diff > 0]

    if depression.size == 0:
        # No depression means not a real pothole OR too shallow to measure
        return {
            "max_rel": 0.0,
            "mean_rel": 0.0,
            "road_depth": road_depth,
            "pothole_depth_values": pothole_region
        }

    max_rel = float(np.max(depression))
    mean_rel = float(np.mean(depression))

    return {
        "max_rel": max_rel,                 # deepest single pixel (relative depth)
        "mean_rel": mean_rel,               # average depth inside pothole
        "road_depth": road_depth,
        "pothole_depth_values": pothole_region
    }
