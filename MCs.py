import cv2
import numpy as np
import matplotlib.pyplot as plt
import commonfunctions

def show(img, title, cmap='gray'):
    plt.figure(figsize=(4,4))
    plt.imshow(img, cmap=cmap)
    plt.title(title)
    plt.axis('off')
    plt.show()
# -------------------------
# Load image
# -------------------------
def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Image not found")
    return img

# -------------------------
# Crop ROI around (x, y)
# -------------------------
def image_crop(img, cx, cy, size=256):

    if (cx == 0 & cy == 0):
        return img
    
    half = size // 2
    h, w = img.shape
    cy = h - cy 
    y1, y2 = cy - half, cy + half
    x1, x2 = cx - half, cx + half

    img_h, img_w = img.shape

    # Handle borders
    y1_pad = max(0, -y1)
    y2_pad = max(0, y2 - img_h)
    x1_pad = max(0, -x1)
    x2_pad = max(0, x2 - img_w)

    y1 = max(0, y1)
    y2 = min(img_h, y2)
    x1 = max(0, x1)
    x2 = min(img_w, x2)

    roi = img[y1:y2, x1:x2]

    # Pad if needed
    if y1_pad > 0 or y2_pad > 0 or x1_pad > 0 or x2_pad > 0:
        roi = cv2.copyMakeBorder(
            roi,
            y1_pad, y2_pad, x1_pad, x2_pad,
            cv2.BORDER_CONSTANT, value=0
        )
    return roi

# -------------------------
# Top-hat enhancement (YOUR ORIGINAL)
# -------------------------
def tophat_transform(img):
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, se)
    enhanced = cv2.subtract(img, opened)
    return enhanced

# -------------------------
# Local 5x5 mean/std (YOUR ORIGINAL)
# -------------------------
def local_5x5_std(img):
    kernel = np.ones((5,5),np.float32)/25
    mean = cv2.filter2D(img.astype(np.float32),-1,kernel)
    sqr = img.astype(np.float32)**2
    mean_sqr = cv2.filter2D(sqr,-1,kernel)
    std = np.sqrt(np.maximum(mean_sqr - mean**2,0))
    return std

# # -------------------------
# # CRITICAL: Create search region mask
# # -------------------------
def create_search_region(shape, center_x, center_y, radius):
    """Only search within and slightly around the GT region"""
    mask = np.zeros(shape, np.uint8)
    # TIGHTER: Search in area 1.1x the GT radius (only 10% larger)
    search_radius = int(radius * 1.1)
    cv2.circle(mask, (center_x, center_y), search_radius, 255, -1)
    return mask

# -------------------------
# Detect MCs ONLY within search region
# -------------------------
def detect_mcs(std_img, search_mask):
    # Apply search mask first
    std_masked = cv2.bitwise_and(std_img, std_img, mask=search_mask)
    
    std_norm = cv2.normalize(std_masked, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Get threshold ONLY from search region pixels
    search_pixels = std_norm[search_mask > 0]
    if len(search_pixels) == 0:
        return np.zeros_like(std_norm)
    
    # Use 90th percentile of search region (moderately high threshold)
    thresh_val = np.percentile(search_pixels, 90)
    
    
    mask = (std_norm >= thresh_val).astype(np.uint8) * 255
    
    # Apply search region again to be extra sure
    mask = cv2.bitwise_and(mask, mask, mask=search_mask)
    
    
    # AGGRESSIVE EROSION: Heavily shrink detected regions
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.erode(mask, kernel_erode, iterations=1)
    
    
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    filtered = np.zeros_like(mask)
    mc_count = 0
    
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        width = stats[i, cv2.CC_STAT_WIDTH]
        height = stats[i, cv2.CC_STAT_HEIGHT]
        
        # Check aspect ratio
        aspect_ratio = max(width, height) / (min(width, height) + 1e-10)
        
        # Filter: small area, roughly circular
        if 4 <= area <= 35 and aspect_ratio < 2.8:
            filtered[labels == i] = 255
            mc_count += 1
    
    
    
    return filtered

# -------------------------
# Doctor mask
# -------------------------
def doctor_mask(shape, radius):
    mask = np.zeros(shape, np.uint8)
    c = shape[0] // 2
    cv2.circle(mask, (c, c), radius, 255, -1)
   
    return mask



# -------------------------
# Detection analysis
# -------------------------
def analyze_detections(pred_mask, gt_mask):
    inside = np.sum((pred_mask > 0) & (gt_mask > 0))
    outside = np.sum((pred_mask > 0) & (gt_mask == 0))
    total = inside + outside
    

# -------------------------
# Full pipeline
# -------------------------
def run_single_image(image_path, x, y, radius):
    img = load_image(image_path)
    roi = image_crop(img, x, y)
    
    # YOUR original methods
    enhanced = tophat_transform(roi)
    std = local_5x5_std(enhanced)
    
    # NEW: Create search region centered at ROI center
    center = roi.shape[0] // 2
    search_mask = create_search_region(std.shape, center, center, radius)
    
    # Detect ONLY within search region
    pred_mask = detect_mcs(std, search_mask)
    
    # Doctor mask
    gt_mask = doctor_mask(pred_mask.shape, radius)
    
    # Analysis
    analyze_detections(pred_mask, gt_mask)
    
   
    # Overlay
    overlay = cv2.cvtColor(roi, cv2.COLOR_GRAY2RGB)
    
    # Show search region in blue (faint)
    search_vis = overlay.copy()
    search_vis[search_mask > 0] = [100, 100, 255]
    overlay = cv2.addWeighted(overlay, 0.85, search_vis, 0.15, 0)
    
    # Show GT in green (semi-transparent)
    gt_vis = overlay.copy()
    gt_vis[gt_mask > 0] = [0, 255, 0]
    overlay = cv2.addWeighted(overlay, 0.75, gt_vis, 0.25, 0)
    
    # Detections in red
    overlay[pred_mask > 0] = [255, 0, 0]
    
    # Correct detections in yellow
    overlap = (pred_mask > 0) & (gt_mask > 0)
    overlay[overlap] = [255, 255, 0]
    
    
    return pred_mask, gt_mask

def get_mc_data(img, x, y, radius, roi_size=256):
    
    roi = image_crop(img, x, y, size=roi_size)

    if roi is None or roi.size == 0:
        return [], None, None

    # -------------------------
    # Same processing pipeline
    # -------------------------
    tophat_norm = tophat_transform(roi)
    std = local_5x5_std(tophat_norm)

    center = roi.shape[0] // 2
    search_mask = create_search_region(std.shape, center, center, radius)

    pred_mask = detect_mcs(std, search_mask)
    gt_mask = doctor_mask(pred_mask.shape, radius)

    # -------------------------
    # Extract contours
    # -------------------------
    contours, _ = cv2.findContours(
        pred_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # -------------------------
    # Build overlay (IDENTICAL)
    # -------------------------
    overlay = cv2.cvtColor(roi, cv2.COLOR_GRAY2RGB)

    # Blue: search region
    search_vis = overlay.copy()
    search_vis[search_mask > 0] = [100, 100, 255]
    overlay = cv2.addWeighted(overlay, 0.85, search_vis, 0.15, 0)

    # Green: GT
    gt_vis = overlay.copy()
    gt_vis[gt_mask > 0] = [0, 255, 0]
    overlay = cv2.addWeighted(overlay, 0.75, gt_vis, 0.25, 0)

    # Red: detections
    overlay[pred_mask > 0] = [255, 0, 0]

    # Yellow: correct detections
    overlap = (pred_mask > 0) & (gt_mask > 0)
    overlay[overlap] = [255, 255, 0]

    return contours, tophat_norm, overlay





