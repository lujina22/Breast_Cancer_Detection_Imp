import cv2
import numpy as np

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
# Top-hat enhancement
# -------------------------
def tophat_transform(img):
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, se)
    enhanced = cv2.subtract(img, opened)
    return enhanced

# -------------------------
# Local 5x5 mean/std
# -------------------------
def local_5x5_std(img):
    kernel = np.ones((5,5),np.float32)/25
    mean = cv2.filter2D(img.astype(np.float32),-1,kernel)
    sqr = img.astype(np.float32)**2
    mean_sqr = cv2.filter2D(sqr,-1,kernel)
    std = np.sqrt(np.maximum(mean_sqr - mean**2,0))
    return std

# -------------------------
# Create search region mask
# -------------------------
def create_search_region(shape, center_x, center_y, radius):
    mask = np.zeros(shape, np.uint8)
    search_radius = int(radius * 1.1)
    cv2.circle(mask, (center_x, center_y), search_radius, 255, -1)
    return mask

# -------------------------
# Detect MCs ONLY within search region
# -------------------------
def detect_mcs(std_img, search_mask):
    std_masked = cv2.bitwise_and(std_img, std_img, mask=search_mask)
    std_norm = cv2.normalize(std_masked, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    search_pixels = std_norm[search_mask > 0]
    if len(search_pixels) == 0:
        return np.zeros_like(std_norm)
    
    thresh_val = np.percentile(search_pixels, 90)
    mask = (std_norm >= thresh_val).astype(np.uint8) * 255
    mask = cv2.bitwise_and(mask, mask, mask=search_mask)
    
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.erode(mask, kernel_erode, iterations=1)
    
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    filtered = np.zeros_like(mask)
    
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        width = stats[i, cv2.CC_STAT_WIDTH]
        height = stats[i, cv2.CC_STAT_HEIGHT]
        aspect_ratio = max(width, height) / (min(width, height) + 1e-10)
        if 4 <= area <= 35 and aspect_ratio < 2.8:
            filtered[labels == i] = 255
    
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
# Evaluation
# -------------------------
def evaluate_pixel_classification(pred_mask, gt_mask):
    pred_flat = (pred_mask > 0).flatten()
    gt_flat = (gt_mask > 0).flatten()
    
    TP = np.sum((pred_flat == 1) & (gt_flat == 1))
    TN = np.sum((pred_flat == 0) & (gt_flat == 0))
    FP = np.sum((pred_flat == 1) & (gt_flat == 0))
    FN = np.sum((pred_flat == 0) & (gt_flat == 1))
    
    total = TP + TN + FP + FN
    
    sensitivity = TP / (TP + FN + 1e-10) * 100
    specificity = TN / (TN + FP + 1e-10) * 100
    accuracy = (TP + TN) / (total + 1e-10) * 100
    precision = TP / (TP + FP + 1e-10) * 100
    f1 = 2 * TP / (2 * TP + FP + FN + 1e-10) * 100
    
    return {
        'accuracy': accuracy,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'f1_score': f1,
        'TP': TP,
        'TN': TN,
        'FP': FP,
        'FN': FN
    }

# -------------------------
# Full pipeline
# -------------------------
def run_single_image(image_path, x, y, radius):
    img = load_image(image_path)
    roi = image_crop(img, x, y)
    enhanced = tophat_transform(roi)
    std = local_5x5_std(enhanced)
    center = roi.shape[0] // 2
    search_mask = create_search_region(std.shape, center, center, radius)
    pred_mask = detect_mcs(std, search_mask)
    gt_mask = doctor_mask(pred_mask.shape, radius)
    metrics = evaluate_pixel_classification(pred_mask, gt_mask)
    return pred_mask, gt_mask, metrics

def get_mc_data(roi, radius=None):
    
    if roi is None or roi.size == 0:
        return [], None, None
    tophat_norm = tophat_transform(roi)
    std = local_5x5_std(tophat_norm)
    search_mask = np.ones_like(std, dtype=np.uint8) * 255
    mc_mask = detect_mcs(std, search_mask)
    contours, _ = cv2.findContours(mc_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    overlay = cv2.cvtColor(roi, cv2.COLOR_GRAY2RGB)
    search_vis = overlay.copy()
    search_vis[search_mask > 0] = [100, 100, 255]
    overlay = cv2.addWeighted(overlay, 0.85, search_vis, 0.15, 0)

    if radius is not None:
        gt_mask = doctor_mask(roi.shape, radius)
        gt_vis = overlay.copy()
        gt_vis[gt_mask > 0] = [0, 255, 0]
        overlay = cv2.addWeighted(overlay, 0.75, gt_vis, 0.25, 0)

    overlay[mc_mask > 0] = [255, 0, 0]

    if radius is not None:
        overlap = (mc_mask > 0) & (gt_mask > 0)
        overlay[overlap] = [255, 255, 0]

    return contours, tophat_norm, overlay