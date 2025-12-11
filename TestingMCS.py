# import pandas as pd
# import cv2
# import numpy as np
# import os

# # --- CONFIGURATION ---
# INPUT_CSV = "train_dataset_FIXED.csv"  # The corrected list from the previous step
# RESULTS_DIR = "Final_Results_Proof"

# # --- 1. YOUR EXACT TOP-HAT CODE ---


# def users_blob_algo(img):
#     # 1. Resize
#     img = cv2.resize(img, (1024, 1024))
    
#     # Ensure Grayscale for processing
#     if len(img.shape) == 3:
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     else:
#         gray = img.copy()

#     # 2. Denoise
#     noise_sigma = gray.std()
#     med_k = 5
#     gray_med = cv2.medianBlur(gray, med_k)
    
#     # CLAHE
#     clip = float(np.interp(noise_sigma, [0, 50], [1.0, 2.0])) 
#     clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8,8))
#     img_clahe = clahe.apply(gray_med)
    
#     # 3. Top-Hat (TUNED FOR BLOBS)
#     H, W = gray.shape
#     diag = np.sqrt(H*H + W*W)
    
#     # --- BIGGER KERNEL ---
#     # We increase this to ~60px. This preserves larger "blobs".
#     base = int(np.clip(diag / 25, 30, 80)) 
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (base, base))
    
#     eroded = cv2.erode(img_clahe, kernel, iterations=1)
#     opened = cv2.dilate(eroded, kernel, iterations=1)
#     tophat = cv2.subtract(img_clahe, opened)
    
#     tophat_norm = cv2.normalize(tophat, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
#     # 4. Threshold
#     threshold_val = np.percentile(tophat_norm, 99.5)
#     _, binary = cv2.threshold(tophat_norm, int(threshold_val), 255, cv2.THRESH_BINARY)
    
#     # 5. Cleanup
#     close_k = int(np.clip(base/3, 5, 15)) # Stronger closing
#     kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
#     binary_closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close, iterations=1)
    
#     # 6. Area Filter (TUNED)
#     # MIN: Increased to 40 (Ignores tiny noise)
#     min_area_px = 40 
    
#     num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_closed, 8)
#     binary_clean = np.zeros_like(binary_closed)
    
#     for i in range(1, num_labels):
#         area = stats[i, cv2.CC_STAT_AREA]
        
#         # --- CRITICAL: ALLOW LARGE BLOBS ---
#         if area >= min_area_px:
#             binary_clean[labels == i] = 255
            
#     # Return GRAYSCALE image to avoid the crash
#     return gray, binary_clean

# def run_test():
#     if not os.path.exists(INPUT_CSV):
#         print(f"Error: {INPUT_CSV} not found. Please run 'generate_cropped_masks.py' first.")
#         return
    
#     print(f"Loading {INPUT_CSV}...")
#     df = pd.read_csv(INPUT_CSV)
#     if not os.path.exists(RESULTS_DIR): os.makedirs(RESULTS_DIR)
    
#     print(f"Testing... (Saving images to {RESULTS_DIR})")
    
#     count = 0
#     for idx, row in df.iterrows():
#         if count >= 20: break 
        
#         # Robust column checking
#         c_path = row.get('crop_path') or row.get('CROP_PATH') or row.get('PATH')
#         m_path = row.get('mask_path') or row.get('MASK_PATH')
        
#         if not c_path or not m_path: continue
#         if not os.path.exists(c_path) or not os.path.exists(m_path): continue
            
#         img = cv2.imread(c_path)
#         gt_mask = cv2.imread(m_path, 0)
        
#         # Run Algo
#         processed_img, my_mask = users_blob_algo(img)
        
#         # --- THE FIX: HANDLE COLOR/GRAY ---
#         if len(processed_img.shape) == 2:
#             vis = cv2.cvtColor(processed_img, cv2.COLOR_GRAY2BGR)
#         else:
#             vis = processed_img.copy() # Already color
        
#         # Doctor (Yellow)
#         gt_resized = cv2.resize(gt_mask, (1024, 1024))
#         # Fix invisible mask
#         if gt_resized.max() <= 1: gt_resized = gt_resized * 255
#         _, gt_resized = cv2.threshold(gt_resized, 127, 255, cv2.THRESH_BINARY)
#         gt_cnts, _ = cv2.findContours(gt_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         cv2.drawContours(vis, gt_cnts, -1, (0, 255, 255), 2)
        
#         # Your Prediction (Green)
#         my_cnts, _ = cv2.findContours(my_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         cv2.drawContours(vis, my_cnts, -1, (0, 255, 0), 2)
        
#         save_path = os.path.join(RESULTS_DIR, f"Blob_Result_{idx}.jpg")
#         cv2.imwrite(save_path, vis)
#         print(f"Saved: {save_path}")
#         count += 1
        
#     print(f"\nDone! Check '{RESULTS_DIR}'. The crash is fixed.")

# if __name__ == "__main__":
#     run_test()













import pandas as pd
import cv2
import numpy as np
import os

# --- CONFIGURATION ---
# We use the full dataset we linked earlier
INPUT_CSV = "train_dataset_FULL.csv" 
# Fallback if full not found
if not os.path.exists(INPUT_CSV): INPUT_CSV = "train_dataset_FIXED.csv"

def users_final_algo(original):
    # --- YOUR EXACT CODE LOGIC ---
    # Force high-quality resizing
    original = cv2.resize(original, (1024, 1024), interpolation=cv2.INTER_CUBIC)

    if len(original.shape) == 3:
        img = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    else:
        img = original.copy()

    # --- 1) Denoise ---
    noise_sigma = img.std()
    if noise_sigma < 10:
        med_k = 3
    elif noise_sigma < 30:
        med_k = 3
    else:
        med_k = 5
    img_med = cv2.medianBlur(img, med_k)

    clip = float(np.interp(noise_sigma, [0, 50], [1.5, 3.0]))
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8,8))
    img_clahe = clahe.apply(img_med)

    # --- 2) Top-Hat (Extreme Tuning) ---
    H, W = img.shape
    diag = np.sqrt(H*H + W*W)
    base = int(np.clip(diag / 20, 40, 100)) # Giant Kernel
    kernel_size = (base, base)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)

    eroded = cv2.erode(img_clahe, kernel, iterations=1)
    opened = cv2.dilate(eroded, kernel, iterations=1)
    tophat = cv2.subtract(img_clahe, opened)

    tophat_norm = cv2.normalize(tophat, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # --- 3) Adaptive Thresholding ---
    threshold_val = np.percentile(tophat_norm, 99.0) # Lowered to 99.0
    _, binary2 = cv2.threshold(tophat_norm, int(threshold_val), 255, cv2.THRESH_BINARY)

    # --- 5) Cleanup ---
    close_k = int(np.clip(base/3, 5, 15))
    if close_k % 2 == 0: close_k += 1
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
    binary_closed = cv2.morphologyEx(binary2, cv2.MORPH_CLOSE, kernel_close, iterations=1)

    # Min Area 50
    min_area_px = 50
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_closed, 8)
    binary_bool = np.zeros_like(binary_closed)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area_px:
            binary_bool[labels == i] = 255
    binary_closed = binary_bool

    # --- 6) Contour Filtering ---
    contours, _ = cv2.findContours(binary_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = min_area_px
    max_area = 100000 # Max limit disabled

    mc_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (min_area <= area <= max_area): continue
        perim = cv2.arcLength(cnt, True)
        if perim == 0: continue
        
        circularity = 4 * np.pi * (area / (perim * perim))
        if not (0.2 <= circularity <= 1.8): continue
        
        mask = np.zeros_like(tophat_norm)
        cv2.drawContours(mask, [cnt], -1, 255, -1)
        mean_val = cv2.mean(tophat_norm, mask=mask)[0]
        if mean_val < 20: continue

        mc_contours.append(cnt)

    # --- 7) Output Mask ---
    mc_mask = np.zeros_like(binary_closed)
    cv2.drawContours(mc_mask, mc_contours, -1, 255, -1)
    
    return mc_mask

def run_accuracy_test():
    if not os.path.exists(INPUT_CSV):
        print(f"Error: Could not find {INPUT_CSV}")
        return

    print(f"Loading {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)
    
    ious = []
    dices = []
    precisions = []
    recalls = []
    
    print(f"Testing {len(df)} images... (This might take a minute)")
    
    for idx, row in df.iterrows():
        # Handle different column names from versions
        c_path = row.get('CROP_PATH') or row.get('crop_path')
        m_path = row.get('MASK_PATH') or row.get('mask_path')
        
        if not c_path or not m_path: continue
        if not os.path.exists(c_path) or not os.path.exists(m_path): continue
            
        # 1. Load Images
        img = cv2.imread(c_path)
        gt_mask = cv2.imread(m_path, 0)
        
        if img is None or gt_mask is None: continue

        # 2. Run YOUR Algo
        pred_mask = users_final_algo(img)
        
        # 3. Prepare Ground Truth (Resize to match 1024x1024)
        gt_resized = cv2.resize(gt_mask, (1024, 1024), interpolation=cv2.INTER_NEAREST)
        
        # Fix invisible mask issue (0/1 -> 0/255)
        if gt_resized.max() <= 1: gt_resized = gt_resized * 255
        
        _, gt_binary = cv2.threshold(gt_resized, 127, 255, cv2.THRESH_BINARY)
        
        # 4. Calculate Metrics
        # Flatten to 1D arrays for easier math
        pred_flat = (pred_mask > 0).astype(int).flatten()
        gt_flat = (gt_binary > 0).astype(int).flatten()
        
        intersection = np.sum(pred_flat * gt_flat)
        pred_area = np.sum(pred_flat)
        gt_area = np.sum(gt_flat)
        union = pred_area + gt_area - intersection
        
        # IoU
        if union == 0:
            iou = 1.0 # Perfect match (both empty)
        else:
            iou = intersection / union
            
        # Dice
        if (pred_area + gt_area) == 0:
            dice = 1.0
        else:
            dice = (2 * intersection) / (pred_area + gt_area)
            
        # Precision (TP / TP + FP)
        if pred_area == 0:
            precision = 1.0 if gt_area == 0 else 0.0
        else:
            precision = intersection / pred_area
            
        # Recall (TP / TP + FN) -> Sensitivity
        if gt_area == 0:
            recall = 1.0 if pred_area == 0 else 0.0
        else:
            recall = intersection / gt_area
            
        ious.append(iou)
        dices.append(dice)
        precisions.append(precision)
        recalls.append(recall)
        
        if idx % 100 == 0:
            print(f"Processed {idx}/{len(df)}...")

    # --- FINAL REPORT ---
    print("\n" + "="*40)
    print("FINAL ALGORITHM ACCURACY REPORT")
    print("="*40)
    print(f"Total Images Tested: {len(ious)}")
    print(f"Average IoU:         {np.mean(ious):.4f}")
    print(f"Average Dice Score:  {np.mean(dices):.4f}")
    print(f"Average Precision:   {np.mean(precisions):.4f}")
    print(f"Average Recall:      {np.mean(recalls):.4f}")
    print("="*40)

if __name__ == "__main__":
    run_accuracy_test()