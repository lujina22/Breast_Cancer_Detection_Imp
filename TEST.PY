# import cv2
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# # --- CONFIGURATION ---
# # Use your clean CSV
# CSV_FILE = "train_dataset_FINAL_CLEAN.csv"

# def robust_tophat(img):
#     # 1. CLAHE (Contrast Enhancement)
#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
#     img_clahe = clahe.apply(img)

#     # 2. Top-Hat with a REASONABLE Kernel
#     # Since we are now using FULL RESOLUTION DDSM, specks are 5-10 pixels.
#     # Kernel should be ~25-30 pixels (bigger than the speck).
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    
#     # Apply Top-Hat
#     tophat = cv2.morphologyEx(img_clahe, cv2.MORPH_TOPHAT, kernel)
    
#     # 3. Threshold (Keep only the brightest 1% of pixels)
#     # This filters out the gray tissue and keeps the white sparks
#     thresh_val = np.percentile(tophat, 99.5) 
#     _, binary = cv2.threshold(tophat, thresh_val, 255, cv2.THRESH_BINARY)
    
#     # 4. Filter Noise (Tiny specs < 2 pixels)
#     # Using 2 pixels because real calcifications are bigger in full res
#     binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((2,2), np.uint8))
    
#     return img_clahe, tophat, binary

# def test_on_crop():
#     # Load dataset
#     df = pd.read_csv(CSV_FILE)
    
#     # Filter for DDSM only
#     ddsm_df = df[df['SOURCE'] == 'DDSM']
    
#     if ddsm_df.empty:
#         print("No DDSM images found in CSV!")
#         return

#     # Pick a random image
#     row = ddsm_df.sample(1).iloc[0]
#     path = row['PATH']
#     print(f"Testing on: {path}")

#     # Load FULL SIZE (Do not resize!)
#     full_img = cv2.imread(path, 0)
    
#     if full_img is None:
#         print("Could not read image.")
#         return

#     h, w = full_img.shape
#     print(f"Original Size: {w}x{h}")

#     # --- THE MAGIC FIX: CROP, DON'T RESIZE ---
#     # We take a 1024x1024 chunk from the CENTER of the image
#     # (The breast is usually in the center-ish)
#     start_y = h // 2 - 512
#     start_x = w // 2 - 512
    
#     # Ensure we don't go out of bounds
#     if start_x < 0: start_x = 0
#     if start_y < 0: start_y = 0
    
#     crop_img = full_img[start_y:start_y+1024, start_x:start_x+1024]
    
#     # Run pipeline on the CROP
#     enhanced, tophat, detected = robust_tophat(crop_img)

#     # Show results
#     plt.figure(figsize=(15, 5))
    
#     plt.subplot(1, 3, 1)
#     plt.imshow(enhanced, cmap='gray')
#     plt.title("1. Enhanced Crop (Full Res)")
    
#     plt.subplot(1, 3, 2)
#     plt.imshow(tophat, cmap='gray')
#     plt.title("2. Top-Hat Result")
    
#     plt.subplot(1, 3, 3)
#     plt.imshow(detected, cmap='gray')
#     plt.title("3. Detected Specks")
    
#     plt.show()

# if __name__ == "__main__":
#     test_on_crop()  


# import cv2
# import numpy as np
# import matplotlib.pyplot as plt

# # --- CONFIGURATION ---
# # Use one of the DDSM images you found in your clean list
# IMAGE_PATH = "archive/jpeg/1.3.6.1.4.1.9590.100.1.2.346281440912431381827919607274014474106/1-291.jpg" # <--- REPLACE WITH A REAL PATH from your CSV

# def detect_and_cluster(img_path):
#     # 1. Load FULL IMAGE
#     full_img = cv2.imread(img_path, 0)
#     if full_img is None: return
    
#     # 2. Crop Center (like before)
#     h, w = full_img.shape
#     crop = full_img[h//2-512 : h//2+512, w//2-512 : w//2+512]
    
#     # 3. Top-Hat Pipeline
#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
#     enhanced = clahe.apply(crop)
    
#     # Kernel size 25 for full res DDSM
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
#     tophat = cv2.morphologyEx(enhanced, cv2.MORPH_TOPHAT, kernel)
    
#     # Threshold (Brightest 0.5%)
#     thresh_val = np.percentile(tophat, 99.5)
#     _, binary = cv2.threshold(tophat, thresh_val, 255, cv2.THRESH_BINARY)
    
#     # 4. Remove Tiny Noise (1-2 pixels)
#     binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((2,2), np.uint8))
    
#     # 5. FIND CONTOURS
#     contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
#     # Get centers of all potential dots
#     centroids = []
#     for cnt in contours:
#         M = cv2.moments(cnt)
#         if M["m00"] != 0:
#             cX = int(M["m10"] / M["m00"])
#             cY = int(M["m01"] / M["m00"])
#             centroids.append((cX, cY))
    
#     # --- 6. THE CLUSTER CHECK (NOISE FILTER) ---
#     # We will verify every dot.
#     valid_centroids = []
#     noise_centroids = []
    
#     search_radius = 50 # Look for neighbors within 50 pixels
#     min_neighbors = 3  # Must have 3 friends to be a "Cluster"
    
#     centroids_arr = np.array(centroids)
    
#     if len(centroids) > 0:
#         for i, (cX, cY) in enumerate(centroids):
#             # Calculate distance to ALL other points
#             # dists = sqrt((x1-x2)^2 + (y1-y2)^2)
#             dists = np.sqrt(np.sum((centroids_arr - [cX, cY])**2, axis=1))
            
#             # Count how many are close (exclude self)
#             neighbors = np.sum(dists < search_radius) - 1
            
#             if neighbors >= min_neighbors:
#                 valid_centroids.append((cX, cY)) # It's a CLUSTER!
#             else:
#                 noise_centroids.append((cX, cY)) # It's lonely noise.
    
#     # --- VISUALIZE ---
#     vis_img = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
    
#     # Draw NOISE in RED X
#     for (nX, nY) in noise_centroids:
#         cv2.drawMarker(vis_img, (nX, nY), (0, 0, 255), markerType=cv2.MARKER_TILTED_CROSS, markerSize=10, thickness=2)
        
#     # Draw CLUSTERS in GREEN CIRCLES
#     for (vX, vY) in valid_centroids:
#         cv2.circle(vis_img, (vX, vY), 10, (0, 255, 0), 2)

#     plt.figure(figsize=(10, 10))
#     plt.imshow(vis_img)
#     plt.title(f"Green = Cluster (Cancer) | Red = Noise\nFound {len(valid_centroids)} Valid Specks")
#     plt.show()

# detect_and_cluster(IMAGE_PATH)  

# import cv2
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import os
# import glob

# # --- CONFIGURATION ---
# # 1. The image you want to test (Paste the path from your error/clean csv)
# IMAGE_PATH_TO_TEST = "archive/jpeg/1.3.6.1.4.1.9590.100.1.2.249417591513868599226701446953615972540/1-184.jpg"

# # 2. The Master CSV (The Answer Key)
# METADATA_CSV = "archive/csv/calc_case_description_train_set.csv" 

# # 3. The Root Folder where all JPEGs are hidden
# JPEG_ROOT = "archive/jpeg/"

# def get_combined_mask_from_csv(target_image_path):
#     # --- 1. IDENTIFY THE IMAGE UID ---
#     # We extract the long number from the folder path
#     # Path format: .../1.3.6.1.4.1.../1-106.jpg
#     # UID is the folder name
#     path_parts = target_image_path.replace("\\", "/").split("/")
#     if len(path_parts) < 2:
#         print("Error: Path format unexpected.")
#         return None
    
#     target_uid = path_parts[-2] # The long folder name
#     print(f"Looking for Answer Key for UID: {target_uid}...")

#     # --- 2. SEARCH THE CSV ---
#     try:
#         df = pd.read_csv(METADATA_CSV)
#     except:
#         print("CRITICAL: Could not read metadata CSV. Check path.")
#         return None

#     # Find all rows that match this image UID
#     # (One image might have multiple calcifications/rows)
#     matches = df[df['image file path'].str.contains(target_uid, na=False)]
    
#     if matches.empty:
#         print("No matches found in CSV for this image.")
#         return None

#     print(f"Found {len(matches)} abnormalities (clusters) for this image.")

#     # --- 3. FIND AND MERGE MASKS ---
#     # We need to find the mask file for EACH abnormality found
    
#     # We assume the image size from the target image
#     original_img = cv2.imread(target_image_path, 0)
#     h, w = original_img.shape
#     combined_mask = np.zeros((h, w), dtype=np.uint8)
    
#     found_masks = 0
    
#     for idx, row in matches.iterrows():
#         mask_dcm_path = row['ROI mask file path']
#         # Extract the Mask's UID Folder from the CSV path
#         # CSV Path: "DOI/.../1.3.6.1.4.1.9590.../000001.dcm"
#         mask_uid = mask_dcm_path.split("/")[-2]
        
#         # Hunt for this UID in your JPEG folder
#         search_pattern = os.path.join(JPEG_ROOT, "**", mask_uid, "*.jpg")
#         candidates = glob.glob(search_pattern, recursive=True)
        
#         if candidates:
#             mask_real_path = candidates[0]
#             # Load and add to master mask
#             m = cv2.imread(mask_real_path, 0)
#             if m is not None:
#                 m_resized = cv2.resize(m, (w, h)) # Ensure exact fit
#                 combined_mask = cv2.bitwise_or(combined_mask, m_resized)
#                 found_masks += 1
#         else:
#             print(f"Warning: Could not locate mask file on disk for UID {mask_uid}")

#     if found_masks == 0:
#         print("Could not find any mask JPEGs on disk.")
#         return None
        
#     return combined_mask

# def verify_and_plot():
#     print(f"Testing: {IMAGE_PATH_TO_TEST}")
    
#     # 1. Load Image
#     full_img = cv2.imread(IMAGE_PATH_TO_TEST, 0)
#     if full_img is None:
#         print("Image not found.")
#         return

#     # 2. Get the Truth (Blue)
#     gt_mask = get_combined_mask_from_csv(IMAGE_PATH_TO_TEST)
#     if gt_mask is None:
#         return

#     # 3. Run YOUR Code (Green) - Using the "Crop" method for speed/accuracy
#     h, w = full_img.shape
#     cy, cx = h // 2, w // 2
#     crop_size = 1024
#     y1, y2 = max(0, cy - crop_size//2), min(h, cy + crop_size//2)
#     x1, x2 = max(0, cx - crop_size//2), min(w, cx + crop_size//2)
    
#     crop_img = full_img[y1:y2, x1:x2]
#     crop_gt = gt_mask[y1:y2, x1:x2]

#     # --- TOP HAT PIPELINE ---
#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
#     enhanced = clahe.apply(crop_img)
    
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
#     tophat = cv2.morphologyEx(enhanced, cv2.MORPH_TOPHAT, kernel)
    
#     thresh_val = np.percentile(tophat, 99.5)
#     _, binary = cv2.threshold(tophat, thresh_val, 255, cv2.THRESH_BINARY)
#     binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((2,2), np.uint8))
#     # ------------------------

#     # 4. Visualize
#     vis_img = cv2.cvtColor(crop_img, cv2.COLOR_GRAY2BGR)
    
#     # Draw TRUTH (Blue)
#     gt_contours, _ = cv2.findContours(crop_gt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     cv2.drawContours(vis_img, gt_contours, -1, (255, 0, 0), 3)

#     # Draw PREDICTION (Green/Red)
#     pred_contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
#     hits = 0
#     misses = 0
    
#     for cnt in pred_contours:
#         # Check center of dot
#         M = cv2.moments(cnt)
#         if M["m00"] == 0: continue
#         cX = int(M["m10"] / M["m00"])
#         cY = int(M["m01"] / M["m00"])
        
#         # Check if inside ANY truth contour
#         hit = False
#         for gt_cnt in gt_contours:
#             if cv2.pointPolygonTest(gt_cnt, (cX, cY), False) >= 0:
#                 hit = True
#                 break
        
#         if hit:
#             cv2.drawContours(vis_img, [cnt], -1, (0, 255, 0), -1) # Green = Correct
#             hits += 1
#         else:
#             cv2.drawContours(vis_img, [cnt], -1, (0, 0, 255), 2) # Red = Noise
#             misses += 1

#     plt.figure(figsize=(12, 12))
#     plt.imshow(vis_img)
#     plt.title(f"BLUE=Truth | GREEN=Hit ({hits}) | RED=Noise ({misses})")
#     plt.axis('off')
#     plt.show()

# if __name__ == "__main__":
#     verify_and_plot()

import pandas as pd
import cv2
import matplotlib.pyplot as plt
import os
import random

# --- CONFIGURATION ---
CSV_FILE = "train_dataset_FIXED.csv" # The paired list we just made

def visualize_overlay():
    print(f"Reading {CSV_FILE}...")
    try:
        df = pd.read_csv(CSV_FILE)
    except:
        print("File not found. Please run the pairing script first.")
        return

    # Pick 5 random pairs to inspect
    # We use a loop to let you close one window and see the next
    indices = random.sample(range(len(df)), min(5, len(df)))
    
    print(f"Showing {len(indices)} random overlays...")

    for idx in indices:
        row = df.iloc[idx]
        img_path = row['CROP_PATH']
        mask_path = row['MASK_PATH']
        
        # 1. Load Both
        if not os.path.exists(img_path) or not os.path.exists(mask_path):
            print(f"Missing file for row {idx}")
            continue
            
        img = cv2.imread(img_path) # Load as Color for visualization
        mask = cv2.imread(mask_path, 0) # Load as Grayscale
        
        # 2. Resize Mask to Match Image
        # (This is critical: DDSM masks sometimes have slightly different padding)
        h, w = img.shape[:2]
        mask_resized = cv2.resize(mask, (w, h))
        
        # 3. Create the Overlay
        # We will make the Mask appear as a RED GLOW on top of the image
        
        # Create a red layer
        zeros = np.zeros_like(mask_resized)
        red_layer = cv2.merge([zeros, zeros, mask_resized]) # BGR format -> Red is last channel? No, OpenCV is BGR. Red is (0,0,255)
        # Actually in OpenCV: Blue, Green, Red. So (0, 0, Mask) is Red.
        
        # Create the overlay: Where mask is white, add red to the image
        # We use addWeighted to blend them
        # Convert mask to 3-channel for blending
        mask_rgb = cv2.cvtColor(mask_resized, cv2.COLOR_GRAY2BGR)
        
        # Make the mask red only
        mask_rgb[:, :, 0] = 0 # Zero Blue
        mask_rgb[:, :, 1] = 0 # Zero Green
        # Red channel keeps the mask value
        
        # Blend: 70% Image + 30% Red Mask
        overlay = cv2.addWeighted(img, 0.7, mask_rgb, 0.3, 0)
        
        # 4. Draw Contours for extra clarity (Yellow Line)
        contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 255, 255), 2) # Yellow line

        # 5. Show Side-by-Side
        plt.figure(figsize=(15, 6))
        
        plt.subplot(1, 3, 1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title("1. Original Crop")
        plt.axis("off")
        
        plt.subplot(1, 3, 2)
        plt.imshow(mask_resized, cmap='gray')
        plt.title("2. The Mask (Answer Key)")
        plt.axis("off")
        
        plt.subplot(1, 3, 3)
        plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        plt.title("3. Overlay (Yellow Line = Truth)")
        plt.axis("off")
        
        plt.show()

if __name__ == "__main__":
    import numpy as np # Forgot import
    visualize_overlay()

