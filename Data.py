# import pandas as pd
# import cv2
# import numpy as np
# import os

# # --- CONFIGURATION ---
# # Use the file you currently have
# INPUT_CSV = "train_dataset_LITE_REPAIRED.csv" 
# # This will be your final, clean file
# OUTPUT_CSV = "train_dataset_FINAL_CLEAN.csv"

# def is_valid_mammogram(image_path):
#     """
#     Returns True ONLY if the image is a full, real mammogram.
#     Returns False if it is a mask, a crop, or broken.
#     """
#     # 1. Fix Path Slashes (Windows/Linux compatibility)
#     clean_path = image_path.replace("\\", "/")
    
#     if not os.path.exists(clean_path):
#         return False, "File Not Found"

#     # 2. Load Image (Grayscale)
#     # We use valid flags to avoid loading errors
#     try:
#         img = cv2.imread(clean_path, 0)
#     except:
#         return False, "Read Error"

#     if img is None:
#         return False, "Corrupt Image"

#     h, w = img.shape

#     # --- CHECK 1: SIZE (Filter out small crops) ---
#     # Full DDSM images are usually HUGE (3000+ height). 
#     # MIAS are 1024.
#     # Anything smaller than 800x800 is likely a useless crop.
#     if h < 800 or w < 800:
#         return False, f"Too Small ({w}x{h})"

#     # --- CHECK 2: CONTENT (Filter out Masks) ---
#     # A mask is mostly black (0) and white (255). It has very few unique gray values.
#     # A real X-ray has thousands of gray values.
    
#     # We take a sample of the center of the image to speed this up
#     center_chunk = img[h//4:h*3//4, w//4:w*3//4]
    
#     # Count how many unique colors are in this chunk
#     unique_colors = len(np.unique(center_chunk))
    
#     # If there are fewer than 20 unique gray colors, it's definitely a mask or a drawing.
#     if unique_colors < 20:
#         return False, "Binary Mask"

#     return True, "Valid"

# def purge_dataset():
#     print(f"Reading {INPUT_CSV}...")
#     try:
#         df = pd.read_csv(INPUT_CSV)
#     except:
#         print("CSV not found. Please verify the filename.")
#         return

#     clean_rows = []
#     removed_counts = {"File Not Found": 0, "Read Error": 0, "Corrupt Image": 0, 
#                       "Too Small": 0, "Binary Mask": 0}

#     print(f"Scanning {len(df)} images... This may take a minute.")

#     for index, row in df.iterrows():
#         # Get path
#         path = row['PATH']
        
#         # Check if it's valid
#         is_valid, reason = is_valid_mammogram(path)
        
#         if is_valid:
#             # Fix the path formatting before saving
#             row['PATH'] = path.replace("\\", "/")
#             clean_rows.append(row)
#         else:
#             # Track why we removed it (just for your info)
#             if "Too Small" in reason: removed_counts["Too Small"] += 1
#             elif "Binary Mask" in reason: removed_counts["Binary Mask"] += 1
#             elif "File Not Found" in reason: removed_counts["File Not Found"] += 1
#             else: removed_counts["Corrupt Image"] += 1
            
#             # Print the first few masks found so you know it's working
#             if "Binary Mask" in reason and removed_counts["Binary Mask"] < 5:
#                 print(f"Removing Mask: {path[-30:]}")

#     # Create new DataFrame
#     clean_df = pd.DataFrame(clean_rows)

#     # Save
#     clean_df.to_csv(OUTPUT_CSV, index=False)

#     print("\n--- PURGE COMPLETE ---")
#     print(f"Original Size: {len(df)}")
#     print(f"Clean Size:    {len(clean_df)}")
#     print("\n--- TRASH REPORT (What was removed) ---")
#     print(f"Masks (Black & White): {removed_counts['Binary Mask']}")
#     print(f"Tiny Crops:            {removed_counts['Too Small']}")
#     print(f"Missing Files:         {removed_counts['File Not Found']}")
    
#     print(f"\nSaved clean dataset to: {OUTPUT_CSV}")

# if __name__ == "__main__":
#     purge_dataset() 

import pandas as pd
import os

# --- CONFIGURATION ---
FIXED_CSV = "train_dataset_FIXED.csv"

# Names of the files we MUST find
TRAIN_FILE = "archive/csv/calc_case_description_train_set.csv"
TEST_FILE  = "archive/csv/calc_case_description_test_set.csv"

def extract_uid(path_str):
    """ Extracts Series UID from path (handles Windows/Linux slashes) """
    if pd.isna(path_str): return None
    clean_path = str(path_str).strip().replace('\\', '/')
    parts = clean_path.split('/')
    if len(parts) >= 2:
        return parts[-2]
    return None

def find_file_path(target_name):
    """ Searches current directory and subdirectories for the file """
    search_dirs = [".", "archive", "archive/csv", "csv"]
    for d in search_dirs:
        full_path = os.path.join(d, target_name)
        if os.path.exists(full_path):
            return full_path
    return None

def link_data():
    print("--- 1. LOADING IMAGES ---")
    if not os.path.exists(FIXED_CSV):
        print(f"Error: {FIXED_CSV} not found.")
        return
    df_images = pd.read_csv(FIXED_CSV)
    print(f"Loaded {len(df_images)} images to link.")

    print("\n--- 2. LOADING MEDICAL RECORDS ---")
    
    # 1. Find Train Set (CRITICAL)
    train_path = find_file_path(TRAIN_FILE)
    if train_path:
        print(f"[OK] Found Train Data: {train_path}")
        df_train = pd.read_csv(train_path)
        print(f"     -> Loaded {len(df_train)} rows.")
    else:
        print(f"[ERROR] Could not find '{TRAIN_FILE}'.")
        print("Please ensure this file is in your folder!")
        return # Stop here because we need this file

    # 2. Find Test Set (Optional but good)
    test_path = find_file_path(TEST_FILE)
    if test_path:
        print(f"[OK] Found Test Data:  {test_path}")
        df_test = pd.read_csv(test_path)
        print(f"     -> Loaded {len(df_test)} rows.")
        df_desc = pd.concat([df_train, df_test], ignore_index=True)
    else:
        print("[WARN] Test set not found (Using only Train set).")
        df_desc = df_train

    print(f"Total Medical Records: {len(df_desc)}")

    print("\n--- 3. MERGING ---")
    # Generate Keys
    df_images['JOIN_KEY'] = df_images['CROP_PATH'].apply(extract_uid)
    df_desc['JOIN_KEY']   = df_desc['cropped image file path'].apply(extract_uid)

    # Prepare Columns
    cols_to_keep = [
        'JOIN_KEY', 
        'pathology', 'calc type', 'calc distribution', 
        'assessment', 'subtlety', 'breast density', 
        'left or right breast', 'image view'
    ]
    # Filter columns that exist
    cols_to_keep = [c for c in cols_to_keep if c in df_desc.columns]
    
    # Drop duplicates in description (keep 1 row per Series UID)
    df_desc_clean = df_desc[cols_to_keep].drop_duplicates(subset=['JOIN_KEY'])
    
    # Merge
    df_final = pd.merge(df_images, df_desc_clean, on='JOIN_KEY', how='left')
    
    # Check Success
    missing = df_final['pathology'].isna().sum()
    if missing == 0:
        print(f"[SUCCESS] All {len(df_final)} images linked successfully!")
    else:
        print(f"[WARNING] {missing} images are still missing diagnoses.")
        print("Check if your 'calc_case_description' files match the images.")

    # Save
    df_final.drop(columns=['JOIN_KEY'], inplace=True, errors='ignore')
    
    # Rename for clarity
    rename_map = {
        'pathology': 'LABEL',
        'calc type': 'CALC_TYPE',
        'calc distribution': 'DISTRIBUTION',
        'assessment': 'BIRADS',
        'subtlety': 'SUBTLETY',
        'breast density': 'DENSITY',
        'left or right breast': 'SIDE',
        'image view': 'VIEW'
    }
    df_final.rename(columns=rename_map, inplace=True)
    
    out_file = "train_dataset_FULL.csv"
    df_final.to_csv(out_file, index=False)
    print(f"\nSaved to: {out_file}")

if __name__ == "__main__":
    link_data()