# import pandas as pd
# import cv2
# import os
# import glob
# import numpy as np

# # --- CONFIGURATION ---
# USER_DATASET_CSV = "train_dataset_FINAL_CLEAN.csv"  # Your clean list of images
# METADATA_CSV = "archive/csv/calc_case_description_train_set.csv" # The Kaggle Answer Key
# IMAGE_ROOT_DIR = "archive/jpeg/" # Where all images/masks are stored

# def get_file_map(root_dir):
#     """
#     Creates a fast lookup dictionary for every file on your hard drive.
#     Key: The filename (e.g., '1-249.jpg'), Value: The full path.
#     This makes finding files instant, instead of searching every time.
#     """
#     print("Indexing all files in archive... (This takes 10 seconds)...")
#     file_map = {}
#     # Find ALL jpgs recursively
#     all_files = glob.glob(os.path.join(root_dir, "**", "*.jpg"), recursive=True)
    
#     for f in all_files:
#         # We store them by their folder name + filename to be specific
#         # e.g., key might be "1.3.6.1.4.1.9590.../1-249.jpg"
#         # But DDSM naming is messy, so let's try indexing by the unique folder UID
#         parts = f.replace("\\", "/").split("/")
#         if len(parts) > 2:
#              # Usually the long UID folder is unique enough
#              uid_folder = parts[-2] 
#              filename = parts[-1]
#              file_map[f"{uid_folder}/{filename}"] = f
             
#              # Also index just by filename as a backup
#              file_map[filename] = f
#     return file_map

# def fill_coordinates_from_meta():
#     print(f"Reading {USER_DATASET_CSV}...")
#     df_main = pd.read_csv(USER_DATASET_CSV)
    
#     print(f"Reading {METADATA_CSV}...")
#     try:
#         df_meta = pd.read_csv(METADATA_CSV)
#     except:
#         print("CRITICAL ERROR: Could not find the metadata CSV at 'archive/csv/'. check the path.")
#         return

#     # Create the file index for speed
#     file_index = get_file_map(IMAGE_ROOT_DIR)
    
#     fixed_count = 0
    
#     print("Linking Images to Masks...")

#     for index, row in df_main.iterrows():
#         if row['SOURCE'] == 'DDSM':
#             current_path = row['PATH'].replace("\\", "/")
            
#             # 1. FIND THIS IMAGE IN THE METADATA CSV
#             # The metadata paths end in .dcm. Our paths end in .jpg.
#             # We try to match based on the unique folder structure (UID)
            
#             # Extract the long UID folder from the current path
#             # Path: archive/jpeg/1.3.6.1.../1-154.jpg
#             path_parts = current_path.split("/")
#             if len(path_parts) < 2: continue
            
#             # The UID is usually the second to last part
#             img_uid_folder = path_parts[-2]
            
#             # Search the metadata CSV for a row containing this UID
#             # (The 'image file path' column in CSV has the UID)
#             match = df_meta[df_meta['image file path'].str.contains(img_uid_folder, na=False)]
            
#             if not match.empty:
#                 # Get the FIRST match (sometimes there are duplicates for crops, take the first)
#                 meta_row = match.iloc[0]
                
#                 # 2. GET THE MASK PATH FROM METADATA
#                 mask_dcm_path = meta_row['ROI mask file path']
#                 # This looks like: "DOI/1.3.6.../000001.dcm"
                
#                 # We need to find the corresponding JPG on your disk
#                 # It's usually in a folder with a similar UID
#                 mask_uid_folder = mask_dcm_path.split("/")[-2]
                
#                 # Try to find a file in our index that matches this Mask UID
#                 # We look for ANY jpg in that specific mask folder
#                 mask_real_path = None
                
#                 # Search our index keys for the mask folder
#                 for key, path in file_index.items():
#                     if mask_uid_folder in path:
#                         mask_real_path = path
#                         break
                
#                 if mask_real_path:
#                     # 3. MEASURE THE MASK
#                     mask = cv2.imread(mask_real_path, 0)
#                     if mask is not None:
#                         # Resize mask to match image if needed (DDSM quirk)
#                         # But usually for calculation we just need the raw numbers
                        
#                         contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#                         if contours:
#                             largest_cnt = max(contours, key=cv2.contourArea)
#                             (x, y), radius = cv2.minEnclosingCircle(largest_cnt)
                            
#                             # 4. UPDATE CSV
#                             df_main.at[index, 'X'] = int(x)
#                             df_main.at[index, 'Y'] = int(y)
#                             df_main.at[index, 'RADIUS'] = int(radius)
#                             fixed_count += 1
#                             print(f"Fixed: {current_path[-20:]} -> R={int(radius)}")
#                 else:
#                     print(f"Mask file missing for: {current_path[-15:]}")
#             else:
#                 print(f"Metadata not found for: {current_path[-15:]}")

#     # Remove rows that still have no coordinates (we can't use them)
#     df_final = df_main.dropna(subset=['X', 'Y', 'RADIUS'])
    
#     OUTPUT_FILE = "train_dataset_FINAL_WITH_COORDS.csv"
#     df_final.to_csv(OUTPUT_FILE, index=False)
    
#     print("\n--- COMPLETE ---")
#     print(f"Original DDSM images: {len(df_main[df_main['SOURCE']=='DDSM'])}")
#     print(f"Fixed with Coordinates: {fixed_count}")
#     print(f"Saved to: {OUTPUT_FILE}")

# if __name__ == "__main__":
#     fill_coordinates_from_meta()

# import pandas as pd
# import os

# # --- CONFIGURATION ---
# # Make sure this points to where you put the CSV you just uploaded
# DICOM_INFO_PATH = "archive/csv/dicom_info.csv" 
# OUTPUT_CSV = "train_dataset_CROPS.csv"

# def extract_crops():
#     print(f"Reading {DICOM_INFO_PATH}...")
    
#     if not os.path.exists(DICOM_INFO_PATH):
#         print("Error: File not found! Check the path.")
#         return

#     # 1. Load the huge info file
#     df = pd.read_csv(DICOM_INFO_PATH)

#     # 2. FILTER: We only want "cropped images" for "Calc" (Calcifications)
#     #    We ignore "Mass" and "full mammogram images"
#     print("Filtering for Calcification Crops...")
    
#     crops = df[ (df['SeriesDescription'] == 'cropped images') & 
#                 (df['PatientID'].str.contains('Calc', na=False)) ].copy()

#     print(f"Found {len(crops)} crop entries in the CSV.")

#     # 3. FIX PATHS
#     # The CSV says: "CBIS-DDSM/jpeg/..."
#     # You have:     "archive/jpeg/..."
#     # We simply swap the start of the string.
    
#     def fix_path(p):
#         # Swap the folder prefix
#         new_path = p.replace("CBIS-DDSM/jpeg", "archive/jpeg")
#         # Fix slashes for Windows
#         return new_path.replace("/", os.sep)

#     crops['PATH'] = crops['image_path'].apply(fix_path)

#     # 4. VERIFY ON DISK (Optional but recommended)
#     print("Verifying which files actually exist on your disk...")
    
#     valid_data = []
#     found_count = 0
    
#     for idx, row in crops.iterrows():
#         full_path = row['PATH']
        
#         # Check if file is really there
#         if os.path.exists(full_path):
#             valid_data.append({
#                 "PATH": full_path,
#                 "TYPE": "MICRO_CALCIFICATION",
#                 "SOURCE": "DDSM_CROP",
#                 "PATIENT_ID": row['PatientID']
#             })
#             found_count += 1
#             if found_count % 500 == 0:
#                 print(f"Verified {found_count} images...")
        
#     # 5. SAVE
#     final_df = pd.DataFrame(valid_data)
#     final_df.to_csv(OUTPUT_CSV, index=False)
    
#     print(f"\n--- SUCCESS ---")
#     print(f"Verified & Saved {len(final_df)} crop images to: {OUTPUT_CSV}")
#     print("These are the zoomed-in ROI images you asked for.")

# if __name__ == "__main__":
#     extract_crops() 
# import pandas as pd
# import os

# # --- CONFIGURATION ---
# CROPS_CSV = "train_dataset_CROPS.csv"           # Your list of photos
# DICOM_INFO = "archive/csv/dicom_info.csv"       # The map you uploaded
# OUTPUT_CSV = "train_dataset_READY_TO_TEST.csv"  # The final list with answers

# def pair_crops_and_masks():
#     print(f"Reading {CROPS_CSV}...")
#     try:
#         df_crops = pd.read_csv(CROPS_CSV)
#     except FileNotFoundError:
#         print("Error: Could not find train_dataset_CROPS.csv")
#         return

#     print(f"Reading {DICOM_INFO}...")
#     try:
#         df_info = pd.read_csv(DICOM_INFO)
#     except FileNotFoundError:
#         print("Error: Could not find dicom_info.csv. Check path 'archive/csv/'")
#         return

#     # 1. Create a Dictionary: { SeriesUID : Path_to_Mask }
#     # We look for rows where SeriesDescription is 'ROI mask images'
#     print("Indexing Masks from dicom_info...")
#     mask_rows = df_info[df_info['SeriesDescription'] == 'ROI mask images']
    
#     # Map the Folder UID -> The Mask File Path
#     # (Because the Crop and the Mask share the same Folder UID)
#     uid_to_mask_path = pd.Series(mask_rows.image_path.values, index=mask_rows.SeriesInstanceUID).to_dict()
    
#     print(f"Found {len(uid_to_mask_path)} mask definitions.")

#     # 2. Match them to your Crops
#     print("Pairing Crops with Masks...")
    
#     found_count = 0
    
#     def find_mask_for_row(crop_path):
#         nonlocal found_count
#         # Extract the Folder UID from the crop path
#         # Path: archive/jpeg/1.3.6.1...UID.../1-123.jpg
#         parts = crop_path.replace('\\', '/').split('/')
#         if len(parts) < 2: return None
        
#         series_uid = parts[-2] # The long folder name
        
#         if series_uid in uid_to_mask_path:
#             # Get the mask path from the dictionary
#             original_mask_path = uid_to_mask_path[series_uid]
            
#             # Convert 'CBIS-DDSM/jpeg' to 'archive/jpeg' to match your PC
#             local_mask_path = original_mask_path.replace("CBIS-DDSM/jpeg", "archive/jpeg")
#             local_mask_path = local_mask_path.replace("/", os.sep) # Fix slashes
            
#             found_count += 1
#             return local_mask_path
#         return None

#     df_crops['MASK_PATH'] = df_crops['PATH'].apply(find_mask_for_row)

#     # 3. Filter and Save
#     # We only keep rows where we successfully found a mask
#     df_final = df_crops.dropna(subset=['MASK_PATH'])
    
#     df_final.to_csv(OUTPUT_CSV, index=False)
    
#     print("\n--- DONE ---")
#     print(f"Paired {found_count} images with their masks.")
#     print(f"Saved to: {OUTPUT_CSV}")
#     print("You now have the Answer Key for every image!")

# if __name__ == "__main__":
#     pair_crops_and_masks() 

# import pandas as pd
# import os

# # --- CONFIGURATION ---
# DICOM_INFO = "archive\csv\dicom_info.csv" # Ensure this file is in your folder
# TRAIN_DESC = "archive\csv\calc_case_description_train_set.csv"
# IMAGE_ROOT = "archive/jpeg"   # Your image folder

# def fix_pairing_paths():
#     print("--- 1. LOADING AND SPLITTING DICOM INFO ---")
#     if not os.path.exists(DICOM_INFO):
#         print(f"Error: {DICOM_INFO} not found.")
#         return
        
#     df_info = pd.read_csv(DICOM_INFO)
    
#     # CRITICAL FIX: Create TWO separate maps to avoid overwriting duplicates
#     # Map 1: UID -> Crop Image
#     df_crops = df_info[df_info['SeriesDescription'].str.contains('cropped images', na=False, case=False)]
#     crop_map = pd.Series(df_crops.image_path.values, index=df_crops.SeriesInstanceUID).to_dict()
    
#     # Map 2: UID -> Mask Image
#     df_masks = df_info[df_info['SeriesDescription'].str.contains('ROI mask images', na=False, case=False)]
#     mask_map = pd.Series(df_masks.image_path.values, index=df_masks.SeriesInstanceUID).to_dict()
    
#     print(f"Indexed {len(crop_map)} crops and {len(mask_map)} masks.")

#     print("--- 2. PAIRING FROM MASTER LIST ---")
#     if not os.path.exists(TRAIN_DESC):
#         print(f"Error: {TRAIN_DESC} not found.")
#         return

#     df_calc = pd.read_csv(TRAIN_DESC)
#     paired_data = []
    
#     for idx, row in df_calc.iterrows():
#         # Get paths from the Master CSV (and clean them)
#         crop_desc_path = str(row['cropped image file path']).strip()
#         mask_desc_path = str(row['ROI mask file path']).strip()
        
#         try:
#             # Extract the Series UID (The folder name, 2nd from the end)
#             # Path format: Folder/StudyUID/SeriesUID/File.dcm
#             crop_uid = crop_desc_path.replace('\\', '/').split('/')[-2]
#             mask_uid = mask_desc_path.replace('\\', '/').split('/')[-2]
            
#             # Lookup in the SEPARATE maps
#             # This ensures we get the Crop JPG for the crop and Mask JPG for the mask
#             real_crop_path = crop_map.get(crop_uid)
#             real_mask_path = mask_map.get(mask_uid)
            
#             if real_crop_path and real_mask_path:
#                 # Fix path for your computer
#                 local_crop = real_crop_path.replace("CBIS-DDSM/jpeg", IMAGE_ROOT).replace("/", os.sep)
#                 local_mask = real_mask_path.replace("CBIS-DDSM/jpeg", IMAGE_ROOT).replace("/", os.sep)
                
#                 paired_data.append({
#                     'PATIENT_ID': row['patient_id'],
#                     'PATHOLOGY': row['pathology'],
#                     'CROP_PATH': local_crop,
#                     'MASK_PATH': local_mask
#                 })
#         except Exception as e:
#             continue
            
#     # Save the Fixed CSV
#     out_file = "train_dataset_CORRECTED.csv"
#     pd.DataFrame(paired_data).to_csv(out_file, index=False)
    
#     print("\n--- SUCCESS ---")
#     print(f"Correctly paired {len(paired_data)} images.")
#     print(f"Saved to: {out_file}")
#     print("Open this file and check: CROP_PATH and MASK_PATH should now be DIFFERENT files.")

# if __name__ == "__main__":
#     fix_pairing_paths()

import pandas as pd
import numpy as np
import os
import cv2

# --- CONFIGURATION ---
DICOM_INFO = "archive\csv\dicom_info.csv" 
TRAIN_DESC = "archive\csv\calc_case_description_train_set.csv"
IMAGE_ROOT = "archive/jpeg"
OUTPUT_CSV = "train_dataset_FIXED.csv"
MASK_SAVE_DIR = "archive/jpeg/cropped_masks" # Where to save the new masks

def fix_dataset():
    print("--- 1. LOADING MAPS ---")
    if not os.path.exists(DICOM_INFO):
        print(f"Error: {DICOM_INFO} not found.")
        return
        
    df_info = pd.read_csv(DICOM_INFO)
    
    # Create separate lookups to avoid ID collisions
    # Crop Map: UID -> Path
    crops = df_info[df_info['SeriesDescription'].str.contains('cropped images', na=False, case=False)]
    crop_map = pd.Series(crops.image_path.values, index=crops.SeriesInstanceUID).to_dict()
    
    # Mask Map: UID -> Path
    masks = df_info[df_info['SeriesDescription'].str.contains('ROI mask images', na=False, case=False)]
    mask_map = pd.Series(masks.image_path.values, index=masks.SeriesInstanceUID).to_dict()
    
    print(f"Indexed {len(crop_map)} crops and {len(mask_map)} full masks.")

    if not os.path.exists(MASK_SAVE_DIR):
        os.makedirs(MASK_SAVE_DIR)

    print("--- 2. GENERATING MATCHING MASKS ---")
    df_calc = pd.read_csv(TRAIN_DESC)
    fixed_rows = []
    
    for idx, row in df_calc.iterrows():
        # Clean paths
        c_desc = str(row['cropped image file path']).strip()
        m_desc = str(row['ROI mask file path']).strip()
        
        try:
            # Extract UIDs
            c_uid = c_desc.replace('\\', '/').split('/')[-2]
            m_uid = m_desc.replace('\\', '/').split('/')[-2]
            
            # Lookup Files
            if c_uid in crop_map and m_uid in mask_map:
                real_c_path = crop_map[c_uid].replace("CBIS-DDSM/jpeg", IMAGE_ROOT).replace("/", os.sep)
                real_m_path = mask_map[m_uid].replace("CBIS-DDSM/jpeg", IMAGE_ROOT).replace("/", os.sep)
                
                if os.path.exists(real_c_path) and os.path.exists(real_m_path):
                    
                    # 3. LOAD IMAGES
                    img_crop = cv2.imread(real_c_path, 0) # Load grayscale
                    img_full_mask = cv2.imread(real_m_path, 0)
                    
                    if img_crop is None or img_full_mask is None: continue
                    
                    # 4. CHECK SIZES
                    h_c, w_c = img_crop.shape
                    h_m, w_m = img_full_mask.shape
                    
                    final_mask_path = real_m_path # Default to original if sizes match
                    
                    # If sizes differ (Mask is huge), we must CROP THE MASK
                    if h_m > h_c + 50 or w_m > w_c + 50:
                        # Find the Tumor in the Full Mask
                        # (The white blobs)
                        points = cv2.findNonZero(img_full_mask)
                        
                        if points is not None:
                            # Get bounding box of the tumor
                            x, y, w, h = cv2.boundingRect(points)
                            center_x = x + w // 2
                            center_y = y + h // 2
                            
                            # Calculate crop coordinates centered on the tumor
                            # We want the mask crop to be size (w_c, h_c)
                            x1 = max(0, center_x - w_c // 2)
                            y1 = max(0, center_y - h_c // 2)
                            x2 = min(w_m, x1 + w_c)
                            y2 = min(h_m, y1 + h_c)
                            
                            # Adjust if we hit borders
                            if x2 - x1 < w_c: x1 = max(0, x2 - w_c)
                            if y2 - y1 < h_c: y1 = max(0, y2 - h_c)
                            
                            # Perform Crop
                            new_mask = img_full_mask[y1:y2, x1:x2]
                            
                            # Resize to exact match (in case of 1-2px off)
                            if new_mask.shape != (h_c, w_c):
                                new_mask = cv2.resize(new_mask, (w_c, h_c), interpolation=cv2.INTER_NEAREST)
                            
                            # Save New Mask
                            new_name = f"{row['patient_id']}_{idx}_mask.jpg"
                            final_mask_path = os.path.join(MASK_SAVE_DIR, new_name)
                            cv2.imwrite(final_mask_path, new_mask)
                            
                        else:
                            # Mask is empty (Black), cannot crop. Skip.
                            continue
                            
                    fixed_rows.append({
                        'PATIENT_ID': row['patient_id'],
                        'CROP_PATH': real_c_path,
                        'MASK_PATH': final_mask_path
                    })
                    
                    if len(fixed_rows) % 100 == 0:
                        print(f"Fixed {len(fixed_rows)} masks...")

        except Exception as e:
            continue

    # 5. SAVE CSV
    df_out = pd.DataFrame(fixed_rows)
    df_out.to_csv(OUTPUT_CSV, index=False)
    
    print("\n--- DONE ---")
    print(f"Generated {len(df_out)} perfectly paired images.")
    print(f"New CSV: {OUTPUT_CSV}")
    print("Use this file for your testing.")

if __name__ == "__main__":
    fix_dataset()