import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

# ================= CONFIGURATION =================
# Input: Path to the original MIAS Info.txt
INFO_FILE_PATH = 'Dataset/all-mias/Info.txt' 

# Output: Desired filenames for the new CSVs
OUTPUT_TRAIN_FILE = 'new_train_dataset.csv'
OUTPUT_TEST_FILE  = 'new_test_dataset.csv'
# =================================================

def parse_mias_info(file_path):
    """
    Parses the MIAS Info.txt file, handling all 322 entries including
    Normal cases and those with missing severity/coordinates.
    """
    data = []
    
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return pd.DataFrame()

    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            
            # Skip empty lines or lines that don't look like data (e.g. headers)
            if len(parts) < 3 or not parts[0].startswith('mdb'):
                continue
            
            # 1. Reference Number (e.g., mdb001)
            ref = parts[0]
            
            # 2. Tissue Character (F, G, D)
            tissue = parts[1] 
            
            # 3. Class of Abnormality (CALC, CIRC, SPIC, MISC, ARCH, ASYM, NORM)
            cls = parts[2]    
            
            # 4. Severity (B, M) - Present only if not NORM
            # 5-7. x, y, radius - Present only if not NORM
            
            if len(parts) == 3:
                # Normal case: mdb003 D NORM
                severity = np.nan
                x = np.nan
                y = np.nan
                radius = np.nan
            elif len(parts) >= 7:
                # Standard Abnormal case: mdb001 G CIRC B 535 425 197
                severity = parts[3]
                try:
                    x = float(parts[4])
                    y = float(parts[5])
                    radius = float(parts[6])
                except ValueError:
                    x, y, radius = np.nan, np.nan, np.nan
            else:
                # Edge cases (some lines in Info.txt might be formatted differently)
                severity = parts[3] if len(parts) > 3 else np.nan
                x, y, radius = np.nan, np.nan, np.nan

            data.append({
                'REF': ref,
                'TISSUE': tissue,
                'CLASS': cls,
                'SEVERITY': severity,
                'X': x,
                'Y': y,
                'RADIUS': radius
            })
            
    return pd.DataFrame(data)

def get_tumor_type(row_class):
    """Maps specific class to general TYPE (Normal, Mass, Microcalcification)."""
    if row_class == 'NORM':
        return 'NORMAL'
    elif row_class == 'CALC':
        return 'MICRO_CALCIFICATION'
    elif row_class in ['CIRC', 'SPIC', 'MISC', 'ARCH', 'ASYM']:
        return 'MASS'
    else:
        return 'MASS' 

# --- MAIN EXECUTION ---

# 1. Load Data
print(f"Reading from {INFO_FILE_PATH}...")
df = parse_mias_info(INFO_FILE_PATH)

if not df.empty:
    # 2. Create Derived Columns
    
    # TYPE column
    df['TYPE'] = df['CLASS'].apply(get_tumor_type)
    
    # PATH column (Dataset/all-mias/mdbXXX.pgm)
    df['PATH'] = df['REF'].apply(lambda x: f"Dataset/all-mias/{x}.pgm")
    
    total_records = len(df)
    print(f"Successfully loaded {total_records} records.")
    
    # 3. Stratified Split (80% Train, 20% Test)
    # Stratify by 'CLASS' ensures rare classes are distributed to both sets
    train_df, test_df = train_test_split(
        df, 
        test_size=0.2, 
        random_state=42, 
        stratify=df['CLASS']
    )
    
    print(f"Training samples: {len(train_df)}")
    print(f"Testing samples:  {len(test_df)}")
    
    # 4. Save with NEW filenames
    train_df.to_csv(OUTPUT_TRAIN_FILE, index=False)
    test_df.to_csv(OUTPUT_TEST_FILE, index=False)
    
    print(f"\nSuccess! Files saved as:")
    print(f" - {OUTPUT_TRAIN_FILE}")
    print(f" - {OUTPUT_TEST_FILE}")
    
    # Preview
    print("\nFirst 3 rows of new Training data:")
    print(train_df[['REF', 'CLASS', 'TYPE', 'PATH']].head(3))

else:
    print("Failed to load data. Please check if Info.txt exists in the correct path.")