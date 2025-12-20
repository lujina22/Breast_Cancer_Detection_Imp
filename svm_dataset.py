import pandas as pd
import numpy as np
import cv2
from Feature_Extraction import extract_mass_features, extract_mc_features, build_master_vector
from input import get_mass_data
from MCs import get_mc_data

def build_dataset(csv_path):
    df = pd.read_csv(csv_path)

    X = []
    y_stage1 = []  # Normal(0) vs Abnormal(1)
    y_stage2 = []  # Mass(0) vs MC(1) only for abnormal

    skipped = []

    for _, row in df.iterrows():
        path = row["PATH"]
        label = row["CLASS"].upper()

        # ---------- LOAD IMAGE ----------
        try:
            if path.endswith(".dcm"):
                import pydicom
                dcm = pydicom.dcmread(path)
                img = dcm.pixel_array.astype(np.float32)
            else:
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    skipped.append(path)
                    continue
        except:
            skipped.append(path)
            continue

        # ---------- MASS FEATURES ----------
        try:
            mass_region, clean_mask = get_mass_data(img)
            mass_f = extract_mass_features(mass_region, clean_mask)
        except:
            mass_f = pd.DataFrame([np.zeros(15)])

        # ---------- MC FEATURES ----------
        try:
            mc_contours, th = get_mc_data(img)
            mc_f = extract_mc_features(mc_contours, th)
        except:
            mc_f = [0,0,0,0,0]

        # ---------- MASTER VECTOR ----------
        fv = build_master_vector(mass_f, mc_f)
        X.append(fv)

        # ---------- LABELS ----------
        # Stage-1 (keep previous mapping for higher accuracy)
        if label == "NORM":
            y_stage1.append(0)
            y_stage2.append(-1)
        else:
            y_stage1.append(1)
            if label in ["MASS", "CIRC", "ARCH", "ASYM"]:
                y_stage2.append(0)
            elif label in ["CALC", "SPIC", "MISC"]:
                y_stage2.append(1)
            else:
                skipped.append(path)
                X.pop()
                continue

    if skipped:
        print(f"Skipped {len(skipped)} images due to errors or unknown labels")
    return np.array(X), np.array(y_stage1), np.array(y_stage2)
