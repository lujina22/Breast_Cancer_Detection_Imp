# test_master_vector.py
from Feature_Extraction import extract_mass_features, extract_mc_features, build_master_vector
import pydicom
import numpy as np
import cv2

from input import get_mass_data
from MCs import get_mc_data


dcm = pydicom.dcmread("images/1-1.dcm")
image_mass = dcm.pixel_array.astype(np.float32)


mc_image = cv2.imread(
    "archive/jpeg/1.3.6.1.4.1.9590.100.1.2.47180989312717299212637285692300339843/1-203.jpg",
    cv2.IMREAD_GRAYSCALE
)

# 1. Get inputs from separate files
# mass_region, clean_mask = get_mass_data(None)
# mc_contours, intensity_image = get_mc_data(mc_image)

mass_region, clean_mask = get_mass_data(image_mass)
mc_contours, intensity_image = get_mc_data(None)

# 2. Extract features
mass_features = extract_mass_features(mass_region, clean_mask)
mc_features   = extract_mc_features(mc_contours, intensity_image)

# 3. Build master vector
master_vector = build_master_vector(mass_features, mc_features)

# 4. Print and inspect
print("Mass features:", mass_features)
print("MC features:", mc_features)
print("Master vector:", master_vector)
print("Length of master vector:", len(master_vector))
