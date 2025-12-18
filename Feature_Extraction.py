import numpy as np
import cv2
import math

import numpy as np
import cv2
import pandas as pd
from skimage.measure import label, regionprops
from skimage.feature import graycomatrix, graycoprops

def extract_mass_features(mass_region_gray, clean_mask):
    """
    Extracts comprehensive shape and texture features from the mass region.
    Combines basic OpenCV metrics with advanced GLCM texture analysis.
    
    mass_region_gray: Grayscale image of the ROI (or full image).
    clean_mask: Binary mask of the specific mass (largest bright region).
    """
    
    # Initialize dictionary for features
    features = {
        'Area': 0, 'Perimeter': 0, 'Circularity': 0,
        'Eccentricity': 0, 'Solidity': 0, 'Extent': 0,
        'Mean_Intensity': 0, 'Max_Intensity': 0, 'Std_Intensity': 0,
        'Contrast': 0, 'Dissimilarity': 0, 'Homogeneity': 0,
        'Energy': 0, 'Correlation': 0, 'ASM': 0
    }

    if mass_region_gray is None or clean_mask is None or np.count_nonzero(clean_mask) == 0:
        return pd.DataFrame([features])

    # --- 1. OpenCV Shape Features (Fast & Simple) ---
    cnts, _ = cv2.findContours(clean_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        cnt = max(cnts, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        perim = cv2.arcLength(cnt, True)
        
        features['Area'] = area
        features['Perimeter'] = perim
        if perim > 0:
            features['Circularity'] = (4 * np.pi * area) / (perim ** 2)

    # --- 2. Intensity Stats ---
    # Mask the original image to get pixels only within the tumor
    mass_pixels = mass_region_gray[clean_mask > 0]
    if len(mass_pixels) > 0:
        features['Mean_Intensity'] = float(np.mean(mass_pixels))
        features['Max_Intensity'] = float(np.max(mass_pixels))
        features['Std_Intensity'] = float(np.std(mass_pixels))

    # --- 3. Advanced Shape & GLCM Texture (skimage) ---
    # We label the mask to treat it as a region
    label_img = label(clean_mask)
    regions = regionprops(label_img, intensity_image=mass_region_gray)

    if regions:
        # Get the largest region (the tumor)
        region = max(regions, key=lambda r: r.area)

        # Advanced Shape
        features['Eccentricity'] = region.eccentricity
        features['Solidity'] = region.solidity
        features['Extent'] = region.extent

        # GLCM Texture Analysis
        # Extract the specific ROI for the GLCM (faster than processing whole image)
        roi = region.intensity_image
        
        # Normalize ROI to 0-255 uint8 for GLCM
        if roi.max() > 0:
            roi = (roi * (255.0 / roi.max())).astype(np.uint8)
        else:
            roi = roi.astype(np.uint8)

        # Calculate GLCM (1 pixel distance, 0 degrees)
        glcm = graycomatrix(roi, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)

        features['Contrast'] = graycoprops(glcm, 'contrast')[0, 0]
        features['Dissimilarity'] = graycoprops(glcm, 'dissimilarity')[0, 0]
        features['Homogeneity'] = graycoprops(glcm, 'homogeneity')[0, 0]
        features['Energy'] = graycoprops(glcm, 'energy')[0, 0]
        features['Correlation'] = graycoprops(glcm, 'correlation')[0, 0]
        features['ASM'] = graycoprops(glcm, 'ASM')[0, 0]

    return pd.DataFrame([features])



def extract_mc_features(mc_contours, tophat_norm):
    
    """
    mc_contours: list of contours for MCs
    intensity_image: your tophat_norm or clahe image
    """
    mc_count = len(mc_contours)
    
    
    
    if mc_count == 0:
        # default values
        mc_avg_area = 0
        mc_std_area = 0
        mc_density  = 0
        mc_mean_int = 0
    
    if mc_count > 0:
        areas = [cv2.contourArea(c) for c in mc_contours]
        mc_avg_area = float(np.mean(areas))
        mc_std_area = float(np.std(areas))
        
        
        # --- cluster density ---
        if mc_count > 1:
            xs, ys = [], []
            for c in mc_contours:
                x, y, w, h = cv2.boundingRect(c)
                xs.extend([x, x + w])
                ys.extend([y, y + h])

            cluster_area = (max(xs)-min(xs)) * (max(ys)-min(ys))
            mc_density = mc_count / (cluster_area + 1e-6)


        # --- average intensity of MCs ---
        mask = np.zeros_like(tophat_norm, dtype=np.uint8)
        for c in mc_contours:
            cv2.drawContours(mask, [c], -1, 255, -1)

        vals = tophat_norm[mask == 255]
        if len(vals) > 0:
            mc_mean_int = float(np.mean(vals))
            
    
    mc_feature_vector = [
        mc_count,
        mc_avg_area,
        mc_std_area,
        mc_density,
        mc_mean_int
    ]

    return mc_feature_vector        


def build_master_vector(mass_features, mc_features):
    """
    Combine mass + microcalcification features into one vector
    """

    # Convert mass features DataFrame → list
    mass_vector = mass_features.iloc[0].values.tolist()

    # mc_features is already a list
    master_vector = mass_vector + mc_features

    return master_vector
