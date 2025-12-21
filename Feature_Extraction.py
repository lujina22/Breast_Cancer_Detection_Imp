import numpy as np
import cv2
import pandas as pd
from skimage.measure import label, regionprops
from skimage.feature import graycomatrix, graycoprops


def extract_mass_features(original_img, mass_mask):
    """
    original_img : full grayscale mammogram
    mass_mask    : binary mask of detected mass
    """

    features = {
        'Area': 0, 'Perimeter': 0, 'Circularity': 0,
        'Eccentricity': 0, 'Solidity': 0, 'Extent': 0,
        'Mean_Intensity': 0, 'Max_Intensity': 0, 'Std_Intensity': 0,
        'Contrast': 0, 'Dissimilarity': 0, 'Homogeneity': 0,
        'Energy': 0, 'Correlation': 0, 'ASM': 0
    }

    if np.count_nonzero(mass_mask) == 0:
        return pd.DataFrame([features])

    # ---------- Shape (OpenCV) ----------
    cnts, _ = cv2.findContours(mass_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(cnts, key=cv2.contourArea)

    area = cv2.contourArea(cnt)
    perim = cv2.arcLength(cnt, True)

    features['Area'] = area
    features['Perimeter'] = perim
    features['Circularity'] = (4 * np.pi * area) / (perim ** 2 + 1e-6)

    # ---------- Intensity ----------
    pixels = original_img[mass_mask > 0]
    features['Mean_Intensity'] = float(np.mean(pixels))
    features['Max_Intensity'] = float(np.max(pixels))
    features['Std_Intensity'] = float(np.std(pixels))

    # ---------- Regionprops ----------
    lbl = label(mass_mask)
    region = max(regionprops(lbl), key=lambda r: r.area)

    features['Eccentricity'] = region.eccentricity
    features['Solidity'] = region.solidity
    features['Extent'] = region.extent

    # ---------- GLCM (ROI-based) ----------
    y0, x0, y1, x1 = region.bbox
    roi = original_img[y0:y1, x0:x1]
    roi_mask = mass_mask[y0:y1, x0:x1]

    roi = roi[roi_mask > 0]
    roi = cv2.normalize(roi, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    glcm = graycomatrix(
        roi.reshape(-1, 1),
        distances=[1],
        angles=[0],
        levels=256,
        symmetric=True,
        normed=True
    )

    features['Contrast'] = graycoprops(glcm, 'contrast')[0, 0]
    features['Dissimilarity'] = graycoprops(glcm, 'dissimilarity')[0, 0]
    features['Homogeneity'] = graycoprops(glcm, 'homogeneity')[0, 0]
    features['Energy'] = graycoprops(glcm, 'energy')[0, 0]
    features['Correlation'] = graycoprops(glcm, 'correlation')[0, 0]
    features['ASM'] = graycoprops(glcm, 'ASM')[0, 0]

    return pd.DataFrame([features])


def extract_mc_features(mc_contours, tophat_norm):
    """
    Extract microcalcification features safely
    """

    # ---------- DEFAULTS (IMPORTANT) ----------
    mc_count = 0
    mc_avg_area = 0.0
    mc_std_area = 0.0
    mc_density  = 0.0
    mc_mean_int = 0.0

    # ---------- VALIDITY CHECK ----------
    if mc_contours is None or tophat_norm is None:
        return [mc_count, mc_avg_area, mc_std_area, mc_density, mc_mean_int]

    mc_count = len(mc_contours)
    if mc_count == 0:
        return [mc_count, mc_avg_area, mc_std_area, mc_density, mc_mean_int]

    # ---------- AREA FEATURES ----------
    areas = [cv2.contourArea(c) for c in mc_contours]
    mc_avg_area = float(np.mean(areas))
    mc_std_area = float(np.std(areas))

    # ---------- DENSITY ----------
    if mc_count > 1:
        xs, ys = [], []
        for c in mc_contours:
            x, y, w, h = cv2.boundingRect(c)
            xs.extend([x, x + w])
            ys.extend([y, y + h])

        cluster_area = (max(xs) - min(xs)) * (max(ys) - min(ys)) + 1e-6
        mc_density = mc_count / cluster_area
    else:
        mc_density = 0.0

    # ---------- INTENSITY ----------
    mask = np.zeros_like(tophat_norm, dtype=np.uint8)
    for c in mc_contours:
        cv2.drawContours(mask, [c], -1, 255, -1)

    vals = tophat_norm[mask == 255]
    if len(vals) > 0:
        mc_mean_int = float(np.mean(vals))

    return [
        mc_count,
        mc_avg_area,
        mc_std_area,
        mc_density,
        mc_mean_int
    ]
 


def build_master_vector(mass_features, mc_features):
    """
    Combine mass + microcalcification features into one vector
    """

    # Convert mass features DataFrame → list
    mass_vector = mass_features.iloc[0].values.tolist()

    # mc_features is already a list
    master_vector = mass_vector + mc_features

    return master_vector
