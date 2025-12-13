import numpy as np
import cv2
import math

def extract_mass_features(mass_region, clean_mask):
    """
    Extract features from mass region.
    mass_region: grayscale region
    clean_mask: binary mask of largest bright region
    """
    # Default values
    mass_area = 0
    mass_perim = 0
    mass_circ = 0
    mass_mean = 0
    mass_max = 0
    mass_std = 0
    mass_texture = 0

    if mass_region is None or clean_mask is None:
        return [mass_area, mass_perim, mass_circ, mass_mean, mass_max, mass_std, mass_texture]

    mask = (clean_mask > 0).astype(np.uint8)
    cnts, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(cnts) > 0:
        cnt = max(cnts, key=cv2.contourArea)
        mass_area = cv2.contourArea(cnt)
        mass_perim = cv2.arcLength(cnt, True)
        if mass_perim > 0:
            mass_circ = (4 * np.pi * mass_area) / (mass_perim * mass_perim)

    mass_pixels = mass_region[mask == 1]
    if len(mass_pixels) > 0:
        mass_mean = float(np.mean(mass_pixels))
        mass_max  = float(np.max(mass_pixels))
        mass_std  = float(np.std(mass_pixels))
        lap = cv2.Laplacian(mass_region, cv2.CV_64F)
        mass_texture = float(np.mean(np.abs(lap[mask == 1])))

    return [mass_area, mass_perim, mass_circ, mass_mean, mass_max, mass_std, mass_texture]



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
    Combine both feature lists into final master vector
    """
    return mass_features + mc_features