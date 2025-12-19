"""
MASS Detection Test Script with IoU and Centroid Hit Evaluation
==================================================================
Tests tumor detection pipeline from Input.ipynb against all MASS images.

Key Features:
1. Circle-to-Circle IoU calculation
2. Centroid hit detection (inside ground truth circle)
3. MIAS coordinate system correction (Y-axis flip: height - y)
4. CSO (Cuckoo Search Optimization) support for parameter optimization

Usage:
    # Use default parameters:
    python test_mass_iou.py --csv train_dataset.csv --image-dir all-mias
    
    # Use pre-optimized parameters:
    python test_mass_iou.py --params-file optimized_params.json
    
    # Run CSO optimization on sample image first:
    python test_mass_iou.py --use-cso --cso-sample mdb267.pgm
"""

import cv2
import numpy as np
import pandas as pd
from skimage import io
from skimage.measure import label, regionprops
from pathlib import Path
import os
import argparse
import json
import joblib
from typing import Dict, Tuple, Optional, List

# Progress bar (optional)
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(iterable, **kwargs):
        return iterable


def detect_tumor_from_notebook(img_path: str, params: Optional[Dict] = None, use_ml: bool = False, ml_model_path: Optional[str] = None) -> Dict:
    """
    Complete tumor detection pipeline extracted from Input.ipynb.
    
    Pipeline Steps:
    1. Load and normalize image
    2. Gaussian blur
    3. Isolate breast region
    4. Histogram equalization (breast only)
    5. CLAHE enhancement
    6. K-means clustering (K=3)
    7. Morphological cleanup
    8. Position-based filtering
    9. Multi-criteria tumor selection
    
    Parameters:
    -----------
    img_path : str
        Path to image file
    params : dict, optional
        Detection parameters. If None, uses defaults.
    use_ml : bool
        If True, use ML classifier for region selection
    ml_model_path : str, optional
        Path to trained ML model (.pkl file)
        
    Returns:
    --------
    dict
        Detection results with keys:
        - 'detected': bool
        - 'centroid': tuple (x, y) or None
        - 'final_mask': np.ndarray or None
        - 'error': str or None
    """
    # Set default parameters if not provided
    if params is None:
        params = {
            'breast_thresh': 30,
            'clahe_clip': 3.0,
            'k_means_k': 3,
            'erosion_iter': 10,
            'y_ratio_thresh': 0.25,
            'extent_thresh': 0.2,
            'border_margin': 50,
            'size_weight': 1.5,
            'compact_weight': 1.2,
            'solidity_weight': 1.0,
            'position_weight': 1.0,
            'intensity_weight': 0.8
        }
    
    try:
        # ========== STEP 1: Load and Normalize ==========
        img = io.imread(img_path)
        img_norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        img_uint8 = np.uint8(img_norm)
        
        # ========== STEP 2: Gaussian Blur ==========
        img_blur = cv2.GaussianBlur(img_uint8, (5, 5), 0)
        
        # ========== STEP 3: Isolate Breast Region ==========
        _, thresh_breast = cv2.threshold(img_blur, params['breast_thresh'], 255, cv2.THRESH_BINARY)
        
        # Use connected components to get the largest region (breast)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(thresh_breast, connectivity=8)
        if num_labels < 2:
            return {'detected': False, 'centroid': None, 'final_mask': None, 'error': 'No breast region found'}
        
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        breast_mask = np.zeros_like(thresh_breast)
        breast_mask[labels == largest_label] = 255
        
        # Refine breast mask
        kernel_mask = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        breast_mask = cv2.morphologyEx(breast_mask, cv2.MORPH_CLOSE, kernel_mask)
        breast_mask = cv2.morphologyEx(breast_mask, cv2.MORPH_OPEN, kernel_mask)
        
        breast_region = cv2.bitwise_and(img_blur, img_blur, mask=breast_mask)
        
        # ========== STEP 4: Histogram Equalization (Breast Only) ==========
        breast_only = breast_region[breast_mask > 0]
        if len(breast_only) == 0:
            return {'detected': False, 'centroid': None, 'final_mask': None, 'error': 'No breast pixels'}
        
        hist, _ = np.histogram(breast_only.flatten(), 256, [0, 256])
        cdf = hist.cumsum()
        cdf_normalized = (cdf * 255) / cdf[-1] if cdf[-1] > 0 else cdf
        
        img_eq = img_blur.copy()
        img_eq[breast_mask > 0] = cdf_normalized[breast_region[breast_mask > 0]]
        
        # ========== STEP 5: CLAHE Enhancement ==========
        clahe = cv2.createCLAHE(clipLimit=params['clahe_clip'], tileGridSize=(8, 8))
        masked_img = cv2.bitwise_and(img_eq, img_eq, mask=breast_mask)
        enhanced_img = clahe.apply(masked_img)
        
        # ========== STEP 6: K-Means Clustering ==========
        pixel_values = enhanced_img.reshape((-1, 1)).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        K = int(params['k_means_k'])  # Background, Tissue, Dense Mass
        
        _, kmeans_labels, centers = cv2.kmeans(
            pixel_values, K, None, criteria, 10, cv2.KMEANS_PP_CENTERS
        )
        
        centers = np.uint8(centers)
        segmented = centers[kmeans_labels.flatten()]
        segmented = segmented.reshape(masked_img.shape)
        
        # Select brightest cluster (likely contains tumor)
        bright_cluster_idx = np.argmax(centers)
        candidate_mask = (segmented == centers[bright_cluster_idx]).astype(np.uint8) * 255
        
        # ========== STEP 7: Morphological Cleanup ==========
        kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        candidate_mask = cv2.morphologyEx(candidate_mask, cv2.MORPH_OPEN, kernel_clean)
        candidate_mask = cv2.morphologyEx(candidate_mask, cv2.MORPH_CLOSE, kernel_clean)
        candidate_mask = cv2.erode(candidate_mask, kernel_clean, iterations=int(params['erosion_iter']))
        
        # ========== STEP 8: Position-Based Filtering ==========
        contours, _ = cv2.findContours(candidate_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        final_tumor_mask = np.zeros_like(candidate_mask)
        height, width = candidate_mask.shape
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 500:  # Filter small noise
                continue
            
            x, y, w, h = cv2.boundingRect(cnt)
            cy = y + h // 2
            extent = area / (w * h) if (w * h) > 0 else 0
            y_ratio = cy / height
            
            # Filter by position and shape
            if y_ratio > params['y_ratio_thresh'] and extent > params['extent_thresh']:
                cv2.drawContours(final_tumor_mask, [cnt], -1, 255, -1)
        
        # ========== STEP 9: Multi-Criteria Tumor Selection ==========
        label_img = label(final_tumor_mask)
        regions = regionprops(label_img, intensity_image=enhanced_img)
        
        border_margin = int(params['border_margin'])
        valid_regions = []
        
        # ========== ML-BASED SELECTION ==========
        if use_ml and ml_model_path and os.path.exists(ml_model_path):
            try:
                # Load ML model
                ml_classifier = joblib.load(ml_model_path)
                
                # Extract features for all valid regions
                from ml_region_classifier import extract_region_features
                
                candidates_with_features = []
                for region in regions:
                    min_row, min_col, max_row, max_col = region.bbox
                    
                    # Check if region is away from borders
                    if (min_row > border_margin and min_col > border_margin and
                        max_row < (height - border_margin) and max_col < (width - border_margin)):
                        
                        features = extract_region_features(region, enhanced_img, (height, width))
                        candidates_with_features.append((region, features))
                
                if candidates_with_features:
                    # Prepare features for prediction
                    feature_dicts = [f for _, f in candidates_with_features]
                    feature_df = pd.DataFrame(feature_dicts)
                    
                    # Remove non-feature columns if present
                    feature_cols = [col for col in feature_df.columns 
                                  if col not in ['centroid_x', 'centroid_y', 'label', 'image_ref', 'distance_to_gt']]
                    X = feature_df[feature_cols]
                    
                    # Predict probabilities
                    y_proba = ml_classifier.predict_proba(X)[:, 1]  # Probability of being tumor
                    
                    # Select region with highest tumor probability
                    best_idx = np.argmax(y_proba)
                    selected_region = candidates_with_features[best_idx][0]
                    reg_y, reg_x = selected_region.centroid
                    
                    # Create final mask
                    final_mask = np.zeros_like(final_tumor_mask)
                    for r, c in selected_region.coords:
                        final_mask[r, c] = 255
                    
                    return {
                        'detected': True,
                        'centroid': (reg_x, reg_y),  # (x, y) format
                        'final_mask': final_mask,
                        'error': None
                    }
                else:
                    return {'detected': False, 'centroid': None, 'final_mask': None, 'error': 'No valid regions for ML'}
                    
            except Exception as e:
                print(f"[WARNING] ML prediction failed: {str(e)}. Falling back to rule-based selection.")
                # Fall through to rule-based selection
        
        # ========== RULE-BASED SELECTION (DEFAULT) ==========
        for region in regions:
            min_row, min_col, max_row, max_col = region.bbox
            
            # Check if region is away from borders
            if (min_row > border_margin and min_col > border_margin and
                max_row < (height - border_margin) and max_col < (width - border_margin)):
                
                area = region.area
                region_y, region_x = region.centroid  # Note: (row, col) = (y, x)
                eccentricity = region.eccentricity
                solidity = region.solidity
                mean_intensity = region.mean_intensity
                y_ratio = region_y / height
                
                # Multi-criteria scoring
                # Size score: prefer 1000-8000 pixels
                if 1000 <= area <= 8000:
                    size_score = 1.5
                elif area < 500:
                    size_score = 0.3
                elif area > 20000:
                    size_score = 0.5
                else:
                    size_score = 1.0
                
                # Compactness score: prefer more circular shapes
                compactness_score = 1.0 - (eccentricity * 0.5)
                
                # Solidity score
                solidity_score = solidity
                
                # Position score: prefer middle regions
                if 0.3 <= y_ratio <= 0.7:
                    position_score = 1.3
                elif y_ratio < 0.2 or y_ratio > 0.8:
                    position_score = 0.5
                else:
                    position_score = 1.0
                
                # Intensity score: prefer brighter regions
                intensity_score = mean_intensity / 255.0
                
                # Weighted total score
                total_score = (size_score * params['size_weight'] + 
                              compactness_score * params['compact_weight'] + 
                              solidity_score * params['solidity_weight'] + 
                              position_score * params['position_weight'] + 
                              intensity_score * params['intensity_weight'])
                
                valid_regions.append((region, total_score, region_x, region_y))
        
        # Select highest scoring region
        if valid_regions:
            valid_regions.sort(key=lambda x: x[1], reverse=True)
            selected_region, score, reg_x, reg_y = valid_regions[0]
            
            # Create final mask
            final_mask = np.zeros_like(final_tumor_mask)
            for r, c in selected_region.coords:
                final_mask[r, c] = 255
            
            return {
                'detected': True,
                'centroid': (reg_x, reg_y),  # (x, y) format
                'final_mask': final_mask,
                'error': None
            }
        else:
            return {'detected': False, 'centroid': None, 'final_mask': None, 'error': 'No valid regions'}
            
    except Exception as e:
        return {'detected': False, 'centroid': None, 'final_mask': None, 'error': str(e)}


def create_circle_mask(img_shape: Tuple[int, int], centroid_x: float, centroid_y: float, radius: float) -> np.ndarray:
    """
    Create a circular binary mask.
    
    Parameters:
    -----------
    img_shape : Tuple[int, int]
        Image dimensions (height, width)
    centroid_x : float
        Circle center X coordinate
    centroid_y : float
        Circle center Y coordinate
    radius : float
        Circle radius
        
    Returns:
    --------
    np.ndarray
        Binary mask (0 or 255)
    """
    mask = np.zeros(img_shape, dtype=np.uint8)
    cv2.circle(mask, (int(centroid_x), int(centroid_y)), int(radius), 255, -1)
    return mask


def calculate_circle_from_mask(mask: np.ndarray, centroid_x: float, centroid_y: float) -> Tuple[Optional[np.ndarray], float]:
    """
    Calculate a circle that encompasses the detected mask.
    Radius = maximum distance from centroid to any point in mask.
    
    Parameters:
    -----------
    mask : np.ndarray
        Binary mask of detected region
    centroid_x : float
        Centroid X coordinate
    centroid_y : float
        Centroid Y coordinate
        
    Returns:
    --------
    Tuple[Optional[np.ndarray], float]
        Circle mask and radius, or (None, 0.0) if mask is empty
    """
    y_coords, x_coords = np.where(mask > 0)
    if len(x_coords) == 0:
        return None, 0.0
    
    # Calculate maximum distance from centroid to mask boundary
    distances = np.sqrt((x_coords - centroid_x)**2 + (y_coords - centroid_y)**2)
    detected_radius = np.max(distances)
    
    # Create circle mask
    circle_mask = create_circle_mask(mask.shape, centroid_x, centroid_y, detected_radius)
    
    return circle_mask, detected_radius


def calculate_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """
    Calculate Intersection over Union (IoU) between two binary masks.
    
    IoU = |A ∩ B| / |A ∪ B|
    
    Parameters:
    -----------
    mask1 : np.ndarray
        First binary mask
    mask2 : np.ndarray
        Second binary mask
        
    Returns:
    --------
    float
        IoU score (0.0 to 1.0)
    """
    mask1_bool = (mask1 > 0).astype(bool)
    mask2_bool = (mask2 > 0).astype(bool)
    
    intersection = np.logical_and(mask1_bool, mask2_bool).sum()
    union = np.logical_or(mask1_bool, mask2_bool).sum()
    
    if union == 0:
        return 0.0
    
    return intersection / union


def is_centroid_inside_circle(det_x: float, det_y: float, gt_x: float, gt_y: float, gt_radius: float) -> Tuple[bool, float]:
    """
    Check if detected centroid is inside ground truth circle.
    
    Parameters:
    -----------
    det_x : float
        Detected centroid X
    det_y : float
        Detected centroid Y
    gt_x : float
        Ground truth centroid X
    gt_y : float
        Ground truth centroid Y
    gt_radius : float
        Ground truth radius
        
    Returns:
    --------
    Tuple[bool, float]
        (is_inside, distance_from_gt_center)
    """
    distance = np.sqrt((det_x - gt_x)**2 + (det_y - gt_y)**2)
    return distance <= gt_radius, distance


def load_optimized_params(params_file: str = 'optimized_params.json') -> Optional[Dict]:
    """
    Load optimized parameters from JSON file.
    
    Parameters:
    -----------
    params_file : str
        Path to JSON file with optimized parameters
        
    Returns:
    --------
    dict or None
        Loaded parameters or None if file doesn't exist
    """
    if os.path.exists(params_file):
        with open(params_file, 'r') as f:
            params = json.load(f)
        print(f"[OK] Loaded optimized parameters from {params_file}")
        return params
    else:
        print(f"[INFO] Parameters file not found: {params_file}")
        return None


def test_all_mass_images(csv_path: str, image_dir: str, params: Optional[Dict] = None, 
                        class_filter: Optional[list] = None, 
                        use_cso: bool = False, cso_sample_image: Optional[str] = None,
                        use_ml: bool = False, ml_model_path: Optional[str] = None) -> Dict:
    """
    Test detection on all MASS images from CSV.
    
    IMPORTANT: Applies Y-coordinate correction for MIAS images.
    MIAS origin is at bottom-left, so we use: y_corrected = height - y_original
    
    Parameters:
    -----------
    csv_path : str
        Path to CSV file with ground truth (train_dataset.csv)
    image_dir : str
        Directory containing MIAS images (all-mias)
    params : dict, optional
        Detection parameters. If None, uses defaults.
    class_filter : list, optional
        Filter MASS images by CLASS (e.g., ['SPIC', 'ARCH']). If None, tests all MASS images.
    use_cso : bool
        If True, run CSO optimization (WARNING: Very slow!)
    cso_sample_image : str, optional
        If use_cso=True, optimize on this image and use params for all images
    use_ml : bool
        If True, use ML classifier for region selection
    ml_model_path : str, optional
        Path to trained ML model (.pkl file)
        
    Returns:
    --------
    dict
        Results dictionary with 'results' DataFrame and 'stats' dict
    """
    # Load CSV
    df = pd.read_csv(csv_path)
    
    # Filter for MASS images with ground truth
    mass_df = df[df['TYPE'] == 'MASS'].copy()
    mass_df = mass_df[mass_df['X'].notna() & mass_df['Y'].notna() & mass_df['RADIUS'].notna()]
    
    # Apply class filter if specified
    if class_filter:
        mass_df = mass_df[mass_df['CLASS'].isin(class_filter)]
        print(f"Found {len(mass_df)} MASS images with ground truth (filtered by CLASS: {', '.join(class_filter)})")
    else:
        print(f"Found {len(mass_df)} MASS images with ground truth (all classes)")
    print("="*80)
    print("\n[IMPORTANT] Applying Y-coordinate correction for MIAS images")
    print("   MIAS origin is at bottom-left, converting to top-left origin\n")
    print("="*80)
    
    # ========== CSO OPTIMIZATION ==========
    if use_cso:
        if cso_sample_image:
            print(f"\n[CSO OPTIMIZATION MODE]")
            print(f"   Optimizing on sample image: {cso_sample_image}")
            print(f"   This will take ~15 minutes...")
            print(f"   Optimized parameters will be used for all images\n")
            
            # Find sample image in dataset
            sample_ref = cso_sample_image.replace('.pgm', '')
            sample_row = mass_df[mass_df['REF'] == sample_ref]
            
            if not sample_row.empty:
                sample_path = sample_row.iloc[0]['PATH']
                if not os.path.exists(sample_path):
                    sample_path = os.path.join(image_dir, f"{sample_ref}.pgm")
                
                if os.path.exists(sample_path):
                    try:
                        from cso_detection_optimizer import CSODetectionOptimizer
                        
                        print(f"   Loading sample image: {sample_path}")
                        sample_img = io.imread(sample_path)
                        sample_img_norm = cv2.normalize(sample_img, None, 0, 255, cv2.NORM_MINMAX)
                        sample_img_blur = cv2.GaussianBlur(np.uint8(sample_img_norm), (5, 5), 0)
                        
                        print("   Running CSO optimization...")
                        optimizer = CSODetectionOptimizer(n_nests=15, max_iterations=30, pa=0.25)
                        params = optimizer.optimize(sample_img_blur)
                        
                        # Save optimized parameters
                        with open('optimized_params.json', 'w') as f:
                            json.dump(params, f, indent=2)
                        print(f"   [OK] Optimization complete! Parameters saved to optimized_params.json")
                        print(f"   Using these parameters for all {len(mass_df)} images\n")
                    except ImportError:
                        print(f"   [ERROR] Could not import cso_detection_optimizer module")
                        print(f"   Falling back to default parameters\n")
                        params = None
                    except Exception as e:
                        print(f"   [ERROR] CSO optimization failed: {str(e)}")
                        print(f"   Falling back to default parameters\n")
                        params = None
                else:
                    print(f"   [ERROR] Sample image not found: {sample_path}")
                    print(f"   Falling back to default parameters\n")
                    params = None
            else:
                print(f"   [ERROR] Sample image {cso_sample_image} not found in dataset")
                print(f"   Falling back to default parameters\n")
                params = None
        else:
            print("\n[WARNING] use_cso=True but no sample image specified!")
            print("   Optimizing for EACH image would take ~15 min per image")
            print("   This is not recommended. Use --cso-sample to optimize once.")
            print("   Falling back to default parameters\n")
            params = None
    elif params is None:
        # Try to load pre-optimized parameters
        loaded_params = load_optimized_params('optimized_params.json')
        if loaded_params:
            params = loaded_params
            print("Using loaded optimized parameters\n")
        else:
            print("Using default parameters\n")
    
    results = []
    
    # Process each MASS image
    iterator = tqdm(mass_df.iterrows(), total=len(mass_df), desc="Testing MASS images") if HAS_TQDM else mass_df.iterrows()
    
    for idx, row in iterator:
        ref = row['REF']
        gt_x_original = float(row['X'])
        gt_y_original = float(row['Y'])
        gt_radius = float(row['RADIUS'])
        
        # Get image path
        img_path = row['PATH']
        if not os.path.exists(img_path):
            # Try alternative path
            img_path = os.path.join(image_dir, f"{ref}.pgm")
        
        if not os.path.exists(img_path):
            print(f"\n[WARNING] Image not found: {ref}")
            continue
        
        # Load image to get dimensions for Y-coordinate correction
        img = io.imread(img_path)
        img_height, img_width = img.shape
        
        # ===== CRITICAL: Y-COORDINATE CORRECTION FOR MIAS =====
        # MIAS images have origin at BOTTOM-LEFT, not top-left
        # We need to flip the Y coordinate
        gt_y_corrected = img_height - gt_y_original
        gt_x = gt_x_original  # X doesn't change
        
        # Run detection with parameters
        detection_result = detect_tumor_from_notebook(img_path, params, use_ml=use_ml, ml_model_path=ml_model_path)
        
        result = {
            'ref': ref,
            'image_height': img_height,
            'image_width': img_width,
            'gt_x_original': gt_x_original,
            'gt_y_original': gt_y_original,
            'gt_x': gt_x,
            'gt_y': gt_y_corrected,
            'gt_radius': gt_radius,
            'detected': detection_result['detected'],
            'error': detection_result['error']
        }
        
        if detection_result['detected']:
            det_x, det_y = detection_result['centroid']
            final_mask = detection_result['final_mask']
            
            # Check centroid hit (is detected centroid inside GT circle?)
            inside, distance = is_centroid_inside_circle(det_x, det_y, gt_x, gt_y_corrected, gt_radius)
            
            result['centroid_x'] = det_x
            result['centroid_y'] = det_y
            result['inside_circle'] = inside
            result['distance_from_gt'] = distance
            
            # Calculate IoU (Circle vs Circle)
            img_shape = (img_height, img_width)
            
            # Ground truth circle
            gt_circle_mask = create_circle_mask(img_shape, gt_x, gt_y_corrected, gt_radius)
            
            # Detected circle (from detected mask)
            detected_circle_mask, detected_radius = calculate_circle_from_mask(final_mask, det_x, det_y)
            
            if detected_circle_mask is not None:
                iou = calculate_iou(detected_circle_mask, gt_circle_mask)
                result['iou'] = iou
                result['detected_radius'] = detected_radius
            else:
                result['iou'] = 0.0
                result['detected_radius'] = 0.0
        else:
            result['centroid_x'] = None
            result['centroid_y'] = None
            result['inside_circle'] = False
            result['distance_from_gt'] = None
            result['iou'] = 0.0
            result['detected_radius'] = None
        
        results.append(result)
    
    # Calculate statistics
    results_df = pd.DataFrame(results)
    
    # Filter successful detections
    successful = results_df[results_df['detected'] == True]
    
    if len(successful) == 0:
        print("\n[ERROR] No successful detections!")
        return {'results': results_df, 'stats': {}}
    
    # Calculate metrics
    total_images = len(results_df)
    successful_count = len(successful)
    hit_count = successful['inside_circle'].sum()
    avg_hit_rate = (hit_count / successful_count) * 100
    avg_iou = successful['iou'].mean()
    
    stats = {
        'total_images': total_images,
        'successful_detections': successful_count,
        'detection_rate': (successful_count / total_images) * 100,
        'hit_count': int(hit_count),
        'avg_hit_rate': avg_hit_rate,
        'avg_iou': avg_iou,
        'std_iou': successful['iou'].std(),
        'min_iou': successful['iou'].min(),
        'max_iou': successful['iou'].max(),
        'median_iou': successful['iou'].median()
    }
    
    return {'results': results_df, 'stats': stats}


def print_results(summary: Dict):
    """Print formatted test results."""
    results_df = summary['results']
    stats = summary['stats']
    
    if not stats:
        print("\n[ERROR] No statistics available")
        return
    
    print("\n" + "="*80)
    print("MASS DETECTION TEST RESULTS")
    print("="*80)
    print(f"\n[Overall Statistics]")
    print(f"   Total MASS images tested:     {stats['total_images']}")
    print(f"   Successful detections:        {stats['successful_detections']} ({stats['detection_rate']:.1f}%)")
    
    print(f"\n[Centroid Hit Rate]")
    print(f"   Centroids inside GT circle:   {stats['hit_count']} / {stats['successful_detections']} ({stats['avg_hit_rate']:.1f}%)")
    
    print(f"\n[IoU Statistics - Circle vs Circle]")
    print(f"   Average IoU:                  {stats['avg_iou']:.4f}")
    print(f"   Median IoU:                   {stats['median_iou']:.4f}")
    print(f"   Standard Deviation:           {stats['std_iou']:.4f}")
    print(f"   Min IoU:                      {stats['min_iou']:.4f}")
    print(f"   Max IoU:                      {stats['max_iou']:.4f}")
    
    # Show top 5 and bottom 5 results
    successful = results_df[results_df['detected'] == True].copy()
    if len(successful) > 0:
        print("\n" + "="*80)
        print("TOP 5 BEST DETECTIONS (by IoU):")
        print("="*80)
        top5 = successful.nlargest(5, 'iou')[['ref', 'iou', 'inside_circle', 'distance_from_gt']]
        print(top5.to_string(index=False))
        
        print("\n" + "="*80)
        print("TOP 5 WORST DETECTIONS (by IoU):")
        print("="*80)
        bottom5 = successful.nsmallest(5, 'iou')[['ref', 'iou', 'inside_circle', 'distance_from_gt']]
        print(bottom5.to_string(index=False))
    
    # Failed detections
    failed = results_df[results_df['detected'] == False]
    if len(failed) > 0:
        print(f"\n" + "="*80)
        print(f"[FAILED DETECTIONS] ({len(failed)} images):")
        print("="*80)
        for _, row in failed.head(10).iterrows():
            print(f"   {row['ref']}: {row['error']}")
        if len(failed) > 10:
            print(f"   ... and {len(failed) - 10} more")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Test MASS detection with IoU and centroid hit evaluation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default parameters:
  python test_mass_iou.py
  
  # Use pre-optimized parameters:
  python test_mass_iou.py --params-file optimized_params.json
  
  # Run CSO optimization on sample image:
  python test_mass_iou.py --use-cso --cso-sample mdb267.pgm
        """
    )
    parser.add_argument('--csv', type=str, default='train_dataset.csv',
                       help='Path to CSV file with ground truth (default: train_dataset.csv)')
    parser.add_argument('--image-dir', type=str, default='all-mias',
                       help='Directory containing MIAS images (default: all-mias)')
    parser.add_argument('--output', type=str, default='mass_test_results.csv',
                       help='Output CSV file for detailed results (default: mass_test_results.csv)')
    parser.add_argument('--stats', type=str, default='mass_test_stats.csv',
                       help='Output CSV file for summary statistics (default: mass_test_stats.csv)')
    parser.add_argument('--use-cso', action='store_true',
                       help='Use CSO optimization (requires --cso-sample)')
    parser.add_argument('--cso-sample', type=str, default=None,
                       help='Sample image to optimize on (e.g., mdb267.pgm). Optimized params will be used for all images.')
    parser.add_argument('--params-file', type=str, default='optimized_params.json',
                       help='JSON file with optimized parameters (default: optimized_params.json)')
    parser.add_argument('--class-filter', type=str, nargs='+', default=None,
                       help='Filter by CLASS within MASS type (e.g., --class-filter SPIC ARCH). Available: SPIC, ARCH, CIRC, MISC, ASYM')
    parser.add_argument('--use-ml', action='store_true',
                       help='Use ML classifier for region selection (requires --ml-model)')
    parser.add_argument('--ml-model', type=str, default='tumor_region_classifier.pkl',
                       help='Path to trained ML model file (default: tumor_region_classifier.pkl)')
    
    args = parser.parse_args()
    
    print("="*80)
    print("MASS DETECTION TEST SCRIPT")
    print("="*80)
    print(f"CSV file:        {args.csv}")
    print(f"Image directory: {args.image_dir}")
    print(f"Output file:     {args.output}")
    
    # Load parameters if file exists
    params = None
    if os.path.exists(args.params_file) and not args.use_cso:
        params = load_optimized_params(args.params_file)
        print(f"Using parameters from: {args.params_file}")
    elif args.use_cso:
        print("CSO optimization mode enabled")
    else:
        print("Using default parameters")
    print()
    
    # Check if ML model exists when --use-ml is specified
    if args.use_ml:
        if not os.path.exists(args.ml_model):
            print(f"[ERROR] ML model not found: {args.ml_model}")
            print(f"Please train the model first using ml_region_classifier.py")
            exit(1)
        print(f"Using ML classifier: {args.ml_model}\n")
    
    # Run test
    summary = test_all_mass_images(
        csv_path=args.csv,
        image_dir=args.image_dir,
        class_filter=args.class_filter,
        params=params,
        use_cso=args.use_cso,
        cso_sample_image=args.cso_sample,
        use_ml=args.use_ml,
        ml_model_path=args.ml_model if args.use_ml else None
    )
    
    # Print results
    print_results(summary)
    
    # Save results to CSV
    summary['results'].to_csv(args.output, index=False)
    print(f"\n[OK] Detailed results saved to: {args.output}")
    
    # Save summary statistics
    if summary['stats']:
        stats_df = pd.DataFrame([summary['stats']])
        stats_df.to_csv(args.stats, index=False)
        print(f"[OK] Summary statistics saved to: {args.stats}")
    
    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)
