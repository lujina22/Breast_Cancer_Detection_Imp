"""
ML-Based Tumor Region Classifier
==================================
Trains a Random Forest classifier to select the correct tumor region
from multiple candidates, improving detection accuracy.

Features:
1. Extract training data from MASS images with ground truth
2. Train Random Forest classifier on region features
3. Integrate with detection pipeline for improved region selection

Usage:
    # Extract training features:
    python ml_region_classifier.py --mode extract --csv train_dataset.csv
    
    # Train model:
    python ml_region_classifier.py --mode train --input training_features.csv
    
    # Both steps combined:
    python ml_region_classifier.py --mode all --csv train_dataset.csv
"""

import cv2
import numpy as np
import pandas as pd
from skimage import io
from skimage.measure import label, regionprops
from skimage.feature import graycomatrix, graycoprops
import os
import argparse
import joblib
from typing import Dict, List, Tuple, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import warnings
warnings.filterwarnings('ignore')

# Progress bar
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(iterable, **kwargs):
        return iterable


def extract_region_features(region, enhanced_img: np.ndarray, img_shape: Tuple[int, int]) -> Dict:
    """
    Extract comprehensive features from a candidate region.
    
    Parameters:
    -----------
    region : regionprops object
        Region to extract features from
    enhanced_img : np.ndarray
        Enhanced image for intensity calculations
    img_shape : Tuple[int, int]
        Image dimensions (height, width)
        
    Returns:
    --------
    dict
        Dictionary of features
    """
    features = {}
    height, width = img_shape
    
    # ========== SHAPE FEATURES ==========
    features['area'] = region.area
    features['perimeter'] = region.perimeter
    features['equivalent_diameter'] = region.equivalent_diameter
    features['eccentricity'] = region.eccentricity
    features['solidity'] = region.solidity
    features['extent'] = region.extent
    features['major_axis_length'] = region.major_axis_length
    features['minor_axis_length'] = region.minor_axis_length
    
    # Circularity: 4π × area / perimeter²
    if region.perimeter > 0:
        features['circularity'] = (4 * np.pi * region.area) / (region.perimeter ** 2)
    else:
        features['circularity'] = 0
    
    # Aspect ratio
    if region.minor_axis_length > 0:
        features['aspect_ratio'] = region.major_axis_length / region.minor_axis_length
    else:
        features['aspect_ratio'] = 1.0
    
    # ========== POSITION FEATURES ==========
    centroid_y, centroid_x = region.centroid
    features['centroid_x_norm'] = centroid_x / width
    features['centroid_y_norm'] = centroid_y / height
    features['y_ratio'] = centroid_y / height
    
    # Distance from image center
    center_x, center_y = width / 2, height / 2
    features['dist_from_center'] = np.sqrt((centroid_x - center_x)**2 + (centroid_y - center_y)**2)
    features['dist_from_center_norm'] = features['dist_from_center'] / np.sqrt(center_x**2 + center_y**2)
    
    # ========== INTENSITY FEATURES ==========
    features['mean_intensity'] = region.mean_intensity
    
    # Extract region pixels for detailed statistics
    min_row, min_col, max_row, max_col = region.bbox
    region_mask = np.zeros(img_shape, dtype=bool)
    for r, c in region.coords:
        region_mask[r, c] = True
    region_pixels = enhanced_img[region_mask]
    
    if len(region_pixels) > 0:
        features['std_intensity'] = np.std(region_pixels)
        features['min_intensity'] = np.min(region_pixels)
        features['max_intensity'] = np.max(region_pixels)
        features['intensity_range'] = features['max_intensity'] - features['min_intensity']
        
        # Skewness and kurtosis (basic approximation)
        features['intensity_skew'] = np.mean((region_pixels - features['mean_intensity'])**3) / (features['std_intensity']**3 + 1e-6)
        features['intensity_kurtosis'] = np.mean((region_pixels - features['mean_intensity'])**4) / (features['std_intensity']**4 + 1e-6)
    else:
        features['std_intensity'] = 0
        features['min_intensity'] = 0
        features['max_intensity'] = 0
        features['intensity_range'] = 0
        features['intensity_skew'] = 0
        features['intensity_kurtosis'] = 0
    
    # ========== TEXTURE FEATURES (GLCM) ==========
    # Extract small patch around region for GLCM
    try:
        patch = enhanced_img[min_row:max_row, min_col:max_col]
        if patch.size > 4:  # Need minimum size
            # Quantize to reduce GLCM computation
            patch_q = (patch / 32).astype(np.uint8)
            glcm = graycomatrix(patch_q, distances=[1], angles=[0], levels=8, symmetric=True, normed=True)
            
            features['glcm_contrast'] = graycoprops(glcm, 'contrast')[0, 0]
            features['glcm_correlation'] = graycoprops(glcm, 'correlation')[0, 0]
            features['glcm_energy'] = graycoprops(glcm, 'energy')[0, 0]
            features['glcm_homogeneity'] = graycoprops(glcm, 'homogeneity')[0, 0]
        else:
            features['glcm_contrast'] = 0
            features['glcm_correlation'] = 0
            features['glcm_energy'] = 0
            features['glcm_homogeneity'] = 0
    except:
        features['glcm_contrast'] = 0
        features['glcm_correlation'] = 0
        features['glcm_energy'] = 0
        features['glcm_homogeneity'] = 0
    
    return features


def extract_candidates_with_features(img_path: str, params: Optional[Dict] = None) -> Tuple[List[Dict], np.ndarray, Tuple[int, int]]:
    """
    Run detection pipeline up to candidate extraction and get features.
    
    Returns:
    --------
    Tuple[List[Dict], np.ndarray, Tuple[int, int]]
        (candidates_with_features, enhanced_img, img_shape)
    """
    # Set default parameters
    if params is None:
        params = {
            'breast_thresh': 30,
            'clahe_clip': 3.0,
            'k_means_k': 3,
            'erosion_iter': 10,
            'y_ratio_thresh': 0.25,
            'extent_thresh': 0.2,
            'border_margin': 50
        }
    
    # Run detection pipeline (copy from test_mass_iou.py)
    img = io.imread(img_path)
    img_norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    img_uint8 = np.uint8(img_norm)
    img_blur = cv2.GaussianBlur(img_uint8, (5, 5), 0)
    
    # Isolate breast region
    _, thresh_breast = cv2.threshold(img_blur, params['breast_thresh'], 255, cv2.THRESH_BINARY)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(thresh_breast, connectivity=8)
    
    if num_labels < 2:
        return [], None, (0, 0)
    
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    breast_mask = np.zeros_like(thresh_breast)
    breast_mask[labels == largest_label] = 255
    
    kernel_mask = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    breast_mask = cv2.morphologyEx(breast_mask, cv2.MORPH_CLOSE, kernel_mask)
    breast_mask = cv2.morphologyEx(breast_mask, cv2.MORPH_OPEN, kernel_mask)
    breast_region = cv2.bitwise_and(img_blur, img_blur, mask=breast_mask)
    
    # Histogram equalization
    breast_only = breast_region[breast_mask > 0]
    if len(breast_only) == 0:
        return [], None, (0, 0)
    
    hist, _ = np.histogram(breast_only.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_normalized = (cdf * 255) / cdf[-1] if cdf[-1] > 0 else cdf
    img_eq = img_blur.copy()
    img_eq[breast_mask > 0] = cdf_normalized[breast_region[breast_mask > 0]]
    
    # CLAHE enhancement
    clahe = cv2.createCLAHE(clipLimit=params['clahe_clip'], tileGridSize=(8, 8))
    masked_img = cv2.bitwise_and(img_eq, img_eq, mask=breast_mask)
    enhanced_img = clahe.apply(masked_img)
    
    # K-means clustering
    pixel_values = enhanced_img.reshape((-1, 1)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    K = int(params['k_means_k'])
    _, kmeans_labels, centers = cv2.kmeans(pixel_values, K, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    
    centers = np.uint8(centers)
    segmented = centers[kmeans_labels.flatten()].reshape(masked_img.shape)
    bright_cluster_idx = np.argmax(centers)
    candidate_mask = (segmented == centers[bright_cluster_idx]).astype(np.uint8) * 255
    
    # Morphological cleanup
    kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    candidate_mask = cv2.morphologyEx(candidate_mask, cv2.MORPH_OPEN, kernel_clean)
    candidate_mask = cv2.morphologyEx(candidate_mask, cv2.MORPH_CLOSE, kernel_clean)
    candidate_mask = cv2.erode(candidate_mask, kernel_clean, iterations=int(params['erosion_iter']))
    
    # Position-based filtering
    contours, _ = cv2.findContours(candidate_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final_tumor_mask = np.zeros_like(candidate_mask)
    height, width = candidate_mask.shape
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 500:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        cy = y + h // 2
        extent = area / (w * h) if (w * h) > 0 else 0
        y_ratio = cy / height
        if y_ratio > params['y_ratio_thresh'] and extent > params['extent_thresh']:
            cv2.drawContours(final_tumor_mask, [cnt], -1, 255, -1)
    
    # Extract regions and features
    label_img = label(final_tumor_mask)
    regions = regionprops(label_img, intensity_image=enhanced_img)
    
    border_margin = int(params['border_margin'])
    candidates = []
    
    for region in regions:
        min_row, min_col, max_row, max_col = region.bbox
        
        # Check if away from borders
        if (min_row > border_margin and min_col > border_margin and
            max_row < (height - border_margin) and max_col < (width - border_margin)):
            
            features = extract_region_features(region, enhanced_img, (height, width))
            features['centroid_x'] = region.centroid[1]  # For labeling later
            features['centroid_y'] = region.centroid[0]
            candidates.append(features)
    
    return candidates, enhanced_img, (height, width)


def generate_training_data(csv_path: str, image_dir: str, class_filter: Optional[List[str]] = None,
                          params_file: Optional[str] = None, output_path: str = 'training_features.csv'):
    """
    Extract training data from MASS images with ground truth.
    
    Parameters:
    -----------
    csv_path : str
        Path to CSV with ground truth
    image_dir : str
        Directory with images
    class_filter : list, optional
        Filter by CLASS
    params_file : str, optional
        JSON file with detection parameters
    output_path : str
        Output CSV file path
    """
    # Load parameters if provided
    params = None
    if params_file and os.path.exists(params_file):
        import json
        with open(params_file, 'r') as f:
            params = json.load(f)
        print(f"[OK] Loaded parameters from {params_file}")
    
    # Load CSV
    df = pd.read_csv(csv_path)
    mass_df = df[df['TYPE'] == 'MASS'].copy()
    mass_df = mass_df[mass_df['X'].notna() & mass_df['Y'].notna() & mass_df['RADIUS'].notna()]
    
    if class_filter:
        mass_df = mass_df[mass_df['CLASS'].isin(class_filter)]
        print(f"Found {len(mass_df)} MASS images (filtered by CLASS: {', '.join(class_filter)})")
    else:
        print(f"Found {len(mass_df)} MASS images")
    
    training_data = []
    
    iterator = tqdm(mass_df.iterrows(), total=len(mass_df), desc="Extracting features") if HAS_TQDM else mass_df.iterrows()
    
    for idx, row in iterator:
        ref = row['REF']
        gt_x = float(row['X'])
        gt_y_original = float(row['Y'])
        gt_radius = float(row['RADIUS'])
        
        # Get image path
        img_path = row['PATH']
        if not os.path.exists(img_path):
            img_path = os.path.join(image_dir, f"{ref}.pgm")
        
        if not os.path.exists(img_path):
            continue
        
        # Load image for Y-coordinate correction
        img = io.imread(img_path)
        img_height = img.shape[0]
        gt_y = img_height - gt_y_original  # Y-coordinate flip
        
        # Extract candidates
        candidates, enhanced_img, img_shape = extract_candidates_with_features(img_path, params)
        
        if not candidates:
            continue
        
        # Label candidates
        for candidate in candidates:
            cand_x = candidate['centroid_x']
            cand_y = candidate['centroid_y']
            
            # Check if centroid inside GT circle
            distance = np.sqrt((cand_x - gt_x)**2 + (cand_y - gt_y)**2)
            label = 1 if distance <= gt_radius else 0
            
            # Add metadata
            candidate['label'] = label
            candidate['image_ref'] = ref
            candidate['distance_to_gt'] = distance
            
            training_data.append(candidate)
    
    # Save to CSV
    df_train = pd.DataFrame(training_data)
    df_train.to_csv(output_path, index=False)
    
    print(f"\n[OK] Extracted {len(df_train)} training examples")
    print(f"     Positive examples (tumor): {(df_train['label']==1).sum()}")
    print(f"     Negative examples (non-tumor): {(df_train['label']==0).sum()}")
    print(f"[OK] Saved to: {output_path}")
    
    return df_train


def train_classifier(input_path: str = 'training_features.csv', output_model: str = 'tumor_region_classifier.pkl'):
    """
    Train Random Forest classifier on extracted features.
    
    Parameters:
    -----------
    input_path : str
        CSV file with training features
    output_model : str
        Output model file path
    """
    print(f"\n{'='*70}")
    print("TRAINING RANDOM FOREST CLASSIFIER")
    print(f"{'='*70}\n")
    
    # Load data
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} training examples")
    
    # Separate features and labels
    feature_cols = [col for col in df.columns if col not in ['label', 'image_ref', 'distance_to_gt', 'centroid_x', 'centroid_y']]
    X = df[feature_cols]
    y = df['label']
    
    print(f"Features: {len(feature_cols)}")
    print(f"Positive: {(y==1).sum()} | Negative: {(y==0).sum()}")
    print(f"Class balance: {(y==1).sum() / len(y) * 100:.1f}% positive\n")
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Train Random Forest
    print("Training Random Forest...")
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',  # Handle class imbalance
        random_state=42,
        n_jobs=-1
    )
    
    # Cross-validation
    cv_scores = cross_val_score(clf, X_train, y_train, cv=5, scoring='f1')
    print(f"Cross-validation F1 Score: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    
    # Train on full training set
    clf.fit(X_train, y_train)
    
    # Test set evaluation
    y_pred = clf.predict(X_test)
    test_f1 = f1_score(y_test, y_pred)
    
    print(f"\n{'='*70}")
    print("TEST SET PERFORMANCE")
    print(f"{'='*70}")
    print(classification_report(y_test, y_pred, target_names=['Non-tumor', 'Tumor']))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"                Predicted")
    print(f"                Non-tumor  Tumor")
    print(f"Actual Non-tumor    {cm[0,0]:4d}    {cm[0,1]:4d}")
    print(f"       Tumor        {cm[1,0]:4d}    {cm[1,1]:4d}")
    
    # Feature importance
    print(f"\n{'='*70}")
    print("TOP 10 MOST IMPORTANT FEATURES")
    print(f"{'='*70}")
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': clf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for idx, row in feature_importance.head(10).iterrows():
        print(f"{row['feature']:30s} {row['importance']:.4f}")
    
    # Save model
    joblib.dump(clf, output_model)
    print(f"\n[OK] Model saved to: {output_model}")
    print(f"[OK] Features used: {len(feature_cols)}")
    
    return clf, feature_importance


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ML-based tumor region classifier')
    parser.add_argument('--mode', type=str, choices=['extract', 'train', 'all'], required=True,
                       help='Mode: extract features, train model, or both')
    parser.add_argument('--csv', type=str, default='train_dataset.csv',
                       help='CSV file with ground truth (for extract mode)')
    parser.add_argument('--image-dir', type=str, default='all-mias',
                       help='Directory with images (for extract mode)')
    parser.add_argument('--class-filter', type=str, nargs='+', default=None,
                       help='Filter by CLASS (e.g., SPIC ARCH ASYM)')
    parser.add_argument('--params-file', type=str, default='optimized_params.json',
                       help='JSON file with detection parameters')
    parser.add_argument('--input', type=str, default='training_features.csv',
                       help='Input CSV for training (for train mode)')
    parser.add_argument('--output', type=str, default='training_features.csv',
                       help='Output CSV for features (for extract mode)')
    parser.add_argument('--model', type=str, default='tumor_region_classifier.pkl',
                       help='Output model file (for train mode)')
    
    args = parser.parse_args()
    
    if args.mode in ['extract', 'all']:
        print(f"\n{'='*70}")
        print("EXTRACTING TRAINING FEATURES")
        print(f"{'='*70}\n")
        generate_training_data(
            csv_path=args.csv,
            image_dir=args.image_dir,
            class_filter=args.class_filter,
            params_file=args.params_file if os.path.exists(args.params_file) else None,
            output_path=args.output
        )
    
    if args.mode in ['train', 'all']:
        train_classifier(
            input_path=args.input if args.mode == 'train' else args.output,
            output_model=args.model
        )
    
    print(f"\n{'='*70}")
    print("COMPLETE")
    print(f"{'='*70}")
