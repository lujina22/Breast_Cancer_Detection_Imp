"""
Cuckoo Search Optimization (CSO) for Tumor Detection Parameters
================================================================
Optimizes detection parameters using CSO algorithm to improve tumor detection accuracy.

Parameters to optimize:
1. Breast isolation threshold
2. CLAHE clipLimit
3. K-means K value
4. Morphological erosion iterations
5. Position filter thresholds (y_ratio, extent)
6. Scoring weights
7. Border margin
8. Area thresholds
"""

import numpy as np
import cv2
from scipy.special import gamma
from typing import Dict, Tuple, List, Optional
import pandas as pd
from skimage import io
from skimage.measure import label, regionprops


class CSODetectionOptimizer:
    """
    Cuckoo Search Optimization for tumor detection parameters.
    
    Parameters:
    -----------
    n_nests : int
        Number of nests (population size), default: 20
    max_iterations : int
        Maximum CSO iterations, default: 50
    pa : float
        Abandonment probability, default: 0.25
    """
    
    def __init__(self, n_nests: int = 20, max_iterations: int = 50, pa: float = 0.25):
        self.n_nests = n_nests
        self.max_iterations = max_iterations
        self.pa = pa
        self.fitness_history = []
        self.best_params = None
        self.best_fitness = float('inf')
        
    def _levy_flight(self, Lambda: float = 1.5) -> float:
        """Generate Lévy flight step for CSO."""
        sigma_u = (gamma(1 + Lambda) * np.sin(np.pi * Lambda / 2) / 
                   (gamma((1 + Lambda) / 2) * Lambda * 2 ** ((Lambda - 1) / 2))) ** (1 / Lambda)
        sigma_v = 1
        
        u = np.random.normal(0, sigma_u)
        v = np.random.normal(0, sigma_v)
        
        step = u / (np.abs(v) ** (1 / Lambda))
        return step
    
    def _detect_tumor_with_params(self, img_blur: np.ndarray, params: Dict) -> Tuple[Optional[Tuple[float, float]], float]:
        """
        Run tumor detection with given parameters.
        
        Returns:
        --------
        Tuple[Optional[Tuple[float, float]], float]
            (detected_centroid (x, y), detection_confidence)
            Returns (None, 0.0) if no tumor detected
        """
        try:
            # Extract parameters
            breast_thresh = int(params['breast_thresh'])
            clahe_clip = params['clahe_clip']
            k_means_k = int(params['k_means_k'])
            erosion_iter = int(params['erosion_iter'])
            y_ratio_thresh = params['y_ratio_thresh']
            extent_thresh = params['extent_thresh']
            border_margin = int(params['border_margin'])
            size_weight = params['size_weight']
            compact_weight = params['compact_weight']
            solidity_weight = params['solidity_weight']
            position_weight = params['position_weight']
            intensity_weight = params['intensity_weight']
            
            # Step 1: Isolate breast region
            _, thresh_breast = cv2.threshold(img_blur, breast_thresh, 255, cv2.THRESH_BINARY)
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(thresh_breast, connectivity=8)
            if num_labels < 2:
                return None, 0.0
            
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            breast_mask = np.zeros_like(thresh_breast)
            breast_mask[labels == largest_label] = 255
            
            kernel_mask = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            breast_mask = cv2.morphologyEx(breast_mask, cv2.MORPH_CLOSE, kernel_mask)
            breast_mask = cv2.morphologyEx(breast_mask, cv2.MORPH_OPEN, kernel_mask)
            breast_region = cv2.bitwise_and(img_blur, img_blur, mask=breast_mask)
            
            # Step 2: Histogram equalization (breast only)
            breast_only = breast_region[breast_mask > 0]
            if len(breast_only) == 0:
                return None, 0.0
            
            hist, _ = np.histogram(breast_only.flatten(), 256, [0, 256])
            cdf = hist.cumsum()
            cdf_normalized = (cdf * 255) / cdf[-1] if cdf[-1] > 0 else cdf
            img_eq = img_blur.copy()
            img_eq[breast_mask > 0] = cdf_normalized[breast_region[breast_mask > 0]]
            
            # Step 3: CLAHE enhancement
            clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
            masked_img = cv2.bitwise_and(img_eq, img_eq, mask=breast_mask)
            enhanced_img = clahe.apply(masked_img)
            
            # Step 4: K-means clustering
            pixel_values = enhanced_img.reshape((-1, 1)).astype(np.float32)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
            _, kmeans_labels, centers = cv2.kmeans(
                pixel_values, k_means_k, None, criteria, 10, cv2.KMEANS_PP_CENTERS
            )
            
            centers = np.uint8(centers)
            segmented = centers[kmeans_labels.flatten()]
            segmented = segmented.reshape(masked_img.shape)
            
            bright_cluster_idx = np.argmax(centers)
            candidate_mask = (segmented == centers[bright_cluster_idx]).astype(np.uint8) * 255
            
            # Step 5: Morphological cleanup
            kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            candidate_mask = cv2.morphologyEx(candidate_mask, cv2.MORPH_OPEN, kernel_clean)
            candidate_mask = cv2.morphologyEx(candidate_mask, cv2.MORPH_CLOSE, kernel_clean)
            candidate_mask = cv2.erode(candidate_mask, kernel_clean, iterations=erosion_iter)
            
            # Step 6: Position filtering
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
                
                if y_ratio > y_ratio_thresh and extent > extent_thresh:
                    cv2.drawContours(final_tumor_mask, [cnt], -1, 255, -1)
            
            # Step 7: Region selection with scoring
            label_img = label(final_tumor_mask)
            regions = regionprops(label_img, intensity_image=enhanced_img)
            
            valid_regions = []
            for region in regions:
                min_row, min_col, max_row, max_col = region.bbox
                if (min_row > border_margin and min_col > border_margin and 
                    max_row < (height - border_margin) and max_col < (width - border_margin)):
                    
                    area = region.area
                    region_y, region_x = region.centroid
                    eccentricity = region.eccentricity
                    solidity = region.solidity
                    mean_intensity = region.mean_intensity
                    y_ratio = region_y / height
                    
                    # Scoring
                    size_score = 1.0
                    if 1000 <= area <= 8000:
                        size_score = 1.5
                    elif area < 500:
                        size_score = 0.3
                    elif area > 20000:
                        size_score = 0.5
                    
                    compactness_score = 1.0 - (eccentricity * 0.5)
                    solidity_score = solidity
                    
                    position_score = 1.0
                    if 0.3 <= y_ratio <= 0.7:
                        position_score = 1.3
                    elif y_ratio < 0.2 or y_ratio > 0.8:
                        position_score = 0.5
                    
                    intensity_score = mean_intensity / 255.0
                    
                    total_score = (size_score * size_weight + 
                                  compactness_score * compact_weight + 
                                  solidity_score * solidity_weight + 
                                  position_score * position_weight + 
                                  intensity_score * intensity_weight)
                    
                    valid_regions.append((region, total_score, region_x, region_y))
            
            if valid_regions:
                valid_regions.sort(key=lambda x: x[1], reverse=True)
                selected_region, score, reg_x, reg_y = valid_regions[0]
                return (reg_x, reg_y), score
            
            return None, 0.0
            
        except Exception as e:
            return None, 0.0
    
    def _fitness_function(self, params: Dict, img_blur: np.ndarray, ground_truth: Optional[Tuple[float, float, float]] = None) -> float:
        """
        Fitness function: optimize detection quality WITHOUT using ground truth.
        
        Optimizes based on:
        1. Detection confidence/score
        2. Number of valid candidates (prefer 1-3, penalize too many/few)
        3. Shape quality (compactness, solidity)
        4. Size appropriateness
        5. Position within breast region
        
        Parameters:
        -----------
        params : Dict
            Detection parameters
        img_blur : np.ndarray
            Preprocessed image
        ground_truth : Optional[Tuple[float, float, float]]
            NOT USED - kept for API compatibility only
            
        Returns:
        --------
        float
            Fitness value (lower is better)
        """
        try:
            # Run detection and get all candidate regions
            detected, confidence = self._detect_tumor_with_params(img_blur, params)
            
            if detected is None:
                # Heavy penalty for no detection
                return 1000.0
            
            # Get detailed detection info by running full pipeline
            # Extract parameters
            breast_thresh = int(params['breast_thresh'])
            clahe_clip = params['clahe_clip']
            k_means_k = int(params['k_means_k'])
            erosion_iter = int(params['erosion_iter'])
            y_ratio_thresh = params['y_ratio_thresh']
            extent_thresh = params['extent_thresh']
            border_margin = int(params['border_margin'])
            size_weight = params['size_weight']
            compact_weight = params['compact_weight']
            solidity_weight = params['solidity_weight']
            position_weight = params['position_weight']
            intensity_weight = params['intensity_weight']
            
            # Run full detection to get all regions
            _, thresh_breast = cv2.threshold(img_blur, breast_thresh, 255, cv2.THRESH_BINARY)
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(thresh_breast, connectivity=8)
            if num_labels < 2:
                return 1000.0
            
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            breast_mask = np.zeros_like(thresh_breast)
            breast_mask[labels == largest_label] = 255
            
            kernel_mask = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            breast_mask = cv2.morphologyEx(breast_mask, cv2.MORPH_CLOSE, kernel_mask)
            breast_mask = cv2.morphologyEx(breast_mask, cv2.MORPH_OPEN, kernel_mask)
            breast_region = cv2.bitwise_and(img_blur, img_blur, mask=breast_mask)
            
            breast_only = breast_region[breast_mask > 0]
            if len(breast_only) == 0:
                return 1000.0
            
            hist, _ = np.histogram(breast_only.flatten(), 256, [0, 256])
            cdf = hist.cumsum()
            cdf_normalized = (cdf * 255) / cdf[-1] if cdf[-1] > 0 else cdf
            img_eq = img_blur.copy()
            img_eq[breast_mask > 0] = cdf_normalized[breast_region[breast_mask > 0]]
            
            clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
            masked_img = cv2.bitwise_and(img_eq, img_eq, mask=breast_mask)
            enhanced_img = clahe.apply(masked_img)
            
            pixel_values = enhanced_img.reshape((-1, 1)).astype(np.float32)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
            _, kmeans_labels, centers = cv2.kmeans(
                pixel_values, k_means_k, None, criteria, 10, cv2.KMEANS_PP_CENTERS
            )
            
            centers = np.uint8(centers)
            segmented = centers[kmeans_labels.flatten()]
            segmented = segmented.reshape(masked_img.shape)
            
            bright_cluster_idx = np.argmax(centers)
            candidate_mask = (segmented == centers[bright_cluster_idx]).astype(np.uint8) * 255
            
            kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            candidate_mask = cv2.morphologyEx(candidate_mask, cv2.MORPH_OPEN, kernel_clean)
            candidate_mask = cv2.morphologyEx(candidate_mask, cv2.MORPH_CLOSE, kernel_clean)
            candidate_mask = cv2.erode(candidate_mask, kernel_clean, iterations=erosion_iter)
            
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
                if y_ratio > y_ratio_thresh and extent > extent_thresh:
                    cv2.drawContours(final_tumor_mask, [cnt], -1, 255, -1)
            
            label_img = label(final_tumor_mask)
            regions = regionprops(label_img, intensity_image=enhanced_img)
            
            valid_regions = []
            for region in regions:
                min_row, min_col, max_row, max_col = region.bbox
                if (min_row > border_margin and min_col > border_margin and 
                    max_row < (height - border_margin) and max_col < (width - border_margin)):
                    
                    area = region.area
                    region_y, region_x = region.centroid
                    eccentricity = region.eccentricity
                    solidity = region.solidity
                    mean_intensity = region.mean_intensity
                    y_ratio = region_y / height
                    
                    size_score = 1.0
                    if 1000 <= area <= 8000:
                        size_score = 1.5
                    elif area < 500:
                        size_score = 0.3
                    elif area > 20000:
                        size_score = 0.5
                    
                    compactness_score = 1.0 - (eccentricity * 0.5)
                    solidity_score = solidity
                    
                    position_score = 1.0
                    if 0.3 <= y_ratio <= 0.7:
                        position_score = 1.3
                    elif y_ratio < 0.2 or y_ratio > 0.8:
                        position_score = 0.5
                    
                    intensity_score = mean_intensity / 255.0
                    
                    total_score = (size_score * size_weight + 
                                  compactness_score * compact_weight + 
                                  solidity_score * solidity_weight + 
                                  position_score * position_weight + 
                                  intensity_score * intensity_weight)
                    
                    valid_regions.append((region, total_score, region_x, region_y, area, eccentricity, solidity))
            
            if not valid_regions:
                return 1000.0  # No valid regions
            
            # Sort by score
            valid_regions.sort(key=lambda x: x[1], reverse=True)
            best_region, best_score, best_x, best_y, best_area, best_ecc, best_sol = valid_regions[0]
            
            # Calculate fitness based on quality metrics (NO GROUND TRUTH)
            fitness = 0.0
            
            # 1. Confidence/Score (higher is better, so subtract from max)
            max_possible_score = 10.0  # Approximate max
            fitness += (max_possible_score - best_score) * 2.0
            
            # 2. Number of candidates (prefer 1-3, penalize too many or too few)
            num_candidates = len(valid_regions)
            if num_candidates == 0:
                fitness += 500.0
            elif num_candidates == 1:
                fitness += 0.0  # Perfect
            elif num_candidates <= 3:
                fitness += (num_candidates - 1) * 10.0  # Small penalty
            else:
                fitness += 50.0 + (num_candidates - 3) * 20.0  # Large penalty for too many
            
            # 3. Shape quality (lower eccentricity, higher solidity is better)
            fitness += best_ecc * 5.0  # Penalize high eccentricity
            fitness += (1.0 - best_sol) * 10.0  # Penalize low solidity
            
            # 4. Size appropriateness (prefer medium size)
            if best_area < 500:
                fitness += 100.0  # Too small
            elif best_area > 20000:
                fitness += 50.0  # Too large
            elif 1000 <= best_area <= 8000:
                fitness += 0.0  # Ideal size
            
            # 5. Score gap (prefer clear winner over ambiguous)
            if len(valid_regions) > 1:
                score_gap = best_score - valid_regions[1][1]
                if score_gap < 0.5:
                    fitness += 20.0  # Penalize ambiguous results
            
            return fitness
            
        except Exception as e:
            return 1000.0  # Error penalty
    
    def _initialize_nests(self) -> List[Dict]:
        """Initialize parameter nests (solutions)."""
        nests = []
        
        for _ in range(self.n_nests):
            params = {
                'breast_thresh': np.random.uniform(20, 50),
                'clahe_clip': np.random.uniform(1.0, 4.0),
                'k_means_k': np.random.choice([3, 4, 5]),
                'erosion_iter': np.random.uniform(5, 20),
                'y_ratio_thresh': np.random.uniform(0.15, 0.35),
                'extent_thresh': np.random.uniform(0.15, 0.3),
                'border_margin': np.random.uniform(30, 80),
                'size_weight': np.random.uniform(1.0, 2.0),
                'compact_weight': np.random.uniform(0.8, 1.5),
                'solidity_weight': np.random.uniform(0.8, 1.5),
                'position_weight': np.random.uniform(0.8, 1.5),
                'intensity_weight': np.random.uniform(0.5, 1.2)
            }
            nests.append(params)
        
        return nests
    
    def optimize(self, img_blur: np.ndarray, ground_truth: Optional[Tuple[float, float, float]] = None) -> Dict:
        """
        Optimize detection parameters using CSO.
        
        NOTE: Ground truth is NOT used during optimization - only for evaluation/testing.
        Optimization is based on detection quality metrics (confidence, shape, size, etc.)
        
        Parameters:
        -----------
        img_blur : np.ndarray
            Preprocessed image (blurred, normalized)
        ground_truth : Optional[Tuple[float, float, float]]
            NOT USED in optimization - kept for API compatibility only
            Use only for evaluation after optimization
            
        Returns:
        --------
        Dict
            Optimized parameters
        """
        print("[NOTE] Optimization does NOT use ground truth.")
        print("    Optimizing based on detection quality metrics only.\n")
        
        # Initialize nests
        nests = self._initialize_nests()
        fitness = np.array([self._fitness_function(nest, img_blur) for nest in nests])
        
        # Find best nest
        best_idx = np.argmin(fitness)
        best_nest = nests[best_idx].copy()
        best_fitness = fitness[best_idx]
        
        self.fitness_history = [best_fitness]
        
        # Parameter bounds for clipping
        bounds = {
            'breast_thresh': (20, 50),
            'clahe_clip': (1.0, 4.0),
            'k_means_k': (3, 5),
            'erosion_iter': (5, 20),
            'y_ratio_thresh': (0.15, 0.35),
            'extent_thresh': (0.15, 0.3),
            'border_margin': (30, 80),
            'size_weight': (1.0, 2.0),
            'compact_weight': (0.8, 1.5),
            'solidity_weight': (0.8, 1.5),
            'position_weight': (0.8, 1.5),
            'intensity_weight': (0.5, 1.2)
        }
        
        # CSO main loop
        for iteration in range(self.max_iterations):
            # Generate new solutions via Lévy flights
            for i in range(self.n_nests):
                step_size = self._levy_flight()
                new_nest = nests[i].copy()
                
                # Apply Lévy flight to each parameter
                for key in new_nest.keys():
                    if key == 'k_means_k':
                        # Discrete parameter
                        new_nest[key] = np.random.choice([3, 4, 5])
                    else:
                        # Continuous parameter
                        param_range = bounds[key][1] - bounds[key][0]
                        step = step_size * np.random.randn() * param_range * 0.1
                        new_nest[key] = new_nest[key] + step
                        new_nest[key] = np.clip(new_nest[key], bounds[key][0], bounds[key][1])
                
                # Evaluate new solution (NO ground truth used)
                new_fitness = self._fitness_function(new_nest, img_blur)
                
                # Replace if better (greedy selection)
                j = np.random.randint(self.n_nests)
                if new_fitness < fitness[i]:
                    nests[i] = new_nest
                    fitness[i] = new_fitness
                elif new_fitness < fitness[j]:
                    nests[i] = new_nest
                    fitness[i] = new_fitness
            
            # Abandon worst nests with probability pa
            n_abandon = int(self.pa * self.n_nests)
            if n_abandon > 0:
                worst_indices = np.argsort(fitness)[-n_abandon:]
                
                for idx in worst_indices:
                    # Generate new random solution
                    new_nest = {}
                    for key, (low, high) in bounds.items():
                        if key == 'k_means_k':
                            new_nest[key] = np.random.choice([3, 4, 5])
                        else:
                            new_nest[key] = np.random.uniform(low, high)
                    nests[idx] = new_nest
                    fitness[idx] = self._fitness_function(new_nest, img_blur)
            
            # Update best solution
            current_best_idx = np.argmin(fitness)
            if fitness[current_best_idx] < best_fitness:
                best_nest = nests[current_best_idx].copy()
                best_fitness = fitness[current_best_idx]
            
            self.fitness_history.append(best_fitness)
            
            # Progress reporting
            if (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration + 1}/{self.max_iterations}, Best Fitness: {best_fitness:.4f}")
        
        self.best_params = best_nest
        self.best_fitness = best_fitness
        
        return best_nest

