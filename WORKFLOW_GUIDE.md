# CSO Optimization Workflow Guide

## File: `Input.ipynb`

### Step-by-Step Cell Execution Order

#### **Phase 1: Setup and Preprocessing** (Run these first)

1. **Cell 0**: Import common functions and setup
   - Imports `commonfunctions`
   - Sets up matplotlib inline
   - Enables autoreload

2. **Cell 1**: Import libraries
   - Imports cv2, numpy, matplotlib, etc.

3. **Cell 2**: Load image
   - Sets `image_name` (e.g., 'mdb155.pgm')
   - Loads image from `all-mias/` folder
   - **IMPORTANT**: Change `image_name` here to test different images

4. **Cell 3**: Normalize image
   - Normalizes to 0-255 range
   - Converts to uint8

5. **Cell 4**: Apply Gaussian blur
   - Creates `img_blur` (needed for later steps)

6. **Cell 5**: (Optional visualization)

#### **Phase 2: Breast Isolation and Equalization** (Critical preprocessing)

7. **Cell 6**: Isolate breast region
   - Creates `breast_mask` and `breast_region`
   - This prevents background from affecting equalization

8. **Cell 7**: Apply histogram equalization (breast only)
   - Creates `img_eq` (equalized image)
   - Only equalizes breast region, not background

9. **Cell 8**: Refine breast mask
   - Creates `clean_mask` (refined breast mask)

10. **Cell 9**: (Optional visualization)

11. **Cell 10**: Apply mask to equalized image
    - Creates `masked_img`

#### **Phase 3: Tumor Detection** (Current algorithm)

12. **Cell 11**: CLAHE + K-means + Morphological filtering
    - Creates `enhanced_img`, `candidate_mask`, `final_tumor_mask`
    - Uses current (non-optimized) parameters

13. **Cell 12**: Automatic tumor detection with heuristics
    - Creates `final_mask` (detected tumor)
    - Uses scoring system to select best region

14. **Cell 13**: (Optional visualization)

15. **Cell 14**: Feature extraction
    - Extracts features from detected tumor
    - Creates `features_df`

16. **Cell 15**: Visualize detected tumor
    - Shows detected centroid

#### **Phase 4: CSO Optimization** (NEW - Run this to optimize)

17. **Cell 16**: CSO Optimization
    - **IMPORTANT**: This cell will take several minutes to run
    - Loads ground truth from `train_dataset.csv` if available
    - Optimizes 12 detection parameters
    - Creates `optimized_params` dictionary
    - Shows convergence plot

18. **Cell 17**: Apply optimized parameters
    - Extracts optimized parameters as variables
    - Sets up parameters for use in detection

#### **Phase 5: Detection with Optimized Parameters** (After optimization)

19. **Cell 18**: Detection with optimized parameters (automatic)
    - Re-runs Cell 11 logic but uses CSO-optimized parameters
    - Creates `final_tumor_mask_opt` and `enhanced_img_opt`
    - **No manual editing needed** - automatically uses optimized values

20. **Cell 19**: Optimized tumor detection with scoring
    - Re-runs Cell 12 logic but uses CSO-optimized scoring weights
    - Creates `final_mask_opt` (optimized detection result)
    - **No manual editing needed** - automatically uses optimized weights
    - Shows comparison visualization

21. **Optional**: Compare results
    - Compare `final_mask` (Cell 12) vs `final_mask_opt` (Cell 19)
    - Compare detected centroids
    - Check if optimization improved accuracy

---

## Quick Start Workflow

### For Quick Testing (No Optimization):
```
Run Cells: 0 → 1 → 2 → 3 → 4 → 6 → 7 → 8 → 10 → 11 → 12 → 14 → 15
```

### For Optimized Detection (RECOMMENDED):
```
1. Run Cells: 0 → 1 → 2 → 3 → 4 → 6 → 7 → 8 → 10
2. Run Cell 16 (CSO Optimization) - Wait for completion (~5-10 minutes)
3. Run Cell 18 (Detection with optimized parameters) - AUTOMATIC
4. Run Cell 19 (Optimized scoring and final detection) - AUTOMATIC
5. Compare results: Cell 12 (original) vs Cell 19 (optimized)
```

---

## Important Notes

1. **Image Selection**: Change `image_name` in Cell 2 to test different images
2. **Ground Truth**: Cell 16 automatically loads ground truth from `train_dataset.csv` if available
3. **Optimization Time**: Cell 16 takes 5-10 minutes depending on iterations (30 iterations by default)
4. **Automatic Application**: Cells 18-19 automatically use optimized parameters - no manual editing needed!

---

## Comparison Workflow

### Original Detection (Default Parameters):
- **Cells 11-12**: Original detection with default parameters
- Results stored in: `final_mask`, `final_tumor_mask`

### Optimized Detection (CSO Parameters):
- **Cells 18-19**: Optimized detection with CSO-tuned parameters  
- Results stored in: `final_mask_opt`, `final_tumor_mask_opt`

### Compare Results:
- Run both workflows and compare:
  - Detection accuracy (if ground truth available)
  - Detected centroids
  - Tumor scores
  - Visualizations

