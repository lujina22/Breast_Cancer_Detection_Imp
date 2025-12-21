"""
Flask Web Application for MASS Detection Pipeline
Provides a web interface for uploading mammogram images and viewing tumor detection results.
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
import cv2
import numpy as np
import os
from werkzeug.utils import secure_filename
import base64
from MASS_Detection import get_mass_data
from Feature_Extraction import extract_mass_features, extract_mc_features, build_master_vector
from MCs import get_mc_data
import joblib

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'pgm', 'tif', 'tiff'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load pre-trained SVM models
try:
    svm_stage1 = joblib.load('svm_stage1_pipeline.pkl')
    svm_stage2 = joblib.load('svm_stage2_pipeline.pkl')
    print("SVM models loaded successfully")
except Exception as e:
    print(f"Warning: Could not load SVM models: {e}")
    svm_stage1 = None
    svm_stage2 = None


def allowed_file(filename):
    """Check if the file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def encode_image_to_base64(image):
    """Convert OpenCV image to base64 string for web display."""
    _, buffer = cv2.imencode('.png', image)
    img_str = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/png;base64,{img_str}"


def classify_tumor(image, mass_mask):
    """
    Classify tumor using SVM models.
    Returns classification result and confidence.
    """
    if svm_stage1 is None or svm_stage2 is None:
        return "Classification unavailable (models not loaded)", 0.0
    
    try:
        # Extract MASS features
        mass_features = extract_mass_features(original_img=image, mass_mask=mass_mask)
        
        # Extract MC features
        mc_contours, tophat_norm = get_mc_data(image)
        mc_features = extract_mc_features(mc_contours, tophat_norm)
        
        # Build master feature vector
        master_vector = build_master_vector(mass_features, mc_features)
        features = np.array(master_vector).reshape(1, -1)
        
        # Stage 1: NORMAL vs ABNORMAL
        stage1_pred = svm_stage1.predict(features)[0]
        
        if stage1_pred == 1:  # NORMAL
            return "NORMAL", 0.0
        
        # Stage 2: MASS vs MICRO_CALCIFICATION (for abnormal cases)
        stage2_pred = svm_stage2.predict(features)[0]
        
        if stage2_pred == 0:  # MASS
            # For MASS cases, we need to determine BENIGN vs MALIGNANT
            # Since we only have 2 stages, we'll use prediction probabilities if available
            if hasattr(svm_stage2, 'predict_proba'):
                proba = svm_stage2.predict_proba(features)[0]
                confidence = max(proba) * 100
                # Higher confidence in MASS class suggests more concerning features
                if confidence > 70:
                    return "MASS - Suspicious", confidence
                else:
                    return "MASS - Detected", confidence
            return "MASS - Detected", 0.0
        else:  # MICRO_CALCIFICATION
            return "MICRO_CALCIFICATION", 0.0
            
    except Exception as e:
        print(f"Classification error: {e}")
        return f"Classification error: {str(e)}", 0.0


def create_visualizations(original_image, mass_mask):
    """
    Create three visualization outputs:
    1. Original image with mask overlay
    2. Isolated tumor region
    3. Centroid with bounding circle
    """
    # Convert grayscale to RGB for colored overlays
    if len(original_image.shape) == 2:
        original_rgb = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
    else:
        original_rgb = original_image.copy()
    
    # 1. Original with mask overlay (semi-transparent red)
    overlay = original_rgb.copy()
    mask_colored = np.zeros_like(original_rgb)
    mask_colored[mass_mask > 0] = [0, 0, 255]  # Red color for tumor
    overlay = cv2.addWeighted(original_rgb, 0.7, mask_colored, 0.3, 0)
    
    # 2. Isolated tumor region
    if len(original_image.shape) == 2:
        isolated = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
    else:
        isolated = original_image.copy()
    isolated[mass_mask == 0] = 0
    
    # 3. Centroid with circle
    centroid_img = original_rgb.copy()
    
    # Calculate centroid and minimum enclosing circle
    contours, _ = cv2.findContours(mass_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # Get the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Calculate moments for centroid
        M = cv2.moments(largest_contour)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            
            # Draw centroid
            cv2.circle(centroid_img, (cx, cy), 8, (0, 255, 0), -1)  # Green dot
            cv2.circle(centroid_img, (cx, cy), 10, (255, 255, 255), 2)  # White border
            
            # Draw minimum enclosing circle
            (x, y), radius = cv2.minEnclosingCircle(largest_contour)
            center = (int(x), int(y))
            radius = int(radius)
            cv2.circle(centroid_img, center, radius, (0, 255, 255), 3)  # Cyan circle
            
            # Add text with coordinates
            text = f"Centroid: ({cx}, {cy})"
            cv2.putText(centroid_img, text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    return overlay, isolated, centroid_img


@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and process the image through the MASS detection pipeline."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Please upload an image file.'}), 400
    
    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Read image (handle both grayscale and color images)
        image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        
        if image is None:
            return jsonify({'error': 'Failed to read image. Please ensure it\'s a valid image file.'}), 400
        
        # Run MASS detection pipeline
        mass_region, mass_mask = get_mass_data(image)
        
        if mass_region is None or mass_mask is None:
            return jsonify({
                'error': 'No tumor detected in the image. The detection algorithm could not identify a MASS region.'
            }), 200  # Return 200 but with error message for UI handling
        
        # Create three visualizations
        overlay, isolated, centroid = create_visualizations(image, mass_mask)
        
        # Classify the tumor
        classification, confidence = classify_tumor(image, mass_mask)
        
        # Convert images to base64 for JSON response
        results = {
            'success': True,
            'original_with_mask': encode_image_to_base64(overlay),
            'isolated_tumor': encode_image_to_base64(isolated),
            'centroid_circle': encode_image_to_base64(centroid),
            'classification': classification,
            'confidence': round(confidence, 2) if confidence > 0 else None
        }
        
        # Clean up uploaded file
        try:
            os.remove(filepath)
        except:
            pass
        
        return jsonify(results)
    
    except Exception as e:
        # Clean up uploaded file in case of error
        try:
            if 'filepath' in locals():
                os.remove(filepath)
        except:
            pass
        
        return jsonify({'error': f'Processing error: {str(e)}'}), 500


@app.route('/outputs/<filename>')
def output_file(filename):
    """Serve output files."""
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)


if __name__ == '__main__':
    print("=" * 60)
    print("MASS Detection Web UI Starting...")
    print("=" * 60)
    print("Access the application at: http://localhost:5000")
    print("=" * 60)
    app.run(debug=True, host='0.0.0.0', port=5000)

