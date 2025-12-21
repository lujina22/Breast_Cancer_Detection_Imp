/**
 * MASS Detection Web UI - Frontend JavaScript
 * Handles file upload, drag-and-drop, image preview, and result display
 */

// DOM Elements
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const browseBtn = document.getElementById('browseBtn');
const previewContainer = document.getElementById('previewContainer');
const previewImage = document.getElementById('previewImage');
const removeBtn = document.getElementById('removeBtn');
const detectBtn = document.getElementById('detectBtn');
const uploadSection = document.getElementById('uploadSection');
const resultsSection = document.getElementById('resultsSection');
const newAnalysisBtn = document.getElementById('newAnalysisBtn');
const errorMessage = document.getElementById('errorMessage');
const errorText = document.getElementById('errorText');
const errorClose = document.getElementById('errorClose');

// State
let selectedFile = null;

// ==========================================
// EVENT LISTENERS
// ==========================================

// Browse button click
browseBtn.addEventListener('click', () => {
    fileInput.click();
});

// File input change
fileInput.addEventListener('change', (e) => {
    handleFileSelect(e.target.files[0]);
});

// Drag and drop events
dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('drag-over');
});

dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('drag-over');
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
    handleFileSelect(e.dataTransfer.files[0]);
});

// Click on drop zone to browse
dropZone.addEventListener('click', (e) => {
    if (e.target === dropZone || e.target.closest('.drop-zone-content')) {
        fileInput.click();
    }
});

// Remove image button
removeBtn.addEventListener('click', () => {
    resetUpload();
});

// Detect button
detectBtn.addEventListener('click', () => {
    if (selectedFile) {
        processImage();
    }
});

// New analysis button
newAnalysisBtn.addEventListener('click', () => {
    resetAll();
});

// Error close button
errorClose.addEventListener('click', () => {
    hideError();
});

// ==========================================
// FILE HANDLING
// ==========================================

/**
 * Handle file selection from input or drag-and-drop
 */
function handleFileSelect(file) {
    if (!file) {
        return;
    }

    // Validate file type
    const validTypes = ['image/png', 'image/jpeg', 'image/jpg', 'image/tiff', 'image/x-portable-graymap'];
    const validExtensions = ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.pgm'];

    const isValidType = validTypes.includes(file.type) ||
        validExtensions.some(ext => file.name.toLowerCase().endsWith(ext));

    if (!isValidType) {
        showError('Invalid file type. Please upload an image file (PNG, JPG, JPEG, PGM, TIF, TIFF).');
        return;
    }

    // Validate file size (max 16MB)
    if (file.size > 16 * 1024 * 1024) {
        showError('File size too large. Maximum allowed size is 16MB.');
        return;
    }

    selectedFile = file;
    displayPreview(file);
}

/**
 * Display image preview
 */
function displayPreview(file) {
    const reader = new FileReader();

    reader.onload = (e) => {
        previewImage.src = e.target.result;
        previewContainer.style.display = 'block';
        dropZone.style.display = 'none';
    };

    reader.onerror = () => {
        showError('Failed to read the image file. Please try again.');
        resetUpload();
    };

    reader.readAsDataURL(file);
}

/**
 * Reset upload section to initial state
 */
function resetUpload() {
    selectedFile = null;
    fileInput.value = '';
    previewImage.src = '';
    previewContainer.style.display = 'none';
    dropZone.style.display = 'block';
}

/**
 * Reset entire application to initial state
 */
function resetAll() {
    resetUpload();
    resultsSection.style.display = 'none';
    uploadSection.style.display = 'block';

    // Scroll to top
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

// ==========================================
// IMAGE PROCESSING
// ==========================================

/**
 * Send image to backend for processing
 */
async function processImage() {
    if (!selectedFile) {
        showError('No file selected.');
        return;
    }

    // Show loading state
    setLoadingState(true);
    hideError();

    // Create FormData
    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (data.error) {
            showError(data.error);
            setLoadingState(false);
            return;
        }

        if (data.success) {
            displayResults(data);
            // Scroll to results
            setTimeout(() => {
                resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }, 300);
        }

    } catch (error) {
        showError(`Network error: ${error.message}. Please check your connection and try again.`);
    } finally {
        setLoadingState(false);
    }
}

/**
 * Display detection results
 */
function displayResults(data) {
    // Set result images
    document.getElementById('resultOverlay').src = data.original_with_mask;
    document.getElementById('resultIsolated').src = data.isolated_tumor;
    document.getElementById('resultCentroid').src = data.centroid_circle;

    // Display classification if available
    if (data.classification) {
        let classificationHTML = `<h3 style="text-align: center; margin-top: 20px; color: #4ade80;">
            ✅ Classification: ${data.classification}`;

        if (data.confidence) {
            classificationHTML += ` (${data.confidence}% confidence)`;
        }

        classificationHTML += '</h3>';

        // Insert classification before results grid
        const resultsGrid = document.querySelector('.results-grid');
        const existingClassification = document.getElementById('classification-result');

        if (existingClassification) {
            existingClassification.remove();
        }

        const classificationDiv = document.createElement('div');
        classificationDiv.id = 'classification-result';
        classificationDiv.innerHTML = classificationHTML;
        resultsGrid.parentNode.insertBefore(classificationDiv, resultsGrid);
    }

    // Show results section
    resultsSection.style.display = 'block';
    uploadSection.style.display = 'none';
}

/**
 * Set loading state for detect button
 */
function setLoadingState(isLoading) {
    const btnText = detectBtn.querySelector('.btn-text');
    const btnLoader = detectBtn.querySelector('.btn-loader');

    if (isLoading) {
        btnText.style.display = 'none';
        btnLoader.style.display = 'inline-flex';
        detectBtn.disabled = true;
    } else {
        btnText.style.display = 'inline';
        btnLoader.style.display = 'none';
        detectBtn.disabled = false;
    }
}

// ==========================================
// ERROR HANDLING
// ==========================================

/**
 * Show error message
 */
function showError(message) {
    errorText.textContent = message;
    errorMessage.style.display = 'flex';

    // Auto-hide after 8 seconds
    setTimeout(() => {
        hideError();
    }, 8000);
}

/**
 * Hide error message
 */
function hideError() {
    errorMessage.style.display = 'none';
}

// ==========================================
// UTILITY FUNCTIONS
// ==========================================

/**
 * Prevent default drag behavior on document
 */
document.addEventListener('dragover', (e) => {
    e.preventDefault();
});

document.addEventListener('drop', (e) => {
    e.preventDefault();
});

// ==========================================
// INITIALIZATION
// ==========================================

console.log('🚀 MASS Detection UI initialized');
console.log('Ready to process mammogram images');
