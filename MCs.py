# MCs.py
import cv2
import numpy as np


def get_mc_data(image):

    # -------------------------------
    # Case 1: MC pipeline disabled
    # -------------------------------
    if image is None:
        mc_contours = []
        tophat_norm = None
        return mc_contours, tophat_norm

    # -------------------------------
    # Use the image passed in
    # -------------------------------
    original = image.copy()

    # Ensure grayscale
    if len(original.shape) == 3:
        img = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    else:
        img = original

    img = cv2.resize(img, (1024, 1024), interpolation=cv2.INTER_CUBIC)

    # -------------------------------
    # Denoise + enhance
    # -------------------------------
    noise_sigma = img.std()
    med_k = 3 if noise_sigma < 30 else 5
    img_med = cv2.medianBlur(img, med_k)

    clip = np.interp(noise_sigma, [0, 50], [1.5, 3.0])
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img_med)

    # -------------------------------
    # Top-hat
    # -------------------------------
    H, W = img.shape
    diag = np.sqrt(H * H + W * W)
    base = int(np.clip(diag / 20, 40, 100))

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (base, base)
    )

    opened = cv2.morphologyEx(img_clahe, cv2.MORPH_OPEN, kernel)
    tophat = cv2.subtract(img_clahe, opened)
    tophat_norm = cv2.normalize(
        tophat, None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)

    # -------------------------------
    # Threshold
    # -------------------------------
    thresh_val = np.percentile(tophat_norm, 99.0)
    _, binary = cv2.threshold(
        tophat_norm, int(thresh_val), 255, cv2.THRESH_BINARY
    )

    # -------------------------------
    # Remove small objects
    # -------------------------------
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    cleaned = np.zeros_like(binary)

    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= 50:
            cleaned[labels == i] = 255

    # -------------------------------
    # Contours
    # -------------------------------
    contours, _ = cv2.findContours(
        cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    mc_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 50:
            continue

        perim = cv2.arcLength(cnt, True)
        if perim == 0:
            continue

        circularity = 4 * np.pi * area / (perim ** 2)
        if 0.2 <= circularity <= 1.8:
            mc_contours.append(cnt)

    return mc_contours, tophat_norm
