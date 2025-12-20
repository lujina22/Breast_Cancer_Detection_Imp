# input.py
import cv2
import numpy as np
import pydicom

def get_mass_data(image):

    # Case: Mass pipeline disabled
    if image is None:
        mass_region = np.zeros((1024, 1024), dtype=np.uint8)
        clean_mask = np.zeros((1024, 1024), dtype=np.uint8)
        return mass_region, clean_mask

    # --- Use the image passed in ---
    img = image.astype(np.float32)

    # --- Normalize to uint8 ---
    img_uint8 = cv2.normalize(
        img, None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)

    # --- Blur & equalize ---
    img_blur = cv2.GaussianBlur(img_uint8, (5, 5), 0)
    img_eq = cv2.equalizeHist(img_blur)

    # --- Threshold ---
    _, thresh = cv2.threshold(img_eq, 5, 255, cv2.THRESH_BINARY)

    # --- Largest connected component ---
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        thresh, connectivity=8
    )

    if num_labels <= 1:
        return None, None

    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])

    clean_mask = np.zeros_like(thresh)
    clean_mask[labels == largest_label] = 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel)

    # --- Apply mask ---
    masked_img = cv2.bitwise_and(img_eq, img_eq, mask=clean_mask)

    # --- K-means segmentation ---
    pixel_values = masked_img.reshape((-1, 1)).astype(np.float32)

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        100,
        0.2
    )

    K = 3
    _, labels, centers = cv2.kmeans(
        pixel_values, K, None, criteria, 10, cv2.KMEANS_PP_CENTERS
    )

    centers = np.uint8(centers)
    segmented = centers[labels.flatten()].reshape(masked_img.shape)

    # --- Brightest cluster ---
    brightest_cluster = np.argmax(centers)
    brightest_mask = np.zeros_like(segmented, dtype=np.uint8)
    brightest_mask[segmented == centers[brightest_cluster]] = 255

    brightest_mask = cv2.morphologyEx(
        brightest_mask, cv2.MORPH_CLOSE, kernel
    )

    mass_region = cv2.bitwise_and(
        masked_img, masked_img, mask=brightest_mask
    )

    return mass_region, brightest_mask
