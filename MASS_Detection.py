# input.py
import cv2
import numpy as np
import math
import random

# ==========================================
# PARAMETERS
# ==========================================
BORDER_MARGIN = 50
K = 8

np.random.seed(42)
random.seed(42)

# ==========================================
# BREAST MASK
# ==========================================
def extract_breast_mask(img):
    _, binary = cv2.threshold(
        img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return np.ones_like(img, dtype=np.uint8) * 255

    largest = max(contours, key=cv2.contourArea)
    mask = np.zeros_like(img, dtype=np.uint8)
    cv2.drawContours(mask, [largest], -1, 255, -1)
    return mask

# ==========================================
# PREPROCESSING
# ==========================================
def preprocess_mammogram(image):
    if image is None:
        raise ValueError("Image is None")

    breast_mask = extract_breast_mask(image)
    img_roi = cv2.bitwise_and(image, image, mask=breast_mask)
    img_eq = cv2.equalizeHist(img_roi)
    img_med = cv2.medianBlur(img_eq, 3)

    data = img_med[breast_mask == 255].reshape(-1, 1).astype(np.float32)
    return img_med, data, breast_mask

# ==========================================
# CSO + KMeans++
# ==========================================
def fitness_function(centroids, data):
    distances = np.abs(data - centroids.T)
    return np.sum(np.min(distances, axis=1) ** 2)

def assign_clusters(centroids, data):
    distances = np.abs(data - centroids.T)
    return np.argmin(distances, axis=1)

def levy_flight(beta=1.5):
    sigma = (
        math.gamma(1 + beta)
        * math.sin(math.pi * beta / 2)
        / (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))
    ) ** (1 / beta)
    u = np.random.normal(0, sigma)
    v = np.random.normal(0, 1)
    return u / (abs(v) ** (1 / beta))

def kmeans_plus_plus(data, k):
    n = data.shape[0]
    centroids = np.zeros((k, 1))
    centroids[0] = data[np.random.randint(n)]

    for i in range(1, k):
        dists = np.min(np.abs(data - centroids[:i].T), axis=1)
        probs = dists ** 2 / np.sum(dists ** 2)
        centroids[i] = data[np.searchsorted(np.cumsum(probs), np.random.rand())]

    return centroids

def KMpp_CSO(data, k, n_nests=15, pa=0.25, max_gen=50):
    nests = np.zeros((n_nests, k, 1))
    nests[0] = kmeans_plus_plus(data, k)

    min_val, max_val = np.min(data), np.max(data)
    for i in range(1, n_nests):
        nests[i] = np.random.uniform(min_val, max_val, (k, 1))

    fitness = np.array([fitness_function(n, data) for n in nests])

    for _ in range(max_gen):
        for i in range(n_nests):
            step = levy_flight()
            new_nest = nests[i] + step * np.random.randn(k, 1)
            new_nest = np.clip(new_nest, min_val, max_val)
            new_fit = fitness_function(new_nest, data)

            j = random.randint(0, n_nests - 1)
            if new_fit < fitness[j]:
                nests[j] = new_nest
                fitness[j] = new_fit

        abandon = int(pa * n_nests)
        worst = np.argsort(fitness)[-abandon:]
        for idx in worst:
            nests[idx] = np.random.uniform(min_val, max_val, (k, 1))
            fitness[idx] = fitness_function(nests[idx], data)

    best = nests[np.argmin(fitness)]
    labels = assign_clusters(best, data)
    return best, labels

# ==========================================
# SEGMENTATION
# ==========================================
def reconstruct_segmented(labels, centroids, breast_mask, shape):
    segmented = np.zeros(shape, dtype=np.uint8)
    segmented_vals = centroids[labels].astype(np.uint8).flatten()
    segmented[breast_mask == 255] = segmented_vals
    return segmented

# ==========================================
# MASS EXTRACTION
# ==========================================
def extract_mass_mask(segmented_img):
    values = np.unique(segmented_img)

    best_val = None
    best_score = -1

    for v in values:
        if v == 0:
            continue

        mask = np.uint8(segmented_img == v) * 255
        area = cv2.countNonZero(mask)
        if area < 400:
            continue

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cnt = max(contours, key=cv2.contourArea)
        perim = cv2.arcLength(cnt, True)
        circularity = 4 * np.pi * area / (perim ** 2 + 1e-6)

        mean_intensity = np.mean(segmented_img[segmented_img == v])
        score = 0.6 * (mean_intensity / 255.0) + 0.4 * circularity

        if score > best_score:
            best_score = score
            best_val = v

    if best_val is None:
        return np.zeros_like(segmented_img, dtype=np.uint8)

    return np.uint8(segmented_img == best_val) * 255

def highest_intensity_component_away_from_border(mask, reference, margin):
    h, w = mask.shape
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)

    best_label = None
    best_intensity = -1

    for i in range(1, num_labels):
        x, y, bw, bh, area = stats[i]
        if x < margin or y < margin or x + bw > w - margin or y + bh > h - margin:
            continue

        mean_i = np.mean(reference[labels == i])
        if mean_i > best_intensity:
            best_intensity = mean_i
            best_label = i

    if best_label is None:
        return np.zeros_like(mask)

    return np.uint8(labels == best_label) * 255

# ==========================================
# WRAPPER
# ==========================================
def get_mass_data(image):
    if image is None:
        return None, None

    img_proc, data, breast_mask = preprocess_mammogram(image)

    centroids, labels = KMpp_CSO(data, K)
    segmented = reconstruct_segmented(labels, centroids, breast_mask, image.shape)

    mass_mask = extract_mass_mask(segmented)
    mass_mask = highest_intensity_component_away_from_border(
        mass_mask, img_proc, BORDER_MARGIN
    )
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mass_mask = cv2.morphologyEx(
        mass_mask, cv2.MORPH_OPEN, kernel, iterations=2
    )

    if cv2.countNonZero(mass_mask) == 0:
        return None, None

    mass_region = cv2.bitwise_and(image, image, mask=mass_mask)
    return mass_region, mass_mask
