# node_metrics.py

import numpy as np
from skimage.morphology import skeletonize
from scipy.ndimage import convolve
from scipy.spatial import cKDTree
from tqdm import tqdm

def find_valence_points(skeleton):
    """
    Compute valence (neighbor count) at each skeleton pixel (0–8).
    """
    skeleton = (skeleton > 0).astype(np.uint8) * 255  # Ensure 0/255 format
    kernel = np.array([[1,1,1],[1,10,1],[1,1,1]], dtype=np.int16)
    convolved = convolve(skeleton.astype(np.int16), kernel, mode='constant', cval=0)
    neighbor_sum = convolved - 2550  # 255 * 10 for center pixel
    neighbor_counts = neighbor_sum // 255
    neighbor_counts = np.clip(neighbor_counts, 0, 8)
    return neighbor_counts

def extract_points(valence_map, valence):
    return np.argwhere(valence_map == valence)

def bipartite_match(pred_points, true_points, max_dist=3):
    if len(pred_points) == 0 or len(true_points) == 0:
        return 0, len(pred_points), len(true_points)

    pred_tree = cKDTree(pred_points)
    matched_pred = set()
    matched_true = set()

    for i, pt in enumerate(true_points):
        dist, idx = pred_tree.query(pt, distance_upper_bound=max_dist)
        if idx != len(pred_points) and idx not in matched_pred:
            matched_pred.add(idx)
            matched_true.add(i)

    tp = len(matched_true)
    fp = len(pred_points) - len(matched_pred)
    fn = len(true_points) - tp
    return tp, max(0, fp), max(0, fn)

def compute_node_metrics(pred, target, max_dist=3):
    pred = (pred > 0.5).astype(np.uint8)
    target = (target > 0.5).astype(np.uint8)

    pred_skel = skeletonize(pred)
    target_skel = skeletonize(target)

    pred_valence = find_valence_points(pred_skel)
    target_valence = find_valence_points(target_skel)

    results = {}

    for val in tqdm(range(1, 5), desc="Computing node metrics by valence"):
        pred_points = extract_points(pred_valence, val)
        true_points = extract_points(target_valence, val)
        print(f"Valence {val} — GT: {len(true_points)}, Pred: {len(pred_points)}")

        tp, fp, fn = bipartite_match(pred_points, true_points, max_dist)
        results[val] = {
            'precision': tp / (tp + fp + 1e-6),
            'recall': tp / (tp + fn + 1e-6),
            'tp': tp,
            'fp': fp,
            'fn': fn
        }
    return results
