import numpy as np
from skimage.morphology import skeletonize
from scipy.ndimage import convolve
from scipy.spatial import cKDTree

def find_valence_points(skeleton):
    """
    Compute valence (neighbor count) at each skeleton pixel (0–8).
    """
    # Ensure 0/255 image
    skeleton = skeleton.astype(np.uint8) * 255

    # Use a kernel that highlights the center (10) and neighbors (1)
    kernel = np.array([[1,1,1],[1,10,1],[1,1,1]], dtype=np.int16)

    # Convert skeleton to int16 before convolution to prevent overflow
    convolved = convolve(skeleton.astype(np.int16), kernel, mode='constant', cval=0)

    # Subtract the center pixel's weighted value (10*255) to get just neighbor sum
    neighbor_sum = convolved - 2550
    neighbor_counts = neighbor_sum // 255
    neighbor_counts = np.clip(neighbor_counts, 0, 8)

    return neighbor_counts


def extract_points(valence_map, valence):
    """
    Extract (y, x) points from the valence map for a given valence.
    """
    points = np.argwhere(valence_map == valence)
    return points

def bipartite_match(pred_points, true_points, max_dist=7):
    if len(pred_points) == 0 or len(true_points) == 0:
        return 0, len(pred_points), len(true_points)

    pred_tree = cKDTree(pred_points)
    true_tree = cKDTree(true_points)

    matched_pred = set()
    matched_true = set()

    for i, pt in enumerate(true_points):
        dist, idx = pred_tree.query(pt, distance_upper_bound=max_dist)
        if idx < len(pred_points) and idx not in matched_pred:
            matched_pred.add(idx)
            matched_true.add(i)

    tp = len(matched_true)
    fp = len(pred_points) - len(matched_pred)
    fn = len(true_points) - tp
    return tp, max(0, fp), max(0, fn)


def compute_node_metrics(pred, target, max_dist=3):
    """
    Full pipeline: skeletonize prediction/target, find valences, and compute precision/recall for 1-4 valent nodes.
    """
    if pred.max() > 1:
        pred = pred / 255.0
    if target.max() > 1:
        target = target / 255.0

    pred = pred > 0.5
    target = target > 0.5

    pred_skel = skeletonize(pred)
    target_skel = skeletonize(target)

    pred_valence = find_valence_points(pred_skel)
    target_valence = find_valence_points(target_skel)

    results = {}

    for val in [1, 2, 3, 4]:
        pred_points = extract_points(pred_valence, valence=val)
        true_points = extract_points(target_valence, valence=val)
        print(f"Valence {val} — GT: {len(true_points)}, Pred: {len(pred_points)}")

        tp, fp, fn = bipartite_match(pred_points, true_points, max_dist=max_dist)

        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)

        results[val] = {
            'precision': precision,
            'recall': recall,
            'tp': tp,
            'fp': fp,
            'fn': fn
        }
    
    return results