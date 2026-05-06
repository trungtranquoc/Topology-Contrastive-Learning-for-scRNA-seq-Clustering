from scipy.optimize import linear_sum_assignment
import numpy as np

def cluster_acc(y_true, y_pred):
    """
    Computes clustering accuracy.
    Args:
        y_true ([type]): [description]
        y_pred ([type]): [description]

    Returns:
        [type]: [description]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    return sum([w[i, j] for i, j in zip(row_ind, col_ind)]) * 1.0 / y_pred.size

def compute_F1_score(y_true, y_pred):
    """
    Computes macro and micro F1 scores for clustering results
    using the Hungarian algorithm for optimal label alignment.
    """
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    assert y_pred.size == y_true.size
    
    # Define matrix dimensions
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    
    # Build the confusion matrix
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    # Optimal assignment using the Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(w.max() - w)

    f1_per_class = []
    total_TP = 0

    # Calculate per-class F1 for Macro, and accumulate global TP for Micro
    for i, j in zip(row_ind, col_ind):
        TP = w[i, j]
        total_TP += TP
        
        # FP: Total predictions for this cluster minus the correct ones
        FP = w[i, :].sum() - TP
        # FN: Total true instances of this class minus the ones found
        FN = w[:, j].sum() - TP
        
        # Compute F1 for this specific class pair
        denominator = 2 * TP + FP + FN
        f1 = (2 * TP / denominator) if denominator > 0 else 0
        f1_per_class.append(f1)

    # F1 Macro: Unweighted average of per-class F1 scores
    F1_macro = np.mean(f1_per_class)

    # F1 Micro: Global calculation
    # In this mutually exclusive assignment context, global_FP == global_FN
    total_samples = y_pred.size
    global_FP = total_samples - total_TP
    global_FN = total_samples - total_TP
    
    micro_denominator = 2 * total_TP + global_FP + global_FN
    F1_micro = (2 * total_TP / micro_denominator) if micro_denominator > 0 else 0

    return F1_macro, F1_micro