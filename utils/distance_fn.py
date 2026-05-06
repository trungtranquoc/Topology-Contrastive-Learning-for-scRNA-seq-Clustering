import torch

def get_euclidean_distance_matrix(tensors: torch.Tensor) -> torch.Tensor:
    """
        Return euclidean distance matrix between all data points.
    """
    diff = tensors.unsqueeze(1) - tensors.unsqueeze(0)  # shape (n, n, d)
    d2 = (diff * diff).sum(-1)
    dist = torch.sqrt(d2 + 1e-12)  # add small epsilon for stability
    return dist
