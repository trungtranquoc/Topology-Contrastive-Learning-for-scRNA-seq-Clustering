import torch

def get_euclidean_distance_matrix(tensors: torch.Tensor) -> torch.Tensor:
    """
        Return euclidean distance matrix between all data points.
    """
    diff = tensors.unsqueeze(1) - tensors.unsqueeze(0)  # shape (n, n, d)
    d2 = (diff * diff).sum(-1)
    dist = torch.sqrt(d2 + 1e-12)  # add small epsilon for stability
    return dist

def get_similarity_distance_matrix(tensors: torch.Tensor) -> torch.Tensor:
    """
        Return similarity distance matrix between all data points.
    """
    similarity = torch.matmul(tensors, tensors.T)
    norms = torch.norm(tensors, dim=1)
    similarity = similarity / (norms.unsqueeze(1) * norms.unsqueeze(0) + 1e-12)  # add small epsilon for stability
    distance = 1 - similarity
    return distance