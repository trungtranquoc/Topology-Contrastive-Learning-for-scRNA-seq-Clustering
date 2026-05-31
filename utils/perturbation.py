import torch

def dropout_event(X_data: torch.Tensor, drop_prob: float):
    """Randomly zero out individual elements of X with probability drop_prob."""
    # Create a mask of the exact same shape as the input data
    drop_mask = torch.empty_like(X_data).bernoulli_(drop_prob).bool()
    X_data = X_data.clone()
    X_data[drop_mask] = 0 # Applies mask element-wise
    return X_data

def gaussian_noise_event(X_data: torch.Tensor, mean: float = 0.0, std: float = 1.0):
    """Add Gaussian noise to X_data."""
    noise = torch.randn_like(X_data) * std + mean
    return X_data + noise

def edge_dropout_event(edge_index: torch.Tensor, drop_prob: float, force_undirected: bool = False):
    """Randomly drop edges from the edge_index with probability drop_prob."""
    row, col = edge_index
    edge_mask = torch.bernoulli(torch.ones(row.size(0), device=edge_index.device) * (1 - drop_prob)).type(torch.bool)

    if force_undirected:
        edge_mask[row > col] = False

    edge_index = edge_index[:, edge_mask]

    if force_undirected:
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        edge_mask = edge_mask.nonzero().repeat((2, 1)).squeeze()

    return edge_index, edge_mask


