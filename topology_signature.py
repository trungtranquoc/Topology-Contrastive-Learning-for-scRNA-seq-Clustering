from torch import nn
import torch
import numpy as np
from typing import Optional, Callable
from aleph import vietoris_rips_from_matrix
from math import prod
from utils.distance_fn import get_euclidean_distance_matrix

class TopologySignatureLoss(nn.Module):
    """
    Compute the topology signature loss. Where the loss function is computed as:
        Loss(D(X), D(Z)) = 1/2 * (|| M_X(y_X) - M_X(y_Z) ||^2 + || M_Z(y_Z) - M_Z(y_X) ||^2)
    where: 
        M_X and M_Z are distance matrices of the point clouds X and Z respectively.
        y_X and y_Z are set of persistence pairs (birth, death) from the Vietoris-Rips filtration of X and Z respectively.
        D(X) and D(Z) are the persistence diagrams of X and Z respectively.

    Args:
        max_dimension: int, the maximum homology dimension to consider (default: 2)
        batch_size: Optional[int], if specified, compute the loss in batches (default: None). Low batch_size can reduce the runtime but may skip those high-dimensional features.
        shuffle_batch: bool, whether to shuffle the data before batching (default: True)
        traceback_error_skip: bool, whether to skip traceback in error messages (default: True)
        turnoff_traceback_warning: bool, whether to turn off traceback warning messages (default: True)
        dist_matrix_fn: callable[[torch.Tensor], np.ndarray]. If no dist_fn specified, use euclidean distance matrix and used numpy linalg.norm to optimize performance (default: None).
    """
    max_dimension: int

    def __init__(self, max_dimension: int = 1):
        
        super(TopologySignatureLoss, self).__init__()
        self.max_dimension = max_dimension

        # Construct distance matrix function
        self.dist_matrix_fn = get_euclidean_distance_matrix

    def _traceback_index(self, edge_distance: float, distance_matrix: np.ndarray) -> tuple[int, int]:
        """Given an edge distance, find the corresponding edge (i, j) in the distance matrix. Find the edge that created the feature by looking at the distance matrix.

        Args:
            edge_distance: edge distance
            distance_matrix: numpy array of distances

        Returns:
            - Edge (i, j) as tuple of indices
        """
        indices = np.where(distance_matrix.detach().cpu() == edge_distance)
        if len(indices[0]) == 0:
            return -1, -1
        indices = (indices[0].tolist(), indices[1].tolist())

        return indices[0][0], indices[1][0]
    
    def _get_indices_pairs(self, distance_matrix: np.ndarray) -> list[tuple[int, int]]:
        """Given a tensor of shape (N, d), compute the distance matrix and return the edge pairs that create the Homology features (from H0 to H_dimension).

        Args:
            tensor: numpy array of shape (N, d)
        Returns:
            - List of edge pairs (i, j) as tuples of indices
        """
        pairs = vietoris_rips_from_matrix(distance_matrix.detach().cpu().numpy(), max_dimension=self.max_dimension)

        # Get edge pairs from H0 first. For H0, the birth is always 0, so we only care about the death
        edge_pairs = [self._traceback_index(point.y, distance_matrix) for point in pairs[0] if point.y != float('inf')]
        
        # Then get edge pairs from H1 to H_dimension
        for dim in range(1, len(pairs)):
            pairs[dim].removeDiagonal()  # Remove diagonal points (0 persistence)
            edge_pairs.extend(self._traceback_index(point.x, distance_matrix) for point in pairs[dim])
            edge_pairs.extend(self._traceback_index(point.y, distance_matrix) for point in pairs[dim])
            
        edge_pairs = [pair for pair in edge_pairs if pair != (-1, -1)]  # Remove invalid pairs
        return edge_pairs
    
    def forward_by_batch(self, X_tensor: np.ndarray, Z_tensor: np.ndarray, batch_size: int = 64) -> torch.Tensor:
        """
            Compute persistence loss but by breaking into batches. Two point clouds are broken into batches (same permutations) and compute the loss for each batch and then average the loss across batches. 
        """
        permutation_indices = torch.randperm(X_tensor.shape[0])
        X_tensor = X_tensor[permutation_indices]
        Z_tensor = Z_tensor[permutation_indices]

        num_batches = (X_tensor.shape[0] + batch_size - 1) // batch_size
        total_loss = torch.tensor(0.0, device=X_tensor.device)

        for i in range(num_batches):
            start_index = i * batch_size
            end_index = min((i + 1) * batch_size, X_tensor.shape[0])
            X_batch = X_tensor[start_index:end_index]
            Z_batch = Z_tensor[start_index:end_index]

            loss = self.forward(X_batch, Z_batch)
            total_loss += loss

        return total_loss / num_batches

    def forward(self, X_tensor: np.ndarray, Z_tensor: np.ndarray, 
                 y_X: Optional[list[tuple[int, int]]] = None, y_Z: Optional[list[tuple[int, int]]] = None) -> torch.Tensor:
        """
            Compute persistence loss between two tensors. Where y_X and y_Z are the persistence index pairs. If y_X and y_Z are not provided, compute them from the distance matrices of X_tensor and Z_tensor respectively. If y_X and y_Z are provided, use them to compute the loss directly without recomputing the persistence pairs. This is useful for batch training where we want to compute the loss for each batch without recomputing the persistence pairs for the whole dataset.
        """
        # Normalize for both to make sure the consistency loss
        X_norm = nn.functional.normalize(X_tensor, dim=0) / prod(s for s in X_tensor.shape[1:])
        Z_norm = nn.functional.normalize(Z_tensor, dim=0) / prod(s for s in Z_tensor.shape[1:])

        mX = self.dist_matrix_fn(X_norm)
        mZ = self.dist_matrix_fn(Z_norm)

        # Index of data cloud before encoding and after encoding
        if y_X is None:
            y_X = self._get_indices_pairs(mX)
        if y_Z is None:
            y_Z = self._get_indices_pairs(mZ)

        # Compute distances for the minimum spanning trees
        mX_yX = [mX[x, y] for x, y in y_X]
        mX_yZ = [mX[x, y] for x, y in y_Z]

        mZ_yX = [mZ[x, y] for x, y in y_X]
        mZ_yZ = [mZ[x, y] for x, y in y_Z]

        loss = 1/2 * sum((x - y)**2 for x, y in zip(mX_yX, mX_yZ)) + sum((x - y)**2 for x, y in zip(mZ_yZ, mZ_yX))
        
        # Normalized by batch size and multiply by the product of the shape of Z_tensor except the batch size
        return loss