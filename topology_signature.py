from torch import nn
import torch
import numpy as np
from typing import Literal, Optional, Callable
from aleph import vietoris_rips_from_matrix
from math import prod
from utils.distance_fn import get_euclidean_distance_matrix, get_similarity_distance_matrix

import torch.nn.functional as F
TopologyLossType = Literal["MSE", "Log-Ratio", "Cosine-Log", "Topo-KL"]
EPS = 1e-6
T = 1.0

def symmetric_kl(p, q, eps=1e-8):
    kl_pq = p * torch.log((p + eps) / (q + eps))
    kl_qp = q * torch.log((q + eps) / (p + eps))
    return 0.5 * torch.sum(kl_pq + kl_qp)

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

    def __init__(self, max_dimension: int = 1, loss_type: TopologyLossType = "MSE", distance_type: Literal["euclidean", "similarity"] = "euclidean"):
        
        super(TopologySignatureLoss, self).__init__()
        self.max_dimension = max_dimension
        self.loss_type = loss_type

        # Construct distance matrix function
        if distance_type == "euclidean":
            self.dist_matrix_fn = get_euclidean_distance_matrix
        elif distance_type == "similarity":
            self.dist_matrix_fn = get_similarity_distance_matrix
        else:
            raise ValueError(f"Invalid distance_type: {distance_type}. Must be 'euclidean' or 'similarity'.")

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
        edge_pairs = []
        if len(pairs) > 0:
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

        if len(y_X) == 0 or len(y_Z) == 0:
            return torch.tensor(0.0, device=X_tensor.device)

        # Extract the critical edges of mX and mZ w.r.t y_X
        val_mX_yX = torch.stack([mX[x, y] for x, y in y_X])
        val_mZ_yX = torch.stack([mZ[x, y] for x, y in y_X])

        # Extract the critical edges of mX and mZ w.r.t y_Z
        val_mX_yZ = torch.stack([mX[x, y] for x, y in y_Z])
        val_mZ_yZ = torch.stack([mZ[x, y] for x, y in y_Z])

        if len(y_X) == 0 or len(y_Z) == 0:
            return torch.tensor(0.0, device=X_tensor.device)

        if self.loss_type == "MSE":
            loss = torch.sum((val_mX_yX - val_mZ_yX) ** 2) + torch.sum((val_mX_yZ - val_mZ_yZ) ** 2)
        elif self.loss_type == "Log-Ratio":
            loss_part1 = torch.sum((torch.log(val_mX_yX + EPS) - torch.log(val_mZ_yX + EPS)) ** 2)
            loss_part2 = torch.sum((torch.log(val_mX_yZ + EPS) - torch.log(val_mZ_yZ + EPS)) ** 2)
            loss = loss_part1 + loss_part2
        elif self.loss_type == "Cosine-Log":
            sim_mX_yX = torch.exp(-val_mX_yX)
            sim_mZ_yX = torch.exp(-val_mZ_yX)
            sim_mX_yZ = torch.exp(-val_mX_yZ)
            sim_mZ_yZ = torch.exp(-val_mZ_yZ)
            
            # 2. Compute the logarithmic binary cross-entropy alignment
            loss_part1 = -torch.sum(sim_mX_yX * torch.log(sim_mZ_yX + EPS) + (1.0 - sim_mX_yX) * torch.log(1.0 - sim_mZ_yX + EPS))
            loss_part2 = -torch.sum(sim_mX_yZ * torch.log(sim_mZ_yZ + EPS) + (1.0 - sim_mX_yZ) * torch.log(1.0 - sim_mZ_yZ + EPS))
            
            loss = loss_part1 + loss_part2

        elif self.loss_type == "Topo-KL":
            # 1. Compute the softmax of the negative distances
            p_mX_yX = F.softmax(-val_mX_yX / T, dim=0)
            p_mZ_yX = F.softmax(-val_mZ_yX / T, dim=0)
            
            p_mX_yZ = F.softmax(-val_mX_yZ / T, dim=0)
            p_mZ_yZ = F.softmax(-val_mZ_yZ / T, dim=0)
            loss_part1 = symmetric_kl(p_mX_yX, p_mZ_yX)
            loss_part2 = symmetric_kl(p_mX_yZ, p_mZ_yZ)
            
            loss = loss_part1 + loss_part2

        # Normalized by batch size and multiply by the product of the shape of Z_tensor except the batch size
        return loss / X_tensor.shape[0] * prod(s for s in Z_tensor.shape[1:])