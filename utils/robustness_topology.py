"""Frozen-model H0 robustness: diagram Wasserstein, bottleneck and persistence.

Callbacks have signature get_representation_fn(X, aux) -> [cells, features].
Appliers have signature (X, aux, level, seed) -> (X_perturbed, aux_perturbed).
1WD/2WD use L-infinity ground cost and the clean maximum distance as scale.
H0_bottleneck is unnormalized; each total persistence uses its own max distance.
Gaussian level means std = 0.1 * level. No training occurs in this module.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

from . import perturbation as perturbation_ops

DEFAULT_SEEDS = (11111, 22222, 33333, 44444, 55555)
DEFAULT_LEVELS = (0, 0.1, 0.2, 0.3, 0.4, 0.5)
GAUSSIAN_BASE_STD = 0.1
CANONICAL_METRICS = ("1WD", "2WD", "H0_bottleneck")
DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_device(device):
    global DEFAULT_DEVICE
    DEFAULT_DEVICE = torch.device(device)


def set_robustness_seed(seed):
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))


@contextmanager
def _seeded(seed):
    """Keep a sweep from changing the caller's Python/NumPy/Torch RNG state."""
    py_state, np_state = random.getstate(), np.random.get_state()
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    try:
        with torch.random.fork_rng(devices=devices):
            set_robustness_seed(seed)
            yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)


def _clone(value):
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    return deepcopy(value)


def _to_numpy(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


def _embedding(fn, x, aux):
    with torch.no_grad():
        result = _to_numpy(fn(_clone(x), _clone(aux)))
    if result.ndim != 2 or not result.size or not np.isfinite(result).all():
        raise ValueError("Representation must be a nonempty finite [cells, features] matrix.")
    return result


def _protocol(levels, seeds):
    levels, seeds = tuple(float(v) for v in levels), tuple(int(v) for v in seeds)
    if not levels or not seeds or any(not np.isfinite(v) or not 0 <= v <= 1 for v in levels):
        raise ValueError("Provide nonempty seeds and levels within [0, 1].")
    return levels, seeds


def _resolve_appliers(spec, appliers, perturbations):
    available = spec.get("perturbation_appliers") if appliers is None else appliers
    if not available:
        raise ValueError("Provide appliers or model['perturbation_appliers'].")
    names = tuple(available) if perturbations is None else tuple(perturbations)
    if not names or any(name not in available for name in names):
        raise ValueError(f"Requested perturbations {names} must exist in {tuple(available)}.")
    return {name: available[name] for name in names}


def _save(results, save_path, verbose):
    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(path, index=False)
        if verbose:
            print(f"Saved robustness results: {path}")
    return results


def make_default_appliers(perturbations=("gene_dropout", "gaussian_noise", "edge_dropout"),
                          gaussian_base_std=GAUSSIAN_BASE_STD, force_undirected=True):
    """Features may be NumPy or Torch; edge_dropout expects a [2, E] tensor.

    Perturbations are delegated to utils.perturbation. Undirected edge handling
    follows edge_dropout_event's semantics. Level zero is an identity.
    Use custom appliers for weighted, heterogeneous or multiple graphs.
    """
    if gaussian_base_std < 0 or not np.isfinite(gaussian_base_std):
        raise ValueError("gaussian_base_std must be finite and nonnegative.")
    names = tuple(perturbations)
    if set(names) - {"gene_dropout", "gaussian_noise", "edge_dropout"}:
        raise ValueError(f"Unknown perturbations: {names}")

    def apply(name, x, aux, level, seed):
        if not np.isfinite(level) or not 0 <= level <= 1:
            raise ValueError("level must be within [0, 1].")
        x, aux = _clone(x), _clone(aux)
        if level == 0:
            return x, aux
        with _seeded(seed):
            if name in ("gene_dropout", "gaussian_noise"):
                # perturbation.py operates on Torch tensors; adapt NumPy callers.
                numpy_input = not isinstance(x, torch.Tensor)
                tensor = torch.as_tensor(np.asarray(x)) if numpy_input else x
                if not tensor.is_floating_point():
                    tensor = tensor.float()
                if name == "gene_dropout":
                    tensor = perturbation_ops.dropout_event(tensor, drop_prob=float(level))
                else:
                    tensor = perturbation_ops.gaussian_noise_event(
                        tensor, mean=0.0, std=gaussian_base_std * float(level))
                x = tensor.detach().cpu().numpy() if numpy_input else tensor
            else:
                if not isinstance(aux, torch.Tensor) or aux.ndim != 2 or aux.shape[0] != 2:
                    raise ValueError("edge_dropout requires a [2, E] edge_index tensor.")
                aux, _ = perturbation_ops.edge_dropout_event(
                    aux, drop_prob=float(level), force_undirected=force_undirected)
        return x, aux

    return {name: (lambda x, aux, level, seed, name=name: apply(name, x, aux, level, seed))
            for name in names}


def default_cosine_distance_matrix(z, device=None):
    z = torch.as_tensor(_to_numpy(z), dtype=torch.float64,
                        device=DEFAULT_DEVICE if device is None else device)
    z = torch.nn.functional.normalize(z, dim=1, eps=1e-12)
    distance = (1 - z @ z.T).clamp(0, 2)
    distance.fill_diagonal_(0)
    return distance.cpu().numpy()


def _distance(z, distance_fn):
    d = np.asarray(distance_fn(z), dtype=np.float64)
    if d.shape != (len(z), len(z)) or not np.isfinite(d).all():
        raise ValueError("distance_fn must return a finite square distance matrix.")
    if np.min(d) < -1e-10 or not np.allclose(d, d.T):
        raise ValueError("Distances must be nonnegative and symmetric.")
    d = np.maximum((d + d.T) / 2, 0)
    np.fill_diagonal(d, 0)
    return np.ascontiguousarray(d)


def _h0_from_distance(distance):
    from aleph import vietoris_rips_from_matrix
    pairs = vietoris_rips_from_matrix(distance, max_dimension=1)
    points = [(float(p.x), float(p.y)) for p in pairs[0]
              if np.isfinite(p.x) and np.isfinite(p.y) and p.y > p.x] if len(pairs) else []
    return np.asarray(points, dtype=np.float64).reshape(-1, 2)


def finite_h0_diagram(z, distance_fn=default_cosine_distance_matrix):
    return _h0_from_distance(_distance(z, distance_fn))


def _diagram_cost(diag_a, diag_b):
    a, b = (np.asarray(d, dtype=np.float64).reshape(-1, 2) for d in (diag_a, diag_b))
    if any(not np.isfinite(d).all() or np.any(d[:, 1] < d[:, 0]) for d in (a, b)):
        raise ValueError("Diagrams must contain finite (birth, death) pairs with death >= birth.")
    n, m = len(a), len(b)
    cost = np.zeros((n + m, n + m))
    if n and m:
        cost[:n, :m] = np.max(np.abs(a[:, None] - b[None, :]), axis=2)
    if n:
        cost[:n, m:] = ((a[:, 1] - a[:, 0]) / 2)[:, None]
    if m:
        cost[n:, :m] = ((b[:, 1] - b[:, 0]) / 2)[None, :]
    return cost


def diagram_wasserstein_distance(diag_a, diag_b, order=1):
    if not np.isfinite(order) or order < 1:
        raise ValueError("Wasserstein order must be finite and >= 1.")
    cost = _diagram_cost(diag_a, diag_b) ** order
    if cost.size == 0:
        return 0.0
    rows, cols = linear_sum_assignment(cost)
    return float(cost[rows, cols].sum() ** (1 / order))


def bottleneck_distance(diag_a, diag_b):
    cost = _diagram_cost(diag_a, diag_b)
    if cost.size == 0:
        return 0.0
    candidates = np.unique(cost)
    lo, hi = 0, len(candidates) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        matching = maximum_bipartite_matching(csr_matrix(cost <= candidates[mid]), perm_type="column")
        if np.all(matching >= 0):
            hi = mid
        else:
            lo = mid + 1
    return float(candidates[lo])


def normalized_wasserstein_distance(z_clean, z_perturbed, order=1,
                                    distance_fn=default_cosine_distance_matrix, eps=1e-12,
                                    diag_clean=None):
    d = _distance(z_clean, distance_fn)
    a = _h0_from_distance(d) if diag_clean is None else diag_clean
    b = finite_h0_diagram(z_perturbed, distance_fn)
    return diagram_wasserstein_distance(a, b, order) / max(float(d.max()), eps)


def h0_bottleneck_distance(z_clean, z_perturbed, distance_fn=default_cosine_distance_matrix):
    return bottleneck_distance(finite_h0_diagram(z_clean, distance_fn),
                               finite_h0_diagram(z_perturbed, distance_fn))


def evaluate_representation_robustness(get_representation_fn, X_clean, aux_clean=None,
        perturbation="gene_dropout", perturbation_appliers=None, levels=DEFAULT_LEVELS,
        seeds=DEFAULT_SEEDS, distance_fn=default_cosine_distance_matrix, verbose=True):
    levels, seeds = _protocol(levels, seeds)
    if not perturbation_appliers or perturbation not in perturbation_appliers:
        raise ValueError(f"Missing applier for {perturbation}.")
    with _seeded(seeds[0]):
        clean = _embedding(get_representation_fn, X_clean, aux_clean)
    d_clean = _distance(clean, distance_fn)
    diag_clean = _h0_from_distance(d_clean)
    scale = max(float(d_clean.max()), 1e-12)
    tp_clean = float(np.diff(diag_clean, axis=1).sum() / scale)
    rows = []
    for level in levels:
        for seed in seeds:
            if level == 0:
                diag, tp = diag_clean, tp_clean
            else:
                with _seeded(seed):
                    x, aux = perturbation_appliers[perturbation](_clone(X_clean), _clone(aux_clean), level, seed)
                    z = _embedding(get_representation_fn, x, aux)
                if z.shape != clean.shape:
                    raise ValueError("Perturbations must preserve cell order and embedding shape.")
                d = _distance(z, distance_fn)
                diag = _h0_from_distance(d)
                tp = float(np.diff(diag, axis=1).sum() / max(float(d.max()), 1e-12))
            row = dict(perturbation=perturbation, level=level, seed=seed,
                       H0_bottleneck=bottleneck_distance(diag_clean, diag),
                       TotalPersistence_H0_clean=tp_clean, TotalPersistence_H0=tp,
                       DeltaPersistence_H0=abs(tp - tp_clean))
            row["1WD"] = diagram_wasserstein_distance(diag_clean, diag, 1) / scale
            row["2WD"] = diagram_wasserstein_distance(diag_clean, diag, 2) / scale
            rows.append(row)
            if verbose:
                print(f"[{perturbation}] level={level:g} seed={seed} 1WD={row['1WD']:.6g}")
    return pd.DataFrame(rows)


def run_robustness_comparison(models, dataset_name, perturbations=None, levels=DEFAULT_LEVELS,
        seeds=DEFAULT_SEEDS, distance_fn=default_cosine_distance_matrix, save_path=None,
        verbose=True, appliers=None):
    levels, seeds = _protocol(levels, seeds)
    perturbations = None if perturbations is None else tuple(perturbations)
    frames = []
    for label, spec in models.items():
        selected = _resolve_appliers(spec, appliers, perturbations)
        for name in selected:
            frame = evaluate_representation_robustness(
                spec["get_representation_fn"], spec["X_clean"], spec.get("aux_clean"),
                name, selected, levels, seeds, spec.get("distance_fn", distance_fn), verbose)
            frame["model"] = label
            frames.append(frame)
    if not frames:
        raise ValueError("Provide at least one model and perturbation.")
    results = pd.concat(frames, ignore_index=True)
    results["dataset"] = dataset_name
    columns = ["dataset", "model", "perturbation", "level", "seed", *CANONICAL_METRICS,
               "TotalPersistence_H0_clean", "TotalPersistence_H0", "DeltaPersistence_H0"]
    return _save(results[columns], save_path, verbose)
