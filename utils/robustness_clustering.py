"""Frozen-encoder robustness measured by KMeans ACC, ARI and NMI.

KMeans is refit for each perturbed embedding, with the same random_state as its
clean reference. Scores use ground-truth labels; ACC uses Hungarian matching.
No encoder training and no inference-time changes to model parameters occur here.
The shared make_default_appliers adapter calls utils.perturbation for all noise
generation; this module does not implement perturbations separately.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import normalize

from .robustness_topology import (
    DEFAULT_LEVELS, DEFAULT_SEEDS, make_default_appliers,
    _clone, _embedding, _protocol, _resolve_appliers, _save, _seeded,
)


def clustering_accuracy(y_true, y_pred):
    """Label-permutation invariant accuracy, including string/noncontiguous labels."""
    true, pred = np.asarray(y_true).reshape(-1), np.asarray(y_pred).reshape(-1)
    if not len(true) or len(true) != len(pred):
        raise ValueError("Labels must be nonempty and have matching lengths.")
    _, true = np.unique(true, return_inverse=True)
    _, pred = np.unique(pred, return_inverse=True)
    contingency = np.zeros((pred.max() + 1, true.max() + 1), dtype=np.int64)
    np.add.at(contingency, (pred, true), 1)
    rows, cols = linear_sum_assignment(-contingency)
    return float(contingency[rows, cols].sum() / len(true))


def _scores(z, labels, num_clusters, seed, normalize_embeddings, n_init):
    if normalize_embeddings:
        z = normalize(z, norm="l2")
    pred = KMeans(n_clusters=num_clusters, n_init=n_init, random_state=seed).fit_predict(z)
    return {"ACC": clustering_accuracy(labels, pred),
            "ARI": float(adjusted_rand_score(labels, pred)),
            "NMI": float(normalized_mutual_info_score(labels, pred))}


def evaluate_clustering_robustness(get_representation_fn, X_clean, Y_true, num_clusters,
        perturbation, applier_fn, aux_clean=None, levels=DEFAULT_LEVELS, seeds=DEFAULT_SEEDS,
        normalize_embeddings=True, n_init=20, verbose=True):
    levels, seeds = _protocol(levels, seeds)
    labels = np.asarray(Y_true).reshape(-1)
    with _seeded(seeds[0]):
        clean = _embedding(get_representation_fn, X_clean, aux_clean)
    if len(labels) != len(clean):
        raise ValueError("Y_true must follow the embedding's cell order and length.")
    if not isinstance(num_clusters, (int, np.integer)) or not 1 <= num_clusters <= len(clean):
        raise ValueError("num_clusters must be an integer between 1 and number of cells.")
    reference = {s: _scores(clean, labels, num_clusters, s, normalize_embeddings, n_init) for s in seeds}
    rows = []
    for level in levels:
        for seed in seeds:
            if level == 0:
                scores = reference[seed]
            else:
                with _seeded(seed):
                    x, aux = applier_fn(_clone(X_clean), _clone(aux_clean), level, seed)
                    z = _embedding(get_representation_fn, x, aux)
                if z.shape != clean.shape:
                    raise ValueError("Perturbations must preserve cell order and embedding shape.")
                scores = _scores(z, labels, num_clusters, seed, normalize_embeddings, n_init)
            rows.append(dict(perturbation=perturbation, level=level, seed=seed, **scores,
                             **{key + "_clean": value for key, value in reference[seed].items()}))
            if verbose:
                print(f"[{perturbation}] level={level:g} seed={seed} ARI={scores['ARI']:.6g}")
    return pd.DataFrame(rows)


def run_clustering_robustness_comparison(models, dataset_name, Y_true, num_clusters,
        appliers=None, levels=DEFAULT_LEVELS, seeds=DEFAULT_SEEDS, save_path=None,
        *, perturbations=None, normalize_embeddings=True, n_init=20, verbose=True):
    """Accept either global appliers or per-model perturbation_appliers, plus a filter."""
    levels, seeds = _protocol(levels, seeds)
    perturbations = None if perturbations is None else tuple(perturbations)
    frames = []
    for label, spec in models.items():
        selected = _resolve_appliers(spec, appliers, perturbations)
        for name, applier in selected.items():
            frame = evaluate_clustering_robustness(
                get_representation_fn=spec["get_representation_fn"], X_clean=spec["X_clean"],
                aux_clean=spec.get("aux_clean"), Y_true=Y_true, num_clusters=num_clusters,
                perturbation=name, applier_fn=applier, levels=levels, seeds=seeds,
                normalize_embeddings=normalize_embeddings, n_init=n_init, verbose=verbose)
            frame["model"] = label
            frames.append(frame)
    if not frames:
        raise ValueError("Provide at least one model and perturbation.")
    results = pd.concat(frames, ignore_index=True)
    results["dataset"] = dataset_name
    columns = ["dataset", "model", "perturbation", "level", "seed",
               "ACC", "ARI", "NMI", "ACC_clean", "ARI_clean", "NMI_clean"]
    return _save(results[columns], save_path, verbose)
