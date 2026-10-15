"""Generic image embedding, clustering and evaluation utilities."""
import json
import os

import numpy as np
from sklearn.metrics import davies_bouldin_score, silhouette_score


def evaluate_clustering(
    embeddings,
    labels,
    output_dir,
    exclude_noise=False,
    silhouette_metric="euclidean",
    silhouette_sample_size=10000,
    random_state=42,
):
    """Compute and save internal clustering metrics.

    HDBSCAN noise (label -1) can be excluded because it is not a real cluster.
    Silhouette is sampled for large datasets to avoid quadratic memory/time use;
    Davies-Bouldin is calculated on all evaluated samples.
    """
    embeddings = np.asarray(embeddings)
    labels = np.asarray(labels)

    if embeddings.ndim != 2:
        raise ValueError("embeddings must be a 2D array")
    if labels.ndim != 1:
        raise ValueError("labels must be a 1D array")
    if len(embeddings) != len(labels):
        raise ValueError("embeddings and labels must contain the same number of samples")
    if silhouette_sample_size is not None and silhouette_sample_size <= 0:
        raise ValueError("silhouette_sample_size must be positive or None")

    total_samples = len(labels)
    noise_mask = labels == -1
    noise_count = int(np.sum(noise_mask))
    evaluation_mask = ~noise_mask if exclude_noise else np.ones(total_samples, dtype=bool)
    evaluation_embeddings = embeddings[evaluation_mask]
    evaluation_labels = labels[evaluation_mask]

    unique_labels = np.unique(evaluation_labels)
    n_clusters = len(unique_labels)
    evaluated_samples = len(evaluation_labels)

    metrics = {
        "status": "ok",
        "total_samples": int(total_samples),
        "evaluated_samples": int(evaluated_samples),
        "excluded_noise": bool(exclude_noise),
        "noise_count": noise_count,
        "noise_ratio": float(noise_count / total_samples) if total_samples else None,
        "n_clusters": int(n_clusters),
        "silhouette_metric": silhouette_metric,
        "silhouette_sample_size": None,
        "silhouette_score": None,
        "davies_bouldin_index": None,
    }

    if n_clusters < 2 or n_clusters >= evaluated_samples:
        metrics["status"] = "undefined"
        metrics["reason"] = (
            "Metrics require at least 2 clusters and fewer clusters than evaluated samples."
        )
    else:
        if silhouette_sample_size is None:
            effective_sample_size = None
        else:
            effective_sample_size = min(silhouette_sample_size, evaluated_samples)

        metrics["silhouette_sample_size"] = effective_sample_size
        metrics["silhouette_score"] = float(
            silhouette_score(
                evaluation_embeddings,
                evaluation_labels,
                metric=silhouette_metric,
                sample_size=effective_sample_size,
                random_state=random_state,
            )
        )
        metrics["davies_bouldin_index"] = float(
            davies_bouldin_score(evaluation_embeddings, evaluation_labels)
        )

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "clustering_metrics.json")
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    print("\n========== Clustering Metrics ==========")
    print(f"Evaluated samples    : {evaluated_samples}")
    print(f"Number of clusters   : {n_clusters}")
    noise_ratio = metrics["noise_ratio"]
    print(f"Noise ratio          : {noise_ratio:.4f}" if noise_ratio is not None else "Noise ratio          : N/A")
    print(f"Silhouette score     : {metrics['silhouette_score']}")
    print(f"Davies-Bouldin index : {metrics['davies_bouldin_index']}")
    if metrics["status"] != "ok":
        print(f"Status               : {metrics['status']} ({metrics['reason']})")
    print(f"Saved metrics        : {output_path}")
    print("========================================")

    return metrics
