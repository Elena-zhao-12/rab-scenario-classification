"""Generic image embedding, clustering and evaluation utilities."""
import argparse
import csv
import os
import re
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

from rab_scenario.evaluation.clustering_metrics import evaluate_clustering


DEFAULT_RUN_IDS = ["EXAMPLE"]


def read_config(config_path):
    """Read the scalar fields needed from this project's generated YAML files."""
    text = config_path.read_text(encoding="utf-8")

    def value(pattern, required=True):
        match = re.search(pattern, text, flags=re.MULTILINE)
        if match:
            return match.group(1).strip()
        if required:
            raise ValueError(f"Missing config field matching {pattern!r}: {config_path}")
        return None

    pca_components = value(r"^\s{2}pca_components:\s*(.+)$")
    metric = value(r"^\s{2}metric:\s*(.+)$", required=False)
    random_state = value(r"^\s{2}random_state:\s*(.+)$", required=False)

    return {
        "model_name": value(r"^model_name:\s*(.+)$"),
        "num_samples": int(value(r"^num_samples:\s*(.+)$")),
        "method": value(r"^\s{2}method:\s*(.+)$"),
        "use_pca": value(r"^\s{2}use_pca:\s*(.+)$").lower() == "true",
        "pca_components": None if pca_components == "null" else int(pca_components),
        "metric": metric or "euclidean",
        "random_state": int(random_state) if random_state is not None else 42,
    }


def find_experiment_dir(results_dir, run_id):
    matches = sorted(results_dir.glob(f"{run_id}_*/**/cluster_labels.npy"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one cluster_labels.npy for {run_id}, found {len(matches)}"
        )
    return matches[0].parent


def find_embedding_files(embedding_root, model_name):
    embedding_matches = sorted(embedding_root.rglob(f"{model_name}_embeddings.npy"))
    names_matches = sorted(embedding_root.rglob(f"{model_name}_image_names.npy"))
    if len(embedding_matches) != 1 or len(names_matches) != 1:
        raise FileNotFoundError(
            f"Expected one embedding/name pair for {model_name}; found "
            f"{len(embedding_matches)} embeddings and {len(names_matches)} name files"
        )
    return embedding_matches[0], names_matches[0]


def load_assignments(assignment_path):
    rows = []
    for line in assignment_path.read_text(encoding="utf-8").splitlines()[2:]:
        if not line.strip():
            continue
        image_name, label = line.rsplit(maxsplit=1)
        rows.append((image_name, int(label)))
    return rows


def validate_alignment(experiment_dir, embeddings, image_names, labels, expected_samples):
    assignments = load_assignments(experiment_dir / "cluster_assignments.txt")
    assignment_names = np.asarray([row[0] for row in assignments], dtype=str)
    assignment_labels = np.asarray([row[1] for row in assignments])
    image_names = np.asarray(image_names, dtype=str)

    lengths = {
        "config": expected_samples,
        "embeddings": len(embeddings),
        "image_names": len(image_names),
        "labels": len(labels),
        "assignments": len(assignments),
    }
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Sample-count mismatch in {experiment_dir}: {lengths}")
    if not np.array_equal(assignment_names, image_names):
        raise ValueError(f"Image-name order does not match embeddings: {experiment_dir}")
    if not np.array_equal(assignment_labels, labels):
        raise ValueError(f"Assignment labels do not match cluster_labels.npy: {experiment_dir}")


def write_summary(summary_rows, output_path):
    fieldnames = [
        "run_id",
        "method",
        "model_name",
        "use_pca",
        "pca_components",
        "total_samples",
        "evaluated_samples",
        "n_clusters",
        "noise_count",
        "noise_ratio",
        "silhouette_score",
        "davies_bouldin_index",
        "status",
        "experiment_dir",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)


def evaluate_experiments(results_dir, embedding_root, run_ids, sample_size=10000, output_dir=None, summary_path=None):
    raw_cache = {}
    feature_cache = {}
    summary_rows = []

    for run_id in run_ids:
        experiment_dir = find_experiment_dir(results_dir, run_id)
        config = read_config(experiment_dir / "config.yaml")
        model_name = config["model_name"]

        if model_name not in raw_cache:
            embeddings_path, names_path = find_embedding_files(embedding_root, model_name)
            raw_cache[model_name] = (
                np.load(embeddings_path),
                np.load(names_path, allow_pickle=True),
            )

        embeddings, image_names = raw_cache[model_name]
        labels = np.load(experiment_dir / "cluster_labels.npy")
        validate_alignment(
            experiment_dir,
            embeddings,
            image_names,
            labels,
            config["num_samples"],
        )

        feature_key = (
            model_name,
            config["use_pca"],
            config["pca_components"],
            config["random_state"],
        )
        if feature_key not in feature_cache:
            print(f"\nPreparing feature space: {feature_key}")
            features = normalize(embeddings, norm="l2")
            if config["use_pca"]:
                features = PCA(
                    n_components=config["pca_components"],
                    random_state=config["random_state"],
                ).fit_transform(features)
            feature_cache[feature_key] = features

        print(f"\nEvaluating {run_id}: {experiment_dir}")
        metrics = evaluate_clustering(
            feature_cache[feature_key],
            labels,
            str((output_dir / run_id) if output_dir else experiment_dir),
            exclude_noise=config["method"] == "hdbscan",
            silhouette_metric=config["metric"],
            silhouette_sample_size=sample_size,
            random_state=config["random_state"],
        )
        summary_rows.append(
            {
                "run_id": run_id,
                "method": config["method"],
                "model_name": model_name,
                "use_pca": config["use_pca"],
                "pca_components": config["pca_components"],
                "total_samples": metrics["total_samples"],
                "evaluated_samples": metrics["evaluated_samples"],
                "n_clusters": metrics["n_clusters"],
                "noise_count": metrics["noise_count"],
                "noise_ratio": metrics["noise_ratio"],
                "silhouette_score": metrics["silhouette_score"],
                "davies_bouldin_index": metrics["davies_bouldin_index"],
                "status": metrics["status"],
                "experiment_dir": os.fspath(experiment_dir),
            }
        )

    summary_path = summary_path or Path("results/summaries/clustering_metrics_summary.csv")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    write_summary(summary_rows, summary_path)
    print(f"\nSaved summary: {summary_path}")
    return summary_rows


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate existing H/K/R clustering experiments without reclustering"
    )
    parser.add_argument("--results_dir", type=Path, default=Path("results/runs"))
    parser.add_argument("--embedding_root", type=Path, default=Path("data/processed"))
    parser.add_argument("--runs", nargs="+", default=DEFAULT_RUN_IDS)
    parser.add_argument(
        "--silhouette_sample_size",
        type=int,
        default=10000,
        help="Maximum samples for Silhouette; use 0 to evaluate all samples",
    )
    parser.add_argument("--output_dir", type=Path, help="Write recalculated metrics here instead of overwriting delivered results")
    parser.add_argument("--summary_path", type=Path, default=Path("results/summaries/clustering_metrics_summary.csv"))
    args = parser.parse_args()
    sample_size = None if args.silhouette_sample_size == 0 else args.silhouette_sample_size
    evaluate_experiments(args.results_dir, args.embedding_root, args.runs, sample_size, args.output_dir, args.summary_path)


if __name__ == "__main__":
    main()
