"""Generic image embedding, clustering and evaluation utilities."""
import os
import argparse
import numpy as np
import time
from datetime import datetime
import yaml

import umap
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA

from rab_scenario.evaluation.clustering_metrics import evaluate_clustering

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def load_embeddings(embedding_dir, model_name=None):
    npy_files = [f for f in os.listdir(embedding_dir) if f.endswith("_embeddings.npy")]
    if not npy_files:
        raise FileNotFoundError(f"No *_embeddings.npy file found in: {embedding_dir}")

    if model_name is not None:
        embeddings_filename = f"{model_name}_embeddings.npy"
        if embeddings_filename not in npy_files:
            available = [f.replace("_embeddings.npy", "") for f in npy_files]
            raise FileNotFoundError(
                f"Model '{model_name}' not found in {embedding_dir}.\n"
                f"Available models: {available}"
            )
    elif len(npy_files) == 1:
        embeddings_filename = npy_files[0]
    else:
        available = [f.replace("_embeddings.npy", "") for f in npy_files]
        raise ValueError(
            f"Multiple embedding files found in {embedding_dir}.\n"
            f"Please specify --model_name. Available: {available}"
        )

    model_name = embeddings_filename.replace("_embeddings.npy", "")

    embeddings_path = os.path.join(embedding_dir, embeddings_filename)
    names_path = os.path.join(embedding_dir, f"{model_name}_image_names.npy")

    if not os.path.exists(embeddings_path):
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")
    if not os.path.exists(names_path):
        raise FileNotFoundError(f"Image names file not found: {names_path}")

    embeddings = np.load(embeddings_path)
    image_names = np.load(names_path, allow_pickle=True)

    print(f"Detected model name    : {model_name}")
    print(f"Loaded embeddings shape: {embeddings.shape}")
    print(f"Loaded {len(image_names)} image names")

    return embeddings, image_names, model_name


def reduce_dimensions(embeddings, n_components=128, random_state=42):
    print(f"Reducing dimensions: {embeddings.shape[1]} -> {n_components} via PCA ...")

    pca = PCA(n_components=n_components, random_state=random_state)
    reduced = pca.fit_transform(embeddings)

    explained = np.sum(pca.explained_variance_ratio_) * 100
    print(f"PCA explained variance: {explained:.2f}%")

    return reduced, pca


def run_kmeans(embeddings_reduced, n_clusters=8, random_state=42, n_init=10):
    print(
        f"\nRunning K-Means with n_clusters={n_clusters}, "
        f"random_state={random_state}, n_init={n_init} ..."
    )

    clusterer = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=n_init
    )

    clusterer.fit(embeddings_reduced)
    labels = clusterer.labels_

    return labels, clusterer


def print_cluster_summary(labels, image_names):
    unique_labels = sorted(set(labels))
    n_clusters = len(unique_labels)

    print("\n========== K-Means Clustering Summary ==========")
    print(f"Total images      : {len(labels)}")
    print(f"Number of clusters: {n_clusters}")
    print("================================================")

    print(f"\n{'Cluster':>10} | {'Count':>8} | {'% of Total':>10}")
    print("-" * 35)

    for label in unique_labels:
        count = np.sum(labels == label)
        print(f"{label:>10} | {count:>8} | {100 * count / len(labels):>9.1f}%")

    print("================================================\n")


def save_results(labels, image_names, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    np.save(os.path.join(output_dir, "cluster_labels.npy"), labels)

    assignment_path = os.path.join(output_dir, "cluster_assignments.txt")
    with open(assignment_path, "w") as f:
        f.write(f"{'image_name':<60} cluster_label\n")
        f.write("-" * 75 + "\n")
        for name, label in zip(image_names, labels):
            f.write(f"{str(name):<60} {label}\n")

    print(f"Saved cluster_labels.npy      -> {output_dir}")
    print(f"Saved cluster_assignments.txt -> {output_dir}")


def plot_clusters_pca2d(embeddings_reduced, labels, output_dir):
    print("Generating 2D PCA cluster visualization ...")

    pca2d = PCA(n_components=2, random_state=42)
    coords = pca2d.fit_transform(embeddings_reduced)

    unique_labels = sorted(set(labels))
    n_clusters = len(unique_labels)

    fig, ax = plt.subplots(figsize=(12, 9))
    cmap = cm.get_cmap("tab20", max(n_clusters, 1))

    for label in unique_labels:
        mask = labels == label
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=15,
            color=cmap(label % 20),
            alpha=0.6,
            label=f"Cluster {label} (n={np.sum(mask)})"
        )

    ax.set_title("K-Means Clustering (2D PCA Projection)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(loc="upper right", fontsize=6, ncol=2)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "cluster_visualization_pca2d.png")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

    print(f"Saved visualization -> {out_path}")


def plot_clusters_umap2d(embeddings_reduced, labels, output_dir, random_state=42):
    print("Generating 2D UMAP cluster visualization ...")

    reducer = umap.UMAP(
        n_components=2,
        metric="euclidean",
        random_state=random_state
    )
    coords = reducer.fit_transform(embeddings_reduced)

    unique_labels = sorted(set(labels))
    n_clusters = len(unique_labels)

    fig, ax = plt.subplots(figsize=(12, 9))
    cmap = cm.get_cmap("tab20", max(n_clusters, 1))

    for label in unique_labels:
        mask = labels == label
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=15,
            color=cmap(label % 20),
            alpha=0.6,
            label=f"Cluster {label} (n={np.sum(mask)})"
        )

    ax.set_title("K-Means Clustering (2D UMAP Projection)")
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.legend(loc="upper right", fontsize=6, ncol=2)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "cluster_visualization_umap2d.png")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

    print(f"Saved visualization -> {out_path}")


def save_config(final_output_dir, embedding_dir, model_name, n_images,
                use_pca, pca_components, n_clusters, random_state, n_init,
                visualize):

    config = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "embedding_dir": embedding_dir,
        "model_name": model_name,
        "num_samples": n_images,
        "clustering": {
            "method": "kmeans",
            "use_pca": use_pca,
            "pca_components": pca_components if use_pca else None,
            "n_clusters": n_clusters,
            "random_state": random_state,
            "n_init": n_init
        },
        "visualization": {
            "enabled": visualize,
            "methods": ["pca2d", "umap2d"] if visualize else [],
            "umap": {
                "n_components": 2,
                "metric": "euclidean",
                "random_state": 42
            } if visualize else None
        }
    }

    config_path = os.path.join(final_output_dir, "config.yaml")

    with open(config_path, "w") as f:
        yaml.dump(config, f)

    print(f"Saved config -> {config_path}")


def run_clustering(embedding_dir, output_dir,
                   model_name=None,
                   use_pca=True,
                   pca_components=128,
                   n_clusters=8,
                   random_state=42,
                   n_init=10,
                   visualize=True):

    start_time = time.time()

    embeddings, image_names, model_name = load_embeddings(embedding_dir, model_name)
    n_images = len(image_names)

    if n_clusters > n_images:
        raise ValueError(
            f"n_clusters ({n_clusters}) cannot be larger than number of images ({n_images})."
        )

    print("L2 normalizing embeddings ...")
    embeddings_norm = normalize(embeddings, norm="l2")

    if use_pca:
        embeddings_reduced, _ = reduce_dimensions(
            embeddings_norm,
            n_components=pca_components,
            random_state=random_state
        )
        pca_tag = f"pca{pca_components}"
    else:
        embeddings_reduced = embeddings_norm
        pca_tag = "nopca"

    # sub_folder = f"{model_name}_n{n_images}_{pca_tag}_kmeans_k{n_clusters}"
    # final_output_dir = os.path.join(output_dir, sub_folder)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    sub_folder = (
        f"{model_name}_n{n_images}_{pca_tag}"
        f"_kmeans_k{n_clusters}_{timestamp}"
    )

    final_output_dir = os.path.join(output_dir, sub_folder)

    print(f"\nResults will be saved to: {final_output_dir}")

    labels, _ = run_kmeans(
        embeddings_reduced,
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=n_init
    )

    print_cluster_summary(labels, image_names)

    save_results(labels, image_names, final_output_dir)

    evaluate_clustering(
        embeddings_reduced,
        labels,
        final_output_dir,
        exclude_noise=False,
        silhouette_metric="euclidean",
        random_state=random_state,
    )

    save_config(
        final_output_dir,
        embedding_dir,
        model_name,
        n_images,
        use_pca,
        pca_components,
        n_clusters,
        random_state,
        n_init,
        visualize
    )

    if visualize:
        plot_clusters_pca2d(embeddings_reduced, labels, final_output_dir)
        plot_clusters_umap2d(embeddings_reduced, labels, final_output_dir)

    runtime = time.time() - start_time
    print(f"Total runtime: {runtime:.2f}s")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="K-Means Clustering on Precomputed Embeddings")

    parser.add_argument("--embedding_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Specify which embedding to use, e.g. resnet50 or dinov2_vits14"
    )

    parser.add_argument("--no_pca", action="store_true")
    parser.add_argument("--pca_components", type=int, default=128)

    parser.add_argument(
        "--n_clusters",
        type=int,
        default=8,
        help="Number of clusters for K-Means"
    )

    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )

    parser.add_argument(
        "--n_init",
        type=int,
        default=10,
        help="Number of K-Means initializations"
    )

    parser.add_argument("--no_visualize", action="store_true")

    args = parser.parse_args()

    run_clustering(
        embedding_dir=args.embedding_dir,
        output_dir=args.output_dir,
        model_name=args.model_name,
        use_pca=not args.no_pca,
        pca_components=args.pca_components,
        n_clusters=args.n_clusters,
        random_state=args.random_state,
        n_init=args.n_init,
        visualize=not args.no_visualize
    )
