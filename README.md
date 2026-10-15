# RAB Scenario Classification — public edition

Generic image embedding, clustering and evaluation code. This edition contains
no company-specific extraction tools, service endpoints, internal paths, real
event identifiers, recorded images, embeddings, annotations or experiment outputs.
It has a fresh Git history. Examples use synthetic data only.

## Install

    python -m venv .venv
    # Activate the environment.
    python -m pip install -r requirements.txt
    python -m pip install -e .

Run commands from the repository root. The Bash pipeline targets Linux or a
suitable Bash environment; Python modules also run on Windows.

## Synthetic example

    python scripts/create_synthetic_example.py
    bash scripts/run_pipeline.sh configs/examples/synthetic_kmeans.sh

The generator creates fictional feature vectors, image names and labels.
No pretrained-model download is needed for this example. You can also invoke:

    python -m rab_scenario.clustering.kmeans --embedding_dir data/processed/example_dataset_dinov2_vits14 --output_dir results/runs/EXAMPLE_demo --model_name dinov2_vits14 --n_clusters 3 --no_pca --no_visualize
    python -m rab_scenario.evaluation.evaluate_experiments --runs EXAMPLE

For semantic evaluation, pass the generated annotations and the assignment
file written by your run:

    python -m rab_scenario.evaluation.semantic_evaluation --annotation_file data/annotations/current/manual_annotations.csv --assignment_file <assignment-file> --run_id EXAMPLE --output_dir results/semantic/EXAMPLE

## Layout

    src/rab_scenario/       Generic embedding, clustering and evaluation modules
    configs/examples/      Synthetic example configuration
    scripts/               Pipeline and synthetic-data generator
    docs/                  Portable usage documentation
    tests/                 Synthetic-data checks
    data/ and results/     Local generated inputs and outputs, ignored by Git

[Documentation](docs/README.md). Proprietary data acquisition is deliberately
excluded; provide your own images or embeddings. Exported code and documentation
are listed in PUBLIC_CONTENTS.json.
