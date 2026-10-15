# RAB Trigger Scenario Classification

Unsupervised organization of Rear Auto Brake (RAB) trigger events from
rear-camera key frames. A master's thesis project on organizing trigger events for review.

**Scope.** The pipeline groups visually similar trigger events so that they can
be reviewed in batches instead of one by one. The resulting groups are
candidates for inspection, not verified scenario categories, and nothing here
decides whether an individual braking intervention was justified. Each event is
represented by a single camera frame, so temporal development and the other
sensing modalities are outside what the method can see.

## What the project does

RAB brakes the vehicle when an obstacle is detected behind it during low-speed
reversing. Reviewing trigger events involves inspecting camera recordings and
vehicle logs, and this manual work becomes harder as the number of events grows.
When events are predominantly unlabelled and recurring categories are not known
in advance, unsupervised grouping can help organize that review.

This project therefore treats the review problem as data structuring: extract
one key frame per trigger event, embed the frames with frozen pretrained
backbones, cluster the embeddings, and evaluate whether the resulting groups
are geometrically coherent and interpretable.

```text
trigger events -> key frames -> frozen embeddings -> clusters -> inspection
```

The public implementation starts with user-supplied key frames or embeddings.
Company-specific event retrieval and key-frame extraction are excluded.

## Results at a glance

The original dataset, numerical experiment results and manual project
annotations are not included in this public edition. The repository provides
the methods and a synthetic example, rather than reproducing the original
experimental findings.

| Component | Available methods | Evaluation |
| --- | --- | --- |
| Frozen image embeddings | DINOv2 and ResNet50 | Compare representations using the same input images |
| Clustering | HDBSCAN and K-Means, with optional PCA | Silhouette, Davies-Bouldin, cluster count and noise ratio |
| Semantic evaluation | Cluster assignments joined to supplied labels | Per-cluster and weighted purity, plus noise composition |

HDBSCAN can leave samples unassigned as noise, while K-Means assigns every
sample. HDBSCAN's internal scores exclude noise and must be interpreted together
with coverage and evaluated sample count. Semantic evaluation uses demonstration
labels by default; supply your own vocabulary for a real annotation scheme.
Visual coherence alone does not establish functional agreement.

## Repository layout

    scripts/                 Pipeline entry point and synthetic-data generator
    configs/examples/        Synthetic K-Means and HDBSCAN configurations
    src/rab_scenario/
      embeddings/            Frozen DINOv2 and ResNet50 feature extraction
      clustering/            HDBSCAN and K-Means
      evaluation/            Internal metrics and semantic evaluation
      tools/                 Image organization for inspection
    data/                    User-supplied or synthetic inputs, ignored by Git
    results/                 Generated results, ignored by Git
    tests/                   Synthetic-data integration test
    docs/                    Portable usage documentation
    local/                   Local inspection views and temporary files

Only the explanatory README files under data/ and results/ are tracked.
The public edition contains no recorded images, real event identifiers,
production embeddings, original annotations or historical experiment outputs.

## Getting started

Create and activate a Python environment, then install the dependencies and
the local package:

    python -m venv .venv
    # Linux: source .venv/bin/activate
    # Windows PowerShell: .venv\Scripts\Activate.ps1
    python -m pip install -r requirements.txt
    python -m pip install -e .

Run commands from the repository root. The Bash pipeline targets Linux or a
suitable Bash environment; Python modules also run on Windows.

Generate fictional embeddings and annotations, then run the K-Means example:

    python scripts/create_synthetic_example.py
    bash scripts/run_pipeline.sh configs/examples/synthetic_kmeans.sh

The example does not require model-weight downloads. To try HDBSCAN on the
same synthetic inputs:

    bash scripts/run_pipeline.sh configs/examples/synthetic_hdbscan.sh

Alternatively, run K-Means directly:

    python -m rab_scenario.clustering.kmeans --embedding_dir data/processed/example_dataset_dinov2_vits14 --output_dir results/runs/EXAMPLE_demo --model_name dinov2_vits14 --n_clusters 3 --no_pca --no_visualize

Recalculate its internal metrics:

    python -m rab_scenario.evaluation.evaluate_experiments --runs EXAMPLE

Default discovery requires one run directory for each identifier. Choose
either the pipeline or direct K-Means example in a fresh checkout; running both
creates two EXAMPLE directories. Use an isolated --results_dir when comparing
repeated runs.

For semantic evaluation, substitute the assignment file from the chosen run:

    python -m rab_scenario.evaluation.semantic_evaluation --annotation_file data/annotations/current/manual_annotations.csv --assignment_file <assignment-file> --run_id EXAMPLE --output_dir results/semantic/EXAMPLE

For your own label scheme, add --vocabulary_file <vocabulary.json>. The JSON
can define trigger_validity_map, scenario_causes and object_families.

New runs go to results/runs/ and retain the input configuration snapshot.
To work with real images, provide a flat image directory under
data/datasets/<dataset>/raw_frames/ and set START_FROM_EMBEDDINGS=false in
a copied configuration. Pretrained weights may download if they are not cached.
These inputs and generated outputs remain local.

## Documentation

- [Navigation](docs/README.md)
- [Installation and examples](docs/getting-started.md)
- [Pipeline and configuration](docs/pipeline.md)
- [Sharing policy](docs/sharing.md)
- [Public-content review](docs/public-audit.md)

Company-specific acquisition tools, infrastructure details, datasets and
historical results are excluded from this edition. The repository was created
without the original Git history. Exported file hashes are listed in
PUBLIC_CONTENTS.json.

Each event uses one frame, so temporal context and other sensing modalities
remain outside the representation. Clusters support inspection and do not
determine whether a braking intervention was justified.
