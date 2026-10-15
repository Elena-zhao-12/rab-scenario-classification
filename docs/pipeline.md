# Pipeline

    User-supplied images -> frozen embeddings -> clusters and internal metrics
    User-supplied labels + assignments -> semantic evaluation

## Modules

| Purpose | Module |
| --- | --- |
| DINOv2 features | rab_scenario.embeddings.dinov2 |
| ResNet50 features | rab_scenario.embeddings.resnet50 |
| HDBSCAN | rab_scenario.clustering.hdbscan |
| K-Means | rab_scenario.clustering.kmeans |
| Existing-run evaluation | rab_scenario.evaluation.evaluate_experiments |
| Semantic evaluation | rab_scenario.evaluation.semantic_evaluation |
| Image organization | rab_scenario.tools.organize_clusters |

Invoke executable modules with python -m. The Bash pipeline automatically
determines its checkout root. RAB_PROJECT_ROOT can override it and PYTHON_BIN
selects the active environment's interpreter.

START_FROM_EMBEDDINGS=true reuses feature/name arrays. ORGANIZE_CLUSTERS=true
requires the corresponding images and creates an ignored local inspection view.
Algorithm parameters remain in configs/. New outputs use
results/runs/<RUN_ID>_<dataset>_<timestamp>/ and retain run_config.sh.

Embeddings, image names, labels and assignments must be row aligned. Internal
evaluation checks that alignment. HDBSCAN excludes noise from Silhouette and
Davies-Bouldin, so interpret those scores alongside coverage. Default experiment
discovery requires one directory per identifier; repeated identifiers must be
evaluated using an isolated results root.

Raw-log acquisition, private service connectivity and deployment infrastructure
are outside this edition.
