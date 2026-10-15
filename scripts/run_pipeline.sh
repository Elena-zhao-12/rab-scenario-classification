#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -ne 1 ]; then
    echo "Usage: bash scripts/run_pipeline.sh <config_file>" >&2
    exit 1
fi
CONFIG_FILE="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
PROJECT_ROOT="${RAB_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
[ -f "$CONFIG_FILE" ] || { echo "Config not found: $CONFIG_FILE" >&2; exit 1; }
source "$CONFIG_FILE"
cd "$PROJECT_ROOT"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_ID="${RUN_ID:-CUSTOM}"
[[ "$RUN_ID" =~ ^[A-Za-z0-9_-]+$ ]] || { echo "Invalid RUN_ID" >&2; exit 1; }
for flag in START_FROM_EMBEDDINGS ORGANIZE_CLUSTERS USE_PCA VISUALIZE; do
    value="${!flag}"
    [[ "$value" == true || "$value" == false ]] || { echo "$flag must be true or false" >&2; exit 1; }
done
case "$EMBED_MODEL" in
    dinov2) MODEL_NAME="dinov2_${DINO_MODEL_VARIANT}"; EMBED_MODULE="rab_scenario.embeddings.dinov2" ;;
    resnet50) MODEL_NAME="resnet50"; EMBED_MODULE="rab_scenario.embeddings.resnet50" ;;
    *) echo "Unknown EMBED_MODEL: $EMBED_MODEL" >&2; exit 1 ;;
esac
case "$CLUSTER_METHOD" in
    hdbscan|kmeans) ;;
    *) echo "Unknown CLUSTER_METHOD: $CLUSTER_METHOD" >&2; exit 1 ;;
esac
RAW_FRAMES_DIR="$PROJECT_ROOT/data/datasets/$DATASET_NAME/raw_frames"
EMBEDDING_DIR="$PROJECT_ROOT/data/processed/${DATASET_NAME}_${MODEL_NAME}"
if [ "$START_FROM_EMBEDDINGS" == true ]; then
    for suffix in embeddings image_names; do
        [ -f "$EMBEDDING_DIR/${MODEL_NAME}_${suffix}.npy" ] || {
            echo "Missing frozen input: $MODEL_NAME $suffix" >&2; exit 1;
        }
    done
else
    [ -d "$RAW_FRAMES_DIR" ] || { echo "Missing raw frames: $RAW_FRAMES_DIR" >&2; exit 1; }
fi
if [ "$ORGANIZE_CLUSTERS" == true ] && [ ! -d "$RAW_FRAMES_DIR" ]; then
    echo "ORGANIZE_CLUSTERS requires raw frames" >&2; exit 1
fi
"$PYTHON_BIN" -c 'import rab_scenario' || {
    echo "Activate the project environment and run: python -m pip install -e ." >&2; exit 1;
}
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_NAME="${RUN_ID}_${DATASET_NAME}_${TIMESTAMP}"
CLUSTER_DIR="$PROJECT_ROOT/results/runs/$RUN_NAME"
mkdir -p "$(dirname "$CLUSTER_DIR")"
mkdir "$CLUSTER_DIR"
cp "$CONFIG_FILE" "$CLUSTER_DIR/run_config.sh"
printf 'Run: %s\nOutput: %s\n' "$RUN_NAME" "$CLUSTER_DIR"
if [ "$START_FROM_EMBEDDINGS" == false ]; then
    EMBED_ARGS=(--input_dir "$RAW_FRAMES_DIR" --output_dir "$EMBEDDING_DIR" --batch_size "$BATCH_SIZE")
    [ "$EMBED_MODEL" != dinov2 ] || EMBED_ARGS+=(--model_variant "$DINO_MODEL_VARIANT")
    "$PYTHON_BIN" -m "$EMBED_MODULE" "${EMBED_ARGS[@]}"
fi
CLUSTER_ARGS=(--embedding_dir "$EMBEDDING_DIR" --output_dir "$CLUSTER_DIR"
    --model_name "$MODEL_NAME" --pca_components "$PCA_COMPONENTS")
[ "$USE_PCA" != false ] || CLUSTER_ARGS+=(--no_pca)
[ "$VISUALIZE" != false ] || CLUSTER_ARGS+=(--no_visualize)
if [ "$CLUSTER_METHOD" == hdbscan ]; then
    CLUSTER_ARGS+=(--min_cluster_size "$MIN_CLUSTER_SIZE" --min_samples "$MIN_SAMPLES" --metric "$METRIC")
else
    CLUSTER_ARGS+=(--n_clusters "$N_CLUSTERS" --random_state "$KMEANS_RANDOM_STATE" --n_init "$KMEANS_N_INIT")
fi
"$PYTHON_BIN" -m "rab_scenario.clustering.$CLUSTER_METHOD" "${CLUSTER_ARGS[@]}"
if [ "$ORGANIZE_CLUSTERS" == true ]; then
    mapfile -t ASSIGNMENTS < <(find "$CLUSTER_DIR" -name cluster_assignments.txt -type f)
    [ "${#ASSIGNMENTS[@]}" -eq 1 ] || { echo "Expected one assignment file" >&2; exit 1; }
    "$PYTHON_BIN" -m rab_scenario.tools.organize_clusters \
        --assignment_file "${ASSIGNMENTS[0]}" --image_dir "$RAW_FRAMES_DIR" \
        --output_dir "$PROJECT_ROOT/local/inspection/$RUN_NAME"
fi
echo "Completed: $RUN_NAME"
echo "Register this run in results/run_registry.csv before including it in a summary."
