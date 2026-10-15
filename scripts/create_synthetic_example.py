"""Generate fictional inputs; never reads an existing dataset."""
from pathlib import Path
import csv
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
def generate(root=ROOT):
    root = Path(root)
    features = root / "data/processed/example_dataset_dinov2_vits14"
    features.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    centers = np.eye(16)[:3] * 4
    matrix = np.vstack([rng.normal(center, .1, (40, 16)) for center in centers]).astype("float32")
    names = np.array([f"example_{i:04d}.png" for i in range(len(matrix))])
    np.save(features / "dinov2_vits14_embeddings.npy", matrix)
    np.save(features / "dinov2_vits14_image_names.npy", names)
    annotations = root / "data/annotations/current/manual_annotations.csv"
    annotations.parent.mkdir(parents=True, exist_ok=True)
    fields = ["image_name", "trigger_validity", "scenario_cause", "object_type", "visibility_quality"]
    with annotations.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for i, name in enumerate(names):
            writer.writerow(dict(image_name=name, trigger_validity="true_positive" if i < 80 else "false_positive", scenario_cause="example_cause", object_type="example_object", visibility_quality="clear"))
    return features, annotations

if __name__ == "__main__":
    features, annotations = generate()
    print(f"Created 120 synthetic features: {features}")
    print(f"Created fictional labels: {annotations}")
