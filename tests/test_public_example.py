"""Public-edition tests use synthetic inputs only."""
from pathlib import Path
import csv
import importlib.util
import tempfile
import unittest
from rab_scenario.evaluation.semantic_evaluation import evaluate

ROOT = Path(__file__).resolve().parents[1]
class PublicExampleTests(unittest.TestCase):
    def test_synthetic_inputs_and_semantic_evaluation(self):
        spec = importlib.util.spec_from_file_location("synthetic_example", ROOT / "scripts/create_synthetic_example.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        parent = ROOT / "local/tests"
        parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as temporary:
            root = Path(temporary)
            features, annotations = module.generate(root)
            import numpy as np
            self.assertEqual(np.load(features / "dinov2_vits14_embeddings.npy").shape, (120, 16))
            assignments = root / "cluster_assignments.txt"
            assignments.write_text("image_name cluster_label\n-------------------------\n" + "".join(f"example_{i:04d}.png {i // 40}\n" for i in range(120)), encoding="utf-8")
            output = root / "semantic"
            evaluate(annotations, assignments, output, "EXAMPLE")
            with (output / "purity_summary.csv").open(encoding="utf-8") as file:
                rows = list(csv.DictReader(file))
            self.assertTrue(rows)
            self.assertTrue(all(float(row["weighted_purity"]) == 1.0 for row in rows))

if __name__ == "__main__":
    unittest.main()
