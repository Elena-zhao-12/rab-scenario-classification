# Getting started

Install requirements.txt and the local package as described in the root README.
Use Python 3.10 or newer with wheels supported by those dependency versions.

    python scripts/create_synthetic_example.py
    bash scripts/run_pipeline.sh configs/examples/synthetic_kmeans.sh
    python -m unittest discover -s tests -p "test_*.py"

The example uses existing synthetic embeddings. To extract your own features,
place a flat image directory under data/datasets/<dataset>/raw_frames/ and
set START_FROM_EMBEDDINGS=false in a configuration. Model weights may download
if they are not cached.

Evaluation defaults point to example data, not a real dataset. Pass input and
output paths explicitly for other data.
