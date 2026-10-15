"""Generic image embedding, clustering and evaluation utilities."""
import os
import argparse
import time
from typing import List

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from transformers import AutoImageProcessor, AutoModel


DINO_MODEL_MAP = {
    "vits14": "facebook/dinov2-small",
    "vitb14": "facebook/dinov2-base",
    "vitl14": "facebook/dinov2-large",
    "vitg14": "facebook/dinov2-giant",
}


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_model_and_processor(model_variant: str, device: torch.device):
    """
    Load DINOv2 backbone and corresponding image processor.

    Args:
        model_variant: one of {"vits14", "vitb14", "vitl14", "vitg14"}
        device: torch device

    Returns:
        processor, model, output_name
    """
    if model_variant not in DINO_MODEL_MAP:
        raise ValueError(
            f"Unsupported model_variant: {model_variant}. "
            f"Choose from: {list(DINO_MODEL_MAP.keys())}"
        )

    model_name = DINO_MODEL_MAP[model_variant]

    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    model.to(device)
    model.eval()

    output_name = f"dinov2_{model_variant}"
    return processor, model, output_name


def collect_image_paths(input_dir: str) -> List[str]:
    """
    Collect all supported image files in input_dir (non-recursive).
    """
    supported_exts = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
    image_paths = [
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.lower().endswith(supported_exts)
    ]
    image_paths.sort()
    return image_paths


def load_batch_images(batch_paths: List[str]) -> List[Image.Image]:
    """
    Load images as RGB PIL images.
    """
    images = []
    for path in batch_paths:
        image = Image.open(path).convert("RGB")
        images.append(image)
    return images


def extract_embeddings(
    input_dir: str,
    output_dir: str,
    model_variant: str = "vitb14",
    batch_size: int = 16,
):
    """
    Extract DINOv2 CLS-token embeddings for all images in input_dir.

    Output:
        - dinov2_<variant>_embeddings.npy
        - dinov2_<variant>_image_names.npy
    """
    start_time = time.time()
    device = get_device()

    print(f"Using device: {device}")
    print(f"Loading DINOv2 model: {model_variant}")

    processor, model, output_name = get_model_and_processor(model_variant, device)

    image_paths = collect_image_paths(input_dir)
    if len(image_paths) == 0:
        raise FileNotFoundError(f"No supported image files found in: {input_dir}")

    print(f"Found {len(image_paths)} images.")

    embeddings = []
    image_names = []

    with torch.no_grad():
        for i in tqdm(range(0, len(image_paths), batch_size), desc="Extracting embeddings"):
            batch_paths = image_paths[i:i + batch_size]
            batch_images = load_batch_images(batch_paths)

            inputs = processor(images=batch_images, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(device)

            outputs = model(pixel_values=pixel_values)
            last_hidden_states = outputs.last_hidden_state  # shape: (B, 1 + num_patches, hidden_size)

            # CLS token: one global embedding per image
            cls_embeddings = last_hidden_states[:, 0, :]    # shape: (B, hidden_size)

            embeddings.append(cls_embeddings.cpu().numpy())
            image_names.extend([os.path.basename(p) for p in batch_paths])

    embeddings = np.vstack(embeddings)

    os.makedirs(output_dir, exist_ok=True)

    embeddings_path = os.path.join(output_dir, f"{output_name}_embeddings.npy")
    image_names_path = os.path.join(output_dir, f"{output_name}_image_names.npy")

    np.save(embeddings_path, embeddings)
    np.save(image_names_path, np.array(image_names, dtype=object))

    print("\nExtraction completed.")
    print(f"Model name           : {output_name}")
    print(f"Total images         : {len(image_names)}")
    print(f"Embedding shape      : {embeddings.shape}")
    print(f"Saved embeddings to  : {embeddings_path}")
    print(f"Saved image names to : {image_names_path}")

    runtime = time.time() - start_time
    hours = int(runtime // 3600)
    minutes = int((runtime % 3600) // 60)
    seconds = runtime % 60
    print(f"Total runtime: {hours:02d}:{minutes:02d}:{seconds:05.2f} (HH:MM:SS)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract single-frame DINOv2 CLS-token embeddings from images"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Path to folder containing extracted trigger frames",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to save embedding .npy files",
    )
    parser.add_argument(
        "--model_variant",
        type=str,
        default="vitb14",
        choices=["vits14", "vitb14", "vitl14", "vitg14"],
        help="Which DINOv2 backbone to use",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for inference",
    )

    args = parser.parse_args()

    extract_embeddings(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        model_variant=args.model_variant,
        batch_size=args.batch_size,
    )