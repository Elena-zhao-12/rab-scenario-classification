"""Generic image embedding, clustering and evaluation utilities."""
import os
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
import time

import torch
import torch.nn as nn
from torchvision import models, transforms


def get_model(device):
    """
    Load pretrained ResNet50 and remove the final classification layer.
    The output will be a 2048-dimensional feature vector.
    """
    model = models.resnet50(pretrained=True)

    # Remove the final fully connected layer (fc)
    model = nn.Sequential(*list(model.children())[:-1])

    model.to(device)
    model.eval()

    return model


def get_transform():
    """
    Define image preprocessing pipeline.
    Must match ImageNet normalization since ResNet50 is pretrained on ImageNet.
    """
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    return transform


def extract_embeddings(input_dir, output_dir, batch_size=32):
    """
    Extract embeddings for all PNG images in input_dir
    and save embeddings + image names in output_dir.
    """

    start_time = time.time()   # start timing
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_model(device)
    transform = get_transform()

    image_paths = [
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.lower().endswith(".png")
    ]

    image_paths.sort()

    embeddings = []
    image_names = []

    with torch.no_grad():
        for i in tqdm(range(0, len(image_paths), batch_size)):

            batch_paths = image_paths[i:i + batch_size]
            batch_images = []

            for path in batch_paths:
                image = Image.open(path).convert("RGB")
                image = transform(image)
                batch_images.append(image)

            batch_tensor = torch.stack(batch_images).to(device)

            outputs = model(batch_tensor)

            # outputs shape: (batch_size, 2048, 1, 1)
            outputs = outputs.squeeze(-1).squeeze(-1)

            embeddings.append(outputs.cpu().numpy())
            image_names.extend([os.path.basename(p) for p in batch_paths])

    embeddings = np.vstack(embeddings)

    os.makedirs(output_dir, exist_ok=True)

    np.save(os.path.join(output_dir, "resnet50_embeddings.npy"), embeddings)
    np.save(os.path.join(output_dir, "resnet50_image_names.npy"), image_names)

    print("Extraction completed.")
    print(f"Total images processed: {len(image_names)}")
    print(f"Embedding shape: {embeddings.shape}")

    end_time = time.time()
    runtime = end_time - start_time

    hours = int(runtime // 3600)
    minutes = int((runtime % 3600) // 60)
    seconds = runtime % 60

    print(f"Total runtime: {hours:02d}:{minutes:02d}:{seconds:05.2f} (HH:MM:SS)")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Path to folder containing PNG images")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Path to save embeddings")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for inference")

    args = parser.parse_args()

    extract_embeddings(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size
    )