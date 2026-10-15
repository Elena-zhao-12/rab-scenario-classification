"""Generic image embedding, clustering and evaluation utilities."""
import os
import argparse
import shutil
from collections import Counter


def parse_assignments(assignment_file):
    assignments = []
    with open(assignment_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("image_name") or line.startswith("-"):
                continue
            parts = line.rsplit(None, 1)
            if len(parts) != 2:
                continue
            image_name = parts[0].strip()
            try:
                label = int(parts[1].strip())
            except ValueError:
                continue
            assignments.append((image_name, label))
    return assignments


def organize_clusters(assignment_file, image_dir, output_dir, use_copy=False):
    assignments = parse_assignments(assignment_file)
    print(f"Parsed {len(assignments)} assignments from: {assignment_file}")

    label_counts = Counter(label for _, label in assignments)
    n_clusters = len([l for l in label_counts if l >= 0])
    print(f"Clusters found: {n_clusters}, Noise points: {label_counts.get(-1, 0)}")

    print(f"\nScanning image directory: {image_dir} ...")
    name_to_path = {}
    for root, _, files in os.walk(image_dir):
        for fname in files:
            name_to_path[fname] = os.path.join(root, fname)
    print(f"Found {len(name_to_path)} files in image directory.\n")

    if len(name_to_path) == 0:
        print(f"[ERROR] No files found in {image_dir}, check the path!")
        return

    stats = {"copied": 0, "linked": 0, "missing": 0, "clusters": set()}

    for i, (image_name, label) in enumerate(assignments):
        if (i + 1) % 500 == 0 or (i + 1) == len(assignments):
            print(f"  Progress: {i+1}/{len(assignments)}")

        src_path = name_to_path.get(image_name) or name_to_path.get(os.path.basename(image_name))

        if src_path is None:
            print(f"  [MISSING] {image_name}")
            stats["missing"] += 1
            continue

        cluster_folder = "cluster_noise" if label == -1 else f"cluster_{label}"
        target_dir = os.path.join(output_dir, cluster_folder)
        os.makedirs(target_dir, exist_ok=True)

        target_path = os.path.join(target_dir, os.path.basename(image_name))
        if os.path.exists(target_path) or os.path.islink(target_path):
            continue

        if use_copy:
            shutil.copy2(src_path, target_path)
            stats["copied"] += 1
        else:
            os.symlink(os.path.abspath(src_path), target_path)
            stats["linked"] += 1

        stats["clusters"].add(cluster_folder)

    action = "Copied" if use_copy else "Symlinked"
    total_placed = stats["copied"] + stats["linked"]
    print("\n========== Organize Summary ==========")
    print(f"Total assignments    : {len(assignments)}")
    print(f"{action:<20} : {total_placed}")
    print(f"Missing (not found)  : {stats['missing']}")
    print(f"Cluster folders      : {sorted(stats['clusters'])}")
    print(f"Output directory     : {output_dir}")
    print("=======================================")

    if stats["missing"] > 0:
        print(f"\n[WARN] {stats['missing']} images not found.")
        print(f"       Make sure --image_dir points to the folder containing the .png files.")
        print(f"       Expected example file: {assignments[0][0]}")
        sample = list(name_to_path.keys())[:3]
        print(f"       Sample files found in image_dir: {sample}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Organize raw_frames images into cluster subfolders based on cluster_assignments.txt"
    )
    parser.add_argument("--assignment_file", type=str, required=True,
                        help="Path to cluster_assignments.txt")
    parser.add_argument("--image_dir", type=str, required=True,
                        help="Path to raw_frames directory (searched recursively)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output root directory; cluster_0/, cluster_1/, cluster_noise/ created here")
    parser.add_argument("--copy", action="store_true",
                        help="Copy files instead of symlinks (default: symlink, saves disk space)")

    args = parser.parse_args()

    organize_clusters(
        assignment_file=args.assignment_file,
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        use_copy=args.copy
    )
