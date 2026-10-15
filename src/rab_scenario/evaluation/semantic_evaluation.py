"""Generic image embedding, clustering and evaluation utilities."""
import argparse
import hashlib
import json
import sys
import csv
import collections
from pathlib import Path


# Demonstration vocabulary only. Supply your own taxonomy for real labels.
TRIGGER_VALIDITY_MAP = {}
SCENARIO_CAUSES = {"example_cause"}
OBJECT_FAMILIES = {"example_object": "example_family"}

LABEL_FIELDS = ["trigger_validity", "scenario_cause", "object_type"]


def primary_label(raw):
    """Return the first entity of a multi-entity `object_type`, lowercased.

    Records in which several entities jointly account for the activation list
    them in one field separated by commas. Purity assumes one class per sample,
    so the leading entity is taken as the primary one.
    """
    value = (raw or "").strip().lower()
    if not value:
        return ""
    if "," in value:
        value = value.split(",", 1)[0]
    return value.strip()


def normalize(field, raw, taxonomy):
    """Map a raw annotation value to a class of the requested taxonomy."""
    if field == "trigger_validity":
        value = (raw or "").strip().lower()
        return TRIGGER_VALIDITY_MAP.get(value, value)

    if field == "scenario_cause":
        # No value of the closed vocabulary contains a separator, so splitting
        # here would only hide an unconsolidated entry such as `a/b`.
        return (raw or "").strip().lower()

    value = primary_label(raw)
    if taxonomy == "coarse" and value:
        return OBJECT_FAMILIES.get(value, "unmapped")
    return value


def vocabulary_report(rows):
    """Print the values that fall outside the expected vocabularies."""
    unresolved = collections.Counter(
        value
        for row in rows
        for value in [normalize("scenario_cause", row.get("scenario_cause"), "fine")]
        if value and value not in SCENARIO_CAUSES
    )
    unmapped = collections.Counter(
        value
        for row in rows
        for value in [primary_label(row.get("object_type"))]
        if value and value not in OBJECT_FAMILIES
    )

    if unresolved:
        print(f"WARNING: {sum(unresolved.values())} rows carry a scenario_cause "
              f"outside the closed vocabulary")
        for value, count in unresolved.most_common():
            print(f"  {count:>4}  {value}")
    else:
        print("Every scenario_cause value belongs to the closed vocabulary")

    if unmapped:
        print(f"WARNING: {sum(unmapped.values())} rows carry an object_type with "
              f"no coarse family; they are counted as `unmapped`")
        for value, count in unmapped.most_common():
            print(f"  {count:>4}  {value}")
    else:
        print("Every object_type value has a coarse family")


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------

def load_assignments(assignment_path):
    """Read image_name -> cluster label from a cluster_assignments.txt."""
    assignments = {}
    lines = assignment_path.read_text(encoding="utf-8").splitlines()[2:]
    for line in lines:
        if not line.strip():
            continue
        image_name, label = line.rsplit(maxsplit=1)
        assignments[image_name.strip()] = int(label)
    return assignments


def load_annotations(annotation_path, assignments):
    """Join explicitly named images to assignments without private ID conventions."""
    rows, mismatches, missing = [], [], []
    with annotation_path.open(encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            name = (row.get("image_name") or "").strip()
            if name not in assignments:
                missing.append(name or "<missing image_name>")
                continue
            label = assignments[name]
            recorded = (row.get("source_cluster_label") or "").strip()
            if recorded and int(recorded) != label:
                mismatches.append((name, recorded, label))
            row["cluster"] = label
            rows.append(row)
    return rows, mismatches, missing, 0


# ---------------------------------------------------------------------------
# Purity
# ---------------------------------------------------------------------------

def cluster_distributions(rows, field, taxonomy, min_confidence=None):
    """Label counts per non-noise cluster, and the counts of the noise group."""
    clusters = collections.defaultdict(collections.Counter)
    noise = collections.Counter()

    for row in rows:
        label = normalize(field, row.get(field), taxonomy)
        if not label:
            continue
        if min_confidence is not None:
            raw_confidence = (row.get("confidence") or "").strip()
            if not raw_confidence or float(raw_confidence) < min_confidence:
                continue
        if row["cluster"] == -1:
            noise[label] += 1
        else:
            clusters[row["cluster"]][label] += 1

    return clusters, noise


def weighted_purity(clusters):
    """Sum of dominant-label counts divided by the total labeled count."""
    dominant = sum(max(counts.values()) for counts in clusters.values())
    total = sum(sum(counts.values()) for counts in clusters.values())
    return (dominant / total if total else None), dominant, total


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_csv(path, fieldnames, rows):
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def distribution_text(counts):
    return "; ".join(f"{label}:{count}" for label, count in counts.most_common())


def evaluate(annotation_path, assignment_path, output_dir, run_id):
    assignments = load_assignments(assignment_path)
    rows, mismatches, missing, recovered = load_annotations(
        annotation_path, assignments
    )

    print(f"Assignments loaded     : {len(assignments)}")
    print(f"Annotated frames joined: {len(rows)}")
    if missing:
        print(f"WARNING: {len(missing)} annotated frames are not in the assignment file")
        for name in missing[:5]:
            print(f"  missing: {name}")
    if mismatches:
        print(f"WARNING: {len(mismatches)} rows disagree with source_cluster_label")
        for name, recorded, actual in mismatches[:5]:
            print(f"  {name}: table={recorded}, assignments={actual}")
    else:
        print("source_cluster_label agrees with the assignment file for every row")

    vocabulary_report(rows)

    has_confidence = bool(rows) and "confidence" in rows[0]
    if not has_confidence:
        print(
            "No confidence column in this table; the confidence-1 sensitivity "
            "columns are left empty"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    per_cluster_rows = []
    noise_rows = []

    for field in LABEL_FIELDS:
        taxonomies = ["fine", "coarse"] if field == "object_type" else ["fine"]

        for taxonomy in taxonomies:
            clusters, noise = cluster_distributions(rows, field, taxonomy)
            purity, dominant, total = weighted_purity(clusters)

            if has_confidence:
                high_clusters, _ = cluster_distributions(
                    rows, field, taxonomy, min_confidence=1.0
                )
                high_purity, _, high_total = weighted_purity(high_clusters)
            else:
                high_purity, high_total = None, None

            summary_rows.append(
                {
                    "run_id": run_id,
                    "label_field": field,
                    "taxonomy": taxonomy,
                    "classes": len({
                        label
                        for counts in clusters.values()
                        for label in counts
                    }),
                    "clusters_covered": len(clusters),
                    "labeled_samples": total,
                    "weighted_purity": round(purity, 4) if purity else None,
                    "perfect_clusters": sum(
                        1
                        for counts in clusters.values()
                        if max(counts.values()) == sum(counts.values())
                    ),
                    "labeled_samples_confidence_1": high_total,
                    "weighted_purity_confidence_1": (
                        round(high_purity, 4) if high_purity else None
                    ),
                    "noise_samples": sum(noise.values()),
                }
            )

            for cluster in sorted(clusters):
                counts = clusters[cluster]
                size = sum(counts.values())
                dominant_label, dominant_count = counts.most_common(1)[0]
                per_cluster_rows.append(
                    {
                        "run_id": run_id,
                        "label_field": field,
                        "taxonomy": taxonomy,
                        "cluster": cluster,
                        "labeled_samples": size,
                        "dominant_label": dominant_label,
                        "dominant_count": dominant_count,
                        "purity": round(dominant_count / size, 4),
                        "label_distribution": distribution_text(counts),
                    }
                )

            noise_rows.append(
                {
                    "run_id": run_id,
                    "label_field": field,
                    "taxonomy": taxonomy,
                    "labeled_samples": sum(noise.values()),
                    "dominant_label": noise.most_common(1)[0][0] if noise else "",
                    "dominant_share": (
                        round(noise.most_common(1)[0][1] / sum(noise.values()), 4)
                        if noise
                        else None
                    ),
                    "label_distribution": distribution_text(noise),
                }
            )

    vocabulary_rows = []
    for field in LABEL_FIELDS:
        raw_values = collections.Counter(
            (row.get(field) or "").strip() for row in rows
        )
        for raw, count in sorted(raw_values.items()):
            if not raw:
                continue
            vocabulary_rows.append(
                {
                    "label_field": field,
                    "raw_value": raw,
                    "count": count,
                    "fine": normalize(field, raw, "fine"),
                    "coarse": normalize(field, raw, "coarse"),
                }
            )

    write_csv(
        output_dir / "purity_summary.csv",
        [
            "run_id", "label_field", "taxonomy", "classes", "clusters_covered",
            "labeled_samples", "weighted_purity", "perfect_clusters",
            "labeled_samples_confidence_1", "weighted_purity_confidence_1",
            "noise_samples",
        ],
        summary_rows,
    )
    write_csv(
        output_dir / "purity_per_cluster.csv",
        [
            "run_id", "label_field", "taxonomy", "cluster", "labeled_samples",
            "dominant_label", "dominant_count", "purity", "label_distribution",
        ],
        per_cluster_rows,
    )
    write_csv(
        output_dir / "noise_composition.csv",
        [
            "run_id", "label_field", "taxonomy", "labeled_samples",
            "dominant_label", "dominant_share", "label_distribution",
        ],
        noise_rows,
    )
    write_csv(
        output_dir / "vocabulary_mapping.csv",
        ["label_field", "raw_value", "count", "fine", "coarse"],
        vocabulary_rows,
    )

    print("\n========== Weighted Purity ==========")
    for row in summary_rows:
        print(
            f"{row['label_field']:<18} {row['taxonomy']:<7}"
            f" classes={row['classes']:>3}"
            f" clusters={row['clusters_covered']:>3}"
            f" labeled={row['labeled_samples']:>4}"
            f" purity={row['weighted_purity']}"
            f" perfect={row['perfect_clusters']}/{row['clusters_covered']}"
        )
    print("=====================================")
    print(f"Saved results to {output_dir}")

    return summary_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--annotation_file",
        type=Path,
        default=Path("data/annotations/current/manual_annotations.csv"),
        help=(
            "Annotation CSV containing image_name and label fields."
        ),
    )
    parser.add_argument(
        "--assignment_file",
        type=Path,
        default=Path(
            "results/runs/EXAMPLE_demo/"
            "cluster_assignments.txt"
        ),
    )
    parser.add_argument(
        "--output_dir", type=Path, default=Path("results/semantic/EXAMPLE")
    )
    parser.add_argument("--run_id", type=str, default="EXAMPLE")
    parser.add_argument("--vocabulary_file", type=Path, help="Optional JSON defining trigger_validity_map, scenario_causes and object_families")
    args = parser.parse_args()
    if args.vocabulary_file:
        vocabulary = json.loads(args.vocabulary_file.read_text(encoding="utf-8"))
        validity = vocabulary.get("trigger_validity_map", {})
        causes = vocabulary.get("scenario_causes", [])
        families = vocabulary.get("object_families", {})
        if not isinstance(validity, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in validity.items()):
            parser.error("trigger_validity_map must map strings to strings")
        if not isinstance(causes, list) or not all(isinstance(v, str) for v in causes):
            parser.error("scenario_causes must be a list of strings")
        if not isinstance(families, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in families.items()):
            parser.error("object_families must map strings to strings")
        TRIGGER_VALIDITY_MAP.clear(); TRIGGER_VALIDITY_MAP.update(validity)
        SCENARIO_CAUSES.clear(); SCENARIO_CAUSES.update(causes)
        OBJECT_FAMILIES.clear(); OBJECT_FAMILIES.update(families)

    evaluate(
        args.annotation_file, args.assignment_file, args.output_dir, args.run_id
    )
    manifest = {
        "run_id": args.run_id,
        "arguments": sys.argv[1:],
        "inputs": [
            {
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path in (args.annotation_file, args.assignment_file, args.vocabulary_file) if path is not None
        ],
    }
    (args.output_dir / "evaluation_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
