"""
Package publication-ready revision artifacts from experiment outputs.

This script does not rerun experiments. It collects the latest available
artifacts from the `results/` directory, copies them into `revision_results/`
with stable manuscript-facing filenames, generates a small metadata manifest,
and writes an index describing each artifact and how it was produced.

It is safe to run repeatedly.

Examples:
    python3 prepare_revision_results.py
    python3 prepare_revision_results.py --results-dir results --output-dir revision_results
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class ArtifactSpec:
    logical_name: str
    source_filename: str
    output_filename: str
    category: str
    description: str
    producer_script: str


ARTIFACT_SPECS: Sequence[ArtifactSpec] = [
    ArtifactSpec(
        logical_name="baseline_table_csv",
        source_filename="comparison_table.csv",
        output_filename="table_baselines.csv",
        category="table",
        description="Baseline comparison table for raw tabular, graph-embedding, and combined models.",
        producer_script="experiments_baselines.py",
    ),
    ArtifactSpec(
        logical_name="baseline_table_md",
        source_filename="comparison_table.md",
        output_filename="table_baselines.md",
        category="table",
        description="Markdown version of the baseline comparison table.",
        producer_script="experiments_baselines.py",
    ),
    ArtifactSpec(
        logical_name="baseline_table_tex",
        source_filename="comparison_table.tex",
        output_filename="table_baselines.tex",
        category="table",
        description="LaTeX-ready baseline comparison table.",
        producer_script="experiments_baselines.py",
    ),
    ArtifactSpec(
        logical_name="ablation_table_csv",
        source_filename="ablation_results.csv",
        output_filename="table_ablations.csv",
        category="table",
        description="Ablation table comparing full, reduced, and baseline configurations.",
        producer_script="experiments_ablation.py",
    ),
    ArtifactSpec(
        logical_name="ablation_table_md",
        source_filename="ablation_results.md",
        output_filename="table_ablations.md",
        category="table",
        description="Markdown version of the ablation table.",
        producer_script="experiments_ablation.py",
    ),
    ArtifactSpec(
        logical_name="ablation_table_tex",
        source_filename="ablation_results.tex",
        output_filename="table_ablations.tex",
        category="table",
        description="LaTeX-ready ablation table.",
        producer_script="experiments_ablation.py",
    ),
    ArtifactSpec(
        logical_name="weight_sensitivity_csv",
        source_filename="weight_sensitivity_results.csv",
        output_filename="table_weight_sensitivity.csv",
        category="table",
        description="Weight sensitivity table comparing heuristic, uniform, and perturbed weight configurations.",
        producer_script="experiments_weight_sensitivity.py",
    ),
    ArtifactSpec(
        logical_name="weight_sensitivity_md",
        source_filename="weight_sensitivity_results.md",
        output_filename="table_weight_sensitivity.md",
        category="table",
        description="Markdown version of the weight sensitivity table.",
        producer_script="experiments_weight_sensitivity.py",
    ),
    ArtifactSpec(
        logical_name="weight_sensitivity_tex",
        source_filename="weight_sensitivity_results.tex",
        output_filename="table_weight_sensitivity.tex",
        category="table",
        description="LaTeX-ready weight sensitivity table.",
        producer_script="experiments_weight_sensitivity.py",
    ),
    ArtifactSpec(
        logical_name="shap_global_importance_csv",
        source_filename="global_shap_importance.csv",
        output_filename="table_global_shap_importance.csv",
        category="table",
        description="Global SHAP importance over embedding dimensions for the main graph classifier.",
        producer_script="experiments_interpretability.py",
    ),
    ArtifactSpec(
        logical_name="shap_summary_bar_png",
        source_filename="shap_summary_bar.png",
        output_filename="figure_shap_global_bar.png",
        category="figure",
        description="Publication-friendly global SHAP bar plot.",
        producer_script="experiments_interpretability.py",
    ),
    ArtifactSpec(
        logical_name="shap_summary_beeswarm_png",
        source_filename="shap_summary_beeswarm.png",
        output_filename="figure_shap_beeswarm.png",
        category="figure",
        description="SHAP beeswarm plot for the graph classifier.",
        producer_script="experiments_interpretability.py",
    ),
    ArtifactSpec(
        logical_name="feature_mapping_csv",
        source_filename="embedding_feature_mapping.csv",
        output_filename="table_embedding_feature_mapping.csv",
        category="table",
        description="Approximate mapping from important embedding dimensions back to Q-Chat and demographic variables.",
        producer_script="experiments_interpretability.py",
    ),
    ArtifactSpec(
        logical_name="case_level_explanations_csv",
        source_filename="case_level_explanations.csv",
        output_filename="table_case_level_explanations.csv",
        category="table",
        description="Case-level SHAP explanations for selected holdout examples.",
        producer_script="experiments_interpretability.py",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect manuscript-ready revision artifacts.")
    parser.add_argument("--results-dir", default="results", help="Directory containing experiment subfolders.")
    parser.add_argument("--output-dir", default="revision_results", help="Directory to write packaged revision artifacts.")
    return parser.parse_args()


def ensure_output_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def latest_subdir_with_file(results_dir: Path, filename: str) -> Optional[Path]:
    candidates = [path for path in results_dir.iterdir() if path.is_dir() and (path / filename).exists()]
    if not candidates:
        return None
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0]


def copy_if_available(source_dir: Optional[Path], filename: str, destination: Path) -> bool:
    if source_dir is None:
        return False
    source = source_dir / filename
    if not source.exists():
        return False
    shutil.copy2(source, destination)
    return True


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def write_csv_rows(path: Path, rows: Sequence[Dict[str, str]], fieldnames: Sequence[str]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_kg_explanation_example(output_dir: Path) -> Tuple[bool, Optional[str]]:
    case_path = output_dir / "table_case_level_explanations.csv"
    mapping_path = output_dir / "table_embedding_feature_mapping.csv"
    out_path = output_dir / "table_kg_explanation_example.csv"

    if not case_path.exists() or not mapping_path.exists():
        return False, "Requires both case-level SHAP explanations and embedding-feature mapping outputs."

    case_rows = read_csv_rows(case_path)
    mapping_rows = read_csv_rows(mapping_path)
    if not case_rows or not mapping_rows:
        return False, "Input explanation files are present but empty."

    best_case_id = case_rows[0]["case_id"]
    selected_case_rows = [row for row in case_rows if row["case_id"] == best_case_id][:5]

    mapping_index: Dict[str, List[Dict[str, str]]] = {}
    for row in mapping_rows:
        mapping_index.setdefault(row["embedding_dim"], []).append(row)

    output_rows: List[Dict[str, str]] = []
    for row in selected_case_rows:
        mapped_features = mapping_index.get(row["embedding_dim"], [])[:3]
        if mapped_features:
            for mapped in mapped_features:
                output_rows.append(
                    {
                        "case_id": row["case_id"],
                        "embedding_dim": row["embedding_dim"],
                        "shap_value": row["shap_value"],
                        "embedding_value": row["embedding_value"],
                        "approx_feature": mapped["approx_feature"],
                        "approx_feature_group": mapped["approx_feature_group"],
                        "approx_correlation": mapped["correlation"],
                        "interpretation_note": "Approximate bridge from local embedding attribution to original graph/input variables.",
                    }
                )
        else:
            output_rows.append(
                {
                    "case_id": row["case_id"],
                    "embedding_dim": row["embedding_dim"],
                    "shap_value": row["shap_value"],
                    "embedding_value": row["embedding_value"],
                    "approx_feature": "",
                    "approx_feature_group": "",
                    "approx_correlation": "",
                    "interpretation_note": "No approximate mapping found for this embedding dimension.",
                }
            )

    fieldnames = [
        "case_id",
        "embedding_dim",
        "shap_value",
        "embedding_value",
        "approx_feature",
        "approx_feature_group",
        "approx_correlation",
        "interpretation_note",
    ]
    write_csv_rows(out_path, output_rows, fieldnames)
    return True, None


def build_index(
    output_dir: Path,
    manifest_rows: Sequence[Dict[str, str]],
    generated_at_utc: str,
) -> None:
    index_path = output_dir / "INDEX.md"
    lines = [
        "# Revision Results",
        "",
        f"Generated at: `{generated_at_utc}`",
        "",
        "This folder packages manuscript-facing artifacts from the latest available experiment outputs under `results/`.",
        "",
        "## Artifacts",
        "",
        "| Output File | Category | Status | Source Run | Producer | Description |",
        "| --- | --- | --- | --- | --- | --- |",
    ]

    for row in manifest_rows:
        lines.append(
            f"| `{row['output_file']}` | {row['category']} | {row['status']} | "
            f"`{row['source_run']}` | `{row['producer_script']}` | {row['description']} |"
        )

    lines.extend(
        [
            "",
            "## How To Produce Missing Artifacts",
            "",
            "1. Run `python3 experiments_baselines.py`",
            "2. Run `python3 experiments_ablation.py`",
            "3. Run `python3 experiments_weight_sensitivity.py`",
            "4. Run `python3 experiments_interpretability.py`",
            "5. Re-run `python3 prepare_revision_results.py`",
            "",
            "## Notes",
            "",
            "- `table_kg_explanation_example.csv` is a derived packaging artifact created from case-level SHAP explanations plus the approximate embedding-feature mapping.",
            "- The embedding-to-feature bridge remains approximate and should be described that way in the manuscript.",
        ]
    )

    index_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir).resolve()
    output_dir = ensure_output_dir(Path(args.output_dir).resolve())
    generated_at_utc = datetime.now(UTC).isoformat().replace("+00:00", "Z")

    manifest_rows: List[Dict[str, str]] = []

    for spec in ARTIFACT_SPECS:
        source_dir = latest_subdir_with_file(results_dir, spec.source_filename) if results_dir.exists() else None
        destination = output_dir / spec.output_filename
        copied = copy_if_available(source_dir, spec.source_filename, destination)

        manifest_rows.append(
            {
                "logical_name": spec.logical_name,
                "output_file": spec.output_filename,
                "category": spec.category,
                "status": "present" if copied else "missing",
                "source_run": source_dir.name if source_dir else "",
                "producer_script": spec.producer_script,
                "description": spec.description,
            }
        )

    kg_example_ok, kg_example_note = build_kg_explanation_example(output_dir)
    manifest_rows.append(
        {
            "logical_name": "kg_explanation_example_csv",
            "output_file": "table_kg_explanation_example.csv",
            "category": "table",
            "status": "present" if kg_example_ok else "missing",
            "source_run": "derived_from_packaged_outputs" if kg_example_ok else "",
            "producer_script": "prepare_revision_results.py",
            "description": "Derived example linking one local SHAP explanation to approximate Q-Chat/demographic feature mappings."
            + ("" if kg_example_ok else f" Missing reason: {kg_example_note}"),
        }
    )

    manifest_path = output_dir / "artifact_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "generated_at_utc": generated_at_utc,
                "results_dir": str(results_dir),
                "output_dir": str(output_dir),
                "artifacts": manifest_rows,
            },
            f,
            indent=2,
        )

    captions_path = output_dir / "artifact_captions.json"
    with open(captions_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                row["output_file"]: {
                    "category": row["category"],
                    "description": row["description"],
                    "producer_script": row["producer_script"],
                    "status": row["status"],
                }
                for row in manifest_rows
            },
            f,
            indent=2,
        )

    build_index(output_dir, manifest_rows, generated_at_utc)

    print(f"Prepared revision artifact package in: {output_dir}")
    present_count = sum(1 for row in manifest_rows if row["status"] == "present")
    print(f"Present artifacts: {present_count}/{len(manifest_rows)}")


if __name__ == "__main__":
    main()
