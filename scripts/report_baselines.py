#!/usr/bin/env python3
"""
Summarise trained model weights, detection thresholds, and per-scene statistics.

Outputs a consolidated CSV that can be used to compare generalisation across water bodies.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Mapping

import pandas as pd

PIXEL_AREA_M2 = 100.0  # Sentinel-2 10 m pixels
DEFAULT_CLASSES = ("floating_debris", "suspended_matter", "plastic_ml")


def summarize_detections(csv_path: Path) -> Dict[str, float]:
    """Aggregate per-class counts and probability averages from detections CSV."""

    summary: Dict[str, float] = {"detections_rows": 0}
    counters = {cls: {"count": 0, "prob_sum": 0.0} for cls in DEFAULT_CLASSES}

    if not csv_path.exists():
        return summary

    for chunk in pd.read_csv(
        csv_path,
        usecols=["class", "probability"],
        chunksize=500_000,
        dtype={"class": "string", "probability": "float32"},
    ):
        grouped = chunk.groupby("class")["probability"].agg(["count", "sum"])
        summary["detections_rows"] += int(grouped["count"].sum())
        for cls, row in grouped.iterrows():
            cls = str(cls)
            if cls not in counters:
                counters[cls] = {"count": 0, "prob_sum": 0.0}
            counters[cls]["count"] += int(row["count"])
            counters[cls]["prob_sum"] += float(row["sum"])

    for cls, stats in counters.items():
        count = stats["count"]
        prob_sum = stats["prob_sum"]
        summary[f"{cls}_count"] = count
        summary[f"{cls}_area_m2"] = float(count * PIXEL_AREA_M2)
        summary[f"{cls}_prob_mean"] = float(prob_sum / count) if count else 0.0

    return summary


def load_threshold_summary(threshold_path: Path) -> Dict[str, float]:
    """Read thresholds.json written by calibration logic."""

    if not threshold_path.exists():
        return {}
    try:
        data = json.loads(threshold_path.read_text(encoding="utf-8"))
    except Exception:
        return {}

    result: Dict[str, float] = {"threshold_metric": data.get("metric")}
    thresholds = data.get("thresholds", {})
    metrics: Mapping[str, Mapping[str, float]] = data.get("per_class_metrics", {})

    for cls, value in thresholds.items():
        result[f"{cls}_threshold"] = float(value)

    for cls, metric_dict in metrics.items():
        for metric_name, metric_value in metric_dict.items():
            key = f"{cls}_{metric_name}"
            result[key] = float(metric_value)

    return result


def load_stacking_summary(stacking_path: Path) -> Dict[str, object]:
    if not stacking_path.exists():
        return {}
    try:
        data = json.loads(stacking_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    out: Dict[str, object] = {
        "stacking_preset": data.get("preset"),
        "stacking_base_models": ";".join(data.get("base_models", []) or []),
        "stacking_meta_model": data.get("meta_model"),
        "stacking_calibrator": data.get("calibrator"),
        "stacking_feature_count": data.get("feature_count"),
        "stacking_sample_count": data.get("sample_count"),
        "stacking_train_sample_count": data.get("train_sample_count"),
    }
    return out


def load_registry(registry_path: Path) -> Dict[str, Dict[str, object]]:
    if not registry_path.exists():
        return {}
    df = pd.read_csv(registry_path)
    if "scene" not in df.columns:
        return {}
    df = df.reset_index().sort_values(["scene", "index"])
    latest = df.groupby("scene", as_index=False).tail(1)
    latest = latest.drop(columns=["index"])
    return latest.set_index("scene").to_dict(orient="index")


def build_report(
    analysis_root: Path,
    registry_map: Dict[str, Dict[str, object]],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    if not analysis_root.exists():
        return rows

    for scene_dir in sorted(analysis_root.iterdir()):
        if not scene_dir.is_dir():
            continue
        scene_id = scene_dir.name
        row: Dict[str, object] = {"scene": scene_id}

        detections_csv = scene_dir / "detections.csv"
        row.update(summarize_detections(detections_csv))

        threshold_json = scene_dir / "thresholds.json"
        row.update(load_threshold_summary(threshold_json))

        stacking_json = scene_dir / "stacking.json"
        row.update(load_stacking_summary(stacking_json))

        if scene_id in registry_map:
            row.update(registry_map[scene_id])

        rows.append(row)

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis-root",
        default="mydata/analysis_outputs",
        help="Directory containing per-scene outputs (default: mydata/analysis_outputs)",
    )
    parser.add_argument(
        "--model-registry",
        default="mydata/model_registry.csv",
        help="Path to model_registry.csv (default: mydata/model_registry.csv)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to write the consolidated CSV (default: mydata/baseline_report.csv)",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=10,
        help="Number of rows to preview in the console (default: 10)",
    )
    args = parser.parse_args()

    analysis_root = Path(args.analysis_root)
    registry_path = Path(args.model_registry)
    registry_map = load_registry(registry_path)

    rows = build_report(analysis_root, registry_map)
    if not rows:
        print("No analysis outputs found; nothing to summarise.")
        return

    df = pd.DataFrame(rows).sort_values("scene").reset_index(drop=True)

    output_path = Path(args.output) if args.output else analysis_root.parent / "baseline_report.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    preview = df.head(max(1, args.max_rows))
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print("\n=== Baseline Report Preview ===")
        print(preview)
        print(f"\nSaved full report to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
