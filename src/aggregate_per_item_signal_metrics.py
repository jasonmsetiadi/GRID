"""Aggregate final TIGER metrics for per-item signal experiments."""

import argparse
import csv
from pathlib import Path

import torch


def _latest_metrics(run_dir: Path):
    paths = list(run_dir.glob("train/runs/*/csv/version_*/metrics.csv"))
    return max(paths, key=lambda path: path.stat().st_mtime_ns) if paths else None


def _final_test_metrics(path: Path):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return {}
    names = [
        name
        for name in rows[0]
        if name and (name == "test" or name.startswith("test/") or name.startswith("test_"))
    ]
    for row in reversed(rows):
        values = {name: row[name] for name in names if row.get(name, "") not in ("", None)}
        if values:
            return values
    return {}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", required=True)
    parser.add_argument("--signals", nargs="+", required=True)
    parser.add_argument("--directions", nargs="+", required=True)
    parser.add_argument("--max-length", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    records = []
    metric_names = set()
    for method in args.methods:
        for direction in args.directions:
            for signal in args.signals:
                run_dir = args.run_root / method / f"{signal}_{direction}"
                metrics_path = _latest_metrics(run_dir)
                if metrics_path is None:
                    print(f"Warning: missing metrics for {method}/{signal}/{direction}")
                    continue
                record = {
                    "dataset": args.dataset,
                    "sid_method": method,
                    "signal": signal,
                    "direction": direction,
                }
                length_path = args.run_root / "lengths" / f"item_lengths_{signal}_{direction}.pt"
                if length_path.exists():
                    lengths = torch.as_tensor(torch.load(length_path, map_location="cpu")).float()
                    record.update(
                        mean_sid_length=float(lengths.mean()),
                        median_sid_length=float(lengths.median()),
                        fraction_at_max_length=float((lengths == args.max_length).float().mean()),
                    )
                metrics = _final_test_metrics(metrics_path)
                metric_names.update(metrics)
                record.update(metrics)
                records.append(record)

    if not records:
        raise RuntimeError("No per-item signal metrics found")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "dataset", "sid_method", "signal", "direction",
        "mean_sid_length", "median_sid_length", "fraction_at_max_length",
    ] + sorted(metric_names)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


if __name__ == "__main__":
    main()
