"""Compare exact per-item length assignments across signals."""

import argparse
import csv
import itertools
from pathlib import Path

import torch
from scipy.stats import kendalltau, pearsonr, spearmanr


def _load(path: str) -> torch.Tensor:
    values = torch.as_tensor(torch.load(path, map_location="cpu")).long().flatten()
    if values.ndim != 1 or values.numel() == 0:
        raise ValueError(f"Expected a non-empty 1D length tensor: {path}")
    return values


def _correlation(function, first: torch.Tensor, second: torch.Tensor) -> float:
    if first.unique().numel() < 2 or second.unique().numel() < 2:
        return float("nan")
    return float(function(first.numpy(), second.numpy()).statistic)


def summarize(name: str, lengths: torch.Tensor, max_length: int):
    histogram = torch.bincount(lengths, minlength=max_length + 1)
    return {
        "signal": name,
        "num_items": lengths.numel(),
        "min_length": int(lengths.min()),
        "max_length": int(lengths.max()),
        "mean_length": float(lengths.float().mean()),
        "median_length": float(lengths.median()),
        "std_length": float(lengths.float().std(unbiased=False)),
        "fraction_at_max_length": float((lengths == max_length).float().mean()),
        **{
            f"length_{length}": int(histogram[length])
            for length in range(1, max_length + 1)
        },
        **{
            f"fraction_length_{length}": float(histogram[length] / lengths.numel())
            for length in range(1, max_length + 1)
        },
    }


def compare(first_name, first, second_name, second):
    size = min(first.numel(), second.numel())
    first = first[:size]
    second = second[:size]
    difference = (first - second).abs()
    return {
        "signal_a": first_name,
        "signal_b": second_name,
        "common_items": size,
        "exact_length_agreement": float((difference == 0).float().mean()),
        "within_one_length_agreement": float((difference <= 1).float().mean()),
        "mean_absolute_length_difference": float(difference.float().mean()),
        "pearson_length_correlation": _correlation(pearsonr, first, second),
        "spearman_length_correlation": _correlation(spearmanr, first, second),
        "kendall_length_correlation": _correlation(kendalltau, first, second),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", action="append", required=True, help="NAME=PATH")
    parser.add_argument("--max-length", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    artifacts = {}
    for value in args.artifact:
        name, path = value.split("=", 1)
        artifacts[name] = _load(path)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summaries = [
        summarize(name, lengths, args.max_length)
        for name, lengths in artifacts.items()
    ]
    with (args.output_dir / "length_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)

    comparisons = [
        compare(first_name, artifacts[first_name], second_name, artifacts[second_name])
        for first_name, second_name in itertools.combinations(artifacts, 2)
    ]
    if comparisons:
        with (args.output_dir / "length_pairwise.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(comparisons[0]))
            writer.writeheader()
            writer.writerows(comparisons)


if __name__ == "__main__":
    main()
