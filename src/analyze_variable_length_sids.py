"""Analyze generated variable-length semantic-ID artifacts."""

import argparse
import csv
import itertools
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from scipy.stats import kendalltau, pearsonr, spearmanr


def load_artifact(path: str) -> Dict[str, torch.Tensor]:
    artifact = torch.load(path, map_location="cpu")
    if not isinstance(artifact, dict):
        raise ValueError(f"Expected a structured SID artifact at {path}")

    required = {"codes", "lengths", "item_ids"}
    missing = required.difference(artifact)
    if missing:
        raise ValueError(f"SID artifact {path} is missing fields: {sorted(missing)}")

    codes = torch.as_tensor(artifact["codes"]).long()
    lengths = torch.as_tensor(artifact["lengths"]).long().flatten()
    item_ids = torch.as_tensor(artifact["item_ids"]).long().flatten()
    if codes.ndim != 2:
        raise ValueError(f"codes must be 2D, got shape {tuple(codes.shape)}")
    if codes.size(0) != lengths.numel() or codes.size(0) != item_ids.numel():
        raise ValueError("codes, lengths, and item_ids must have matching row counts")
    if torch.any(lengths < 1) or torch.any(lengths > codes.size(1)):
        raise ValueError("SID lengths must be between 1 and the code width")
    if item_ids.unique().numel() != item_ids.numel():
        raise ValueError("SID artifact contains duplicate item_ids")

    result = {"codes": codes, "lengths": lengths, "item_ids": item_ids}
    if "residual_norms" in artifact:
        residual_norms = torch.as_tensor(artifact["residual_norms"])
        if residual_norms.shape != codes.shape:
            raise ValueError("residual_norms must have the same shape as codes")
        result["residual_norms"] = residual_norms
    return result


def effective_sids(artifact: Dict[str, torch.Tensor]) -> Dict[int, Tuple[int, ...]]:
    return {
        int(item_id): tuple(
            int(code) for code in codes[: int(length)].tolist()
        )
        for item_id, codes, length in zip(
            artifact["item_ids"], artifact["codes"], artifact["lengths"]
        )
    }


def _correlation(function, first: torch.Tensor, second: torch.Tensor) -> float:
    if first.numel() < 2 or first.unique().numel() < 2 or second.unique().numel() < 2:
        return float("nan")
    return float(function(first.numpy(), second.numpy()).statistic)


def summarize_artifact(
    name: str,
    artifact: Dict[str, torch.Tensor],
    max_length: int,
) -> Dict[str, object]:
    lengths = artifact["lengths"]
    sids = effective_sids(artifact)
    unique_count = len(set(sids.values()))
    duplicate_count = len(sids) - unique_count
    row: Dict[str, object] = {
        "signal": name,
        "num_items": len(sids),
        "min_length": int(lengths.min()),
        "max_length": int(lengths.max()),
        "mean_length": float(lengths.float().mean()),
        "median_length": float(lengths.median()),
        "std_length": float(lengths.float().std(unbiased=False)),
        "fraction_at_max_length": float((lengths == max_length).float().mean()),
        "unique_sids": unique_count,
        "duplicate_items": duplicate_count,
        "collision_rate": duplicate_count / max(len(sids), 1),
    }
    for length in range(1, max_length + 1):
        row[f"length_{length}"] = int((lengths == length).sum())
        row[f"fraction_length_{length}"] = float(
            (lengths == length).float().mean()
        )

    if "residual_norms" in artifact:
        residuals = artifact["residual_norms"]
        selected = residuals[
            torch.arange(len(lengths)), lengths - 1
        ].float()
        row["selected_residual_mean"] = float(selected.mean())
        row["selected_residual_median"] = float(selected.median())
        row["selected_residual_max"] = float(selected.max())
    else:
        row["selected_residual_mean"] = ""
        row["selected_residual_median"] = ""
        row["selected_residual_max"] = ""
    return row


def compare_artifacts(
    first_name: str,
    first: Dict[str, torch.Tensor],
    second_name: str,
    second: Dict[str, torch.Tensor],
) -> Dict[str, object]:
    first_sids = effective_sids(first)
    second_sids = effective_sids(second)
    common_ids = sorted(set(first_sids).intersection(second_sids))
    if not common_ids:
        raise ValueError(f"Artifacts {first_name} and {second_name} have no common items")

    first_lengths = torch.tensor([len(first_sids[item]) for item in common_ids])
    second_lengths = torch.tensor([len(second_sids[item]) for item in common_ids])
    differences = (first_lengths - second_lengths).abs()
    first_top = set(
        item
        for item, _ in sorted(
            first_sids.items(), key=lambda pair: len(pair[1]), reverse=True
        )[: max(1, len(first_sids) // 10)]
    )
    second_top = set(
        item
        for item, _ in sorted(
            second_sids.items(), key=lambda pair: len(pair[1]), reverse=True
        )[: max(1, len(second_sids) // 10)]
    )
    union = first_top | second_top
    return {
        "signal_a": first_name,
        "signal_b": second_name,
        "common_items": len(common_ids),
        "exact_length_agreement": float((differences == 0).float().mean()),
        "within_one_length_agreement": float((differences <= 1).float().mean()),
        "mean_absolute_length_difference": float(differences.float().mean()),
        "pearson_length_correlation": _correlation(pearsonr, first_lengths, second_lengths),
        "spearman_length_correlation": _correlation(spearmanr, first_lengths, second_lengths),
        "kendall_length_correlation": _correlation(kendalltau, first_lengths, second_lengths),
        "exact_sid_agreement": float(
            sum(first_sids[item] == second_sids[item] for item in common_ids)
            / len(common_ids)
        ),
        "top_length_item_jaccard": len(first_top & second_top) / max(len(union), 1),
    }


def _parse_artifacts(values: List[str]) -> Dict[str, str]:
    artifacts = {}
    for value in values:
        if "=" not in value:
            raise ValueError("Each --artifact must use NAME=PATH format")
        name, path = value.split("=", 1)
        if not name or not path:
            raise ValueError("Each --artifact must contain a name and path")
        artifacts[name] = path
    if not artifacts:
        raise ValueError("At least one --artifact is required")
    return artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact",
        action="append",
        required=True,
        help="Generated SID artifact in NAME=PATH format; repeat for each signal",
    )
    parser.add_argument("--max-length", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    paths = _parse_artifacts(args.artifact)
    artifacts = {name: load_artifact(path) for name, path in paths.items()}
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = [
        summarize_artifact(name, artifact, args.max_length)
        for name, artifact in artifacts.items()
    ]
    summary_fields = list(summary_rows[0])
    with (args.output_dir / "sid_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    pairwise_rows = [
        compare_artifacts(first_name, artifacts[first_name], second_name, artifacts[second_name])
        for first_name, second_name in itertools.combinations(artifacts, 2)
    ]
    if pairwise_rows:
        with (args.output_dir / "sid_pairwise.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(pairwise_rows[0]))
            writer.writeheader()
            writer.writerows(pairwise_rows)


if __name__ == "__main__":
    main()
