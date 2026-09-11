"""Compare per-item budget assignments across budget signals."""

import argparse
import csv
import os
from itertools import combinations

import tensorflow as tf
import torch
from scipy.stats import kendalltau, pearsonr, spearmanr

from src.data.loading.components.iterators import TFRecordIterator
from src.utils.file_utils import list_files


DEFAULT_SIGNALS = [
    "interaction_count",
    "cooccurrence",
    "ppmi",
    "neighborhood_entropy",
    "graph_centrality",
    "item2vec",
    "bpr",
    "lightgcn",
]


def load_interaction_counts(data_dir: str) -> torch.Tensor:
    files = list_files(
        folder_path=os.path.join(data_dir, "training"), suffix="*tfrecord.gz"
    )
    if not files:
        raise FileNotFoundError(f"No training TFRecords found under {data_dir}")
    iterator = TFRecordIterator()
    iterator.update_list_of_file_paths(files)
    iterator.should_drop_last_batch = False
    counts = {}
    for batch in iterator.iter_batches(batch_size=1024):
        sequence = batch["sequence_data"]
        if isinstance(sequence, tf.SparseTensor):
            values = sequence.values.numpy()
        else:
            values = sequence.numpy().reshape(-1)
        for item_id in values:
            item_id = int(item_id)
            counts[item_id] = counts.get(item_id, 0) + 1
    result = torch.zeros(max(counts) + 1, dtype=torch.float32)
    for item_id, count in counts.items():
        result[item_id] = count
    return result


def top_overlap(first: torch.Tensor, second: torch.Tensor, fraction: float) -> float:
    top_k = max(1, int(first.numel() * fraction))
    first_top = set(torch.topk(first, top_k).indices.tolist())
    second_top = set(torch.topk(second, top_k).indices.tolist())
    return len(first_top & second_top) / len(first_top | second_top)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="beauty")
    parser.add_argument("--direction", choices=["direct", "inverse"], default="direct")
    parser.add_argument("--logs-dir", default="logs")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--output-dir", default="logs/budget_analysis")
    parser.add_argument("--signals", nargs="+", default=DEFAULT_SIGNALS)
    args = parser.parse_args()

    budgets = {}
    for signal in args.signals:
        path = os.path.join(
            args.logs_dir,
            args.dataset,
            f"{signal}_{args.direction}_budgets.pt",
        )
        if os.path.exists(path):
            budgets[signal] = torch.load(path, map_location="cpu").long()
        else:
            print(f"Skipping missing budget artifact: {path}")

    if len(budgets) < 2:
        raise ValueError("At least two budget artifacts are required.")

    interaction_counts = load_interaction_counts(
        args.data_dir or os.path.join("data", "amazon_data", args.dataset)
    )
    common_length = min(
        interaction_counts.numel(), *(values.numel() for values in budgets.values())
    )
    interaction_counts = interaction_counts[:common_length]
    sorted_items = torch.argsort(interaction_counts)
    tail_items = sorted_items[: max(1, common_length // 5)]
    head_items = sorted_items[-max(1, common_length // 5) :]
    mid_items = sorted_items[max(1, common_length // 5) : -max(1, common_length // 5)]

    os.makedirs(args.output_dir, exist_ok=True)
    summary_path = os.path.join(
        args.output_dir, f"{args.dataset}_{args.direction}_summary.csv"
    )
    with open(summary_path, "w", newline="") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=[
                "signal",
                "num_items",
                "mean_budget",
                "std_budget",
                "median_budget",
                "min_budget",
                "max_budget",
                "fraction_budget_1",
                "fraction_budget_10",
                "head_mean_budget",
                "mid_mean_budget",
                "tail_mean_budget",
                "head_median_budget",
                "mid_median_budget",
                "tail_median_budget",
                *[f"budget_{budget}" for budget in range(1, 11)],
                *[f"head_budget_{budget}" for budget in range(1, 11)],
                *[f"mid_budget_{budget}" for budget in range(1, 11)],
                *[f"tail_budget_{budget}" for budget in range(1, 11)],
            ],
        )
        writer.writeheader()
        for signal, values in budgets.items():
            histogram = torch.bincount(values, minlength=11)
            head_histogram = torch.bincount(values[head_items], minlength=11)
            mid_histogram = torch.bincount(values[mid_items], minlength=11)
            tail_histogram = torch.bincount(values[tail_items], minlength=11)
            writer.writerow(
                {
                    "signal": signal,
                    "num_items": values.numel(),
                    "mean_budget": float(values.float().mean()),
                    "std_budget": float(values.float().std()),
                    "median_budget": float(values.median()),
                    "min_budget": int(values.min()),
                    "max_budget": int(values.max()),
                    "fraction_budget_1": float((values == 1).float().mean()),
                    "fraction_budget_10": float((values == 10).float().mean()),
                    "head_mean_budget": float(values[head_items].float().mean()),
                    "mid_mean_budget": float(values[mid_items].float().mean()),
                    "tail_mean_budget": float(values[tail_items].float().mean()),
                    "head_median_budget": float(values[head_items].median()),
                    "mid_median_budget": float(values[mid_items].median()),
                    "tail_median_budget": float(values[tail_items].median()),
                    **{
                        f"budget_{budget}": int(histogram[budget])
                        for budget in range(1, 11)
                    },
                    **{
                        f"head_budget_{budget}": int(head_histogram[budget])
                        for budget in range(1, 11)
                    },
                    **{
                        f"mid_budget_{budget}": int(mid_histogram[budget])
                        for budget in range(1, 11)
                    },
                    **{
                        f"tail_budget_{budget}": int(tail_histogram[budget])
                        for budget in range(1, 11)
                    },
                }
            )

    pairwise_path = os.path.join(
        args.output_dir, f"{args.dataset}_{args.direction}_pairwise.csv"
    )
    with open(pairwise_path, "w", newline="") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=[
                "signal_a",
                "signal_b",
                "pearson",
                "spearman",
                "kendall",
                "exact_agreement",
                "within_one_agreement",
                "mean_absolute_difference",
                "top10_jaccard",
            ],
        )
        writer.writeheader()
        for signal_a, signal_b in combinations(budgets, 2):
            length = min(budgets[signal_a].numel(), budgets[signal_b].numel())
            first = budgets[signal_a][:length]
            second = budgets[signal_b][:length]
            difference = (first - second).abs()
            writer.writerow(
                {
                    "signal_a": signal_a,
                    "signal_b": signal_b,
                    "pearson": float(pearsonr(first.numpy(), second.numpy()).statistic),
                    "spearman": float(spearmanr(first.numpy(), second.numpy()).statistic),
                    "kendall": float(kendalltau(first.numpy(), second.numpy()).statistic),
                    "exact_agreement": float((difference == 0).float().mean()),
                    "within_one_agreement": float((difference <= 1).float().mean()),
                    "mean_absolute_difference": float(difference.float().mean()),
                    "top10_jaccard": top_overlap(first, second, 0.10),
                }
            )

    print(f"Wrote {summary_path}")
    print(f"Wrote {pairwise_path}")


if __name__ == "__main__":
    main()
