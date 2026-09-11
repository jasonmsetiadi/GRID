"""Create interaction-count-based semantic-ID budgets from training records."""

import argparse
import math
import os
from collections import Counter, defaultdict
from itertools import combinations

import tensorflow as tf
import torch

from src.data.loading.components.iterators import TFRecordIterator
from src.utils.file_utils import list_files


def create_item_budgets(
    data_dir: str,
    output_path: str,
    signal: str = "interaction_count",
    top_k: int = 20,
    direction: str = "direct",
    min_budget: int = 1,
    max_budget: int = 10,
) -> None:
    files = list_files(
        folder_path=os.path.join(data_dir, "training"), suffix="*tfrecord.gz"
    )
    if not files:
        raise FileNotFoundError(f"No training TFRecords found under {data_dir}")
    if min_budget < 1 or max_budget < min_budget:
        raise ValueError("Budget bounds must satisfy 1 <= min_budget <= max_budget.")
    if signal not in {
        "interaction_count",
        "cooccurrence",
        "ppmi",
        "neighborhood_entropy",
        "graph_centrality",
    }:
        raise ValueError(
            "signal must be interaction_count, cooccurrence, ppmi, "
            "neighborhood_entropy, or graph_centrality."
        )
    if top_k < 1:
        raise ValueError("top_k must be positive.")
    if direction not in {"direct", "inverse"}:
        raise ValueError("direction must be 'direct' or 'inverse'.")

    iterator = TFRecordIterator()
    iterator.update_list_of_file_paths(files)
    iterator.should_drop_last_batch = False
    scores = Counter()
    item_sequence_counts = Counter()
    pair_counts = Counter()
    sequence_count = 0
    for batch in iterator.iter_batches(batch_size=1024):
        sequence = batch["sequence_data"]
        if isinstance(sequence, tf.SparseTensor):
            sequence = tf.sparse.to_dense(sequence)
        for raw_sequence in sequence.numpy():
            item_ids = [int(item_id) for item_id in raw_sequence.reshape(-1)]
            if signal == "interaction_count":
                scores.update(item_ids)
            else:
                unique_item_ids = set(item_ids)
                cooccurring_items = max(len(unique_item_ids) - 1, 0)
                for item_id in unique_item_ids:
                    scores[item_id] += cooccurring_items
                if signal in {"ppmi", "neighborhood_entropy", "graph_centrality"}:
                    item_sequence_counts.update(unique_item_ids)
                    pair_counts.update(combinations(sorted(unique_item_ids), 2))
                sequence_count += 1

    if signal == "ppmi":
        ppmi_contributions = defaultdict(list)
        for (item_i, item_j), pair_count in pair_counts.items():
            pmi = math.log(
                pair_count * sequence_count
                / (item_sequence_counts[item_i] * item_sequence_counts[item_j])
            )
            if pmi > 0:
                ppmi_contributions[item_i].append(pmi)
                ppmi_contributions[item_j].append(pmi)
        scores = Counter({item_id: 0.0 for item_id in item_sequence_counts})
        for item_id, contributions in ppmi_contributions.items():
            scores[item_id] = sum(sorted(contributions, reverse=True)[:top_k])
    elif signal == "neighborhood_entropy":
        neighbor_counts = defaultdict(Counter)
        for (item_i, item_j), pair_count in pair_counts.items():
            neighbor_counts[item_i][item_j] = pair_count
            neighbor_counts[item_j][item_i] = pair_count
        scores = Counter()
        for item_id, neighbors in neighbor_counts.items():
            total = sum(neighbors.values())
            scores[item_id] = -sum(
                (count / total) * math.log(count / total)
                for count in neighbors.values()
            )
        scores.update({item_id: 0.0 for item_id in item_sequence_counts})
    elif signal == "graph_centrality":
        # Weighted degree centrality: total edge weight incident to the item.
        scores = Counter()
        for (item_i, item_j), pair_count in pair_counts.items():
            scores[item_i] += pair_count
            scores[item_j] += pair_count
        scores.update({item_id: 0.0 for item_id in item_sequence_counts})

    if not scores:
        raise ValueError("Training records contained no item IDs.")

    max_item_id = max(scores)
    interaction_counts = torch.zeros(max_item_id + 1, dtype=torch.float32)
    for item_id, score in scores.items():
        interaction_counts[item_id] = score

    scores = torch.log1p(interaction_counts)
    nonzero_scores = scores[interaction_counts > 0]
    if nonzero_scores.numel() == 0:
        normalized_scores = torch.zeros_like(scores)
    elif nonzero_scores.numel() == 1 or nonzero_scores.max() == nonzero_scores.min():
        normalized_scores = torch.ones_like(scores)
    else:
        normalized_scores = (scores - nonzero_scores.min()) / (
            nonzero_scores.max() - nonzero_scores.min()
        )
        normalized_scores[interaction_counts == 0] = 0

    if direction == "inverse":
        positive_mask = interaction_counts > 0
        normalized_scores[positive_mask] = 1 - normalized_scores[positive_mask]

    budgets = torch.round(
        min_budget + normalized_scores * (max_budget - min_budget)
    ).long()
    budgets[interaction_counts == 0] = min_budget

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(budgets, output_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument(
        "--signal",
        choices=[
            "interaction_count",
            "cooccurrence",
            "ppmi",
            "neighborhood_entropy",
            "graph_centrality",
        ],
        default="interaction_count",
    )
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--direction", choices=["direct", "inverse"], default="direct")
    parser.add_argument("--min-budget", type=int, default=1)
    parser.add_argument("--max-budget", type=int, default=10)
    args = parser.parse_args()
    create_item_budgets(
        data_dir=args.data_dir,
        output_path=args.output_path,
        signal=args.signal,
        top_k=args.top_k,
        direction=args.direction,
        min_budget=args.min_budget,
        max_budget=args.max_budget,
    )


if __name__ == "__main__":
    main()
