"""Train BPR user/item embeddings and derive per-item budgets."""

import argparse
import os
import random
from collections import Counter
from typing import List, Tuple

import tensorflow as tf
import torch
from torch import nn

from src.data.loading.components.iterators import TFRecordIterator
from src.utils.file_utils import list_files


def _sparse_rows(value):
    if not isinstance(value, tf.SparseTensor):
        return [row.reshape(-1).tolist() for row in value.numpy()]
    rows = [[] for _ in range(int(value.dense_shape[0]))]
    for index, item in zip(value.indices.numpy(), value.values.numpy()):
        rows[int(index[0])].append(int(item))
    return rows


def load_interactions(data_dir: str) -> List[Tuple[int, List[int]]]:
    files = list_files(
        folder_path=os.path.join(data_dir, "training"), suffix="*tfrecord.gz"
    )
    if not files:
        raise FileNotFoundError(f"No training TFRecords found under {data_dir}")

    iterator = TFRecordIterator()
    iterator.update_list_of_file_paths(files)
    iterator.should_drop_last_batch = False
    interactions = []
    for batch in iterator.iter_batches(batch_size=1024):
        user_rows = _sparse_rows(batch["user_id"])
        item_rows = _sparse_rows(batch["sequence_data"])
        for user_row, item_row in zip(user_rows, item_rows):
            if user_row and item_row:
                interactions.append((user_row[0], list(set(item_row))))
    if not interactions:
        raise ValueError("Training records contained no user-item interactions.")
    return interactions


def train_bpr(
    interactions: List[Tuple[int, List[int]]],
    embedding_dim: int = 64,
    negative_samples: int = 1,
    epochs: int = 3,
    batch_size: int = 4096,
    learning_rate: float = 0.01,
    max_pairs: int = 5_000_000,
    seed: int = 42,
) -> torch.Tensor:
    if max_pairs < 1:
        raise ValueError("max_pairs must be positive.")
    random.seed(seed)
    torch.manual_seed(seed)

    user_to_index = {}
    positives = []
    item_counts = Counter()
    for user_id, item_ids in interactions:
        user_index = user_to_index.setdefault(user_id, len(user_to_index))
        for item_id in item_ids:
            positives.append((user_index, item_id))
            item_counts[item_id] += 1

    if len(positives) > max_pairs:
        positives = random.sample(positives, max_pairs)
    max_item_id = max(item_counts)
    num_items = max_item_id + 1
    user_embeddings = nn.Embedding(len(user_to_index), embedding_dim)
    item_embeddings = nn.Embedding(num_items, embedding_dim)
    optimizer = torch.optim.Adam(
        list(user_embeddings.parameters()) + list(item_embeddings.parameters()),
        lr=learning_rate,
    )
    frequencies = torch.zeros(num_items, dtype=torch.float32)
    for item_id, count in item_counts.items():
        frequencies[item_id] = count
    negative_distribution = frequencies.pow(0.75)
    negative_distribution /= negative_distribution.sum()
    pair_tensor = torch.tensor(positives, dtype=torch.long)

    for _ in range(epochs):
        order = torch.randperm(pair_tensor.size(0))
        for batch_indices in order.split(batch_size):
            batch = pair_tensor[batch_indices]
            users = user_embeddings(batch[:, 0])
            positive_items = item_embeddings(batch[:, 1])
            negatives = torch.multinomial(
                negative_distribution,
                batch.size(0) * negative_samples,
                replacement=True,
            ).view(batch.size(0), negative_samples)
            negative_items = item_embeddings(negatives)
            positive_scores = (users * positive_items).sum(dim=1, keepdim=True)
            negative_scores = torch.einsum("bd,bnd->bn", users, negative_items)
            loss = -torch.nn.functional.logsigmoid(
                positive_scores - negative_scores
            ).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return item_embeddings.weight.detach()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--embedding-output-path", required=True)
    parser.add_argument("--budget-output-path", required=True)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--negative-samples", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--max-pairs", type=int, default=5_000_000)
    parser.add_argument("--min-budget", type=int, default=1)
    parser.add_argument("--max-budget", type=int, default=10)
    parser.add_argument("--budget-direction", choices=["direct", "inverse"], default="direct")
    args = parser.parse_args()

    from src.train_item2vec_budgets import scores_to_budgets

    embeddings = train_bpr(
        interactions=load_interactions(args.data_dir),
        embedding_dim=args.embedding_dim,
        negative_samples=args.negative_samples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_pairs=args.max_pairs,
    )
    budgets = scores_to_budgets(
        torch.linalg.vector_norm(embeddings, dim=1),
        min_budget=args.min_budget,
        max_budget=args.max_budget,
        direction=args.budget_direction,
    )
    os.makedirs(os.path.dirname(args.embedding_output_path), exist_ok=True)
    torch.save(embeddings, args.embedding_output_path)
    torch.save(budgets, args.budget_output_path)


if __name__ == "__main__":
    main()
