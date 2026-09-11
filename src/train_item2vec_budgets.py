"""Train item2vec embeddings and derive per-item semantic-ID budgets."""

import argparse
import os
import random
from collections import Counter
from typing import List

import tensorflow as tf
import torch
from torch import nn

from src.data.loading.components.iterators import TFRecordIterator
from src.utils.file_utils import list_files


def load_training_sequences(data_dir: str) -> List[List[int]]:
    files = list_files(
        folder_path=os.path.join(data_dir, "training"), suffix="*tfrecord.gz"
    )
    if not files:
        raise FileNotFoundError(f"No training TFRecords found under {data_dir}")

    iterator = TFRecordIterator()
    iterator.update_list_of_file_paths(files)
    iterator.should_drop_last_batch = False
    sequences = []
    for batch in iterator.iter_batches(batch_size=1024):
        sequence = batch["sequence_data"]
        if isinstance(sequence, tf.SparseTensor):
            sequence = tf.sparse.to_dense(sequence)
        for raw_sequence in sequence.numpy():
            item_ids = [int(item_id) for item_id in raw_sequence.reshape(-1)]
            if item_ids:
                sequences.append(item_ids)
    if not sequences:
        raise ValueError("Training records contained no item sequences.")
    return sequences


def train_item2vec(
    sequences: List[List[int]],
    embedding_dim: int = 64,
    window_size: int = 5,
    negative_samples: int = 5,
    epochs: int = 3,
    batch_size: int = 4096,
    max_pairs: int = 5_000_000,
    learning_rate: float = 0.01,
    seed: int = 42,
) -> torch.Tensor:
    if max_pairs < 1:
        raise ValueError("max_pairs must be positive.")
    random.seed(seed)
    torch.manual_seed(seed)
    max_item_id = max(item_id for sequence in sequences for item_id in sequence)
    num_items = max_item_id + 1

    pairs = []
    pairs_seen = 0
    counts = Counter(item_id for sequence in sequences for item_id in sequence)
    for sequence in sequences:
        for center_index, center_item in enumerate(sequence):
            start = max(0, center_index - window_size)
            end = min(len(sequence), center_index + window_size + 1)
            for context_index in range(start, end):
                if context_index != center_index:
                    pair = (center_item, sequence[context_index])
                    pairs_seen += 1
                    if len(pairs) < max_pairs:
                        pairs.append(pair)
                    else:
                        replacement_index = random.randrange(pairs_seen)
                        if replacement_index < max_pairs:
                            pairs[replacement_index] = pair
    if not pairs:
        raise ValueError("Training sequences did not produce item2vec pairs.")

    center_embeddings = nn.Embedding(num_items, embedding_dim)
    context_embeddings = nn.Embedding(num_items, embedding_dim)
    optimizer = torch.optim.Adam(
        list(center_embeddings.parameters()) + list(context_embeddings.parameters()),
        lr=learning_rate,
    )
    frequencies = torch.zeros(num_items, dtype=torch.float32)
    for item_id, count in counts.items():
        frequencies[item_id] = count
    negative_distribution = frequencies.pow(0.75)
    negative_distribution /= negative_distribution.sum()

    pair_tensor = torch.tensor(pairs, dtype=torch.long)
    for _ in range(epochs):
        order = torch.randperm(pair_tensor.size(0))
        for batch_indices in order.split(batch_size):
            batch = pair_tensor[batch_indices]
            centers = center_embeddings(batch[:, 0])
            contexts = context_embeddings(batch[:, 1])
            positive_logits = (centers * contexts).sum(dim=1)

            negatives = torch.multinomial(
                negative_distribution,
                batch.size(0) * negative_samples,
                replacement=True,
            ).view(batch.size(0), negative_samples)
            negative_vectors = context_embeddings(negatives)
            negative_logits = torch.einsum("bd,bnd->bn", centers, negative_vectors)

            loss = -torch.nn.functional.logsigmoid(positive_logits).mean()
            loss -= torch.nn.functional.logsigmoid(-negative_logits).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return (center_embeddings.weight.detach() + context_embeddings.weight.detach()) / 2


def scores_to_budgets(
    scores: torch.Tensor,
    min_budget: int = 1,
    max_budget: int = 10,
    direction: str = "direct",
) -> torch.Tensor:
    if min_budget < 1 or max_budget < min_budget:
        raise ValueError("Budget bounds must satisfy 1 <= min_budget <= max_budget.")
    if direction not in {"direct", "inverse"}:
        raise ValueError("direction must be 'direct' or 'inverse'.")
    positive_scores = scores[scores > 0]
    if positive_scores.numel() == 0:
        normalized_scores = torch.zeros_like(scores)
    elif positive_scores.max() == positive_scores.min():
        normalized_scores = torch.ones_like(scores)
    else:
        normalized_scores = (scores - positive_scores.min()) / (
            positive_scores.max() - positive_scores.min()
        )
        normalized_scores[scores <= 0] = 0
    if direction == "inverse":
        positive_mask = scores > 0
        normalized_scores[positive_mask] = 1 - normalized_scores[positive_mask]
    return torch.round(
        min_budget + normalized_scores * (max_budget - min_budget)
    ).long()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--embedding-output-path", required=True)
    parser.add_argument("--budget-output-path", required=True)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--window-size", type=int, default=5)
    parser.add_argument("--negative-samples", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--max-pairs", type=int, default=5_000_000)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--min-budget", type=int, default=1)
    parser.add_argument("--max-budget", type=int, default=10)
    parser.add_argument("--budget-direction", choices=["direct", "inverse"], default="direct")
    args = parser.parse_args()

    sequences = load_training_sequences(args.data_dir)
    embeddings = train_item2vec(
        sequences=sequences,
        embedding_dim=args.embedding_dim,
        window_size=args.window_size,
        negative_samples=args.negative_samples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_pairs=args.max_pairs,
        learning_rate=args.learning_rate,
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
