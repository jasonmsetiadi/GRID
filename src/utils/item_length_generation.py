"""Generate exact per-item SID lengths from training interactions."""

import argparse
import math
import os
from collections import Counter, defaultdict
from itertools import combinations
from typing import Dict, Iterable, List, Tuple

import tensorflow as tf
import torch

from src.data.loading.components.iterators import TFRecordIterator
from src.utils.file_utils import list_files

METHODS = {
    "interaction_count",
    "cooccurrence",
    "ppmi",
    "neighborhood_entropy",
    "graph_centrality",
    "item2vec",
    "bpr",
    "lightgcn",
}


def load_sequences(data_dir: str) -> List[List[int]]:
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
        for row in sequence.numpy():
            values = [int(value) for value in row.reshape(-1)]
            if values:
                sequences.append(values)
    if not sequences:
        raise ValueError("Training records contained no item sequences")
    return sequences


def _graph_scores(sequences: Iterable[List[int]]) -> Dict[int, float]:
    scores = Counter()
    for sequence in sequences:
        items = set(sequence)
        for item in items:
            scores[item] += max(len(items) - 1, 0)
    return dict(scores)


def _pair_statistics(sequences: Iterable[List[int]]):
    item_counts = Counter()
    pair_counts = Counter()
    sequence_count = 0
    for sequence in sequences:
        items = set(sequence)
        item_counts.update(items)
        pair_counts.update(combinations(sorted(items), 2))
        sequence_count += 1
    return item_counts, pair_counts, sequence_count


def _deterministic_scores(sequences: List[List[int]], method: str) -> Dict[int, float]:
    if method == "interaction_count":
        return dict(Counter(item for sequence in sequences for item in sequence))
    if method == "cooccurrence":
        return _graph_scores(sequences)

    item_counts, pair_counts, sequence_count = _pair_statistics(sequences)
    if method == "graph_centrality":
        scores = Counter()
        for (first, second), count in pair_counts.items():
            scores[first] += count
            scores[second] += count
        return dict(scores)
    if method == "neighborhood_entropy":
        neighbors = defaultdict(Counter)
        for (first, second), count in pair_counts.items():
            neighbors[first][second] = count
            neighbors[second][first] = count
        scores = {}
        for item, item_neighbors in neighbors.items():
            total = sum(item_neighbors.values())
            scores[item] = -sum(
                (count / total) * math.log(count / total)
                for count in item_neighbors.values()
            )
        return scores
    if method == "ppmi":
        scores = Counter({item: 0.0 for item in item_counts})
        for (first, second), count in pair_counts.items():
            pmi = math.log(
                count * sequence_count
                / (item_counts[first] * item_counts[second])
            )
            if pmi > 0:
                scores[first] += pmi
                scores[second] += pmi
        return dict(scores)
    raise ValueError(f"Unsupported deterministic item-length method: {method}")


def _learned_scores(sequences: List[List[int]], method: str) -> Dict[int, float]:
    """Train a small collaborative model and score items by embedding norm."""
    if method == "item2vec":
        pairs = []
        for sequence in sequences:
            for center_index, center in enumerate(sequence):
                start = max(0, center_index - 5)
                end = min(len(sequence), center_index + 6)
                pairs.extend(
                    (center, sequence[index])
                    for index in range(start, end)
                    if index != center_index
                )
        return _train_item2vec(pairs, max(max(sequence) for sequence in sequences) + 1)

    interactions = _load_interactions_from_sequences(sequences)
    if method == "bpr":
        return _train_bpr(interactions)
    if method == "lightgcn":
        return _train_lightgcn(interactions)
    raise ValueError(f"Unsupported learned item-length method: {method}")


def _load_interactions_from_sequences(
    sequences: List[List[int]],
) -> List[Tuple[int, List[int]]]:
    return [(index, list(set(sequence))) for index, sequence in enumerate(sequences)]


def _train_item2vec(pairs: List[Tuple[int, int]], num_items: int) -> Dict[int, float]:
    if not pairs:
        raise ValueError("Training sequences did not produce item2vec pairs")
    pairs = pairs[:5_000_000]
    pair_tensor = torch.tensor(pairs, dtype=torch.long)
    dimension = 32
    center = torch.nn.Embedding(num_items, dimension)
    context = torch.nn.Embedding(num_items, dimension)
    optimizer = torch.optim.Adam(list(center.parameters()) + list(context.parameters()), lr=0.01)
    frequencies = torch.bincount(pair_tensor[:, 1], minlength=num_items).float().pow(0.75)
    frequencies = frequencies / frequencies.sum().clamp_min(1)
    for _ in range(2):
        for indices in torch.randperm(len(pair_tensor)).split(4096):
            batch = pair_tensor[indices]
            vectors = center(batch[:, 0])
            positive = context(batch[:, 1])
            negative_ids = torch.multinomial(
                frequencies, batch.size(0) * 3, replacement=True
            ).view(batch.size(0), 3)
            negative = context(negative_ids)
            positive_score = (vectors * positive).sum(dim=1)
            negative_score = torch.einsum("bd,bnd->bn", vectors, negative)
            loss = -torch.nn.functional.logsigmoid(positive_score).mean()
            loss -= torch.nn.functional.logsigmoid(-negative_score).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    values = torch.linalg.vector_norm(center.weight + context.weight, dim=1)
    return {index: float(value) for index, value in enumerate(values) if value > 0}


def _train_bpr(interactions: List[Tuple[int, List[int]]]) -> Dict[int, float]:
    user_items = {user: set(items) for user, items in interactions}
    num_items = max(item for items in user_items.values() for item in items) + 1
    positives = [
        (user, item) for user, items in user_items.items() for item in items
    ]
    user_index = {user: index for index, user in enumerate(user_items)}
    users = torch.nn.Embedding(len(user_index), 32)
    items = torch.nn.Embedding(num_items, 32)
    optimizer = torch.optim.Adam(list(users.parameters()) + list(items.parameters()), lr=0.01)
    pair_tensor = torch.tensor(
        [(user_index[user], item) for user, item in positives], dtype=torch.long
    )
    frequencies = torch.bincount(pair_tensor[:, 1], minlength=num_items).float().pow(0.75)
    frequencies = frequencies / frequencies.sum().clamp_min(1)
    for _ in range(2):
        for indices in torch.randperm(len(pair_tensor)).split(4096):
            batch = pair_tensor[indices]
            user_vectors = users(batch[:, 0])
            positive_vectors = items(batch[:, 1])
            negative_ids = torch.multinomial(frequencies, batch.size(0), replacement=True)
            negative_vectors = items(negative_ids)
            difference = (user_vectors * (positive_vectors - negative_vectors)).sum(dim=1)
            loss = -torch.nn.functional.logsigmoid(difference).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    values = torch.linalg.vector_norm(items.weight, dim=1)
    return {index: float(value) for index, value in enumerate(values) if value > 0}


def _train_lightgcn(interactions: List[Tuple[int, List[int]]]) -> Dict[int, float]:
    user_index = {user: index for index, (user, _) in enumerate(interactions)}
    positives = [
        (user_index[user], item)
        for user, items in interactions
        for item in set(items)
    ]
    num_users = len(user_index)
    num_items = max(item for _, item in positives) + 1
    num_nodes = num_users + num_items
    initial = torch.nn.Embedding(num_nodes, 32)
    optimizer = torch.optim.Adam(initial.parameters(), lr=0.01)

    user_nodes = torch.tensor([user for user, _ in positives], dtype=torch.long)
    item_nodes = torch.tensor(
        [num_users + item for _, item in positives], dtype=torch.long
    )
    source = torch.cat([user_nodes, item_nodes])
    target = torch.cat([item_nodes, user_nodes])
    degree = torch.bincount(source, minlength=num_nodes).float().clamp_min(1)
    edge_weight = (degree[source] * degree[target]).rsqrt()
    pair_tensor = torch.tensor(positives, dtype=torch.long)
    item_frequency = torch.bincount(
        pair_tensor[:, 1], minlength=num_items
    ).float().pow(0.75)
    item_frequency /= item_frequency.sum().clamp_min(1)

    def propagate():
        embeddings = initial.weight
        propagated = [embeddings]
        for _ in range(2):
            next_embeddings = torch.zeros_like(embeddings)
            next_embeddings.index_add_(
                0, target, embeddings[source] * edge_weight.unsqueeze(1)
            )
            embeddings = next_embeddings
            propagated.append(embeddings)
        return torch.stack(propagated).mean(dim=0)

    for _ in range(2):
        for indices in torch.randperm(len(pair_tensor)).split(4096):
            batch = pair_tensor[indices]
            embeddings = propagate()
            user_vectors = embeddings[batch[:, 0]]
            positive_vectors = embeddings[num_users + batch[:, 1]]
            negative_ids = torch.multinomial(
                item_frequency, batch.size(0), replacement=True
            )
            negative_vectors = embeddings[num_users + negative_ids]
            difference = (user_vectors * (positive_vectors - negative_vectors)).sum(dim=1)
            loss = -torch.nn.functional.logsigmoid(difference).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    values = torch.linalg.vector_norm(propagate()[num_users:], dim=1)
    return {index: float(value) for index, value in enumerate(values) if value > 0}


def scores_to_lengths(
    scores: Dict[int, float],
    min_length: int,
    max_length: int,
    direction: str = "direct",
) -> torch.Tensor:
    if min_length < 1 or max_length < min_length:
        raise ValueError("Length bounds must satisfy 1 <= min_length <= max_length")
    if direction not in {"direct", "inverse"}:
        raise ValueError("direction must be 'direct' or 'inverse'")
    if not scores:
        raise ValueError("Cannot generate lengths without item scores")

    output = torch.zeros(max(scores) + 1, dtype=torch.long)
    values = torch.tensor([float(score) for score in scores.values()])
    positive = values > 0
    if positive.any() and values[positive].max() != values[positive].min():
        normalized = (values - values[positive].min()) / (
            values[positive].max() - values[positive].min()
        )
        normalized[~positive] = 0
    elif positive.any():
        normalized = torch.where(positive, torch.ones_like(values), torch.zeros_like(values))
    else:
        normalized = torch.zeros_like(values)
    if direction == "inverse":
        normalized[positive] = 1 - normalized[positive]

    lengths = torch.round(min_length + normalized * (max_length - min_length)).long()
    for index, item in enumerate(scores):
        output[item] = lengths[index]
    output[output == 0] = min_length
    return output


def generate_item_lengths(
    data_dir: str,
    method: str,
    min_length: int,
    max_length: int,
    direction: str = "direct",
) -> torch.Tensor:
    if method not in METHODS:
        raise ValueError(f"Unknown per-item length method: {method}")
    sequences = load_sequences(data_dir)
    if method in {"item2vec", "bpr", "lightgcn"}:
        scores = _learned_scores(sequences, method)
    else:
        scores = _deterministic_scores(sequences, method)
    return scores_to_lengths(scores, min_length, max_length, direction)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--method", choices=sorted(METHODS), required=True)
    parser.add_argument("--min-length", type=int, default=1)
    parser.add_argument("--max-length", type=int, required=True)
    parser.add_argument("--direction", choices=["direct", "inverse"], default="direct")
    parser.add_argument("--output-path", required=True)
    args = parser.parse_args()
    lengths = generate_item_lengths(
        data_dir=args.data_dir,
        method=args.method,
        min_length=args.min_length,
        max_length=args.max_length,
        direction=args.direction,
    )
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    torch.save(lengths, args.output_path)


if __name__ == "__main__":
    main()
