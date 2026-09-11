"""Train a lightweight LightGCN/BPR model and derive item budgets."""

import argparse
import os

import torch
from torch import nn

from src.train_bpr_budgets import load_interactions
from src.train_item2vec_budgets import scores_to_budgets


def train_lightgcn(
    interactions,
    embedding_dim: int = 64,
    layers: int = 2,
    negative_samples: int = 1,
    epochs: int = 3,
    batch_size: int = 4096,
    learning_rate: float = 0.01,
    max_pairs: int = 5_000_000,
    seed: int = 42,
) -> torch.Tensor:
    if layers < 1 or max_pairs < 1:
        raise ValueError("layers and max_pairs must be positive.")
    torch.manual_seed(seed)

    user_to_index = {}
    positives = []
    max_item_id = 0
    for user_id, item_ids in interactions:
        user_index = user_to_index.setdefault(user_id, len(user_to_index))
        for item_id in item_ids:
            positives.append((user_index, item_id))
            max_item_id = max(max_item_id, item_id)
    if len(positives) > max_pairs:
        generator = torch.Generator().manual_seed(seed)
        positives = [
            positives[index]
            for index in torch.randperm(len(positives), generator=generator)[:max_pairs]
        ]

    num_users = len(user_to_index)
    num_items = max_item_id + 1
    num_nodes = num_users + num_items
    initial_embeddings = nn.Embedding(num_nodes, embedding_dim)
    optimizer = torch.optim.Adam(initial_embeddings.parameters(), lr=learning_rate)

    edge_users = torch.tensor([user for user, _ in positives], dtype=torch.long)
    edge_items = torch.tensor(
        [num_users + item for _, item in positives], dtype=torch.long
    )
    edge_src = torch.cat([edge_users, edge_items])
    edge_dst = torch.cat([edge_items, edge_users])
    degrees = torch.bincount(edge_src, minlength=num_nodes).float().clamp_min(1)
    edge_weights = (degrees[edge_src] * degrees[edge_dst]).rsqrt()

    item_counts = torch.bincount(edge_items - num_users, minlength=num_items).float()
    negative_distribution = item_counts.pow(0.75)
    negative_distribution /= negative_distribution.sum()
    pair_tensor = torch.tensor(positives, dtype=torch.long)

    def propagate():
        embeddings = initial_embeddings.weight
        propagated = [embeddings]
        for _ in range(layers):
            next_embeddings = torch.zeros_like(embeddings)
            next_embeddings.index_add_(
                0, edge_dst, embeddings[edge_src] * edge_weights.unsqueeze(1)
            )
            embeddings = next_embeddings
            propagated.append(embeddings)
        return torch.stack(propagated, dim=0).mean(dim=0)

    for _ in range(epochs):
        order = torch.randperm(pair_tensor.size(0))
        for batch_indices in order.split(batch_size):
            batch = pair_tensor[batch_indices]
            all_embeddings = propagate()
            user_embeddings = all_embeddings[batch[:, 0]]
            positive_embeddings = all_embeddings[num_users + batch[:, 1]]
            negatives = torch.multinomial(
                negative_distribution,
                batch.size(0) * negative_samples,
                replacement=True,
            ).view(batch.size(0), negative_samples)
            negative_embeddings = all_embeddings[num_users + negatives]
            positive_scores = (user_embeddings * positive_embeddings).sum(dim=1, keepdim=True)
            negative_scores = torch.einsum(
                "bd,bnd->bn", user_embeddings, negative_embeddings
            )
            loss = -torch.nn.functional.logsigmoid(
                positive_scores - negative_scores
            ).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return propagate()[num_users:].detach()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--embedding-output-path", required=True)
    parser.add_argument("--budget-output-path", required=True)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--negative-samples", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--max-pairs", type=int, default=5_000_000)
    parser.add_argument("--min-budget", type=int, default=1)
    parser.add_argument("--max-budget", type=int, default=10)
    parser.add_argument("--budget-direction", choices=["direct", "inverse"], default="direct")
    args = parser.parse_args()

    embeddings = train_lightgcn(
        interactions=load_interactions(args.data_dir),
        embedding_dim=args.embedding_dim,
        layers=args.layers,
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
