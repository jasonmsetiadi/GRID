from typing import Optional

import torch


def select_lengths_from_residuals(
    residuals: torch.Tensor,
    threshold: float,
    min_length: int = 1,
    max_length: Optional[int] = None,
) -> torch.Tensor:
    """Select a SID prefix length from residual vectors."""
    if residuals.ndim != 3:
        raise ValueError(
            "Expected residuals with shape [batch, features, layers], "
            f"got {tuple(residuals.shape)}"
        )
    num_layers = residuals.size(-1)
    if num_layers == 0:
        raise ValueError("At least one residual layer is required")
    if min_length < 1 or min_length > num_layers:
        raise ValueError("min_length must be between 1 and the number of layers")

    max_length = num_layers if max_length is None else min(max_length, num_layers)
    if max_length < min_length:
        raise ValueError("max_length must be >= min_length")

    residual_norms = torch.linalg.vector_norm(residuals, dim=1)
    eligible = residual_norms[:, min_length - 1 : max_length] <= threshold
    found = eligible.any(dim=1)
    first = eligible.to(torch.int64).argmax(dim=1) + min_length
    return torch.where(found, first, torch.full_like(first, max_length))
