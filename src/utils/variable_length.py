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


def select_lengths(
    mode: str,
    method: str,
    residuals: Optional[torch.Tensor] = None,
    item_ids: Optional[torch.Tensor] = None,
    item_lengths: Optional[torch.Tensor] = None,
    threshold: Optional[float] = None,
    min_length: int = 1,
    max_length: Optional[int] = None,
) -> torch.Tensor:
    """Select exact SID lengths using a content or per-item strategy."""
    if mode == "content_based":
        if method != "residual_threshold":
            raise ValueError(
                "Unsupported content-based length method: "
                f"{method!r}. Supported method: 'residual_threshold'."
            )
        if residuals is None or threshold is None:
            raise ValueError(
                "content_based/residual_threshold requires residuals and threshold"
            )
        return select_lengths_from_residuals(
            residuals=residuals,
            threshold=threshold,
            min_length=min_length,
            max_length=max_length,
        )

    if mode == "per_item":
        if item_ids is None or item_lengths is None:
            raise ValueError(
                f"per_item/{method} requires item_ids and generated item lengths"
            )
        if item_lengths.ndim != 1:
            raise ValueError(
                "Expected a 1D per-item length tensor, "
                f"got shape {tuple(item_lengths.shape)}"
            )

        item_ids = torch.as_tensor(
            item_ids, dtype=torch.long, device=item_lengths.device
        )
        if item_ids.numel() == 0:
            return item_ids.clone()
        if item_ids.min() < 0 or item_ids.max() >= item_lengths.numel():
            raise IndexError(
                "Per-item length tensor does not contain every requested item_id"
            )

        lengths = item_lengths[item_ids].long()
        effective_max = item_lengths.numel() if max_length is None else max_length
        if torch.any(lengths < min_length) or torch.any(lengths > effective_max):
            raise ValueError(
                "Per-item lengths must be within the configured "
                f"range [{min_length}, {effective_max}]"
            )
        return lengths.to(device=item_ids.device)

    raise ValueError(
        f"Unsupported length selection mode: {mode!r}. "
        "Supported modes: 'content_based' and 'per_item'."
    )
