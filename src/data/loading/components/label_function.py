from abc import ABC, abstractmethod
from typing import Optional

import torch

from src.data.loading.components.interfaces import (
    LabelFunctionOutput,
)


class LabelFunction(ABC):
    """
    An interface for the LabelFunction classes. The LabelFunction classes are used to transform the input sequence for training or inference and collecting the labels and label prediction locations.
    """

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def transform_label(self, sequence: torch.Tensor) -> LabelFunctionOutput:
        """
        Function to transform the input sequence for training or inference and collecting the labels and label prediction locations.
        The exact functionality needs to be implemented in the child class.

        Parameters
        ----------
        sequence: torch.Tensor of size (batch_size, sequence_length)
            The matrix for input sequences.

        Returns
        ----------
        A LabelFunctionOutput object with the following attributes:
            sequence: torch.Tensor of size (batch_size, new_sequence_length)
                The original input sequence transformed for training or inference. The new_sequence_length can be the same as the original sequence length or different depending on the LabelFunction.
            labels: torch.Tensor of size (number of labels,)
                The labels across the sequences in the batch stacked together as labels for training.
            label_location: torch.Tensor of size (number of masked tokens, 2)
                The (row, col) location of the label_prediction corresponding to the labels in the batch stacked together.
        """
        raise NotImplementedError("Need to implement in the child class.")

    def get_input_attention_mask(
        self, sequence: torch.Tensor, padding_token: int
    ) -> torch.Tensor:
        """
        Function to get the input attention mask for the input sequence.
        Defaults to returning the mask for the non-padding tokens but
        can be overridden in the child class to return a different mask.


        Parameters
        ----------
        sequence: torch.Tensor of size (batch_size, sequence_length)
            The matrix for input sequences.
        padding_token: int
            The index of token for padding.

        Returns
        ----------
        A tensor of size (batch_size, sequence_length) with 1s for non-padding tokens and 0s for padding tokens.
        """
        return sequence != padding_token

class Identity(LabelFunction):
    """
    LabelFunction class to return the original input for the non-masked values of the sequence. It's useful in situations where we don't to transform the input sequence.
    """

    def transform_label(
        self, sequence: torch.Tensor, padding_token: int = 0, masking_token: int = 0
    ) -> LabelFunctionOutput:
        """
        Returns the original input for the non-masked values of the sequence.

        Parameters
        ----------
        sequence: torch.Tensor of size (batch_size, sequence_length)
            The matrix for input sequences.
        padding_token: int
            The index of token for padding.
            The padding_token is defined in the DataloaderConfig of LightningDataModule and is passed to the collate_fn which then passes it to the LabelFunction's transform_label function.
            Thus, to change the padding_token, the user needs to change the padding_token in the DataloaderConfig.

        Returns
        ----------
        A LabelFunctionOutput object with the following attributes:
            sequence: torch.Tensor of size (batch_size, sequence_length)
                The original input sequence.
            labels: torch.Tensor of size (number of non-padding tokens in batch,)
                Collecting all the non-padding tokens in the batch stacked together as labels for training.
            label_location: torch.Tensor of size (number of non-padding tokens in batch, 2)
                The location of the label predictions, that is the (row, col) location of all non-padding tokens in the batch stacked together.
        """
        content_mask = sequence != padding_token
        labels = sequence[content_mask]
        label_location = content_mask.nonzero()

        return LabelFunctionOutput(
            sequence=sequence, labels=labels, label_location=label_location
        )


class NextKTokenMasking(LabelFunction):
    """
    LabelFunction to create masking to use the last K tokens at the end of each sequence as labels.
    Implements the `All Action` training objective from Pinnerformer (https://arxiv.org/pdf/2205.04507)
    """

    def __init__(self, next_k: int = 5):
        """
        Initialize the LabelFunction with the number of tokens to mask as labels.

        Parameters
        ----------
        next_k: int
            Number of tokens K to mask as labels.
        """
        self.next_k = next_k

    def transform_label(
        self, sequence: torch.Tensor, padding_token: int, masking_token: int
    ) -> LabelFunctionOutput:
        """
        For each row of sequence, we save the original last next_k tokens as labels and replace them with 1 masking token and next_k - 1 padding tokens.
        For all the next_k tokens, we use the first masked token as the label prediction.

        Parameters
        ----------
        sequence: torch.Tensor of size (batch_size, sequence_length)
            The matrix for input sequences.
        padding_token: int
            The index of token for padding.
        masking_token: int
            The index of token for masked tokens.

        The padding_token and the masking_token are defined in the DataloaderConfig of LightningDataModule and are passed to the collate_fn which then passes it to the LabelFunction's transform_label function.
        Thus, to change the padding_token and the masking_token, the user needs to change them in the DataloaderConfig.

        Returns
        ----------
        A LabelFunctionOutput object with the following attributes:
            sequence: torch.Tensor of size (batch_size, sequence_length),
                The original input sequence transformed such that for each row, the last next_k non-padding tokens are replaced with 1 masking token and next_k - 1 padding tokens.
            labels: torch.Tensor of size (batch_size * self.next_k,)
                The original last next_k tokens of each row chosen as labels stacked together.
            label_location: torch.Tensor of size (batch_size * self.next_k, 2)
                The location of the label predictions, for each row the first masked token's location is returned as the label prediction location for all the next_k tokens.
        """
        content_mask = sequence != padding_token

        # check each sequence in batch is greater than next_k + 1
        unpadded_seq_lengths = content_mask.sum(1)  # shape: (batch_size,)
        if torch.any(unpadded_seq_lengths < self.next_k + 1):
            raise ValueError(
                f"Sequence lengths: {unpadded_seq_lengths[unpadded_seq_lengths < self.next_k + 1]} should be greater than next_k + 1: {self.next_k + 1}"
            )

        # for each row, we want elements from unpadded_seq_lengths - next_k to unpadded_seq_lengths - 1 as labels
        label_start_indices = unpadded_seq_lengths - self.next_k  # shape: (batch_size,)

        # for each row, we select [label_start_indices, label_start_indices + 1, ..., label_start_indices + next_k - 1]
        label_col_offset = torch.arange(self.next_k)  # shape: (next_k,)
        label_col_indices = (
            label_start_indices.unsqueeze(1) + label_col_offset
        )  # shape: (batch_size, next_k)
        label_col_indices = label_col_indices.reshape(
            -1
        )  # shape: (batch_size * next_k,)

        # To get the row indices, we repeat each row index next_k times
        row_orig_indices = torch.arange(sequence.size(0))  # shape: (batch_size,)
        row_interleaved_indices = row_orig_indices.repeat_interleave(
            self.next_k
        )  # shape: (batch_size * next_k,)

        labels = sequence[row_interleaved_indices, label_col_indices]

        # for prediction, for each row, we replace the last next_k tokens with
        # 1 masking token at the label_start_indices, and next_k - 1 with padding tokens
        sequence[row_interleaved_indices, label_col_indices] = padding_token
        sequence[row_orig_indices, label_start_indices] = masking_token

        # for each row, we use the label_start_index (which was masked) as label prediction for all the next_k labels
        label_location = torch.stack(
            (
                row_interleaved_indices,
                label_start_indices.repeat_interleave(self.next_k),
            ),
            dim=1,
        )

        return LabelFunctionOutput(
            sequence=sequence,
            labels=labels,
            label_location=label_location,
        )


class NextItemTokenMasking(NextKTokenMasking):
    """Mask the final item, using separators for variable-length IDs."""

    def __init__(
        self,
        next_k: int,
        separator_token: Optional[int] = None,
        semantic_id_mode: str = "fixed",
    ):
        super().__init__(next_k=next_k)
        self.separator_token = separator_token
        self.semantic_id_mode = semantic_id_mode

    def transform_label(
        self, sequence: torch.Tensor, padding_token: int, masking_token: int
    ) -> LabelFunctionOutput:
        if self.separator_token is None:
            return super().transform_label(
                sequence=sequence,
                padding_token=padding_token,
                masking_token=masking_token,
            )

        if self.semantic_id_mode == "fixed":
            targets = []
            target_locations = []
            transformed = sequence.clone()
            for row_index, row in enumerate(sequence):
                separator_positions = torch.where(row == self.separator_token)[0]
                if separator_positions.numel() == 0:
                    raise ValueError("Semantic-ID sequences require separators.")
                final_separator = int(separator_positions[-1])
                start = final_separator - self.next_k
                target = row[start:final_separator]
                targets.append(target)
                target_locations.append((row_index, start))
                transformed[row_index, start] = masking_token
                if final_separator > start + 1:
                    transformed[row_index, start + 1 : final_separator] = padding_token
            labels = torch.stack(targets)
            return LabelFunctionOutput(
                sequence=transformed,
                labels=labels,
                label_location=torch.tensor(target_locations, device=sequence.device),
            )

        targets = []
        target_locations = []
        transformed = sequence.clone()
        for row_index, row in enumerate(sequence):
            separator_positions = torch.where(row == self.separator_token)[0]
            if separator_positions.numel() == 0:
                raise ValueError("Variable-length semantic-ID sequences require separators.")
            final_separator = int(separator_positions[-1])
            previous_separator = (
                int(separator_positions[-2]) if separator_positions.numel() > 1 else -1
            )
            start = previous_separator + 1
            target = row[start : final_separator + 1]
            targets.append(target)
            target_locations.append((row_index, start))
            transformed[row_index, start] = masking_token
            if final_separator > start:
                transformed[row_index, start + 1 : final_separator + 1] = padding_token

        max_target_length = max(target.numel() for target in targets)
        labels = torch.full(
            (len(targets), max_target_length), padding_token, dtype=sequence.dtype
        )
        for row_index, target in enumerate(targets):
            labels[row_index, : target.numel()] = target

        return LabelFunctionOutput(
            sequence=transformed,
            labels=labels,
            label_location=torch.tensor(target_locations, device=sequence.device),
        )
