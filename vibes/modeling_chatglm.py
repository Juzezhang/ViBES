""" PyTorch ChatGLM model. """

import math
import sys
import torch
import torch.utils.checkpoint
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss, LayerNorm, MSELoss, BCEWithLogitsLoss
from torch.nn.utils import skip_init
from typing import Optional, Tuple, Union, List, Dict, Any

from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    SequenceClassifierOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging, is_torch_npu_available
from transformers.generation.logits_process import LogitsProcessor
from transformers.generation.utils import ModelOutput
from transformers import PreTrainedModel, GenerationMixin

from .configuration_chatglm import ChatGLMConfig
import copy
try:
    from transformers.utils import is_flash_attn_greater_or_equal_2_10, is_flash_attn_2_available

    if is_flash_attn_2_available():
        from flash_attn import flash_attn_func, flash_attn_varlen_func
        from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa
except:
    pass

# flags required to enable jit fusion kernels

if sys.platform != 'darwin' and not is_torch_npu_available():
    torch._C._jit_set_profiling_mode(False)
    torch._C._jit_set_profiling_executor(False)
    torch._C._jit_override_can_fuse_on_cpu(True)
    torch._C._jit_override_can_fuse_on_gpu(True)

logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "THUDM/ChatGLM"
_CONFIG_FOR_DOC = "ChatGLMConfig"

class ModalityBatchProcessor:
    """
    Corrected version: Keep batch dimension, only optimize within batch
    """
    def __init__(self, n_modalities: int = 2, n_experts: int = 2): ## TODO: to modify to exact number of modalities and experts
        self.n_modalities = n_modalities
        self.n_experts = n_experts


    def _get_position_mapping(self, modality_mask: torch.Tensor):
        """
        Get index mapping for True positions in modality mask

        Args:
            modality_mask: [batch_size, seq_length] boolean tensor

        Returns:
            position_mapping: List of tensors containing True position indices for each batch
        """
        batch_size, seq_length = modality_mask.shape
        position_mappings = []

        for batch_idx in range(batch_size):
            # Get position indices where modality is True for current batch
            true_positions = torch.nonzero(modality_mask[batch_idx], as_tuple=False).flatten()
            position_mappings.append(true_positions)

        return position_mappings


    def create_batch_aware_modality_data(self, input_ids: torch.Tensor, modality_masks: torch.Tensor):
        """
        Create batch-aware data structure for each modality

        Args:
            input_ids: [batch_size, seq_length]  
            modality_masks: [n_modalities, batch_size, seq_length]

        Returns:
            Data organized by modality, but maintaining batch dimension
        """
        if len(input_ids.shape) == 2:
            batch_size, seq_length = input_ids.shape
        elif len(input_ids.shape) == 3:
            batch_size, seq_length, hidden_dim = input_ids.shape
        else:
            raise ValueError(f"Input IDs must have 2 or 3 dimensions, but got {len(input_ids.shape)}")

        modality_data = {}

        for modality_idx in range(self.n_modalities):
            # Calculate maximum length for this modality across batches
            max_tokens_in_modality = 0
            batch_tokens = []
            batch_masks = []

            for batch_idx in range(batch_size):
                mask = modality_masks[modality_idx, batch_idx]
                tokens = input_ids[batch_idx][mask]
                max_tokens_in_modality = max(max_tokens_in_modality, len(tokens))
                batch_tokens.append(tokens)

            if max_tokens_in_modality > 0:
                # Perform minimal padding within this modality
                padded_tokens = []
                attention_masks = []

                for tokens in batch_tokens:
                    if len(tokens) < max_tokens_in_modality:
                        # Only pad within modality, not globally
                        pad_length = max_tokens_in_modality - len(tokens)
                        if tokens.dim() == 1:
                            padded = F.pad(tokens, (0, pad_length), value=0)
                        elif tokens.dim() == 2:
                            padded = F.pad(tokens, (0, 0, 0, pad_length), value=0)
                        else:
                            raise ValueError(f"Tokens must have 1 or 2 dimensions, but got {tokens.dim()}")
                        mask = torch.cat([
                            torch.ones(len(tokens), dtype=torch.bool, device=tokens.device),
                            torch.zeros(pad_length, dtype=torch.bool, device=tokens.device)
                        ])
                    else:
                        padded = tokens
                        mask = torch.ones(len(tokens), dtype=torch.bool, device=tokens.device)

                    padded_tokens.append(padded)
                    attention_masks.append(mask)

                modality_data[modality_idx] = {
                    'tokens': torch.stack(padded_tokens),  # [batch_size, max_tokens_in_modality]
                    'attention_mask': torch.stack(attention_masks),  # [batch_size, max_tokens_in_modality]
                    'original_positions': self._get_position_mapping(modality_masks[modality_idx])
                }

        return modality_data


    def create_batch_aware_expert_data_old(self, input_ids: torch.Tensor, modality_masks: torch.Tensor):
        """
        Create batch-aware data structure for each modality

        Args:
            input_ids: [batch_size, seq_length]  
            modality_masks: [n_modalities, batch_size, seq_length]

        Returns:
            Data organized by modality, but maintaining batch dimension
        """
        if len(input_ids.shape) == 2:
            batch_size, seq_length = input_ids.shape
        elif len(input_ids.shape) == 3:
            batch_size, seq_length, hidden_dim = input_ids.shape
        else:
            raise ValueError(f"Input IDs must have 2 or 3 dimensions, but got {len(input_ids.shape)}")

        expert_data = {}

        for expert_idx in range(self.n_experts):
            # Calculate maximum length for this modality across batches
            max_tokens_in_modality = 0
            batch_tokens = []
            batch_masks = []

            for batch_idx in range(batch_size):
                mask = modality_masks[expert_idx, batch_idx]
                tokens = input_ids[batch_idx][mask]
                max_tokens_in_modality = max(max_tokens_in_modality, len(tokens))
                batch_tokens.append(tokens)

            if max_tokens_in_modality > 0:
                # Perform minimal padding within this modality
                padded_tokens = []
                attention_masks = []

                for tokens in batch_tokens:
                    if len(tokens) < max_tokens_in_modality:
                        # Only pad within modality, not globally
                        pad_length = max_tokens_in_modality - len(tokens)
                        if tokens.dim() == 1:
                            padded = F.pad(tokens, (0, pad_length), value=0)
                        elif tokens.dim() == 2:
                            padded = F.pad(tokens, (0, 0, 0, pad_length), value=0)
                        else:
                            raise ValueError(f"Tokens must have 1 or 2 dimensions, but got {tokens.dim()}")
                        mask = torch.cat([
                            torch.ones(len(tokens), dtype=torch.bool, device=tokens.device),
                            torch.zeros(pad_length, dtype=torch.bool, device=tokens.device)
                        ])
                    else:
                        padded = tokens
                        mask = torch.ones(len(tokens), dtype=torch.bool, device=tokens.device)

                    padded_tokens.append(padded)
                    attention_masks.append(mask)

                expert_data[expert_idx] = {
                    'tokens': torch.stack(padded_tokens),  # [batch_size, max_tokens_in_expert]
                    'attention_mask': torch.stack(attention_masks),  # [batch_size, max_tokens_in_expert]
                    'original_positions': self._get_position_mapping(modality_masks[expert_idx])
                }

        return expert_data


    def create_batch_aware_expert_data(self, input_ids: torch.Tensor, modality_masks: torch.Tensor):
        """
        Create batch-aware data structure for each expert (vectorized, identical outputs to *_old)

        Args:
            input_ids: [batch_size, seq_length] or [batch_size, seq_length, hidden_dim]
            modality_masks: [n_experts, batch_size, seq_length] (bool)

        Returns:
            Dict per expert with minimally padded tokens/attention_mask and original positions
        """
        if len(input_ids.shape) == 2:
            batch_size, seq_length = input_ids.shape
            has_hidden = False
            hidden_dim = None
        elif len(input_ids.shape) == 3:
            batch_size, seq_length, hidden_dim = input_ids.shape
            has_hidden = True
        else:
            raise ValueError(f"Input IDs must have 2 or 3 dimensions, but got {len(input_ids.shape)}")

        device = input_ids.device
        expert_data = {}
        arange_seq = torch.arange(seq_length, device=device).unsqueeze(0).expand(batch_size, -1)  # [B, S]

        for expert_idx in range(self.n_experts):
            mask = modality_masks[expert_idx]  # [B, S] bool
            lengths = mask.sum(dim=1)  # [B]
            max_tokens = int(lengths.max().item()) if lengths.numel() > 0 else 0
            if max_tokens == 0:
                continue

            # Stable compaction: move True positions to the front preserving order
            keys = (~mask).to(torch.int64) * (seq_length + 1) + arange_seq  # [B, S]
            sort_idx = torch.argsort(keys, dim=1, stable=True)  # [B, S]
            gather_idx = sort_idx[:, :max_tokens]  # [B, max_tokens]

            if has_hidden:
                expanded_idx = gather_idx.unsqueeze(-1).expand(batch_size, max_tokens, hidden_dim)
                padded_tokens = torch.gather(input_ids, 1, expanded_idx)  # [B, max_tokens, H]
            else:
                padded_tokens = input_ids.gather(1, gather_idx)  # [B, max_tokens]

            attention_mask = (torch.arange(max_tokens, device=device).unsqueeze(0).expand(batch_size, -1)
                             < lengths.unsqueeze(1))  # [B, max_tokens]

            # Zero out padded positions to exactly match *_old behavior
            if has_hidden:
                padded_tokens = padded_tokens.masked_fill(~attention_mask.unsqueeze(-1), 0)
            else:
                padded_tokens = padded_tokens.masked_fill(~attention_mask, 0)

            expert_data[expert_idx] = {
                'tokens': padded_tokens,
                'attention_mask': attention_mask,
                'original_positions': self._get_position_mapping(modality_masks[expert_idx]),
            }

        return expert_data

    def create_batch_aware_modality_inputs_labels(self, input_ids: torch.Tensor, labels: torch.Tensor, modality_masks: torch.Tensor, ignore_modality_index = 0):
        """
        Create batch-aware data structure for each modality (fully vectorized compaction)

        Args:
            input_ids: [batch_size, seq_length]
            modality_masks: [n_modalities, batch_size, seq_length]

        Returns:
            Dict per modality with minimally padded tokens/labels/attention_mask and original positions
        """
        if len(input_ids.shape) == 2:
            batch_size, seq_length = input_ids.shape
        elif len(input_ids.shape) == 3:
            batch_size, seq_length, hidden_dim = input_ids.shape
        else:
            raise ValueError(f"Input IDs must have 2 or 3 dimensions, but got {len(input_ids.shape)}")

        modality_data = {}
        device = input_ids.device
        n_modalities = modality_masks.shape[0]

        # Precompute shared helpers
        arange_seq = torch.arange(seq_length, device=device).unsqueeze(0).expand(batch_size, -1)  # [B, S]

        for modality_idx in range(n_modalities):
            mask = modality_masks[modality_idx]  # [B, S] bool
            lengths = mask.sum(dim=1)  # [B]
            max_tokens_in_modality = int(lengths.max().item())
            if max_tokens_in_modality == 0:
                continue

            # Use stable argsort-based compaction to move True tokens to the front preserving order
            # Keys: True -> 0..S-1, False -> (S+1)+pos to ensure all False after Trues, keeping order
            keys = (~mask).to(torch.int64) * (seq_length + 1) + arange_seq  # [B, S]
            sort_idx = torch.argsort(keys, dim=1, stable=True)  # [B, S], Trues first in original order
            gather_idx = sort_idx[:, :max_tokens_in_modality]

            # Gather tokens (and labels) in one shot
            padded_tokens = input_ids.gather(1, gather_idx)
            attention_masks = (torch.arange(max_tokens_in_modality, device=device)
                               .unsqueeze(0).expand(batch_size, -1)) < lengths.unsqueeze(1)
            # Zero-out padded positions to match old behavior
            padded_tokens = padded_tokens.masked_fill(~attention_masks, 0)
            if labels is not None:
                padded_labels = labels.gather(1, gather_idx)
                padded_labels = padded_labels.masked_fill(~attention_masks, 0)
            else:
                padded_labels = None

            # Apply ignore index consistent with old behavior: any 0 becomes -100
            if labels is not None and ignore_modality_index == modality_idx:
                padded_labels = torch.where(padded_labels == 0, torch.full_like(padded_labels, -100), padded_labels)

            if labels is not None:
                modality_data[modality_idx] = {
                    'tokens': padded_tokens,
                    'labels': padded_labels,
                    'attention_mask': attention_masks,
                    'original_positions': self._get_position_mapping(modality_masks[modality_idx]),
                    'original_length': labels.shape[1]
                }
            else:
                modality_data[modality_idx] = {
                    'tokens': padded_tokens,
                    'labels': None,
                    'attention_mask': attention_masks,
                    'original_positions': self._get_position_mapping(modality_masks[modality_idx]),
                    'original_length': input_ids.shape[1]
                }

        return modality_data

    def create_batch_aware_modality_inputs_labels_old(self, input_ids: torch.Tensor, labels: torch.Tensor, modality_masks: torch.Tensor, ignore_modality_index = 0):
        """
        Create batch-aware data structure for each modality

        Args:
            input_ids: [batch_size, seq_length]  
            modality_masks: [n_modalities, batch_size, seq_length]

        Returns:
            Data organized by modality, but maintaining batch dimension
        """
        if len(input_ids.shape) == 2:
            batch_size, seq_length = input_ids.shape
        elif len(input_ids.shape) == 3:
            batch_size, seq_length, hidden_dim = input_ids.shape
        else:
            raise ValueError(f"Input IDs must have 2 or 3 dimensions, but got {len(input_ids.shape)}")

        modality_data = {}

        for modality_idx in range(self.n_modalities):
            # Calculate maximum length for this modality across batches
            max_tokens_in_modality = 0
            batch_tokens = []
            batch_labels = []
            batch_masks = []
            for batch_idx in range(batch_size):
                mask = modality_masks[modality_idx, batch_idx]
                tokens = input_ids[batch_idx][mask]
                if labels is not None:
                    labels_idx = labels[batch_idx][mask]
                else:
                    labels_idx = None
                max_tokens_in_modality = max(max_tokens_in_modality, len(tokens))
                batch_tokens.append(tokens)
                batch_labels.append(labels_idx)

            if max_tokens_in_modality > 0:
                # Perform minimal padding within this modality
                padded_tokens = []
                padded_labels = []
                attention_masks = []

                for tokens, labels_idx in zip(batch_tokens, batch_labels):
                    if len(tokens) < max_tokens_in_modality:
                        # Only pad within modality, not globally
                        pad_length = max_tokens_in_modality - len(tokens)
                        if tokens.dim() == 1:
                            padded = F.pad(tokens, (0, pad_length), value=0)
                            padded_label = F.pad(labels_idx, (0, pad_length), value=0)
                        elif tokens.dim() == 2:
                            padded = F.pad(tokens, (0, 0, 0, pad_length), value=0)
                            padded_label = F.pad(labels_idx, (0, 0, 0, pad_length), value=0)
                        else:
                            raise ValueError(f"Tokens must have 1 or 2 dimensions, but got {tokens.dim()}")
                        mask = torch.cat([
                            torch.ones(len(tokens), dtype=torch.bool, device=tokens.device),
                            torch.zeros(pad_length, dtype=torch.bool, device=tokens.device)
                        ])
                    else:
                        padded = tokens
                        padded_label = labels_idx
                        mask = torch.ones(len(tokens), dtype=torch.bool, device=tokens.device)

                    padded_tokens.append(padded)
                    padded_labels.append(padded_label)
                    attention_masks.append(mask)


                padded_tokens = torch.stack(padded_tokens)
                if labels is not None:
                    padded_labels = torch.stack(padded_labels)
                else:
                    padded_labels = None
                attention_masks = torch.stack(attention_masks)
                if modality_idx == ignore_modality_index and labels is not None:
                    padded_labels[padded_labels == 0] = -100

                if labels is not None:
                    modality_data[modality_idx] = {
                        'tokens': padded_tokens,  # [batch_size, max_tokens_in_modality]
                        'labels': padded_labels,  # [batch_size, max_tokens_in_modality]
                        'attention_mask': attention_masks,  # [batch_size, max_tokens_in_modality]
                        'original_positions': self._get_position_mapping(modality_masks[modality_idx]),
                        'original_length': labels.shape[1]
                    }
                else:
                    modality_data[modality_idx] = {
                        'tokens': padded_tokens,  # [batch_size, max_tokens_in_modality]
                        'labels': None,  # [batch_size, max_tokens_in_modality]
                        'attention_mask': attention_masks,  # [batch_size, max_tokens_in_modality]
                        'original_positions': self._get_position_mapping(modality_masks[modality_idx]),
                        'original_length': input_ids.shape[1]
                    }

        return modality_data

    def create_batch_aware_modality_inputs_labels_combined_0_1(self, input_ids: torch.Tensor, labels: torch.Tensor, modality_masks: torch.Tensor, ignore_modality_index = 0):
        """
        Create batch-aware data structure for each modality

        Args:
            input_ids: [batch_size, seq_length]  
            modality_masks: [n_modalities, batch_size, seq_length]

        Returns:
            Data organized by modality, but maintaining batch dimension
        """
        if len(input_ids.shape) == 2:
            batch_size, seq_length = input_ids.shape
        elif len(input_ids.shape) == 3:
            batch_size, seq_length, hidden_dim = input_ids.shape
        else:
            raise ValueError(f"Input IDs must have 2 or 3 dimensions, but got {len(input_ids.shape)}")

        modality_data = {}

        mod_0_1 = torch.logical_or(modality_masks[0], modality_masks[1])
        modality_masks_new = torch.stack([mod_0_1, modality_masks[2]], dim=0)

        for modality_idx in range(2):
            # Calculate maximum length for this modality across batches
            max_tokens_in_modality = 0
            batch_tokens = []
            batch_labels = []
            batch_masks = []
            for batch_idx in range(batch_size):
                mask = modality_masks_new[modality_idx, batch_idx]
                tokens = input_ids[batch_idx][mask]
                if labels is not None:
                    labels_idx = labels[batch_idx][mask]
                else:
                    labels_idx = None
                max_tokens_in_modality = max(max_tokens_in_modality, len(tokens))
                batch_tokens.append(tokens)
                batch_labels.append(labels_idx)

            if max_tokens_in_modality > 0:
                # Perform minimal padding within this modality
                padded_tokens = []
                padded_labels = []
                attention_masks = []

                for tokens, labels_idx in zip(batch_tokens, batch_labels):
                    if len(tokens) < max_tokens_in_modality:
                        # Only pad within modality, not globally
                        pad_length = max_tokens_in_modality - len(tokens)
                        if tokens.dim() == 1:
                            padded = F.pad(tokens, (0, pad_length), value=0)
                            padded_label = F.pad(labels_idx, (0, pad_length), value=0)
                        elif tokens.dim() == 2:
                            padded = F.pad(tokens, (0, 0, 0, pad_length), value=0)
                            padded_label = F.pad(labels_idx, (0, 0, 0, pad_length), value=0)
                        else:
                            raise ValueError(f"Tokens must have 1 or 2 dimensions, but got {tokens.dim()}")
                        mask = torch.cat([
                            torch.ones(len(tokens), dtype=torch.bool, device=tokens.device),
                            torch.zeros(pad_length, dtype=torch.bool, device=tokens.device)
                        ])
                    else:
                        padded = tokens
                        padded_label = labels_idx
                        mask = torch.ones(len(tokens), dtype=torch.bool, device=tokens.device)

                    padded_tokens.append(padded)
                    padded_labels.append(padded_label)
                    attention_masks.append(mask)


                padded_tokens = torch.stack(padded_tokens)
                if labels is not None:
                    padded_labels = torch.stack(padded_labels)
                else:
                    padded_labels = None
                attention_masks = torch.stack(attention_masks)
                if modality_idx == ignore_modality_index and labels is not None:
                    padded_labels[padded_labels == 0] = -100

                if labels is not None:
                    modality_data[modality_idx] = {
                        'tokens': padded_tokens,  # [batch_size, max_tokens_in_modality]
                        'labels': padded_labels,  # [batch_size, max_tokens_in_modality]
                        'attention_mask': attention_masks,  # [batch_size, max_tokens_in_modality]
                        'original_positions': self._get_position_mapping(modality_masks_new[modality_idx]),
                        'original_length': labels.shape[1]
                    }
                else:
                    modality_data[modality_idx] = {
                        'tokens': padded_tokens,  # [batch_size, max_tokens_in_modality]
                        'labels': None,  # [batch_size, max_tokens_in_modality]
                        'attention_mask': attention_masks,  # [batch_size, max_tokens_in_modality]
                        'original_positions': self._get_position_mapping(modality_masks[modality_idx]),
                        'original_length': input_ids.shape[1]
                    }

            else:
                    modality_data[modality_idx] = {
                        'labels': None,
                        'original_length': 0
                    }        
        return modality_data


    ### TODO: change to create_batch_aware_expert_inputs_labels, need to verify
    def create_batch_aware_expert_inputs_labels(self, input_ids: torch.Tensor, labels: torch.Tensor, position_encoding_indices: torch.Tensor, modality_masks: torch.Tensor, ignore_modality_index = 0):
        """
        Create batch-aware data structure for each modality (vectorized version)

        Args:
            input_ids: [batch_size, seq_length]  
            position_encoding_indices: [batch_size, seq_length]
            modality_masks: [n_modalities, batch_size, seq_length]

        Returns:
            Data organized by modality, but maintaining batch dimension
        """
        if len(input_ids.shape) == 2:
            batch_size, seq_length = input_ids.shape
        elif len(input_ids.shape) == 3:
            batch_size, seq_length, hidden_dim = input_ids.shape
        else:
            raise ValueError(f"Input IDs must have 2 or 3 dimensions, but got {len(input_ids.shape)}")

        expert_data = {}
        n_modalities = modality_masks.shape[0]

        # Vectorized modality combination
        mod_0_1 = torch.logical_or(modality_masks[0], modality_masks[1])
        if n_modalities > 2:
            mod_2 = modality_masks[2:].any(dim=0)  # Vectorized OR across modalities 2+
        else:
            mod_2 = torch.zeros_like(mod_0_1)
        modality_masks_new = torch.stack([mod_0_1, mod_2], dim=0)

        for modality_idx in range(2):
            mask = modality_masks_new[modality_idx]  # [batch_size, seq_length]

            # Calculate max tokens per batch using vectorized sum
            tokens_per_batch = mask.sum(dim=1)  # [batch_size]
            max_tokens_in_modality = int(tokens_per_batch.max().item())

            if max_tokens_in_modality == 0:
                expert_data[modality_idx] = {
                    'labels': None,
                    'original_length': 0
                }
                continue

            # Pre-allocate tensors
            device = input_ids.device
            padded_tokens = torch.zeros(batch_size, max_tokens_in_modality, dtype=input_ids.dtype, device=device)
            attention_masks = torch.zeros(batch_size, max_tokens_in_modality, dtype=torch.int64, device=device)
            if position_encoding_indices is not None:
                padded_position_encoding_indices = torch.zeros(batch_size, max_tokens_in_modality, dtype=position_encoding_indices.dtype, device=device)
            else:
                padded_position_encoding_indices = None
            if labels is not None:
                padded_labels = torch.zeros(batch_size, max_tokens_in_modality, dtype=labels.dtype, device=device)
            else:
                padded_labels = None

            # Fill tensors using advanced indexing (still need loop for ragged sequences)
            for batch_idx in range(batch_size):
                batch_mask = mask[batch_idx]
                num_tokens = int(tokens_per_batch[batch_idx].item())

                if num_tokens > 0:
                    padded_tokens[batch_idx, :num_tokens] = input_ids[batch_idx][batch_mask]
                    attention_masks[batch_idx, :num_tokens] = True
                    if position_encoding_indices is not None:
                        padded_position_encoding_indices[batch_idx, :num_tokens] = position_encoding_indices[batch_idx][batch_mask]
                    if labels is not None:
                        padded_labels[batch_idx, :num_tokens] = labels[batch_idx][batch_mask]

            # Apply ignore index for padding
            if modality_idx == ignore_modality_index and labels is not None:
                padded_labels[padded_labels == 0] = -100

            # Build result dictionary
            if labels is not None:
                expert_data[modality_idx] = {
                    'tokens': padded_tokens,
                    'labels': padded_labels,
                    'attention_mask': attention_masks,
                    'position_encoding_indices': padded_position_encoding_indices,
                    'original_positions': self._get_position_mapping(modality_masks_new[modality_idx]),
                    'original_length': labels.shape[1]
                }
            else:
                expert_data[modality_idx] = {
                    'tokens': padded_tokens,
                    'labels': None,
                    'attention_mask': attention_masks,
                    'position_encoding_indices': padded_position_encoding_indices,
                    'original_positions': self._get_position_mapping(modality_masks_new[modality_idx]),
                    'original_length': input_ids.shape[1]
                }

        return expert_data, modality_masks_new

    def restore_to_original_sequence(self, modality_outputs: list, original_input_shape: tuple, modality_masks: torch.Tensor):
        """
        Restore processed modality outputs back to original sequence positions

        Args:
            modality_outputs: List containing processed outputs for each modality
                             [tensor0, tensor1, ...] where each tensor is [batch_size, max_tokens_in_modality, hidden_dim]
            original_input_shape: Original input shape (batch_size, seq_length)
            modality_masks: [n_modalities, batch_size, seq_length] - same as used in create_batch_aware_modality_data

        Returns:
            restored_sequence: [batch_size, seq_length, hidden_dim] - restored to original positions
        """
        batch_size, seq_length = original_input_shape

        # Get hidden dimension from first available modality output
        hidden_dim = None
        for outputs in modality_outputs:
            if outputs is not None:
                hidden_dim = outputs.shape[-1]
                break

        if hidden_dim is None:
            raise ValueError("No valid outputs found in modality_outputs")

        # Initialize restored sequence with zeros
        device = None
        dtype = None
        for outputs in modality_outputs:
            if outputs is not None:
                device = outputs.device
                dtype = outputs.dtype
                break

        restored_sequence = torch.zeros(batch_size, seq_length, hidden_dim, device=device, dtype=dtype)

        # Restore each modality's outputs to their original positions
        for modality_idx, outputs in enumerate(modality_outputs):
            if outputs is None:
                continue

            # outputs: [batch_size, max_tokens_in_modality, hidden_dim]

            # Get original positions for this modality
            position_mappings = self._get_position_mapping(modality_masks[modality_idx])

            for batch_idx in range(batch_size):
                original_positions = position_mappings[batch_idx]

                if len(original_positions) == 0:
                    continue

                # Get outputs for this specific batch - this should be [max_tokens_in_modality, hidden_dim]
                single_batch_outputs = outputs[batch_idx]  # [max_tokens_in_modality, hidden_dim]

                # Only take the number of tokens that correspond to original positions
                num_tokens = min(len(original_positions), single_batch_outputs.shape[1])
                valid_outputs = single_batch_outputs[:, :num_tokens]  # [heads, num_tokens, hidden_dim]
                valid_positions = original_positions[:num_tokens]  # [num_tokens]

                if len(valid_outputs) > 0:
                    restored_sequence[batch_idx, valid_positions] = valid_outputs

        return restored_sequence


    def restore_to_original_feature(self, modality_outputs: list, original_input_shape: tuple, modality_masks: torch.Tensor):
        """
        Restore processed modality outputs back to original sequence positions

        Args:
            modality_outputs: List containing processed outputs for each modality
                             [tensor0, tensor1, ...] where each tensor is [batch_size, max_tokens_in_modality, hidden_dim]
            original_input_shape: Original input shape (batch_size, seq_length)
            modality_masks: [n_modalities, batch_size, seq_length] - same as used in create_batch_aware_modality_data

        Returns:
            restored_sequence: [batch_size, n_heads, seq_length, hidden_dim] - restored to original positions
        """ 
        batch_size, n_heads, seq_length = original_input_shape


        # Get hidden dimension from first available modality output
        hidden_dim = None
        for outputs in modality_outputs:
            if outputs is not None:
                hidden_dim = outputs.shape[-1]
                break

        if hidden_dim is None:
            raise ValueError("No valid outputs found in modality_outputs")

        # Initialize restored sequence with zeros
        device = None
        dtype = None
        for outputs in modality_outputs:
            if outputs is not None:
                device = outputs.device
                dtype = outputs.dtype
                break

        restored_sequence = torch.zeros(batch_size, n_heads, seq_length, hidden_dim, device=device, dtype=dtype)

        # Restore each modality's outputs to their original positions
        for modality_idx, outputs in enumerate(modality_outputs):
            if outputs is None:
                continue

            # outputs: [batch_size, max_tokens_in_modality, hidden_dim]

            # Get original positions for this modality
            position_mappings = self._get_position_mapping(modality_masks[modality_idx])

            for batch_idx in range(batch_size):

                seq_length_expert = modality_masks[modality_idx, batch_idx].sum()

                original_positions = position_mappings[batch_idx]

                if len(original_positions) == 0:
                    continue

                # Get outputs for this specific batch - this should be [max_tokens_in_modality, hidden_dim]
                single_batch_outputs = outputs[batch_idx, :,:seq_length_expert,:]  # [max_tokens_in_modality, hidden_dim]

                # Only take the number of tokens that correspond to original positions
                # Restore outputs to their original positions in the sequence
                restored_sequence[batch_idx, :, original_positions] = single_batch_outputs

        return restored_sequence

    def get_modality_specific_inputs(self, hidden_states: torch.Tensor, modality_masks: torch.Tensor):
        """
        Extract modality-specific inputs for processing

        Args:
            hidden_states: [batch_size, seq_length, hidden_dim]
            modality_masks: [n_modalities, batch_size, seq_length]

        Returns:
            modality_inputs: Dict containing inputs for each modality
        """
        batch_size, seq_length, hidden_dim = hidden_states.shape
        modality_inputs = {}

        for modality_idx in range(self.n_modalities):
            # Calculate maximum length for this modality across batches
            max_tokens_in_modality = 0
            batch_states = []

            for batch_idx in range(batch_size):
                mask = modality_masks[modality_idx, batch_idx]
                states = hidden_states[batch_idx][mask]
                max_tokens_in_modality = max(max_tokens_in_modality, len(states))
                batch_states.append(states)

            if max_tokens_in_modality > 0:
                # Perform minimal padding within this modality
                padded_states = []
                attention_masks = []

                for states in batch_states:
                    if len(states) < max_tokens_in_modality:
                        # Pad with zeros
                        pad_length = max_tokens_in_modality - len(states)
                        padded = F.pad(states, (0, 0, 0, pad_length), value=0.0)
                        mask = torch.cat([
                            torch.ones(len(states), dtype=torch.bool, device=states.device),
                            torch.zeros(pad_length, dtype=torch.bool, device=states.device)
                        ])
                    else:
                        padded = states
                        mask = torch.ones(len(states), dtype=torch.bool, device=states.device)

                    padded_states.append(padded)
                    attention_masks.append(mask)

                modality_inputs[modality_idx] = {
                    'hidden_states': torch.stack(padded_states),  # [batch_size, max_tokens_in_modality, hidden_dim]
                    'attention_mask': torch.stack(attention_masks),  # [batch_size, max_tokens_in_modality]
                    'original_positions': self._get_position_mapping(modality_masks[modality_idx])
                }

        return modality_inputs


def default_init(cls, *args, **kwargs):
    return cls(*args, **kwargs)


class InvalidScoreLogitsProcessor(LogitsProcessor):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if torch.isnan(scores).any() or torch.isinf(scores).any():
            scores.zero_()
            scores[..., 198] = 5e4
        return scores


def split_tensor_along_last_dim(
        tensor: torch.Tensor,
        num_partitions: int,
        contiguous_split_chunks: bool = False,
) -> List[torch.Tensor]:
    """Split a tensor along its last dimension.

    Arguments:
        tensor: input tensor.
        num_partitions: number of partitions to split the tensor
        contiguous_split_chunks: If True, make each chunk contiguous
                                 in memory.

    Returns:
        A list of Tensors
    """
    # Get the size and dimension.
    last_dim = tensor.dim() - 1
    last_dim_size = tensor.size()[last_dim] // num_partitions
    # Split.
    tensor_list = torch.split(tensor, last_dim_size, dim=last_dim)
    # Note: torch.split does not create contiguous tensors by default.
    if contiguous_split_chunks:
        return tuple(chunk.contiguous() for chunk in tensor_list)

    return tensor_list


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, rope_ratio=1, original_impl=False, device=None, dtype=None):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=device).to(dtype=dtype) / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.dim = dim
        self.original_impl = original_impl
        self.rope_ratio = rope_ratio

    def forward_impl(
            self, seq_len: int, n_elem: int, dtype: torch.dtype, device: torch.device, base: int = 10000
    ):
        """Enhanced Transformer with Rotary Position Embedding.

        Derived from: https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/
        transformers/rope/__init__.py. MIT License:
        https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/license.
        """
        # $\Theta = {\theta_i = 10000^{\frac{2(i-1)}{d}}, i \in [1, 2, ..., \frac{d}{2}]}$
        base = base * self.rope_ratio
        theta = 1.0 / (base ** (torch.arange(0, n_elem, 2, dtype=torch.float, device=device) / n_elem))

        # Create position indexes `[0, 1, ..., seq_len - 1]`
        seq_idx = torch.arange(seq_len, dtype=torch.float, device=device)

        # Calculate the product of position index and $\theta_i$
        idx_theta = torch.outer(seq_idx, theta).float()

        cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)

        # this is to mimic the behaviour of complex32, else we will get different results
        if dtype in (torch.float16, torch.bfloat16, torch.int8):
            cache = cache.bfloat16() if dtype == torch.bfloat16 else cache.half()
        return cache

    def forward(self, max_seq_len, offset=0):
        """
        Forward pass to generate rotary positional embeddings.
        
        Args:
            max_seq_len: Maximum sequence length
            offset: Offset for position encoding
            
        Returns:
            Rotary positional embedding tensor
        """
        return self.forward_impl(
            max_seq_len, self.dim, dtype=self.inv_freq.dtype, device=self.inv_freq.device
        )


@torch.jit.script
def apply_rotary_pos_emb(x: torch.Tensor, rope_cache: torch.Tensor) -> torch.Tensor:
    # x: [b, np, sq, hn]
    b, np, sq, hn = x.size(0), x.size(1), x.size(2), x.size(3)
    rot_dim = rope_cache.shape[-2] * 2
    x, x_pass = x[..., :rot_dim], x[..., rot_dim:]
    # truncate to support variable sizes
    rope_cache = rope_cache[:sq]
    xshaped = x.reshape(b, np, sq, rot_dim // 2, 2)
    rope_cache = rope_cache.view(-1, 1, sq, xshaped.size(3), 2)
    x_out2 = torch.stack(
        [
            xshaped[..., 0] * rope_cache[..., 0] - xshaped[..., 1] * rope_cache[..., 1],
            xshaped[..., 1] * rope_cache[..., 0] + xshaped[..., 0] * rope_cache[..., 1],
        ],
        -1,
    )
    x_out2 = x_out2.flatten(3)
    return torch.cat((x_out2, x_pass), dim=-1)


# @torch.jit.script
def apply_rotary_pos_emb_batch(x: torch.Tensor, rope_cache: torch.Tensor) -> torch.Tensor:
    # x: [b, np, sq, hn]
    # rope_cache: [b, S_total, dim_half, 2]  (batched cos/sin cache)
    b, np, sq, hn = x.size(0), x.size(1), x.size(2), x.size(3)

    # Rotation dimension = last second dimension of rope_cache * 2
    dim_half = rope_cache.shape[-2]
    rot_dim = dim_half * 2
    assert rot_dim <= hn, "rotary dim exceeds head dim"

    # Split out the first rot_dim dimensions for RoPE, and the remaining passthrough part
    x_rot, x_pass = x[..., :rot_dim], x[..., rot_dim:]           # [b, np, sq, rot_dim]

    # Only take the first sq positions of rope (slice on sequence dimension)
    rope = rope_cache[:, :sq, :, :]                              # [b, sq, dim_half, 2]

    # Reshape for broadcasting: x_rot->[b,np,sq,dim_half,2], rope->[b,1,sq,dim_half,2]
    x_rot = x_rot.view(b, np, sq, dim_half, 2)
    rope  = rope.unsqueeze(1)                                     # [b, 1, sq, dim_half, 2]

    # Complex rotation: (x0 + i x1) * (c + i s)
    x_out2 = torch.stack(
        [
            x_rot[..., 0] * rope[..., 0] - x_rot[..., 1] * rope[..., 1],
            x_rot[..., 1] * rope[..., 0] + x_rot[..., 0] * rope[..., 1],
        ],
        dim=-1,
    )                                                             # [b,np,sq,dim_half,2]

    # Flatten back to rot_dim and concatenate with passthrough part
    x_out2 = x_out2.flatten(3)                                    # [b,np,sq,rot_dim]
    return torch.cat((x_out2, x_pass), dim=-1)                    # [b,np,sq,hn]


def interpolate_rotary_pos_emb(
    rotary_pos_emb: torch.Tensor,
    insert_positions: List[Tuple[int, int]],
    num_new_tokens: List[int],
    interpolation_type: str = "linear",
    base: int = 10000,
    rope_ratio: float = 1.0
) -> torch.Tensor:
    """
    Interpolate rotary position embeddings to insert new positions between existing ones.

    This function is designed for multi-modal scenarios where tokens from a second modality
    need to be inserted at specific positions within the first modality's sequence.

    IMPORTANT: This function now correctly interpolates angles (theta) instead of directly
    interpolating cos/sin values, which preserves the rotational properties of the embeddings.

    Args:
        rotary_pos_emb: Original rotary position embeddings with shape [seq_len, dim, 2]
                       where the last dimension contains cos and sin values
        insert_positions: List of tuples (start_idx, end_idx) indicating where to insert.
                         For example, [(50, 51)] means insert between position 50 and 51
        num_new_tokens: List of integers indicating how many tokens to insert at each position
        interpolation_type: Type of interpolation - "linear" or "cubic" (default: "linear")
        base: Base value for computing inverse frequencies (default: 10000)
        rope_ratio: Rope ratio for scaling the base (default: 1.0)

    Returns:
        Interpolated rotary position embeddings with shape [new_seq_len, dim, 2]
        where new_seq_len = original_seq_len + sum(num_new_tokens)

    Example:
        # Insert 30 tokens between position 50 and 51
        new_rope = interpolate_rotary_pos_emb(
            rotary_pos_emb,  # shape: [100, 32, 2]
            insert_positions=[(50, 51)],
            num_new_tokens=[30]
        )  # returns shape: [130, 32, 2]
    """
    assert len(insert_positions) == len(num_new_tokens), \
        "Length of insert_positions must match length of num_new_tokens"

    seq_len, dim, two = rotary_pos_emb.shape
    assert two == 2, "Last dimension must be 2 (cos and sin values)"

    # Calculate the new total sequence length
    total_new_tokens = sum(num_new_tokens)
    new_seq_len = seq_len + total_new_tokens

    # Create a mapping from new positions to old positions
    old_to_new_mapping = []
    current_new_pos = 0

    # Sort insert positions to process them in order
    sorted_inserts = sorted(zip(insert_positions, num_new_tokens), key=lambda x: x[0][0])

    # Build the position mapping
    last_old_pos = 0
    for (start_idx, end_idx), n_tokens in sorted_inserts:
        # Add positions before the insertion point
        for old_pos in range(last_old_pos, start_idx + 1):
            old_to_new_mapping.append((current_new_pos, float(old_pos)))
            current_new_pos += 1

        # Add interpolated positions
        # These will be interpolated between start_idx and end_idx
        interpolated_positions = []
        for i in range(n_tokens):
            # Linear interpolation weight
            alpha = (i + 1) / (n_tokens + 1)
            interpolated_pos = start_idx + alpha * (end_idx - start_idx)
            interpolated_positions.append((current_new_pos, interpolated_pos))
            current_new_pos += 1

        old_to_new_mapping.extend(interpolated_positions)
        last_old_pos = end_idx

    # Add remaining positions after the last insertion
    for old_pos in range(last_old_pos, seq_len):
        old_to_new_mapping.append((current_new_pos, float(old_pos)))
        current_new_pos += 1

    # Create the new rotary embeddings
    new_rotary_pos_emb = torch.zeros(
        (new_seq_len, dim, 2), 
        dtype=rotary_pos_emb.dtype, 
        device=rotary_pos_emb.device
    )

    # Compute inverse frequencies for angle calculation
    # Following the same pattern as RotaryEmbedding.forward_impl
    base_with_ratio = base * rope_ratio
    inv_freq = 1.0 / (base_with_ratio ** (torch.arange(0, dim * 2, 2, dtype=torch.float32, device=rotary_pos_emb.device) / (dim * 2)))

    # Generate rotary embeddings by interpolating angles
    for new_idx, position in old_to_new_mapping:
        # Calculate theta (angle) for this position
        theta = position * inv_freq

        # Compute cos and sin from the angle
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)

        # Store in the final rotary embeddings
        new_rotary_pos_emb[new_idx, :, 0] = cos_theta
        new_rotary_pos_emb[new_idx, :, 1] = sin_theta

    # Convert to appropriate dtype if needed
    if rotary_pos_emb.dtype in (torch.float16, torch.bfloat16):
        new_rotary_pos_emb = new_rotary_pos_emb.to(rotary_pos_emb.dtype)

    return new_rotary_pos_emb


def create_modality_aware_rotary_embeddings(
    rotary_embedding_module: RotaryEmbedding,
    first_modality_length: int,
    second_modality_positions: List[Tuple[int, int, int]],
    base: int = 10000
) -> torch.Tensor:
    """
    Create rotary position embeddings for multi-modal inputs where the second modality
    tokens are inserted at specific positions within the first modality sequence.

    Args:
        rotary_embedding_module: The RotaryEmbedding module instance
        first_modality_length: Length of the first modality sequence
        second_modality_positions: List of tuples (start_pos, end_pos, num_tokens)
                                  indicating where and how many second modality tokens to insert
        base: Base value for rotary embeddings computation

    Returns:
        Rotary position embeddings with shape [total_length, dim, 2]

    Example:
        # First modality has 100 tokens, insert 30 second modality tokens between positions 50-51
        rope = create_modality_aware_rotary_embeddings(
            rotary_emb_module,
            first_modality_length=100,
            second_modality_positions=[(50, 51, 30)]
        )
    """
    # Generate rotary embeddings for the first modality
    first_modality_rope = rotary_embedding_module.forward_impl(
        seq_len=first_modality_length,
        n_elem=rotary_embedding_module.dim,
        dtype=rotary_embedding_module.inv_freq.dtype,
        device=rotary_embedding_module.inv_freq.device,
        base=base
    )

    # If no second modality positions, return the first modality embeddings
    if not second_modality_positions:
        return first_modality_rope

    # Extract insert positions and number of tokens
    insert_positions = [(pos[0], pos[1]) for pos in second_modality_positions]
    num_new_tokens = [pos[2] for pos in second_modality_positions]

    # Interpolate to create the final rotary embeddings
    final_rope = interpolate_rotary_pos_emb(
        first_modality_rope,
        insert_positions=insert_positions,
        num_new_tokens=num_new_tokens,
        base=base,
        rope_ratio=rotary_embedding_module.rope_ratio
    )

    return final_rope


def create_rotary_embeddings_from_modality_masks(
    rotary_embedding_module: RotaryEmbedding,
    modality_masks: torch.Tensor,
    primary_modality_idx: int = 0,
    interpolation_type: str = "linear",
    base: int = 10000
) -> torch.Tensor:
    """
    Create rotary position embeddings based on modality masks for three modalities.
    Modalities 0 and 1 form the primary sequence, modality 2 is interpolated.

    Special interpolation logic for modality 2:
    - First token: interpolated at the midpoint between modality 0's last position and modality 1's first position
    - Remaining tokens: two tokens interpolated between each pair of modality 1 positions

    Args:
        rotary_embedding_module: The RotaryEmbedding module instance
        modality_masks: Tensor of shape [n_modalities, batch_size, seq_length] or [n_modalities, seq_length]
                       where 1 indicates the presence of that modality at that position
        primary_modality_idx: Not used in this version, kept for compatibility
        interpolation_type: Type of interpolation - "linear" or "cubic" (default: "linear")
        base: Base value for rotary embeddings computation

    Returns:
        Rotary position embeddings with shape [seq_length, dim, 2]

    Example:
        # Modality masks for sequence [0,0,0,1,1,2,2,2,2,2,0,0,0]
        modality_masks = torch.tensor([
            [[1,1,1,0,0,0,0,0,0,0,1,1,1]],  # Modality 0
            [[0,0,0,1,1,0,0,0,0,0,0,0,0]],  # Modality 1
            [[0,0,0,0,0,1,1,1,1,1,0,0,0]]   # Modality 2
        ])
    """
    # Handle both [n_modalities, seq_length] and [n_modalities, batch_size, seq_length]
    if modality_masks.dim() == 3:
        # For batched inputs, we assume all batches have the same modality pattern
        # Take the first batch as reference
        modality_masks = modality_masks[:, 0, :]

    n_modalities, seq_length = modality_masks.shape
    assert n_modalities >= 3, "This function requires at least 3 modalities"

    # Get positions for modality 0 and 1 (primary modalities)
    modality0_mask = modality_masks[0].bool()
    modality1_mask = modality_masks[1].bool()
    modality2_mask = modality_masks[2].bool()

    modality0_positions = torch.nonzero(modality0_mask, as_tuple=False).flatten()
    modality1_positions = torch.nonzero(modality1_mask, as_tuple=False).flatten()
    modality2_positions = torch.nonzero(modality2_mask, as_tuple=False).flatten()

    # Combine modality 0 and 1 to form the primary sequence
    primary_positions = torch.cat([modality0_positions, modality1_positions])
    primary_positions, _ = torch.sort(primary_positions)

    # Generate rotary embeddings for primary sequence
    primary_length = len(primary_positions)
    primary_rope = rotary_embedding_module.forward_impl(
        seq_len=primary_length,
        n_elem=rotary_embedding_module.dim,
        dtype=rotary_embedding_module.inv_freq.dtype,
        device=rotary_embedding_module.inv_freq.device,
        base=base
    )

    # Initialize the final rotary embeddings
    final_rope = torch.zeros(
        (seq_length, rotary_embedding_module.dim // 2, 2),
        dtype=primary_rope.dtype,
        device=primary_rope.device
    )

    # Place primary modality embeddings at their original positions
    final_rope[primary_positions] = primary_rope

    # Process modality 2 with simplified interpolation logic
    if len(modality2_positions) > 0:
        # Prepare for angle-based interpolation
        base_with_ratio = base * rotary_embedding_module.rope_ratio
        inv_freq = 1.0 / (base_with_ratio ** (torch.arange(
            0, rotary_embedding_module.dim, 2, 
            dtype=torch.float32, 
            device=final_rope.device
        ) / rotary_embedding_module.dim))

        # Find consecutive groups of modality 2 positions
        modality2_groups = _find_consecutive_groups(modality2_positions)

        # Process each group of modality 2 tokens
        for group_idx, group in enumerate(modality2_groups):
            group_start = group[0].item()
            group_end = group[-1].item()

            # Find the modality 0 position just before this group to identify the current cycle
            mod0_before_group = modality0_positions[modality0_positions < group_start]
            if len(mod0_before_group) > 0:
                # Find the last modality 0 before this group
                last_mod0_before = mod0_before_group[-1].item()

                # Find modality 1 positions in the current cycle
                # These are modality 1 positions after the last modality 0 and before this modality 2 group
                current_cycle_mod1 = modality1_positions[(modality1_positions > last_mod0_before) & (modality1_positions < group_start)]
            else:
                # No modality 0 before, use all modality 1 before this group
                current_cycle_mod1 = modality1_positions[modality1_positions < group_start]

            if len(current_cycle_mod1) > 0:
                # Get the first modality 1 position in the current cycle
                first_mod1 = current_cycle_mod1[0].item()
                first_mod1_idx = (primary_positions == first_mod1).nonzero(as_tuple=False).item()

                # Process the first token of modality 2
                # It should be 0.5 before the first modality 1 token
                first_token_pos = group[0].item()
                interpolated_position = first_mod1_idx - 0.5
                theta = interpolated_position * inv_freq
                final_rope[first_token_pos, :, 0] = torch.cos(theta)
                final_rope[first_token_pos, :, 1] = torch.sin(theta)

                # Process remaining tokens
                # Each pair goes after each modality 1 token
                remaining_tokens = group[1:]
                for i, pos in enumerate(remaining_tokens):
                    mod1_idx = i // 2  # Which modality 1 token this pair belongs to
                    within_pair = i % 2  # Position within the pair (0 or 1)

                    if mod1_idx < len(current_cycle_mod1):
                        # Get the modality 1 token this pair follows
                        current_mod1 = current_cycle_mod1[mod1_idx].item()
                        current_idx = (primary_positions == current_mod1).nonzero(as_tuple=False).item()

                        # Interpolate after this modality 1 position
                        # Use 1/3 and 2/3 for the two tokens in each pair
                        # Check if there's a next modality 1 in the current cycle
                        if mod1_idx < len(current_cycle_mod1) - 1:
                            # Get the next modality 1 position
                            next_mod1 = current_cycle_mod1[mod1_idx + 1].item()
                            next_idx = (primary_positions == next_mod1).nonzero(as_tuple=False).item()
                            alpha = (within_pair + 1) / 3.0
                            interpolated_position = (1 - alpha) * current_idx + alpha * next_idx
                        else:
                            # This is the last modality 1 in the cycle, extrapolate
                            interpolated_position = current_idx + (within_pair + 1) * 0.333
                    else:
                        # More modality 2 tokens than expected, extrapolate
                        last_mod1 = current_cycle_mod1[-1].item()
                        last_mod1_idx = (primary_positions == last_mod1).nonzero(as_tuple=False).item()
                        overflow = i - len(current_cycle_mod1) * 2
                        interpolated_position = last_mod1_idx + (overflow + 1) * 0.5

                    theta = interpolated_position * inv_freq
                    final_rope[pos.item(), :, 0] = torch.cos(theta)
                    final_rope[pos.item(), :, 1] = torch.sin(theta)
            else:
                # No modality 1 before this group, shouldn't happen in normal usage
                # Just use sequential positions starting from 0
                for i, pos in enumerate(group):
                    # theta = i * inv_freq
                    theta = pos * inv_freq # changed by  juze, here should be the following number
                    final_rope[pos, :, 0] = torch.cos(theta)
                    final_rope[pos, :, 1] = torch.sin(theta)

    # Process any additional modalities (if n_modalities > 3)
    # Use the original interpolation logic for backward compatibility
    for modality_idx in range(3, n_modalities):
        modality_mask = modality_masks[modality_idx].bool()
        modality_positions = torch.nonzero(modality_mask, as_tuple=False).flatten()

        if len(modality_positions) == 0:
            continue

        # Use the original interpolation logic for additional modalities
        groups = _find_consecutive_groups(modality_positions)

        for group in groups:
            start_idx = group[0].item()
            end_idx = group[-1].item()

            # Find the nearest primary modality positions before and after this group
            before_pos = primary_positions[primary_positions < start_idx]
            after_pos = primary_positions[primary_positions > end_idx]

            if len(before_pos) > 0 and len(after_pos) > 0:
                # Interpolate between the nearest positions
                before_idx = before_pos[-1].item()
                after_idx = after_pos[0].item()

                # Get the rotary embeddings at these positions
                before_rope_idx = (primary_positions == before_idx).nonzero(as_tuple=False).item()
                after_rope_idx = (primary_positions == after_idx).nonzero(as_tuple=False).item()

                # Prepare for angle-based interpolation
                base_with_ratio = base * rotary_embedding_module.rope_ratio
                inv_freq = 1.0 / (base_with_ratio ** (torch.arange(
                    0, rotary_embedding_module.dim, 2, 
                    dtype=torch.float32, 
                    device=final_rope.device
                ) / rotary_embedding_module.dim))

                # Interpolate for each position in the group
                for i, pos in enumerate(group):
                    # Calculate interpolation weight
                    alpha = (i + 1) / (len(group) + 1)

                    # Interpolate the position index
                    interpolated_position = (1 - alpha) * before_rope_idx + alpha * after_rope_idx

                    # Calculate theta (angle) for the interpolated position
                    theta = interpolated_position * inv_freq

                    # Compute cos and sin from the interpolated angle
                    final_rope[pos, :, 0] = torch.cos(theta)
                    final_rope[pos, :, 1] = torch.sin(theta)

    return final_rope


def create_rotary_embeddings_from_modality_masks_multiple_modalities_fast(
    rotary_embedding_module: RotaryEmbedding,
    modality_masks: torch.Tensor,
    modality_fps: Dict[int, float] = None,
    primary_modality_idx: int = 0,
    interpolation_type: str = "linear",
    base: int = 10000
) -> torch.Tensor:
    """
    Fast version for fixed pattern: [variable M0] + 26×M1 + 53×M2 + 14×M3 + 14×M4 + 14×M5
    Uses precomputed position mappings for maximum performance.

    Pattern assumptions:
    - Modality 1: 26 tokens (base fps = 12.5)
    - Modality 2: 53 tokens (1 start + 52 regular, fps = 25.0)
    - Modality 3: 14 tokens (1 start + 13 regular, fps = 6.25)
    - Modality 4: 14 tokens (1 start + 13 regular, fps = 6.25)  
    - Modality 5: 14 tokens (1 start + 13 regular, fps = 6.25)

    Args:
        rotary_embedding_module: The RotaryEmbedding module instance
        modality_masks: Tensor of shape [n_modalities, batch_size, seq_length] or [n_modalities, seq_length]
        modality_fps: Dictionary mapping modality index to fps value

    Returns:
        Rotary position embeddings with shape [seq_length, dim, 2]
    """
    # Default fps configuration for fixed pattern
    if modality_fps is None:
        modality_fps = {1: 12.5, 2: 25.0, 3: 6.25, 4: 6.25, 5: 6.25}

    # Handle batch dimension
    if modality_masks.dim() == 3:
        modality_masks = modality_masks[:, 0, :]

    n_modalities, seq_length = modality_masks.shape

    # Get positions for all modalities
    modality_positions = {}
    for i in range(n_modalities):
        mask = modality_masks[i].bool()
        positions = torch.nonzero(mask, as_tuple=False).flatten()
        if len(positions) > 0:
            modality_positions[i] = positions

    # Combine modality 0 and 1 to form the primary sequence
    primary_positions = []
    if 0 in modality_positions:
        primary_positions.append(modality_positions[0])
    if 1 in modality_positions:
        primary_positions.append(modality_positions[1])

    if primary_positions:
        primary_positions = torch.cat(primary_positions)
        primary_positions, _ = torch.sort(primary_positions)
    else:
        raise ValueError("Primary modalities (0 or 1) must be present")

    if 1 not in modality_positions:
        return None

    mod1_positions = modality_positions[1]
    if len(mod1_positions) == 0:
        return None

    # Generate rotary embeddings for primary sequence
    primary_length = len(primary_positions)
    primary_rope = rotary_embedding_module.forward_impl(
        seq_len=primary_length,
        n_elem=rotary_embedding_module.dim,
        dtype=rotary_embedding_module.inv_freq.dtype,
        device=rotary_embedding_module.inv_freq.device,
        base=base
    )

    # Initialize the final rotary embeddings
    final_rope = torch.zeros(
        (seq_length, rotary_embedding_module.dim // 2, 2),
        dtype=primary_rope.dtype,
        device=primary_rope.device
    )

    # Place primary modality embeddings at their original positions
    final_rope[primary_positions] = primary_rope

    # Prepare for angle-based interpolation
    base_with_ratio = base * rotary_embedding_module.rope_ratio
    inv_freq = 1.0 / (base_with_ratio ** (torch.arange(
        0, rotary_embedding_module.dim, 2, 
        dtype=rotary_embedding_module.inv_freq.dtype, 
        device=rotary_embedding_module.inv_freq.device
    ) / rotary_embedding_module.dim))

    # Create pos→rope mapping for fast lookups
    pos_to_rope_idx = {}
    for i, pos in enumerate(primary_positions):
        pos_to_rope_idx[pos.item()] = i

    # Get mod1 rope indices
    mod1_rope_indices = []
    for mod1_pos in mod1_positions:
        rope_idx = pos_to_rope_idx[mod1_pos.item()]
        mod1_rope_indices.append(rope_idx)

    # Fast path: precomputed position mappings for fixed pattern
    if len(mod1_positions) == 26:  # Expected pattern
        K = 25  # Number of intervals between mod1 anchors

        # Precomputed interval assignments for each modality
        # Modality 2: 52 regular tokens distributed across 25 intervals
        if 2 in modality_positions and len(modality_positions[2]) == 53:
            positions = modality_positions[2]
            _apply_modality2_fast_pattern(positions, mod1_rope_indices, inv_freq, final_rope, K)

        # Modalities 3,4,5: 13 regular tokens each
        for modality_idx in [3, 4, 5]:
            if modality_idx in modality_positions and len(modality_positions[modality_idx]) == 14:
                positions = modality_positions[modality_idx]
                _apply_modality345_fast_pattern(positions, modality_idx, mod1_rope_indices, inv_freq, final_rope, K)
    else:
        # Fallback to general algorithm for non-standard patterns
        for modality_idx in range(2, min(6, n_modalities)):
            if modality_idx not in modality_positions:
                continue
            positions = modality_positions[modality_idx]
            _apply_general_positions(
                positions, modality_idx, mod1_rope_indices,
                inv_freq, final_rope, modality_fps
            )

    return final_rope


def _apply_modality2_fast_pattern(
    positions: torch.Tensor,
    mod1_rope_indices: List[int],
    inv_freq: torch.Tensor,
    final_rope: torch.Tensor,
    K: int
):
    """
    Fast pattern for modality 2: 1 start + 52 regular tokens
    Precomputed distribution: ~2 tokens per interval (52/25 = 2.08)
    """
    # Start token: -1/5 offset from first mod1
    start_pos = positions[0].item()
    start_interpolated_position = mod1_rope_indices[0] - 0.2  # -1/5
    theta = start_interpolated_position * inv_freq
    final_rope[start_pos, :, 0] = torch.cos(theta)
    final_rope[start_pos, :, 1] = torch.sin(theta)

    # Regular tokens: precomputed distribution across 25 intervals
    regular_positions = positions[1:]

    # Precomputed assignment: 2 tokens in first 2 intervals, 3 tokens in remaining
    token_idx = 0
    for interval_idx in range(K):
        start_rope_idx = mod1_rope_indices[interval_idx]
        end_rope_idx = mod1_rope_indices[interval_idx + 1]

        # Precomputed pattern: intervals 0-1 get 2 tokens, others get 2-3
        if interval_idx < 23:  # First 23 intervals get 2 tokens
            interval_token_count = 2
        else:  # Last 2 intervals get 3 tokens (23*2 + 2*3 = 52)
            interval_token_count = 3

        # Apply uniform interpolation in this interval
        for i in range(interval_token_count):
            if token_idx < len(regular_positions):
                pos = regular_positions[token_idx].item()
                alpha = (i + 1) / (interval_token_count + 1)
                interpolated_position = start_rope_idx + alpha * (end_rope_idx - start_rope_idx)

                theta = interpolated_position * inv_freq
                final_rope[pos, :, 0] = torch.cos(theta)
                final_rope[pos, :, 1] = torch.sin(theta)
                token_idx += 1


def _apply_modality345_fast_pattern(
    positions: torch.Tensor,
    modality_idx: int,
    mod1_rope_indices: List[int],
    inv_freq: torch.Tensor,
    final_rope: torch.Tensor,
    K: int
):
    """
    Fast pattern for modalities 3,4,5: 1 start + 13 regular tokens each
    Precomputed distribution: ~0.52 tokens per interval (13/25 = 0.52)
    """
    # Start token with appropriate offset
    start_offset = -(modality_idx - 1) / 5.0  # -2/5, -3/5, -4/5
    start_pos = positions[0].item()
    start_interpolated_position = mod1_rope_indices[0] + start_offset
    theta = start_interpolated_position * inv_freq
    final_rope[start_pos, :, 0] = torch.cos(theta)
    final_rope[start_pos, :, 1] = torch.sin(theta)

    # Regular tokens: precomputed sparse distribution
    regular_positions = positions[1:]

    # Precomputed pattern: place 1 token every ~2 intervals
    # 13 tokens across 25 intervals: intervals [0,2,4,6,8,10,12,14,16,18,20,22,24]
    token_idx = 0
    for interval_idx in range(0, K, 2):  # Every 2nd interval
        if token_idx >= len(regular_positions):
            break

        start_rope_idx = mod1_rope_indices[interval_idx]
        end_rope_idx = mod1_rope_indices[interval_idx + 1]

        pos = regular_positions[token_idx].item()
        # Place in middle of interval
        interpolated_position = (start_rope_idx + end_rope_idx) / 2

        theta = interpolated_position * inv_freq
        final_rope[pos, :, 0] = torch.cos(theta)
        final_rope[pos, :, 1] = torch.sin(theta)
        token_idx += 1

    # Handle remaining tokens if any (due to rounding)
    remaining_intervals = [1, 3, 5]  # Fill odd intervals for remaining tokens
    for i, interval_idx in enumerate(remaining_intervals):
        if token_idx >= len(regular_positions) or interval_idx >= K:
            break

        start_rope_idx = mod1_rope_indices[interval_idx]
        end_rope_idx = mod1_rope_indices[interval_idx + 1]

        pos = regular_positions[token_idx].item()
        interpolated_position = (start_rope_idx + end_rope_idx) / 2

        theta = interpolated_position * inv_freq
        final_rope[pos, :, 0] = torch.cos(theta)
        final_rope[pos, :, 1] = torch.sin(theta)
        token_idx += 1


def _apply_general_positions(
    positions: torch.Tensor,
    modality_idx: int,
    mod1_rope_indices: List[int],
    inv_freq: torch.Tensor,
    final_rope: torch.Tensor,
    modality_fps: Dict[int, float]
):
    """
    Fallback to general algorithm for non-standard patterns.
    """
    if len(positions) == 0 or len(mod1_rope_indices) < 2:
        return

    # Start token
    if modality_idx >= 2 and modality_idx <= 5:
        start_offset = -(modality_idx - 1) / 5.0
        start_pos = positions[0].item()
        start_interpolated_position = mod1_rope_indices[0] + start_offset
        theta = start_interpolated_position * inv_freq
        final_rope[start_pos, :, 0] = torch.cos(theta)
        final_rope[start_pos, :, 1] = torch.sin(theta)

        regular_positions = positions[1:]
    else:
        regular_positions = positions

    # Distribute regular tokens uniformly across intervals
    if len(regular_positions) > 0:
        K = len(mod1_rope_indices) - 1
        for i, pos in enumerate(regular_positions):
            interval_idx = min(int(i * K / len(regular_positions)), K - 1)
            start_rope_idx = mod1_rope_indices[interval_idx]
            end_rope_idx = mod1_rope_indices[interval_idx + 1]

            alpha = 0.5  # Place in middle of interval
            interpolated_position = start_rope_idx + alpha * (end_rope_idx - start_rope_idx)

            theta = interpolated_position * inv_freq
            final_rope[pos.item(), :, 0] = torch.cos(theta)
            final_rope[pos.item(), :, 1] = torch.sin(theta)


def compute_rotary_embeddings_from_precomputed_indices(
    rotary_embedding_module,
    position_indices,
    seq_length: Optional[int] = None,
    batch_size: Optional[int] = None,
    # base: int = 10000
):
    """
    Compute actual rotary embeddings from precomputed position indices.
    This function can be used in the model to efficiently compute rotary embeddings
    using the position indices stored in the dataset.

    Args:
        rotary_embedding_module: The RotaryEmbedding module instance
        position_indices: List or tensor of precomputed position indices
                           - For single sequence: shape [seq_length] or [seq_length,]
                           - For batch: shape [batch_size, seq_length]
        base: Base value for rotary embeddings computation

    Returns:
        Rotary position embeddings:
        - For single sequence: shape [seq_length, dim, 2]
        - For batch: shape [batch_size, seq_length, dim, 2]

    Usage example in model:
        # Load precomputed indices from dataset
        position_indices = batch["position_encoding_indices"]  # shape [batch_size, seq_length]

        # Compute rotary embeddings efficiently
        rotary_pos_emb = compute_rotary_embeddings_from_precomputed_indices(
            self.rotary_pos_emb, position_indices
        )
    """

    if position_indices is None:
        if seq_length is None:
            raise ValueError("seq_length must be provided when position_indices is None")
        device = rotary_embedding_module.inv_freq.device
        base_indices = torch.arange(seq_length, dtype=torch.float32, device=device)
        bsz = 1 if batch_size is None else max(int(batch_size), 1)
        position_indices = base_indices.unsqueeze(0).repeat(bsz, 1)

    if isinstance(position_indices, list):
        position_indices = torch.tensor(position_indices, dtype=torch.float32, device=rotary_embedding_module.inv_freq.device)
    else:
        position_indices = position_indices.to(device=rotary_embedding_module.inv_freq.device, dtype=torch.float32)

    # Handle both single sequence and batch cases
    if position_indices.dim() == 1:
        # Single sequence case: [seq_length] -> [1, seq_length]
        position_indices = position_indices.unsqueeze(0)
        batch_size = 1
        seq_length = position_indices.shape[1]
        squeeze_output = True  # Flag to squeeze output back to single sequence
    else:
        # Batch case: [batch_size, seq_length]
        batch_size, seq_length = position_indices.shape
        squeeze_output = False

    # Use inv_freq from the module
    inv_freq = rotary_embedding_module.inv_freq

    # Initialize the final rotary embeddings
    final_rope = torch.zeros(
        (batch_size, seq_length, rotary_embedding_module.dim // 2, 2),
        dtype=rotary_embedding_module.inv_freq.dtype,
        device=rotary_embedding_module.inv_freq.device
    )

    # Vectorized computation for all positions in all batches
    # position_indices: [batch_size, seq_length]
    # inv_freq: [dim//2]
    # theta: [batch_size, seq_length, dim//2]
    theta = position_indices.unsqueeze(-1) * inv_freq.unsqueeze(0).unsqueeze(0)

    # Compute cos and sin embeddings
    final_rope[:, :, :, 0] = torch.cos(theta)
    final_rope[:, :, :, 1] = torch.sin(theta)

    # Squeeze back to single sequence if input was single sequence
    if squeeze_output:
        final_rope = final_rope.squeeze(0)

    return final_rope


def create_rotary_embeddings_from_modality_masks_multiple_modalities(
    rotary_embedding_module: RotaryEmbedding,
    modality_masks: torch.Tensor,
    modality_fps: Dict[int, float] = None,
    primary_modality_idx: int = 0,
    interpolation_type: str = "linear",
    base: int = 10000
) -> torch.Tensor:
    """
    Create rotary position embeddings based on modality masks for multiple modalities with different fps.
    Uses global timestamp sorting to ensure proper temporal alignment across all modalities.

    Key improvements:
    - Global timestamp calculation and sorting for all modalities
    - Ensures modalities 2-5 are in the same time windows (e.g., 3-4s)
    - Proper handling of time conflicts by priority-based ordering
    - Start tokens positioned before first modality 1 token with proper offsets
    - Tokens are sorted globally, then assigned positions following modality 1 sequence

    Args:
        rotary_embedding_module: The RotaryEmbedding module instance
        modality_masks: Tensor of shape [n_modalities, batch_size, seq_length] or [n_modalities, seq_length]
                       where 1 indicates the presence of that modality at that position
        modality_fps: Dictionary mapping modality index to fps value
                     Default: {1: 12.5, 2: 25.0, 3: 6.25, 4: 6.25, 5: 6.25}
        primary_modality_idx: Not used in this version, kept for compatibility
        interpolation_type: Type of interpolation - "linear" or "cubic" (default: "linear")
        base: Base value for rotary embeddings computation

    Returns:
        Rotary position embeddings with shape [seq_length, dim, 2]

    Example:
        # Modality masks for sequence [0,0,0,1,1,2,2,2,2,2,3,4,5]
        modality_masks = torch.tensor([
            [[1,1,1,0,0,0,0,0,0,0,0,0,0]],  # Modality 0
            [[0,0,0,1,1,0,0,0,0,0,0,0,0]],  # Modality 1
            [[0,0,0,0,0,1,1,1,1,1,0,0,0]],  # Modality 2 (face)
            [[0,0,0,0,0,0,0,0,0,0,1,0,0]],  # Modality 3 (upper)
            [[0,0,0,0,0,0,0,0,0,0,0,1,0]],  # Modality 4 (lower)
            [[0,0,0,0,0,0,0,0,0,0,0,0,1]]   # Modality 5 (hand)
        ])
    """
    # Default fps configuration
    if modality_fps is None:
        modality_fps = {1: 12.5, 2: 25.0, 3: 6.25, 4: 6.25, 5: 6.25}

    # Handle both [n_modalities, seq_length] and [n_modalities, batch_size, seq_length]
    if modality_masks.dim() == 3:
        modality_masks = modality_masks[:, 0, :]

    n_modalities, seq_length = modality_masks.shape
    assert n_modalities >= 2, "This function requires at least 2 modalities"

    # Get positions for all modalities
    modality_positions = {}
    for i in range(n_modalities):
        mask = modality_masks[i].bool()
        positions = torch.nonzero(mask, as_tuple=False).flatten()
        if len(positions) > 0:
            modality_positions[i] = positions

    # Combine modality 0 and 1 to form the primary sequence
    primary_positions = []
    if 0 in modality_positions:
        primary_positions.append(modality_positions[0])
    if 1 in modality_positions:
        primary_positions.append(modality_positions[1])

    if primary_positions:
        primary_positions = torch.cat(primary_positions)
        primary_positions, _ = torch.sort(primary_positions)
    else:
        raise ValueError("Primary modalities (0 or 1) must be present")

    if 1 not in modality_positions:
        return None

    mod1_positions = modality_positions[1]
    if len(mod1_positions) == 0:
        return None

    # Generate rotary embeddings for primary sequence
    primary_length = len(primary_positions)
    primary_rope = rotary_embedding_module.forward_impl(
        seq_len=primary_length,
        n_elem=rotary_embedding_module.dim,
        dtype=rotary_embedding_module.inv_freq.dtype,
        device=rotary_embedding_module.inv_freq.device,
        base=base
    )

    # Initialize the final rotary embeddings
    final_rope = torch.zeros(
        (seq_length, rotary_embedding_module.dim // 2, 2),
        dtype=primary_rope.dtype,
        device=primary_rope.device
    )

    # Place primary modality embeddings at their original positions
    final_rope[primary_positions] = primary_rope


    # Prepare for angle-based interpolation
    base_with_ratio = base * rotary_embedding_module.rope_ratio
    inv_freq = 1.0 / (base_with_ratio ** (torch.arange(
        0, rotary_embedding_module.dim, 2, 
        dtype=rotary_embedding_module.inv_freq.dtype, 
        device=rotary_embedding_module.inv_freq.device
    ) / rotary_embedding_module.dim))

    # Process other modalities with global timestamp approach
    if 1 not in modality_positions:
        return final_rope

    if len(mod1_positions) == 0:
        return final_rope

    # Find consecutive groups of modality 1 positions (cycles)
    mod1_groups = _find_consecutive_groups(mod1_positions)

    for group_idx, mod1_group in enumerate(mod1_groups):
        # Get timing information for this mod1 cycle
        first_mod1_pos = mod1_group[0].item()
        first_mod1_idx = (primary_positions == first_mod1_pos).nonzero(as_tuple=False).item()

        # Calculate time duration for this mod1 cycle
        base_fps = modality_fps.get(1, 12.5)
        cycle_duration = len(mod1_group) / base_fps  # Duration in seconds

        # Create a global token list for this cycle with timestamps
        all_cycle_tokens = []

        # Step 1: Process modalities 2-5 and collect all tokens with timestamps
        for modality_idx in range(2, min(6, n_modalities)):
            if modality_idx not in modality_positions:
                continue

            positions = modality_positions[modality_idx]
            fps = modality_fps.get(modality_idx, 12.5)

            # Find positions that belong to this cycle
            cycle_positions = []
            for pos in positions:
                # Determine if this position belongs to current cycle
                if pos.item() >= first_mod1_pos:
                    if group_idx + 1 < len(mod1_groups):
                        next_mod1_pos = mod1_groups[group_idx + 1][0].item()
                        if pos.item() < next_mod1_pos:
                            cycle_positions.append(pos)
                    else:
                        cycle_positions.append(pos)

            if len(cycle_positions) > 0:
                # Add start token (first position) with special timing
                start_pos = cycle_positions[0]
                start_offset = -(6 - modality_idx) / 5.0
                all_cycle_tokens.append({
                    'position': start_pos.item(),
                    'modality': modality_idx,
                    'timestamp': start_offset,  # Negative timestamp for start tokens
                    'is_start_token': True,
                    'priority': modality_idx
                })

                # Add regular tokens with calculated timestamps
                if len(cycle_positions) > 1:
                    regular_positions = cycle_positions[1:]
                    time_per_token = 1.0 / fps

                    for i, pos in enumerate(regular_positions):
                        # Calculate actual timestamp for this token
                        token_timestamp = i * time_per_token
                        # Normalize timestamp to interval scale (K intervals between mod1 anchors)
                        # This ensures proper interval assignment
                        K = len(mod1_group) - 1  # Number of intervals between mod1 anchors
                        normalized_timestamp = (token_timestamp / cycle_duration) * K

                        all_cycle_tokens.append({
                            'position': pos.item(),
                            'modality': modality_idx,
                            'timestamp': normalized_timestamp,
                            'is_start_token': False,
                            'priority': modality_idx
                        })

        # Step 2: Sort all tokens by timestamp, then by modality priority
        # This ensures proper ordering when timestamps are identical
        all_cycle_tokens.sort(key=lambda x: (x['timestamp'], x['priority']))

        # Step 3: Separate start tokens and regular tokens, then uniformly interpolate
        # Start tokens: handle separately with fixed offsets
        start_tokens = [token for token in all_cycle_tokens if token['is_start_token']]
        regular_tokens = [token for token in all_cycle_tokens if not token['is_start_token']]

        # Handle start tokens first
        for token_info in start_tokens:
            pos = token_info['position']
            timestamp = token_info['timestamp']  # This is the negative offset

            # Start tokens: positioned before first mod1 token with fixed offset
            interpolated_position = first_mod1_idx + timestamp
            theta = interpolated_position * inv_freq
            final_rope[pos, :, 0] = torch.cos(theta)
            final_rope[pos, :, 1] = torch.sin(theta)

        # Handle regular tokens with uniform interpolation between mod1 positions
        if len(regular_tokens) > 0 and len(mod1_group) > 0:
            # Create pos→rope mapping to avoid O(N²) lookups
            pos_to_rope_idx = {}
            for i, pos in enumerate(primary_positions):
                pos_to_rope_idx[pos.item()] = i

            # Get the rope indices for this mod1 group using the mapping
            mod1_rope_indices = []
            for mod1_pos in mod1_group:
                rope_idx = pos_to_rope_idx[mod1_pos.item()]
                mod1_rope_indices.append(rope_idx)

            if len(mod1_rope_indices) == 1:
                # Only one mod1 token, place all regular tokens after it
                base_rope_idx = mod1_rope_indices[0]
                for i, token_info in enumerate(regular_tokens):
                    pos = token_info['position']
                    # Uniformly space tokens after the single mod1 token
                    interpolated_position = base_rope_idx + (i + 1) * (1.0 / (len(regular_tokens) + 1))
                    theta = interpolated_position * inv_freq
                    final_rope[pos, :, 0] = torch.cos(theta)
                    final_rope[pos, :, 1] = torch.sin(theta)
            else:
                # Multiple mod1 tokens, need to determine which interval each token belongs to
                # and then uniformly interpolate within each interval

                K = len(mod1_rope_indices) - 1  # Number of intervals between mod1 anchors

                # Group regular tokens by which interval they belong to based on normalized timestamp
                intervals = []  # List of lists, each containing tokens for one interval
                for i in range(K):
                    intervals.append([])
                tail_tokens = []  # Tokens that go beyond the last mod1 anchor (for extrapolation)

                # Assign each regular token to its corresponding interval based on timestamp
                for token_info in regular_tokens:
                    timestamp = token_info['timestamp']

                    # Determine which interval this token belongs to
                    if timestamp < 0:
                        # Should not happen for regular tokens, but handle gracefully
                        interval_idx = 0
                        intervals[interval_idx].append(token_info)
                    elif timestamp >= K:
                        # Beyond the last mod1 anchor, add to tail for extrapolation
                        tail_tokens.append(token_info)
                    else:
                        # Find the appropriate interval based on normalized timestamp
                        interval_idx = int(timestamp)
                        intervals[interval_idx].append(token_info)

                # Process tokens within intervals (interpolation)
                for interval_idx, interval_tokens in enumerate(intervals):
                    if len(interval_tokens) > 0:
                        start_rope_idx = mod1_rope_indices[interval_idx]
                        end_rope_idx = mod1_rope_indices[interval_idx + 1]

                        # Uniformly distribute tokens in this interval
                        for i, token_info in enumerate(interval_tokens):
                            pos = token_info['position']
                            # Uniform interpolation within the interval
                            alpha = (i + 1) / (len(interval_tokens) + 1)
                            interpolated_position = start_rope_idx + alpha * (end_rope_idx - start_rope_idx)

                            theta = interpolated_position * inv_freq
                            final_rope[pos, :, 0] = torch.cos(theta)
                            final_rope[pos, :, 1] = torch.sin(theta)

                # Process tail tokens (extrapolation beyond last mod1 anchor)
                if len(tail_tokens) > 0:
                    # Use the last interval's span for extrapolation
                    last_start_rope_idx = mod1_rope_indices[-2]
                    last_end_rope_idx = mod1_rope_indices[-1]
                    interval_span = last_end_rope_idx - last_start_rope_idx

                    # Extrapolate tokens beyond the last mod1 anchor
                    for i, token_info in enumerate(tail_tokens):
                        pos = token_info['position']
                        # Extrapolate using the last interval's span
                        extrapolation_offset = (i + 1) * (interval_span / (len(tail_tokens) + 1))
                        interpolated_position = last_end_rope_idx + extrapolation_offset

                        theta = interpolated_position * inv_freq
                        final_rope[pos, :, 0] = torch.cos(theta)
                        final_rope[pos, :, 1] = torch.sin(theta)

    return final_rope


def _find_consecutive_groups(positions: torch.Tensor) -> List[torch.Tensor]:
    """
    Find groups of consecutive positions in a sorted tensor.

    Args:
        positions: Sorted tensor of positions

    Returns:
        List of tensors, each containing a group of consecutive positions
    """
    if len(positions) == 0:
        return []

    groups = []
    current_group = [positions[0]]

    for i in range(1, len(positions)):
        if positions[i] == positions[i-1] + 1:
            current_group.append(positions[i])
        else:
            groups.append(torch.tensor(current_group, device=positions.device))
            current_group = [positions[i]]

    groups.append(torch.tensor(current_group, device=positions.device))
    return groups


class DummyRotaryEmbedding:
    """
    Dummy RotaryEmbedding class for position encoding calculation during preprocessing.
    This mimics the interface needed by the position encoding function.
    """
    def __init__(self, dim=4096, rope_ratio=1.0):
        self.dim = dim
        self.rope_ratio = rope_ratio
        # Create dummy inv_freq with appropriate dtype and device
        self.inv_freq = torch.ones(dim // 2, dtype=torch.float32)

    def forward_impl(self, seq_len, n_elem, dtype, device, base=10000):
        """
        Dummy implementation that returns position indices instead of actual embeddings.
        This is used during preprocessing to compute position mappings.
        """
        # Return simple position indices as a placeholder
        # The actual rotary embeddings will be computed at runtime using these indices
        positions = torch.arange(seq_len, dtype=dtype, device=device).unsqueeze(-1).unsqueeze(-1)
        return positions.expand(-1, n_elem // 2, 2)


def calculate_position_encoding_indices(modality_masks, modality_fps=None):
    """
    Calculate position encoding indices for each token based on modality masks.
    This function extracts the position mapping logic from the rotary embedding computation
    and returns the interpolated position indices that can be stored in the dataset.

    Args:
        modality_masks: Tensor of modality masks [n_modalities, batch_size, seq_length] or
                       List of modality masks [mask_0, mask_1, mask_2, mask_3, mask_4, mask_5]
                       where each mask is a list of booleans indicating token presence
        modality_fps: Dictionary mapping modality index to fps value
                     Default: {1: 12.5, 2: 25.0, 3: 6.25, 4: 6.25, 5: 6.25}

    Returns:
        List of float values representing the interpolated position index for each token.
        These indices can be used later to compute the actual rotary embeddings efficiently.
    """
    if modality_fps is None:
        # modality_fps = {1: 12.5, 2: 25.0, 3: 6.25, 4: 6.25, 5: 6.25}
        modality_fps = {1: 12.5, 2: 43.75}

    # Handle both tensor and list formats
    if isinstance(modality_masks, torch.Tensor):
        # Tensor format: [n_modalities, batch_size, seq_length]
        # Use the first batch for calculation
        modality_masks_tensor = modality_masks[:, 0, :].bool()
    else:
        # List format: convert to tensor
        seq_length = len(modality_masks[0])
        n_modalities = len(modality_masks)
        modality_masks_tensor = torch.zeros((n_modalities, seq_length), dtype=torch.bool)
        for i, mask in enumerate(modality_masks):
            modality_masks_tensor[i] = torch.tensor(mask, dtype=torch.bool)

    # Create dummy rotary embedding module for position calculation
    dummy_rotary = DummyRotaryEmbedding()

    # Compute position mappings using the same logic as the model
    position_indices = extract_position_indices_from_rotary_computation(
        dummy_rotary, modality_masks_tensor, modality_fps
    )

    return position_indices.tolist()


def extract_position_indices_from_rotary_computation(rotary_embedding_module, modality_masks, modality_fps):
    """
    Extract position indices from the rotary embedding computation without computing actual embeddings.
    This function follows the same logic as create_rotary_embeddings_from_modality_masks_multiple_modalities
    but only tracks the position indices.
    """
    # Handle batch dimension
    if modality_masks.dim() == 3:
        modality_masks = modality_masks[:, 0, :]

    n_modalities, seq_length = modality_masks.shape

    # Initialize position indices array
    position_indices = torch.arange(seq_length, dtype=torch.float32)

    # Get positions for all modalities
    modality_positions = {}
    for i in range(n_modalities):
        mask = modality_masks[i].bool()
        positions = torch.nonzero(mask, as_tuple=False).flatten()
        if len(positions) > 0:
            modality_positions[i] = positions

    # Combine modality 0 and 1 to form the primary sequence
    primary_positions = []
    if 0 in modality_positions:
        primary_positions.append(modality_positions[0])
    if 1 in modality_positions:
        primary_positions.append(modality_positions[1])

    if not primary_positions:
        return position_indices

    primary_positions = torch.cat(primary_positions)
    primary_positions, _ = torch.sort(primary_positions)

    if 1 not in modality_positions:
        return position_indices

    mod1_positions = modality_positions[1]
    if len(mod1_positions) == 0:
        return position_indices

    # Set primary position indices (these remain as their sequential positions)
    for i, pos in enumerate(primary_positions):
        position_indices[pos] = float(i)

    # Create pos→rope mapping for fast lookups
    pos_to_rope_idx = {}
    for i, pos in enumerate(primary_positions):
        pos_to_rope_idx[pos.item()] = i

    # Find consecutive groups of modality 1 positions (cycles)
    mod1_groups = _find_consecutive_groups(mod1_positions)

    for group_idx, mod1_group in enumerate(mod1_groups):
        # Get timing information for this mod1 cycle
        first_mod1_pos = mod1_group[0].item()
        first_mod1_idx = pos_to_rope_idx[first_mod1_pos]

        # Calculate time duration for this mod1 cycle
        base_fps = modality_fps.get(1, 12.5)
        cycle_duration = len(mod1_group) / base_fps  # Duration in seconds

        # Create a global token list for this cycle with timestamps
        all_cycle_tokens = []

        # Process modalities 2-5 and collect all tokens with timestamps
        for modality_idx in range(2, min(3, n_modalities)):
            if modality_idx not in modality_positions:
                continue

            positions = modality_positions[modality_idx]
            fps = modality_fps.get(modality_idx, 12.5)

            # Find positions that belong to this cycle
            cycle_positions = []
            for pos in positions:
                if pos.item() >= first_mod1_pos:
                    if group_idx + 1 < len(mod1_groups):
                        next_mod1_pos = mod1_groups[group_idx + 1][0].item()
                        if pos.item() < next_mod1_pos:
                            cycle_positions.append(pos)
                    else:
                        cycle_positions.append(pos)

            if len(cycle_positions) > 0:
                # Add start token (first position) with special timing
                start_pos = cycle_positions[0]
                # For unified motion modality (modality 2), use -0.5 offset for begin_of_motion token
                start_offset = -0.5

                all_cycle_tokens.append({
                    'position': start_pos.item(),
                    'modality': modality_idx,
                    'timestamp': start_offset,
                    'is_start_token': True,
                    'priority': modality_idx
                })

                # Add regular tokens with calculated timestamps
                if len(cycle_positions) > 1:
                    regular_positions = cycle_positions[1:]
                    time_per_token = 1.0 / fps

                    for i, pos in enumerate(regular_positions):
                        token_timestamp = i * time_per_token
                        K = len(mod1_group) - 1  # Number of intervals between mod1 anchors
                        normalized_timestamp = (token_timestamp / cycle_duration) * K

                        all_cycle_tokens.append({
                            'position': pos.item(),
                            'modality': modality_idx,
                            'timestamp': normalized_timestamp,
                            'is_start_token': False,
                            'priority': modality_idx
                        })

        # Sort all tokens by timestamp, then by modality priority
        all_cycle_tokens.sort(key=lambda x: (x['timestamp'], x['priority']))

        # Separate start tokens and regular tokens
        start_tokens = [token for token in all_cycle_tokens if token['is_start_token']]
        regular_tokens = [token for token in all_cycle_tokens if not token['is_start_token']]

        # Handle start tokens first
        for token_info in start_tokens:
            pos = token_info['position']
            timestamp = token_info['timestamp']  # This is the negative offset

            # Start tokens: positioned before first mod1 token with fixed offset
            interpolated_position = first_mod1_idx + timestamp
            position_indices[pos] = interpolated_position

        # Handle regular tokens with uniform interpolation between mod1 positions
        if len(regular_tokens) > 0 and len(mod1_group) > 0:
            # Get mod1 rope indices
            mod1_rope_indices = []
            for mod1_pos in mod1_group:
                rope_idx = pos_to_rope_idx[mod1_pos.item()]
                mod1_rope_indices.append(rope_idx)

            if len(mod1_rope_indices) == 1:
                # Only one mod1 token, place all regular tokens after it
                base_rope_idx = mod1_rope_indices[0]
                for i, token_info in enumerate(regular_tokens):
                    pos = token_info['position']
                    interpolated_position = base_rope_idx + (i + 1) * (1.0 / (len(regular_tokens) + 1))
                    position_indices[pos] = interpolated_position
            else:
                # Multiple mod1 tokens, determine interval assignment
                K = len(mod1_rope_indices) - 1  # Number of intervals between mod1 anchors

                # Group regular tokens by interval
                intervals = []
                for i in range(K):
                    intervals.append([])
                tail_tokens = []

                # Assign each regular token to its corresponding interval
                for token_info in regular_tokens:
                    timestamp = token_info['timestamp']

                    if timestamp < 0:
                        interval_idx = 0
                        intervals[interval_idx].append(token_info)
                    elif timestamp >= K:
                        tail_tokens.append(token_info)
                    else:
                        interval_idx = int(timestamp)
                        intervals[interval_idx].append(token_info)

                # Process tokens within intervals (interpolation)
                for interval_idx, interval_tokens in enumerate(intervals):
                    if len(interval_tokens) > 0:
                        start_rope_idx = mod1_rope_indices[interval_idx]
                        end_rope_idx = mod1_rope_indices[interval_idx + 1]

                        # Uniformly distribute tokens in this interval
                        for i, token_info in enumerate(interval_tokens):
                            pos = token_info['position']
                            alpha = (i + 1) / (len(interval_tokens) + 1)
                            interpolated_position = start_rope_idx + alpha * (end_rope_idx - start_rope_idx)
                            position_indices[pos] = interpolated_position

                # Process tail tokens (extrapolation beyond last mod1 anchor)
                if len(tail_tokens) > 0:
                    last_start_rope_idx = mod1_rope_indices[-2]
                    last_end_rope_idx = mod1_rope_indices[-1]
                    interval_span = last_end_rope_idx - last_start_rope_idx

                    for i, token_info in enumerate(tail_tokens):
                        pos = token_info['position']
                        extrapolation_offset = (i + 1) * (interval_span / (len(tail_tokens) + 1))
                        interpolated_position = last_end_rope_idx + extrapolation_offset
                        position_indices[pos] = interpolated_position

    return position_indices


def extract_insertion_positions_from_modality_data(
    modality_data: List[Dict],
    primary_modality_idx: int = 0
) -> List[Tuple[int, int, int]]:
    """
    Extract insertion positions from modality data generated by create_batch_aware_modality_inputs_labels.

    This function analyzes the position mappings to determine where secondary modality tokens
    should be inserted relative to the primary modality.

    Args:
        modality_data: List of dictionaries containing 'original_positions' for each modality
        primary_modality_idx: Index of the primary modality (default: 0)

    Returns:
        List of tuples (start_pos, end_pos, num_tokens) for insertion positions

    Example:
        # Given modality positions like:
        # Modality 0: [0,1,2,3,6,7,8,9]  (primary)
        # Modality 1: [4,5]               (secondary)
        # Returns: [(3, 4, 2)] - insert 2 tokens between primary positions 3 and 4
    """
    if len(modality_data) < 2:
        return []

    # Get primary modality positions (assuming first batch)
    primary_positions = modality_data[primary_modality_idx]['original_positions'][0]

    insertion_positions = []

    # Process each secondary modality
    for modality_idx in range(len(modality_data)):
        if modality_idx == primary_modality_idx:
            continue

        # Get secondary modality positions (assuming first batch)
        secondary_positions = modality_data[modality_idx]['original_positions'][0]

        if len(secondary_positions) == 0:
            continue

        # Find consecutive groups in secondary positions
        groups = _find_consecutive_groups(secondary_positions)

        for group in groups:
            group_start = group[0].item()
            group_end = group[-1].item()
            num_tokens = len(group)

            # Find the primary positions that surround this group
            before_positions = primary_positions[primary_positions < group_start]
            after_positions = primary_positions[primary_positions > group_end]

            if len(before_positions) > 0 and len(after_positions) > 0:
                # Find the insertion point in primary modality sequence
                insert_after = before_positions[-1].item()
                insert_before = after_positions[0].item()

                # Find the index in primary sequence
                primary_insert_idx = (primary_positions == insert_after).nonzero(as_tuple=False).item()

                insertion_positions.append((primary_insert_idx, primary_insert_idx + 1, num_tokens))

    return insertion_positions


class RMSNorm(torch.nn.Module):
    def __init__(self, normalized_shape, eps=1e-5, device=None, dtype=None, **kwargs):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(normalized_shape, device=device, dtype=dtype))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor):
        """
        Apply RMS normalization to hidden states.
        
        Args:
            hidden_states: Input tensor [..., hidden_size]
            
        Returns:
            Normalized tensor with same shape as input
        """
        input_dtype = hidden_states.dtype
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)

        return (self.weight * hidden_states).to(input_dtype)


class CoreAttention(torch.nn.Module):
    def __init__(self, config: ChatGLMConfig, layer_number):
        super(CoreAttention, self).__init__()
        self.config = config
        self.apply_query_key_layer_scaling = config.apply_query_key_layer_scaling
        self.attention_softmax_in_fp32 = config.attention_softmax_in_fp32
        if self.apply_query_key_layer_scaling:
            self.attention_softmax_in_fp32 = True
        self.layer_number = max(1, layer_number)
        self.is_causal = True

        projection_size = config.kv_channels * config.num_attention_heads

        # Per attention head and per partition values.
        self.hidden_size_per_partition = projection_size
        self.hidden_size_per_attention_head = projection_size // config.num_attention_heads
        self.num_attention_heads_per_partition = config.num_attention_heads

        coeff = None
        self.norm_factor = math.sqrt(self.hidden_size_per_attention_head)
        if self.apply_query_key_layer_scaling:
            coeff = self.layer_number
            self.norm_factor *= coeff
        self.coeff = coeff

        self.attention_dropout = torch.nn.Dropout(config.attention_dropout)

    def forward(self, query_layer, key_layer, value_layer, attention_mask):
        """
        Core attention computation.
        
        Args:
            query_layer: Query tensor [batch, num_heads, seq_len_q, head_dim]
            key_layer: Key tensor [batch, num_heads, seq_len_k, head_dim]
            value_layer: Value tensor [batch, num_heads, seq_len_k, head_dim]
            attention_mask: Attention mask tensor
            
        Returns:
            Context layer after attention computation
        """
        # [batch, num_heads, seq_len_q, seq_len_k]
        output_size = (query_layer.size(0), query_layer.size(1), query_layer.size(2), key_layer.size(2))

        # [b, np, sq, hn] -> [b * np, sq, hn]
        query_layer = query_layer.view(output_size[0] * output_size[1], output_size[2], -1)
        # [b, np, sk, hn] -> [b * np, sk, hn]
        key_layer = key_layer.view(output_size[0] * output_size[1], output_size[3], -1)

        # preallocting input tensor: [b * np, sq, sk]
        matmul_input_buffer = torch.empty(
            output_size[0] * output_size[1], output_size[2], output_size[3], dtype=query_layer.dtype,
            device=query_layer.device
        )

        # Raw attention scores. [b * np, sq, sk]
        matmul_result = torch.baddbmm(
            matmul_input_buffer,
            query_layer,  # [b * np, sq, hn]
            key_layer.transpose(1, 2),  # [b * np, hn, sk]
            beta=0.0,
            alpha=(1.0 / self.norm_factor),
        )

        # Change view to [batch, num_heads, seq_len, seq_len]
        attention_scores = matmul_result.view(*output_size)

        # Apply attention mask and compute attention probabilities with dropout
        # attention scores and attention mask [batch, num_heads, seq_len, seq_len]
        if self.attention_softmax_in_fp32:
            attention_scores = attention_scores.float()
        if self.coeff is not None:
            attention_scores = attention_scores * self.coeff
        if attention_mask is None and attention_scores.shape[2] == attention_scores.shape[3]:
            attention_mask = torch.ones(output_size[0], 1, output_size[2], output_size[3],
                                        device=attention_scores.device, dtype=torch.bool)
            attention_mask.tril_()
            attention_mask = ~attention_mask
        if attention_mask is not None:
            attention_scores = attention_scores.masked_fill(attention_mask, float("-inf"))
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = attention_probs.type_as(value_layer)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.attention_dropout(attention_probs)

        # query layer shape: [b * np, sq, hn]
        # value layer shape: [b, np, sk, hn]
        # attention shape: [b, np, sq, sk]
        # context layer shape: [b, np, sq, hn]
        output_size = (value_layer.size(0), value_layer.size(1), query_layer.size(1), value_layer.size(3))
        # change view [b * np, sk, hn]
        value_layer = value_layer.view(output_size[0] * output_size[1], value_layer.size(2), -1)
        # change view [b * np, sq, sk]
        attention_probs = attention_probs.view(output_size[0] * output_size[1], output_size[2], -1)
        # matmul: [b * np, sq, hn]
        context_layer = torch.bmm(attention_probs, value_layer)
        # change view [b, np, sq, hn]
        context_layer = context_layer.view(*output_size)
        # [b, np, sq, hn] --> [b, sq, np, hn]
        context_layer = context_layer.transpose(1, 2).contiguous()
        # [b, sq, np, hn] --> [b, sq, hp]
        new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size_per_partition,)
        context_layer = context_layer.reshape(*new_context_layer_shape)

        return context_layer


class SdpaAttention(CoreAttention):
    def forward(self, query_layer, key_layer, value_layer, attention_mask):
        """
        Forward pass using scaled dot-product attention (SDPA).
        
        Args:
            query_layer: Query tensor
            key_layer: Key tensor
            value_layer: Value tensor
            attention_mask: Optional attention mask
            
        Returns:
            context_layer: Output context tensor after attention
        """
        if attention_mask is None and query_layer.shape[2] == key_layer.shape[2]:
            context_layer = torch.nn.functional.scaled_dot_product_attention(query_layer, key_layer, value_layer,
                                                                             is_causal=True,
                                                                             dropout_p=self.config.attention_dropout if self.training else 0.0)
        else:
            if attention_mask is not None:
                attention_mask = ~attention_mask
            context_layer = torch.nn.functional.scaled_dot_product_attention(query_layer, key_layer, value_layer,
                                                                             attention_mask,
                                                                             dropout_p=self.config.attention_dropout if self.training else 0.0)
        context_layer = context_layer.transpose(1, 2).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size_per_partition,)
        context_layer = context_layer.reshape(*new_context_layer_shape)
        return context_layer


def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


# Copied from transformers.models.llama.modeling_llama.LlamaFlashAttention2
class FlashAttention2(CoreAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    def forward(self, query_states, key_states, value_states, attention_mask):
        """
        Forward pass using Flash Attention 2.
        
        Args:
            query_states: Query tensor
            key_states: Key tensor
            value_states: Value tensor
            attention_mask: Optional attention mask
            
        Returns:
            attn_output: Output tensor after flash attention
        """
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)
        batch_size, query_length = query_states.shape[:2]
        if not self._flash_attn_uses_top_left_mask:
            causal = self.is_causal
        else:
            # TODO: Remove the `query_length != 1` check once Flash Attention for RoCm is bumped to 2.1. For details, please see the comment in LlamaFlashAttention2 __init__.
            causal = self.is_causal and query_length != 1
        dropout = self.config.attention_dropout if self.training else 0.0
        # Contains at least one padding token in the sequence
        if attention_mask is not None:
            query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
                query_states, key_states, value_states, attention_mask, query_length
            )

            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

            attn_output_unpad = flash_attn_varlen_func(
                query_states,
                key_states,
                value_states,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_in_batch_q,
                max_seqlen_k=max_seqlen_in_batch_k,
                dropout_p=dropout,
                softmax_scale=None,
                causal=causal,
            )

            attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
        else:
            attn_output = flash_attn_func(
                query_states, key_states, value_states, dropout, softmax_scale=None, causal=causal
            )
        attn_output = attn_output.reshape(batch_size, query_length, self.hidden_size_per_partition).contiguous()
        return attn_output

    def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
        batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

        key_layer = index_first_axis(
            key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        value_layer = index_first_axis(
            value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        if query_length == kv_seq_len:
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, self.num_attention_heads_per_partition, head_dim),
                indices_k
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )  # There is a memcpy here, that is very bad.
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            # The -q_len: slice assumes left padding.
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)

        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )


CORE_ATTENTION_CLASSES = {
    "eager": CoreAttention,
    "sdpa": SdpaAttention,
    "flash_attention_2": FlashAttention2
}


class SelfAttention(torch.nn.Module):
    """Parallel self-attention layer abstract class.

    Self-attention layer takes input with size [s, b, h]
    and returns output of the same size.
    """

    def __init__(self, config: ChatGLMConfig, layer_number, device=None):
        super(SelfAttention, self).__init__()
        self.layer_number = max(1, layer_number)
        self.config = config

        self.projection_size = config.kv_channels * config.num_attention_heads

        # Per attention head and per partition values.
        self.hidden_size_per_attention_head = self.projection_size // config.num_attention_heads
        self.num_attention_heads_per_partition = config.num_attention_heads

        self.multi_query_attention = config.multi_query_attention
        self.qkv_hidden_size = 3 * self.projection_size
        if self.multi_query_attention:
            self.num_multi_query_groups_per_partition = config.multi_query_group_num
            self.qkv_hidden_size = (
                    self.projection_size + 2 * self.hidden_size_per_attention_head * config.multi_query_group_num
            )
        self.query_key_value = nn.Linear(config.hidden_size, self.qkv_hidden_size,
                                         bias=config.add_bias_linear or config.add_qkv_bias,
                                         device=device, **_config_to_kwargs(config)
                                         )

        self.core_attention = CORE_ATTENTION_CLASSES[config._attn_implementation](config, self.layer_number)

        # Output.
        self.dense = nn.Linear(self.projection_size, config.hidden_size, bias=config.add_bias_linear,
                               device=device, **_config_to_kwargs(config)
                               )

    def set_attention_implementation(self, attn_impl: str):
        """Swap the core attention backend at runtime."""
        self.config._attn_implementation = attn_impl
        self.core_attention = CORE_ATTENTION_CLASSES[attn_impl](self.config, self.layer_number)

    def _allocate_memory(self, inference_max_sequence_len, batch_size, device=None, dtype=None):
        if self.multi_query_attention:
            num_attention_heads = self.num_multi_query_groups_per_partition
        else:
            num_attention_heads = self.num_attention_heads_per_partition
        return torch.empty(
            inference_max_sequence_len,
            batch_size,
            num_attention_heads,
            self.hidden_size_per_attention_head,
            dtype=dtype,
            device=device,
        )

    def forward(
            self, hidden_states, attention_mask, rotary_pos_emb, kv_cache=None, use_cache=True
    ):
        """
        Forward pass for self-attention layer.
        
        Args:
            hidden_states: Input hidden states [batch_size, seq_len, hidden_size]
            attention_mask: Attention mask tensor
            rotary_pos_emb: Rotary positional embeddings
            kv_cache: Key-value cache for efficient inference
            use_cache: Whether to use caching
            
        Returns:
            output: Output tensor after attention
            kv_cache: Updated key-value cache
        """
        # Pre-allocate memory for key-values for inference
        # Compute query, key, and value projections: [batch, seq_len, hidden] --> [batch, seq_len, (num_heads * 3 * head_dim)]
        mixed_x_layer = self.query_key_value(hidden_states)

        if self.multi_query_attention:
            (query_layer, key_layer, value_layer) = mixed_x_layer.split(
                [
                    self.num_attention_heads_per_partition * self.hidden_size_per_attention_head,
                    self.num_multi_query_groups_per_partition * self.hidden_size_per_attention_head,
                    self.num_multi_query_groups_per_partition * self.hidden_size_per_attention_head,
                ],
                dim=-1,
            )
            query_layer = query_layer.view(
                query_layer.size()[:-1] + (self.num_attention_heads_per_partition, self.hidden_size_per_attention_head)
            )
            key_layer = key_layer.view(
                key_layer.size()[:-1] + (self.num_multi_query_groups_per_partition, self.hidden_size_per_attention_head)
            )
            value_layer = value_layer.view(
                value_layer.size()[:-1]
                + (self.num_multi_query_groups_per_partition, self.hidden_size_per_attention_head)
            )
        else:
            new_tensor_shape = mixed_x_layer.size()[:-1] + \
                               (self.num_attention_heads_per_partition,
                                3 * self.hidden_size_per_attention_head)
            mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)

            # [b, sq, np, 3 * hn] --> 3 [b, sq, np, hn]
            (query_layer, key_layer, value_layer) = split_tensor_along_last_dim(mixed_x_layer, 3)

        # [b, sq, np, hn] -> [b, np, sq, hn]
        query_layer, key_layer, value_layer = [k.transpose(1, 2) for k in [query_layer, key_layer, value_layer]]

        # apply relative positional encoding (rotary embedding)
        if rotary_pos_emb is not None:
            query_layer = apply_rotary_pos_emb(query_layer, rotary_pos_emb)
            key_layer = apply_rotary_pos_emb(key_layer, rotary_pos_emb)

        # adjust key and value for inference
        if kv_cache is not None:
            cache_k, cache_v = kv_cache
            key_layer = torch.cat((cache_k, key_layer), dim=2)
            value_layer = torch.cat((cache_v, value_layer), dim=2)
        if use_cache:
            if kv_cache is None:
                kv_cache = torch.cat((key_layer.unsqueeze(0).unsqueeze(0), value_layer.unsqueeze(0).unsqueeze(0)),
                                     dim=1)
            else:
                kv_cache = (key_layer, value_layer)
        else:
            kv_cache = None

        if self.multi_query_attention:
            key_layer = key_layer.unsqueeze(2)
            key_layer = key_layer.expand(
                -1, -1, self.num_attention_heads_per_partition // self.num_multi_query_groups_per_partition, -1, -1
            )
            key_layer = key_layer.contiguous().view(
                key_layer.size()[:1] + (self.num_attention_heads_per_partition,) + key_layer.size()[3:]
            )
            value_layer = value_layer.unsqueeze(2)
            value_layer = value_layer.expand(
                -1, -1, self.num_attention_heads_per_partition // self.num_multi_query_groups_per_partition, -1, -1
            )
            value_layer = value_layer.contiguous().view(
                value_layer.size()[:1] + (self.num_attention_heads_per_partition,) + value_layer.size()[3:]
            )

        # Normalize attention mask to match current query/key lengths
        if attention_mask is not None:
            seq_q = query_layer.size(2)
            seq_k = key_layer.size(2)

            attn_impl = getattr(self.core_attention, "__class__", None)
            # Flash attention expects 2D key padding mask
            if isinstance(self.core_attention, FlashAttention2):
                if attention_mask.dim() == 4:
                    attention_mask = attention_mask.squeeze(1)
                if attention_mask.dim() == 3:
                    attention_mask = attention_mask[:, 0]
                attention_mask = attention_mask[:, :seq_k].to(dtype=torch.bool, device=key_layer.device)
            else:
                if attention_mask.dim() == 2:
                    pad_q = attention_mask[:, :seq_q].to(dtype=torch.bool, device=query_layer.device)
                    pad_k = attention_mask[:, :seq_k].to(dtype=torch.bool, device=key_layer.device)
                    causal = torch.ones(seq_q, seq_k, dtype=torch.bool, device=query_layer.device).tril_()
                    allowed = pad_q.unsqueeze(-1) & pad_k.unsqueeze(-2) & causal
                    attention_mask = (~allowed).unsqueeze(1)
                else:
                    attention_mask = attention_mask[:, :, :seq_q, :seq_k].to(dtype=torch.bool, device=query_layer.device)

        # Core attention computation
        context_layer = self.core_attention(query_layer, key_layer, value_layer, attention_mask)

        # Output projection: [seq_len, batch_size, hidden_size]

        output = self.dense(context_layer)

        return output, kv_cache


class ModalityUntiedSelfAttention(torch.nn.Module):
    """Parallel self-attention layer abstract class.

    Self-attention layer takes input with size [s, b, h]
    and returns output of the same size.
    """

    def __init__(self, config: ChatGLMConfig, layer_number, device=None):
        super(ModalityUntiedSelfAttention, self).__init__()
        self.layer_number = max(1, layer_number)
        self.config = config
        self.n_modalities = 2
        self.batch_processor = ModalityBatchProcessor(self.n_modalities)
        self.projection_size = config.kv_channels * config.num_attention_heads

        # Per attention head and per partition values.
        self.hidden_size_per_attention_head = self.projection_size // config.num_attention_heads
        self.num_attention_heads_per_partition = config.num_attention_heads
        self.multi_query_attention = config.multi_query_attention
        self.qkv_hidden_size = 3 * self.projection_size
        if self.multi_query_attention:
            self.num_multi_query_groups_per_partition = config.multi_query_group_num
            self.qkv_hidden_size = (
                    self.projection_size + 2 * self.hidden_size_per_attention_head * config.multi_query_group_num
            )

        self.query_key_value = torch.nn.ModuleList(
                    [
                        nn.Linear(config.hidden_size, self.qkv_hidden_size,
                        bias=config.add_bias_linear or config.add_qkv_bias,
                        device=device, **_config_to_kwargs(config)
                        ),
                        nn.Linear(config.mot_settings['mot_hidden_size'], self.qkv_hidden_size,
                        bias=config.add_bias_linear or config.add_qkv_bias,
                        device=device, **_config_to_kwargs(config)
                        )
                    ]
                )
        self.core_attention = CORE_ATTENTION_CLASSES[config._attn_implementation](config, self.layer_number)

        # Output projection layer
        self.dense = torch.nn.ModuleList(
                    [
                        nn.Linear(self.projection_size, config.hidden_size, bias=config.add_bias_linear,
                                                device=device, **_config_to_kwargs(config)
                                                ),
                        nn.Linear(self.projection_size, config.mot_settings['mot_hidden_size'], bias=config.add_bias_linear,
                                                device=device, **_config_to_kwargs(config)
                                                )
                    ]
                )

    def set_attention_implementation(self, attn_impl: str):
        self.config._attn_implementation = attn_impl
        self.core_attention = CORE_ATTENTION_CLASSES[attn_impl](self.config, self.layer_number)

    def _process_qkv(self, x, modality_masks):
        """
        Process query, key, and value projections for each modality.
        """
        expert_outputs_xq, expert_outputs_xk, expert_outputs_xv = [], [], []
        for i in range(self.n_modalities):
            expert_input = x[modality_masks[i]]
            xq = self.query_key_value[i](expert_input)
            xk = self.query_key_value[i](expert_input)
            xv = self.query_key_value[i](expert_input)

            expert_outputs_xq.append(xq)
            expert_outputs_xk.append(xk)
            expert_outputs_xv.append(xv)

        return expert_outputs_xq, expert_outputs_xk, expert_outputs_xv

    def _allocate_memory(self, inference_max_sequence_len, batch_size, device=None, dtype=None):
        if self.multi_query_attention:
            num_attention_heads = self.num_multi_query_groups_per_partition
        else:
            num_attention_heads = self.num_attention_heads_per_partition
        return torch.empty(
            inference_max_sequence_len,
            batch_size,
            num_attention_heads,
            self.hidden_size_per_attention_head,
            dtype=dtype,
            device=device,
        )


    def forward(
            self, hidden_states, attention_mask, rotary_pos_emb, kv_cache=None, use_cache=True,
            modality_masks=None, modality_masks_full=None, past_key_values=None
    ):
        # Pre-allocate memory for key-values for inference.
        # Query, Key, and Value

        if isinstance(kv_cache, tuple):
            kv_cache = list(kv_cache)

        if past_key_values is not None and isinstance(past_key_values, (list, tuple)):
            for cache_idx in range(min(len(past_key_values), len(kv_cache))):
                cache_entry = past_key_values[cache_idx]
                if cache_entry is None:
                    continue
                kv_cache[cache_idx] = cache_entry

        mask_for_mapping = (
            modality_masks_full
            if modality_masks_full is not None
            else modality_masks
        )

        mixed_x_layer = []
        for i in range(len(hidden_states)):
            if hidden_states[i] != []:
                expert_output = self.query_key_value[i](hidden_states[i])
                mixed_x_layer.append(expert_output)
            else:
                mixed_x_layer.append([])

        query_layer_list, key_layer_list, value_layer_list = [], [], []

        for i in range(len(mixed_x_layer)):


            if modality_masks[i].sum() == 0:
                if i < len(kv_cache) and kv_cache[i] is not None:
                    cache_k, cache_v = kv_cache[i]
                    key_layer = cache_k
                    value_layer = cache_v
                    if self.multi_query_attention:
                        key_layer = key_layer.unsqueeze(2)
                        key_layer = key_layer.expand(
                            -1, -1, self.num_attention_heads_per_partition // self.num_multi_query_groups_per_partition, -1, -1
                        )
                        key_layer = key_layer.contiguous().view(
                            key_layer.size()[:1] + (self.num_attention_heads_per_partition,) + key_layer.size()[3:]
                        )
                        value_layer = value_layer.unsqueeze(2)
                        value_layer = value_layer.expand(
                            -1, -1, self.num_attention_heads_per_partition // self.num_multi_query_groups_per_partition, -1, -1
                        )
                        value_layer = value_layer.contiguous().view(
                            value_layer.size()[:1] + (self.num_attention_heads_per_partition,) + value_layer.size()[3:]
                        )
                    query_layer_list.append(None)
                    key_layer_list.append(key_layer)
                    value_layer_list.append(value_layer)
                else:
                    query_layer_list.append(None)
                    key_layer_list.append(None)
                    value_layer_list.append(None)
                continue

            if self.multi_query_attention:

                (query_layer, key_layer, value_layer) = mixed_x_layer[i].split(
                    [
                        self.num_attention_heads_per_partition * self.hidden_size_per_attention_head,
                        self.num_multi_query_groups_per_partition * self.hidden_size_per_attention_head,
                        self.num_multi_query_groups_per_partition * self.hidden_size_per_attention_head,
                    ],
                    dim=-1,
                )
                query_layer = query_layer.view(
                    query_layer.size()[:-1] + (self.num_attention_heads_per_partition, self.hidden_size_per_attention_head)
                )
                key_layer = key_layer.view(
                    key_layer.size()[:-1] + (self.num_multi_query_groups_per_partition, self.hidden_size_per_attention_head)
                )
                value_layer = value_layer.view(
                    value_layer.size()[:-1]
                    + (self.num_multi_query_groups_per_partition, self.hidden_size_per_attention_head)
                )
            else:

                # TODO: modify this part
                new_tensor_shape = mixed_x_layer.size()[:-1] + \
                                (self.num_attention_heads_per_partition,
                                    3 * self.hidden_size_per_attention_head)
                mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)

                # [b, sq, np, 3 * hn] --> 3 [b, sq, np, hn]
                (query_layer, key_layer, value_layer) = split_tensor_along_last_dim(mixed_x_layer, 3)

            # [b, sq, np, hn] -> [b, np, sq, hn]
            query_layer, key_layer, value_layer = [k.transpose(1, 2) for k in [query_layer, key_layer, value_layer]]

            # apply relative positional encoding (rotary embedding)
            if rotary_pos_emb is not None:
                if modality_masks_full is not None:
                    query_layer = apply_rotary_pos_emb_batch(query_layer, rotary_pos_emb)
                    key_layer = apply_rotary_pos_emb_batch(key_layer, rotary_pos_emb)
                else:
                    query_layer = apply_rotary_pos_emb_batch(query_layer, rotary_pos_emb)
                    key_layer = apply_rotary_pos_emb_batch(key_layer, rotary_pos_emb)


            # adjust key and value for inference
            if kv_cache[i] is not None:
                cache_k, cache_v = kv_cache[i]
                key_layer = torch.cat((cache_k, key_layer), dim=2)
                value_layer = torch.cat((cache_v, value_layer), dim=2)

            if use_cache:
                kv_cache[i] = (key_layer, value_layer)
            else:
                kv_cache[i] = None


            if self.multi_query_attention:
                key_layer = key_layer.unsqueeze(2)
                key_layer = key_layer.expand(
                    -1, -1, self.num_attention_heads_per_partition // self.num_multi_query_groups_per_partition, -1, -1
                )
                key_layer = key_layer.contiguous().view(
                    key_layer.size()[:1] + (self.num_attention_heads_per_partition,) + key_layer.size()[3:]
                )
                value_layer = value_layer.unsqueeze(2)
                value_layer = value_layer.expand(
                    -1, -1, self.num_attention_heads_per_partition // self.num_multi_query_groups_per_partition, -1, -1
                )
                value_layer = value_layer.contiguous().view(
                    value_layer.size()[:1] + (self.num_attention_heads_per_partition,) + value_layer.size()[3:]
                )
            query_layer_list.append(query_layer)
            key_layer_list.append(key_layer)
            value_layer_list.append(value_layer)

        # Determine a valid query tensor for shape inference
        valid_query = next((ql for ql in query_layer_list if isinstance(ql, torch.Tensor)), None)
        if valid_query is None:
            valid_key = next((kl for kl in key_layer_list if isinstance(kl, torch.Tensor)), None)
            if valid_key is not None:
                batch_size, n_heads, _, _ = valid_key.shape
            else:
                fallback_hidden = next(
                    (hs for hs in hidden_states if isinstance(hs, torch.Tensor) and hs.numel() > 0),
                    None,
                )
                if fallback_hidden is None:
                    return hidden_states, kv_cache, all_hidden_states, all_self_attentions
                batch_size = fallback_hidden.size(0)
                n_heads = self.num_attention_heads_per_partition
        else:
            batch_size, n_heads, _, _ = valid_query.shape

        # Restore query, key, and value layers to their original sequence positions
        seq_length_original = mask_for_mapping[0].shape[-1]
        query_layer_list_restored = self.batch_processor.restore_to_original_feature(
            query_layer_list,
            original_input_shape=(batch_size, n_heads, seq_length_original),
            modality_masks=mask_for_mapping,
        )
        key_layer_list_restored = self.batch_processor.restore_to_original_feature(
            key_layer_list,
            original_input_shape=(batch_size, n_heads, seq_length_original),
            modality_masks=mask_for_mapping,
        )
        value_layer_list_restored = self.batch_processor.restore_to_original_feature(
            value_layer_list,
            original_input_shape=(batch_size, n_heads, seq_length_original),
            modality_masks=mask_for_mapping,
        )

        context_layer_cache = []
        for batch_idx in range(batch_size):
            if mask_for_mapping[0, batch_idx].sum() != 0:
                context_layer_cache.append(
                    self.core_attention(
                        query_layer_list_restored[batch_idx:batch_idx + 1, :, mask_for_mapping[0, batch_idx]],
                        key_layer_list_restored[batch_idx:batch_idx + 1, :, mask_for_mapping[0, batch_idx]],
                        value_layer_list_restored[batch_idx:batch_idx + 1, :, mask_for_mapping[0, batch_idx]],
                        None,
                    )
                )
            else:
                context_layer_cache.append([])

        context_layer = torch.zeros(
            batch_size, n_heads, mask_for_mapping.shape[-1], query_layer_list_restored.shape[-1],
            device=query_layer_list_restored.device,
            dtype=query_layer_list_restored.dtype,
        )

        if mask_for_mapping[1].sum() != 0:
            context_layer_rest = self.core_attention(
                query_layer_list_restored,
                key_layer_list_restored,
                value_layer_list_restored,
                None,
            )
            for batch_idx in range(batch_size):
                mask_rest = mask_for_mapping[1, batch_idx]
                tokens_rest = int(mask_rest.sum().item())
                if tokens_rest > 0:
                    context_layer[batch_idx, :, mask_rest] = context_layer_rest[batch_idx, :, :tokens_rest, :]

        for batch_idx, cache_entry in enumerate(context_layer_cache):
            mask_primary = mask_for_mapping[0, batch_idx]
            tokens_primary = int(mask_primary.sum().item())
            if tokens_primary > 0 and isinstance(cache_entry, torch.Tensor):
                context_layer[batch_idx, :, mask_primary] = cache_entry.squeeze(0)

        context_layer_batches = self.batch_processor.create_batch_aware_modality_data(context_layer, mask_for_mapping)

        output = []
        for modality_idx, batch_entry in enumerate(context_layer_batches):
            if batch_entry != []:
                output.append(self.dense[modality_idx](batch_entry['tokens']))
            else:
                output.append([])


        return output, kv_cache


    def forward_second_expert(
            self, hidden_states, attention_mask, rotary_pos_emb, kv_cache=None, use_cache=True,
            expert_masks=None, 
            expert_masks_full=None
    ):
        """
        Forward pass for the second expert attention layer.
        
        Args:
            hidden_states: Input hidden states [batch_size, seq_len, hidden_size]
            attention_mask: Attention mask tensor
            rotary_pos_emb: Rotary positional embeddings
            kv_cache: Key-value cache for efficient inference
            use_cache: Whether to use caching
            expert_masks: Masks for expert routing
            expert_masks_full: Full expert masks
            
        Returns:
            output: Output tensor after attention
            kv_cache: Updated key-value cache
        """
        # Compute query, key, value projections for second expert
        mixed_x_layer = self.query_key_value[1](hidden_states)

        if self.multi_query_attention:

            (query_layer, key_layer, value_layer) = mixed_x_layer.split(
                [
                    self.num_attention_heads_per_partition * self.hidden_size_per_attention_head,
                    self.num_multi_query_groups_per_partition * self.hidden_size_per_attention_head,
                    self.num_multi_query_groups_per_partition * self.hidden_size_per_attention_head,
                ],
                dim=-1,
            )
            query_layer = query_layer.view(
                query_layer.size()[:-1] + (self.num_attention_heads_per_partition, self.hidden_size_per_attention_head)
            )
            key_layer = key_layer.view(
                key_layer.size()[:-1] + (self.num_multi_query_groups_per_partition, self.hidden_size_per_attention_head)
            )
            value_layer = value_layer.view(
                value_layer.size()[:-1]
                + (self.num_multi_query_groups_per_partition, self.hidden_size_per_attention_head)
            )
        else:

            # TODO: modify this part
            new_tensor_shape = mixed_x_layer.size()[:-1] + \
                            (self.num_attention_heads_per_partition,
                                3 * self.hidden_size_per_attention_head)
            mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)

            # [b, sq, np, 3 * hn] --> 3 [b, sq, np, hn]
            (query_layer, key_layer, value_layer) = split_tensor_along_last_dim(mixed_x_layer, 3)


        # [b, sq, np, hn] -> [b, np, sq, hn]
        query_layer, key_layer, value_layer = [k.transpose(1, 2) for k in [query_layer, key_layer, value_layer]]

        # apply relative positional encoding (rotary embedding)
        if rotary_pos_emb is not None:

            _, batch_size , seq_len = expert_masks.shape
            query_layer = apply_rotary_pos_emb_batch(query_layer, rotary_pos_emb)
            key_layer = apply_rotary_pos_emb_batch(key_layer, rotary_pos_emb)

        # adjust key and value for inference
        if kv_cache[1] is not None:
            cache_k_2, cache_v_2 = kv_cache[1]
            key_layer = torch.cat((cache_k_2, key_layer), dim=2)
            value_layer = torch.cat((cache_v_2, value_layer), dim=2)

        if use_cache:
            kv_cache[1] = (key_layer, value_layer)
        # else:


        if self.multi_query_attention:
            key_layer = key_layer.unsqueeze(2)
            key_layer = key_layer.expand(
                -1, -1, self.num_attention_heads_per_partition // self.num_multi_query_groups_per_partition, -1, -1
            )
            key_layer = key_layer.contiguous().view(
                key_layer.size()[:1] + (self.num_attention_heads_per_partition,) + key_layer.size()[3:]
            )
            value_layer = value_layer.unsqueeze(2)
            value_layer = value_layer.expand(
                -1, -1, self.num_attention_heads_per_partition // self.num_multi_query_groups_per_partition, -1, -1
            )
            value_layer = value_layer.contiguous().view(
                value_layer.size()[:1] + (self.num_attention_heads_per_partition,) + value_layer.size()[3:]
            )

            if kv_cache[0] is not None:
                cache_k_1, cache_v_1 = kv_cache[0]
                cache_k_1 = cache_k_1.unsqueeze(2)
                cache_k_1 = cache_k_1.expand(
                    -1, -1, self.num_attention_heads_per_partition // self.num_multi_query_groups_per_partition, -1, -1
                )
                cache_k_1 = cache_k_1.contiguous().view(
                    cache_k_1.size()[:1] + (self.num_attention_heads_per_partition,) + cache_k_1.size()[3:]
                )
                cache_v_1 = cache_v_1.unsqueeze(2)
                cache_v_1 = cache_v_1.expand(
                    -1, -1, self.num_attention_heads_per_partition // self.num_multi_query_groups_per_partition, -1, -1
                )
                cache_v_1 = cache_v_1.contiguous().view(
                    cache_v_1.size()[:1] + (self.num_attention_heads_per_partition,) + cache_v_1.size()[3:]
                )

        batch_size, n_heads, _ , _ = query_layer.shape


        if use_cache and (expert_masks is not None) and (kv_cache[0] is not None):
            seq_length_original = attention_mask.shape[-1]

            # In KV cache, the query will have only one token, so we don't need to restore it
            query_layer_list_restored = query_layer

            key_layer_list_restored = self.batch_processor.restore_to_original_feature(
                [cache_k_1, key_layer], 
                original_input_shape=(batch_size, n_heads, seq_length_original),  # batch_size, n_heads, seq_length
                modality_masks=expert_masks_full[:,:,:seq_length_original]
            )
            value_layer_list_restored = self.batch_processor.restore_to_original_feature(
                [cache_v_1, value_layer], 
                original_input_shape=(batch_size, n_heads, seq_length_original),  # batch_size, n_heads, seq_length
                modality_masks=expert_masks_full[:,:,:seq_length_original]
            )

            ## TODO: if we use flash attention, we should set attention_mask to None because flash attention will handle the mask
            context_layer = self.core_attention(query_layer_list_restored, key_layer_list_restored, value_layer_list_restored, None)
            # ## TODO: if we use core attention, we definitely need to set attention_mask
            # context_layer = self.core_attention(query_layer_list_restored, key_layer_list_restored, value_layer_list_restored, attention_mask)

            output = self.dense[1](context_layer)

        else:

            seq_length_original = expert_masks[0].shape[-1]

            query_layer_list_restored = self.batch_processor.restore_to_original_feature(
                [None, query_layer], 
                original_input_shape=(batch_size, n_heads, seq_length_original),  # batch_size, n_heads, seq_length
                modality_masks=expert_masks
            )

            key_layer_list_restored = self.batch_processor.restore_to_original_feature(
                [cache_k_1, key_layer], 
                original_input_shape=(batch_size, n_heads, seq_length_original),  # batch_size, n_heads, seq_length
                modality_masks=expert_masks
            )
            value_layer_list_restored = self.batch_processor.restore_to_original_feature(
                [cache_v_1, value_layer], 
                original_input_shape=(batch_size, n_heads, seq_length_original),  # batch_size, n_heads, seq_length
                modality_masks=expert_masks
            )


            context_layer = self.core_attention(query_layer_list_restored, key_layer_list_restored, value_layer_list_restored, attention_mask)
            context_layer_batches = self.batch_processor.create_batch_aware_expert_data(context_layer, expert_masks)

            output = self.dense[1](context_layer_batches[1]['tokens'])

        return output, kv_cache


    def forward_first_expert(
            self, hidden_states, attention_mask, rotary_pos_emb, kv_cache=None, use_cache=True,
    ):
        # Pre-allocate memory for key-values for inference.
        # Query, Key, and Value

        mixed_x_layer = self.query_key_value[0](hidden_states[0])

        if self.multi_query_attention:

            (query_layer, key_layer, value_layer) = mixed_x_layer.split(
                [
                    self.num_attention_heads_per_partition * self.hidden_size_per_attention_head,
                    self.num_multi_query_groups_per_partition * self.hidden_size_per_attention_head,
                    self.num_multi_query_groups_per_partition * self.hidden_size_per_attention_head,
                ],
                dim=-1,
            )
            query_layer = query_layer.view(
                query_layer.size()[:-1] + (self.num_attention_heads_per_partition, self.hidden_size_per_attention_head)
            )
            key_layer = key_layer.view(
                key_layer.size()[:-1] + (self.num_multi_query_groups_per_partition, self.hidden_size_per_attention_head)
            )
            value_layer = value_layer.view(
                value_layer.size()[:-1]
                + (self.num_multi_query_groups_per_partition, self.hidden_size_per_attention_head)
            )
        else:

            # TODO: modify this part
            new_tensor_shape = mixed_x_layer.size()[:-1] + \
                            (self.num_attention_heads_per_partition,
                                3 * self.hidden_size_per_attention_head)
            mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)

            # [b, sq, np, 3 * hn] --> 3 [b, sq, np, hn]
            (query_layer, key_layer, value_layer) = split_tensor_along_last_dim(mixed_x_layer, 3)

        # [b, sq, np, hn] -> [b, np, sq, hn]
        query_layer, key_layer, value_layer = [k.transpose(1, 2) for k in [query_layer, key_layer, value_layer]]

        # apply relative positional encoding (rotary embedding)
        if rotary_pos_emb is not None:
            query_layer = apply_rotary_pos_emb_batch(query_layer, rotary_pos_emb)
            key_layer = apply_rotary_pos_emb_batch(key_layer, rotary_pos_emb)


        if kv_cache[0] is not None:
            kv_cache[0] = (torch.cat((kv_cache[0][0], key_layer), dim=2), torch.cat((kv_cache[0][1], value_layer), dim=2))
            key_layer, value_layer = kv_cache[0]
        else:
            kv_cache[0] = (key_layer, value_layer)


        if self.multi_query_attention:
            key_layer = key_layer.unsqueeze(2)
            key_layer = key_layer.expand(
                -1, -1, self.num_attention_heads_per_partition // self.num_multi_query_groups_per_partition, -1, -1
            )
            key_layer = key_layer.contiguous().view(
                key_layer.size()[:1] + (self.num_attention_heads_per_partition,) + key_layer.size()[3:]
            )
            value_layer = value_layer.unsqueeze(2)
            value_layer = value_layer.expand(
                -1, -1, self.num_attention_heads_per_partition // self.num_multi_query_groups_per_partition, -1, -1
            )
            value_layer = value_layer.contiguous().view(
                value_layer.size()[:1] + (self.num_attention_heads_per_partition,) + value_layer.size()[3:]
            )


        # core attention computation
        context_layer = self.core_attention(query_layer, key_layer, value_layer, None)


        output = []
        output.append(self.dense[0](context_layer))

        return output, kv_cache


class OriginalSelfAttention(torch.nn.Module):
    """Parallel self-attention layer abstract class.

    Self-attention layer takes input with size [s, b, h]
    and returns output of the same size.
    """

    def __init__(self, config: ChatGLMConfig, layer_number, device=None):
        super(OriginalSelfAttention, self).__init__()
        self.layer_number = max(1, layer_number)
        self.config = config
        self.n_modalities = 2
        self.batch_processor = ModalityBatchProcessor(self.n_modalities)
        self.projection_size = config.kv_channels * config.num_attention_heads

        # Per attention head and per partition values.
        self.hidden_size_per_attention_head = self.projection_size // config.num_attention_heads
        self.num_attention_heads_per_partition = config.num_attention_heads
        self.multi_query_attention = config.multi_query_attention
        self.qkv_hidden_size = 3 * self.projection_size
        if self.multi_query_attention:
            self.num_multi_query_groups_per_partition = config.multi_query_group_num
            self.qkv_hidden_size = (
                    self.projection_size + 2 * self.hidden_size_per_attention_head * config.multi_query_group_num
            )

        self.query_key_value = torch.nn.ModuleList(
                    [
                        nn.Linear(config.hidden_size, self.qkv_hidden_size,
                        bias=config.add_bias_linear or config.add_qkv_bias,
                        device=device, **_config_to_kwargs(config)
                        )
                    ]
                )
        self.core_attention = CORE_ATTENTION_CLASSES[config._attn_implementation](config, self.layer_number)

        # Output.
        self.dense = torch.nn.ModuleList(
                    [
                        nn.Linear(self.projection_size, config.hidden_size, bias=config.add_bias_linear,
                                                device=device, **_config_to_kwargs(config)
                                                )
                    ]
                )

    def set_attention_implementation(self, attn_impl: str):
        self.config._attn_implementation = attn_impl
        self.core_attention = CORE_ATTENTION_CLASSES[attn_impl](self.config, self.layer_number)

    def _process_qkv(self, x, modality_masks):
        """
        Process query, key, and value projections for each modality.
        """
        expert_outputs_xq, expert_outputs_xk, expert_outputs_xv = [], [], []
        for i in range(self.n_modalities):
            expert_input = x[modality_masks[i]]
            xq = self.query_key_value[i](expert_input)
            xk = self.query_key_value[i](expert_input)
            xv = self.query_key_value[i](expert_input)

            expert_outputs_xq.append(xq)
            expert_outputs_xk.append(xk)
            expert_outputs_xv.append(xv)

        return expert_outputs_xq, expert_outputs_xk, expert_outputs_xv

    def _allocate_memory(self, inference_max_sequence_len, batch_size, device=None, dtype=None):
        if self.multi_query_attention:
            num_attention_heads = self.num_multi_query_groups_per_partition
        else:
            num_attention_heads = self.num_attention_heads_per_partition
        return torch.empty(
            inference_max_sequence_len,
            batch_size,
            num_attention_heads,
            self.hidden_size_per_attention_head,
            dtype=dtype,
            device=device,
        )


    def forward(
            self, hidden_states, attention_mask, rotary_pos_emb, kv_cache=None, use_cache=True,
            modality_masks=None, modality_masks_full=None, past_key_values=None
    ):
        # Pre-allocate memory for key-values for inference.
        # Query, Key, and Value

        if isinstance(kv_cache, tuple):
            kv_cache = list(kv_cache)

        if past_key_values is not None and isinstance(past_key_values, (list, tuple)):
            for cache_idx in range(min(len(past_key_values), len(kv_cache))):
                cache_entry = past_key_values[cache_idx]
                if cache_entry is None:
                    continue
                kv_cache[cache_idx] = cache_entry

        mask_for_mapping = (
            modality_masks_full
            if modality_masks_full is not None
            else modality_masks
        )

        mixed_x_layer = []
        for i in range(len(hidden_states)):
            if hidden_states[i] != []:
                expert_output = self.query_key_value[i](hidden_states[i])
                mixed_x_layer.append(expert_output)
            else:
                mixed_x_layer.append([])

        query_layer_list, key_layer_list, value_layer_list = [], [], []

        for i in range(len(mixed_x_layer)):


            if modality_masks[i].sum() == 0:
                if i < len(kv_cache) and kv_cache[i] is not None:
                    cache_k, cache_v = kv_cache[i]
                    key_layer = cache_k
                    value_layer = cache_v
                    if self.multi_query_attention:
                        key_layer = key_layer.unsqueeze(2)
                        key_layer = key_layer.expand(
                            -1, -1, self.num_attention_heads_per_partition // self.num_multi_query_groups_per_partition, -1, -1
                        )
                        key_layer = key_layer.contiguous().view(
                            key_layer.size()[:1] + (self.num_attention_heads_per_partition,) + key_layer.size()[3:]
                        )
                        value_layer = value_layer.unsqueeze(2)
                        value_layer = value_layer.expand(
                            -1, -1, self.num_attention_heads_per_partition // self.num_multi_query_groups_per_partition, -1, -1
                        )
                        value_layer = value_layer.contiguous().view(
                            value_layer.size()[:1] + (self.num_attention_heads_per_partition,) + value_layer.size()[3:]
                        )
                    query_layer_list.append(None)
                    key_layer_list.append(key_layer)
                    value_layer_list.append(value_layer)
                else:
                    query_layer_list.append(None)
                    key_layer_list.append(None)
                    value_layer_list.append(None)
                continue

            if self.multi_query_attention:

                (query_layer, key_layer, value_layer) = mixed_x_layer[i].split(
                    [
                        self.num_attention_heads_per_partition * self.hidden_size_per_attention_head,
                        self.num_multi_query_groups_per_partition * self.hidden_size_per_attention_head,
                        self.num_multi_query_groups_per_partition * self.hidden_size_per_attention_head,
                    ],
                    dim=-1,
                )
                query_layer = query_layer.view(
                    query_layer.size()[:-1] + (self.num_attention_heads_per_partition, self.hidden_size_per_attention_head)
                )
                key_layer = key_layer.view(
                    key_layer.size()[:-1] + (self.num_multi_query_groups_per_partition, self.hidden_size_per_attention_head)
                )
                value_layer = value_layer.view(
                    value_layer.size()[:-1]
                    + (self.num_multi_query_groups_per_partition, self.hidden_size_per_attention_head)
                )
            else:

                # TODO: modify this part
                new_tensor_shape = mixed_x_layer.size()[:-1] + \
                                (self.num_attention_heads_per_partition,
                                    3 * self.hidden_size_per_attention_head)
                mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)

                # [b, sq, np, 3 * hn] --> 3 [b, sq, np, hn]
                (query_layer, key_layer, value_layer) = split_tensor_along_last_dim(mixed_x_layer, 3)

            # [b, sq, np, hn] -> [b, np, sq, hn]
            query_layer, key_layer, value_layer = [k.transpose(1, 2) for k in [query_layer, key_layer, value_layer]]

            # apply relative positional encoding (rotary embedding)
            if rotary_pos_emb is not None:
                if modality_masks_full is not None:
                    query_layer = apply_rotary_pos_emb_batch(query_layer, rotary_pos_emb)
                    key_layer = apply_rotary_pos_emb_batch(key_layer, rotary_pos_emb)
                else:
                    query_layer = apply_rotary_pos_emb_batch(query_layer, rotary_pos_emb)
                    key_layer = apply_rotary_pos_emb_batch(key_layer, rotary_pos_emb)


            # adjust key and value for inference
            if kv_cache[i] is not None:
                cache_k, cache_v = kv_cache[i]
                key_layer = torch.cat((cache_k, key_layer), dim=2)
                value_layer = torch.cat((cache_v, value_layer), dim=2)

            if use_cache:
                kv_cache[i] = (key_layer, value_layer)
            else:
                kv_cache[i] = None


            if self.multi_query_attention:
                key_layer = key_layer.unsqueeze(2)
                key_layer = key_layer.expand(
                    -1, -1, self.num_attention_heads_per_partition // self.num_multi_query_groups_per_partition, -1, -1
                )
                key_layer = key_layer.contiguous().view(
                    key_layer.size()[:1] + (self.num_attention_heads_per_partition,) + key_layer.size()[3:]
                )
                value_layer = value_layer.unsqueeze(2)
                value_layer = value_layer.expand(
                    -1, -1, self.num_attention_heads_per_partition // self.num_multi_query_groups_per_partition, -1, -1
                )
                value_layer = value_layer.contiguous().view(
                    value_layer.size()[:1] + (self.num_attention_heads_per_partition,) + value_layer.size()[3:]
                )
            query_layer_list.append(query_layer)
            key_layer_list.append(key_layer)
            value_layer_list.append(value_layer)

        # Determine a valid query tensor for shape inference
        valid_query = next((ql for ql in query_layer_list if isinstance(ql, torch.Tensor)), None)
        if valid_query is None:
            valid_key = next((kl for kl in key_layer_list if isinstance(kl, torch.Tensor)), None)
            if valid_key is not None:
                batch_size, n_heads, _, _ = valid_key.shape
            else:
                fallback_hidden = next(
                    (hs for hs in hidden_states if isinstance(hs, torch.Tensor) and hs.numel() > 0),
                    None,
                )
                if fallback_hidden is None:
                    return hidden_states, kv_cache, all_hidden_states, all_self_attentions
                batch_size = fallback_hidden.size(0)
                n_heads = self.num_attention_heads_per_partition
        else:
            batch_size, n_heads, _, _ = valid_query.shape


        seq_length_original = mask_for_mapping[0].shape[-1]
        query_layer_list_restored = self.batch_processor.restore_to_original_feature(
            query_layer_list,
            original_input_shape=(batch_size, n_heads, seq_length_original),
            modality_masks=mask_for_mapping,
        )
        key_layer_list_restored = self.batch_processor.restore_to_original_feature(
            key_layer_list,
            original_input_shape=(batch_size, n_heads, seq_length_original),
            modality_masks=mask_for_mapping,
        )
        value_layer_list_restored = self.batch_processor.restore_to_original_feature(
            value_layer_list,
            original_input_shape=(batch_size, n_heads, seq_length_original),
            modality_masks=mask_for_mapping,
        )

        context_layer_cache = []
        for batch_idx in range(batch_size):
            if mask_for_mapping[0, batch_idx].sum() != 0:
                context_layer_cache.append(
                    self.core_attention(
                        query_layer_list_restored[batch_idx:batch_idx + 1, :, mask_for_mapping[0, batch_idx]],
                        key_layer_list_restored[batch_idx:batch_idx + 1, :, mask_for_mapping[0, batch_idx]],
                        value_layer_list_restored[batch_idx:batch_idx + 1, :, mask_for_mapping[0, batch_idx]],
                        None,
                    )
                )
            else:
                context_layer_cache.append([])

        context_layer = torch.zeros(
            batch_size, n_heads, mask_for_mapping.shape[-1], query_layer_list_restored.shape[-1],
            device=query_layer_list_restored.device,
            dtype=query_layer_list_restored.dtype,
        )

        if mask_for_mapping[1].sum() != 0:
            context_layer_rest = self.core_attention(
                query_layer_list_restored,
                key_layer_list_restored,
                value_layer_list_restored,
                None,
            )
            for batch_idx in range(batch_size):
                mask_rest = mask_for_mapping[1, batch_idx]
                tokens_rest = int(mask_rest.sum().item())
                if tokens_rest > 0:
                    context_layer[batch_idx, :, mask_rest] = context_layer_rest[batch_idx, :, :tokens_rest, :]

        for batch_idx, cache_entry in enumerate(context_layer_cache):
            mask_primary = mask_for_mapping[0, batch_idx]
            tokens_primary = int(mask_primary.sum().item())
            if tokens_primary > 0 and isinstance(cache_entry, torch.Tensor):
                context_layer[batch_idx, :, mask_primary] = cache_entry.squeeze(0)

        context_layer_batches = self.batch_processor.create_batch_aware_modality_data(context_layer, mask_for_mapping)

        output = []
        for modality_idx, batch_entry in enumerate(context_layer_batches):
            if batch_entry != []:
                output.append(self.dense[modality_idx](batch_entry['tokens']))
            else:
                output.append([])


        return output, kv_cache


    def forward_first_expert(
            self, hidden_states, attention_mask, rotary_pos_emb, kv_cache=None, use_cache=True,
    ):
        # Pre-allocate memory for key-values for inference.
        # Query, Key, and Value

        mixed_x_layer = self.query_key_value[0](hidden_states[0])

        if self.multi_query_attention:

            (query_layer, key_layer, value_layer) = mixed_x_layer.split(
                [
                    self.num_attention_heads_per_partition * self.hidden_size_per_attention_head,
                    self.num_multi_query_groups_per_partition * self.hidden_size_per_attention_head,
                    self.num_multi_query_groups_per_partition * self.hidden_size_per_attention_head,
                ],
                dim=-1,
            )
            query_layer = query_layer.view(
                query_layer.size()[:-1] + (self.num_attention_heads_per_partition, self.hidden_size_per_attention_head)
            )
            key_layer = key_layer.view(
                key_layer.size()[:-1] + (self.num_multi_query_groups_per_partition, self.hidden_size_per_attention_head)
            )
            value_layer = value_layer.view(
                value_layer.size()[:-1]
                + (self.num_multi_query_groups_per_partition, self.hidden_size_per_attention_head)
            )
        else:

            # TODO: modify this part
            new_tensor_shape = mixed_x_layer.size()[:-1] + \
                            (self.num_attention_heads_per_partition,
                                3 * self.hidden_size_per_attention_head)
            mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)

            # [b, sq, np, 3 * hn] --> 3 [b, sq, np, hn]
            (query_layer, key_layer, value_layer) = split_tensor_along_last_dim(mixed_x_layer, 3)

        # [b, sq, np, hn] -> [b, np, sq, hn]
        query_layer, key_layer, value_layer = [k.transpose(1, 2) for k in [query_layer, key_layer, value_layer]]

        # apply relative positional encoding (rotary embedding)
        if rotary_pos_emb is not None:
            query_layer = apply_rotary_pos_emb_batch(query_layer, rotary_pos_emb)
            key_layer = apply_rotary_pos_emb_batch(key_layer, rotary_pos_emb)


        if kv_cache[0] is not None:
            kv_cache[0] = (torch.cat((kv_cache[0][0], key_layer), dim=2), torch.cat((kv_cache[0][1], value_layer), dim=2))
            key_layer, value_layer = kv_cache[0]
        else:
            kv_cache[0] = (key_layer, value_layer)


        if self.multi_query_attention:
            key_layer = key_layer.unsqueeze(2)
            key_layer = key_layer.expand(
                -1, -1, self.num_attention_heads_per_partition // self.num_multi_query_groups_per_partition, -1, -1
            )
            key_layer = key_layer.contiguous().view(
                key_layer.size()[:1] + (self.num_attention_heads_per_partition,) + key_layer.size()[3:]
            )
            value_layer = value_layer.unsqueeze(2)
            value_layer = value_layer.expand(
                -1, -1, self.num_attention_heads_per_partition // self.num_multi_query_groups_per_partition, -1, -1
            )
            value_layer = value_layer.contiguous().view(
                value_layer.size()[:1] + (self.num_attention_heads_per_partition,) + value_layer.size()[3:]
            )


        # core attention computation
        context_layer = self.core_attention(query_layer, key_layer, value_layer, None)


        output = []
        output.append(self.dense[0](context_layer))

        return output, kv_cache


def _config_to_kwargs(args):
    common_kwargs = {
        "dtype": args.torch_dtype,
    }
    return common_kwargs


class MLP(torch.nn.Module):
    """MLP.

    MLP will take the input with h hidden state, project it to 4*h
    hidden dimension, perform nonlinear transformation, and project the
    state back into h hidden dimension.
    """

    def __init__(self, config: ChatGLMConfig, device=None):
        super(MLP, self).__init__()

        self.add_bias = config.add_bias_linear

        # Project to 4h. If using swiglu double the output width, see https://arxiv.org/pdf/2002.05202.pdf
        self.dense_h_to_4h = nn.Linear(
            config.hidden_size,
            config.ffn_hidden_size * 2,
            bias=self.add_bias,
            device=device,
            **_config_to_kwargs(config)
        )

        def swiglu(x):
            x = torch.chunk(x, 2, dim=-1)
            return F.silu(x[0]) * x[1]

        self.activation_func = swiglu

        # Project back to h.
        self.dense_4h_to_h = nn.Linear(
            config.ffn_hidden_size,
            config.hidden_size,
            bias=self.add_bias,
            device=device,
            **_config_to_kwargs(config)
        )

    def forward(self, hidden_states):
        """
        Forward pass through MLP layer.
        
        Args:
            hidden_states: Input tensor [seq_len, batch_size, hidden_size]
            
        Returns:
            Output tensor [seq_len, batch_size, hidden_size]
        """
        # Project to 4*hidden_size, apply activation, then project back
        intermediate_parallel = self.dense_h_to_4h(hidden_states)
        intermediate_parallel = self.activation_func(intermediate_parallel)
        # Project back to hidden_size
        output = self.dense_4h_to_h(intermediate_parallel)
        return output


class GLMBlock(torch.nn.Module):
    """A single transformer layer.

    Transformer layer takes input with size [s, b, h] and returns an
    output of the same size.
    """

    def __init__(self, config: ChatGLMConfig, layer_number, device=None):
        super(GLMBlock, self).__init__()
        self.layer_number = layer_number

        self.apply_residual_connection_post_layernorm = config.apply_residual_connection_post_layernorm

        self.fp32_residual_connection = config.fp32_residual_connection

        LayerNormFunc = RMSNorm if config.rmsnorm else LayerNorm
        # Layernorm on the input data.
        self.input_layernorm = LayerNormFunc(config.hidden_size, eps=config.layernorm_epsilon, device=device,
                                             dtype=config.torch_dtype)

        # Self attention.
        self.self_attention = SelfAttention(config, layer_number, device=device)
        self.hidden_dropout = config.hidden_dropout

        # Layernorm on the attention output
        self.post_attention_layernorm = LayerNormFunc(config.hidden_size, eps=config.layernorm_epsilon, device=device,
                                                      dtype=config.torch_dtype)

        # MLP
        self.mlp = MLP(config, device=device)

    def forward(
            self, hidden_states, attention_mask, rotary_pos_emb, kv_cache=None, use_cache=True,
    ):
        """
        Forward pass for a single transformer block.
        
        Args:
            hidden_states: Input hidden states [seq_len, batch_size, hidden_size]
            attention_mask: Attention mask tensor
            rotary_pos_emb: Rotary positional embeddings
            kv_cache: Key-value cache for efficient inference
            use_cache: Whether to use caching
            
        Returns:
            output: Output tensor [seq_len, batch_size, hidden_size]
            kv_cache: Updated key-value cache
        """
        # Layer norm at the beginning of the transformer layer.
        layernorm_output = self.input_layernorm(hidden_states)
        # Self attention.
        attention_output, kv_cache = self.self_attention(
            layernorm_output,
            attention_mask,
            rotary_pos_emb,
            kv_cache=kv_cache,
            use_cache=use_cache
        )

        # Residual connection.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = hidden_states

        layernorm_input = torch.nn.functional.dropout(attention_output, p=self.hidden_dropout, training=self.training)
        layernorm_input = residual + layernorm_input

        # Layer norm post the self attention.
        layernorm_output = self.post_attention_layernorm(layernorm_input)

        # MLP.
        mlp_output = self.mlp(layernorm_output)

        # Second residual connection.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = layernorm_input

        output = torch.nn.functional.dropout(mlp_output, p=self.hidden_dropout, training=self.training)
        output = residual + output

        return output, kv_cache


class GLMBlockMot(torch.nn.Module):
    """A single transformer layer.

    Transformer layer takes input with size [s, b, h] and returns an
    output of the same size.
    """

    def __init__(self, config: ChatGLMConfig, layer_number, device=None):
        super(GLMBlockMot, self).__init__()
        self.layer_number = layer_number
        self.n_modalities = 2

        self.apply_residual_connection_post_layernorm = config.apply_residual_connection_post_layernorm
        self.fp32_residual_connection = config.fp32_residual_connection

        LayerNormFunc = RMSNorm if config.rmsnorm else LayerNorm

        # Layernorm on the input data.
        self.local_experts_input_layernorm = torch.nn.ModuleList([
            LayerNormFunc(config.hidden_size, eps=config.layernorm_epsilon, device=device, dtype=config.torch_dtype),
            LayerNormFunc(config.mot_settings['mot_hidden_size'], eps=config.layernorm_epsilon, device=device, dtype=config.torch_dtype)
        ])
        # Self attention.
        self.self_attention = ModalityUntiedSelfAttention(config, layer_number, device=device)
        self.hidden_dropout = config.hidden_dropout

        # Layernorm on the attention output
        self.local_experts_post_attention_layernorm = torch.nn.ModuleList([
            LayerNormFunc(config.hidden_size, eps=config.layernorm_epsilon, device=device, dtype=config.torch_dtype),
            LayerNormFunc(config.mot_settings['mot_hidden_size'], eps=config.layernorm_epsilon, device=device, dtype=config.torch_dtype)
        ])

        # MLP
        config_expert = copy.deepcopy(config)
        config_expert.hidden_size = config.mot_settings['mot_hidden_size']
        config_expert.ffn_hidden_size = config.mot_settings['mot_ffn_hidden_size'] ### original is 13696, we set it to 4096 following the pi0 paper
        self.local_experts_mlp = torch.nn.ModuleList([
            MLP(config, device=device),
            MLP(config_expert, device=device)
        ])

    def forward(
            self, hidden_states, attention_mask, rotary_pos_emb, kv_cache=None, use_cache=True, modality_masks=None, modality_masks_full=None, past_key_values=None
    ):
        """
        Forward pass for motion-aware transformer block.
        
        Args:
            hidden_states: Input hidden states (list of tensors for each modality)
            attention_mask: Attention mask tensor
            rotary_pos_emb: Rotary positional embeddings
            kv_cache: Key-value cache for efficient inference
            use_cache: Whether to use caching
            modality_masks: Masks for different modalities
            modality_masks_full: Full modality masks
            past_key_values: Past key-value pairs from previous forward passes
            
        Returns:
            output: Output tensor (list of tensors for each modality)
            kv_cache: Updated key-value cache
        """
        # Layer norm at the beginning of the transformer layer
        if hidden_states[0] != []:
            if hidden_states[0].shape[1] == 1:
                modality_masks = modality_masks[:, :, :1]
        layernorm_output = []
        for i in range(len(hidden_states)):
            if hidden_states[i] != []:
                expert_output = self.local_experts_input_layernorm[i](hidden_states[i])
                layernorm_output.append(expert_output)
            else:
                layernorm_output.append([])

        # Self attention.
        attention_output, kv_cache = self.self_attention.forward_second_expert_with_cached_kv(
            layernorm_output,
            attention_mask,
            rotary_pos_emb,
            kv_cache=kv_cache,
            use_cache=use_cache,
            modality_masks=modality_masks,
            modality_masks_full=modality_masks_full,
            past_key_values=past_key_values  # Pass cached KV from T1
        )

        # Residual connection.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = hidden_states

        # Compute layernorm input with residual connection
        layernorm_input = []
        for i in range(len(attention_output)):
            if attention_output[i] != [] and residual[i] != []:
                layernorm_input.append(torch.nn.functional.dropout(attention_output[i], p=self.hidden_dropout, training=self.training))
                layernorm_input[i] = residual[i] + layernorm_input[i]
            else:
                layernorm_input.append([])


        # Layer norm post the self attention.

        layernorm_output = []
        for i in range(len(layernorm_input)):
            expert_input = layernorm_input[i]
            if expert_input != []:
                expert_output = self.local_experts_post_attention_layernorm[i](expert_input)
                layernorm_output.append(expert_output)
            else:
                layernorm_output.append([])

        # MLP.

        mlp_output = []
        for i in range(len(layernorm_output)):
            if layernorm_output[i] != []:
                expert_output = self.local_experts_mlp[i](layernorm_output[i])
                mlp_output.append(expert_output)
            else:
                mlp_output.append([])

        # Second residual connection.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = layernorm_input

        output = []
        for i in range(len(mlp_output)):
            if mlp_output[i] != []:
                output.append(torch.nn.functional.dropout(mlp_output[i], p=self.hidden_dropout, training=self.training))
                output[i] = residual[i] + output[i]
            else:
                output.append([])

        return output, kv_cache

    def forward_first_expert(
            self, hidden_states, attention_mask, rotary_pos_emb, kv_cache=None, use_cache=True
    ):
        """
        Forward pass for first expert in motion-aware transformer block.
        
        Args:
            hidden_states: Input hidden states (list of tensors for each modality)
            attention_mask: Attention mask tensor
            rotary_pos_emb: Rotary positional embeddings
            kv_cache: Key-value cache for efficient inference
            use_cache: Whether to use caching
            
        Returns:
            output: Output tensor (list of tensors for each modality)
            kv_cache: Updated key-value cache
        """
        # Motion-aware processing workflow
        layernorm_output = []
        for i in range(len(hidden_states)):
            if hidden_states[i] != []:
                expert_output = self.local_experts_input_layernorm[i](hidden_states[i])
                layernorm_output.append(expert_output)
            else:
                layernorm_output.append([])

        # Self attention.
        attention_output, kv_cache = self.self_attention.forward_first_expert(
            layernorm_output,
            attention_mask,
            rotary_pos_emb,
            kv_cache=kv_cache,
            use_cache=use_cache,
        )

        # Residual connection.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = hidden_states

        layernorm_input = []
        for i in range(len(attention_output)):
            if attention_output[i] != [] and residual[i] != []:
                layernorm_input.append(torch.nn.functional.dropout(attention_output[i], p=self.hidden_dropout, training=self.training))
                layernorm_input[i] = residual[i] + layernorm_input[i]
            else:
                layernorm_input.append([])


        layernorm_output = []
        for i in range(len(layernorm_input)):
            expert_input = layernorm_input[i]
            if expert_input != []:
                expert_output = self.local_experts_post_attention_layernorm[i](expert_input)
                layernorm_output.append(expert_output)
            else:
                layernorm_output.append([])

        mlp_output = []
        for i in range(len(layernorm_output)):
            if layernorm_output[i] != []:
                expert_output = self.local_experts_mlp[i](layernorm_output[i])
                mlp_output.append(expert_output)
            else:
                mlp_output.append([])

        # Second residual connection.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = layernorm_input

        output = []
        for i in range(len(mlp_output)):
            if mlp_output[i] != []:
                output.append(torch.nn.functional.dropout(mlp_output[i], p=self.hidden_dropout, training=self.training))
                output[i] = residual[i] + output[i]
            else:
                output.append([])

        return output, kv_cache

    def forward_second_expert(
            self, hidden_states, attention_mask, rotary_pos_emb, kv_cache=None, use_cache=True, expert_masks=None, expert_masks_full=None
    ):
        """
        Forward pass for the second expert in the transformer layer.

        Args:
            hidden_states: Input hidden states tensor [seq_len, batch_size, hidden_size]
            attention_mask: Attention mask tensor
            rotary_pos_emb: Rotary positional embeddings
            kv_cache: Key-value cache for efficient inference
            use_cache: Whether to use caching
            expert_masks: Masks for expert-specific processing
            expert_masks_full: Full masks for expert processing

        Returns:
            output: Output tensor after processing
            kv_cache: Updated key-value cache
        """
        # Apply input layer normalization for second expert
        layernorm_output = self.local_experts_input_layernorm[1](hidden_states)

        # Self-attention with expert-specific masks
        attention_output, kv_cache = self.self_attention.forward_second_expert(
            layernorm_output,
            attention_mask,
            rotary_pos_emb,
            kv_cache=kv_cache,
            use_cache=use_cache,
            expert_masks=expert_masks,
            expert_masks_full=expert_masks_full,
        )

        # Residual connection with dropout
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = hidden_states

        layernorm_input = torch.nn.functional.dropout(attention_output, p=self.hidden_dropout, training=self.training)
        layernorm_input = residual + layernorm_input

        # Post-attention layer normalization
        layernorm_output = self.local_experts_post_attention_layernorm[1](layernorm_input)

        # MLP processing
        mlp_output = self.local_experts_mlp[1](layernorm_output)

        # Second residual connection
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = layernorm_input

        # Final output with dropout and residual connection
        output = torch.nn.functional.dropout(mlp_output, p=self.hidden_dropout, training=self.training)
        output = residual + output

        return output, kv_cache


class GLMBlockOriginal(torch.nn.Module):
    """A single transformer layer.

    Transformer layer takes input with size [s, b, h] and returns an
    output of the same size.
    """

    def __init__(self, config: ChatGLMConfig, layer_number, device=None):
        super(GLMBlockOriginal, self).__init__()
        self.layer_number = layer_number
        self.n_modalities = 2

        self.apply_residual_connection_post_layernorm = config.apply_residual_connection_post_layernorm

        self.fp32_residual_connection = config.fp32_residual_connection

        LayerNormFunc = RMSNorm if config.rmsnorm else LayerNorm
        # Layernorm on the input data.
        self.local_experts_input_layernorm = torch.nn.ModuleList([
            LayerNormFunc(config.hidden_size, eps=config.layernorm_epsilon, device=device, dtype=config.torch_dtype)
        ])

        # Self attention.
        self.self_attention = OriginalSelfAttention(config, layer_number, device=device)
        self.hidden_dropout = config.hidden_dropout

        # Layernorm on the attention output                                                      
        self.local_experts_post_attention_layernorm = torch.nn.ModuleList([
            LayerNormFunc(config.hidden_size, eps=config.layernorm_epsilon, device=device, dtype=config.torch_dtype)
        ])


        # MLP
        self.local_experts_mlp = torch.nn.ModuleList([
            MLP(config, device=device)
        ])

    def forward(
            self, hidden_states, attention_mask, rotary_pos_emb, kv_cache=None, use_cache=True,
            modality_masks=None
    ):
        """
        Forward pass for original transformer block with modality support.
        
        Args:
            hidden_states: Input hidden states (list of tensors for each modality)
            attention_mask: Attention mask tensor
            rotary_pos_emb: Rotary positional embeddings
            kv_cache: Key-value cache for efficient inference
            use_cache: Whether to use caching
            modality_masks: Masks for different modalities
            
        Returns:
            output: Output tensor (list of tensors for each modality)
            kv_cache: Updated key-value cache
        """
        # Layer norm at the beginning of the transformer layer
        if hidden_states[0].shape[1] == 1:
            modality_masks = modality_masks[:, :, :1]
        layernorm_output = []
        for i in range(1):
            if hidden_states[i] != []:
                expert_output = self.local_experts_input_layernorm[i](hidden_states[i])
                layernorm_output.append(expert_output)
            else:
                layernorm_output.append([])

        # Self attention.
        attention_output, kv_cache = self.self_attention(
            layernorm_output,
            attention_mask,
            rotary_pos_emb,
            kv_cache=kv_cache,
            use_cache=use_cache,
            modality_masks=modality_masks
        )

        # Residual connection.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = hidden_states

        layernorm_input = []
        for i in range(len(attention_output)):
            if attention_output[i] != []:
                layernorm_input.append(torch.nn.functional.dropout(attention_output[i], p=self.hidden_dropout, training=self.training))
                layernorm_input[i] = residual[i] + layernorm_input[i]
            else:
                layernorm_input.append([])

        # Layer norm post the self attention.

        layernorm_output = []
        for i in range(len(layernorm_input)):
            expert_input = layernorm_input[i]
            if expert_input != []:
                expert_output = self.local_experts_post_attention_layernorm[i](expert_input)
                layernorm_output.append(expert_output)
            else:
                layernorm_output.append([])

        # MLP processing for each modality
        mlp_output = []
        for i in range(len(layernorm_output)):
            if layernorm_output[i] != []:
                expert_output = self.local_experts_mlp[i](layernorm_output[i])
                mlp_output.append(expert_output)
            else:
                mlp_output.append([])

        # Second residual connection.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = layernorm_input

        output = []
        for i in range(len(mlp_output)):
            if mlp_output[i] != []:
                output.append(torch.nn.functional.dropout(mlp_output[i], p=self.hidden_dropout, training=self.training))
                output[i] = residual[i] + output[i]
            else:
                output.append([])

        if len(hidden_states) > 1:
            output.append(hidden_states[1])

        return output, kv_cache

    def forward_first_expert(
            self, hidden_states, attention_mask, rotary_pos_emb, kv_cache=None, use_cache=True
    ):
        """
        Forward pass for the first expert in the transformer layer.

        Args:
            hidden_states: Input hidden states tensor [seq_len, batch_size, hidden_size]
            attention_mask: Attention mask tensor
            rotary_pos_emb: Rotary positional embeddings
            kv_cache: Key-value cache for efficient inference
            use_cache: Whether to use caching

        Returns:
            output: Output tensor after processing
            kv_cache: Updated key-value cache
        """
        # Apply input layer normalization for each modality
        layernorm_output = []
        for i in range(len(hidden_states)):
            if hidden_states[i] != []:
                expert_output = self.local_experts_input_layernorm[i](hidden_states[i])
                layernorm_output.append(expert_output)
            else:
                layernorm_output.append([])

        # Self-attention processing
        attention_output, kv_cache = self.self_attention.forward_first_expert(
            layernorm_output,
            attention_mask,
            rotary_pos_emb,
            kv_cache=kv_cache,
            use_cache=use_cache,
        )

        # Residual connection with dropout
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = hidden_states

        layernorm_input = []
        for i in range(len(attention_output)):
            if attention_output[i] != [] and residual[i] != []:
                layernorm_input.append(torch.nn.functional.dropout(attention_output[i], p=self.hidden_dropout, training=self.training))
                layernorm_input[i] = residual[i] + layernorm_input[i]
            else:
                layernorm_input.append([])

        # Post-attention layer normalization for each modality
        layernorm_output = []
        for i in range(len(layernorm_input)):
            expert_input = layernorm_input[i]
            if expert_input != []:
                expert_output = self.local_experts_post_attention_layernorm[i](expert_input)
                layernorm_output.append(expert_output)
            else:
                layernorm_output.append([])

        # MLP processing for each modality
        mlp_output = []
        for i in range(len(layernorm_output)):
            if layernorm_output[i] != []:
                expert_output = self.local_experts_mlp[i](layernorm_output[i])
                mlp_output.append(expert_output)
            else:
                mlp_output.append([])

        # Second residual connection
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = layernorm_input

        # Final output with dropout and residual connection
        output = []
        for i in range(len(mlp_output)):
            if mlp_output[i] != []:
                output.append(torch.nn.functional.dropout(mlp_output[i], p=self.hidden_dropout, training=self.training))
                output[i] = residual[i] + output[i]
            else:
                output.append([])

        return output, kv_cache


class GLMTransformer(torch.nn.Module):
    """Transformer class."""

    def __init__(self, config: ChatGLMConfig, device=None):
        super(GLMTransformer, self).__init__()

        self.fp32_residual_connection = config.fp32_residual_connection
        self.post_layer_norm = config.post_layer_norm

        # Number of layers.
        self.num_layers = config.num_layers

        # Transformer layers.
        def build_layer(layer_number):
            return GLMBlock(config, layer_number, device=device)

        self.layers = torch.nn.ModuleList([build_layer(i + 1) for i in range(self.num_layers)])

        if self.post_layer_norm:
            LayerNormFunc = RMSNorm if config.rmsnorm else LayerNorm
            # Final layer norm before output.
            self.final_layernorm = LayerNormFunc(config.hidden_size, eps=config.layernorm_epsilon, device=device,
                                                 dtype=config.torch_dtype)

        self.gradient_checkpointing = False

    def _get_layer(self, layer_number):
        return self.layers[layer_number]

    def forward(
            self, hidden_states, attention_mask, rotary_pos_emb, kv_caches=None,
            use_cache: Optional[bool] = True,
            output_hidden_states: Optional[bool] = False,
    ):
        if not kv_caches:
            kv_caches = [None for _ in range(self.num_layers)]
        presents = () if use_cache else None
        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        all_self_attentions = None
        all_hidden_states = () if output_hidden_states else None
        for index in range(self.num_layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer = self._get_layer(index)
            if self.gradient_checkpointing and self.training:
                layer_ret = torch.utils.checkpoint.checkpoint(
                    layer,
                    hidden_states,
                    attention_mask,
                    rotary_pos_emb,
                    kv_caches[index],
                    use_cache,
                    use_reentrant=False
                )
            else:
                layer_ret = layer(
                    hidden_states,
                    attention_mask,
                    rotary_pos_emb,
                    kv_cache=kv_caches[index],
                    use_cache=use_cache
                )
            hidden_states, kv_cache = layer_ret
            if use_cache:
                # token by token decoding, use tuple format
                if kv_caches[0] is not None:
                    presents = presents + (kv_cache,)
                # prefilling in decoding, use tensor format to save cuda memory
                else:
                    if len(presents) == 0:
                        presents = kv_cache
                    else:
                        presents = torch.cat((presents, kv_cache.to(presents.device)), dim=0)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # Final layer norm.
        if self.post_layer_norm:
            hidden_states = self.final_layernorm(hidden_states)

        return hidden_states, presents, all_hidden_states, all_self_attentions


class GLMTransformerMot(torch.nn.Module):
    """Transformer class."""

    def __init__(self, config: ChatGLMConfig, device=None):
        super(GLMTransformerMot, self).__init__()

        self.fp32_residual_connection = config.fp32_residual_connection
        self.post_layer_norm = config.post_layer_norm
        self.n_experts = 2
        self.n_modalities = 6
        # Number of layers.
        self.num_layers = config.num_hidden_layers
        self.num_layers_seccond_modality = config.num_layers

        # Transformer layers.
        def build_layer_original(layer_number):
            return GLMBlockOriginal(config, layer_number, device=device)

        # Transformer layers.
        def build_layer_mot(layer_number):
            return GLMBlockMot(config, layer_number, device=device)

        if self.num_layers_seccond_modality == 40:
            self.layers = torch.nn.ModuleList([build_layer_mot(i + 1) for i in range(self.num_layers)])
        elif self.num_layers_seccond_modality < 40:
            mot_layers = [idx for idx in range(config.num_layers)]  # use mot for the first num_layers layers
            self.layers = nn.ModuleList([
                build_layer_mot(layer_idx+1) if layer_idx in mot_layers 
                else build_layer_original(layer_idx+1) 
                for layer_idx in range(config.num_hidden_layers)
            ])
        else:
            raise ValueError(f"Invalid layer number: {config.num_layers}")


        if self.post_layer_norm:
            LayerNormFunc = RMSNorm if config.rmsnorm else LayerNorm
            # Final layer norm before output.
            self.local_experts_final_layernorm = torch.nn.ModuleList([
                LayerNormFunc(config.hidden_size, eps=config.layernorm_epsilon, device=device,
                                                    dtype=config.torch_dtype),
                LayerNormFunc(config.mot_settings['mot_hidden_size'], eps=config.layernorm_epsilon, device=device,
                                                    dtype=config.torch_dtype)
            ])

        self.gradient_checkpointing = False

    def _get_layer(self, layer_number):
        return self.layers[layer_number]

    def forward(
            self, hidden_states, attention_mask, rotary_pos_emb, kv_caches=None,
            use_cache: Optional[bool] = True,
            output_hidden_states: Optional[bool] = False,
            modality_masks: Optional[torch.Tensor] = None,
            modality_masks_full: Optional[torch.Tensor] = None,
            past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
    ):
        if not kv_caches:
            kv_caches = [[None, None] for _ in range(self.num_layers_seccond_modality)]
        presents = kv_caches if use_cache else None
        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        all_self_attentions = None
        all_hidden_states = () if output_hidden_states else None
        for index in range(self.num_layers_seccond_modality):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer = self._get_layer(index)
            layer_past_key_values = None
            if past_key_values is not None:
                if isinstance(past_key_values, (list, tuple)) and index < len(past_key_values):
                    layer_past_key_values = past_key_values[index]
                else:
                    layer_past_key_values = past_key_values

            if self.gradient_checkpointing and self.training:
                layer_ret = torch.utils.checkpoint.checkpoint(
                    layer,
                    hidden_states,
                    attention_mask,
                    rotary_pos_emb,
                    kv_caches[index],
                    use_cache,
                    use_reentrant=False,
                    modality_masks=modality_masks,
                    modality_masks_full=modality_masks_full,
                    past_key_values=layer_past_key_values
                )
            else:
                layer_ret = layer(
                    hidden_states,
                    attention_mask,
                    rotary_pos_emb,
                    kv_cache=kv_caches[index],
                    use_cache=use_cache,
                    modality_masks=modality_masks,
                    modality_masks_full=modality_masks_full,
                    past_key_values=layer_past_key_values
                )
            hidden_states, kv_cache = layer_ret
            if use_cache:
                presents[index] = kv_cache

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # Final layer norm.
        if self.post_layer_norm:
            if hidden_states[0] != []:
                if hidden_states[0].shape[1] == 1:
                    modality_masks = modality_masks[:, :, :1]
            merged_output = []
            for i in range(len(hidden_states)):
                if modality_masks[i].sum() != 0:
                    expert_input = hidden_states[i]
                    expert_output = self.local_experts_final_layernorm[i](expert_input)
                    merged_output.append(expert_output)
                else:
                    merged_output.append([])

            hidden_states = merged_output

        return hidden_states, presents, all_hidden_states, all_self_attentions

    def forward_second_expert(
            self, hidden_states, attention_mask, rotary_pos_emb, kv_caches=None,
            use_cache: Optional[bool] = True,
            output_hidden_states: Optional[bool] = False,
            expert_masks: Optional[torch.Tensor] = None,
            expert_masks_full: Optional[torch.Tensor] = None,
            past_key_values: Optional[Tuple] = None,
    ):
        """
        Forward pass with KV cache reuse from T1
        """
        if not kv_caches:
            kv_caches = [[None, None] for _ in range(self.num_layers_seccond_modality)]
        presents = kv_caches if use_cache else None
        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        all_self_attentions = None
        all_hidden_states = () if output_hidden_states else None

        for index in range(self.num_layers_seccond_modality):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer = self._get_layer(index)

            if self.gradient_checkpointing and self.training:
                layer_ret = torch.utils.checkpoint.checkpoint(
                    layer.forward_second_expert,
                    hidden_states,
                    attention_mask,
                    rotary_pos_emb,
                    kv_caches[index],
                    use_cache,
                    use_reentrant=False,
                    expert_masks=expert_masks,
                    expert_masks_full=expert_masks_full,
                )
            else:
                layer_ret = layer.forward_second_expert(
                    hidden_states,
                    attention_mask,
                    rotary_pos_emb,
                    kv_cache=kv_caches[index],
                    use_cache=use_cache,
                    expert_masks=expert_masks,
                    expert_masks_full=expert_masks_full,
                )
            hidden_states, kv_cache = layer_ret
            if use_cache:
                presents[index] = kv_cache

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # Final layer norm.
        if self.post_layer_norm:
            hidden_states = self.local_experts_final_layernorm[1](hidden_states)


        return hidden_states, presents, all_hidden_states, all_self_attentions


    def forward_first_expert(
            self, hidden_states, attention_mask, rotary_pos_emb, kv_caches=None,
            use_cache: Optional[bool] = True,
            output_hidden_states: Optional[bool] = False,
            layer_limit: Optional[int] = None,
            is_generate: Optional[bool] = False,
    ):

        if is_generate:
            num_layers_first_expert_forward = self.num_layers
        else:
            num_layers_first_expert_forward = self.num_layers_seccond_modality


        if not kv_caches:
            kv_caches = [[None, None] for _ in range(num_layers_first_expert_forward)]
        presents = kv_caches
        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        all_self_attentions = None
        all_hidden_states = () if output_hidden_states else None
        total_layers = len(self.layers)
        effective_limit = num_layers_first_expert_forward
        for index in range(effective_limit):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer = self._get_layer(index)
            if self.gradient_checkpointing and self.training:
                layer_ret = torch.utils.checkpoint.checkpoint(
                    layer.forward_first_expert,
                    hidden_states,
                    attention_mask,
                    rotary_pos_emb,
                    kv_caches[index],
                    use_cache,
                    use_reentrant=False,
                )
            else:
                layer_ret = layer.forward_first_expert(
                    hidden_states,
                    attention_mask,
                    rotary_pos_emb,
                    kv_cache=kv_caches[index],
                    use_cache=use_cache,
                )
            hidden_states, kv_cache = layer_ret
            presents[index] = kv_cache

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # Final layer norm.
        if self.post_layer_norm:
            merged_output = []
            if hidden_states[0] != []:
                expert_input = hidden_states[0]
                expert_output = self.local_experts_final_layernorm[0](expert_input)
                merged_output.append(expert_output)
            else:
                merged_output.append([])

            hidden_states = merged_output

        return hidden_states, presents, all_hidden_states, all_self_attentions


class ChatGLMPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and
    a simple interface for downloading and loading pretrained models.
    """

    is_parallelizable = False
    supports_gradient_checkpointing = True
    config_class = ChatGLMConfig
    base_model_prefix = "transformer"
    _no_split_modules = ["GLMBlock"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True

    def _init_weights(self, module: nn.Module):
        """Initialize the weights."""
        return

    def get_masks(self, input_ids, past_key_values, padding_mask=None, position_ids=None):
        """
        Generate attention masks for the model.
        
        Args:
            input_ids: Input token IDs
            past_key_values: Past key-value pairs from previous forward passes
            padding_mask: Optional padding mask
            position_ids: Optional position IDs
            
        Returns:
            full_attention_mask: Full attention mask tensor
        """
        if self.config._attn_implementation == "flash_attention_2":
            if padding_mask is not None and not padding_mask.all():
                return padding_mask
            return None
        if self.training:
            batch_size, seq_length = padding_mask.shape
        else:
            batch_size, seq_length = input_ids.shape
        full_attention_mask = torch.ones(batch_size, seq_length, seq_length, device=input_ids.device)
        full_attention_mask.tril_()
        past_length = 0
        if past_key_values and not self.training:
            # Calculate past length based on cached key-value pairs
            # Note: first dim is layer num, second dim is modality num
            if past_key_values[0][1] is not None and position_ids is not None:
                past_length = padding_mask.shape[1] - seq_length + past_key_values[0][1][0].shape[2]
            elif past_key_values[0][0] is not None:
                past_length = padding_mask.shape[1] - seq_length

        if past_length:
            full_attention_mask = torch.cat((torch.ones(batch_size, seq_length, past_length,
                                                        device=input_ids.device), full_attention_mask), dim=-1)
        if padding_mask is not None:
            full_attention_mask = full_attention_mask * padding_mask.unsqueeze(1)
        if not past_length and padding_mask is not None:
            full_attention_mask -= padding_mask.unsqueeze(-1) - 1
        full_attention_mask = (full_attention_mask < 0.5).bool()
        full_attention_mask.unsqueeze_(1)
        return full_attention_mask

    def get_position_ids(self, input_ids, device):
        """
        Generate position IDs for input tokens.
        
        Args:
            input_ids: Input token IDs
            device: Device to place position IDs on
            
        Returns:
            position_ids: Position ID tensor
        """
        batch_size, seq_length = input_ids.shape
        position_ids = torch.arange(seq_length, dtype=torch.long, device=device).unsqueeze(0).repeat(batch_size, 1)
        return position_ids

class Embedding(torch.nn.Module):
    """Language model embeddings."""

    def __init__(self, config: ChatGLMConfig, device=None):
        super(Embedding, self).__init__()

        self.hidden_size = config.hidden_size
        # Word embeddings (parallel).
        self.word_embeddings = nn.Embedding(
            config.padded_vocab_size,
            self.hidden_size,
            dtype=config.torch_dtype,
            device=device
        )
        self.fp32_residual_connection = config.fp32_residual_connection

    def forward(self, input_ids):
        """
        Forward pass for language model embeddings.
        
        Args:
            input_ids: Input token IDs
            
        Returns:
            words_embeddings: Word embedding tensor
        """
        words_embeddings = self.word_embeddings(input_ids)
        embeddings = words_embeddings
        # If the input flag for fp32 residual connection is set, convert for float.
        if self.fp32_residual_connection:
            embeddings = embeddings.float()
        return embeddings


class EmbeddingMot(torch.nn.Module):
    """Language model embeddings."""

    def __init__(self, config: ChatGLMConfig, device=None):
        super(EmbeddingMot, self).__init__()

        self.n_modalities = 2
        self.hidden_size = config.hidden_size

        self.word_embeddings = torch.nn.ModuleList([
            nn.Embedding(
                config.padded_vocab_size,
                config.hidden_size,
                dtype=config.torch_dtype,
                device=device
            ),
            nn.Embedding(
                config.mot_settings['mot_vocab_size'],
                config.mot_settings['mot_hidden_size'],
                dtype=config.torch_dtype,
                device=device
            )
        ])
        self.fp32_residual_connection = config.fp32_residual_connection


    def forward(self, input_ids, modality_masks):
        """
        Forward pass for motion embeddings with modality support.
        
        Args:
            input_ids: Input token IDs with modality information
            modality_masks: Masks for different modalities
            
        Returns:
            words_embeddings: List of embedding tensors for each modality
        """
        if input_ids[0]['original_length'] == 1:
            modality_masks = modality_masks[:, :, :1]
        words_embeddings = []
        for i in range(self.n_modalities):
            if modality_masks[i].sum() == 0:
                words_embeddings.append([])
            else:
                modality_batches_i = input_ids[i]['tokens']
                words_embeddings.append(self.word_embeddings[i](modality_batches_i))

        embeddings = words_embeddings
        # If the input flag for fp32 residual connection is set, convert for float.
        if self.fp32_residual_connection:
            for i in range(self.n_modalities):
                if embeddings[i] is not None:
                    embeddings[i] = embeddings[i].float()

        return embeddings

    def forward_first_expert(self, input_ids):
        """
        Forward pass for first expert embedding layer.
        
        Args:
            input_ids: Input token IDs
            
        Returns:
            words_embeddings: List containing embedding tensor for first expert
        """
        words_embeddings = [self.word_embeddings[0](input_ids)]

        if self.fp32_residual_connection and isinstance(words_embeddings[0], torch.Tensor):
            words_embeddings[0] = words_embeddings[0].float()

        return words_embeddings

    def forward_second_expert(self, input_ids):
        """
        Forward pass for second expert embedding layer.
        
        Args:
            input_ids: Input token IDs
            
        Returns:
            words_embeddings: Embedding tensor for second expert
        """
        words_embeddings = self.word_embeddings[1](input_ids)

        embeddings = words_embeddings
        # If the input flag for fp32 residual connection is set, convert for float.
        if self.fp32_residual_connection:
            embeddings = embeddings.float()

        return embeddings

class EmbeddingMotModality_2(torch.nn.Module):
    """Language model embeddings."""

    def __init__(self, config: ChatGLMConfig, device=None):
        super(EmbeddingMotModality_2, self).__init__()

        self.n_modalities = 2
        self.hidden_size = config.hidden_size
        # Word embeddings (parallel).
        # Initialize word embeddings for text and motion modalities
        self.word_embeddings = torch.nn.ModuleList([
            nn.Embedding(
                config.padded_vocab_size,
                config.hidden_size,
                dtype=config.torch_dtype,
                device=device
            ),
            nn.Embedding(
                config.mot_settings['mot_vocab_size'],
                config.mot_settings['mot_hidden_size'],
                dtype=config.torch_dtype,
                device=device
            )
        ])
        self.fp32_residual_connection = config.fp32_residual_connection

    def forward(self, input_ids):
        """
        Forward pass for embedding layer.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            
        Returns:
            embeddings: Token embeddings [batch_size, seq_len, hidden_size]
        """
        # Get embeddings for motion tokens (using index 1 of the embedding list)
        embeddings = self.word_embeddings[1](input_ids)

        # If the input flag for fp32 residual connection is set, convert for float.
        if self.fp32_residual_connection:
            embeddings = embeddings.float()

        return embeddings


class ChatGLMModelMotDualTransformer(ChatGLMPreTrainedModel):
    def __init__(self, config: ChatGLMConfig, device=None, empty_init=True):
        super().__init__(config)
        if empty_init:
            init_method = skip_init
        else:
            init_method = default_init
        init_kwargs = {}
        if device is not None:
            init_kwargs["device"] = device
        self.embedding = init_method(EmbeddingMot, config, **init_kwargs)

        self.num_layers = config.num_layers
        self.multi_query_group_num = config.multi_query_group_num
        self.kv_channels = config.kv_channels
        self.t1_layer_limit = getattr(config, "t1_layer_limit", config.num_layers)

        # Rotary positional embeddings
        self.seq_length = config.seq_length
        rotary_dim = (
            config.hidden_size // config.num_attention_heads if config.kv_channels is None else config.kv_channels
        )

        self.rotary_pos_emb = RotaryEmbedding(rotary_dim // 2, rope_ratio=config.rope_ratio,
                                              original_impl=config.original_rope,
                                              device=device, dtype=config.torch_dtype)
        self.encoder = init_method(GLMTransformerMot, config, **init_kwargs)

        self.output_layer = nn.ModuleList([
            init_method(nn.Linear, config.hidden_size, config.padded_vocab_size, bias=False,
                                        dtype=config.torch_dtype, **init_kwargs),
            init_method(nn.Linear, config.mot_settings['mot_hidden_size'], config.mot_settings['mot_vocab_size'], bias=False,
                                        dtype=config.torch_dtype, **init_kwargs)
        ])

    def _build_attention_mask(self, input_ids, attention_mask, past_key_values):
        """Normalize attention mask shape based on the active attention implementation."""
        attn_impl = getattr(self.config, "_attn_implementation", "flash_attention_2")

        if attn_impl == "flash_attention_2":
            if attention_mask is None:
                return None
            if attention_mask.dim() == 2:
                return attention_mask.to(dtype=torch.bool, device=input_ids.device)
            # Flash attention expects 2D masks; collapse higher-rank masks if needed
            return attention_mask.squeeze().to(dtype=torch.bool, device=input_ids.device)

        # Eager/SDPA paths expect a 4D attention mask aligned with current sequence length.
        if attention_mask is None:
            return self.get_masks(input_ids, past_key_values, padding_mask=None)

        if attention_mask.dim() >= 3:
            return self.get_masks(input_ids, past_key_values, padding_mask=attention_mask)

        batch_size, seq_length = input_ids.shape
        device = input_ids.device
        dtype = torch.bool

        pad_mask = attention_mask.to(dtype=dtype, device=device)
        # allowed positions: causal & both tokens not padded
        causal_mask = torch.ones(batch_size, seq_length, seq_length, device=device, dtype=dtype).tril_()
        allowed = causal_mask & pad_mask.unsqueeze(-1) & pad_mask.unsqueeze(-2)
        forbidden = ~allowed
        forbidden = forbidden.unsqueeze(1)  # [b, 1, seq, seq]
        return forbidden


    def set_t1_layer_limit(self, layer_limit: int):
        """
        Set the layer limit for the first transformer (T1).
        
        Args:
            layer_limit: Maximum number of layers to process in T1
        """
        total_layers = len(getattr(self.encoder, "layers", []))
        if total_layers:
            self.t1_layer_limit = max(0, min(layer_limit, total_layers))
        else:
            self.t1_layer_limit = max(0, layer_limit)
        if hasattr(self.config, "t1_layer_limit"):
            self.config.t1_layer_limit = self.t1_layer_limit

    def get_input_embeddings(self):
        """Get the input embeddings layer."""
        return self.embedding.word_embeddings

    def set_input_embeddings(self, value):
        """Set the input embeddings layer."""
        self.embedding.word_embeddings = value

    def forward(
            self,
            input_ids,
            modality_batches,
            position_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.BoolTensor] = None,
            full_attention_mask: Optional[torch.BoolTensor] = None,
            past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            modality_masks: Optional[torch.Tensor] = None,
            modality_masks_full: Optional[torch.Tensor] = None,
            position_encoding_indices: Optional[torch.Tensor] = None,
            frozen_hidden_states: Optional[torch.Tensor] = None,
            frozen_mask: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass for dual transformer model with modality support.
        
        Args:
            input_ids: Input token IDs
            modality_batches: Batched modality data
            position_ids: Optional position IDs
            attention_mask: Optional attention mask
            full_attention_mask: Optional full attention mask
            past_key_values: Past key-value pairs from previous forward passes
            inputs_embeds: Optional pre-computed input embeddings
            use_cache: Whether to use caching
            output_attentions: Whether to output attention weights
            output_hidden_states: Whether to output hidden states from all layers
            return_dict: Whether to return a dictionary instead of a tuple
            modality_masks: Masks for different modalities
            modality_masks_full: Full modality masks
            position_encoding_indices: Optional position encoding indices
            frozen_hidden_states: Optional frozen hidden states
            frozen_mask: Optional frozen mask
            
        Returns:
            BaseModelOutputWithPast or tuple containing model outputs
        """
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        n_modalities = modality_masks.shape[0]
        mod_0_1 = torch.logical_or(modality_masks[0], modality_masks[1])
        mod_2 = modality_masks[2]
        for modality_idx in range(3, n_modalities):
            mod_2 = torch.logical_or(mod_2, modality_masks[modality_idx])
        modality_masks_new = torch.stack([mod_0_1, mod_2], dim=0)


        if modality_masks_full is not None:
            mod_0_1_full = torch.logical_or(modality_masks_full[0], modality_masks_full[1])
            mod_2_full = modality_masks_full[2]
            for modality_idx in range(3, n_modalities):
                mod_2_full = torch.logical_or(mod_2_full, modality_masks_full[modality_idx])
            modality_masks_new_full = torch.stack([mod_0_1_full, mod_2_full], dim=0)
        else:
            modality_masks_new_full = None


        seq_length = modality_batches[0]['original_length']
        if position_ids is not None and isinstance(position_ids, torch.Tensor):
            max_position = int(position_ids.max().item()) + 1
            seq_length = max(seq_length, max_position)

        if inputs_embeds is None:
            inputs_embeds = self.embedding(modality_batches, modality_masks_new)

        if full_attention_mask is None:
            full_attention_mask = self._build_attention_mask(input_ids, attention_mask, past_key_values)


        batch_size = None
        if input_ids is not None:
            batch_size = input_ids.shape[0]
        elif 0 in modality_batches:
            batch_size = modality_batches[0]['tokens'].shape[0]

        rotary_pos_emb = compute_rotary_embeddings_from_precomputed_indices(
            rotary_embedding_module=self.rotary_pos_emb,
            position_indices=position_encoding_indices,
            seq_length=seq_length,
            batch_size=batch_size,
        )

        if position_ids is not None:
            for i in range(len(position_ids)):
                rotary_pos_emb = rotary_pos_emb[:, position_ids[i]] ## Todo need to be updated to support batch

        # Run encoder.
        hidden_states, presents, all_hidden_states, all_self_attentions = self.encoder(
            inputs_embeds, full_attention_mask, rotary_pos_emb=rotary_pos_emb,
            kv_caches=past_key_values, use_cache=use_cache, output_hidden_states=output_hidden_states,
            modality_masks=modality_masks_new,
            modality_masks_full=modality_masks_new_full,
        )
        if presents is not None and type(presents) is torch.Tensor:
            presents = presents.split(1, dim=0)
            presents = list(presents)
            presents = [list(x.squeeze(0).split(1, dim=0)) for x in presents]
            presents = [tuple([x.squeeze(0) for x in y]) for y in presents]
            presents = tuple(presents)

        if not return_dict:
            return tuple(v for v in [hidden_states, presents, all_hidden_states, all_self_attentions] if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )

    def forward_first_expert(
            self,
            input_ids,
            position_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.BoolTensor] = None,
            full_attention_mask: Optional[torch.BoolTensor] = None,
            past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            use_cache: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            position_encoding_indices: Optional[torch.Tensor] = None,
            is_generate: Optional[bool] = False,
    ):
        """
        Process first modality (modalities 0,1) with limited layers and generate KV cache
        """
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        seq_length = input_ids.shape[1]
        if position_ids is not None and isinstance(position_ids, torch.Tensor):
            max_position = int(position_ids.max().item()) + 1
            seq_length = max(seq_length, max_position)

        if inputs_embeds is None:
            inputs_embeds = self.embedding.forward_first_expert(input_ids)

        if full_attention_mask is None:
            if (attention_mask is not None and not attention_mask.all()) or (past_key_values and seq_length != 1):
                full_attention_mask = self.get_masks(input_ids, past_key_values, padding_mask=attention_mask, position_ids=position_ids)

        # Create rotary position embeddings
        batch_size = input_ids.shape[0]
        rotary_pos_emb = self.rotary_pos_emb(seq_length)
        rotary_pos_emb = rotary_pos_emb[None, :, :, :].repeat(batch_size, 1, 1, 1)

        if position_ids is not None:
            for i in range(len(position_ids)):
                rotary_pos_emb = rotary_pos_emb[:, position_ids[i]]

        # Run encoder with limited layers (only first few layers for T1)
        # Note: This will be handled by the encoder's layer limit mechanism
        hidden_states, presents, all_hidden_states, all_self_attentions = self.encoder.forward_first_expert(
            inputs_embeds,
            full_attention_mask,
            rotary_pos_emb=rotary_pos_emb,
            kv_caches=past_key_values,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            layer_limit=self.t1_layer_limit,
            is_generate=is_generate,
        )


        if presents is not None and type(presents) is torch.Tensor:
            presents = presents.split(1, dim=0)
            presents = list(presents)
            presents = [list(x.squeeze(0).split(1, dim=0)) for x in presents]
            presents = [tuple([x.squeeze(0) for x in y]) for y in presents]
            presents = tuple(presents)

        if not return_dict:
            return tuple(v for v in [hidden_states, presents, all_hidden_states, all_self_attentions] if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


    def forward_second_expert(
            self,
            input_ids,
            position_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.BoolTensor] = None,
            full_attention_mask: Optional[torch.BoolTensor] = None,
            past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            use_cache: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            expert_masks: Optional[torch.Tensor] = None,
            expert_masks_full: Optional[torch.Tensor] = None,
            position_encoding_indices: Optional[torch.Tensor] = None,
            **kwargs
    ):
        """
        Process all modalities (0,1,2,3,4,5) in interleaved format with KV cache reuse
        """
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        seq_length = input_ids.shape[1]

        if inputs_embeds is None:
            inputs_embeds = self.embedding.forward_second_expert(input_ids)

        past_length = 0

        ## TODO: this part is important during inference, we need to ensure the full_attention_mask used for cache is correct
        if full_attention_mask is None:
            if (attention_mask is not None and not attention_mask.all()) or (past_key_values and seq_length != 1):
                full_attention_mask = self.get_masks(input_ids, past_key_values, padding_mask=attention_mask)
            else:
                full_attention_mask = attention_mask

        # Create rotary position embeddings
        batch_size = input_ids.shape[0]

        rotary_pos_emb = compute_rotary_embeddings_from_precomputed_indices(
            rotary_embedding_module=self.rotary_pos_emb,
            position_indices=position_encoding_indices,
            seq_length=seq_length,
            batch_size=batch_size,
        )


        #         rotary_pos_emb = rotary_pos_emb[:, position_ids[i]]

        # Run encoder with KV cache reuse
        # The encoder will handle merging the cached KV from T1
        hidden_states, presents, all_hidden_states, all_self_attentions = self.encoder.forward_second_expert(
            inputs_embeds, full_attention_mask, rotary_pos_emb=rotary_pos_emb,
            kv_caches=past_key_values, use_cache=use_cache, output_hidden_states=output_hidden_states,
            expert_masks=expert_masks,
            expert_masks_full=expert_masks_full,
        )
        if presents is not None and type(presents) is torch.Tensor:
            presents = presents.split(1, dim=0)
            presents = list(presents)
            presents = [list(x.squeeze(0).split(1, dim=0)) for x in presents]
            presents = [tuple([x.squeeze(0) for x in y]) for y in presents]
            presents = tuple(presents)

        if not return_dict:
            return tuple(v for v in [hidden_states, presents, all_hidden_states, all_self_attentions] if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class ChatGLMTransformer01(ChatGLMPreTrainedModel):
    """
    Transformer 1: Processes modalities 0,1 (text+audio)
    - Frozen during training, no gradient updates
    - Supports layer limit (layer_limit)
    - Provides KV cache to T2
    """

    def __init__(self, config: ChatGLMConfig, device=None, empty_init=True, layer_limit=None):
        super().__init__(config)

        if empty_init:
            init_method = skip_init
        else:
            init_method = default_init
        init_kwargs = {}
        if device is not None:
            init_kwargs["device"] = device

        # Only processes modalities 0,1, so only needs one embedding
        self.embedding = init_method(EmbeddingMot, config, **init_kwargs)

        self.num_layers = config.num_layers
        self.layer_limit = layer_limit or config.num_layers
        self.multi_query_group_num = config.multi_query_group_num
        self.kv_channels = config.kv_channels

        # Rotary positional embeddings
        self.seq_length = config.seq_length
        rotary_dim = (
            config.hidden_size // config.num_attention_heads if config.kv_channels is None else config.kv_channels
        )

        self.rotary_pos_emb = RotaryEmbedding(rotary_dim // 2, rope_ratio=config.rope_ratio,
                                              original_impl=config.original_rope,
                                              device=device, dtype=config.torch_dtype)

        # Create encoder with layer limit support
        self.encoder = init_method(GLMTransformerMot, config, **init_kwargs)

        # Only use first output layer (text+audio)
        self.output_layer = init_method(nn.Linear, config.hidden_size, config.padded_vocab_size, bias=False,
                                       dtype=config.torch_dtype, **init_kwargs)

        # Freeze all parameters
        self.eval()
        for param in self.parameters():
            param.requires_grad = False

    def get_input_embeddings(self):
        """Get the input embeddings layer."""
        return self.embedding.word_embeddings

    def set_input_embeddings(self, value):
        """Set the input embeddings layer."""
        self.embedding.word_embeddings = value

    def forward_modalities_01(
        self,
        input_ids,
        modality_masks,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.BoolTensor] = None,
        full_attention_mask: Optional[torch.BoolTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        modality_masks_full: Optional[torch.Tensor] = None,
        position_encoding_indices: Optional[torch.Tensor] = None,
        layer_limit: Optional[int] = None,
    ):
        """
        Process modalities 0,1 and generate KV cache
        """
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Ensure only processing modalities 0,1
        if modality_masks.shape[0] != 2:
            # If input has 6 modalities, merge to 2
            mod_0_1 = torch.logical_or(modality_masks[0], modality_masks[1])
            modality_masks = torch.stack([mod_0_1, torch.zeros_like(mod_0_1)], dim=0)

        # Create modality_batches for embedding
        modality_batches = self._create_modality_batches(input_ids, modality_masks)

        seq_length = modality_batches[0]['original_length']

        if position_ids is not None and isinstance(position_ids, torch.Tensor):
            max_position = int(position_ids.max().item()) + 1
            seq_length = max(seq_length, max_position)

        if inputs_embeds is None:
            inputs_embeds = self.embedding(modality_batches, modality_masks)

        if full_attention_mask is None:
            if (attention_mask is not None and not attention_mask.all()) or (past_key_values and seq_length != 1):
                full_attention_mask = self.get_masks(input_ids, past_key_values, padding_mask=attention_mask)

        batch_size = input_ids.shape[0] if input_ids is not None else modality_batches[0]['tokens'].shape[0]
        # Create rotary position embeddings
        rotary_pos_emb = compute_rotary_embeddings_from_precomputed_indices(
            rotary_embedding_module=self.rotary_pos_emb,
            position_indices=position_encoding_indices,
            seq_length=seq_length,
            batch_size=batch_size,
        )

        # Use limited layers for forward pass
        actual_layer_limit = layer_limit or self.layer_limit
        encoder_outputs = self.encoder.forward_with_layer_limit(
            hidden_states=inputs_embeds,
            attention_mask=full_attention_mask,
            rotary_pos_emb=rotary_pos_emb,
            kv_caches=past_key_values,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            modality_masks=modality_masks,
            modality_masks_full=modality_masks_full,
            layer_limit=actual_layer_limit,
        )

        hidden_states = encoder_outputs[0]

        # Use first expert output (modalities 0,1)
        if isinstance(hidden_states, list):
            final_hidden_states = hidden_states[0]  # Use first expert output
        else:
            final_hidden_states = hidden_states

        # Pass through output layer
        lm_logits = self.output_layer(final_hidden_states)

        if not return_dict:
            output = (lm_logits,) + encoder_outputs[1:]
            return output

        return BaseModelOutputWithPast(
            last_hidden_state=final_hidden_states,
            past_key_values=encoder_outputs.past_key_values if hasattr(encoder_outputs, 'past_key_values') else None,
            hidden_states=encoder_outputs.hidden_states if hasattr(encoder_outputs, 'hidden_states') else None,
            attentions=encoder_outputs.attentions if hasattr(encoder_outputs, 'attentions') else None,
        )

    def _create_modality_batches(self, input_ids, modality_masks):
        """Create modality_batches for embedding"""
        batch_size, seq_len = input_ids.shape

        # Create simple modality_batches structure
        modality_batches = [{
            'input_ids': input_ids,
            'labels': input_ids.clone(),
            'original_length': seq_len
        }]

        return modality_batches


class ChatGLMTransformerInterleaved(ChatGLMPreTrainedModel):
    """
    Transformer 2: Processes all modalities (0,1,2,3,4,5)
    - Main training target
    - Reuses T1's KV cache
    - Supports interleaved attention
    """

    def __init__(self, config: ChatGLMConfig, device=None, empty_init=True):
        super().__init__(config)

        if empty_init:
            init_method = skip_init
        else:
            init_method = default_init
        init_kwargs = {}
        if device is not None:
            init_kwargs["device"] = device

        # Initialize embedding layer for motion tokens
        # Note: This processes modalities 0,1, so only needs one embedding
        self.embedding = init_method(EmbeddingMot, config, **init_kwargs)

        # Model configuration parameters
        self.num_layers = config.num_layers
        self.multi_query_group_num = config.multi_query_group_num
        self.kv_channels = config.kv_channels

        # Initialize rotary positional embeddings
        self.seq_length = config.seq_length
        rotary_dim = (
            config.hidden_size // config.num_attention_heads if config.kv_channels is None else config.kv_channels
        )

        self.rotary_pos_emb = RotaryEmbedding(rotary_dim // 2, rope_ratio=config.rope_ratio,
                                              original_impl=config.original_rope,
                                              device=device, dtype=config.torch_dtype)

        # Create encoder with KV cache reuse support
        self.encoder = init_method(GLMTransformerMot, config, **init_kwargs)

        # Use motion output layer
        self.output_layer = init_method(nn.Linear, getattr(config, 'mot_settings', {}).get('mot_hidden_size', config.hidden_size), 
                                       getattr(config, 'mot_settings', {}).get('mot_vocab_size', config.vocab_size), bias=False,
                                       dtype=config.torch_dtype, **init_kwargs)

    def get_input_embeddings(self):
        """Get the input embeddings layer."""
        return self.embedding.word_embeddings

    def set_input_embeddings(self, value):
        """Set the input embeddings layer."""
        self.embedding.word_embeddings = value

    def forward_second_expert_with_cached_kv(
        self,
        input_ids,
        modality_masks,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.BoolTensor] = None,
        full_attention_mask: Optional[torch.BoolTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        modality_masks_full: Optional[torch.Tensor] = None,
        position_encoding_indices: Optional[torch.Tensor] = None,
        training: bool = True,
    ):
        """
        Forward pass for the second expert with cached key-value pairs from the first expert.
        
        This method processes all modalities (0,1,2,3,4,5) and reuses the KV cache from T1
        (the first expert transformer) for efficient inference.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            modality_masks: Masks for different modalities [num_modalities, batch_size, seq_len]
            position_ids: Optional position IDs for tokens
            attention_mask: Optional attention mask tensor
            full_attention_mask: Optional full attention mask
            past_key_values: Past key-value pairs from previous forward passes
            inputs_embeds: Optional pre-computed input embeddings
            use_cache: Whether to use caching for key-value pairs
            output_attentions: Whether to output attention weights
            output_hidden_states: Whether to output hidden states from all layers
            return_dict: Whether to return a dictionary instead of a tuple
            modality_masks_full: Full modality masks (takes precedence if provided)
            position_encoding_indices: Optional position encoding indices
            training: Whether the model is in training mode
            
        Returns:
            BaseModelOutputWithPast or tuple containing:
            - last_hidden_state: Final hidden states [batch_size, seq_len, hidden_size]
            - past_key_values: Updated key-value cache
            - hidden_states: All hidden states from each layer (if output_hidden_states=True)
            - attentions: All attention weights (if output_attentions=True)
        """
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Merge modalities 0,1 and 2,3,4,5 into 2 groups
        n_modalities = modality_masks.shape[0]
        mod_0_1 = torch.logical_or(modality_masks[0], modality_masks[1])
        mod_2 = modality_masks[2]
        for modality_idx in range(3, n_modalities):
            mod_2 = torch.logical_or(mod_2, modality_masks[modality_idx])
        modality_masks_new = torch.stack([mod_0_1, mod_2], dim=0)

        if modality_masks_full is not None:
            mod_0_1_full = torch.logical_or(modality_masks_full[0], modality_masks_full[1])
            mod_2_full = modality_masks_full[2]
            for modality_idx in range(3, n_modalities):
                mod_2_full = torch.logical_or(mod_2_full, modality_masks_full[modality_idx])
            modality_masks_new_full = torch.stack([mod_0_1_full, mod_2_full], dim=0)
        else:
            modality_masks_new_full = None

        # Create modality_batches for embedding
        modality_batches = self._create_modality_batches(input_ids, modality_masks_new)

        seq_length = modality_batches[0]['original_length']
        if position_ids is not None and isinstance(position_ids, torch.Tensor):
            max_position = int(position_ids.max().item()) + 1
            seq_length = max(seq_length, max_position)

        if inputs_embeds is None:
            inputs_embeds = self.embedding(modality_batches, modality_masks_new)

        if full_attention_mask is None:
            if (attention_mask is not None and not attention_mask.all()) or (past_key_values and seq_length != 1):
                full_attention_mask = self.get_masks(input_ids, past_key_values, padding_mask=attention_mask)

        batch_size = input_ids.shape[0] if input_ids is not None else modality_batches[0]['tokens'].shape[0]
        # Create rotary position embeddings
        rotary_pos_emb = compute_rotary_embeddings_from_precomputed_indices(
            rotary_embedding_module=self.rotary_pos_emb,
            position_indices=position_encoding_indices,
            seq_length=seq_length,
            batch_size=batch_size,
        )

        # Forward with cached KV from T1
        encoder_outputs = self.encoder.forward_second_expert_with_cached_kv(
            hidden_states=inputs_embeds,
            attention_mask=full_attention_mask,
            rotary_pos_emb=rotary_pos_emb,
            kv_caches=past_key_values,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            modality_masks=modality_masks_new,
            modality_masks_full=modality_masks_new_full,
            past_key_values=past_key_values,
        )

        hidden_states = encoder_outputs[0]

        # Use second expert output (motion)
        if isinstance(hidden_states, list):
            final_hidden_states = hidden_states[-1]  # Use second expert output
        else:
            final_hidden_states = hidden_states

        # Pass through output layer
        lm_logits = self.output_layer(final_hidden_states)

        if not return_dict:
            output = (lm_logits,) + encoder_outputs[1:]
            return output

        return BaseModelOutputWithPast(
            last_hidden_state=final_hidden_states,
            past_key_values=encoder_outputs.past_key_values if hasattr(encoder_outputs, 'past_key_values') else None,
            hidden_states=encoder_outputs.hidden_states if hasattr(encoder_outputs, 'hidden_states') else None,
            attentions=encoder_outputs.attentions if hasattr(encoder_outputs, 'attentions') else None,
        )

    def _create_modality_batches(self, input_ids, modality_masks):
        """Create modality_batches for embedding"""
        batch_size, seq_len = input_ids.shape

        # Create simple modality_batches structure
        modality_batches = [{
            'input_ids': input_ids,
            'labels': input_ids.clone(),
            'original_length': seq_len
        }]

        return modality_batches


class ChatGLMForConditionalGenerationMotExpertNum2(ChatGLMPreTrainedModel, GenerationMixin):
    def __init__(self, config: ChatGLMConfig, empty_init=True, device=None, t1_layer_limit=None):
        super().__init__(config)

        self.max_sequence_length = config.max_length
        self.config = config

        # Layer limit for T1 (how many layers to use when processing modalities 0,1)
        self.t1_layer_limit = t1_layer_limit or getattr(config, "t1_layer_limit", config.num_layers)
        self.config.t1_layer_limit = self.t1_layer_limit

        # Use existing transformer structure
        self.transformer = ChatGLMModelMotDualTransformer(config, empty_init=empty_init, device=device)
        self.transformer.set_t1_layer_limit(self.t1_layer_limit)

        # Configuration for dual transformer behavior
        self.n_modalities = 6  # 6 input modalities: text, audio, face, upper, lower, hand
        self.n_transformer_experts = 2  # 2 transformer experts in the model
        self.batch_processor = ModalityBatchProcessor(self.n_modalities)

    def _extract_past_from_model_output(self, outputs):
        return "past_key_values", outputs.get("past_key_values", None)

    def _update_model_kwargs_for_generation(
            self,
            outputs: ModelOutput,
            model_kwargs: Dict[str, Any],
            is_encoder_decoder: bool = False,
    ) -> Dict[str, Any]:
        # update past_key_values
        cache_name, cache = self._extract_past_from_model_output(outputs)
        model_kwargs[cache_name] = cache

        # update attention mask
        if "attention_mask" in model_kwargs:
            attention_mask = model_kwargs["attention_mask"]
            model_kwargs["attention_mask"] = torch.cat(
                [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
            )

        # update position ids
        if "position_ids" in model_kwargs:
            position_ids = model_kwargs["position_ids"]
            new_position_id = position_ids[..., -1:].clone()
            new_position_id += 1
            model_kwargs["position_ids"] = torch.cat(
                [position_ids, new_position_id], dim=-1
            )

        model_kwargs["is_first_forward"] = False
        return model_kwargs

    def prepare_inputs_for_generation(
            self,
            input_ids: torch.LongTensor,
            past_key_values: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            use_cache: Optional[bool] = None,
            is_first_forward: bool = True,
            modality_masks: Optional[torch.Tensor] = None,
            modality_masks_full: Optional[torch.Tensor] = None,
            position_encoding_indices: Optional[torch.Tensor] = None,
            **kwargs
    ) -> dict:
        # only last token for input_ids if past is not None
        if position_ids is None:
            position_ids = self.get_position_ids(input_ids, device=input_ids.device)
        if not is_first_forward:
            if past_key_values is not None:
                position_ids = position_ids[..., -1:]
                input_ids = input_ids[:, -1:]
                modality_masks = modality_masks[..., -1:]
        if position_encoding_indices is not None:
            position_encoding_indices = position_encoding_indices[..., -1:]
            return {
                "input_ids": input_ids,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
                "attention_mask": attention_mask,
                "return_last_logit": True,
                "use_cache": use_cache,
                "modality_masks": modality_masks,
                "modality_masks_full": modality_masks_full,
                "position_encoding_indices": position_encoding_indices,
            }
        else:
            return {
                "input_ids": input_ids,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
                "attention_mask": attention_mask,
                "return_last_logit": True,
                "use_cache": use_cache,
                "modality_masks": modality_masks,
                "modality_masks_full": modality_masks_full,
            }


    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            return_last_logit: Optional[bool] = False,
            modality_masks: Optional[torch.Tensor] = None,
            modality_masks_full: Optional[torch.Tensor] = None,
            position_encoding_indices: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass for conditional generation model with expert routing.
        
        Args:
            input_ids: Input token IDs
            position_ids: Optional position IDs
            attention_mask: Optional attention mask
            inputs_embeds: Optional pre-computed input embeddings
            labels: Optional labels for training
            use_cache: Whether to use caching
            output_attentions: Whether to output attention weights
            output_hidden_states: Whether to output hidden states from all layers
            return_dict: Whether to return a dictionary instead of a tuple
            return_last_logit: Whether to return only the last logit
            modality_masks: Masks for different modalities
            modality_masks_full: Full modality masks
            position_encoding_indices: Optional position encoding indices
            
        Returns:
            CausalLMOutputWithPast or tuple containing model outputs
        """
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # raw_modality_masks = modality_masks

        expert_batches, expert_masks = self.batch_processor.create_batch_aware_expert_inputs_labels(
            input_ids, labels, position_encoding_indices, modality_masks
        )

        t1_outputs = self.transformer.forward_first_expert(
            input_ids=expert_batches[0]['tokens'],
            attention_mask=expert_batches[0]['attention_mask'],
            inputs_embeds=None,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        past_key_values = getattr(t1_outputs, "past_key_values", None)

        transformer_outputs = self.transformer.forward_second_expert(
            input_ids=expert_batches[1]['tokens'],
            position_ids=position_ids,
            attention_mask=attention_mask, ### attention_mask or expert_batches[1]['attention_mask']?
            past_key_values=past_key_values,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            expert_masks=expert_masks,
            position_encoding_indices=expert_batches[1]['position_encoding_indices'],
        )

        hidden_states = transformer_outputs[0]
        final_output = self.transformer.output_layer[1](hidden_states)
        lm_logits = final_output

        if return_last_logit:
            last_modality_mask = modality_masks[:, :, -1] ### TODO: MODIFY FOR BATCH SIZE
            lm_logits = lm_logits[:, -1:]
        loss = None
        if labels is not None:
            lm_logits = lm_logits.to(torch.float32)
            labels_modality_2 = expert_batches[1]['labels']
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels_modality_2[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            lm_logits = lm_logits.to(final_output.dtype)
            loss = loss.to(final_output.dtype)


        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


    # Process output and add face tokens
    def add_face_body_tokens_to_output(self, output_ids, output_modality_masks):
        """
        Add face tokens after audio token segments and update modality masks
        """
        print("\n=== Adding Body Tokens ===")

        # Find all audio token positions (152353 <= token_id <= 168735)
        audio_positions = []
        for i, token_id in enumerate(output_ids[0]):
            if 152353 <= token_id <= 168735:
                audio_positions.append(i)

        print(f"Found {len(audio_positions)} audio tokens")

        if len(audio_positions) == 0:
            print("No audio tokens found, returning original output")
            return output_ids, output_modality_masks

        # Group audio positions by 26 tokens per group
        audio_groups = []
        current_group = []

        for i, pos in enumerate(audio_positions):
            current_group.append(pos)
            # If current group reaches 26 tokens, or next position is not continuous, end current group
            if len(current_group) == 26 or (i + 1 < len(audio_positions) and audio_positions[i + 1] != pos + 1):
                audio_groups.append(current_group)
                current_group = []

        # Handle the last group
        if current_group:
            audio_groups.append(current_group)

        print(f"Audio tokens divided into {len(audio_groups)} groups:")
        for i, group in enumerate(audio_groups):
            print(f"  Group {i+1}: {len(group)} tokens, positions {group[0]}-{group[-1]}")

        # Calculate total number of body tokens to insert
        total_body_tokens = len(audio_groups) * (52 + 13 + 13 + 13 + 1) ## 92

        new_length = output_ids.shape[1] + total_body_tokens

        print(f"Will add {total_body_tokens} modality motion, new length: {new_length}")

        # Create new output_ids and modality_masks
        # new_output_ids = torch.zeros(output_ids.shape[0], new_length, dtype=output_ids.dtype, device=output_ids.device)
        new_output_ids = 168736 * torch.ones(output_ids.shape[0], new_length, dtype=output_ids.dtype, device=output_ids.device)
        new_modality_masks = torch.zeros(output_modality_masks.shape[0], output_ids.shape[0], new_length, 
                                        dtype=output_modality_masks.dtype, device=output_modality_masks.device)

        # Copy original data and insert face tokens
        current_pos = 0
        new_pos = 0

        for group_idx, group in enumerate(audio_groups):
            # Copy content before audio group
            segment_length = group[0] - current_pos
            if segment_length > 0:
                new_output_ids[:, new_pos:new_pos + segment_length] = output_ids[:, current_pos:group[0]]
                new_modality_masks[:, :, new_pos:new_pos + segment_length] = output_modality_masks[:, :, current_pos:group[0]]
                new_pos += segment_length

            # Copy audio tokens
            audio_length = len(group)
            new_output_ids[:, new_pos:new_pos + audio_length] = output_ids[:, group[0]:group[-1] + 1]
            new_modality_masks[:, :, new_pos:new_pos + audio_length] = output_modality_masks[:, :, group[0]:group[-1] + 1]
            new_pos += audio_length

            # Add 52 face tokens (using token IDs in range 168736-168787)
            # Temporarily fill with 0 here, actual face tokens will add 168736 offset later
            # face_tokens = torch.arange(0, 53, dtype=output_ids.dtype, device=output_ids.device).unsqueeze(0)
            face_tokens = torch.zeros(1, 92, dtype=output_ids.dtype, device=output_ids.device)
            face_tokens[0, 0] = 1280
            new_output_ids[:, new_pos:new_pos + 92] = face_tokens

            # Set modality mask for face tokens (third modality set to True)
            new_modality_masks[0, :, new_pos:new_pos + 92] = False  # Not text modality
            new_modality_masks[1, :, new_pos:new_pos + 92] = False  # Not audio modality
            new_modality_masks[2, :, new_pos:new_pos + 92] = True   # Is face modality, face, is in the third modality

            new_pos += 92

            current_pos = group[-1] + 1

        # Copy remaining content
        if current_pos < output_ids.shape[1]:
            remaining_length = output_ids.shape[1] - current_pos
            new_output_ids[:, new_pos:new_pos + remaining_length] = output_ids[:, current_pos:]
            new_modality_masks[:, :, new_pos:new_pos + remaining_length] = output_modality_masks[:, :, current_pos:]

        print(f"Successfully added body tokens, {total_body_tokens} body tokens, final length: {new_output_ids.shape[1]}")
        return new_output_ids, new_modality_masks


    # Process output and add face tokens
    def add_body_tokens_to_output(self, output_ids, output_modality_masks):
        """
        Add face tokens after audio token segments and update modality masks
        """
        print("\n=== Adding Body Tokens ===")

        # Find all audio token positions (152353 <= token_id <= 168735)
        audio_positions = []
        for i, token_id in enumerate(output_ids[0]):
            # ### only for demo, please don't forget remove this code.
            #     continue
            if 152353 <= token_id <= 168735:
                audio_positions.append(i)

        print(f"Found {len(audio_positions)} audio tokens")

        if len(audio_positions) == 0:
            print("No audio tokens found, returning original output")
            return output_ids, output_modality_masks

        # Group audio positions by 26 tokens per group
        audio_groups = []
        current_group = []

        for i, pos in enumerate(audio_positions):
            current_group.append(pos)

            # ### only for demo, please don't forget remove this code.
            #     continue
            # If current group reaches 26 tokens, or next position is not continuous, end current group
            if len(current_group) == 26 or (i + 1 < len(audio_positions) and audio_positions[i + 1] != pos + 1):
                audio_groups.append(current_group)
                current_group = []

        # Handle the last group
        if current_group:
            audio_groups.append(current_group)

        print(f"Audio tokens divided into {len(audio_groups)} groups:")
        for i, group in enumerate(audio_groups):
            print(f"  Group {i+1}: {len(group)} tokens, positions {group[0]}-{group[-1]}")

        # Calculate total number of motion tokens to insert
        total_motion_tokens = len(audio_groups) * 40   ## 40 = 13 upper + 13 lower + 13 hand + 1 motion

        new_length = output_ids.shape[1] + total_motion_tokens

        print(f"Will add {total_motion_tokens} motion tokens, new length: {new_length}")

        # Create new output_ids and modality_masks
        # new_output_ids = torch.zeros(output_ids.shape[0], new_length, dtype=output_ids.dtype, device=output_ids.device)
        new_output_ids = 168736 * torch.ones(output_ids.shape[0], new_length, dtype=output_ids.dtype, device=output_ids.device)
        new_modality_masks = torch.zeros(output_modality_masks.shape[0], output_ids.shape[0], new_length, 
                                        dtype=output_modality_masks.dtype, device=output_modality_masks.device)

        # Copy original data and insert face tokens
        current_pos = 0
        new_pos = 0

        for group_idx, group in enumerate(audio_groups):
            # Copy content before audio group
            segment_length = group[0] - current_pos
            if segment_length > 0:
                new_output_ids[:, new_pos:new_pos + segment_length] = output_ids[:, current_pos:group[0]]
                new_modality_masks[:, :, new_pos:new_pos + segment_length] = output_modality_masks[:, :, current_pos:group[0]]
                new_pos += segment_length

            # Copy audio tokens
            audio_length = len(group)
            new_output_ids[:, new_pos:new_pos + audio_length] = output_ids[:, group[0]:group[-1] + 1]
            new_modality_masks[:, :, new_pos:new_pos + audio_length] = output_modality_masks[:, :, group[0]:group[-1] + 1]
            new_pos += audio_length


            # Add 14 upper tokens (using token IDs in range 168788-168801)
            motion_tokens = torch.zeros(1, 40, dtype=output_ids.dtype, device=output_ids.device)
            motion_tokens[0, 0] = 1280  ### 169505 - 169249 = 256 (<|begin_of_upper|> -> 169505)
            new_output_ids[:, new_pos:new_pos + 40] = motion_tokens

            # Set modality mask for motion tokens (third modality set to True)
            new_modality_masks[0, :, new_pos:new_pos + 40] = False  # Not text modality
            new_modality_masks[1, :, new_pos:new_pos + 40] = False  # Not audio modality
            new_modality_masks[2, :, new_pos:new_pos + 40] = True   # Is motion modality, motion is in the third modality
            new_pos += 40

            current_pos = group[-1] + 1

        # Copy remaining content
        if current_pos < output_ids.shape[1]:
            remaining_length = output_ids.shape[1] - current_pos
            new_output_ids[:, new_pos:new_pos + remaining_length] = output_ids[:, current_pos:]
            new_modality_masks[:, :, new_pos:new_pos + remaining_length] = output_modality_masks[:, :, current_pos:]

        print(f"Successfully added motion tokens, {total_motion_tokens} motion tokens, final length: {new_output_ids.shape[1]}")
        return new_output_ids, new_modality_masks


    # Process output and add face tokens
    def add_face_tokens_to_output(self, output_ids, output_modality_masks):
        """
        Add face tokens after audio token segments and update modality masks
        """
        print("\n=== Adding Face Tokens ===")

        # Find all audio token positions (152353 <= token_id <= 168735)
        audio_positions = []
        for i, token_id in enumerate(output_ids[0]):
            if 152353 <= token_id <= 168735:
                audio_positions.append(i)

        print(f"Found {len(audio_positions)} audio tokens")

        if len(audio_positions) == 0:
            print("No audio tokens found, returning original output")
            return output_ids, output_modality_masks

        # Group audio positions by 26 tokens per group
        audio_groups = []
        current_group = []

        for i, pos in enumerate(audio_positions):
            current_group.append(pos)
            # If current group reaches 26 tokens, or next position is not continuous, end current group
            if len(current_group) == 26 or (i + 1 < len(audio_positions) and audio_positions[i + 1] != pos + 1):
                audio_groups.append(current_group)
                current_group = []

        # Handle the last group
        if current_group:
            audio_groups.append(current_group)

        print(f"Audio tokens divided into {len(audio_groups)} groups:")
        for i, group in enumerate(audio_groups):
            print(f"  Group {i+1}: {len(group)} tokens, positions {group[0]}-{group[-1]}")

        # Calculate total number of face tokens to insert
        total_face_tokens = len(audio_groups) * 53

        new_length = output_ids.shape[1] + total_face_tokens

        print(f"Will add {total_face_tokens} face tokens, new length: {new_length}")

        # Create new output_ids and modality_masks
        # new_output_ids = torch.zeros(output_ids.shape[0], new_length, dtype=output_ids.dtype, device=output_ids.device)
        new_output_ids = 168736 * torch.ones(output_ids.shape[0], new_length, dtype=output_ids.dtype, device=output_ids.device)
        new_modality_masks = torch.zeros(output_modality_masks.shape[0], output_ids.shape[0], new_length, 
                                        dtype=output_modality_masks.dtype, device=output_modality_masks.device)

        # Copy original data and insert face tokens
        current_pos = 0
        new_pos = 0

        for group_idx, group in enumerate(audio_groups):
            # Copy content before audio group
            segment_length = group[0] - current_pos
            if segment_length > 0:
                new_output_ids[:, new_pos:new_pos + segment_length] = output_ids[:, current_pos:group[0]]
                new_modality_masks[:, :, new_pos:new_pos + segment_length] = output_modality_masks[:, :, current_pos:group[0]]
                new_pos += segment_length

            # Copy audio tokens
            audio_length = len(group)
            new_output_ids[:, new_pos:new_pos + audio_length] = output_ids[:, group[0]:group[-1] + 1]
            new_modality_masks[:, :, new_pos:new_pos + audio_length] = output_modality_masks[:, :, group[0]:group[-1] + 1]
            new_pos += audio_length

            # Add 52 face tokens (using token IDs in range 168736-168787)
            # Temporarily fill with 0 here, actual face tokens will add 168736 offset later
            # face_tokens = torch.arange(0, 53, dtype=output_ids.dtype, device=output_ids.device).unsqueeze(0)
            face_tokens = torch.zeros(1, 53, dtype=output_ids.dtype, device=output_ids.device)
            face_tokens[0, 0] = 1280
            new_output_ids[:, new_pos:new_pos + 53] = face_tokens

            # Set modality mask for face tokens (third modality set to True)
            new_modality_masks[0, :, new_pos:new_pos + 53] = False  # Not text modality
            new_modality_masks[1, :, new_pos:new_pos + 53] = False  # Not audio modality
            new_modality_masks[2, :, new_pos:new_pos + 53] = True   # Is face modality, face, is in the third modality

            new_pos += 53

            current_pos = group[-1] + 1

        # Copy remaining content
        if current_pos < output_ids.shape[1]:
            remaining_length = output_ids.shape[1] - current_pos
            new_output_ids[:, new_pos:new_pos + remaining_length] = output_ids[:, current_pos:]
            new_modality_masks[:, :, new_pos:new_pos + remaining_length] = output_modality_masks[:, :, current_pos:]

        print(f"Successfully added face tokens, final length: {new_output_ids.shape[1]}")
        return new_output_ids, new_modality_masks


    def forward_second_expert(
            self,
            input_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[Tuple[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            return_last_logit: Optional[bool] = False,
            modality_masks: Optional[torch.Tensor] = None,
            modality_masks_full: Optional[torch.Tensor] = None,
            position_encoding_indices: Optional[torch.Tensor] = None,
    ):
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        raw_modality_masks = modality_masks
        raw_modality_masks_full = modality_masks_full

        expert_batches, expert_masks = self.batch_processor.create_batch_aware_expert_inputs_labels(
            input_ids, labels, position_encoding_indices, raw_modality_masks
        )

        if raw_modality_masks_full is not None:
            n_modalities = raw_modality_masks_full.shape[0]
            # Vectorized modality combination
            expert_0_1 = torch.logical_or(raw_modality_masks_full[0], raw_modality_masks_full[1])
            if n_modalities > 2:
                expert_2 = raw_modality_masks_full[2:].any(dim=0)  # Vectorized OR across modalities 2+
            else:
                expert_2 = torch.zeros_like(expert_0_1)
            expert_masks_full = torch.stack([expert_0_1, expert_2], dim=0)


        transformer_outputs = self.transformer.forward_second_expert(
            input_ids=expert_batches[1]['tokens'],
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            expert_masks=expert_masks,
            expert_masks_full=expert_masks_full,
            position_encoding_indices=expert_batches[1]['position_encoding_indices'],
        )

        hidden_states = transformer_outputs[0]

        final_output = self.transformer.output_layer[1](hidden_states)
        lm_logits = final_output

        if return_last_logit:
            last_modality_mask = modality_masks[:, :, -1] ### TODO: MODIFY FOR BATCH SIZE
            lm_logits = lm_logits[:, -1:]
        loss = None
        if labels is not None:
            lm_logits = lm_logits.to(torch.float32)
            labels_modality_2 = expert_batches[1]['labels']
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels_modality_2[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            lm_logits = lm_logits.to(final_output.dtype)
            loss = loss.to(final_output.dtype)

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


    def forward_first_expert(
            self,
            input_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[Tuple[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            return_last_logit: Optional[bool] = False,
            modality_masks: Optional[torch.Tensor] = None,
            modality_masks_full: Optional[torch.Tensor] = None,
            position_encoding_indices: Optional[torch.Tensor] = None,
    ):
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        raw_modality_masks = modality_masks

        #     input_ids, labels, raw_modality_masks
        expert_batches, modality_masks_new = self.batch_processor.create_batch_aware_expert_inputs_labels(
            input_ids, labels, None, raw_modality_masks
        )

        is_generate = (return_last_logit is True) or (use_cache is True) or (past_key_values is not None)


        transformer_outputs = self.transformer.forward_first_expert(
            input_ids=expert_batches[0]['tokens'],
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            position_encoding_indices=position_encoding_indices,
            is_generate = is_generate
        )

        hidden_states = transformer_outputs[0]

        final_output = self.transformer.output_layer[0](hidden_states[0])
        lm_logits = final_output

        if return_last_logit:
            last_modality_mask = modality_masks[:, :, -1] ### TODO: MODIFY FOR BATCH SIZE
            lm_logits = lm_logits[:, -1:]
        loss = None
        if labels is not None:
            lm_logits = lm_logits.to(torch.float32)
            labels_modality_2 = expert_batches[0]['labels']
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels_modality_2[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            lm_logits = lm_logits.to(final_output.dtype)
            loss = loss.to(final_output.dtype)

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


    @staticmethod
    def _reorder_cache(
            past: Tuple[Tuple[torch.Tensor, torch.Tensor], ...], beam_idx: torch.LongTensor
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], ...]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or
        [`~PreTrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.

        Output shares the same memory storage as `past`.
        """
        return tuple(
            (
                layer_past[0].index_select(0, beam_idx.to(layer_past[0].device)),
                layer_past[1].index_select(0, beam_idx.to(layer_past[1].device)),
            )
            for layer_past in past
        )


    @torch.no_grad()
    def generate_first_expert(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 1024,
        do_sample: bool = True,
        temperature: float = 0.2,
        top_p: float = 0.8,
        top_k: Optional[int] = None,
        repetition_penalty: float = 1.0,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        modality_masks: Optional[torch.Tensor] = None,
        position_encoding_indices: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """
        Generate text using the model with support for modality masks.

        Args:
            input_ids: Input token ids [batch_size, seq_length]
            attention_mask: Attention mask [batch_size, seq_length]
            max_new_tokens: Maximum number of tokens to generate
            do_sample: Whether to use sampling (True) or greedy decoding (False)
            temperature: Temperature for sampling
            top_p: Top-p (nucleus) sampling parameter
            top_k: Top-k sampling parameter
            repetition_penalty: Penalty for repeating tokens
            eos_token_id: End of sequence token id
            pad_token_id: Padding token id
            modality_masks: Masks for different modalities [n_modalities, batch_size, seq_length]
            **kwargs: Additional keyword arguments

        Returns:
            Generated token ids [batch_size, seq_length + generated_length]
        """
        # Set default values
        if eos_token_id is None:
            eos_token_id = self.config.eos_token_id
        if pad_token_id is None:
            pad_token_id = self.config.pad_token_id if hasattr(self.config, 'pad_token_id') else eos_token_id

        # Get device and dtype
        device = input_ids.device
        dtype = self.transformer.embedding.word_embeddings[0].weight.dtype

        # Initialize variables
        batch_size = input_ids.shape[0]
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=device)

        # Get initial position ids
        position_ids = self.get_position_ids(input_ids, device=device)

        # Initialize past key values
        past_key_values = None

        # Prepare initial model kwargs
        model_kwargs = {
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": bool(kwargs.get("use_cache", False)),
            "modality_masks": modality_masks,
            "position_encoding_indices": position_encoding_indices,
            "is_first_forward": True,
        }

        # Keep track of which sequences are finished
        if eos_token_id is not None:
            eos_token_id_tensor = torch.tensor(eos_token_id).to(device)

        # Generate tokens one by one
        for step in range(max_new_tokens):
            # Prepare inputs using prepare_inputs_for_generation
            model_inputs = self.prepare_inputs_for_generation(
                input_ids,
                **model_kwargs
            )

            # Forward pass
            outputs = self.forward_first_expert(
                **model_inputs,
                return_dict=True,
            )

            # Get logits for the last token
            next_token_logits = outputs.logits[:, -1, :]

            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for i in range(batch_size):
                    for token_id in set(input_ids[i].tolist()):
                        if token_id != pad_token_id:
                            next_token_logits[i, token_id] /= repetition_penalty

            # Apply temperature
            if do_sample and temperature > 0:
                next_token_logits = next_token_logits / temperature

            # Get next tokens
            if do_sample:
                # Apply top-k filtering
                if top_k is not None and top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = -float('Inf')

                # Apply top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # Shift the indices to the right to keep also the first token above the threshold
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0

                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = -float('Inf')

                # Sample from the distribution
                probs = F.softmax(next_token_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                # Greedy decoding
                next_tokens = torch.argmax(next_token_logits, dim=-1)

            # Determine modality for the next token based on token ID ranges
            # Audio tokens (152353-168735) use modality 1, others use modality 0 (text)
            is_audio_token = (next_tokens >= 152353) & (next_tokens <= 168735)

            # Create modality masks for the new token
            next_modality_ids_0 = ~is_audio_token  # Text modality
            next_modality_ids_1 = is_audio_token   # Audio modality
            next_modality_ids_2 = torch.zeros(batch_size, dtype=torch.bool, device=device)  # Other modalities

            # Stack modality masks and add sequence dimension
            next_modality_masks = torch.stack([
                next_modality_ids_0, next_modality_ids_1, next_modality_ids_2
            ], dim=0).unsqueeze(-1)  # shape: [num_modalities, batch_size, 1]

            # Update unfinished sequences
            if eos_token_id is not None:
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)
                # Check if any sequence has reached eos_token_id
                unfinished_sequences = unfinished_sequences * (next_tokens not in eos_token_id)

            # Concatenate with previous tokens
            input_ids = torch.cat([input_ids, next_tokens.unsqueeze(1)], dim=-1)

            # Update modality masks
            modality_masks = torch.cat([modality_masks, next_modality_masks], dim=-1)
            model_kwargs["modality_masks"] = modality_masks

            # Update model kwargs for next iteration
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=False,
            )

            # Check if all sequences are finished
            if unfinished_sequences.max() == 0:
                break

        # Return results - always return three values for consistency
        past_key_values = outputs.past_key_values if hasattr(outputs, 'past_key_values') else None
        return input_ids, modality_masks, past_key_values


    @torch.no_grad()
    def generate_first_expert_cache(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 1024,
        do_sample: bool = True,
        temperature: float = 0.2,
        top_p: float = 0.8,
        top_k: Optional[int] = None,
        repetition_penalty: float = 1.0,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        modality_masks: Optional[torch.Tensor] = None,
        position_encoding_indices: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """
        Generate text using the model with support for modality masks.

        Args:
            input_ids: Input token ids [batch_size, seq_length]
            attention_mask: Attention mask [batch_size, seq_length]
            max_new_tokens: Maximum number of tokens to generate
            do_sample: Whether to use sampling (True) or greedy decoding (False)
            temperature: Temperature for sampling
            top_p: Top-p (nucleus) sampling parameter
            top_k: Top-k sampling parameter
            repetition_penalty: Penalty for repeating tokens
            eos_token_id: End of sequence token id
            pad_token_id: Padding token id
            modality_masks: Masks for different modalities [n_modalities, batch_size, seq_length]
            position_encoding_indices: Position encoding indices [batch_size, seq_length]
            **kwargs: Additional keyword arguments

        Returns:
            Generated token ids [batch_size, seq_length + generated_length]
        """
        # Set default values
        if eos_token_id is None:
            eos_token_id = self.config.eos_token_id
        if pad_token_id is None:
            pad_token_id = self.config.pad_token_id if hasattr(self.config, 'pad_token_id') else eos_token_id

        # Get device and dtype
        device = input_ids.device
        dtype = self.transformer.embedding.word_embeddings[0].weight.dtype

        # Initialize variables
        batch_size = input_ids.shape[0]
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=device)

        # Get initial position ids
        position_ids = self.get_position_ids(input_ids, device=device)

        # Initialize past key values
        past_key_values = None

        # Prepare initial model kwargs
        model_kwargs = {
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            # "use_cache": True,
            "use_cache": False, # TODO: change to True
            "modality_masks": modality_masks,
        }

        model_inputs = self.prepare_inputs_for_generation(
            input_ids,
            **model_kwargs
        )

        # Forward pass
        outputs = self.forward_first_expert(
            **model_inputs,
            return_dict=True,
        )

        return outputs['past_key_values']


    @torch.no_grad()
    def generate_second_expert(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 1024,
        do_sample: bool = True,
        temperature: float = 0.2,
        top_p: float = 0.8,
        top_k: Optional[int] = None,
        repetition_penalty: float = 1.0,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        modality_masks: Optional[torch.Tensor] = None,
        position_encoding_indices: Optional[torch.Tensor] = None,
        past_key_values: Optional[torch.Tensor] = None,
        body_part: Optional[str] = None,
        **kwargs,
    ):
        """
        Generate text using the model with support for modality masks.

        Args:
            input_ids: Input token ids [batch_size, seq_length]
            attention_mask: Attention mask [batch_size, seq_length]
            max_new_tokens: Maximum number of tokens to generate
            do_sample: Whether to use sampling (True) or greedy decoding (False)
            temperature: Temperature for sampling
            top_p: Top-p (nucleus) sampling parameter
            top_k: Top-k sampling parameter
            repetition_penalty: Penalty for repeating tokens
            eos_token_id: End of sequence token id
            pad_token_id: Padding token id
            modality_masks: Masks for different modalities [n_modalities, batch_size, seq_length]
            position_encoding_indices: Position encoding indices [batch_size, seq_length]
            body_part: Body part to generate
            **kwargs: Additional keyword arguments

        Returns:
            Generated token ids [batch_size, seq_length + generated_length]
        """

        # Get device and dtype
        device = input_ids.device

        # Initialize variables
        batch_size = input_ids.shape[0]
        # unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=device)

        # Get initial position ids
        position_ids = self.get_position_ids(input_ids, device=device)

        # # Initialize past key values
        if past_key_values is not None:
            use_cache = True
        else:
            use_cache = False

        mask = modality_masks[2,0]

        idxs = torch.nonzero(mask, as_tuple=False).view(-1)[0] + 1 ## TODO: change to batch size
        modality_masks_current = modality_masks[:, :, :idxs]

        # Prepare initial model kwargs
        model_kwargs = {
            "attention_mask": attention_mask[:, :idxs],
            "position_ids": position_ids[:, :idxs],
            "past_key_values": past_key_values,
            "use_cache": use_cache,
            "modality_masks": modality_masks_current,
            "modality_masks_full": modality_masks[:, :, :idxs],
            "position_encoding_indices": position_encoding_indices[:, :idxs],
            "is_first_forward": False,
        }

        seq_len = input_ids.shape[1]

        # Generate tokens one by one
        for seq_index in range(idxs, seq_len):

            if modality_masks[0, 0, seq_index-1] == True or modality_masks[1, 0, seq_index-1] == True:  ## TODO: change to batch size
                model_kwargs["modality_masks"] = modality_masks[:, :, :seq_index+1]
                model_kwargs["modality_masks_full"] = modality_masks
                model_kwargs["position_encoding_indices"] = position_encoding_indices[:, :seq_index+1]
                model_kwargs["attention_mask"] = attention_mask[:, :seq_index+1]
                continue

            # Prepare inputs using prepare_inputs_for_generation
            model_inputs = self.prepare_inputs_for_generation(
                input_ids[:, :seq_index],
                **model_kwargs
            )

            # Forward pass
            outputs = self.forward_second_expert(
                    **model_inputs,
                    return_dict=True,
                )
            # Get logits for the last token
            next_token_logits = outputs.logits[:, -1, :]

            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for i in range(batch_size):
                    for token_id in set(input_ids[i].tolist()):
                        if token_id != pad_token_id:
                            next_token_logits[i, token_id] /= repetition_penalty

            # Apply temperature
            if do_sample and temperature > 0:
                next_token_logits = next_token_logits / temperature

            # Get next tokens
            if do_sample:
                # Apply top-k filtering
                if top_k is not None and top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = -float('Inf')

                # Apply top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # Shift the indices to the right to keep also the first token above the threshold
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0

                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = -float('Inf')

                # Sample from the distribution
                probs = F.softmax(next_token_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                # Greedy decoding
                next_tokens = torch.argmax(next_token_logits, dim=-1)


            #     pass

            # Concatenate with previous tokens
            if input_ids[0, seq_index] == 0:
                input_ids[:, seq_index] = next_tokens.unsqueeze(1)
            else:
                pass


            model_kwargs["modality_masks"] = modality_masks[:, :, :seq_index+1]
            model_kwargs["modality_masks_full"] = modality_masks
            model_kwargs["position_encoding_indices"] = position_encoding_indices[:, :seq_index+1]
            # model_kwargs["attention_mask"] = attention_mask[:, :seq_index+1]  ## _update_model_kwargs_for_generation will handle this
            # Update model kwargs for next iteration
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=False,
            )

            # next_tokens = next_tokens * unfinished_sequences
            # unfinished_sequences = unfinished_sequences * (next_tokens != 1540)

            # if next_tokens == 1540: ## TODO: change to batch size
            #     break

            # # Check if all sequences are finished
            #     break


        return input_ids, modality_masks


    @torch.no_grad()
    def generate_second_expert_t2m(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 1024,
        do_sample: bool = True,
        temperature: float = 0.2,
        top_p: float = 0.8,
        top_k: Optional[int] = None,
        repetition_penalty: float = 1.0,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        modality_masks: Optional[torch.Tensor] = None,
        position_encoding_indices: Optional[torch.Tensor] = None,
        past_key_values: Optional[torch.Tensor] = None,
        body_part: Optional[str] = None,
        **kwargs,
    ):
        """
        Generate text using the model with support for modality masks.

        Args:
            input_ids: Input token ids [batch_size, seq_length]
            attention_mask: Attention mask [batch_size, seq_length]
            max_new_tokens: Maximum number of tokens to generate
            do_sample: Whether to use sampling (True) or greedy decoding (False)
            temperature: Temperature for sampling
            top_p: Top-p (nucleus) sampling parameter
            top_k: Top-k sampling parameter
            repetition_penalty: Penalty for repeating tokens
            eos_token_id: End of sequence token id
            pad_token_id: Padding token id
            modality_masks: Masks for different modalities [n_modalities, batch_size, seq_length]
            position_encoding_indices: Position encoding indices [batch_size, seq_length]
            body_part: Body part to generate
            **kwargs: Additional keyword arguments

        Returns:
            Generated token ids [batch_size, seq_length + generated_length]
        """

        # Get device and dtype
        device = input_ids.device

        # Initialize variables
        batch_size = input_ids.shape[0]
        # unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=device)

        # Get initial position ids
        position_ids = self.get_position_ids(input_ids, device=device)

        # # Initialize past key values
        if past_key_values is not None:
            use_cache = True
        else:
            use_cache = False

        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=device)

        # Prepare initial model kwargs
        model_kwargs = {
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
            "modality_masks": modality_masks,
            "modality_masks_full": modality_masks,
            "position_encoding_indices": position_encoding_indices,
            "is_first_forward": False,
        }

        seq_len = input_ids.shape[1]


        # Generate tokens one by one
        for _ in range(max_new_tokens):

            # Prepare inputs using prepare_inputs_for_generation
            model_inputs = self.prepare_inputs_for_generation(
                input_ids,
                **model_kwargs
            )

            # Forward pass
            outputs = self.forward_second_expert(
                    **model_inputs,
                    return_dict=True,
                )
            # Get logits for the last token
            next_token_logits = outputs.logits[:, -1, :]

            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for i in range(batch_size):
                    for token_id in set(input_ids[i].tolist()):
                        if token_id != pad_token_id:
                            next_token_logits[i, token_id] /= repetition_penalty

            # Apply temperature
            if do_sample and temperature > 0:
                next_token_logits = next_token_logits / temperature

            # Get next tokens
            if do_sample:
                # Apply top-k filtering
                if top_k is not None and top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = -float('Inf')

                # Apply top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # Shift the indices to the right to keep also the first token above the threshold
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0

                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = -float('Inf')

                # Sample from the distribution
                probs = F.softmax(next_token_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                # Greedy decoding
                next_tokens = torch.argmax(next_token_logits, dim=-1)


            input_ids = torch.cat([input_ids, next_tokens.unsqueeze(1)], dim=-1)


            ## TODO: change to batch size
            position_encoding_indices = torch.arange(input_ids.shape[-1], dtype=torch.long, device=device).unsqueeze(0).unsqueeze(0).repeat(batch_size, 1,1)


            next_modality_masks = torch.zeros(modality_masks.shape[0], modality_masks.shape[1], 1, dtype=modality_masks.dtype, device=modality_masks.device)
            next_modality_masks[2, :, :] = True
            modality_masks = torch.cat([modality_masks, next_modality_masks], dim=-1)

            model_kwargs["modality_masks"] = modality_masks
            model_kwargs["modality_masks_full"] = modality_masks
            model_kwargs["position_encoding_indices"] = position_encoding_indices
            # Update model kwargs for next iteration
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=False,
            )

            # generation_time = time.time() - generation_start
            # print(f"Generation completed, time: {generation_time:.2f} seconds")

            next_tokens = next_tokens * unfinished_sequences
            unfinished_sequences = unfinished_sequences * (next_tokens != 513)

            # if next_tokens == 1540: ## TODO: change to batch size
            #     break

            # Check if all sequences are finished
            if unfinished_sequences.max() == 0:
                break

        return input_ids, modality_masks


    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 1024,
        do_sample: bool = True,
        temperature: float = 0.2,
        top_p: float = 0.8,
        top_k: Optional[int] = None,
        repetition_penalty: float = 1.0,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        modality_masks: Optional[torch.Tensor] = None,
        position_encoding_indices: Optional[torch.Tensor] = None,
        body_part: Optional[str] = None,
        **kwargs,
    ):
        """
        Generate text using the model with support for modality masks.
        Generates using the first modality with automatic modality switching.

        Args:
            input_ids: Input token ids [batch_size, seq_length]
            attention_mask: Attention mask [batch_size, seq_length]
            max_new_tokens: Maximum number of tokens to generate
            do_sample: Whether to use sampling (True) or greedy decoding (False)
            temperature: Temperature for sampling
            top_p: Top-p (nucleus) sampling parameter
            top_k: Top-k sampling parameter
            repetition_penalty: Penalty for repeating tokens
            eos_token_id: End of sequence token id
            pad_token_id: Padding token id
            modality_masks: Masks for different modalities [n_modalities, batch_size, seq_length]
            body_part: Body part to generate
            **kwargs: Additional keyword arguments

        Returns:
            Generated token ids [batch_size, seq_length + generated_length]
        """
        # Set default values
        if eos_token_id is None:
            eos_token_id = self.config.eos_token_id
        if pad_token_id is None:
            pad_token_id = self.config.pad_token_id if hasattr(self.config, 'pad_token_id') else eos_token_id

        # Get device and dtype
        device = input_ids.device
        dtype = self.transformer.embedding.word_embeddings[0].weight.dtype

        # Initialize variables
        batch_size = input_ids.shape[0]

        # Get initial position ids
        position_ids = self.get_position_ids(input_ids, device=device)

        # Initialize modality masks if not provided
        if modality_masks is None:
            # Default to modality 0 (text) for all tokens
            modality_masks = torch.zeros(3, batch_size, input_ids.shape[1], dtype=torch.bool, device=device)
            modality_masks[0] = True  # Set modality 0 (text) as active

        # Generate using first modality
        print("Generating with first modality...")
        first_modality_input_ids, first_modality_masks, first_modality_cache = self.generate_first_expert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            modality_masks=modality_masks,
            position_encoding_indices=position_encoding_indices,
            **kwargs
        )

        if body_part == "face":
            first_modality_input_ids_with_body, first_modality_masks_with_body = self.add_face_tokens_to_output(first_modality_input_ids, first_modality_masks)
            position_encoding_indices = calculate_position_encoding_indices(first_modality_masks_with_body, modality_fps={1: 12.5, 2: 25.0})
        elif body_part == "body":
            first_modality_input_ids_with_body, first_modality_masks_with_body = self.add_body_tokens_to_output(first_modality_input_ids, first_modality_masks)
            position_encoding_indices = calculate_position_encoding_indices(first_modality_masks_with_body, modality_fps={1: 12.5, 2: 18.75})
        elif body_part == "face_body":
            first_modality_input_ids_with_body, first_modality_masks_with_body = self.add_face_body_tokens_to_output(first_modality_input_ids, first_modality_masks)
        else:
            raise ValueError(f"Invalid body part: {body_part}")

        # position_encoding_indices = calculate_position_encoding_indices(first_modality_masks_with_body)
        position_encoding_indices = torch.tensor(position_encoding_indices, device=device).unsqueeze(0)
        attention_mask = torch.ones(first_modality_input_ids_with_body.shape[0], first_modality_input_ids_with_body.shape[1], dtype=torch.int64, device=first_modality_input_ids_with_body.device)


        # Step 2: Generate using second modality with cached KV from first modality
        print("Generating with second modality using cached KV...")

        second_modality_input_ids, second_modality_masks = self.generate_second_expert(
            input_ids=first_modality_input_ids_with_body,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            modality_masks=first_modality_masks_with_body,
            position_encoding_indices=position_encoding_indices,
            past_key_values=first_modality_cache,
            body_part=body_part,
            **kwargs
        )

        # Return the final results from second modality generation
        return second_modality_input_ids, second_modality_masks


class ChatGLMForSequenceClassification(ChatGLMPreTrainedModel):
    def __init__(self, config: ChatGLMConfig, empty_init=True, device=None):
        super().__init__(config)

        self.num_labels = config.num_labels
        self.transformer = ChatGLMModel(config, empty_init=empty_init, device=device)

        self.classifier_head = nn.Linear(config.hidden_size, config.num_labels, bias=True, dtype=config.torch_dtype)
        if config.classifier_dropout is not None:
            self.dropout = nn.Dropout(config.classifier_dropout)
        else:
            self.dropout = None
        self.config = config

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            full_attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
            inputs_embeds: Optional[torch.LongTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor, ...], SequenceClassifierOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            full_attention_mask=full_attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = transformer_outputs[0]
        pooled_hidden_states = hidden_states[:, -1]
        if self.dropout is not None:
            pooled_hidden_states = self.dropout(pooled_hidden_states)
        logits = self.classifier_head(pooled_hidden_states)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze().float(), labels.squeeze())
                else:
                    loss = loss_fct(logits.float(), labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels).float(), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits.float(), labels.view(-1, self.num_labels))

        if not return_dict:
            output = (logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


__all__ = [
    "ChatGLMPreTrainedModel",
    "ChatGLMModel",
    "ChatGLMForConditionalGenerationMotExpertNum2",
    "ChatGLMForSequenceClassification",
]
