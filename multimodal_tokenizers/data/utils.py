import torch
import rich
import pickle
import numpy as np


def lengths_to_mask(lengths):
    max_len = max(lengths)
    mask = torch.arange(max_len, device=lengths.device).expand(
        len(lengths), max_len) < lengths.unsqueeze(1)
    return mask

# padding to max length in one batch
def collate_tensors(batch):
    if isinstance(batch[0], np.ndarray):
        batch = [torch.tensor(b).float() for b in batch]

    dims = batch[0].dim()
    max_size = [max([b.size(i) for b in batch]) for i in range(dims)]
    size = (len(batch), ) + tuple(max_size)
    canvas = batch[0].new_zeros(size=size)
    for i, b in enumerate(batch):
        sub_tensor = canvas[i]
        for d in range(dims):
            sub_tensor = sub_tensor.narrow(d, 0, b.size(d))
        sub_tensor.add_(b)
    return canvas

def huggingface_dataset_collate(batch):
    """
    Simplified collate function for HuggingFace datasets that contain already preprocessed text.
    The text is already in the right format, length, and structure for training.
    
    Args:
        batch: A list of samples from the HuggingFace dataset
        
    Returns:
        A dict with 'text' containing the text data directly
    """
    # Extract text data from the batch
    formatted_texts = []
    for item in batch:
        if 'text' in item:
            # Text is already preprocessed, use directly
            formatted_texts.append(item['text'])
    
    return {
        'text': formatted_texts
    }

def _infer_collate_key(sample):
    if "text" in sample or "speaker_a_audio" in sample:
        return "hf_text"
    split_name = sample.get("split_name", "")
    if split_name == "test":
        return "test"
    token_fields = ("face_token", "hand_token", "lower_token", "upper_token", "audio_token")
    if any(field in sample for field in token_fields):
        return "lm_tokens"
    if split_name == "token":
        return "token"
    if split_name == "vae":
        return "vae_global"
    if "face_p2" in sample:
        return "vq_face_pairs"
    if "face_p1" in sample:
        return "vq_face_single"
    has_upper = "upper" in sample
    has_lower = "lower" in sample or "lower_54" in sample
    has_hand = "hand" in sample
    has_face = "face" in sample or "face_with_head" in sample
    if has_upper and not (has_lower or has_hand or has_face):
        return "vq_upper"
    if has_lower and not (has_upper or has_hand or has_face):
        return "vq_lower"
    if has_face and not (has_upper or has_lower or has_hand):
        return "vq_face"
    if has_upper or has_lower or has_hand or has_face:
        return "vq_mixed"
    return "default_face"

def _get_collate_key(sample):
    return sample.get("collate_key") or _infer_collate_key(sample)

def _require_keys(batch, keys, schema):
    missing = [key for key in keys if any(key not in b for b in batch)]
    if missing:
        sample_keys = sorted(batch[0].keys())
        raise KeyError(f"[{schema}] Missing keys in batch: {missing}. Sample keys: {sample_keys}")

def _collate_by_spec(batch, schema, tensor_keys, list_keys, required=None):
    required = required or []
    _require_keys(batch, required, schema)
    tensor_keys = [k for k in tensor_keys if all(k in b for b in batch)]
    list_keys = [k for k in list_keys if all(k in b for b in batch)]
    adapted_batch = {}
    for key in tensor_keys:
        adapted_batch[key] = collate_tensors([b[key].float() for b in batch])
    for key in list_keys:
        adapted_batch[key] = [b[key] for b in batch]
    return adapted_batch

def conversation_collate(batch):
    """
    Unified collate function that handles different types of data.
    Defaults to face data collation if no specific format is detected.
    """
    notnone_batches = [b for b in batch if b is not None]
    if not notnone_batches:
        return {}

    schema = _get_collate_key(notnone_batches[0])
    schema_set = {_get_collate_key(b) for b in notnone_batches}
    if len(schema_set) > 1:
        raise ValueError(f"Mixed collate_key values in batch: {sorted(schema_set)}")

    if schema == "hf_text":
        return huggingface_dataset_collate(notnone_batches)

    if schema == "vq_face_pairs":
        return _collate_by_spec(
            notnone_batches,
            schema,
            tensor_keys=["face_p1", "face_p2"],
            list_keys=["p1_name", "p2_name", "motion_len_1", "motion_len_2", "id_name", "dataset_name"],
            required=["face_p1", "face_p2"],
        )

    if schema == "vq_face_single":
        return _collate_by_spec(
            notnone_batches,
            schema,
            tensor_keys=["face_p1"],
            list_keys=["motion_len_1", "id_name", "dataset_name"],
            required=["face_p1"],
        )

    if schema == "vq_face":
        return _collate_by_spec(
            notnone_batches,
            schema,
            tensor_keys=["face", "face_with_head", "shape", "pose"],
            list_keys=["motion_len", "id_name", "dataset_name"],
            required=["face"],
        )

    if schema == "vq_upper":
        return _collate_by_spec(
            notnone_batches,
            schema,
            tensor_keys=["upper", "shape"],
            list_keys=["motion_len", "id_name", "dataset_name"],
            required=["upper"],
        )

    if schema == "vq_lower":
        lower_key = "lower" if "lower" in notnone_batches[0] else "lower_54"
        return _collate_by_spec(
            notnone_batches,
            schema,
            tensor_keys=[lower_key, "shape", "trans"],
            list_keys=["motion_len", "id_name", "dataset_name"],
            required=[lower_key],
        )

    if schema == "vae_global":
        return _collate_by_spec(
            notnone_batches,
            schema,
            tensor_keys=["lower", "shape", "trans"],
            list_keys=["motion_len", "id_name", "dataset_name"],
            required=["lower"],
        )

    if schema == "vq_mixed":
        modality_keys = ["pose", "face", "face_with_head", "hand", "upper", "lower", "shape", "trans"]
        present_keys = [k for k in modality_keys if k in notnone_batches[0]]
        if not present_keys:
            raise KeyError(f"[{schema}] No modality keys found in batch.")
        _require_keys(notnone_batches, present_keys, schema)
        return _collate_by_spec(
            notnone_batches,
            schema,
            tensor_keys=present_keys,
            list_keys=["motion_len", "id_name", "dataset_name"],
        )

    if schema == "token":
        return _collate_by_spec(
            notnone_batches,
            schema,
            tensor_keys=["pose", "beta", "trans", "facial"],
            list_keys=["seq_name", "dataset_name"],
        )

    if schema == "test":
        tensor_fields = [
            "face", "hand", "lower", "upper", "tar_pose", "tar_beta",
            "tar_trans", "tar_exps", "audio_token", "m_tokens_len",
        ]
        list_fields = ["raw_audio", "a_tokens_len"]
        return _collate_by_spec(
            notnone_batches,
            schema,
            tensor_keys=tensor_fields,
            list_keys=list_fields,
        )

    if schema == "lm_tokens":
        return _collate_by_spec(
            notnone_batches,
            schema,
            tensor_keys=["face_token", "hand_token", "lower_token", "upper_token", "audio_token"],
            list_keys=["tasks", "m_tokens_len", "a_tokens_len", "text"],
        )

    return _collate_by_spec(
        notnone_batches,
        "default_face",
        tensor_keys=["face", "shape", "pose"],
        list_keys=["motion_len", "id_name", "dataset_name"],
        required=["face"],
    )

# Legacy collate functions delegate to the main conversation_collate function
def face_collate_fn(batch):
    """Legacy face collate function that delegates to conversation_collate."""
    return conversation_collate(batch)
    
def lom_collate(batch):
    """Legacy lom collate function that delegates to conversation_collate."""
    return conversation_collate(batch)

def load_pkl(path, description=None, progressBar=False):
    if progressBar:
        with rich.progress.open(path, 'rb', description=description) as file:
            data = pickle.load(file)
    else:
        with open(path, 'rb') as file:
            data = pickle.load(file)
    return data
