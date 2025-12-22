import torch
import rich
import pickle
import numpy as np
import random


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

def format_interleaved_conversation(audio_tokens_a=None, audio_tokens_b=None, face_tokens_a=None, face_tokens_b=None, interleaved_text=None, max_seq_length=1024, num_rounds=11):
    """
    Format conversation data with interleaved tokens in the following format:
    <|user|>\n <|audio_XXX|>...<|face_XXX|>...
    <|assistant|>\n <|audio_XXX|>...<|face_XXX|>...
    
    Can create conversations from two types of inputs:
    1. From raw audio and face tokens (by providing audio_tokens_a/b and face_tokens_a/b)
    2. From already interleaved text (by providing interleaved_text)
    
    Args:
        audio_tokens_a: Audio tokens for speaker A (optional)
        audio_tokens_b: Audio tokens for speaker B (optional)
        face_tokens_a: Face tokens for speaker A (optional)
        face_tokens_b: Face tokens for speaker B (optional)
        interleaved_text: Already interleaved text in format <|speaker_A|>...<|speaker_B|>... (optional)
        max_seq_length: Maximum sequence length for the LLM (default: 1024)
        num_rounds: Number of dialogue rounds in each sequence (default: 11)
        
    Returns:
        If interleaved_text is provided, returns a list where each element is a conversation with system prompt, 
        formatted and length-limited dialogue sequence.
        If token parameters are provided, returns a single string containing one formatted dialogue turn.
    """
    # Define system template
    system_template = "<|system|>\nThe user will provide an audio conversation along with face motion data. Do it step by step.First, think about the audio conversation and respond in a interleaved manner, with 13 face token followed by 26 audio tokens.\n"
    system_token_count = 49  # Confirmed token count for the system template
    
    # Processing method 1: Process already interleaved text
    if interleaved_text is not None:
        # If input is empty, return empty list
        if not interleaved_text or interleaved_text.strip() == "":
            return []
        
        # Convert <|speaker_X|> format to <|user|>\n and <|assistant|>\n format
        if "<|speaker_A|>" in interleaved_text or "<|speaker_B|>" in interleaved_text:
            # Randomly decide whether speaker_A is user or assistant
            if random.random() < 0.5:
                # speaker_A is user, speaker_B is assistant
                converted_text = interleaved_text.replace("<|speaker_A|>", "<|user|>\n")
                converted_text = converted_text.replace("<|speaker_B|>", "<|assistant|>\n")
            else:
                # speaker_B is user, speaker_A is assistant
                converted_text = interleaved_text.replace("<|speaker_B|>", "<|user|>\n")
                converted_text = converted_text.replace("<|speaker_A|>", "<|assistant|>\n")
        else:
            # Already in chat format
            converted_text = interleaved_text
        
        # Fix formatting issues: ensure proper separation between <|user|> and <|assistant|>
        # Fix cases where <|user|><|assistant|> are adjacent
        converted_text = converted_text.replace("<|user|><|assistant|>", "<|user|>\n<|assistant|>")
        
        # Split text by user turns - using regex for precise matching
        import re
        parts = re.split(r'(?=<\|user\|>)', converted_text)
        
        # Skip empty first part (if exists)
        if parts and not parts[0].strip():
            parts = parts[1:]
        
        # If no user turns, return empty list
        if not parts:
            return []
        
        # Reconstruct complete user-assistant dialogue turns
        complete_turns = []
        for part in parts:
            # Check if this part contains a complete user-assistant interaction
            if "<|user|>" in part and "<|assistant|>" in part:
                # Ensure part starts with <|user|>
                if not part.strip().startswith("<|user|>"):
                    continue
                    
                # Ensure there's content (audio and face tokens) between <|user|> and <|assistant|>
                user_content = re.search(r'<\|user\|>.*?(?=<\|assistant\|>)', part, re.DOTALL)
                if not user_content or not re.search(r'<\|audio_\d+\|>', user_content.group(0)):
                    continue
                    
                # Ensure there's content (audio and face tokens) after <|assistant|>
                assistant_content = re.search(r'<\|assistant\|>.*?($|(?=<\|user\|>))', part, re.DOTALL)
                if not assistant_content or not re.search(r'<\|audio_\d+\|>', assistant_content.group(0)):
                    continue
                
                complete_turns.append(part)
        
        # If not enough dialogue rounds, return empty list
        if len(complete_turns) < num_rounds:
            return []
        
        # Create fixed-length sequences, each containing exactly num_rounds of dialogue
        conversations = []
        for i in range(0, len(complete_turns) - num_rounds + 1):
            # Take exactly num_rounds consecutive turns
            selected_turns = complete_turns[i:i+num_rounds]
            
            # Create conversation with system prompt
            conversation = system_template + "".join(selected_turns)
            
            # Ensure total length doesn't exceed max_seq_length
            if count_tokens_simple(conversation) <= max_seq_length:
                conversations.append(conversation)
        
        return conversations
    
    # Processing method 2: Create one dialogue turn from raw tokens (original functionality preserved)
    else:
        # Decide randomly whether A is user or assistant
        if random.random() < 0.5:
            # A is user, B is assistant
            user_audio = audio_tokens_a
            user_face = face_tokens_a
            assistant_audio = audio_tokens_b
            assistant_face = face_tokens_b
        else:
            # B is user, A is assistant
            user_audio = audio_tokens_b
            user_face = face_tokens_b
            assistant_audio = audio_tokens_a
            assistant_face = face_tokens_a
        
        # Ensure tokens are in list format
        if isinstance(user_audio, np.ndarray):
            user_audio = user_audio.flatten().tolist()
        if isinstance(user_face, np.ndarray):
            user_face = user_face.flatten().tolist()
        if isinstance(assistant_audio, np.ndarray):
            assistant_audio = assistant_audio.flatten().tolist()
        if isinstance(assistant_face, np.ndarray):
            assistant_face = assistant_face.flatten().tolist()
        
        # Calculate token counts more precisely for our format
        # Each <|audio_X|> or <|face_X|> counts as 1 token
        user_token_count = 2 + 1 + 1 + len(user_audio) + len(user_face)  # <|user|> + \n + space + audio tokens + face tokens
        assistant_token_count = 3 + 1 + 1 + len(assistant_audio) + len(assistant_face)  # \n + <|assistant|> + \n + space + audio tokens + face tokens
        
        total_token_count = user_token_count + assistant_token_count
        
        # If the sequence is too long, truncate tokens while preserving format
        if total_token_count > max_seq_length:
            # Calculate available tokens for content (minus the markers)
            available_content_tokens = max_seq_length - 9  # Markers: <|user|>, <|assistant|>, \n, spaces
            
            # Distribute tokens evenly between user and assistant
            available_per_side = available_content_tokens // 2
            
            # Calculate proportion to keep from each modality
            user_content_tokens = len(user_audio) + len(user_face)
            assistant_content_tokens = len(assistant_audio) + len(assistant_face)
            
            if user_content_tokens > available_per_side:
                # Need to truncate user tokens
                user_ratio = available_per_side / user_content_tokens
                audio_ratio = len(user_audio) / (len(user_audio) + len(user_face))
                face_ratio = len(user_face) / (len(user_audio) + len(user_face))
                
                # Calculate tokens to keep for each modality
                user_audio_keep = int(len(user_audio) * user_ratio * audio_ratio)
                user_face_keep = int(len(user_face) * user_ratio * face_ratio)
                
                # Ensure we keep at least some tokens
                user_audio_keep = max(1, user_audio_keep)
                user_face_keep = max(1, user_face_keep)
                
                # Truncate
                user_audio = user_audio[:user_audio_keep]
                user_face = user_face[:user_face_keep]
            
            if assistant_content_tokens > available_per_side:
                # Need to truncate assistant tokens
                assistant_ratio = available_per_side / assistant_content_tokens
                audio_ratio = len(assistant_audio) / (len(assistant_audio) + len(assistant_face))
                face_ratio = len(assistant_face) / (len(assistant_audio) + len(assistant_face))
                
                # Calculate tokens to keep for each modality
                assistant_audio_keep = int(len(assistant_audio) * assistant_ratio * audio_ratio)
                assistant_face_keep = int(len(assistant_face) * assistant_ratio * face_ratio)
                
                # Ensure we keep at least some tokens
                assistant_audio_keep = max(1, assistant_audio_keep)
                assistant_face_keep = max(1, assistant_face_keep)
                
                # Truncate
                assistant_audio = assistant_audio[:assistant_audio_keep]
                assistant_face = assistant_face[:assistant_face_keep]
        
        # Format tokens as strings
        user_audio_str = ''.join([f"<|audio_{token}|>" for token in user_audio])
        user_face_str = ''.join([f"<|face_{token}|>" for token in user_face])
        assistant_audio_str = ''.join([f"<|audio_{token}|>" for token in assistant_audio])
        assistant_face_str = ''.join([f"<|face_{token}|>" for token in assistant_face])
        
        # Create the full formatted text
        formatted_text = f"<|user|>\n {user_audio_str}{user_face_str}\n<|assistant|>\n {assistant_audio_str}{assistant_face_str}"
        
        return formatted_text

    
def count_tokens_simple(text):
    """
    Simple estimation of token count in text (for GLM-4-Voice tokenizer)
    
    Args:
        text: Input text
        
    Returns:
        Estimated token count
    """
    # Count special tokens
    special_tokens = text.count("<|system|>") + text.count("<|user|>") + text.count("<|assistant|>")
    special_tokens += text.count("<|speaker_A|>") + text.count("<|speaker_B|>")
    
    # Count audio and face tokens
    import re
    audio_tokens = len(re.findall(r"<\|audio_\d+\|>", text))
    face_tokens = len(re.findall(r"<\|face_\d+\|>", text))
    
    # Count newlines
    newlines = text.count("\n")
    
    # Estimate regular text tokens
    words = len([w for w in text.split() if not (
        w.startswith("<|") and w.endswith("|>") or 
        re.match(r"<\|audio_\d+\|>", w) or 
        re.match(r"<\|face_\d+\|>", w)
    )])
    
    # Total token count
    total = special_tokens + audio_tokens + face_tokens + newlines + words
    
    return total

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

def conversation_collate(batch):
    """
    Unified collate function that handles different types of data.
    Defaults to face data collation if no specific format is detected.
    """
    notnone_batches = [b for b in batch if b is not None]
    
    if not notnone_batches:
        return {}
    
    # Check if this is a Hugging Face Dataset format (from direct loading)
    if 'text' in notnone_batches[0]:
        # Text is already preprocessed, use simplified collate
        return huggingface_dataset_collate(notnone_batches)
    elif 'speaker_a_audio' in notnone_batches[0]:
        # For raw speaker data format, use the regular HF collate
        return huggingface_dataset_collate(notnone_batches)
        
    # Get split name from the first item
    split_name = notnone_batches[0].get("split_name", "")
    select_part = batch[0]["select_part"]
    # ===== CONVERSATION DATASET (VQ mode) =====
    if split_name == 'vq' and "face_p2" in notnone_batches[0]:
        adapted_batch = {
            "face_p1": collate_tensors([b["face_p1"].float() for b in notnone_batches]),
            "face_p2": collate_tensors([b["face_p2"].float() for b in notnone_batches]),
            "p1_name": [b["p1_name"] for b in notnone_batches],
            "p2_name": [b["p2_name"] for b in notnone_batches],
            "motion_len_1": [b["motion_len_1"] for b in notnone_batches],
            "motion_len_2": [b["motion_len_2"] for b in notnone_batches],
            "id_name": [b["id_name"] for b in notnone_batches],
            "dataset_name": [b["dataset_name"] for b in notnone_batches],
        }
        return adapted_batch
    elif split_name == 'vq' and "face_p1" in notnone_batches[0]:
        adapted_batch = {
            "face_p1": collate_tensors([b["face_p1"].float() for b in notnone_batches]),
            "id_name": [b["id_name"] for b in notnone_batches],
            "motion_len_1": [b["motion_len_1"] for b in notnone_batches],
            "dataset_name": [b["dataset_name"] for b in notnone_batches],
        }
        return adapted_batch
    
    elif split_name == 'vq' and select_part == 'compositional' and (notnone_batches[0]['dataset_name'] == 'tfhp' or notnone_batches[0]['dataset_name'] == 'YouTube_Talking') and "face" in notnone_batches[0]:
        adapted_batch = {
            "face": collate_tensors([b["face"].float() for b in notnone_batches]),
            "face_with_head": collate_tensors([b["face_with_head"].float() for b in notnone_batches]),
            "motion_len": [b["motion_len"] for b in notnone_batches],
            "id_name": [b["id_name"] for b in notnone_batches],
            "dataset_name": [b["dataset_name"] for b in notnone_batches],
        }
        return adapted_batch

    elif split_name == 'vq' and select_part == 'compositional' and (notnone_batches[0]['dataset_name'] == 'tfhp' or notnone_batches[0]['dataset_name'] == 'YouTube_Talking') and "upper" in notnone_batches[0]:
        adapted_batch = {
            "upper": collate_tensors([b["upper"].float() for b in notnone_batches]),
            "motion_len": [b["motion_len"] for b in notnone_batches],
            "id_name": [b["id_name"] for b in notnone_batches],
            "dataset_name": [b["dataset_name"] for b in notnone_batches],
        }
        return adapted_batch

    elif split_name == 'vq' and select_part == 'compositional' and notnone_batches[0]['dataset_name'] != 'tfhp' and notnone_batches[0]['dataset_name'] != 'YouTube_Talking':
        adapted_batch = {
            "upper": collate_tensors([b["upper"].float() for b in notnone_batches]),
            "shape": collate_tensors([b["shape"].float() for b in notnone_batches]),
            "trans": collate_tensors([b["trans"].float() for b in notnone_batches]),
            "hand": collate_tensors([b["hand"].float() for b in notnone_batches]),
            "lower": collate_tensors([b["lower"].float() for b in notnone_batches]),
            "face": collate_tensors([b["face"].float() for b in notnone_batches]),
            "face_with_head": collate_tensors([b["face_with_head"].float() for b in notnone_batches]),
            "motion_len": [b["motion_len"] for b in notnone_batches],
            "id_name": [b["id_name"] for b in notnone_batches],
            "dataset_name": [b["dataset_name"] for b in notnone_batches],
        }
        return adapted_batch

    elif split_name == 'vq' and "upper" in notnone_batches[0]:
        adapted_batch = {
            "upper": collate_tensors([b["upper"].float() for b in notnone_batches]),
            "shape": collate_tensors([b["shape"].float() for b in notnone_batches]),
            "motion_len": [b["motion_len"] for b in notnone_batches],
            "id_name": [b["id_name"] for b in notnone_batches],
            "dataset_name": [b["dataset_name"] for b in notnone_batches],
        }
        return adapted_batch

    elif split_name == 'vq' and ("lower" in notnone_batches[0] or "lower_54" in notnone_batches[0]):
        adapted_batch = {
            "lower": collate_tensors([b["lower"].float() for b in notnone_batches]),
            "shape": collate_tensors([b["shape"].float() for b in notnone_batches]),
            "trans": collate_tensors([b["trans"].float() for b in notnone_batches]),
            "motion_len": [b["motion_len"] for b in notnone_batches],
            "id_name": [b["id_name"] for b in notnone_batches],
            "dataset_name": [b["dataset_name"] for b in notnone_batches],
        }
        return adapted_batch

    elif split_name == 'vae' and select_part == "global":
        adapted_batch = {
            "lower": collate_tensors([b["lower"].float() for b in notnone_batches]),
            "shape": collate_tensors([b["shape"].float() for b in notnone_batches]),
            "trans": collate_tensors([b["trans"].float() for b in notnone_batches]),
            "motion_len": [b["motion_len"] for b in notnone_batches],
            "id_name": [b["id_name"] for b in notnone_batches],
            "dataset_name": [b["dataset_name"] for b in notnone_batches],
        }
        return adapted_batch
    
    # ===== MIXED DATASET (VQ mode with multiple body parts) =====
    elif split_name == 'vq' and ("hand" in notnone_batches[0] or "upper" in notnone_batches[0] or "lower" in notnone_batches[0]):
        adapted_batch = {
            "pose": collate_tensors([b["pose"].float() for b in notnone_batches]) if "pose" in notnone_batches[0] else None,
            "face": collate_tensors([b["face"].float() for b in notnone_batches]) if "face" in notnone_batches[0] else None,
            "hand": collate_tensors([b["hand"].float() for b in notnone_batches]) if "hand" in notnone_batches[0] else None,
            "upper": collate_tensors([b["upper"].float() for b in notnone_batches]) if "upper" in notnone_batches[0] else None,
            "lower": collate_tensors([b["lower"].float() for b in notnone_batches]) if "lower" in notnone_batches[0] else None,
            "shape": collate_tensors([b["shape"].float() for b in notnone_batches]) if "shape" in notnone_batches[0] else None,
            "trans": collate_tensors([b["trans"].float() for b in notnone_batches]) if "trans" in notnone_batches[0] else None,
            "motion_len": [b.get("motion_len", 0) for b in notnone_batches],
            "id_name": [b.get("id_name", "") for b in notnone_batches],
            "dataset_name": [b.get("dataset_name", "") for b in notnone_batches],
        }
        # Remove None entries
        adapted_batch = {k: v for k, v in adapted_batch.items() if v is not None}
        return adapted_batch
    elif select_part == 'compositional':
        adapted_batch = {
            "pose": collate_tensors([b["pose"].float() for b in notnone_batches]),
            "face": collate_tensors([b["face"].float() for b in notnone_batches]),
            "hand": collate_tensors([b["hand"].float() for b in notnone_batches]),
            "lower": collate_tensors([b["lower"].float() for b in notnone_batches]),
            "upper": collate_tensors([b["upper"].float() for b in notnone_batches]),
            "shape": collate_tensors([b["shape"].float() for b in notnone_batches]),
            "trans": collate_tensors([b["trans"].float() for b in notnone_batches]),
            "motion_len": [b["motion_len"] for b in notnone_batches],
            "id_name": [b["id_name"] for b in notnone_batches],
            "dataset_name": [b["dataset_name"] for b in notnone_batches],
        }
        # Remove None entries
        adapted_batch = {k: v for k, v in adapted_batch.items() if v is not None}
        return adapted_batch
    # ===== TEST mode =====
    elif split_name == 'test':
        adapted_batch = {}
        # Add all tensor fields
        tensor_fields = ["face", "hand", "lower", "upper", "tar_pose", "tar_beta", 
                        "tar_trans", "tar_exps", "audio_token", "m_tokens_len"]
        for field in tensor_fields:
            if field in notnone_batches[0]:
                adapted_batch[field] = collate_tensors([b[field].float() for b in notnone_batches])
        
        # Add list fields
        list_fields = ["raw_audio", "a_tokens_len"]
        for field in list_fields:
            if field in notnone_batches[0]:
                adapted_batch[field] = [b[field] for b in notnone_batches]
                
        return adapted_batch
    
    # ===== LM mode =====
    elif any(field in notnone_batches[0] for field in ["face_token", "hand_token", "lower_token", "upper_token", "audio_token"]):
        adapted_batch = {}
        # Add all tensor fields
        token_fields = ["face_token", "hand_token", "lower_token", "upper_token", "audio_token"]
        for field in token_fields:
            if field in notnone_batches[0]:
                adapted_batch[field] = collate_tensors([b[field].float() for b in notnone_batches])
        
        # Add list fields
        list_fields = ["tasks", "m_tokens_len", "a_tokens_len", "text"]
        for field in list_fields:
            if field in notnone_batches[0]:
                adapted_batch[field] = [b[field] for b in notnone_batches]
                
        return adapted_batch
        
    # ===== FACE DATASET (default) =====
    else:
        # Default to face data collation
        face_data = "face" in notnone_batches[0]
        shape_data = "shape" in notnone_batches[0]
        pose_data = "pose" in notnone_batches[0]
        
        adapted_batch = {}
        
        # Add face data if available
        if face_data:
            adapted_batch['face'] = collate_tensors([b["face"].float() for b in notnone_batches])
            
        # Add shape data if available
        if shape_data:
            adapted_batch['shape'] = collate_tensors([b["shape"].float() for b in notnone_batches])
            
        # Add pose data if available
        if pose_data:
            adapted_batch['pose'] = collate_tensors([b["pose"].float() for b in notnone_batches])
            
        # Add metadata
        if "motion_len" in notnone_batches[0]:
            adapted_batch['motion_len'] = [b.get("motion_len", 0) for b in notnone_batches]
        if "dataset_name" in notnone_batches[0]:
            adapted_batch['dataset_name'] = [b.get("dataset_name", "") for b in notnone_batches]
        if "id_name" in notnone_batches[0]:
            adapted_batch['id_name'] = [b.get("id_name", "") for b in notnone_batches]
            
        return adapted_batch

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

def add_system_template_and_chunk(text, max_seq_length=1024):
    """
    Add system template to conversation data and ensure it fits within max_seq_length.
    
    Args:
        text: The original conversation text in format:
              <|user|>\n <|audio_XXX|>...<|face_XXX|>...
              <|assistant|>\n <|audio_XXX|>...<|face_XXX|>...
              或
              <|speaker_A|> <|audio_XXX|>...<|face_XXX|>...
              <|speaker_B|> <|audio_XXX|>...<|face_XXX|>...
        max_seq_length: Maximum sequence length (default: 1024)
        
    Returns:
        Formatted text with system template, chunked to max_seq_length
    """
    # Define system template
    system_template = "<|system|>\nThe user will provide an audio conversation along with face motion data. Do it step by step.First, think about the audio conversation and respond in a interleaved manner, with 13 face token followed by 26 audio tokens.\n"
    
    # Estimate system template token count
    system_token_count = 49  # Confirmed token count for the system template
    
    # If no text provided, just return the system template
    if not text or text.strip() == "":
        return system_template
    
    # 先将speaker格式转换为chat格式
    if "<|speaker_A|>" in text or "<|speaker_B|>" in text:
        text = convert_speaker_format_to_chat_format(text)
    
    # We need to chunk the conversation into user-assistant pairs
    # Split by user marker
    parts = text.split("<|user|>")
    
    # Skip empty first part if needed
    if parts and not parts[0].strip():
        parts = parts[1:]
    
    if not parts:
        return system_template
    
    # Reconstruct turns, ensuring we keep user-assistant pairs together
    turns = []
    for i, part in enumerate(parts):
        if i == 0:
            turn = "<|user|>" + part
        else:
            turn = "<|user|>" + part
            
        # Check if this is a complete turn with both user and assistant
        if "<|assistant|>" in turn:
            turns.append(turn)
        else:
            # Incomplete turn - add only if it's the last part
            if i == len(parts) - 1:
                turns.append(turn)
    
    # Estimate tokens for each turn more precisely
    turn_tokens = []
    for turn in turns:
        # Count special tokens and words
        special_tokens = turn.count("<|user|>") + turn.count("<|assistant|>")
        
        # Count audio and face tokens - each is exactly 1 token
        audio_tokens = turn.count("<|audio_")
        face_tokens = turn.count("<|face_")
        
        # Count newlines
        formatting = turn.count("\n")
        
        # Total tokens for this turn
        total = special_tokens + audio_tokens + face_tokens + formatting
        turn_tokens.append(total)
    
    # Start with the system template
    result = system_template
    current_tokens = system_token_count
    
    # Add turns until we reach the token limit
    for i, (turn, tokens) in enumerate(zip(turns, turn_tokens)):
        if current_tokens + tokens <= max_seq_length:
            result += turn
            current_tokens += tokens
        else:
            # This turn would exceed the limit
            break
    
    return result

def convert_speaker_format_to_chat_format(text):
    """
    Convert <|speaker_A|> and <|speaker_B|> format to <|user|>\n and <|assistant|>\n format
    
    Args:
        text: Text marked with <|speaker_A|> and <|speaker_B|> tags
        
    Returns:
        Text marked with <|user|>\n and <|assistant|>\n tags
    """
    # Randomly decide whether speaker_A is user or assistant
    if random.random() < 0.5:
        # speaker_A is user, speaker_B is assistant
        text = text.replace("<|speaker_A|>", "<|user|>\n")
        text = text.replace("<|speaker_B|>", "<|assistant|>\n")
    else:
        # speaker_B is user, speaker_A is assistant
        text = text.replace("<|speaker_B|>", "<|user|>\n")
        text = text.replace("<|speaker_A|>", "<|assistant|>\n")
    
    return text

def create_fixed_length_conversation(text, num_rounds=11):
    """
    Create a fixed-length conversation with exactly the specified number of rounds.
    Each conversation includes the system template and exactly num_rounds of user-assistant interactions.
    
    Args:
        text: The original conversation text
        num_rounds: Number of conversation rounds to include (default: 11)
        
    Returns:
        List of formatted conversations, each with system template and exactly num_rounds
    """
    # Define system template
    system_template = "<|system|>\nThe user will provide an audio conversation along with face motion data. Do it step by step.First, think about the audio conversation and respond in a interleaved manner, with 13 face token followed by 26 audio tokens.\n"
    
    # If no text provided, return empty list
    if not text or text.strip() == "":
        return []
    
    # 先将speaker格式转换为chat格式
    if "<|speaker_A|>" in text or "<|speaker_B|>" in text:
        text = convert_speaker_format_to_chat_format(text)
    
    # Split by user marker to get all user turns
    parts = text.split("<|user|>")
    
    # Skip empty first part if needed
    if parts and not parts[0].strip():
        parts = parts[1:]
    
    if not parts:
        return []
    
    # Reconstruct complete user-assistant pairs
    complete_turns = []
    for i, part in enumerate(parts):
        turn = "<|user|>" + part
        
        # Only include turns that have both user and assistant parts
        if "<|assistant|>" in turn:
            complete_turns.append(turn)
    
    # If we don't have enough turns, return empty list
    if len(complete_turns) < num_rounds:
        return []
    
    # Create conversations with exactly num_rounds
    conversations = []
    for i in range(0, len(complete_turns) - num_rounds + 1):
        # Take exactly num_rounds consecutive turns
        selected_turns = complete_turns[i:i+num_rounds]
        
        # Create conversation with system template
        conversation = system_template + "".join(selected_turns)
        
        conversations.append(conversation)
    
    return conversations



def process_interleaved_text(text, num_rounds=11):
    """
    Process already interleaved text, convert speaker format to chat format, and create fixed-round dialogue sequences.
    
    Args:
        text: Already interleaved text in format <|speaker_A|>...<|speaker_B|>...
        num_rounds: Number of dialogue rounds in each sequence (default: 11)
        
    Returns:
        List of formatted conversations, each with system template and exactly num_rounds
    """
    # Define system template
    system_template = "<|system|>\nThe user will provide an audio conversation along with face motion data. Do it step by step.First, think about the audio conversation and respond in a interleaved manner, with 13 face token followed by 26 audio tokens.\n"
    
    # If input is empty, return empty list
    if not text or text.strip() == "":
        return []
    
    # Convert <|speaker_X|> format to <|user|>\n and <|assistant|>\n format
    if "<|speaker_A|>" in text or "<|speaker_B|>" in text:
        # Randomly decide if speaker_A is user or assistant
        if random.random() < 0.5:
            # speaker_A is user, speaker_B is assistant
            converted_text = text.replace("<|speaker_A|>", "<|user|>\n")
            converted_text = converted_text.replace("<|speaker_B|>", "<|assistant|>\n")
        else:
            # speaker_B is user, speaker_A is assistant
            converted_text = text.replace("<|speaker_B|>", "<|user|>\n")
            converted_text = converted_text.replace("<|speaker_A|>", "<|assistant|>\n")
    else:
        # Already in chat format
        converted_text = text
    
    # Split text by user marker
    parts = converted_text.split("<|user|>")
    
    # Skip empty first part if needed
    if parts and not parts[0].strip():
        parts = parts[1:]
    
    # If no user turns, return empty list
    if not parts:
        return []
    
    # Reconstruct complete user-assistant pairs
    complete_turns = []
    for i, part in enumerate(parts):
        turn = "<|user|>" + part
        
        # Only include turns that have both user and assistant parts
        if "<|assistant|>" in turn:
            complete_turns.append(turn)
    
    # If not enough dialogue rounds, return empty list
    if len(complete_turns) < num_rounds:
        return []
    
    # Create fixed-length sequences, each containing exactly num_rounds of dialogue
    conversations = []
    for i in range(0, len(complete_turns) - num_rounds + 1):
        # Take exactly num_rounds consecutive turns
        selected_turns = complete_turns[i:i+num_rounds]
        
        # Create conversation with system prompt
        conversation = system_template + "".join(selected_turns)
        
        conversations.append(conversation)
    
    return conversations

def huggingface_dataset_collate_with_system(batch):
    """
    Simplified collate function that works with already preprocessed text data.
    This function is kept for backward compatibility but now works the same as huggingface_dataset_collate.
    
    Args:
        batch: A list of samples from the HuggingFace dataset
        
    Returns:
        A dict with 'text' containing the text data directly
    """
    return huggingface_dataset_collate(batch)
