"""
Token processing utilities for inference.
Handles token extraction, parsing, and formatting from model outputs.
"""
from typing import List, Dict, Optional


def extract_token_ids_from_string(token_strings: List[str]) -> List[int]:
    """
    Extract token IDs from token string representations like '<|audio_1|>'.
    
    Args:
        token_strings: List of token strings to parse
        
    Returns:
        List of extracted token IDs
        
    Example:
        >>> extract_token_ids_from_string(['<|audio_123|>', '<|face_456|>'])
        [123, 456]
    """
    return [int(val.split('_')[1].split('|')[0]) for val in token_strings if isinstance(val, str)]


def extract_modality_tokens_from_response(
    full_response: str,
    modality_names: Optional[List[str]] = None
) -> Dict[str, List[int]]:
    """
    Extract tokens for different modalities from a model response.
    
    Args:
        full_response: Full decoded response string from the model
        modality_names: List of modality names to extract (default: ['audio', 'face', 'upper', 'lower', 'hand'])
        
    Returns:
        Dictionary mapping modality names to lists of token IDs
        
    Example:
        >>> response = "<|audio_1|><|face_2|><|upper_3|>"
        >>> extract_modality_tokens_from_response(response)
        {'audio': [1], 'face': [2], 'upper': [3], 'lower': [], 'hand': []}
    """
    if modality_names is None:
        modality_names = ['audio', 'face', 'upper', 'lower', 'hand']
    
    response_split = full_response.split("<|")
    tokens = {}
    
    for modality in modality_names:
        modality_strings = [
            s for s in response_split 
            if f"{modality}_" in s
        ]
        tokens[modality] = extract_token_ids_from_string(modality_strings)
    
    return tokens


def parse_token_string(token_string: str) -> Optional[int]:
    """
    Parse a single token string to extract the token ID.
    
    Args:
        token_string: Single token string like '<|audio_123|>'
        
    Returns:
        Token ID if parsing succeeds, None otherwise
        
    Example:
        >>> parse_token_string('<|audio_123|>')
        123
        >>> parse_token_string('invalid')
        None
    """
    try:
        if not isinstance(token_string, str) or '_' not in token_string:
            return None
        parts = token_string.split('_')
        if len(parts) < 2:
            return None
        token_id_str = parts[1].split('|')[0]
        return int(token_id_str)
    except (ValueError, IndexError):
        return None

