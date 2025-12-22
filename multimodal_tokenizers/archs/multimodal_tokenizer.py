import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import os
import glob
import safetensors
from transformers import WhisperFeatureExtractor, WhisperTokenizerFast

# Import GLM-4-Voice components
from reference_project.GLM_4_Voice.speech_tokenizer.configuration_whisper import WhisperVQConfig
from reference_project.GLM_4_Voice.speech_tokenizer.modeling_whisper import WhisperVQEncoder
from reference_project.GLM_4_Voice.speech_tokenizer.utils import extract_speech_token, load_quantize_encoder

# Import face VQ-VAE model
from conver_agent.archs.lom_vq import VQVAEConvZeroDSUS_PaperVersion


class MultimodalTokenizer:
    """
    A multimodal tokenizer that combines audio and face expression tokenization capabilities.
    
    This tokenizer extends the GLM-4-Voice audio tokenizer to handle face tokens,
    allowing for joint processing of audio and facial expressions.
    """
    
    def __init__(
        self,
        audio_model_path,
        face_model_path,
        device="cuda" if torch.cuda.is_available() else "cpu",
        face_input_dim=112,  # 6D head + 6D jaw + 100 expression
        face_codebook_size=256
    ):
        """
        Initialize the multimodal tokenizer.
        
        Args:
            audio_model_path: Path to the GLM-4-Voice audio tokenizer model
            face_model_path: Path to the face VQ-VAE tokenizer model
            device: Device to run the models on (cuda or cpu)
            face_input_dim: Dimension of the face input features
            face_codebook_size: Size of the face codebook
        """
        self.device = device
        
        # Initialize audio tokenizer components
        self.audio_model_path = audio_model_path
        self.audio_encoder = self._init_audio_encoder(audio_model_path)
        self.audio_feature_extractor = WhisperFeatureExtractor.from_pretrained(audio_model_path)
        
        # Initialize face tokenizer components
        self.face_model_path = face_model_path
        self.face_input_dim = face_input_dim
        self.face_codebook_size = face_codebook_size
        self.face_encoder = self._init_face_encoder(face_model_path, face_input_dim, face_codebook_size)
        
        # Setup tokenization parameters
        self._setup_tokenization_parameters()
    
    def _init_audio_encoder(self, model_path):
        """
        Initialize the audio encoder from GLM-4-Voice.
        
        Args:
            model_path: Path to the pretrained GLM-4-Voice model
            
        Returns:
            Initialized WhisperVQEncoder model
        """
        print(f"Loading audio encoder from {model_path}")
        audio_encoder = load_quantize_encoder(model_path)
        audio_encoder.eval()
        return audio_encoder
    
    def _init_face_encoder(self, model_path, input_dim, codebook_size):
        """
        Initialize the face VQ-VAE encoder.
        
        Args:
            model_path: Path to the face VQ-VAE model checkpoint
            input_dim: Dimension of face parameters
            codebook_size: Size of the face codebook
            
        Returns:
            Initialized face VQ-VAE model
        """
        print(f"Loading face encoder from {model_path}")
        
        # Initialize face VQ-VAE model
        face_encoder = VQVAEConvZeroDSUS_PaperVersion(
            vae_layer=3,
            code_num=codebook_size,
            codebook_size=codebook_size,
            vae_quantizer_lambda=1.0,
            vae_test_dim=input_dim
        )
        
        # Load pre-trained weights
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'state_dict' in checkpoint:
                checkpoint = checkpoint['state_dict']
            
            # Remove 'model.' prefix if it exists
            new_state_dict = {}
            for k, v in checkpoint.items():
                if k.startswith('model.'):
                    new_state_dict[k[6:]] = v
                else:
                    new_state_dict[k] = v
            
            missing, unexpected = face_encoder.load_state_dict(new_state_dict, strict=False)
            if missing:
                print(f"Missing keys in face encoder: {missing}")
            if unexpected:
                print(f"Unexpected keys in face encoder: {unexpected}")
                
        face_encoder.to(self.device)
        face_encoder.eval()
        return face_encoder
    
    def _setup_tokenization_parameters(self):
        """Setup parameters for tokenization."""
        # Audio tokenization parameters
        self._resample_buffer = {}
        
        # Face tokenization parameters
        self.face_token_range = (self.audio_feature_extractor.num_mel_bins, 
                                self.audio_feature_extractor.num_mel_bins + self.face_codebook_size)
        
        print(f"Audio token range: (0, {self.audio_feature_extractor.num_mel_bins})")
        print(f"Face token range: {self.face_token_range}")
    
    def tokenize_audio(self, audio_path_or_data):
        """
        Tokenize audio data into discrete tokens.
        
        Args:
            audio_path_or_data: Path to audio file or audio data (as numpy array or tensor)
            
        Returns:
            List of audio token sequences
        """
        # Handle different input types
        if isinstance(audio_path_or_data, str):
            audio_inputs = [audio_path_or_data]
        elif isinstance(audio_path_or_data, list) and all(isinstance(x, str) for x in audio_path_or_data):
            audio_inputs = audio_path_or_data
        else:
            # Assume it's already loaded audio data
            if isinstance(audio_path_or_data, torch.Tensor):
                audio_data = audio_path_or_data.cpu().numpy()
            else:
                audio_data = audio_path_or_data
                
            if audio_data.ndim == 1:  # Single audio
                audio_data = [audio_data]
            audio_inputs = [(data, 16000) for data in audio_data]  # Assume 16kHz
        
        # Tokenize audio
        with torch.no_grad():
            audio_tokens = extract_speech_token(self.audio_encoder, self.audio_feature_extractor, audio_inputs)
        
        return audio_tokens
    
    def tokenize_face(self, face_data):
        """
        Tokenize face parameters into discrete tokens.
        
        Args:
            face_data: Face parameters as numpy array or tensor
                       Shape: (batch_size, sequence_length, face_input_dim) or
                              (sequence_length, face_input_dim)
                              
        Returns:
            Tokenized face parameters
        """
        # Ensure data is a tensor
        if isinstance(face_data, np.ndarray):
            face_data = torch.from_numpy(face_data).float()
        
        # Handle different input shapes
        if face_data.dim() == 2:  # (sequence_length, face_input_dim)
            face_data = face_data.unsqueeze(0)  # (1, sequence_length, face_input_dim)
        
        # Ensure the data is on the right device
        face_data = face_data.to(self.device)
        
        # Tokenize face data
        with torch.no_grad():
            outputs = self.face_encoder(face_data)
            
            # Get token indices
            if hasattr(outputs, 'quantized_token_ids'):
                face_tokens = outputs.quantized_token_ids
            else:
                # Extract token indices from the VQ-VAE output
                face_tokens = outputs[1]  # Assuming [1] contains token indices
                
        # Offset the face tokens to avoid overlap with audio tokens
        face_tokens = face_tokens + self.face_token_range[0]
            
        return face_tokens
    
    def detokenize_face(self, face_tokens):
        """
        Convert face tokens back to face parameters.
        
        Args:
            face_tokens: Tokenized face parameters
                        Shape: (batch_size, sequence_length) or (sequence_length,)
                        
        Returns:
            Reconstructed face parameters
        """
        # Ensure data is a tensor
        if isinstance(face_tokens, np.ndarray):
            face_tokens = torch.from_numpy(face_tokens).long()
        
        # Handle different input shapes
        if face_tokens.dim() == 1:  # (sequence_length,)
            face_tokens = face_tokens.unsqueeze(0)  # (1, sequence_length)
        
        # Remove offset from face tokens
        face_tokens = face_tokens - self.face_token_range[0]
        
        # Ensure tokens are within valid range
        face_tokens = torch.clamp(face_tokens, 0, self.face_codebook_size - 1)
        
        # Ensure the data is on the right device
        face_tokens = face_tokens.to(self.device)
        
        # Decode face tokens
        with torch.no_grad():
            # Use the decoder's embedding to convert tokens back to features
            # Note: actual implementation will depend on the specific VQ-VAE model
            reconstructed_faces = self.face_encoder.decode_by_token_ids(face_tokens)
            
        return reconstructed_faces
    
    def combine_tokens(self, audio_tokens, face_tokens, interleave=False):
        """
        Combine audio and face tokens into a single sequence.
        
        Args:
            audio_tokens: Audio token sequence
            face_tokens: Face token sequence
            interleave: If True, interleave audio and face tokens
                       If False, concatenate them (audio first, then face)
                       
        Returns:
            Combined token sequence
        """
        if interleave:
            # Interleave audio and face tokens
            # This requires audio_tokens and face_tokens to have the same length
            combined_tokens = []
            min_len = min(len(audio_tokens), len(face_tokens))
            
            for i in range(min_len):
                combined_tokens.append(audio_tokens[i])
                combined_tokens.append(face_tokens[i])
                
            # Add remaining tokens from the longer sequence
            if len(audio_tokens) > min_len:
                combined_tokens.extend(audio_tokens[min_len:])
            if len(face_tokens) > min_len:
                combined_tokens.extend(face_tokens[min_len:])
        else:
            # Simply concatenate the token sequences (audio first, then face)
            combined_tokens = audio_tokens + face_tokens
            
        return combined_tokens
    
    def separate_tokens(self, combined_tokens, audio_length=None, face_length=None):
        """
        Separate combined tokens back into audio and face tokens.
        
        Args:
            combined_tokens: Combined token sequence
            audio_length: Length of audio token sequence (for concatenated mode)
            face_length: Length of face token sequence (for concatenated mode)
            
        Returns:
            Tuple of (audio_tokens, face_tokens)
        """
        # Identify audio and face tokens based on their values
        audio_tokens = []
        face_tokens = []
        
        for token in combined_tokens:
            if token < self.face_token_range[0]:
                audio_tokens.append(token)
            else:
                face_tokens.append(token)
                
        return audio_tokens, face_tokens
    
    def save(self, output_dir):
        """
        Save the multimodal tokenizer.
        
        Args:
            output_dir: Directory to save the tokenizer to
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save configuration
        config = {
            "audio_model_path": self.audio_model_path,
            "face_model_path": self.face_model_path,
            "face_input_dim": self.face_input_dim,
            "face_codebook_size": self.face_codebook_size,
            "face_token_range": self.face_token_range,
        }
        
        # Save configuration
        torch.save(config, os.path.join(output_dir, "multimodal_tokenizer_config.pt"))
        
        # Note: The actual models (audio_encoder and face_encoder) are not saved
        # as they are loaded from their respective paths
        
        print(f"Multimodal tokenizer configuration saved to {output_dir}")
    
    @classmethod
    def from_pretrained(cls, model_dir):
        """
        Load a multimodal tokenizer from a saved directory.
        
        Args:
            model_dir: Directory containing the saved tokenizer
            
        Returns:
            Initialized MultimodalTokenizer
        """
        config_path = os.path.join(model_dir, "multimodal_tokenizer_config.pt")
        if not os.path.exists(config_path):
            raise ValueError(f"Configuration file not found at {config_path}")
            
        # Load configuration
        config = torch.load(config_path)
        
        # Initialize tokenizer from configuration
        tokenizer = cls(
            audio_model_path=config["audio_model_path"],
            face_model_path=config["face_model_path"],
            face_input_dim=config["face_input_dim"],
            face_codebook_size=config["face_codebook_size"]
        )
        
        return tokenizer 