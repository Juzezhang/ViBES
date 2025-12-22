"""
This script implements the GLM-based conversational agent model.
It is adapted from the original conversational agent implementation.

Author: Juze Zhang
License: Check the original repository for licensing details.
"""
import os
import re
from typing import List, Union, Dict, Any, Optional
import numpy as np
import math
import time
import heapq
import torch
from torch import Tensor, nn
from torch.distributions.distribution import Distribution
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForCausalLM, AutoModel
import random
from .tools.token_emb import NewTokenEmb
from huggingface_hub import login
from transformers import TextIteratorStreamer
import threading


class ConverAgentGLM(nn.Module):

    def __init__(
        self,
        model_path: str,
        tokenizer_path: str = None,
        flow_path: str = None,
        stage: str = "lm_pretrain",
        new_token_type: str = "insert",
        motion_framerate: float = 30.0,
        audio_samplerate: float = 16000.0,
        motion_down_sampling: int = 1,
        audio_down_sampling: int = 320,   # audio down sample rate
        predict_ratio: float = 0.2,
        inbetween_ratio: float = 0.25,
        max_length: int = 512,
        lora: bool = False,
        quota_ratio: float = 0.5,
        noise_density: float = 0.15,
        mean_noise_span_length: int = 3,
        flash_attention: bool = False,
        modalities: dict = None,
        **kwargs,
    ) -> None:
        super().__init__()

        # Parameters
        self.max_length = max_length
        self.motion_framerate = motion_framerate
        self.audio_samplerate = audio_samplerate
        self.motion_down_sampling = motion_down_sampling
        self.audio_down_sampling = audio_down_sampling
        self.predict_ratio = predict_ratio
        self.inbetween_ratio = inbetween_ratio
        self.mask_ratio_audio = 0.08

        self.noise_density = noise_density
        self.mean_noise_span_length = mean_noise_span_length
        self.quota_ratio = quota_ratio
        self.stage = stage
        
        # 保存paths
        self.model_path = model_path
        self.tokenizer_path = model_path  # 始终使用model_path作为tokenizer路径
        self.flow_path = flow_path

        # Set up quantization config if available
        try:
            from transformers import BitsAndBytesConfig
            use_quantization = kwargs.get("use_quantization", False)
            if use_quantization:
                # Configure quantization
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True
                )
            else:
                quantization_config = None
        except (ImportError, ModuleNotFoundError):
            print("bitsandbytes not found, will not use quantization")
            quantization_config = None

        # Initialize GLM tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_path, 
            trust_remote_code=True
        )
        
        self.language_model = AutoModel.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            quantization_config=quantization_config,
            # device_map={"": 0}
        )
        self.lm_type = 'dec'
        
        # 如果存在flow path，加载用于语音解码的flow模型
        if self.flow_path:
            try:
                self.flow_model = AutoModel.from_pretrained(
                    self.flow_path,
                    trust_remote_code=True
                )
                print(f"Loaded GLM-4-Voice flow model from {self.flow_path}")
            except Exception as e:
                print(f"Failed to load flow model: {e}")
                self.flow_model = None
        else:
            self.flow_model = None
        
        # Ensure pad token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # # Add special tokens for modalities
        # if modalities:
        #     # Initialize codebook sizes from modalities config
        #     for modality, settings in modalities.items():
        #         # Set codebook sizes based on modality
        #         if modality == "motion":
        #             self.motion_codebook_size = settings["codebook_size"]
        #         elif modality == "face":
        #             self.face_codebook_size = settings["codebook_size"]
        #         elif modality == "hand":
        #             self.hand_codebook_size = settings["codebook_size"]
        #         elif modality == "upper":
        #             self.upper_codebook_size = settings["codebook_size"]
        #         elif modality == "lower":
        #             self.lower_codebook_size = settings["codebook_size"]
        #         elif modality == "audio":
        #             self.audio_codebook_size = settings["codebook_size"]

        #         prefix = settings["prefix"]
        #         codebook_size = settings["codebook_size"] + 3  # Adding 3 for special tokens
        #         print(f"Adding {codebook_size} tokens for {modality} with prefix {prefix}")
                
        #         # Generate tokens for the current modality
        #         # GLM-specific token format (keeping the same as Qwen for consistency)
        #         tokens = [f"[{prefix}{i}]" for i in range(codebook_size)]
        #         self.tokenizer.add_tokens(tokens, special_tokens=False)

        
        # Generate tokens for the current modality
        # GLM-specific token format (keeping the same as Qwen for consistency)
        tokens = [f"<|face_{i}|>" for i in range(256)]
        self.tokenizer.add_tokens(tokens, special_tokens=False)
            

        # Resize token embeddings based on new tokens
        if new_token_type == "insert":
            self.language_model.resize_token_embeddings(len(self.tokenizer))
        elif new_token_type == "mlp":
            shared = NewTokenEmb(self.language_model.shared,
                                 self.motion_codebook_size + 3)
            self.language_model.resize_token_embeddings(len(self.tokenizer))
            self.language_model.shared = shared

        # Lora setup for parameter-efficient fine-tuning
        if lora:
            from peft import LoraConfig, TaskType, get_peft_model
            # GLM-specific LoRA configuration (adapted from Qwen config)
            peft_config = LoraConfig(
                bias="none",
                task_type="CAUSAL_LM",
                r=kwargs.get("lora_rank", 16),
                lora_alpha=kwargs.get("lora_alpha", 32),
                lora_dropout=kwargs.get("lora_dropout", 0.05),
                target_modules=kwargs.get("lora_target_modules", 
                                          ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"])
            )
            self.language_model = get_peft_model(self.language_model, peft_config)

    def audio_token_to_string(self, tokens, lengths=None):
        if tokens is None:
            return []

        # 调试信息
        print(f"Audio token_to_string: tokens shape={tokens.shape if isinstance(tokens, torch.Tensor) else 'not tensor'}")
        print(f"Audio token_to_string: lengths={lengths[:5] if lengths is not None else None}")

        if isinstance(tokens, list):
            tokens = torch.tensor(tokens)

        if lengths is None:
            # Assume all tokens are valid if lengths not provided
            lengths = [tokens.shape[1]] * tokens.shape[0]

        token_strings = []
        for i, length in enumerate(lengths):
            try:
                # Ensure length is valid
                if length is None:
                    token_strings.append("")
                    continue
                    
                # Process only valid tokens up to the specified length
                valid_tokens = tokens[i, :length]
                
                # Convert audio tokens to string with appropriate format
                audio_str = ' '.join([f"[atoken{int(t.item() if isinstance(t, torch.Tensor) else t)}]" for t in valid_tokens])
                token_strings.append(audio_str)
            except Exception as e:
                print(f"Error in audio_token_to_string for item {i}: {e}")
                # Add empty string as fallback
                token_strings.append("")
            
        return token_strings

    def motion_token_to_string(self, tokens, lengths=None):
        if tokens is None:
            return []

        # Debug information
        print(f"Motion token_to_string: tokens shape={tokens.shape if isinstance(tokens, torch.Tensor) else 'not tensor'}")
        print(f"Motion token_to_string: lengths={lengths[:5] if lengths is not None else None}")

        if isinstance(tokens, list):
            tokens = torch.tensor(tokens)

        if lengths is None:
            # Assume all tokens are valid if lengths not provided
            lengths = [tokens.shape[1]] * tokens.shape[0]

        token_strings = []
        for i, length in enumerate(lengths):
            try:
                # Ensure length is valid
                if length is None:
                    token_strings.append("")
                    continue
                    
                # Process only valid tokens up to the specified length
                valid_tokens = tokens[i, :length]
                
                # Convert motion tokens to string with appropriate format
                motion_str = ' '.join([f"[mtoken{int(t.item() if isinstance(t, torch.Tensor) else t)}]" for t in valid_tokens])
                token_strings.append(motion_str)
            except Exception as e:
                print(f"Error in motion_token_to_string for item {i}: {e}")
                # Add empty string as fallback
                token_strings.append("")
            
        return token_strings

    def face_token_to_string(self, tokens, lengths=None):
        if tokens is None:
            return []

        # 调试信息
        print(f"Face token_to_string: tokens shape={tokens.shape if isinstance(tokens, torch.Tensor) else 'not tensor'}")
        print(f"Face token_to_string: lengths={lengths[:5] if lengths is not None else None}")

        if isinstance(tokens, list):
            tokens = torch.tensor(tokens)

        if lengths is None:
            # Assume all tokens are valid if lengths not provided
            lengths = [tokens.shape[1]] * tokens.shape[0]

        token_strings = []
        for i, length in enumerate(lengths):
            try:
                # Ensure length is valid
                if length is None:
                    token_strings.append("")
                    continue
                    
                # Process only valid tokens up to the specified length
                valid_tokens = tokens[i, :length]
                
                # Convert face tokens to string with appropriate format
                face_str = ' '.join([f"[ftoken{int(t.item() if isinstance(t, torch.Tensor) else t)}]" for t in valid_tokens])
                token_strings.append(face_str)
            except Exception as e:
                print(f"Error in face_token_to_string for item {i}: {e}")
                # Add empty string as fallback
                token_strings.append("")
            
        return token_strings

    def compositional_motion_token_to_string(self, face_tokens, hand_tokens, lower_tokens, upper_tokens, motion_lengths=None):
        """Convert all motion-related tokens to string representation."""
        face_strings = self.face_token_to_string(face_tokens, motion_lengths)
        hand_strings = self.motion_token_to_string(hand_tokens, motion_lengths) if hand_tokens is not None else [''] * (len(face_strings) if face_strings else 0)
        upper_strings = self.motion_token_to_string(upper_tokens, motion_lengths) if upper_tokens is not None else [''] * (len(face_strings) if face_strings else 0)
        lower_strings = self.motion_token_to_string(lower_tokens, motion_lengths) if lower_tokens is not None else [''] * (len(face_strings) if face_strings else 0)
        motion_string = [''] * (len(face_strings) if face_strings else 0)  # Full motion representation placeholder
        
        return face_strings, hand_strings, upper_strings, lower_strings, motion_string

    def placeholder_fulfill(self, template, length, audio_length,
                          face_string, hand_string, upper_string, lower_string,
                          motion_string, audio_string, text):
        """Replace placeholders in templates with actual content."""
        if template is None:
            return ""
            
        result = template.replace("<Caption_Placeholder>", text if text else "")
        result = result.replace("<Audio_Placeholder>", audio_string if audio_string else "")
        result = result.replace("<Motion_Placeholder>", motion_string if motion_string else "")
        result = result.replace("<Face_Placeholder>", face_string if face_string else "")
        result = result.replace("<Hand_Placeholder>", hand_string if hand_string else "")
        result = result.replace("<Upper_Placeholder>", upper_string if upper_string else "")
        result = result.replace("<Lower_Placeholder>", lower_string if lower_string else "")
        result = result.replace("<Interleaved_Placeholder>", face_string if face_string else "")  # Use face_string as a fallback for interleaved tokens
        result = result.replace("<Length_Placeholder>", str(length) if length is not None else "0")
        result = result.replace("<Audio_Length_Placeholder>", str(audio_length) if audio_length is not None else "0")
        
        return result

    def placeholder_fulfill_interleaved(self, template, length, audio_length,
                                     interleaved_string, audio_string, text):
        """Replace placeholders in templates with actual content using interleaved tokens."""
        if template is None:
            return ""
            
        result = template.replace("<Caption_Placeholder>", text if text else "")
        result = result.replace("<Audio_Placeholder>", audio_string if audio_string else "")
        result = result.replace("<Interleaved_Placeholder>", interleaved_string if interleaved_string else "")
        # For backward compatibility
        result = result.replace("<Motion_Placeholder>", interleaved_string if interleaved_string else "")
        result = result.replace("<Face_Placeholder>", interleaved_string if interleaved_string else "")
        result = result.replace("<Hand_Placeholder>", "")
        result = result.replace("<Upper_Placeholder>", "")
        result = result.replace("<Lower_Placeholder>", "")
        result = result.replace("<Length_Placeholder>", str(length) if length is not None else "0")
        result = result.replace("<Audio_Length_Placeholder>", str(audio_length) if audio_length is not None else "0")
        
        return result

    def template_fulfill(self,
                         tasks,
                         lengths,
                         audio_lengths,
                         face_strings,
                         hand_strings,
                         upper_strings,
                         lower_strings,
                         motion_string,
                         audio_strings,
                         texts,
                         stage='test'):
        """Fill templates with content."""
        inputs = []
        outputs = []
        if audio_lengths is None or audio_lengths[0] is None:
            audio_strings = [''] * len(lengths)

        for i in range(len(lengths)):
            input_template = random.choice(tasks[i]['input'])
            output_template = random.choice(tasks[i]['output'])
            length = lengths[i]
            audio_length = audio_lengths[i] if audio_lengths else None
            inputs.append(
                self.placeholder_fulfill(input_template, length, audio_length,
                                         face_strings[i], hand_strings[i],
                                         upper_strings[i], lower_strings[i], motion_string[i],
                                         audio_strings[i], texts[i]))
            outputs.append(
                self.placeholder_fulfill(output_template, length, audio_length,
                                         face_strings[i], hand_strings[i],
                                         upper_strings[i], lower_strings[i], motion_string[i],
                                         audio_strings[i], texts[i]))

        return inputs, outputs

    def template_fulfill_interleaved(self,
                                  tasks,
                                  lengths,
                                  audio_lengths,
                                  interleaved_strings,
                                  audio_strings,
                                  texts,
                                  stage='test'):
        """Fill templates with content using interleaved tokens."""
        inputs = []
        outputs = []
        if audio_lengths is None or audio_lengths[0] is None:
            audio_strings = [''] * len(lengths)

        for i in range(len(lengths)):
            input_template = random.choice(tasks[i]['input'])
            output_template = random.choice(tasks[i]['output'])
            length = lengths[i]
            audio_length = audio_lengths[i] if audio_lengths else None
            inputs.append(
                self.placeholder_fulfill_interleaved(input_template, length, audio_length,
                                               interleaved_strings[i], audio_strings[i], texts[i]))
            outputs.append(
                self.placeholder_fulfill_interleaved(output_template, length, audio_length,
                                                interleaved_strings[i], audio_strings[i], texts[i]))

        return inputs, outputs

    def forward(self, 
                texts: List[str],
                **kwargs):
        """Forward pass through the model.
        
        Note: The texts parameter is expected to already contain interleaved token strings.
        """
        self.tokenizer.padding_side = "right"
        
        # # Format sequences for GLM - the text input already contains the interleaved tokens
        # full_sequences = []
        # for i, text in enumerate(texts):
        #     # GLM-4 specific chat format with empty answer
        #     full_sequences.append(f"[Round {i+1}]\n\n问：{text}\n\n答：")
        full_sequences = texts
        # Move to appropriate device
        device = self.language_model.device

        # Tokenize with GLM specific settings
        # tokenized_inputs = self.tokenizer(full_sequences,
        #                       padding='max_length',
        #                       max_length=self.max_length,
        #                       truncation=True,
        #                       return_attention_mask=True,
        #                       return_tensors="pt")

        tokenized_inputs = self.tokenizer(full_sequences,
                              truncation=True,
                              return_tensors="pt")
        attention_mask = tokenized_inputs.attention_mask
        input_ids = tokenized_inputs.input_ids

        # 手动将attention_mask补齐到1024长度
        batch_size, current_length = attention_mask.shape
        if current_length < self.max_length:
            padding_length = self.max_length - current_length
            padding = torch.zeros((batch_size, padding_length), dtype=attention_mask.dtype)
            attention_mask = torch.cat([attention_mask, padding], dim=1)
            input_ids = torch.cat([input_ids, padding], dim=1)


        labels = input_ids.clone()
        
        # For each sequence, find where the instruction ends and only compute loss on the output
        for i, sequence in enumerate(full_sequences):
            # Find positions of all user and assistant markers
            input_ids_list = input_ids[i].tolist()
            
            # First, set all labels to -100 (we'll selectively enable only assistant responses)
            labels[i, :] = -100
            
            # Get tokenized versions of the markers
            user_marker = self.tokenizer.encode("<|user|>", add_special_tokens=False)
            assistant_marker = self.tokenizer.encode("<|assistant|>", add_special_tokens=False)
            
            # Find all positions of user and assistant markers
            user_positions = []
            assistant_positions = []
            
            for idx in range(len(input_ids_list) - len(user_marker) + 1):
                if input_ids_list[idx:idx+len(user_marker)] == user_marker:
                    user_positions.append(idx)
                    
            for idx in range(len(input_ids_list) - len(assistant_marker) + 1):
                if input_ids_list[idx:idx+len(assistant_marker)] == assistant_marker:
                    assistant_positions.append(idx)
            
            # Process each assistant section - compute loss only for assistant responses
            for j, asst_pos in enumerate(assistant_positions):
                # Find end of this assistant section (next user marker or end of sequence)
                end_pos = user_positions[j+1] if j+1 < len(user_positions) else len(input_ids_list)
                
                # Enable loss computation for this assistant section
                # Start after the assistant marker and its following newline token
                start_pos = asst_pos + len(assistant_marker) + 1  # +1 for newline after marker
                labels[i, start_pos:end_pos] = torch.tensor(input_ids_list[start_pos:end_pos], device=labels.device)

        labels = labels.to(device)
        attention_mask = attention_mask.to(device)
        input_ids = input_ids.to(device)
        
        # Forward pass through the model
        outputs = self.language_model(input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    labels=labels)

        return outputs

    def generate_direct(self,
                        input: List[str] = None,
                        max_length: int = 512,
                        num_beams: int = 1,
                        do_sample: bool = True,
                        temperature: float = 0.7,
                        top_p: float = 0.9,
                        repetition_penalty: float = 1.05,
                        bad_words_ids: List[int] = None):
        """Generate text output directly from input prompts."""
        self.device = self.language_model.device
        
        # Format inputs for GLM
        formatted_inputs = []
        for prompt in input:
            if isinstance(prompt, str):
                # Apply GLM chat format
                formatted_inputs.append(f"[Round 1]\n\n问：{prompt}\n\n答：")
            else:
                formatted_inputs.append("")
        
        # Set tokenizer options for generation
        self.tokenizer.padding_side = 'left'
        
        # Tokenize the inputs
        source_encoding = self.tokenizer(
            formatted_inputs,
            padding='longest',
            max_length=self.max_length // 2,  # Use half of max length for input
            truncation=True,
            return_tensors="pt"
        )
        
        source_input_ids = source_encoding.input_ids.to(self.device)
        source_attention_mask = source_encoding.attention_mask.to(self.device)

        # Generate with GLM-specific parameters
        outputs = self.language_model.generate(
            input_ids=source_input_ids,
            attention_mask=source_attention_mask,
            pad_token_id=self.tokenizer.pad_token_id,
            do_sample=do_sample,
            max_new_tokens=max_length,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            use_cache=True,
            bad_words_ids=bad_words_ids
        )
        
        # Reset padding side for other operations
        self.tokenizer.padding_side = 'right'

        # Decode outputs to get full text including prompts
        outputs_string = self.tokenizer.batch_decode(
            outputs,
            skip_special_tokens=True
        )
        
        # Extract only the generated answers
        cleaned_outputs = []
        for output in outputs_string:
            # Extract answer part after '答：'
            parts = output.split('答：')
            if len(parts) > 1:
                answer = parts[1].strip()
                cleaned_outputs.append(answer)
            else:
                cleaned_outputs.append("")
        
        return cleaned_outputs

    def generate_conditional(self,
                             texts: Optional[List[str]] = None,
                             **kwargs):
        """Generate tokens using a decoder-only model conditioned on text inputs.
        
        Args:
            texts: List of input prompts (should not contain tokens)
            
        Returns:
            Dictionary with generated tokens and text
        """
        self.device = self.language_model.device
        
        if not texts:
            return {
                'tokens': [],
                'text': []
            }
        
        # Generate tokens from input prompts 
        generated_text = self.generate_direct(
            texts, max_length=self.max_length, num_beams=1, do_sample=True
        )
        
        # Initialize results container
        result = {
            'text': generated_text
        }
        
        # Parse generated text to extract tokens 
        interleaved_tokens_out = []
        
        for text in generated_text:
            # Try all possible token formats in order of preference
            formats_to_try = [
                (r'\[itoken(\d+)\]', 'interleaved'),
                (r'\[ftoken(\d+)\]', 'face'),
                (r'\[mtoken(\d+)\]', 'motion'),
                (r'\[token(\d+)\]', 'generic')
            ]
            
            token_seq = []
            for pattern, _ in formats_to_try:
                matches = re.findall(pattern, text)
                if matches:
                    token_seq = [int(token) for token in matches]
                    break
            
            interleaved_tokens_out.append(token_seq)
        
        # Store results
        result['tokens'] = interleaved_tokens_out
        
        return result

    def generate_streaming(self,
                           input: str,
                           max_length: int = 512,
                           temperature: float = 0.7,
                           top_p: float = 0.9,
                           repetition_penalty: float = 1.05):
        """Generate text in a streaming fashion for real-time applications.
        
        Args:
            input: Input prompt
            max_length: Maximum length of generated text
            temperature: Temperature for sampling
            top_p: Top-p sampling parameter
            repetition_penalty: Repetition penalty parameter
            
        Returns:
            Generator that yields text chunks as they are generated
        """
        self.device = self.language_model.device
        
        # Format for GLM chat
        formatted_input = f"[Round 1]\n\n问：{input}\n\n答："
        
        # Tokenize the input
        input_ids = self.tokenizer(
            formatted_input,
            return_tensors="pt"
        ).input_ids.to(self.device)
        
        # Stream tokens
        streamer = TextIteratorStreamer(
            self.tokenizer, 
            skip_prompt=True,
            skip_special_tokens=True
        )
        
        # Generate with streaming
        generation_kwargs = {
            "input_ids": input_ids,
            "max_new_tokens": max_length,
            "temperature": temperature,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
            "streamer": streamer,
            "do_sample": True
        }
        
        # Start generation in a separate thread
        threading.Thread(
            target=self.language_model.generate, 
            kwargs=generation_kwargs
        ).start()
        
        # Return the streamer
        return streamer 

    def interleaved_token_to_string(self, tokens, lengths=None):
        """Convert interleaved tokens to string representation.
        
        Args:
            tokens: Tensor of interleaved token IDs
            lengths: List of valid sequence lengths
            
        Returns:
            List of string representations of token sequences
        """
        if tokens is None:
            return []

        # Debug information
        print(f"Interleaved token_to_string: tokens shape={tokens.shape if isinstance(tokens, torch.Tensor) else 'not tensor'}")
        print(f"Interleaved token_to_string: lengths={lengths[:5] if lengths is not None else None}")

        if isinstance(tokens, list):
            tokens = torch.tensor(tokens)

        if lengths is None:
            # Assume all tokens are valid if lengths not provided
            lengths = [tokens.shape[1]] * tokens.shape[0]

        token_strings = []
        for i, length in enumerate(lengths):
            try:
                # Ensure length is valid
                if length is None:
                    token_strings.append("")
                    continue
                    
                # Process only valid tokens up to the specified length
                valid_tokens = tokens[i, :length]
                
                # Convert interleaved tokens to string with appropriate format
                # Using a generic token format that doesn't specify body part
                token_str = ' '.join([f"[itoken{int(t.item() if isinstance(t, torch.Tensor) else t)}]" for t in valid_tokens])
                token_strings.append(token_str)
            except Exception as e:
                print(f"Error in interleaved_token_to_string for item {i}: {e}")
                # Add empty string as fallback
                token_strings.append("")
            
        return token_strings 