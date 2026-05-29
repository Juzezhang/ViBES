import os
import time as _time
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from transformers import Trainer, TrainingArguments, TrainerCallback
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoConfig
)
from transformers.modeling_utils import load_sharded_checkpoint
from datasets import load_from_disk
import torch
import numpy as np
import gc
import json
import math
from vibes.modeling_chatglm import ChatGLMForConditionalGenerationMotExpertNum2
from expert_io import is_expert1_checkpoint, load_expert1_checkpoint

def load_tokenized_dataset(dataset_path, verbose=True):
    """
    Load preprocessed CANDOR tokenized dataset
    
    Args:
        dataset_path: Path to tokenized dataset
        verbose: Whether to output detailed information
        
    Returns:
        Processed dataset
    """
    if verbose:
        print(f"Loading tokenized dataset from {dataset_path}")
    
    try:
        # Directly load tokenized dataset
        tokenized_dataset = load_from_disk(dataset_path)
        # tokenized_dataset = tokenized_dataset.select(range(10, 11))
        if verbose:
            print(f"Dataset loaded with {len(tokenized_dataset)} examples")
            
            # Check data format
            sample = tokenized_dataset[0]
            print(f"Sample keys: {list(sample.keys())}")
            print(f"Input IDs length: {len(sample['input_ids'])}")
            print(f"Labels length: {len(sample['labels'])}")
            
            # Verify dataset contains required fields
            if not all(k in sample for k in ['input_ids', 'attention_mask', 'labels']):
                print("Warning: Dataset is missing some required fields")
        
        return tokenized_dataset
        
    except Exception as e:
        print(f"Error loading tokenized dataset: {e}")
        raise

def tokenized_data_collator(batch):
    """
    Collator for already tokenized datasets
    This collator directly uses preprocessed tokens, no need to tokenize again
    Handles variable-length sequences by padding to the maximum length in the batch
    
    Args:
        batch: Data batch
        
    Returns:
        Processed batch with padded sequences
    """
    # Find the maximum length in the batch
    max_length = max([len(item["input_ids"]) for item in batch])
    
    # Initialize lists for batched data
    input_ids_list = []
    attention_mask_list = []
    labels_list = []
    modality_masks_0_list = []
    modality_masks_1_list = []
    modality_masks_2_list = []
    position_encoding_indices_list = []
    # Process each item with padding
    for item in batch:
        # Get current lengths
        curr_len = len(item["input_ids"])
        padding_len = max_length - curr_len
        
        # Pad input_ids (usually with 0 or tokenizer.pad_token_id)
        padded_input_ids = item["input_ids"] + [0] * padding_len
        input_ids_list.append(padded_input_ids)
        
        # Pad attention_mask (with 0s)
        padded_attention_mask = item["attention_mask"] + [0] * padding_len
        attention_mask_list.append(padded_attention_mask)
        
        # Pad labels (usually with -100 to ignore in loss calculation)
        padded_labels = item["labels"] + [-100] * padding_len
        labels_list.append(padded_labels)
        # if (torch.tensor(padded_labels) != -100).sum() == 0:
        #     pass
        #     print(f"Found {padded_labels.sum()} -100 labels in the batch")

        # Pad modality_masks_0 (with False)
        padded_modality_masks_0 = item["modality_masks_0"] + [False] * padding_len
        modality_masks_0_list.append(padded_modality_masks_0)

        padded_modality_masks_1 = item["modality_masks_1"] + [False] * padding_len
        modality_masks_1_list.append(padded_modality_masks_1)

        padded_modality_masks_2 = item["modality_masks_2"] + [False] * padding_len
        modality_masks_2_list.append(padded_modality_masks_2)


        position_encoding_indices = item["position_encoding_indices"] + [0] * padding_len
        position_encoding_indices_list.append(position_encoding_indices)


    modality_masks_list = [modality_masks_0_list, modality_masks_1_list, modality_masks_2_list]

    input_ids_tensor = torch.tensor(input_ids_list)
    input_ids_tensor[input_ids_tensor>=168736] -= 168736

    labels_tensor = torch.tensor(labels_list)
    labels_tensor[labels_tensor>=168736] -= 168736


    # Convert to tensors
    return {
        "input_ids": input_ids_tensor,
        "attention_mask": torch.tensor(attention_mask_list),
        "labels": labels_tensor,
        "modality_masks": torch.tensor(modality_masks_list),
        "position_encoding_indices": torch.tensor(position_encoding_indices_list),
    }

def resize_glm_vocab(model, new_size):
    """Adjust the vocabulary size of the GLM model
    
    Note: This function adjusts both the input embedding layer and the output layer
    This is the same method used in training_glm.py
    """
    # Adjust the input embedding layer
    # model.resize_token_embeddings(new_size)
    

    # old_input_layer = model.transformer.embedding.word_embeddings[0]
    # hidden_dim = old_input_layer.weight.shape[1]
    # new_input_layer = torch.nn.Linear(hidden_dim, new_size, bias=False)
    # with torch.no_grad():
    #     new_input_layer.weight.data = new_input_layer.weight.data.to(old_input_layer.weight.dtype)
    #     new_input_layer.weight[:old_input_layer.weight.shape[0], :] = old_input_layer.weight.data
    # model.transformer.embedding.word_embeddings[0] = new_input_layer

    # Adjust the input embedding layers
    for i in range(len(model.transformer.embedding.word_embeddings)):
        old_embedding = model.transformer.embedding.word_embeddings[i]
        embedding_dim = old_embedding.weight.shape[1]
        new_embedding = torch.nn.Embedding(new_size, embedding_dim, dtype=old_embedding.weight.dtype)
        
        # Copy weights (preserve existing knowledge)
        with torch.no_grad():
            # Ensure new embedding uses the same data type as the old one
            new_embedding.weight.data = new_embedding.weight.data.to(old_embedding.weight.dtype)
            new_embedding.weight[:old_embedding.weight.shape[0], :] = old_embedding.weight.data
            
        # Move the new embedding to the same device as the old one
        new_embedding = new_embedding.to(device=old_embedding.weight.device, dtype=old_embedding.weight.dtype)
        
        # Replace the embedding layer
        model.transformer.embedding.word_embeddings[i] = new_embedding


        # Adjust the output layer
        old_layer = model.transformer.output_layer[i]
        hidden_dim = old_layer.weight.shape[1]
        new_layer = torch.nn.Linear(hidden_dim, new_size, bias=False)
        
        # Copy weights (preserve existing knowledge)
        with torch.no_grad():
            # Ensure new layer uses the same data type as the old layer
            new_layer.weight.data = new_layer.weight.data.to(old_layer.weight.dtype)
            new_layer.weight[:old_layer.weight.shape[0], :] = old_layer.weight.data
            
        # Move the new layer to the same device as the old layer
        new_layer = new_layer.to(device=old_layer.weight.device, dtype=old_layer.weight.dtype)
        
        # Replace the layer
        model.transformer.output_layer[i] = new_layer
    model.config.vocab_size = new_size
    
    return model



def resize_glm_vocab_expert2(model, new_size):
    """Adjust the vocabulary size of the GLM model
    
    Note: This function adjusts both the input embedding layer and the output layer
    This is the same method used in training_glm.py
    """
    # Adjust the input embedding layer
    # model.resize_token_embeddings(new_size)
    
    # Adjust the input embedding layers
    
    old_embedding = model.transformer.embedding.word_embeddings[1]
    embedding_dim = old_embedding.weight.shape[1]
    new_embedding = torch.nn.Embedding(new_size, embedding_dim, dtype=old_embedding.weight.dtype)
    
    # Copy weights (preserve existing knowledge)
    with torch.no_grad():
        # Ensure new embedding uses the same data type as the old one
        new_embedding.weight.data = new_embedding.weight.data.to(old_embedding.weight.dtype)
        if old_embedding.weight.shape[0] < new_embedding.weight.shape[0]:
            new_embedding.weight[:old_embedding.weight.shape[0], :] = old_embedding.weight.data
        else:
            new_embedding.weight[:new_embedding.weight.shape[0], :] = old_embedding.weight.data[:new_embedding.weight.shape[0], :]
        
    # Move the new embedding to the same device as the old one
    new_embedding = new_embedding.to(device=old_embedding.weight.device, dtype=old_embedding.weight.dtype)
    
    # Replace the embedding layer
    model.transformer.embedding.word_embeddings[1] = new_embedding

    # Adjust the output layer
    old_layer = model.transformer.output_layer[1]
    hidden_dim = old_layer.weight.shape[1]
    new_layer = torch.nn.Linear(hidden_dim, new_size, bias=False)
    
    # Copy weights (preserve existing knowledge)
    with torch.no_grad():
        # Ensure new layer uses the same data type as the old layer
        new_layer.weight.data = new_layer.weight.data.to(old_layer.weight.dtype)
        if old_layer.weight.shape[0] < new_layer.weight.shape[0]:
            new_layer.weight[:old_layer.weight.shape[0], :] = old_layer.weight.data
        else:
            new_layer.weight[:new_layer.weight.shape[0], :] = old_layer.weight.data[:new_layer.weight.shape[0], :]
        
    # Move the new layer to the same device as the old layer
    new_layer = new_layer.to(device=old_layer.weight.device, dtype=old_layer.weight.dtype)
    
    # Replace the layer
    model.transformer.output_layer[1] = new_layer
    # model.config.vocab_size = new_size
    model.config.mot_settings['mot_vocab_size'] = new_size
    return model

def initialize_mot_expert_parameters(model, verbose=True, init_strategy="xavier_uniform"):
    """
    Initialize the second modality expert parameters with proper random initialization.
    Since the two experts have different dimensions, we cannot copy weights directly.
    
    Args:
        model: The ChatGLMForConditionalGenerationMot model  
        verbose: Whether to print initialization details
        init_strategy: Initialization strategy ("xavier_uniform", "xavier_normal", "kaiming_uniform", "kaiming_normal")
    """
    if verbose:
        print(f"Initializing MOT expert parameters with strategy: {init_strategy}")
        print("Note: Two experts have different dimensions, using random initialization")
    
    initialized_params = []
    
    def apply_initialization(param, param_name="", layer_type="linear"):
        """Apply the chosen initialization strategy based on parameter type"""
        with torch.no_grad():
            if layer_type == "embedding":
                # Use normal distribution initialization for embedding layers
                torch.nn.init.normal_(param, mean=0.0, std=0.02)
            elif layer_type == "layernorm":
                # Initialize LayerNorm weight to 1, bias to 0
                if len(param.shape) == 1:  # LayerNorm weight
                    torch.nn.init.ones_(param)
                else:
                    torch.nn.init.zeros_(param)
            elif layer_type == "linear":
                if init_strategy == "xavier_uniform":
                    torch.nn.init.xavier_uniform_(param)
                elif init_strategy == "xavier_normal":
                    torch.nn.init.xavier_normal_(param)
                elif init_strategy == "kaiming_uniform":
                    torch.nn.init.kaiming_uniform_(param, a=math.sqrt(5))
                elif init_strategy == "kaiming_normal":
                    torch.nn.init.kaiming_normal_(param)
                else:
                    # Default to xavier_uniform
                    torch.nn.init.xavier_uniform_(param)
            elif layer_type == "bias":
                # Bias is usually initialized to 0
                torch.nn.init.zeros_(param)
        
        if verbose:
            print(f"  Initialized {param_name} ({layer_type}) with {init_strategy}")

    # 1. Initialize the second modality's embedding layer
    if hasattr(model.transformer.embedding, 'word_embeddings'):
        if len(model.transformer.embedding.word_embeddings) > 1:
            target_emb = model.transformer.embedding.word_embeddings[1].weight
            apply_initialization(target_emb, "embedding.word_embeddings.1", "embedding")
            initialized_params.append("transformer.embedding.word_embeddings.1")

    # 2. Initialize the second expert in transformer layers
    encoder = model.transformer.encoder
    
    for layer_idx, layer in enumerate(encoder.layers):
        if verbose:
            print(f"  Processing Layer {layer_idx}...")
            
        # Input layer normalization
        if hasattr(layer, 'local_experts_input_layernorm') and len(layer.local_experts_input_layernorm) > 1:
            target_norm = layer.local_experts_input_layernorm[1].weight
            apply_initialization(target_norm, f"Layer {layer_idx}: input_layernorm.1", "layernorm")
            initialized_params.append(f"Layer {layer_idx}: local_experts_input_layernorm.1")
        
        # Post-attention layer normalization
        if hasattr(layer, 'local_experts_post_attention_layernorm') and len(layer.local_experts_post_attention_layernorm) > 1:
            target_norm = layer.local_experts_post_attention_layernorm[1].weight
            apply_initialization(target_norm, f"Layer {layer_idx}: post_attention_layernorm.1", "layernorm")
            initialized_params.append(f"Layer {layer_idx}: local_experts_post_attention_layernorm.1")
        
        # MLP layers
        if hasattr(layer, 'local_experts_mlp') and len(layer.local_experts_mlp) > 1:
            # dense_h_to_4h weights and bias
            mlp_expert = layer.local_experts_mlp[1]
            
            target_h_to_4h_weight = mlp_expert.dense_h_to_4h.weight
            apply_initialization(target_h_to_4h_weight, f"Layer {layer_idx}: mlp.dense_h_to_4h.weight.1", "linear")
            
            if mlp_expert.dense_h_to_4h.bias is not None:
                target_h_to_4h_bias = mlp_expert.dense_h_to_4h.bias
                apply_initialization(target_h_to_4h_bias, f"Layer {layer_idx}: mlp.dense_h_to_4h.bias.1", "bias")
            
            # dense_4h_to_h weights and bias
            target_4h_to_h_weight = mlp_expert.dense_4h_to_h.weight
            apply_initialization(target_4h_to_h_weight, f"Layer {layer_idx}: mlp.dense_4h_to_h.weight.1", "linear")
            
            if mlp_expert.dense_4h_to_h.bias is not None:
                target_4h_to_h_bias = mlp_expert.dense_4h_to_h.bias
                apply_initialization(target_4h_to_h_bias, f"Layer {layer_idx}: mlp.dense_4h_to_h.bias.1", "bias")
            
            initialized_params.extend([
                f"Layer {layer_idx}: local_experts_mlp.1.dense_h_to_4h",
                f"Layer {layer_idx}: local_experts_mlp.1.dense_4h_to_h"
            ])
        
        # Attention layers
        if (hasattr(layer, 'self_attention') and 
            hasattr(layer.self_attention, 'query_key_value') and
            isinstance(layer.self_attention.query_key_value, torch.nn.ModuleList) and
            len(layer.self_attention.query_key_value) > 1):
            
            target_qkv_weight = layer.self_attention.query_key_value[1].weight
            apply_initialization(target_qkv_weight, f"Layer {layer_idx}: attention.qkv.weight.1", "linear")
            
            if layer.self_attention.query_key_value[1].bias is not None:
                target_qkv_bias = layer.self_attention.query_key_value[1].bias
                apply_initialization(target_qkv_bias, f"Layer {layer_idx}: attention.qkv.bias.1", "bias")
            
            initialized_params.append(f"Layer {layer_idx}: self_attention.query_key_value.1")
        
        if (hasattr(layer, 'self_attention') and 
            hasattr(layer.self_attention, 'dense') and
            isinstance(layer.self_attention.dense, torch.nn.ModuleList) and
            len(layer.self_attention.dense) > 1):
            
            target_dense_weight = layer.self_attention.dense[1].weight
            apply_initialization(target_dense_weight, f"Layer {layer_idx}: attention.dense.weight.1", "linear")
            
            if layer.self_attention.dense[1].bias is not None:
                target_dense_bias = layer.self_attention.dense[1].bias
                apply_initialization(target_dense_bias, f"Layer {layer_idx}: attention.dense.bias.1", "bias")
            
            initialized_params.append(f"Layer {layer_idx}: self_attention.dense.1")

    # 3. Final layer normalization
    if (hasattr(encoder, 'local_experts_final_layernorm') and 
        len(encoder.local_experts_final_layernorm) > 1):
        target_norm = encoder.local_experts_final_layernorm[1].weight
        apply_initialization(target_norm, "final_layernorm.1", "layernorm")
        initialized_params.append("transformer.encoder.local_experts_final_layernorm.1")
    
    # 4. Output layer
    if (hasattr(model.transformer, 'output_layer') and 
        len(model.transformer.output_layer) > 1):
        target_output_weight = model.transformer.output_layer[1].weight
        apply_initialization(target_output_weight, "output_layer.1", "linear")
        initialized_params.append("transformer.output_layer.1")

    if verbose:
        print(f"Initialized {len(initialized_params)} expert parameters:")
        for param in initialized_params:
            print(f"  - {param}")
        print(f"MOT expert parameter initialization completed with {init_strategy}!")
    
    return model



# Add command line argument support
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train GLM model on CANDOR dataset")
    parser.add_argument("--tokenized_dataset", type=str, required=True, 
                        help="Path to the preprocessed tokenized dataset (created by preprocess_candor_dataset_w_transcript_tokenized.py)")
    parser.add_argument("--output_dir", type=str, default="./glm4voice-conversational-agent-eos-streaming-transcription-debug", 
                        help="Output directory for model checkpoints")
    parser.add_argument("--batch_size", type=int, default=8, 
                        help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=20, 
                        help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-5, 
                        help="Learning rate")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Resume training from a checkpoint path")
    parser.add_argument("--pretrained_model_path", type=str, default=None,
                        help="Pretrained model path")
    parser.add_argument("--glm_base_path", type=str, default="THUDM/glm-4-voice-9b",
                        help="GLM-4-Voice base providing the frozen Expert-0 (text/audio) weights. "
                             "Checkpoints store only the trained motion Expert-1; Expert-0 is "
                             "reconstructed from this base at load time.")
    # Add DeepSpeed local_rank argument
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Local rank for distributed training (used by DeepSpeed)")
    # Add save frequency parameter
    parser.add_argument("--save_steps", type=int, default=500,
                        help="Save a checkpoint every N steps")
    parser.add_argument("--save_total_limit", type=int, default=5,
                        help="Maximum number of checkpoints to keep")
    parser.add_argument("--save_strategy", type=str, default="steps", choices=["steps", "no"],
                        help="Checkpoint save strategy. Use 'no' for fast smoke tests.")
    parser.add_argument("--skip_final_save", action="store_true",
                        help="Skip final trainer.save_model(), useful for no-save smoke tests.")
    # Original model has 40 layers, we can use this to train the model with less layers
    parser.add_argument("--layer_num", type=int, default=5,
                        help="Number of layers in the model")
    parser.add_argument("--t1_layer_limit", type=int, default=None,
                        help="Number of Transformer-1 layers to execute (<= 40, defaults to layer_num)")
    # --- efficiency knobs ---
    # The trainable motion expert (Expert-1) is small (~0.9 GB); the frozen text/audio expert +
    # embeddings (~19 GB) dominate. ZeRO-3 would shard AND all-gather those frozen weights every
    # step (huge wasted comm -> low GPU util). The full model fits replicated on one 46 GB GPU, so
    # ZeRO-2 (params replicated, only the tiny Expert-1 grads/optimizer sharded) is much faster here.
    parser.add_argument("--zero_stage", type=int, default=2, choices=[0, 1, 2, 3],
                        help="DeepSpeed ZeRO stage (default 2; 3 only if the model does not fit replicated)")
    parser.add_argument("--no_gradient_checkpointing", action="store_true",
                        help="Disable gradient checkpointing (faster; fine when the model fits without it)")
    parser.add_argument("--max_steps", type=int, default=0,
                        help="If >0, stop after this many optimizer steps (for benchmarking)")
    parser.add_argument("--logging_steps", type=int, default=100,
                        help="Log train loss every N steps (lower = more frequent loss readouts)")
    parser.add_argument("--dataloader_num_workers", type=int, default=0,
                        help="Dataloader workers. 0 is best for smoke tests and avoids worker startup latency.")
    parser.add_argument("--max_train_samples", type=int, default=0,
                        help="If >0, train on only the first N dataset rows (for smoke tests)")
    parser.add_argument("--attn_implementation", type=str, default="flash_attention_2",
                        choices=["flash_attention_2", "eager", "sdpa"],
                        help="Attention backend for masked interleaved training")
    parser.add_argument("--vibes_config", type=str, default="./vibes",
                        help="MoME config+tokenizer dir. Default ./vibes (9B backbone). Use vibes_0.5b "
                             "(hidden 1024 / 24 layers) with --glm_base_path <ViBES-Audio> --layer_num 24 "
                             "for the 0.5B variant.")
    args = parser.parse_args()
    
    # Output training configuration information
    print(f"Training configuration:")
    print(f"  - Tokenized dataset: {args.tokenized_dataset}")
    print(f"  - Output directory: {args.output_dir}")
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - Epochs: {args.epochs}")
    print(f"  - Learning rate: {args.learning_rate}")
    print(f"  - Resume from checkpoint: {args.resume_from_checkpoint if args.resume_from_checkpoint else 'No'}")
    print(f"  - Pretrained model path: {args.pretrained_model_path if args.pretrained_model_path else 'No'}")
    print(f"  - Local rank: {args.local_rank}")
    print(f"  - Save checkpoint every {args.save_steps} steps")
    print(f"  - Keep up to {args.save_total_limit} checkpoints")
    print(f"  - Attention backend: {args.attn_implementation}")
    print("  - Training attention mode: masked interleaved (override with VIBES_TRAIN_ATTENTION_MODE=legacy_cache)")

    # Clear memory
    gc.collect()
    torch.cuda.empty_cache()
    print("Memory cleared")

    # Create DeepSpeed configuration
    zero_opt = {
        "stage": args.zero_stage,
        "allgather_partitions": True,
        "allgather_bucket_size": 5e8,
        "reduce_scatter": True,
        "contiguous_gradients": True,
        "overlap_comm": True,
        "reduce_bucket_size": "auto",
    }
    if args.zero_stage == 3:
        # Stage-3 also shards parameters; these keys only apply then.
        zero_opt.update({
            "sub_group_size": 1e9,
            "stage3_prefetch_bucket_size": "auto",
            "stage3_param_persistence_threshold": "auto",
            "stage3_max_live_parameters": 1e9,
            "stage3_max_reuse_distance": 1e9,
            "stage3_gather_16bit_weights_on_model_save": True,
        })
    ds_config = {
        "train_micro_batch_size_per_gpu": "auto",
        "zero_allow_untested_optimizer": True,
        "bf16": {
            "enabled": "auto"
        },
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": "auto",
                "betas": "auto",
                "eps": "auto",
                "weight_decay": "auto",
                # Use torch's AdamW instead of DeepSpeed's FusedAdam: FusedAdam JIT-compiles a CUDA
                # op (needs a working `ninja` executable + nvcc on the compute node), which the
                # relocated conda env can't do reliably. The trainable expert is tiny (~0.9 GB), so
                # the fused-vs-torch optimizer speed difference is negligible.
                "torch_adam": True
            }
        },
        "zero_optimization": zero_opt
    }

    # Save configuration to file
    with open('ds_config.json', 'w') as f:
        json.dump(ds_config, f, indent=4)

    print(f"Created DeepSpeed configuration (ZeRO Stage-{args.zero_stage})")

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.vibes_config,
        trust_remote_code=True
    )
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Original tokenizer vocab size: {tokenizer.vocab_size}")
    print(f"Tokenizer eos_token: {tokenizer.eos_token}")

    # Add special tokens for face, upper body, and lower body modalities
    # Load tokenizer
    print("\n=== Step 1: Load Tokenizer ===")
    tokenizer_path = args.vibes_config
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    ## tokenizer.convert_tokens_to_ids("<|face_10|>")

    # Load model in a DeepSpeed-friendly way
    print("Loading model in BF16 format...")
    # When using DeepSpeed, initial model should be on CPU
    device_map = "cpu" 

    # Load model using AutoModel with trust_remote_code
    # The model directory should contain modeling_chatglm.py with the dual-transformer class
    if args.pretrained_model_path:
        print("\n=== Step 2: Load Model From Pretrained Checkpoint ===")
        config = AutoConfig.from_pretrained(tokenizer_path, trust_remote_code=True)
        config.num_layers = args.layer_num
        config.torch_dtype = torch.bfloat16
        config._attn_implementation = args.attn_implementation
        language_model = ChatGLMForConditionalGenerationMotExpertNum2(config, empty_init=True, device="cpu")
        language_model = language_model.to(dtype=torch.bfloat16)
        if is_expert1_checkpoint(args.pretrained_model_path):
            print(f"Detected Expert-1-only checkpoint; reconstructing Expert-0 from {args.glm_base_path}")
            _, missing, unexpected = load_expert1_checkpoint(language_model, args.pretrained_model_path, args.glm_base_path)
            unexpected = [k for k in unexpected if "rotary_pos_emb" not in k]
            if unexpected:
                print(f"Warning: {len(unexpected)} unexpected keys, e.g. {unexpected[:3]}")
        else:
            load_sharded_checkpoint(language_model, args.pretrained_model_path)
    else:
        # Load model configuration and create model instance
        print("\n=== Step 2: Load Base Model ===")
        config = AutoConfig.from_pretrained(tokenizer_path, trust_remote_code=True)
        config.num_layers = args.layer_num
        config.torch_dtype = torch.bfloat16
        config._attn_implementation = args.attn_implementation
        language_model = ChatGLMForConditionalGenerationMotExpertNum2(config, empty_init=True, device="cpu")
        language_model = language_model.to(dtype=torch.bfloat16)


        # if args.layer_num == 40:
        #     language_model = ChatGLMForConditionalGenerationMotExpertNum2.from_pretrained(
        #         "/path/to/model_files/glm-4-voice-9b-mot",
        #         torch_dtype=torch.bfloat16,
        #         attn_implementation="flash_attention_2",
        #     )
        # elif args.layer_num == 5:
        #     language_model = ChatGLMForConditionalGenerationMotExpertNum2.from_pretrained(
        #         "/path/to/model_files/glm-4-voice-9b-mot_layernum_5_modalitynum_3",
        #         torch_dtype=torch.bfloat16,
        #         attn_implementation="flash_attention_2",
        #     )
        # else:
        #     raise ValueError(f"Invalid layer number: {args.layer_num}")
        

        # language_model = AutoModel.from_pretrained(
        #     "/path/to/model_files/glm-4-voice-9b-mot_layernum_5_modalitynum_6",
        #     trust_remote_code=True,
        #     torch_dtype=torch.bfloat16,
        # )

        language_model_original = AutoModel.from_pretrained(
            args.glm_base_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )

        state_dict = language_model_original.state_dict()
        new_state_dict = dict()
        for name, param in state_dict.items():
            if "transformer.embedding" in name:
                print(name)
            if "transformer.output_layer" in name:
                print(name)
            name = name.replace("transformer.embedding.word_embeddings", "transformer.embedding.word_embeddings.0")
            name = name.replace("input_layernorm", "local_experts_input_layernorm.0")
            name = name.replace("post_attention_layernorm", "local_experts_post_attention_layernorm.0")
            name = name.replace("mlp.dense_h_to_4h", "local_experts_mlp.0.dense_h_to_4h")
            name = name.replace("mlp.dense_4h_to_h", "local_experts_mlp.0.dense_4h_to_h")
            name = name.replace("self_attention.query_key_value", "self_attention.query_key_value.0")
            name = name.replace("self_attention.dense", "self_attention.dense.0")
            name = name.replace("transformer.encoder.final_layernorm", "transformer.encoder.local_experts_final_layernorm.0")
            name = name.replace("transformer.output_layer", "transformer.output_layer.0")
            new_state_dict[name] = param

        language_model.load_state_dict(new_state_dict, strict=False)

        # Initialize MOT expert parameters
        language_model = initialize_mot_expert_parameters(language_model, init_strategy="xavier_uniform")

    # Check model device before resizing
    model_device = next(language_model.parameters()).device
    print(f"Model is initially on device: {model_device}")

    # Configure T1 layer usage limit (defaults to layer_num when not provided)
    t1_layer_limit = args.t1_layer_limit if args.t1_layer_limit is not None else args.layer_num
    total_t1_layers = len(getattr(language_model.transformer.encoder, "layers", []))
    if total_t1_layers:
        t1_layer_limit = max(1, min(t1_layer_limit, total_t1_layers))
    language_model.config.t1_layer_limit = t1_layer_limit
    if hasattr(language_model, "t1_layer_limit"):
        language_model.t1_layer_limit = t1_layer_limit
    if hasattr(language_model, "transformer") and hasattr(language_model.transformer, "set_t1_layer_limit"):
        language_model.transformer.set_t1_layer_limit(t1_layer_limit)
    print(f"Transformer-1 layer limit set to {t1_layer_limit}")
    
    # DeepSpeed requires model to start on CPU for proper initialization with ZeRO stage 3
    if model_device.type != 'cpu':
        print("Moving model to CPU for DeepSpeed compatibility...")
        language_model.to('cpu')
        model_device = next(language_model.parameters()).device
        print(f"Model moved to: {model_device}")

    # Adjust model vocabulary size and update config
    language_model = resize_glm_vocab_expert2(language_model, 1282) # 512+256+256+256 + 2
    # print(f"Original modality2 model vocab_size: {len(tokenizer)}")
    # print(f"Resized modality2 model vocab_size (config): {language_model.config.vocab_size}")

    # freeze all parameters
    for param in language_model.parameters():
        param.requires_grad = False

    for name, param in language_model.named_parameters():
        if any(x in name for x in [
            'transformer.embedding.word_embeddings.1',
            'local_experts_input_layernorm.1',
            'local_experts_post_attention_layernorm.1',
            'local_experts_mlp.1.dense_h_to_4h',
            'local_experts_mlp.1.dense_4h_to_h',
            'self_attention.query_key_value.1',
            'self_attention.dense.1',
            'transformer.encoder.local_experts_final_layernorm.1',
            'transformer.output_layer.1'
        ]):
            param.requires_grad = True
            print(f"Unfrozen parameter: {name}")

    # Output model parameter information
    trainable_params = sum(p.numel() for p in language_model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in language_model.parameters())
    # print(f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / all_params:.2f}% of {all_params:,} total parameters)")
    print(f"Trainable parameters: {trainable_params / 1e9:.2f}B ({100 * trainable_params / all_params:.2f}%)")
    print(f"Total parameters: {all_params / 1e9:.2f}B")

    # Load tokenized dataset
    tokenized_dataset_path = args.tokenized_dataset
    if not os.path.exists(tokenized_dataset_path):
        raise FileNotFoundError(f"Tokenized dataset not found: {tokenized_dataset_path}")

    print(f"Using tokenized dataset: {tokenized_dataset_path}")
    dataset = load_tokenized_dataset(tokenized_dataset_path)
    if args.max_train_samples and args.max_train_samples > 0:
        n = min(args.max_train_samples, len(dataset))
        dataset = dataset.select(range(n))
        print(f"Selected first {n} samples for smoke training")
    data_collator = tokenized_data_collator
    print("Using tokenized data collator")

    # Set training parameters
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=1,
        gradient_checkpointing=not args.no_gradient_checkpointing,
        # evaluation_strategy="no",
        max_steps=args.max_steps if args.max_steps and args.max_steps > 0 else -1,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        save_total_limit=args.save_total_limit,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        warmup_steps=10,
        weight_decay=0.01,
        fp16=False,
        bf16=True,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_pin_memory=True,
        deepspeed="ds_config.json",
        # deepspeed_port=29505,  # change this port to avoid conflict
    )

    # Initialize Trainer
    # Use ExpertSaveTrainer so each checkpoint stores ONLY the trained motion expert
    # (Expert-1) instead of the full ~20GB model; Expert-0 (frozen GLM-4-Voice base) is
    # reconstructed at load time via expert_io.load_expert1_checkpoint.
    from expert_io import make_expert_save_trainer
    ExpertSaveTrainer = make_expert_save_trainer(base=os.path.basename(args.glm_base_path.rstrip("/")))
    trainer = ExpertSaveTrainer(
        model=language_model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Benchmark instrumentation: when VIBES_BENCH_TIMING is set, record per-step wall time so a
    # downstream harness can compute steady-state throughput (warmup excluded). No-op otherwise.
    if os.environ.get("VIBES_BENCH_TIMING"):
        class _BenchTimingCallback(TrainerCallback):
            def __init__(self, path, world_size, micro_bs):
                self.path, self.world, self.bs = path, world_size, micro_bs
                self._t = None
                self.f = open(path, "w"); self.f.write("step,step_time_s,samples\n"); self.f.flush()
            def on_step_begin(self, a, s, c, **kw):
                if torch.cuda.is_available(): torch.cuda.synchronize()
                self._t = _time.perf_counter()
            def on_step_end(self, a, s, c, **kw):
                if self._t is None: return
                if torch.cuda.is_available(): torch.cuda.synchronize()
                dt = _time.perf_counter() - self._t
                self.f.write(f"{s.global_step},{dt:.5f},{self.world * self.bs}\n"); self.f.flush()
        ws = int(os.environ.get("WORLD_SIZE", "1"))
        if int(os.environ.get("LOCAL_RANK", "0")) == 0:
            trainer.add_callback(_BenchTimingCallback(
                os.environ["VIBES_BENCH_TIMING"], ws, args.batch_size))

    # Start training
    if args.resume_from_checkpoint:
        print(f"Resuming training from checkpoint: {args.resume_from_checkpoint}")
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    else:
        trainer.train()

    # Save model
    if args.skip_final_save:
        print("Training completed, final model save skipped by --skip_final_save")
    else:
        final_model_path = os.path.join(args.output_dir, "final_model")
        trainer.save_model(final_model_path)
        print(f"Training completed, model saved to: {final_model_path}")
