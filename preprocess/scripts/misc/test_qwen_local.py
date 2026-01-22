import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Update to match the same model as in the config file
model_path = "model_files/qwen2-1.5b-instruct"

print(f"Testing model loading from local path: {model_path}")

try:
    # Try loading tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=True,
        trust_remote_code=True
    )
    print("✓ Tokenizer loaded successfully")
    
    # Try to use quantization if available
    try:
        from transformers import BitsAndBytesConfig
        
        # Configure quantization
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )
        
        # Load the model with quantization
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            trust_remote_code=True,
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16
        )
        print("✓ Model loaded successfully with 4-bit quantization")
    except (ImportError, ModuleNotFoundError):
        print("⚠️ bitsandbytes not found, loading model without quantization")
        # Fall back to loading without quantization
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        )
        print("✓ Model loaded successfully without quantization")
    
    # Test with a simple prompt
    prompt = "<|im_start|>user\nGenerate motion for a person who is waving goodbye enthusiastically<|im_end|>\n<|im_start|>assistant\n"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    
    # Generate a response
    output = model.generate(
        input_ids, 
        max_new_tokens=50,
        temperature=0.7,
        top_p=0.9
    )
    response = tokenizer.decode(output[0], skip_special_tokens=False)
    print(f"Test response: {response}")
    
    print("✓ Model generates responses correctly from local path")
    
except Exception as e:
    print(f"Error loading model: {str(e)}")
    print(f"Type: {type(e)}")
    import traceback
    traceback.print_exc() 