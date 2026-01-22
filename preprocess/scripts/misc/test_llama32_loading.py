import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import AutoConfig

# Set your model path to the local Llama 3.2 path
model_path = "./model_files/Llama-3.2-3B-Instruct"

print(f"Testing model loading from: {model_path}")

try:
    # Try loading tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=True,
        trust_remote_code=True
    )
    print("✓ Tokenizer loaded successfully")
    
    # First load the config
    config = AutoConfig.from_pretrained(model_path)
    
    # Modify the rope_scaling to match expected format
    config.rope_scaling = {"type": "linear", "factor": 32.0}
    print("✓ Modified config with proper rope_scaling format")
    
    # Load the model with the modified config
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        load_in_8bit=True  # Use 8-bit quantization to save memory
    )
    print("✓ Model loaded successfully")
    
    # Test a simple prompt
    prompt = "<s>[INST] Hello, how are you? [/INST]"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    
    # Generate a small response to verify functionality
    output = model.generate(
        input_ids, 
        max_new_tokens=20,
        temperature=0.7,
        top_p=0.9
    )
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"Test response: {response}")
    
    print("✓ Model generates responses correctly")
    
except Exception as e:
    print(f"Error loading model: {str(e)}")
    print(f"Type: {type(e)}")
    import traceback
    traceback.print_exc() 