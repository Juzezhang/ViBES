import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login

# Get HuggingFace token from environment variable
hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    print(f"Authenticating with HuggingFace using token...")
    login(token=hf_token)
    print("✓ Authentication complete")
else:
    print("No HF_TOKEN found, proceeding without authentication")

# Set your model path to Mistral instead of Llama 3
model_path = "mistralai/Mistral-7B-Instruct-v0.2"  # Open source alternative

print(f"Testing model loading from: {model_path}")

try:
    # Try loading tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=True,
        trust_remote_code=True
    )
    print("✓ Tokenizer loaded successfully")
    
    # Try loading model (just a small portion to verify)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        load_in_8bit=True  # Use 8-bit quantization to load faster
    )
    print("✓ Model loaded successfully")
    
    # Test a simple prompt
    prompt = "<s>[INST] Hello, how are you? [/INST]"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    
    # Generate a small response to verify functionality
    output = model.generate(input_ids, max_new_tokens=20)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"Test response: {response}")
    
    print("✓ Model generates responses correctly")
    
except Exception as e:
    print(f"Error loading model: {str(e)}")
    print(f"Type: {type(e)}")
    import traceback
    traceback.print_exc() 