#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Example script for loading the processed CANDOR dataset with system template and chunking.
This demonstrates how to use the conversation_collate and add_system_template_and_chunk 
functions from the utils module.

Usage:
    python scripts/example_candor_dataloader.py --dataset_path ./processed_candor_dataset
"""

import os
import argparse
from datasets import load_dataset
from torch.utils.data import DataLoader
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import utilities from the project
from conver_agent.data.utils import conversation_collate, add_system_template_and_chunk

def main():
    parser = argparse.ArgumentParser(description="Load and process CANDOR dataset with system template")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the processed CANDOR dataset")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for the dataloader")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle the dataset")
    parser.add_argument("--max_seq_length", type=int, default=1024, help="Maximum sequence length for chunking")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for the dataloader")
    parser.add_argument("--sample_count", type=int, default=2, help="Number of samples to print")
    
    args = parser.parse_args()
    
    print(f"Loading dataset from {args.dataset_path}")
    
    # Check if the path is a directory or a file
    if os.path.isdir(args.dataset_path):
        # Look for jsonl file
        jsonl_path = os.path.join(args.dataset_path, "candor_dataset.jsonl")
        if os.path.exists(jsonl_path):
            dataset_path = jsonl_path
        else:
            # Try with json
            json_path = os.path.join(args.dataset_path, "candor_dataset.json")
            if os.path.exists(json_path):
                dataset_path = json_path
            else:
                raise ValueError(f"Could not find candor_dataset.jsonl or candor_dataset.json in {args.dataset_path}")
    else:
        dataset_path = args.dataset_path
    
    # Load the dataset
    candor_dataset = load_dataset("json", data_files=dataset_path)
    
    # Print dataset info
    print(f"Dataset loaded with {len(candor_dataset['train'])} examples")
    print(f"Dataset features: {candor_dataset['train'].features}")
    
    # Create a dataloader
    dataloader = DataLoader(
        candor_dataset["train"],
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        num_workers=args.num_workers,
        collate_fn=conversation_collate
    )
    
    # Print some samples
    print(f"\nPrinting {args.sample_count} samples with system template and chunking:")
    for i, batch in enumerate(dataloader):
        if i >= args.sample_count:
            break
            
        print(f"\n--- Sample {i+1} ---")
        for j, text in enumerate(batch["text"]):
            print(f"\nExample {j+1} (length: {len(text)} chars):")
            print("-" * 40)
            # Print the first 500 characters and last 200 characters
            if len(text) > 700:
                print(text[:500])
                print("...")
                print(text[-200:])
            else:
                print(text)
            print("-" * 40)
    
    # Example of direct usage of the add_system_template_and_chunk function
    print("\nExample of direct usage of add_system_template_and_chunk:")
    print("-" * 60)
    
    # Get a sample from the dataset
    sample = candor_dataset["train"][0]["text"]
    
    # Apply system template and chunking
    formatted_sample = add_system_template_and_chunk(sample, max_seq_length=args.max_seq_length)
    
    # Print the result
    print(f"Original sample length: {len(sample)} chars")
    print(f"Formatted sample length: {len(formatted_sample)} chars")
    print("\nFormatted sample (first 500 chars):")
    print(formatted_sample[:500])
    if len(formatted_sample) > 700:
        print("...")
        print(formatted_sample[-200:])
    
    print("\nDone!")

if __name__ == "__main__":
    main() 