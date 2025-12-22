import random
import pickle
import os
import numpy as np
import codecs as cs
import torch
from torch.utils import data
from os.path import join as pjoin
from rich.progress import track
import json
import spacy
import pandas as pd
import math
import textgrid as tg
from .utils.split_transcript import split_and_merge_sentences
import librosa
from numpy.lib import stride_tricks
from pathlib import Path
from conver_agent.utils.token_utils import prepare_multimodal_tokens_for_lm, combine_audio_face_tokens

class MixedDatasetLLM(data.Dataset):
    def __init__(
        self,
        dataset_configs,
        split,
        args,
        tiny=False,
        debug=False,
        stage='lm_pretrain',
        task_path=None,
        **kwargs,
    ):
        """
        Simplified dataset class that only loads and returns preprocessed interleaved text data.
        
        Parameters:
        - dataset_configs: List of configurations for different datasets.
        - split: Specifies the data split (train/val/test).
        - args: Additional arguments.
        - tiny: Whether to use a small subset for debugging.
        - debug: If True, enables debug mode.
        - stage: Specifies the training stage.
        - task_path: Path to the task instructions file.
        """
        # Set max data size based on debug mode
        self.debug = debug
        if tiny or debug:
            self.maxdata = 10
        else:
            self.maxdata = float('inf')  # No limit, use all data

        self.stage = stage
        self.args = args
        self.split = split
        
        # Dictionaries to store data and metadata
        self.data_dict = {}
        self.metadata = []

        # Load each dataset based on its configuration
        for config in dataset_configs:
            dataset_name = config.get("name")
            if dataset_name == "LibriSpeech":
                self._load_librispeech(config)
                self.data_dict.update(self.data_dict_librispeech)
                self.metadata.extend(self.metadata_librispeech)
                print(f"Loaded LibriSpeech dataset with {len(self.metadata_librispeech)} samples")
            elif dataset_name == "CANDOR":
                # Skip test split
                if self.split == "test":
                    print(f"Skipping CANDOR dataset for test split")
                    self.data_dict_candor = {}
                    self.metadata_candor = []
                    continue
                self._load_candor(config)
                self.data_dict.update(self.data_dict_candor)
                self.metadata.extend(self.metadata_candor)
                print(f"Loaded CANDOR dataset with {len(self.metadata_candor)} samples")
            else:
                raise NotImplementedError(f"Unknown dataset name {dataset_name}")


    def _load_librispeech(self, config):
        """
        Load the LibriSpeech dataset.
        
        Parameters:
        - config: Configuration dictionary for LibriSpeech.
        """
        data_root = self.args["LIBRISPEECH"].ROOT
        code_path_audio = config.get("code_path_audio")
        token_root_librispeech = pjoin(data_root, code_path_audio)
        instructions_file = config.get("instructions_file")
        used_trainset = config.get("used_trainset")

        # Determine the instructions path
        if self.task_path:
            instructions_path = self.task_path
        elif self.stage == 'lm_pretrain':
            instructions_path = pjoin(data_root, instructions_file)
        elif self.stage in ['lm_instruct', 'lm_causal_instruct']:
            instructions_path = pjoin(data_root, instructions_file)
        else:
            raise NotImplementedError(f"stage {self.stage} not implemented")

        # Load instructions and tasks
        self.instructions_librispeech = json.load(open(instructions_path, 'r'))
        self.tasks_librispeech = []
        for task in self.instructions_librispeech.keys():
            for subtask in self.instructions_librispeech[task].keys():
                self.tasks_librispeech.append(self.instructions_librispeech[task][subtask])

        # Preload all text data
        self.text_dict_librispeech = {}
        self.token_dict_librispeech = {}
        self.metadata_librispeech_original = []

        # Load data for each specified trainset
        for trainset in used_trainset:
            # Load text data from pickle files
            with open(pjoin(token_root_librispeech, trainset + '-texts.pkl'), 'rb') as f:
                self.text_dict_librispeech[trainset] = pickle.load(f)

            with open(pjoin(token_root_librispeech, trainset + '.pkl'), 'rb') as f:
                self.token_dict_librispeech[trainset] = pickle.load(f)

            # Collect metadata for each sample in the dataset
            for speaker_id in self.token_dict_librispeech[trainset].keys():
                for chapter_id in self.token_dict_librispeech[trainset][speaker_id].keys():
                    for utterance_id in self.token_dict_librispeech[trainset][speaker_id][chapter_id].keys():
                        for part_id in self.token_dict_librispeech[trainset][speaker_id][chapter_id][utterance_id].keys():
                            self.metadata_librispeech_original.append((trainset, speaker_id, chapter_id, utterance_id, part_id))

            # Prepare metadata and data dictionary for LibriSpeech
            enumerator_librispeech = enumerate(self.metadata_librispeech_original)
            self.metadata_librispeech = []
            self.data_dict_librispeech = {}
            for i, name in enumerator_librispeech:
                if len(self.metadata_librispeech) > self.maxdata:
                    break
                try:
                    # Retrieve metadata for the current item
                    trainset, speaker_id, chapter_id, utterance_id, part_id = name
                    # Get the corresponding text data
                    text = self.text_dict_librispeech[trainset][speaker_id][chapter_id][utterance_id][part_id]

                    # Create a unique save name for each data entry
                    for tasks in self.tasks_librispeech:
                        save_name = 'librispeech' + '_' + trainset + '_' + speaker_id + '_' + chapter_id + '_' + utterance_id + '_' + part_id + '_' + tasks['class']
                        # Store the data entry
                        self.data_dict_librispeech[save_name] = {
                            'text': text,
                            'tasks': tasks,
                        }
                        self.metadata_librispeech.append(save_name)

                except:
                    pass

    def _load_candor(self, config):
        """
        Load the CANDOR dataset from preprocessed files.
        
        Parameters:
        - config: Configuration dictionary for CANDOR.
        """
        # Skip loading test split
        if self.split == "test":
            print(f"Skipping CANDOR dataset for test split")
            self.data_dict_candor = {}
            self.metadata_candor = []
            return
        
        data_root = self.args["CANDOR"].ROOT if "CANDOR" in self.args else "/simurgh/u/juze/datasets/CANDOR"
        
        print(f"Loading preprocessed CANDOR dataset...")
        
        # Get configuration parameters
        preprocessed_dir = config.get("preprocessed_dir", "./processed_candor_dataset")
        dataset_file = config.get("dataset_file", "candor_dataset.jsonl")
        
        # Ensure path is absolute
        if not os.path.isabs(preprocessed_dir):
            preprocessed_dir = os.path.join(data_root, preprocessed_dir)
        
        # Path to preprocessed dataset file
        dataset_path = os.path.join(preprocessed_dir, dataset_file)
        
        # Check if preprocessed file exists
        if not os.path.exists(dataset_path):
            print(f"Preprocessed dataset file not found: {dataset_path}")
            print("Please run the preprocessing script first: python scripts/preprocess_candor_dataset.py")
            self.data_dict_candor = {}
            self.metadata_candor = []
            return
        
        try:
            # Use Hugging Face datasets to load the dataset
            from datasets import load_dataset
            
            print(f"Loading dataset from {dataset_path}")
            
            # Load the dataset
            candor_dataset = load_dataset('json', data_files=dataset_path)['train']
            print(f"Loaded {len(candor_dataset)} conversation sequences")
            
            # Create data dictionary
            self.data_dict_candor = {}
            self.metadata_candor = []
            
            # Process each conversation entry
            for idx, entry in enumerate(candor_dataset):
                if self.debug and len(self.metadata_candor) > self.maxdata:
                    break
                    
                # Use the whole conversation as one sample
                sample_id = f"candor_{entry['conv_id']}_{idx}"
                
                # Store required data - only keep text and basic metadata
                self.data_dict_candor[sample_id] = {
                    'text': entry['text'],  # Already in ABABABAB format
                    'tasks': {
                        'class': 'conversation',
                        'task': 'audio_face_generation',
                        'instruction': 'Generate appropriate facial expressions for the given audio.',
                        'input_modality': 'audio',
                        'output_modality': 'face'
                    },
                    'conv_id': entry['conv_id'],
                }
                
                self.metadata_candor.append(sample_id)
            
            print(f"Processed {len(self.metadata_candor)} CANDOR samples")
        
        except Exception as e:
            print(f"Error loading preprocessed CANDOR dataset: {e}")
            import traceback
            traceback.print_exc()
            self.data_dict_candor = {}
            self.metadata_candor = []

    def __len__(self):
        """
        Returns the number of samples in the dataset.
        """
        return len(self.metadata)

    def __getitem__(self, ind):
        """
        Retrieves a sample from the dataset.

        Parameters:
        - ind: Index of the sample to retrieve.

        Returns:
        - dict: Sample data containing only the text field.
        """
        key = self.metadata[ind]
        # Get the sample data
        data = self.data_dict[key]
        
        # Only return text and basic info, not audio or face_token
        return {
            'text': data['text'],
            'name': key
        }