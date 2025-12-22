import numpy as np
import torch
import os 
from os.path import join as pjoin
from .mixed_dataset.utils.word_vectorizer import WordVectorizer
from .mixed_dataset.scripts.motion_process import (process_file, recover_from_ric)
from . import BASEDataModule
from .mixed_dataset import MixedDatasetVQ, MixedDatasetCB, MixedDatasetLLM, MixedDatasetVQArtalk
from .mixed_dataset import FaceVQDataset, UpperVQDataset, LowerVQDataset, GlobalVQDataset
from .utils import conversation_collate, huggingface_dataset_collate
from datasets import load_dataset

class MixedDataModule(BASEDataModule):
    def __init__(self, cfg, **kwargs):

        super().__init__(collate_fn=conversation_collate)
        self.cfg = cfg
        self.save_hyperparameters(logger=False)
        # Basic info of the dataset
        cfg.DATASET.JOINT_TYPE = 'smplx'
        self.njoints = 55
        dataset_configs = cfg.DATASET.datasets
        dataset_configs_test = cfg.DATASET.datasets_test
        # # Path to the dataset
        self.hparams.args = cfg.DATASET
        self.hparams.dataset_configs=dataset_configs
        self.hparams.dataset_configs_test=dataset_configs_test
        self.hparams.debug = cfg.DEBUG
        self.hparams.stage = cfg.TRAIN.STAGE
        self.hparams.audio_down = cfg.model.params.lm.params.audio_down_sampling
        # self.hparams.w_vectorizer = WordVectorizer(cfg.DATASET.WORD_VERTILIZER_PATH, "our_vab")
        self.hparams.motion_representation = cfg.DATASET.motion_representation
        self.hparams.smpl_path = cfg.DATASET.SMPL_PATH
        self.hparams.njoints = 55
            
        # Select dataset class based on stage
        if cfg.TRAIN.STAGE == "vae" or cfg.TRAIN.STAGE == "vqvae":
            # Use FaceVQDataset by default for all VAE/VQ stages
            if cfg.Selected_part == 'upper':
                self.Dataset = UpperVQDataset
                self.DatasetEval = UpperVQDataset
            elif cfg.Selected_part == 'lower' or cfg.Selected_part == 'lower_54' or cfg.Selected_part == 'lower_global':
                self.Dataset = LowerVQDataset
                self.DatasetEval = LowerVQDataset
            elif cfg.Selected_part == 'face':
                self.Dataset = FaceVQDataset
                self.DatasetEval = FaceVQDataset
            elif cfg.Selected_part == 'compositional':
                self.Dataset = MixedDatasetVQ
                self.DatasetEval = MixedDatasetVQ
            elif cfg.Selected_part == 'global':
                self.Dataset = GlobalVQDataset
                self.DatasetEval = GlobalVQDataset
        elif cfg.TRAIN.STAGE == "token":
            self.Dataset = MixedDatasetVQ
            self.DatasetEval = MixedDatasetVQ
        # elif cfg.TRAIN.STAGE == "token_artalk":
            # self.Dataset = MixedDatasetVQArtalk
            # self.DatasetEval = MixedDatasetVQArtalk
        elif 'lm' in cfg.TRAIN.STAGE:
            # Instead of using MixedDatasetLLM class, directly use HuggingFace datasets
            # Get configuration for CANDOR dataset
            for config in dataset_configs:
                if config.get("name") == "CANDOR":
                    candor_config = config
                    break
            else:
                raise ValueError("CANDOR dataset configuration not found in dataset_configs")
            
            # Get dataset paths
            data_root = cfg.DATASET.CANDOR.ROOT if hasattr(cfg.DATASET, "CANDOR") else "/simurgh/u/juze/datasets/CANDOR"
            preprocessed_dir = candor_config.get("preprocessed_dir", "processed_candor_dataset")
            dataset_file = candor_config.get("dataset_file", "candor_dataset.jsonl")
            
            # Ensure path is absolute
            if not os.path.isabs(preprocessed_dir):
                preprocessed_dir = os.path.join(data_root, preprocessed_dir)
            
            # Path to preprocessed dataset file
            dataset_path = os.path.join(preprocessed_dir, dataset_file)
            
            # Check if preprocessed file exists
            if not os.path.exists(dataset_path):
                raise FileNotFoundError(f"Preprocessed dataset file not found: {dataset_path}. Please run the preprocessing script first.")
            
            print(f"Loading dataset directly from {dataset_path}")
            # Directly load dataset using HuggingFace datasets
            dataset = load_dataset('json', data_files=dataset_path)['train']
            print(f"Loaded {len(dataset)} conversation sequences")
            
            # Override train_dataset and other dataset properties to directly use the loaded dataset
            self._train_dataset = dataset
            self._val_dataset = dataset
            self._test_dataset = dataset
            
            # Keep the classes for compatibility, but mark them so they aren't used
            self.Dataset = None 
            self.DatasetEval = None
            
            # Set the appropriate collate_fn for HuggingFace datasets
            self.dataloader_options = {"collate_fn": huggingface_dataset_collate}
            
            # Override the dataset getter methods
            def train_dataset_getter(self):
                return self._train_dataset
                
            def val_dataset_getter(self):
                return self._val_dataset
                
            def test_dataset_getter(self):
                return self._test_dataset
            
            # Replace the property getters
            MixedDataModule.train_dataset = property(train_dataset_getter)
            MixedDataModule.val_dataset = property(val_dataset_getter) 
            MixedDataModule.test_dataset = property(test_dataset_getter)
        else:
            raise RuntimeError("Haven't setup this code!")

        # # Get additional info of the dataset
        # self._sample_set = self.get_sample_set(overrides={"split": "test", "tiny": True})


    def feats2joints(self, features):
        mean = torch.tensor(self.hparams.mean).to(features)
        std = torch.tensor(self.hparams.std).to(features)
        features = features * std + mean
        return recover_from_ric(features, self.njoints)

    def joints2feats(self, features):
        example_data = np.load(os.path.join(self.hparams.data_root, 'joints', '000021.npy'))
        example_data = example_data.reshape(len(example_data), -1, 3)
        example_data = torch.from_numpy(example_data)
        features = process_file(features, self.njoints, example_data, 't2m')[0]
        return features

    def normalize(self, features):
        mean = torch.tensor(self.hparams.mean).to(features)
        std = torch.tensor(self.hparams.std).to(features)
        features = (features - mean) / std
        return features

    def denormalize(self, features):
        mean = torch.tensor(self.hparams.mean).to(features)
        std = torch.tensor(self.hparams.std).to(features)
        features = features * std + mean
        return features

    def renorm4t2m(self, features):
        # renorm to t2m norms for using t2m evaluators
        ori_mean = torch.tensor(self.hparams.mean).to(features)
        ori_std = torch.tensor(self.hparams.std).to(features)
        eval_mean = torch.tensor(self.hparams.mean_eval).to(features)
        eval_std = torch.tensor(self.hparams.std_eval).to(features)
        features = features * ori_std + ori_mean
        features = (features - eval_mean) / eval_std
        return features

    def mm_mode(self, mm_on=True):
        if mm_on:
            self.is_mm = True
            self.name_list = self.test_dataset.name_list
            self.mm_list = np.random.choice(self.name_list,
                                            self.cfg.METRIC.MM_NUM_SAMPLES,
                                            replace=False)
            self.test_dataset.name_list = self.mm_list
        else:
            self.is_mm = False
            self.test_dataset.name_list = self.name_list
