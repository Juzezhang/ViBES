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
from omegaconf import OmegaConf

class MixedDataModule(BASEDataModule):
    def __init__(self, cfg, **kwargs):

        super().__init__(collate_fn=conversation_collate)
        self.cfg = cfg
        self.save_hyperparameters(logger=False)
        # Basic info of the dataset
        cfg.DATASET.JOINT_TYPE = 'smplx'
        self.njoints = 55
        dataset_configs = self._apply_dataset_defaults(cfg.DATASET.datasets)
        dataset_configs_test = self._apply_dataset_defaults(
            OmegaConf.select(cfg, "DATASET.datasets_test")
        )
        if dataset_configs_test is None:
            dataset_configs_test = dataset_configs
        # # Path to the dataset
        self.hparams.args = cfg.DATASET
        self.hparams.dataset_configs=dataset_configs
        self.hparams.dataset_configs_test=dataset_configs_test
        self.hparams.debug = cfg.DEBUG
        self.hparams.stage = cfg.TRAIN.STAGE
        self.hparams.selected_part = cfg.Selected_part
        # Force-disable all dataset caching.
        self.hparams.use_cache = False
        self.hparams.save_cache = False
        audio_down = OmegaConf.select(cfg, "DATASET.audio_down")
        if audio_down is None:
            audio_down = 640
        self.hparams.audio_down = audio_down
        # self.hparams.w_vectorizer = WordVectorizer(cfg.DATASET.WORD_VERTILIZER_PATH, "our_vab")
        self.hparams.motion_representation = cfg.DATASET.motion_representation
        self.hparams.smpl_path = cfg.DATASET.SMPLX_MODEL_DIR
        self.hparams.njoints = 55

        # Get normalization settings for preprocessed data (passed to all datasets, used when preprocessed_dir is present)
        normalization_dir = OmegaConf.select(cfg, "DATASET.normalization_dir")
        normalize_cfg = OmegaConf.select(cfg, "DATASET.normalize")
        self.hparams.normalization_dir = normalization_dir
        self.hparams.normalize = True if normalize_cfg is None else normalize_cfg

        # Check if using preprocessed datasets (has preprocessed_dir in any dataset config)
        use_preprocessed = self._check_preprocessed_datasets(dataset_configs)

        # Select dataset class based on stage
        if cfg.TRAIN.STAGE == "vae" or cfg.TRAIN.STAGE == "vqvae":
            # If using preprocessed data, always use MixedDatasetVQ (which has preprocessed loading logic)
            if use_preprocessed:
                self.Dataset = MixedDatasetVQ
                self.DatasetEval = MixedDatasetVQ
            # Use FaceVQDataset by default for all VAE/VQ stages
            elif cfg.Selected_part == 'upper':
                self.Dataset = UpperVQDataset
                self.DatasetEval = UpperVQDataset
            elif cfg.Selected_part == 'lower' or cfg.Selected_part == 'lower_54' or cfg.Selected_part == 'lower_global':
                self.Dataset = LowerVQDataset
                self.DatasetEval = LowerVQDataset
            elif cfg.Selected_part == 'face':
                self.Dataset = FaceVQDataset
                self.DatasetEval = FaceVQDataset
            elif cfg.Selected_part in ['compositional', 'full_rot', 'full_h3d', 'full_genmo', 'upper_lower_global']:
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
            full_dataset = load_dataset('json', data_files=dataset_path)['train']
            print(f"Loaded {len(full_dataset)} conversation sequences")

            # Split into train / val / test so evaluation doesn't use training data.
            # Use 90% train, 5% val, 5% test.
            split_1 = full_dataset.train_test_split(test_size=0.1, seed=42)
            train_dataset = split_1['train']
            remaining = split_1['test']
            split_2 = remaining.train_test_split(test_size=0.5, seed=42)
            val_dataset = split_2['train']
            test_dataset = split_2['test']
            print(f"  Split: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")

            # Set instance attributes; the base class properties already check
            # self._xxx_dataset and return them directly when not None.
            self._train_dataset = train_dataset
            self._val_dataset = val_dataset
            self._test_dataset = test_dataset

            # Keep the classes for compatibility, but mark them so they aren't used
            self.Dataset = None
            self.DatasetEval = None

            # Set the appropriate collate_fn for HuggingFace datasets
            self.dataloader_options = {"collate_fn": huggingface_dataset_collate}
        else:
            raise RuntimeError("Haven't setup this code!")

        # # Get additional info of the dataset
        # self._sample_set = self.get_sample_set(overrides={"split": "test", "tiny": True})

    def _check_preprocessed_datasets(self, dataset_configs):
        """Check if any dataset in configs uses preprocessed format.

        Preprocessed datasets are detected by the presence of 'preprocessed_dir' field.
        """
        if dataset_configs is None:
            return False
        for config in dataset_configs:
            if config.get("preprocessed_dir"):
                return True
        return False

    def _apply_dataset_defaults(self, dataset_configs):
        if dataset_configs is None:
            return dataset_configs

        global_defaults = {
            "pose_length": OmegaConf.select(self.cfg, "DATASET.pose_length"),
            "stride": OmegaConf.select(self.cfg, "DATASET.stride"),
            "pose_fps": OmegaConf.select(self.cfg, "DATASET.pose_fps"),
            "unit_length": OmegaConf.select(self.cfg, "DATASET.unit_length"),
            "pre_frames": OmegaConf.select(self.cfg, "DATASET.pre_frames"),
            "audio_fps": OmegaConf.select(self.cfg, "DATASET.audio_fps"),
            "audio_down": OmegaConf.select(self.cfg, "DATASET.audio_down"),
            "foot_contact_path": OmegaConf.select(self.cfg, "DATASET.foot_contact_path"),
            "motion_unit": OmegaConf.select(self.cfg, "DATASET.motion_unit"),
        }
        dataset_override_keys = (
            "training_speakers",
            "testing_speakers",
            "additional_data",
            "pose_rep",
            "pose_rep_mirror",
            "pose_rep_subdirs",
            "pose_rep_face",
            "foot_contact_path",
            "motion_unit",
            "code_path",
            "code_path_audio",
        )
        for config in dataset_configs:
            dataset_name = OmegaConf.select(config, "name")
            for key, value in global_defaults.items():
                if value is not None and OmegaConf.select(config, key) is None:
                    config[key] = value
            if dataset_name:
                dataset_defaults = OmegaConf.select(self.cfg, f"DATASET.{dataset_name}")
                if dataset_defaults is not None:
                    for key in dataset_override_keys:
                        value = OmegaConf.select(dataset_defaults, key)
                        if value is not None and OmegaConf.select(config, key) is None:
                            config[key] = value
        return dataset_configs


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
