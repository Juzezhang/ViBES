import random
import pickle
import os
import numpy as np
import torch
from torch.utils import data
from os.path import join as pjoin
import json
import math
import glob
from tqdm import tqdm
from .data_tools import (
    JOINT_MASK_LOWER,
    JOINT_MASK_FULL,
    BEAT_SMPLX_LOWER,
)
from multimodal_tokenizers.utils.rotation_conversions import axis_angle_to_6d, axis_angle_to_matrix, rotation_6d_to_axis_angle, axis_angle_to_6d_np
import pandas as pd
import codecs as cs
import smplx

class GlobalVQDataset(data.Dataset):
    """
    Dataset for processing facial expression data for the VAE/VQ-VAE training stage.
    Follows the pattern of MixedDatasetVQ but focuses on facial expressions.
    """
    
    def __init__(
        self,
        args,
        dataset_configs,
        dataset_configs_test,
        split,
        tiny=False,
        debug=False,
        stage='lm_pretrain',
        task_path=None,
        mean=None,
        std=None,
        motion_representation="rotation",
        smpl_path=None,
        njoints=55,
        use_cache=False,  # Whether to load data from cache when available
        save_cache=False,  # Whether to save processed data to cache
        cache_format="pkl", # Format to use for caching: "h5", "npz", or "pkl"
        **kwargs,
    ):
        """
        Initializes the dataset class.

        Parameters:
        - dataset_configs: List of configurations for different datasets.
        - split: Specifies the data split (train/val/test).
        - args: Additional arguments.
        - unit_length: Length of the units for data processing.
        - fps: Frames per second for motion data.
        - tiny: Whether to use a small subset for debugging.
        - debug: If True, enables debug mode.
        - stage: Specifies the training stage.
        - task_path: Path to the task instructions file.
        - motion_representation: Specifies the motion representation.
        - use_cache: Whether to load data from cache when available.
        - save_cache: Whether to save processed data to cache.
        - cache_format: Format to use for caching ("h5", "npz", or "pkl").
        """

        # Store debug flag for cache path generation
        self.debug = debug
        
        # Set max data size depending on debug mode
        if tiny or debug:
            self.maxdata = 10
        else:
            self.maxdata = 1e10

        self.args = args
        self.task_path = task_path
        self.stage = stage
        self.split = split
        # if split == 'test':
        #     self.maxdata = 1e10

        self.njoints = njoints
        self.use_cache = use_cache
        self.save_cache = save_cache
        self.cache_format = cache_format
        self.smpl_path = smpl_path  # Store the path for later use if needed
        self.motion_representation = motion_representation

        # Store kwargs for SMPLX initialization if needed
        if motion_representation == "rotation":
            # Initialize the joint masks for different body parts
            self.joint_mask_lower = JOINT_MASK_LOWER
            # We'll initialize SMPLX only when needed, not upfront
            self.smplx_2020 = None
        elif motion_representation == "h3d":
            self.h3d_mean = mean
            self.h3d_std = std
        # Dictionary to store data and metadata
        self.data_dict = {}
        self.metadata = []
        
        # Load each dataset based on its type from the configuration
        if split == 'test':
            dataset_configs_selected = dataset_configs_test
        else:
            dataset_configs_selected = dataset_configs
            
        for config in dataset_configs_selected:
            dataset_name = config.get("name")
            if dataset_name == "BEAT2":
                self._load_beat2(config)
                self.data_dict.update(self.data_dict_beat2)
                self.metadata.extend(self.metadata_beat2)
            elif dataset_name == "AMASS":
                self._load_amass(config)
                self.data_dict.update(self.data_dict_amass)
                self.metadata.extend(self.metadata_amass)
            elif dataset_name == "CANDOR":
                self._load_candor(config)
                self.data_dict.update(self.data_dict_candor)
                self.metadata.extend(self.metadata_candor)
            elif dataset_name == "TFHP":
                self._load_tfhp(config)
                self.data_dict.update(self.data_dict_tfhp)
                self.metadata.extend(self.metadata_tfhp)
            elif dataset_name == "YouTube_Talking":
                self._load_youtube_talking(config)
                self.data_dict.update(self.data_dict_youtube_talking)
                self.metadata.extend(self.metadata_youtube_talking)
            elif dataset_name == "YouTube_Talking_Synthetic":
                self._load_youtube_talking_synthetic(config)
                self.data_dict.update(self.data_dict_youtube_talking_synthetic)
                self.metadata.extend(self.metadata_youtube_talking_synthetic)
            else:
                # Skip datasets without face data
                continue
                
        print(f"[GlobalVQDataset] Loaded {len(self.metadata)} samples")
        
    def _get_cache_path(self, dataset_path, dataset_type):
        """Generate a cache file path based on dataset path and type."""
        cache_dir = os.path.join(dataset_path, 'cache')
        config_str = f"{dataset_type}_global_{self.split}_vq"
        
        if self.cache_format == "h5":
            ext = ".h5"
        elif self.cache_format == "npz":
            ext = ".npz"
        else:  # Default to pickle
            ext = ".pkl"
            
        cache_path = os.path.join(cache_dir, f"{config_str}{ext}")
        return cache_path
        
    def _save_to_cache(self, cache_path, data_dict, metadata, dataset_name):
        """Save processed data to cache."""
        if not self.save_cache:
            return
            
        try:
            # Create cache directory if it doesn't exist
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            
            print(f"Saving {dataset_name} global dataset to cache: {cache_path}")
            
            # Convert any remaining PyTorch tensors to NumPy arrays for consistency
            numpy_data_dict = {}
            for key, item in data_dict.items():
                numpy_item = {}
                for attr_key, attr_value in item.items():
                    if isinstance(attr_value, torch.Tensor):
                        # Convert torch tensors to numpy arrays
                        numpy_item[attr_key] = attr_value.cpu().numpy()
                    else:
                        numpy_item[attr_key] = attr_value
                numpy_data_dict[key] = numpy_item
            
            # Save based on the selected format
            if self.cache_format == "pkl":
                with open(cache_path, 'wb') as f:
                    pickle.dump({
                        'data_dict': numpy_data_dict,
                        'metadata': metadata
                    }, f, protocol=pickle.HIGHEST_PROTOCOL)
                
            elif self.cache_format == "npz":
                np.savez(cache_path, 
                    data_dict=numpy_data_dict, 
                    metadata=metadata
                )
                
            print(f"Successfully saved {dataset_name} global cache to {cache_path}")
            
        except Exception as e:
            print(f"Error saving {dataset_name} global cache: {str(e)}")
            import traceback
            traceback.print_exc()
            
    def _load_from_cache(self, cache_path, dataset_name):
        """Load processed data from cache."""
        if not self.use_cache:
            return None, None
        
        if not os.path.exists(cache_path):
            print(f"Global cache file not found: {cache_path}")
            return None, None
        
        try:
            print(f"Loading {dataset_name} global dataset from cache: {cache_path}")
            
            if self.cache_format == "pkl":
                with open(cache_path, 'rb') as f:
                    cached_data = pickle.load(f)
                    data_dict = cached_data['data_dict']
                    metadata = cached_data['metadata']
            elif self.cache_format == "npz":
                loaded = np.load(cache_path, allow_pickle=True)
                
                if loaded['metadata'].size == 1:
                    metadata = loaded['metadata'].item()
                else:
                    metadata = loaded['metadata']
                
                if loaded['data_dict'].size == 1:
                    data_dict = loaded['data_dict'].item()
                else:
                    print(f"Warning: Unexpected data_dict structure in {cache_path}")
                    data_dict = {}
            
            print(f"Successfully loaded {dataset_name} global cache from {cache_path}")
            return data_dict, metadata
            
        except Exception as e:
            print(f"Error loading {dataset_name} global cache: {str(e)}")
            import traceback
            traceback.print_exc()
            return None, None

    def _initialize_smplx_if_needed(self):
        """
        Initialize the SMPLX model if it hasn't been initialized yet.
        This is done lazily to avoid unnecessary GPU memory usage when loading from cache.
        """
        if self.smplx_2020 is None and (self.motion_representation == "rotation"):
            print("Initializing SMPLX model for data processing...")
            self.smplx_2020 = smplx.create(self.smpl_path,
                model_type='smplx',
                gender='NEUTRAL_2020',
                use_face_contour=False,
                num_betas=300,
                num_expression_coeffs=100,
                ext='npz',
                use_pca=False,
                ).cuda().eval()


    def _load_beat2(self, config):
        """
        Load the BEAT2 dataset based on the configuration.

        Parameters:
            config: Configuration dictionary for BEAT2.
        """
        # Get cache path for this specific dataset
        data_root = self.args["BEAT2"].ROOT
        cache_path = self._get_cache_path(data_root, "BEAT2")
        
        # Check if we should load from cache
        data_dict, metadata = self._load_from_cache(cache_path, "BEAT2")
        if data_dict is not None:
            self.data_dict_beat2 = data_dict
            self.metadata_beat2 = metadata
            return
        
        # We need to process data from scratch, so initialize SMPLX if needed
        self._initialize_smplx_if_needed()
        
        print(f"Processing BEAT2 dataset...")
        
        # Set up dataset parameters
        self.data_root_beat2 = data_root
        self.ori_length = config.pose_length
        additional_data = config.additional_data
        training_speakers = config.training_speakers
        pose_rep = config.pose_rep
        pose_fps_beat2 = config.pose_fps
        
        # Load split rules from CSV file
        split_rule = pd.read_csv(pjoin(data_root, "train_test_split.csv"))
        
        # Filter files based on training speakers and split type
        if self.split == 'token':
            # For token split, only filter by training speakers
            self.selected_file = split_rule.loc[
                ((split_rule['id'].str.split("_").str[0].astype(int).isin(training_speakers)))
            ]
        else:
            # For other splits, filter by both split type and training speakers
            self.selected_file = split_rule.loc[
                (split_rule['type'] == self.split) &
                ((split_rule['id'].str.split("_").str[0].astype(int).isin(training_speakers)))
            ]
            # Include additional data if specified
            if additional_data:
                split_b = split_rule.loc[
                    (split_rule['type'] == 'additional') & 
                    (split_rule['id'].str.split("_").str[0].astype(int).isin(training_speakers))
                ]
                self.selected_file = pd.concat([self.selected_file, split_b])

        self.data_dict_beat2 = {}
        self.metadata_beat2 = []

        # Process each file in the selected files
        for index, file_name in tqdm(self.selected_file.iterrows()):
            f_name = file_name["id"]
            pose_file = pjoin(self.data_root_beat2, pose_rep, f_name + ".npz")
            
            # try:
            # Load pose data from NPZ file
            pose_data = np.load(pose_file, allow_pickle=True)
            poses = pose_data["poses"]
            n, c = poses.shape[0], poses.shape[1]  # n: number of frames, c: pose dimension
            trans = pose_data["trans"]
            betas = pose_data["betas"]
            # Repeat betas for each frame
            betas = np.repeat(pose_data["betas"].reshape(1, 300), poses.shape[0], axis=0)

            # expressions = pose_data["expressions"]

            # Apply joint mask to filter relevant joints
            pose_processed = poses * JOINT_MASK_FULL
            pose_processed = pose_processed[:, JOINT_MASK_FULL.astype(bool)]
            
            if self.motion_representation == 'rotation':
                # Calculate foot contacts using existing function or load from cache
                foot_contacts_path = pjoin(self.data_root_beat2, 'foot_contacts_25', f_name + '.npy')
                if os.path.exists(foot_contacts_path):
                    contacts = np.load(foot_contacts_path)
                else:
                    contacts = self.comput_foot_contacts(pose_data)
                    os.makedirs(pjoin(self.data_root_beat2, 'foot_contacts_25'), exist_ok=True)
                    np.save(foot_contacts_path, contacts)
                # Concatenate foot contacts to pose data
                pose_processed = np.concatenate([pose_processed, contacts], axis=1)

            # Extract different components from processed pose
            tar_pose = pose_processed[ :, :165]  # Body pose (55 joints * 3)
            tar_contact = pose_processed[ :, 165:169]  # Foot contacts (4 values)
            # tar_exps = expressions  # Facial expressions
            tar_trans = trans  # Translation

            # # Extract and convert jaw pose data
            # tar_pose_jaw = tar_pose[:, 66:69]  # Jaw pose (3D rotation)
            # tar_pose_jaw_6d = axis_angle_to_6d_np(tar_pose_jaw).reshape(n, 6)  # Convert to 6D representation
            
            # # Concatenate jaw pose and expressions for face data
            # tar_pose_face = np.concatenate([tar_pose_jaw_6d, tar_exps], axis=1)

            # # Extract and convert hand pose data
            # tar_pose_hands = tar_pose[:, 25 * 3:55 * 3].reshape(n, 30, 3)  # 30 hand joints
            # tar_pose_hands_6d = axis_angle_to_6d_np(tar_pose_hands).reshape(n, 30 * 6)

            # # Extract and convert upper body pose data
            # tar_pose_upper = tar_pose[:, self.joint_mask_upper.astype(bool)].reshape(n, 13, 3)  # 13 upper body joints
            # tar_pose_upper_6d = axis_angle_to_6d_np(tar_pose_upper).reshape(n, 13 * 6)

            # Extract and convert lower body pose data
            tar_pose_leg = tar_pose[:, self.joint_mask_lower.astype(bool)].reshape(n, 9, 3)  # 9 lower body joints
            tar_pose_leg_6d = axis_angle_to_6d_np(tar_pose_leg).reshape(n, 9 * 6)
            
            # Combine lower body pose with translation and contacts
            tar_pose_lower = np.concatenate([tar_pose_leg_6d, tar_trans, tar_contact], axis=1)
            
            # # Convert full pose to 6D representation
            # tar_pose_6d = axis_angle_to_6d_np(tar_pose.reshape(n, 55, 3)).reshape(n, 55 * 6)

            # Calculate time segments
            round_seconds_skeleton = tar_pose_lower.shape[0] // pose_fps_beat2
            if round_seconds_skeleton == 0:
                round_seconds_skeleton = 1
            clip_s_t, clip_e_t = 0, round_seconds_skeleton - 0  # Start and end time in seconds
            clip_s_f_pose, clip_e_f_pose = clip_s_t * pose_fps_beat2, clip_e_t * pose_fps_beat2  # Start and end frame

            # Determine stride and cut length based on split type
            if self.split == 'test' or self.split == 'token':  # stride = length for test
                cut_length = clip_e_f_pose - clip_s_f_pose
                stride = cut_length
                self.max_length = cut_length
            else:
                stride = int(config.stride)
                cut_length = int(self.ori_length)
            
            # Calculate number of subdivisions
            num_subdivision = math.floor((clip_e_f_pose - clip_s_f_pose - cut_length) / stride) + 1

            # Create segments of motion data
            for i in range(num_subdivision):  # cut into segments (e.g., 2s clips)

                start_idx = clip_s_f_pose + i * config.stride
                fin_idx = start_idx + cut_length
                
                # Extract segment from each data type
                # sample_pose = tar_pose_6d[start_idx:fin_idx]
                # sample_face = tar_pose_face[start_idx:fin_idx]
                # sample_hand = tar_pose_hands_6d[start_idx:fin_idx]
                # sample_upper = tar_pose_upper_6d[start_idx:fin_idx]
                sample_lower = tar_pose_lower[start_idx:fin_idx]
                sample_trans = tar_trans[start_idx:fin_idx]
                sample_shape = betas[start_idx:fin_idx]
                # sample_expressions = expressions[start_idx:fin_idx]

                # Create unique name for this segment
                new_name = 'beat2_' + '%s_%d' % (f_name,i)
                
                # Store processed data
                self.data_dict_beat2[new_name] = {
                    # 'face': sample_face,
                    # 'hand': sample_hand,
                    # 'upper': sample_upper,
                    'lower': sample_lower,
                    # 'pose': sample_pose,
                    'shape': sample_shape,
                    'trans': sample_trans,
                    # 'exps': sample_expressions,
                    'id': f_name,
                    'dataset_name': 'beat2',
                }
                self.metadata_beat2.append(new_name)
                
            # Break early for debugging if max data reached
            if index >= self.maxdata:
                break
        
        # Save processed data to cache
        self._save_to_cache(cache_path, self.data_dict_beat2, self.metadata_beat2, "BEAT2")


    def _load_amass(self, config):
        """
        Load the AMASS dataset based on the configuration.

        Parameters:
        - config: Configuration dictionary for AMASS.
        """
        # Get cache path for this specific dataset
        data_root = self.args["AMASS"].ROOT
        cache_path = self._get_cache_path(data_root, "AMASS")
        
        # Check if we should load from cache
        data_dict, metadata = self._load_from_cache(cache_path, "AMASS")
        if data_dict is not None:
            self.data_dict_amass = data_dict
            self.metadata_amass = metadata
            return
        
        # We need to process data from scratch, so initialize SMPLX if needed
        self._initialize_smplx_if_needed()
        
        print(f"Processing AMASS dataset...")
        
        # Set up dataset parameters
        self.data_root_amass = data_root
        pose_fps_amass = config.pose_fps
        
        # Load train and test splits from text files
        split_file_train = pjoin(self.data_root_amass, 'train.txt')
        
        # Load training data IDs
        id_list_train = []
        with cs.open(split_file_train, "r") as f:
            for line in f.readlines():
                id_list_train.append(line.strip())

        split_file_test = pjoin(self.data_root_amass, 'test.txt')
        
        # Load test data IDs
        id_list_test = []
        with cs.open(split_file_test, "r") as f:
            for line in f.readlines():
                id_list_test.append(line.strip())

        # Select appropriate ID list based on split
        if self.split == 'train':
            id_list_amass = id_list_train
        elif self.split == 'test':
            id_list_amass = id_list_test
        else:
            # For other splits, use both train and test
            id_list_amass = id_list_train + id_list_test

        self.ori_length = config.pose_length
        # Calculate maximum lengths for data
        # self.max_length = int(config.pose_length)

        self.metadata_amass = []
        self.data_dict_amass = {}

        ##################  AMASS  ##################
        # Process each file
        for index, file_name in tqdm(enumerate(id_list_amass)):
            try:
                # Load pose data from aligned AMASS data
                pose_file = pjoin(self.data_root_amass, 'amass_data_align_25', file_name+'.npz')
                pose_data = np.load(pose_file, allow_pickle=True)
                
                # Calculate stride for downsampling to target FPS
                stride = int(pose_fps_amass / pose_fps_amass)

                # Extract and downsample pose data
                poses = pose_data["poses"][::stride]
                n, c = poses.shape[0], poses.shape[1]  # n: number of frames, c: pose dimension
                tar_trans = pose_data["trans"][::stride]

                # Pad betas to 300 dimensions (AMASS uses 16, but we need 300 for SMPLX)
                padded_betas = np.zeros(300)
                padded_betas[:16] = pose_data["betas"]
                betas = np.repeat(padded_betas.reshape(1, 300), n, axis=0)
                
                # Apply joint mask to filter relevant joints
                pose_processed = poses * JOINT_MASK_FULL
                pose_processed = pose_processed[:, JOINT_MASK_FULL.astype(bool)]
                
                # if self.select_type == 'full_rot' or self.select_type == 'separate_rot':
                # if self.motion_representation == 'rotation':
                # Calculate foot contacts for rotation representation
                foot_contacts_path = pjoin(self.data_root_amass, 'foot_contacts_25', file_name + '.npy')
                if os.path.exists(foot_contacts_path):
                    contacts = np.load(foot_contacts_path)
                else:
                    contacts = self.comput_foot_contacts(pose_data)
                    os.makedirs(pjoin(self.data_root_amass, 'foot_contacts_25'), exist_ok=True)
                    np.save(foot_contacts_path, contacts)
                # Concatenate foot contacts to pose data
                pose_processed = np.concatenate([pose_processed, contacts], axis=1)
                
                # Extract body pose (55 joints * 3)
                tar_pose = pose_processed[:, :165]

                # Get foot contacts
                tar_contact = contacts


                # Extract and convert lower body pose data
                tar_pose_leg = tar_pose[:, self.joint_mask_lower.astype(bool)].reshape(n, 9, 3)
                tar_pose_leg_6d = axis_angle_to_6d_np(tar_pose_leg).reshape(n, 9 * 6)
                
                # Combine lower body pose with translation and contacts
                tar_pose_lower = np.concatenate([tar_pose_leg_6d, tar_trans, tar_contact], axis=1)

                # Calculate time segments
                round_seconds_skeleton = tar_pose_lower.shape[0] // pose_fps_amass
                if round_seconds_skeleton == 0:
                    round_seconds_skeleton = 1
                clip_s_t, clip_e_t = 0, round_seconds_skeleton - 0
                clip_s_f_pose, clip_e_f_pose = clip_s_t * pose_fps_amass, clip_e_t * pose_fps_amass
            
                # Determine stride and cut length based on split type
                if self.split == 'test' or self.split == 'token':  # stride = length for test
                    cut_length = clip_e_f_pose - clip_s_f_pose
                    stride = cut_length
                    # self.max_length = cut_length
                else:
                    stride = int(config.stride)
                    cut_length = int(self.ori_length)
                
                # Calculate number of subdivisions
                num_subdivision = math.floor((clip_e_f_pose - clip_s_f_pose - cut_length) / stride) + 1

                # Create segments of motion data
                for i in range(num_subdivision):
                    start_idx = clip_s_f_pose + i * stride
                    fin_idx = start_idx + cut_length
                    
                    # Skip if segment goes out of bounds
                    if fin_idx > tar_pose_lower.shape[0]:
                        continue
                    
                    # Extract segment from each data type    
                    # sample_pose = tar_pose_6d[start_idx:fin_idx]
                    # sample_face = tar_pose_face[start_idx:fin_idx]  # Empty for AMASS
                    # sample_hand = tar_pose_hands_6d[start_idx:fin_idx]  # Empty for AMASS
                    # sample_upper = tar_pose_upper_6d[start_idx:fin_idx]
                    sample_lower = tar_pose_lower[start_idx:fin_idx]
                    sample_trans = tar_trans[start_idx:fin_idx]
                    sample_shape = betas[start_idx:fin_idx]

                    # Create unique name for this segment
                    new_name = 'amass_' + '%s_%d' % (file_name, i)

                    # Store processed data
                    self.data_dict_amass[new_name] = {
                        # 'face': sample_face,
                        # 'hand': sample_hand,
                        # 'upper': sample_upper,
                        'lower': sample_lower,
                        # 'pose': sample_pose,
                        'shape': sample_shape,
                        'trans': sample_trans,
                        'id': file_name,
                        'dataset_name': 'amass',
                    }
                    self.metadata_amass.append(new_name)
            except Exception as e:
                # Skip files that can't be processed
                # print(f"Error processing file {f_name}: {str(e)}")
                continue

            # Break early for debugging if max data reached
            if index >= self.maxdata:
                break
        
        # Save processed data to cache
        self._save_to_cache(cache_path, self.data_dict_amass, self.metadata_amass, "AMASS")


    def _load_candor(self, config):
        """
        Load facial expressions from the CANDOR dataset.
        """
        # Get cache path for this specific dataset
        data_root = self.args["CANDOR"].ROOT
        cache_path = self._get_cache_path(data_root, "CANDOR_face")
        
        # Check if we should load from cache
        data_dict, metadata = self._load_from_cache(cache_path, "CANDOR_face")
        if data_dict is not None:
            self.data_dict_candor = data_dict
            self.metadata_candor = metadata
            return
        
        max_segment_length = config.pose_length
        stride = config.stride
        
        print(f"Processing CANDOR dataset for face expressions...")
        
        self.data_root_candor = data_root
        
        # Load the dataset structure
        structure_path = pjoin(self.data_root_candor, 'candor_structure.json')
        if not os.path.exists(structure_path):
            print(f"CANDOR structure file not found: {structure_path}")
            self.data_dict_candor = {}
            self.metadata_candor = []
            return
            
        with open(structure_path, 'r') as f:
            candor_structure = json.load(f)
        
        self.data_dict_candor = {}
        self.metadata_candor = []
        
        # Determine train/val/test split (assuming 80/10/10)
        total_seqs = len(candor_structure)
        # keys = list(candor_structure.keys())
        # random.seed(42)  # For reproducibility
        # random.shuffle(keys)
        split_file = pjoin(data_root, self.split + '_processed.txt')
        # Data id list
        id_list_candor = []
        with cs.open(split_file, "r") as f:
            for line in f.readlines():
                id_list_candor.append(line.strip())

        
        for sequence_id in tqdm(id_list_candor, desc=f"Processing CANDOR {self.split} sequences"):
            try:
                # Process each participant in the conversation
                participants = os.listdir(pjoin(self.data_root_candor, 'FLAME_coeffs', sequence_id))

                for idx, participant_file in enumerate(participants):
                    flame_path = pjoin(self.data_root_candor, 'FLAME_coeffs', sequence_id, participant_file)
                    
                    if not os.path.exists(flame_path):
                        continue
                        
                    flame_data = np.load(flame_path)
                    
                    # Extract expression and pose parameters
                    expressions = flame_data['exp']  # Shape: [n_frames, 50]
                    pose = flame_data['pose']        # Shape: [n_frames, 6] (global + jaw)
                    shape = flame_data['shape']      # Shape: [n_frames, 100/300] (betas)
                    
                    # Get sequence length
                    seq_length = expressions.shape[0]
                    
                    # Process head pose (positions 0:3 in pose)
                    head_pose = pose[:, :3]  # Global head orientation in axis-angle
                    head_pose_6d = axis_angle_to_6d_np(head_pose).reshape(seq_length, 6)
                    
                    # Process jaw pose (positions 3:6 in pose)
                    jaw_pose = pose[:, 3:6]  # Jaw rotation in axis-angle
                    jaw_pose_6d = axis_angle_to_6d_np(jaw_pose).reshape(seq_length, 6)
                    
                    # Pad expressions to 100 dimensions as needed
                    if expressions.shape[1] < 100:
                        padded_exps = np.zeros((seq_length, 100), dtype=expressions.dtype)
                        padded_exps[:, :expressions.shape[1]] = expressions
                        expressions = padded_exps
                    elif expressions.shape[1] > 100:
                        expressions = expressions[:, :100]
                    
                    # Concatenate head pose, jaw pose and expressions for 112D face representation
                    face_features = np.concatenate([head_pose_6d, jaw_pose_6d, expressions], axis=1)
                    
                    
                    if seq_length <= max_segment_length:
                        # Short enough to use as a single segment
                        segment_name = f'candor_face_{sequence_id}_{idx}'
                        self.data_dict_candor[segment_name] = {
                            'face': face_features,
                            'shape': shape,
                            'pose': pose[:, :3],  # Global head orientation
                            'id': sequence_id,
                            'participant': idx,
                            'video_name': participant_file.split('.')[0],
                            'dataset_name': 'candor'
                        }
                        self.metadata_candor.append(segment_name)
                    else:
                        # Split into overlapping segments
                        num_segments = (seq_length - max_segment_length) // stride + 1
                        
                        for seg_idx in range(num_segments):
                            start_idx = seg_idx * stride
                            end_idx = min(start_idx + max_segment_length, seq_length)
                            
                            segment_name = f'candor_face_{sequence_id}_{idx}_{seg_idx}'
                            self.data_dict_candor[segment_name] = {
                                'face': face_features[start_idx:end_idx],
                                'shape': shape[start_idx:end_idx],
                                'pose': pose[start_idx:end_idx, :3],  # Global head orientation
                                'id': sequence_id,
                                'participant': idx,
                                'video_name': participant_file.split('.')[0],
                                'dataset_name': 'candor'
                            }
                            self.metadata_candor.append(segment_name)
            except Exception as e:
                print(f"Error processing sequence {sequence_id}: {str(e)}")
                continue
                
            # For fast debug
            if len(self.metadata_candor) >= self.maxdata:
                break
        
        # Save processed data to cache
        self._save_to_cache(cache_path, self.data_dict_candor, self.metadata_candor, "CANDOR_face")

    def _load_tfhp(self, config):
        """
        Load facial expressions from the TFHP dataset.
        """
        # Get cache path for this specific dataset
        data_root = self.args["TFHP"].ROOT
        cache_path = self._get_cache_path(data_root, "TFHP_face")
        
        # Check if we should load from cache
        data_dict, metadata = self._load_from_cache(cache_path, "TFHP_face")
        if data_dict is not None:
            self.data_dict_tfhp = data_dict
            self.metadata_tfhp = metadata
            return
        
        max_segment_length = config.pose_length
        stride = config.stride
        
        print(f"Processing TFHP dataset for face expressions...")
        
        self.data_root_tfhp = data_root
        split_file = pjoin(data_root, self.split + '.txt')
        # Data id list
        id_list_tfhp = []
        with cs.open(split_file, "r") as f:
            for line in f.readlines():
                id_list_tfhp.append(line.strip())


        self.data_dict_tfhp = {}
        self.metadata_tfhp = []
        
        # Determine train/val/test split (assuming 80/10/10)
        total_seqs = len(id_list_tfhp)
        keys = list(id_list_tfhp)

        
        for sequence_id in tqdm(id_list_tfhp, desc=f"Processing TFHP {self.split} sequences"):
            try:

                files_coef = sorted(os.listdir(pjoin(self.data_root_tfhp, 'coef', sequence_id)))
                if len(files_coef) == 0:
                    continue
                for file_coef in files_coef:

                    if not file_coef.endswith('.npz'):
                        continue
                    # Process each participant in the conversation
                    flame_path = pjoin(self.data_root_tfhp, 'coef', sequence_id, file_coef)
                    
                    flame_data = np.load(flame_path)
                    
                    # Extract expression and pose parameters
                    expressions = flame_data['exp']  # Shape: [n_frames, 50]
                    pose = flame_data['pose']        # Shape: [n_frames, 6] (global + jaw)
                    shape = flame_data['shape']      # Shape: [n_frames, 100/300] (betas)
                    
                    # Get sequence length
                    seq_length = expressions.shape[0]
                    
                    # Process head pose (positions 0:3 in pose)
                    head_pose = pose[:, :3]  # Global head orientation in axis-angle
                    head_pose_6d = axis_angle_to_6d_np(head_pose).reshape(seq_length, 6)
                    
                    # Process jaw pose (positions 3:6 in pose)
                    jaw_pose = pose[:, 3:6]  # Jaw rotation in axis-angle
                    jaw_pose_6d = axis_angle_to_6d_np(jaw_pose).reshape(seq_length, 6)
                    
                    # Pad expressions to 100 dimensions as needed
                    if expressions.shape[1] < 100:
                        padded_exps = np.zeros((seq_length, 100), dtype=expressions.dtype)
                        padded_exps[:, :expressions.shape[1]] = expressions
                        expressions = padded_exps
                    elif expressions.shape[1] > 100:
                        expressions = expressions[:, :100]
                    
                    # Concatenate head pose, jaw pose and expressions for 112D face representation
                    face_features = np.concatenate([head_pose_6d, jaw_pose_6d, expressions], axis=1)
                    
                    
                    if seq_length <= max_segment_length:
                        # Short enough to use as a single segment
                        # segment_name = f'tfhp_face_{sequence_id}_{file_coef.split('.')[0]}'
                        sequence_id_name = sequence_id.replace('/', '_')
                        file_coef_name = file_coef.split('.')[0]
                        segment_name = f'tfhp_face_{sequence_id_name}_{file_coef_name}'
                        self.data_dict_tfhp[segment_name] = {
                            'face': face_features,
                            'shape': shape,
                            'pose': pose[:, :3],  # Global head orientation
                            'id': sequence_id,
                            'video_name': file_coef_name,
                            'dataset_name': 'TFHP'
                        }
                        self.metadata_tfhp.append(segment_name)
                    else:
                        # Split into overlapping segments
                        num_segments = (seq_length - max_segment_length) // stride + 1
                        for seg_idx in range(num_segments):
                            start_idx = seg_idx * stride
                            end_idx = min(start_idx + max_segment_length, seq_length)
                            file_coef_name = file_coef.split('.')[0]
                            sequence_id_name = sequence_id.replace('/', '_')
                            segment_name = f'tfhp_face_{sequence_id_name}_{file_coef_name}_{seg_idx}'
                            # segment_name = 0
                            self.data_dict_tfhp[segment_name] = {
                                'face': face_features[start_idx:end_idx],
                                'shape': shape[start_idx:end_idx],
                                'pose': pose[start_idx:end_idx, :3],  # Global head orientation
                                'id': sequence_id,
                                'video_name': file_coef_name,
                                'dataset_name': 'TFHP'
                            }
                            self.metadata_tfhp.append(segment_name)
            except Exception as e:
                print(f"Error processing sequence {sequence_id}: {str(e)}")
                continue

            # For fast debug
            if len(self.metadata_tfhp) >= self.maxdata:
                break
        
        # Save processed data to cache
        self._save_to_cache(cache_path, self.data_dict_tfhp, self.metadata_tfhp, "TFHP_face")

    def _load_youtube_talking(self, config):
        """
        Load facial expressions from the YouTube_Talking dataset.
        """
        # Get cache path for this specific dataset
        data_root = self.args["YouTube_Talking"].ROOT
        cache_path = self._get_cache_path(data_root, "YouTube_Talking_face")
        
        # Check if we should load from cache
        data_dict, metadata = self._load_from_cache(cache_path, "YouTube_Talking_face")
        if data_dict is not None:
            self.data_dict_youtube_talking = data_dict
            self.metadata_youtube_talking = metadata
            return
        
        max_segment_length = config.pose_length
        stride = config.stride
        
        print(f"Processing YouTube_Talking dataset for face expressions...")
        
        self.data_root_youtube_talking = data_root
        split_file = pjoin(data_root, self.split + '_processed.txt')
        # Data id list
        id_list_youtube_talking = []
        with cs.open(split_file, "r") as f:
            for line in f.readlines():
                id_list_youtube_talking.append(line.strip())

        self.data_dict_youtube_talking = {}
        self.metadata_youtube_talking = []
        
        # Determine train/val/test split (assuming 80/10/10)
        total_seqs = len(id_list_youtube_talking)
        keys = list(id_list_youtube_talking)

        
        for sequence_id in tqdm(id_list_youtube_talking, desc=f"Processing YouTube_Talking {self.split} sequences"):
            try:
                # Process each participant in the conversation
                flame_path = pjoin(self.data_root_youtube_talking, 'FLAME_coeffs_25', sequence_id + '.npz')
                flame_data = np.load(flame_path)
                
                # Extract expression and pose parameters
                expressions = flame_data['exp']  # Shape: [n_frames, 50]
                pose = flame_data['pose']        # Shape: [n_frames, 6] (global + jaw)
                shape = flame_data['shape']      # Shape: [n_frames, 100/300] (betas)
                
                # Get sequence length
                seq_length = expressions.shape[0]
                
                # Process head pose (positions 0:3 in pose)
                head_pose = pose[:, :3]  # Global head orientation in axis-angle
                head_pose_6d = axis_angle_to_6d_np(head_pose).reshape(seq_length, 6)
                
                # Process jaw pose (positions 3:6 in pose)
                jaw_pose = pose[:, 3:6]  # Jaw rotation in axis-angle
                jaw_pose_6d = axis_angle_to_6d_np(jaw_pose).reshape(seq_length, 6)
                
                # Pad expressions to 100 dimensions as needed
                if expressions.shape[1] < 100:
                    padded_exps = np.zeros((seq_length, 100), dtype=expressions.dtype)
                    padded_exps[:, :expressions.shape[1]] = expressions
                    expressions = padded_exps
                elif expressions.shape[1] > 100:
                    expressions = expressions[:, :100]
                
                # Concatenate head pose, jaw pose and expressions for 112D face representation
                face_features = np.concatenate([head_pose_6d, jaw_pose_6d, expressions], axis=1)
                
                if seq_length <= max_segment_length:
                    # Short enough to use as a single segment
                    sequence_id_name = sequence_id.replace('/', '_')
                    segment_name = f'youtube_talking_face_{sequence_id_name}'
                    self.data_dict_youtube_talking[segment_name] = {
                        'face': face_features,
                        'shape': shape,
                        'pose': pose[:, :3],  # Global head orientation
                        'id': sequence_id,
                        'dataset_name': 'YouTube_Talking'
                    }
                    self.metadata_youtube_talking.append(segment_name)
                else:
                    # Split into overlapping segments
                    num_segments = (seq_length - max_segment_length) // stride + 1
                    for seg_idx in range(num_segments):
                        start_idx = seg_idx * stride
                        end_idx = min(start_idx + max_segment_length, seq_length)
                        sequence_id_name = sequence_id.replace('/', '_')
                        segment_name = f'youtube_talking_face_{sequence_id_name}_{seg_idx}'
                        self.data_dict_youtube_talking[segment_name] = {
                            'face': face_features[start_idx:end_idx],
                            'shape': shape[start_idx:end_idx],
                            'pose': pose[start_idx:end_idx, :3],  # Global head orientation
                            'id': sequence_id,
                            'dataset_name': 'YouTube_Talking'
                        }
                        self.metadata_youtube_talking.append(segment_name)
            except Exception as e:
                print(f"Error processing sequence {sequence_id}: {str(e)}")
                continue

            # For fast debug
            if len(self.metadata_youtube_talking) >= self.maxdata:
                break
        
        # Save processed data to cache
        self._save_to_cache(cache_path, self.data_dict_youtube_talking, self.metadata_youtube_talking, "YouTube_Talking_face")

    def _load_youtube_talking_synthetic(self, config):
        """
        Load facial expressions from the YouTube_Talking dataset.
        """
        # Get cache path for this specific dataset
        data_root = self.args["YouTube_Talking_Synthetic"].ROOT
        cache_path = self._get_cache_path(data_root, "YouTube_Talking_Synthetic_face")
        
        # Check if we should load from cache
        data_dict, metadata = self._load_from_cache(cache_path, "YouTube_Talking_Synthetic_face")
        if data_dict is not None:
            self.data_dict_youtube_talking_synthetic = data_dict
            self.metadata_youtube_talking_synthetic = metadata
            return
        
        max_segment_length = config.pose_length
        stride = config.stride
        
        print(f"Processing YouTube_Talking_Synthetic dataset for face expressions...")
        
        self.data_root_youtube_talking_synthetic = data_root
        split_file = pjoin(data_root, self.split + '.txt')
        # Data id list
        id_list_youtube_talking = []
        with cs.open(split_file, "r") as f:
            for line in f.readlines():
                id_list_youtube_talking.append(line.strip())

        self.data_dict_youtube_talking_synthetic = {}
        self.metadata_youtube_talking_synthetic = []
        
        # Determine train/val/test split (assuming 80/10/10)
        total_seqs = len(id_list_youtube_talking)
        keys = list(id_list_youtube_talking)

        
        for sequence_id in tqdm(id_list_youtube_talking, desc=f"Processing YouTube_Talking_Synthetic {self.split} sequences"):
            try:
                # Process each participant in the conversation
                flame_path = pjoin(self.data_root_youtube_talking_synthetic, 'FLAME_coeffs_25', sequence_id + '.npz')
                flame_data = np.load(flame_path)
                
                # Extract expression and pose parameters
                expressions = flame_data['exp']  # Shape: [n_frames, 50]
                pose = flame_data['pose']        # Shape: [n_frames, 6] (global + jaw)
                shape = flame_data['shape']      # Shape: [n_frames, 100/300] (betas)
                
                # Get sequence length
                seq_length = expressions.shape[0]
                
                # Process head pose (positions 0:3 in pose)
                head_pose = pose[:, :3]  # Global head orientation in axis-angle
                head_pose_6d = axis_angle_to_6d_np(head_pose).reshape(seq_length, 6)
                
                # Process jaw pose (positions 3:6 in pose)
                jaw_pose = pose[:, 3:6]  # Jaw rotation in axis-angle
                jaw_pose_6d = axis_angle_to_6d_np(jaw_pose).reshape(seq_length, 6)
                
                # Pad expressions to 100 dimensions as needed
                if expressions.shape[1] < 100:
                    padded_exps = np.zeros((seq_length, 100), dtype=expressions.dtype)
                    padded_exps[:, :expressions.shape[1]] = expressions
                    expressions = padded_exps
                elif expressions.shape[1] > 100:
                    expressions = expressions[:, :100]
                
                # Concatenate head pose, jaw pose and expressions for 112D face representation
                face_features = np.concatenate([head_pose_6d, jaw_pose_6d, expressions], axis=1)
                
                if seq_length <= max_segment_length:
                    # Short enough to use as a single segment
                    sequence_id_name = sequence_id.replace('/', '_')
                    segment_name = f'youtube_talking_synthetic_face_{sequence_id_name}'
                    self.data_dict_youtube_talking_synthetic[segment_name] = {
                        'face': face_features,
                        'shape': shape,
                        'pose': pose[:, :3],  # Global head orientation
                        'id': sequence_id,
                        'dataset_name': 'YouTube_Talking'
                    }
                    self.metadata_youtube_talking_synthetic.append(segment_name)
                else:
                    # Split into overlapping segments
                    num_segments = (seq_length - max_segment_length) // stride + 1
                    for seg_idx in range(num_segments):
                        start_idx = seg_idx * stride
                        end_idx = min(start_idx + max_segment_length, seq_length)
                        sequence_id_name = sequence_id.replace('/', '_')
                        segment_name = f'youtube_talking_synthetic_face_{sequence_id_name}_{seg_idx}'
                        self.data_dict_youtube_talking_synthetic[segment_name] = {
                            'face': face_features[start_idx:end_idx],
                            'shape': shape[start_idx:end_idx],
                            'pose': pose[start_idx:end_idx, :3],  # Global head orientation
                            'id': sequence_id,
                            'dataset_name': 'YouTube_Talking'
                        }
                        self.metadata_youtube_talking_synthetic.append(segment_name)
            except Exception as e:
                print(f"Error processing sequence {sequence_id}: {str(e)}")
                continue

            # For fast debug
            if len(self.metadata_youtube_talking_synthetic) >= self.maxdata:
                break
        
        # Save processed data to cache
        self._save_to_cache(cache_path, self.data_dict_youtube_talking_synthetic, self.metadata_youtube_talking_synthetic, "YouTube_Talking_Synthetic_face")


    def comput_foot_contacts(self, m_data):
        """
        Compute foot contacts from motion data.
        This method requires SMPLX, so we ensure it's initialized.
        
        Parameters:
        - m_data: Motion data dictionary containing betas, poses, trans, and expressions
        
        Returns:
        - contacts: Binary array indicating foot contacts (1 for contact, 0 for no contact)
        """
        # Make sure SMPLX is initialized
        self._initialize_smplx_if_needed()
        
        # Extract motion components
        betas, poses, trans, exps = m_data["betas"], m_data["poses"], m_data["trans"], m_data["expressions"]
        n, c = poses.shape[0], poses.shape[1]  # n: number of frames, c: pose dimension
        
        # Determine the dimension of betas
        beta_dim = betas.shape[-1]  # get the last dimension of betas

        if beta_dim == 16:
            # AMASS dataset has 16 beta parameters, need to pad to 300
            padded_betas = np.zeros(300)
            padded_betas[:16] = betas
            exps = torch.zeros([n, 100], dtype=torch.float32).cuda()  # AMASS dataset has no expressions
        else:
            # BEAT2 dataset already has 300 beta parameters
            padded_betas = betas
            exps = torch.from_numpy(m_data["expressions"]).cuda().float()  # BEAT2 dataset has expressions

        # Prepare betas for all frames
        betas = padded_betas.reshape(1, 300)  # can be 16 or 300
        betas = np.tile(betas, (n, 1))

        # Convert all data to tensors on GPU
        betas = torch.from_numpy(betas).cuda().float()
        poses = torch.from_numpy(poses.reshape(n, c)).cuda().float()
        trans = torch.from_numpy(trans.reshape(n, 3)).cuda().float()
        
        # Process in batches to avoid memory issues
        max_length = 128
        s, r = n // max_length, n % max_length  # s: number of full batches, r: remainder
        all_tensor = []
        
        # Process full batches
        for i in range(s):
            with torch.no_grad():
                # Run SMPLX forward pass to get joint positions
                joints = self.smplx_2020(
                    betas=betas[i * max_length:(i + 1) * max_length],
                    transl=trans[i * max_length:(i + 1) * max_length],
                    expression=exps[i * max_length:(i + 1) * max_length],
                    jaw_pose=poses[i * max_length:(i + 1) * max_length, 66:69],
                    global_orient=poses[i * max_length:(i + 1) * max_length, :3],
                    body_pose=poses[i * max_length:(i + 1) * max_length, 3:21 * 3 + 3],
                    left_hand_pose=poses[i * max_length:(i + 1) * max_length, 25 * 3:40 * 3],
                    right_hand_pose=poses[i * max_length:(i + 1) * max_length, 40 * 3:55 * 3],
                    return_verts=True,
                    return_joints=True,
                    leye_pose=poses[i * max_length:(i + 1) * max_length, 69:72],
                    reye_pose=poses[i * max_length:(i + 1) * max_length, 72:75],
                )['joints'][:, (7, 8, 10, 11), :].reshape(max_length, 4, 3).cpu()  # Extract foot joints
            all_tensor.append(joints)
            
        # Process remainder frames if any
        if r != 0:
            with torch.no_grad():
                joints = self.smplx_2020(
                    betas=betas[s * max_length:s * max_length + r],
                    transl=trans[s * max_length:s * max_length + r],
                    expression=exps[s * max_length:s * max_length + r],
                    jaw_pose=poses[s * max_length:s * max_length + r, 66:69],
                    global_orient=poses[s * max_length:s * max_length + r, :3],
                    body_pose=poses[s * max_length:s * max_length + r, 3:21 * 3 + 3],
                    left_hand_pose=poses[s * max_length:s * max_length + r, 25 * 3:40 * 3],
                    right_hand_pose=poses[s * max_length:s * max_length + r, 40 * 3:55 * 3],
                    return_verts=True,
                    return_joints=True,
                    leye_pose=poses[s * max_length:s * max_length + r, 69:72],
                    reye_pose=poses[s * max_length:s * max_length + r, 72:75],
                )['joints'][:, (7, 8, 10, 11), :].reshape(r, 4, 3).cpu()
            all_tensor.append(joints)
            
        # Concatenate all batches
        joints = torch.cat(all_tensor, axis=0)  # all, 4, 3
        
        # Calculate foot velocities
        feetv = torch.zeros(joints.shape[1], joints.shape[0])
        joints = joints.permute(1, 0, 2)  # 4, all, 3
        feetv[:, :-1] = (joints[:, 1:] - joints[:, :-1]).norm(dim=-1)
        
        # Determine contacts based on velocity threshold
        contacts = (feetv < 0.01).numpy().astype(float)  # Contact when velocity < 0.01
        contacts = contacts.transpose(1, 0)  # all, 4

        return contacts


    def __len__(self):
        """Returns the number of samples in the dataset."""
        return len(self.metadata)
        
    def __getitem__(self, item):
        """
        Retrieves a sample from the dataset based on the given index.
        Converts NumPy arrays to PyTorch tensors for model usage.
        """
        dataset_name = self.metadata[item]
        data = self.data_dict[dataset_name]
        
        # Create a copy of the data dictionary to avoid modifying the original
        formatted_data = {}
        
        # Convert NumPy arrays to PyTorch tensors for model usage
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                # Convert NumPy arrays to tensors
                formatted_data[key] = torch.from_numpy(value).float()
            else:
                # Keep other data types as is
                formatted_data[key] = value
                
        # Add additional information
        motion_len = formatted_data['lower'].shape[0]
        formatted_data.update({
            "id_name": formatted_data.get('id', ""),
            "dataset_name": formatted_data.get('dataset_name', ""),
            "split_name": "vae",
            "select_part": "global",
            "motion_len": motion_len,
        })
        
        return formatted_data

# Note: face_collate_fn has been moved to conver_agent/data/utils.py 