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
    JOINT_MASK_FACE,
    BEAT_SMPLX_FACE,
)
from conver_agent.utils.rotation_conversions import axis_angle_to_6d, axis_angle_to_matrix, rotation_6d_to_axis_angle, axis_angle_to_6d_np
import pandas as pd
import codecs as cs


class FaceVQDataset(data.Dataset):
    """
    Dataset for processing facial expression data for the VAE/VQ-VAE training stage.
    Follows the pattern of MixedDatasetVQ but focuses on facial expressions.
    """
    
    def __init__s(
        self,
        args,
        dataset_configs,
        split,
        tiny=False,
        debug=False,
        stage='vae',
        use_cache=False,
        save_cache=False,
        cache_format="pkl",
        **kwargs,
    ):
        """
        Initialize the face dataset.
        
        Args:
            args: Arguments containing dataset paths and configurations
            dataset_configs: List of configurations for different datasets
            split: 'train', 'val', or 'test'
            tiny: Whether to use a small subset for debugging
            debug: If True, enables debug mode
            stage: Specifies the training stage
            use_cache: Whether to load data from cache when available
            save_cache: Whether to save processed data to cache
            cache_format: Format to use for caching ("h5", "npz", or "pkl")
        """
        # Set max data size depending on debug mode
        if tiny or debug:
            self.maxdata = 100
        else:
            self.maxdata = 1e10

        self.args = args
        self.dataset_configs = dataset_configs
        self.stage = stage
        self.split = split
        self.use_cache = use_cache
        self.save_cache = save_cache
        self.cache_format = cache_format
        
        # Configure face-specific parameters
        self.face_exp_dim = 100  # Expression parameters dimension, padded to 100
        self.face_jaw_dim = 6    # 6D rotation representation for jaw (3 DoF)
        
        # Dictionary to store data and metadata
        self.data_dict = {}
        self.metadata = []
        
        # Load each dataset based on its type from the configuration
        for config in dataset_configs:
            dataset_name = config.get("name")
            
            if dataset_name == "BEAT2":
                self._load_beat2(config)
                self.data_dict.update(self.data_dict_beat2)
                self.metadata.extend(self.metadata_beat2)
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
                
        print(f"[FaceVQDataset] Loaded {len(self.metadata)} samples")
        
    def _get_cache_path(self, dataset_path, dataset_type):
        """Generate a cache file path based on dataset path and type."""
        cache_dir = os.path.join(dataset_path, 'cache')
        config_str = f"{dataset_type}_face_{self.split}_vq"
        
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
            
            print(f"Saving {dataset_name} face dataset to cache: {cache_path}")
            
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
                
            print(f"Successfully saved {dataset_name} face cache to {cache_path}")
            
        except Exception as e:
            print(f"Error saving {dataset_name} face cache: {str(e)}")
            import traceback
            traceback.print_exc()
            
    def _load_from_cache(self, cache_path, dataset_name):
        """Load processed data from cache."""
        if not self.use_cache:
            return None, None
        
        if not os.path.exists(cache_path):
            print(f"Face cache file not found: {cache_path}")
            return None, None
        
        try:
            print(f"Loading {dataset_name} face dataset from cache: {cache_path}")
            
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
            
            print(f"Successfully loaded {dataset_name} face cache from {cache_path}")
            return data_dict, metadata
            
        except Exception as e:
            print(f"Error loading {dataset_name} face cache: {str(e)}")
            import traceback
            traceback.print_exc()
            return None, None
            
    def _load_beat2(self, config):
        """
        Load facial expressions from the BEAT2 dataset.
        """
        # Get cache path for this specific dataset
        data_root = self.args["BEAT2"].ROOT
        cache_path = self._get_cache_path(data_root, "BEAT2_face")
        
        # Check if we should load from cache
        data_dict, metadata = self._load_from_cache(cache_path, "BEAT2_face")
        if data_dict is not None:
            self.data_dict_beat2 = data_dict
            self.metadata_beat2 = metadata
            return
        
        print(f"Processing BEAT2 dataset for face expressions...")
        
        self.data_root_beat2 = data_root
        self.ori_length = config.pose_length
        additional_data = config.additional_data
        training_speakers = config.training_speakers
        pose_fps_beat2 = config.pose_fps
        pose_rep = config.pose_rep

        # Load split rules
        split_rule = pd.read_csv(pjoin(data_root, "train_test_split.csv"))
        
        # Filter based on training speakers
        if self.split == 'token':
            self.selected_file = split_rule.loc[
                ((split_rule['id'].str.split("_").str[0].astype(int).isin(training_speakers)))
            ]
        else:
            self.selected_file = split_rule.loc[
                (split_rule['type'] == self.split) &
                ((split_rule['id'].str.split("_").str[0].astype(int).isin(training_speakers)))
            ]
            if additional_data:
                split_b = split_rule.loc[
                    (split_rule['type'] == 'additional') & 
                    (split_rule['id'].str.split("_").str[0].astype(int).isin(training_speakers))
                ]
                self.selected_file = pd.concat([self.selected_file, split_b])

        self.data_dict_beat2 = {}
        self.metadata_beat2 = []
        
        # Process each file
        for index, file_name in tqdm(self.selected_file.iterrows()):
            f_name = file_name["id"]
            pose_file = pjoin(self.data_root_beat2, pose_rep, f_name + ".npz")
            
            try:
                # Load pose data
                pose_data = np.load(pose_file, allow_pickle=True)
                poses = pose_data["poses"]
                n, c = poses.shape[0], poses.shape[1]
                trans = pose_data["trans"]
                betas = pose_data["betas"]
                betas = np.repeat(pose_data["betas"].reshape(1, 300), poses.shape[0], axis=0)
                expressions = pose_data["expressions"]
                
                # Extract and convert jaw pose data
                tar_pose_jaw = poses[:, 66:69]  # Jaw rotation in axis-angle
                tar_pose_jaw_6d = axis_angle_to_6d_np(tar_pose_jaw).reshape(n, 6)

                # Extract and convert head (global) pose data 
                tar_pose_head = poses[:, 45:48]  # Global head orientation in axis-angle
                tar_pose_head_6d = axis_angle_to_6d_np(tar_pose_head).reshape(n, 6)
                
                # Ensure expression has 100 dimensions
                if expressions.shape[1] < 100:
                    padded_exps = np.zeros((n, 100), dtype=expressions.dtype)
                    padded_exps[:, :expressions.shape[1]] = expressions
                    expressions = padded_exps
                elif expressions.shape[1] > 100:
                    expressions = expressions[:, :100]
                
                # Concatenate head pose, jaw pose and expressions for 112D face representation
                tar_pose_face = np.concatenate([tar_pose_head_6d, tar_pose_jaw_6d, expressions], axis=1)
                
                # Calculate segments
                round_seconds_skeleton = tar_pose_face.shape[0] // pose_fps_beat2
                if round_seconds_skeleton == 0:
                    round_seconds_skeleton = 1
                clip_s_t, clip_e_t = 0, round_seconds_skeleton - 0
                clip_s_f_pose, clip_e_f_pose = clip_s_t * pose_fps_beat2, clip_e_t * pose_fps_beat2
            
                if self.split == 'test' or self.split == 'token':  # stride = length for test
                    cut_length = clip_e_f_pose - clip_s_f_pose
                    stride = cut_length
                else:
                    stride = int(config.stride)
                    cut_length = int(self.ori_length)
                
                num_subdivision = math.floor((clip_e_f_pose - clip_s_f_pose - cut_length) / stride) + 1
                
                # Create segments
                for i in range(num_subdivision):
                    start_idx = clip_s_f_pose + i * stride
                    fin_idx = start_idx + cut_length
                    
                    # Skip if out of bounds
                    if fin_idx > tar_pose_face.shape[0]:
                        continue
                        
                    sample_face = tar_pose_face[start_idx:fin_idx]
                    sample_shape = betas[start_idx:fin_idx]
                    sample_global_pose = poses[start_idx:fin_idx, :3]  # Global head orientation
                    
                    new_name = 'beat2_face_' + '%s_%d' % (f_name, i)
                    
                    # Store processed data
                    self.data_dict_beat2[new_name] = {
                        'face': sample_face,
                        'shape': sample_shape,
                        'pose': sample_global_pose,
                        'id': f_name,
                        'dataset_name': 'beat2',
                    }
                    self.metadata_beat2.append(new_name)
            except Exception as e:
                print(f"Error processing file {f_name}: {str(e)}")
                continue
                
            # For fast debug
            if index >= self.maxdata:
                break
        
        # Save processed data to cache
        self._save_to_cache(cache_path, self.data_dict_beat2, self.metadata_beat2, "BEAT2_face")
    
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
        motion_len = formatted_data['face'].shape[0]
        formatted_data.update({
            "id_name": formatted_data.get('id', ""),
            "dataset_name": formatted_data.get('dataset_name', ""),
            "split_name": "vq",
            "motion_len": motion_len,
        })
        
        return formatted_data

# Note: face_collate_fn has been moved to conver_agent/data/utils.py 