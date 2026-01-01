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
import h5py
from .utils.split_transcript import split_and_merge_sentences
import smplx
from tqdm import tqdm
from .data_tools import joints_list
from smplx import FLAME
from multimodal_tokenizers.data.mixed_dataset.data_tools import (
    joints_list, 
    JOINT_MASK_FACE,
    JOINT_MASK_UPPER,
    JOINT_MASK_HANDS,
    JOINT_MASK_LOWER,
    JOINT_MASK_FULL,
    BEAT_SMPLX_JOINTS,
    BEAT_SMPLX_FULL,
    BEAT_SMPLX_FACE,
    BEAT_SMPLX_UPPER,
    BEAT_SMPLX_HANDS,
    BEAT_SMPLX_LOWER
)
from multimodal_tokenizers.utils.rotation_conversions import axis_angle_to_6d, axis_angle_to_matrix, rotation_6d_to_axis_angle, axis_angle_to_6d_np

class MixedDatasetVQ(data.Dataset):
    def __init__(
        self,
        args,
        dataset_configs,
        split,
        # unit_length=4,
        # fps=20,
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
            # self.maxdata = 10
            self.maxdata = 1e10
        else:
            self.maxdata = 1e10
            # self.maxdata = 2

        self.args = args
        self.dataset_configs = dataset_configs
        self.task_path = task_path
        self.stage = stage
        self.split = split
        self.njoints = njoints
        self.use_cache = use_cache
        self.save_cache = save_cache
        self.cache_format = cache_format
        self.smpl_path = smpl_path  # Store the path for later use if needed
        self.motion_representation = motion_representation

        # Store kwargs for SMPLX initialization if needed
        if motion_representation == "rotation":
            # self.select_type = kwargs['cfg']['Selected_type']
            # self.select_part = kwargs['cfg']["Representation_type"].get(self.select_type)
            # Initialize the joint masks for different body parts
            self.joint_mask_upper = JOINT_MASK_UPPER
            self.joint_mask_lower = JOINT_MASK_LOWER
            self.joint_mask_hand = JOINT_MASK_HANDS
            self.joint_mask_face = JOINT_MASK_FACE
            self.joint_mask_full = JOINT_MASK_FULL
            # We'll initialize SMPLX only when needed, not upfront
            self.smplx_2020 = None
        elif motion_representation == "h3d":
            self.h3d_mean = mean
            self.h3d_std = std

        # Dictionary to store data and metadata
        self.data_dict = {}
        self.metadata = []

        # Load each dataset based on its type from the configuration
        for config in dataset_configs:
            # dataset_name = config.get("type")
            dataset_name = config.get("name")

            if dataset_name == "amass_h3d":
                self._load_amass_h3d(config)
                self.data_dict.update(self.data_dict_amass_h3d)
                self.metadata.extend(self.metadata_amass_h3d)
            elif dataset_name == "AMASS":
                self._load_amass(config)
                self.data_dict.update(self.data_dict_amass)
                self.metadata.extend(self.metadata_amass)
            elif dataset_name == "AMASS_talking":
                self._load_amass_talking(config)
                self.data_dict.update(self.data_dict_amass_talking)
                self.metadata.extend(self.metadata_amass_talking)
            elif dataset_name == "AMASS_P2P":
                self._load_amass_p2p(config)
                self.data_dict.update(self.data_dict_amass_p2p)
                self.metadata.extend(self.metadata_amass_p2p)
            elif dataset_name == "BEAT2":
                # if self.split == 'test':
                #     continue
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
            elif dataset_name == "YouTube_Talking_Upper":
                self._load_youtube_talking_upper(config)
                self.data_dict.update(self.data_dict_youtube_talking_upper)
                self.metadata.extend(self.metadata_youtube_talking_upper)
            elif dataset_name == "YouTube_Talking_Face":
                self._load_youtube_talking_face(config)
                self.data_dict.update(self.data_dict_youtube_talking_face)
                self.metadata.extend(self.metadata_youtube_talking_face)
            elif dataset_name == "YouTube_Talking_Synthetic":
                self._load_youtube_talking_synthetic(config)
                self.data_dict.update(self.data_dict_youtube_talking_synthetic)
                self.metadata.extend(self.metadata_youtube_talking_synthetic)
            else:
                raise NotImplementedError(f"Unknown dataset name {dataset_name}")

        print(len(self.metadata))

    def _get_cache_path(self, dataset_path, dataset_type):
        """
        Generate a cache file path based on dataset path and type.
        
        Parameters:
        - dataset_path: Path to the original dataset
        - dataset_type: Type of the dataset (e.g., 'beat2', 'amass')
        
        Returns:
        - Path to the cache file
        """
        # Use the dataset's own directory for caching
        cache_dir = os.path.join(dataset_path, 'cache')
        
        # Generate a unique filename based on dataset configuration
        # Use only attributes we know are available
        config_str = f"{dataset_type}_{self.split}_vq"
        
        # Set the file extension based on the cache format
        if self.cache_format == "h5":
            ext = ".h5"
        elif self.cache_format == "npz":
            ext = ".npz"
        else:  # Default to pickle
            ext = ".pkl"
            
        cache_path = os.path.join(cache_dir, f"{config_str}{ext}")
        return cache_path

    def _save_to_cache(self, cache_path, data_dict, metadata, dataset_name):
        """
        Save processed data to cache.
        
        Parameters:
        - cache_path: Path where to save the cache file
        - data_dict: Dictionary containing processed data (NumPy arrays)
        - metadata: List of metadata entries
        - dataset_name: Type of the dataset (for logging)
        """
        if not self.save_cache:
            return
            
        try:
            # Create cache directory if it doesn't exist
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            
            print(f"Saving {dataset_name} dataset to cache: {cache_path}")
            
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
                # Simple pickle save
                with open(cache_path, 'wb') as f:
                    pickle.dump({
                        'data_dict': numpy_data_dict,
                        'metadata': metadata
                    }, f, protocol=pickle.HIGHEST_PROTOCOL)
                
            elif self.cache_format == "npz":
                # Faster uncompressed saving
                np.savez(cache_path, 
                    data_dict=numpy_data_dict, 
                    metadata=metadata
                )
                
            elif self.cache_format == "h5":
                # Keep existing HDF5 implementation
                with h5py.File(cache_path, 'w') as f:
                    # Save metadata as a JSON string
                    metadata_str = json.dumps(metadata)
                    # Convert metadata to a list of bytes to ensure it's not treated as a scalar
                    metadata_bytes = np.frombuffer(metadata_str.encode('utf-8'), dtype='uint8')
                    # Now we can safely use compression since it's definitely an array
                    f.create_dataset('metadata', data=metadata_bytes, compression='gzip')
                    
                    # Create a group for data_dict
                    data_group = f.create_group('data_dict')
                    
                    # Save each entry in numpy_data_dict
                    for key, item in numpy_data_dict.items():
                        # Create a group for each item (entry)
                        item_group = data_group.create_group(key)
                        
                        # Save each attribute of the item
                        for attr_key, attr_value in item.items():
                            if attr_value is None:
                                # Store None as a special value
                                item_group.attrs[attr_key] = 'None'
                            elif isinstance(attr_value, (str, int, float, bool)):
                                # Store simple types as attributes
                                item_group.attrs[attr_key] = attr_value
                            elif isinstance(attr_value, dict):
                                # Store dictionaries as JSON-encoded strings
                                attr_group = item_group.create_group(attr_key)
                                for k, v in attr_value.items():
                                    if isinstance(v, (str, int, float, bool, list)):
                                        attr_group.attrs[k] = json.dumps(v)
                            elif isinstance(attr_value, np.ndarray):
                                # Only apply compression to non-scalar arrays
                                if attr_value.size > 1:
                                    item_group.create_dataset(
                                        attr_key, 
                                        data=attr_value,
                                        compression='gzip', 
                                        compression_opts=4
                                    )
                                else:
                                    # For scalar arrays, store without compression
                                    item_group.create_dataset(attr_key, data=attr_value)
                            elif attr_key == 'tasks' or attr_key == 'emotion_label':
                                # Handle special cases like tasks
                                item_group.attrs[attr_key] = json.dumps(attr_value)
                            else:
                                # Try to convert other types to strings
                                try:
                                    item_group.attrs[attr_key] = str(attr_value)
                                except:
                                    print(f"Warning: Could not save attribute {attr_key} of type {type(attr_value)}")
                
            print(f"Successfully saved {dataset_name} cache to {cache_path}")
            
        except Exception as e:
            print(f"Error saving {dataset_name} cache: {str(e)}")
            # Print more details about the error for debugging
            import traceback
            traceback.print_exc()

    def _load_from_cache(self, cache_path, dataset_name):
        """
        Load processed data from cache.
        
        Parameters:
        - cache_path: Path to the cache file
        - dataset_name: Type of the dataset (for logging)
        
        Returns:
        - tuple (data_dict, metadata) if successful, (None, None) otherwise
        """
        if not self.use_cache:
            return None, None
        
        if not os.path.exists(cache_path):
            print(f"Cache file not found: {cache_path}")
            return None, None
        
        try:
            print(f"Loading {dataset_name} dataset from cache: {cache_path}")
            
            if self.cache_format == "pkl":
                # Simple pickle load
                with open(cache_path, 'rb') as f:
                    cached_data = pickle.load(f)
                    data_dict = cached_data['data_dict']
                    metadata = cached_data['metadata']
                    
            elif self.cache_format == "npz":
                # Ultra-simple NPZ load - just two objects
                loaded = np.load(cache_path, allow_pickle=True)
                
                # More robust metadata loading - handle both direct arrays and object arrays
                if loaded['metadata'].size == 1:
                    # Try to extract as a single object
                    metadata = loaded['metadata'].item()
                else:
                    # Load as a direct array (e.g., list of strings)
                    metadata = loaded['metadata']
                
                # More robust data_dict loading
                if loaded['data_dict'].size == 1:
                    # Normal case - data_dict is stored as an object array
                    data_dict = loaded['data_dict'].item()
                else:
                    # Handle edge case where it might be stored differently
                    print(f"Warning: Unexpected data_dict structure in {cache_path}")
                    data_dict = {}
            
            elif self.cache_format == "h5":
                # Keep existing HDF5 implementation
                data_dict = {}
                with h5py.File(cache_path, 'r') as f:
                    # Load metadata
                    try:
                        # Try to load metadata as a byte array
                        metadata_bytes = f['metadata'][()]
                        metadata_str = metadata_bytes.tobytes().decode('utf-8')
                        metadata = json.loads(metadata_str)
                    except:
                        # Fallback: try loading directly as a string attribute
                        try:
                            metadata_str = f.attrs.get('metadata', '[]')
                            metadata = json.loads(metadata_str)
                        except:
                            # Last resort: just create an empty list
                            print(f"Warning: Could not load metadata from {cache_path}")
                            metadata = []
                    
                    # Load data_dict
                    data_group = f['data_dict']
                    for key in data_group.keys():
                        item_group = data_group[key]
                        item = {}
                        
                        # Load each attribute
                        for attr_key in item_group.keys():
                            try:
                                attr_value = item_group[attr_key][()]
                                item[attr_key] = attr_value
                            except:
                                print(f"Warning: Could not load attribute {attr_key} from {cache_path}")
                        
                        # Load attributes stored as metadata
                        for attr_key in item_group.attrs.keys():
                            attr_value = item_group.attrs[attr_key]
                            if attr_value == 'None':
                                item[attr_key] = None
                            elif attr_key in ['tasks', 'emotion_label'] or isinstance(attr_value, str) and attr_value.startswith('{') and attr_value.endswith('}'):
                                # Try to parse as JSON
                                try:
                                    item[attr_key] = json.loads(attr_value)
                                except:
                                    item[attr_key] = attr_value
                            else:
                                item[attr_key] = attr_value
                        
                        data_dict[key] = item
            
            print(f"Successfully loaded {dataset_name} cache from {cache_path}")
            return data_dict, metadata
            
        except Exception as e:
            print(f"Error loading {dataset_name} cache: {str(e)}")
            # Print more details about the error for debugging
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

    def _initialize_flame_if_needed(self):
        """
        Initialize the FLAME model if it hasn't been initialized yet.
        This is done lazily to avoid unnecessary GPU memory usage when loading from cache.
        """
        if self.flame is None:
            print("Initializing SMPLX model for data processing...")
            self.flame = smplx.create(self.smpl_path,
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
        
        # data_root = config.get("data_root")
        self.data_root_beat2 = data_root
        self.ori_length = config.pose_length
        additional_data = config.additional_data
        training_speakers = config.training_speakers
        pose_rep = config.pose_rep
        pose_rep_mirror = config.pose_rep_mirror
        pose_fps_beat2 = config.pose_fps
        foot_contact_path = config.foot_contact_path
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
            pose_file_mirror = pjoin(self.data_root_beat2, pose_rep_mirror, "M_" + f_name + ".npz")
            # try:
            # Load pose data
            pose_data = np.load(pose_file, allow_pickle=True)
            pose_data_mirror = np.load(pose_file_mirror, allow_pickle=True)
            poses = pose_data["poses"]
            poses_mirror = pose_data_mirror["poses"]
            n, c = poses.shape[0], poses.shape[1]
            trans = pose_data["trans"]
            trans_mirror = pose_data_mirror["trans"]
            betas = pose_data["betas"]
            betas = np.repeat(pose_data["betas"].reshape(1, 300), poses.shape[0], axis=0)
            expressions = pose_data["expressions"]
            expressions_mirror = pose_data_mirror["expressions"]
            # Process pose data
            pose_processed = poses * JOINT_MASK_FULL
            pose_processed = pose_processed[:, JOINT_MASK_FULL.astype(bool)]
            pose_processed_mirror = poses_mirror * JOINT_MASK_FULL
            pose_processed_mirror = pose_processed_mirror[:, JOINT_MASK_FULL.astype(bool)]
            if self.motion_representation == 'rotation':
                # contacts = self.comput_foot_contacts(pose_data)
                # Calculate foot contacts using existing function
                foot_contacts_path = pjoin(self.data_root_beat2, foot_contact_path, f_name + '.npy')
                foot_contacts_path_mirror = pjoin(self.data_root_beat2, foot_contact_path, "M_" + f_name + '.npy')
                if os.path.exists(foot_contacts_path):
                    contacts = np.load(foot_contacts_path)
                    contacts_mirror = np.load(foot_contacts_path_mirror)
                else:
                    contacts = self.comput_foot_contacts(pose_data)
                    contacts_mirror = self.comput_foot_contacts(pose_data_mirror)
                    os.makedirs(pjoin(self.data_root_beat2, foot_contact_path), exist_ok=True)
                    np.save(foot_contacts_path, contacts)
                    np.save(foot_contacts_path_mirror, contacts_mirror)
                pose_processed = np.concatenate([pose_processed, contacts], axis=1)
                pose_processed_mirror = np.concatenate([pose_processed_mirror, contacts_mirror], axis=1)
            tar_pose = pose_processed[ :, :165]
            tar_pose_mirror = pose_processed_mirror[ :, :165]
            tar_contact = pose_processed[ :, 165:169]
            tar_contact_mirror = pose_processed_mirror[ :, 165:169]
            tar_exps = expressions
            tar_exps_mirror = expressions_mirror
            tar_trans = trans
            tar_trans_mirror = trans_mirror

            # Extract and convert jaw pose data
            tar_pose_jaw = tar_pose[:, 66:69]
            tar_pose_jaw_6d = axis_angle_to_6d_np(tar_pose_jaw).reshape(n, 6)
            tar_pose_jaw_mirror = tar_pose_mirror[:, 66:69]
            tar_pose_jaw_6d_mirror = axis_angle_to_6d_np(tar_pose_jaw_mirror).reshape(n, 6)
            
            # Extract and convert head (global) pose data 
            tar_pose_head = tar_pose[:, 45:48]  # Global head orientation in axis-angle
            tar_pose_head_mirror = tar_pose_mirror[:, 45:48]
            tar_pose_head_6d = axis_angle_to_6d_np(tar_pose_head).reshape(n, 6)
            tar_pose_head_6d_mirror = axis_angle_to_6d_np(tar_pose_head_mirror).reshape(n, 6)
            # Concatenate jaw pose and expressions for face
            tar_pose_face = np.concatenate([tar_pose_jaw_6d, tar_exps], axis=1)
            tar_pose_face_mirror = np.concatenate([tar_pose_jaw_6d_mirror, tar_exps_mirror], axis=1)
            tar_pose_face_with_head = np.concatenate([tar_pose_head_6d, tar_pose_jaw_6d, expressions], axis=1)
            tar_pose_face_with_head_mirror = np.concatenate([tar_pose_head_6d_mirror, tar_pose_jaw_6d_mirror, expressions_mirror], axis=1)


            # Extract and convert hand pose data
            tar_pose_hands = tar_pose[:, 25 * 3:55 * 3].reshape(n, 30, 3)
            tar_pose_hands_mirror = tar_pose_mirror[:, 25 * 3:55 * 3].reshape(n, 30, 3)
            tar_pose_hands_6d = axis_angle_to_6d_np(tar_pose_hands).reshape(n, 30 * 6)
            tar_pose_hands_6d_mirror = axis_angle_to_6d_np(tar_pose_hands_mirror).reshape(n, 30 * 6)
            # Extract and convert upper body pose data
            tar_pose_upper = tar_pose[:, self.joint_mask_upper.astype(bool)].reshape(n, 13, 3)
            tar_pose_upper_mirror = tar_pose_mirror[:, self.joint_mask_upper.astype(bool)].reshape(n, 13, 3)
            tar_pose_upper_6d = axis_angle_to_6d_np(tar_pose_upper).reshape(n, 13 * 6)
            tar_pose_upper_6d_mirror = axis_angle_to_6d_np(tar_pose_upper_mirror).reshape(n, 13 * 6)
            # Extract and convert lower body pose data
            tar_pose_leg = tar_pose[:, self.joint_mask_lower.astype(bool)].reshape(n, 9, 3)
            tar_pose_leg_mirror = tar_pose_mirror[:, self.joint_mask_lower.astype(bool)].reshape(n, 9, 3)
            tar_pose_leg_6d = axis_angle_to_6d_np(tar_pose_leg).reshape(n, 9 * 6)
            tar_pose_leg_6d_mirror = axis_angle_to_6d_np(tar_pose_leg_mirror).reshape(n, 9 * 6)
            
            # Convert other data to tensors
            tar_pose_lower = np.concatenate([tar_pose_leg_6d, tar_trans, tar_contact], axis=1)
            tar_pose_lower_mirror = np.concatenate([tar_pose_leg_6d_mirror, tar_trans_mirror, tar_contact_mirror], axis=1)
            tar_pose_6d = axis_angle_to_6d_np(tar_pose.reshape(n, 55, 3)).reshape(n, 55 * 6)
            tar_pose_6d_mirror = axis_angle_to_6d_np(tar_pose_mirror.reshape(n, 55, 3)).reshape(n, 55 * 6)

            round_seconds_skeleton = tar_pose_6d.shape[0] // pose_fps_beat2
            if round_seconds_skeleton == 0:
                round_seconds_skeleton = 1
            clip_s_t, clip_e_t = 0, round_seconds_skeleton - 0  # assume [10, 90]s
            clip_s_f_pose, clip_e_f_pose = clip_s_t * pose_fps_beat2, clip_e_t * pose_fps_beat2  # [150,90*15]

            if self.split == 'test' or self.split == 'token':  # stride = length for test
                cut_length = clip_e_f_pose - clip_s_f_pose
                stride = cut_length
                self.max_length = cut_length
            else:
                stride = int(config.stride)
                cut_length = int(self.ori_length)
            num_subdivision = math.floor((clip_e_f_pose - clip_s_f_pose - cut_length) / stride) + 1

            for i in range(num_subdivision):  # cut into around 2s chip, (self npose)

                start_idx = clip_s_f_pose + i * config.stride
                fin_idx = start_idx + cut_length
                sample_pose = tar_pose_6d[start_idx:fin_idx]
                sample_pose_mirror = tar_pose_6d_mirror[start_idx:fin_idx]
                sample_face = tar_pose_face[start_idx:fin_idx]
                sample_face_mirror = tar_pose_face_mirror[start_idx:fin_idx]
                sample_face_with_head = tar_pose_face_with_head[start_idx:fin_idx]
                sample_face_with_head_mirror = tar_pose_face_with_head_mirror[start_idx:fin_idx]
                sample_hand = tar_pose_hands_6d[start_idx:fin_idx]
                sample_hand_mirror = tar_pose_hands_6d_mirror[start_idx:fin_idx]
                sample_upper = tar_pose_upper_6d[start_idx:fin_idx]
                sample_upper_mirror = tar_pose_upper_6d_mirror[start_idx:fin_idx]
                sample_lower = tar_pose_lower[start_idx:fin_idx]
                sample_lower_mirror = tar_pose_lower_mirror[start_idx:fin_idx]
                sample_trans = tar_trans[start_idx:fin_idx]
                sample_trans_mirror = tar_trans_mirror[start_idx:fin_idx]
                sample_shape = betas[start_idx:fin_idx]
                sample_expressions = expressions[start_idx:fin_idx]
                sample_expressions_mirror = expressions_mirror[start_idx:fin_idx]

                new_name = 'beat2_' + '%s_%d' % (f_name,i)
                # Store processed data
                self.data_dict_beat2[new_name] = {
                    'face': sample_face,
                    'face_with_head': sample_face_with_head,
                    'hand': sample_hand,
                    'upper': sample_upper,
                    'lower': sample_lower,
                    'pose': sample_pose,
                    'shape': sample_shape,
                    'trans': sample_trans,
                    'exps': sample_expressions,
                    'id': f_name,
                    'dataset_name': 'beat2',
                }
                self.metadata_beat2.append(new_name)

                new_name_mirror = 'beat2_' + 'M_' + '%s_%d' % (f_name,i)
                # Store processed data
                self.data_dict_beat2[new_name_mirror] = {
                    'face': sample_face_mirror,
                    'face_with_head': sample_face_with_head_mirror,
                    'hand': sample_hand_mirror,
                    'upper': sample_upper_mirror,
                    'lower': sample_lower_mirror,
                    'pose': sample_pose_mirror,
                    'shape': sample_shape,
                    'trans': sample_trans_mirror,
                    'exps': sample_expressions_mirror,
                    'id': 'M_' + f_name,
                    'dataset_name': 'beat2',
                }
                self.metadata_beat2.append(new_name_mirror)



            # for fast debug
            if index >= self.maxdata:
                break
        
        # Save processed data to cache
        self._save_to_cache(cache_path, self.data_dict_beat2, self.metadata_beat2, "BEAT2")

    def _load_candor(self, config):
        """
        Load the CANDOR dataset based on the configuration.
        """
        # Get cache path for this specific dataset
        data_root = self.args["CANDOR"].ROOT
        cache_path = self._get_cache_path(data_root, "CANDOR")
                
        # We need to process data from scratch, so initialize SMPLX if needed
        # self._initialize_flame_if_needed()
        
        # Check if we should load from cache
        data_dict, metadata = self._load_from_cache(cache_path, "CANDOR")
        if data_dict is not None:
            self.data_dict_candor = data_dict
            self.metadata_candor = metadata
            return
        
        print(f"Processing CANDOR dataset...")
        
        self.data_root_candor = data_root

        structure_path = pjoin(self.data_root_candor, 'candor_structure.json')
        with open(structure_path, 'r') as f:
            candor_structure = json.load(f)

        # Split the dataset based on the 'split' parameter
        conversation_ids = list(candor_structure.keys())
        # import random
        # random.seed(42)  # Fixed seed for reproducibility
        # random.shuffle(conversation_ids)
        # # Deterministic split based on conversation ID
        # if self.split == "train":
        #     # Use the first 80% of shuffled IDs for training
        #     conversation_ids = conversation_ids[:int(len(conversation_ids) * 0.8)]
        # elif self.split == "val":
        #     # Use 80-90% of shuffled IDs for validation
        #     conversation_ids = conversation_ids[int(len(conversation_ids) * 0.8):int(len(conversation_ids) * 0.9)]
        # elif self.split == "test":
        #     # Use the last 10% of shuffled IDs for testing
        #     conversation_ids = conversation_ids[int(len(conversation_ids) * 0.9):]
        # # If split is not specified or is another value, use all conversations

        self.data_dict_candor = {}
        self.metadata_candor = []

        for index, sequence_id in enumerate(tqdm(conversation_ids)):

            try:

                if len(candor_structure[sequence_id]) < 2:
                    continue
                # Load FLAME coefficients for both participants in the conversation
                flame_coeffs_participant1 = np.load(pjoin(self.data_root_candor, 'FLAME_coeffs', sequence_id, candor_structure[sequence_id][0]))
                flame_coeffs_participant2 = np.load(pjoin(self.data_root_candor, 'FLAME_coeffs', sequence_id, candor_structure[sequence_id][1]))
                    
                # Extract expression and pose parameters for both participants
                expressions_p1 = flame_coeffs_participant1['exp']
                pose_p1 = flame_coeffs_participant1['pose']

                expressions_p2 = flame_coeffs_participant2['exp']
                pose_p2 = flame_coeffs_participant2['pose']

                # Get sequence lengths
                seq_length_p1 = expressions_p1.shape[0]
                seq_length_p2 = expressions_p2.shape[0]
                
                # Process head pose for participant 1
                head_pose_p1 = pose_p1[:, :3]
                head_pose_6d_p1 = axis_angle_to_6d_np(head_pose_p1).reshape(seq_length_p1, 6)
                # head_pose_6d_p1 = head_pose_p1
                
                # Process jaw pose for participant 1
                jaw_pose_p1 = pose_p1[:, 3:]
                expressions_p1_padded = np.zeros((seq_length_p1, 100))
                expressions_p1_padded[:, :50] = expressions_p1
                jaw_pose_6d_p1 = axis_angle_to_6d_np(jaw_pose_p1).reshape(seq_length_p1, 6)
                
                # Concatenate jaw pose and expressions for participant 1's face
                face_features_p1 = np.concatenate([head_pose_6d_p1, jaw_pose_6d_p1, expressions_p1_padded], axis=1)

                # Process head pose for participant 1
                head_pose_p2 = pose_p2[:, :3]
                head_pose_6d_p2 = axis_angle_to_6d_np(head_pose_p2).reshape(seq_length_p2, 6)


                # Process jaw pose for participant 2
                jaw_pose_p2 = pose_p2[:, 3:]
                expressions_p2_padded = np.zeros((seq_length_p2, 100))
                expressions_p2_padded[:, :50] = expressions_p2
                jaw_pose_6d_p2 = axis_angle_to_6d_np(jaw_pose_p2).reshape(seq_length_p2, 6)
                
                # Concatenate jaw pose and expressions for participant 2's face
                face_features_p2 = np.concatenate([head_pose_6d_p2, jaw_pose_6d_p2, expressions_p2_padded], axis=1)

                # Store processed data for this sequence
                self.data_dict_candor[sequence_id] = {
                    'face_p1': face_features_p1,
                    'p1_name': candor_structure[sequence_id][0].split('.')[0],
                    'face_p2': face_features_p2,
                    'p2_name': candor_structure[sequence_id][1].split('.')[0],
                    'id': sequence_id,
                    'dataset_name': 'candor',
                }
                self.metadata_candor.append(sequence_id)
            except Exception as e:
                print(f"Error processing sequence {sequence_id}: {e}")
                continue
            if index >= self.maxdata:
                break
        
        # Save processed data to cache
        self._save_to_cache(cache_path, self.data_dict_candor, self.metadata_candor, "CANDOR")

    def _load_tfhp(self, config):
        """
        Load the TFHP dataset based on the configuration.
        """
        # Get cache path for this specific dataset
        data_root = self.args["TFHP"].ROOT
        cache_path = self._get_cache_path(data_root, "TFHP")
                
        # Check if we should load from cache
        data_dict, metadata = self._load_from_cache(cache_path, "TFHP")
        if data_dict is not None:
            self.data_dict_tfhp = data_dict
            self.metadata_tfhp = metadata
            return
        
        print(f"Processing TFHP dataset...")
        
        self.data_root_tfhp = data_root

        self.data_dict_tfhp = {}
        self.metadata_tfhp = []
        index = 0
        # for index, sequence_id in enumerate(tqdm(sequence_ids)):
        for root, dirs, files in tqdm(os.walk(pjoin(self.data_root_tfhp, 'coef'))):

            try:
                index += 1
                for file in files:
                    if not file.endswith('.npz'):
                        continue

                    coef_path = os.path.join(root, file)
                    clean_name = coef_path.replace(pjoin(self.data_root_tfhp, 'coef'), '').replace('.npz', '')[1:]
                    # Load FLAME coefficients for both participants in the conversation
                    flame_coeffs = np.load(coef_path)

                    # Extract expression and pose parameters for both participants
                    expressions = flame_coeffs['exp']
                    pose = flame_coeffs['pose']

                    # Get sequence lengths
                    seq_length = expressions.shape[0]
                    
                    # Process head pose for participant 1
                    head_pose = pose[:, :3]
                    head_pose_6d = axis_angle_to_6d_np(head_pose).reshape(seq_length, 6)
                    
                    # Process jaw pose for participant 1
                    jaw_pose = pose[:, 3:]
                    expressions_padded = np.zeros((seq_length, 100))
                    expressions_padded[:, :50] = expressions
                    jaw_pose_6d = axis_angle_to_6d_np(jaw_pose).reshape(seq_length, 6)
                    
                    # Concatenate jaw pose and expressions for participant 1's face
                    face_features = np.concatenate([jaw_pose_6d, expressions_padded], axis=1)
                    face_features_with_head = np.concatenate([head_pose_6d, jaw_pose_6d, expressions_padded], axis=1)

                    # Store processed data for this sequence
                    # self.data_dict_tfhp[clean_name] = {
                    #     'face': face_features,
                    #     'id': clean_name,
                    #     'dataset_name': 'tfhp',
                    # }
                    self.data_dict_tfhp[clean_name] = {
                        # 'face_p1': face_features,
                        # 'p1_name': '',
                        # 'face_p2': '',
                        # 'p2_name': '',
                        'face': face_features,
                        'face_with_head': face_features_with_head,
                        'id': clean_name,
                        'dataset_name': 'tfhp',
                    }
                    self.metadata_tfhp.append(clean_name)
            except Exception as e:
                print(f"Error processing sequence {clean_name}: {e}")
                continue
            if index >= self.maxdata:
                break
        
        # Save processed data to cache
        self._save_to_cache(cache_path, self.data_dict_tfhp, self.metadata_tfhp, "TFHP")

    def _load_youtube_talking(self, config):
        """
        Load the TFHP dataset based on the configuration.
        """
        # Get cache path for this specific dataset
        data_root = self.args["YouTube_Talking"].ROOT
        cache_path = self._get_cache_path(data_root, "YouTube_Talking")
                
        # Check if we should load from cache
        data_dict, metadata = self._load_from_cache(cache_path, "YouTube_Talking")
        if data_dict is not None:
            self.data_dict_youtube_talking = data_dict
            self.metadata_youtube_talking = metadata
            return
        
        print(f"Processing YouTube_Talking dataset...")
        
        self.data_root_youtube_talking = data_root

        self.data_dict_youtube_talking = {}
        self.metadata_youtube_talking = []
        index = 0
        # for index, sequence_id in enumerate(tqdm(sequence_ids)):
        for root, dirs, files in tqdm(os.walk(pjoin(self.data_root_youtube_talking, 'FLAME_coeffs_25'))):

            try:
                index += 1
                for file in files:
                    if not file.endswith('.npz'):
                        continue

                    coef_path = os.path.join(root, file)
                    clean_name = coef_path.replace(pjoin(self.data_root_youtube_talking, 'FLAME_coeffs_25'), '').replace('.npz', '')[1:]
                    # Load FLAME coefficients for both participants in the conversation
                    flame_coeffs = np.load(coef_path)

                    # Extract expression and pose parameters for both participants
                    expressions = flame_coeffs['exp']
                    pose = flame_coeffs['pose']

                    # Get sequence lengths
                    seq_length = expressions.shape[0]
                    
                    # Process head pose for participant 1
                    head_pose = pose[:, :3]
                    head_pose_6d = axis_angle_to_6d_np(head_pose).reshape(seq_length, 6)
                    
                    # Process jaw pose for participant 1
                    jaw_pose = pose[:, 3:]
                    expressions_padded = np.zeros((seq_length, 100))
                    expressions_padded[:, :50] = expressions
                    jaw_pose_6d = axis_angle_to_6d_np(jaw_pose).reshape(seq_length, 6)
                    
                    # Concatenate jaw pose and expressions for participant 1's face
                    face_features = np.concatenate([head_pose_6d, jaw_pose_6d, expressions_padded], axis=1)

                    # Store processed data for this sequence
                    self.data_dict_youtube_talking[clean_name] = {
                        'face_p1': face_features,
                        'id': clean_name,
                        'dataset_name': 'YouTube_Talking',
                    }
                    self.metadata_youtube_talking.append(clean_name)
            except Exception as e:
                print(f"Error processing sequence {clean_name}: {e}")
                continue
            if index >= self.maxdata:
                break
        
        # Save processed data to cache
        self._save_to_cache(cache_path, self.data_dict_youtube_talking, self.metadata_youtube_talking, "YouTube_Talking")

    def _load_youtube_talking_face(self, config):
        """
        Load the TFHP dataset based on the configuration.
        """
        # Get cache path for this specific dataset
        data_root = self.args["YouTube_Talking"].ROOT
        cache_path = self._get_cache_path(data_root, "YouTube_Talking")
                
        # Check if we should load from cache
        data_dict, metadata = self._load_from_cache(cache_path, "YouTube_Talking")
        if data_dict is not None:
            self.data_dict_youtube_talking_face = data_dict
            self.metadata_youtube_talking_face = metadata
            return
        
        print(f"Processing YouTube_Talking dataset...")
        
        self.data_root_youtube_talking = data_root

        self.data_dict_youtube_talking_face = {}
        self.metadata_youtube_talking_face = []
        index = 0
        # for index, sequence_id in enumerate(tqdm(sequence_ids)):
        for root, dirs, files in tqdm(os.walk(pjoin(self.data_root_youtube_talking, 'FLAME_coeffs_25'))):

            try:
                index += 1
                for file in files:
                    if not file.endswith('.npz'):
                        continue

                    coef_path = os.path.join(root, file)
                    clean_name = coef_path.replace(pjoin(self.data_root_youtube_talking, 'FLAME_coeffs_25'), '').replace('.npz', '')[1:]
                    # Load FLAME coefficients for both participants in the conversation
                    flame_coeffs = np.load(coef_path)

                    # Extract expression and pose parameters for both participants
                    expressions = flame_coeffs['exp']
                    pose = flame_coeffs['pose']

                    # Get sequence lengths
                    seq_length = expressions.shape[0]
                    
                    # Process head pose for participant 1
                    head_pose = pose[:, :3]
                    head_pose_6d = axis_angle_to_6d_np(head_pose).reshape(seq_length, 6)
                    
                    # Process jaw pose for participant 1
                    jaw_pose = pose[:, 3:]
                    expressions_padded = np.zeros((seq_length, 100))
                    expressions_padded[:, :50] = expressions
                    jaw_pose_6d = axis_angle_to_6d_np(jaw_pose).reshape(seq_length, 6)
                    
                    # Concatenate jaw pose and expressions for participant 1's face
                    face_features = np.concatenate([jaw_pose_6d, expressions_padded], axis=1)
                    face_features_with_head = np.concatenate([head_pose_6d, jaw_pose_6d, expressions_padded], axis=1)
                    # Store processed data for this sequence
                    self.data_dict_youtube_talking_face[clean_name] = {
                        'face': face_features,
                        'face_with_head': face_features_with_head,
                        'id': clean_name,
                        'dataset_name': 'YouTube_Talking',
                    }
                    self.metadata_youtube_talking_face.append(clean_name)
            except Exception as e:
                print(f"Error processing sequence {clean_name}: {e}")
                continue
            if index >= self.maxdata:
                break
        
        # Save processed data to cache
        self._save_to_cache(cache_path, self.data_dict_youtube_talking_face, self.metadata_youtube_talking_face, "YouTube_Talking_Face")

    def _load_youtube_talking_upper(self, config):
        """
        Load the TFHP dataset based on the configuration.
        """
        # Get cache path for this specific dataset
        data_root = self.args["YouTube_Talking"].ROOT
        cache_path = self._get_cache_path(data_root, "YouTube_Talking")
                
        # Check if we should load from cache
        data_dict, metadata = self._load_from_cache(cache_path, "YouTube_Talking")
        if data_dict is not None:
            self.data_dict_youtube_talking_upper = data_dict
            self.metadata_youtube_talking_upper = metadata
            return
        
        print(f"Processing YouTube_Talking dataset...")
        
        self.data_root_youtube_talking = data_root

        self.data_dict_youtube_talking_upper = {}
        self.metadata_youtube_talking_upper = []
        index = 0
        # for index, sequence_id in enumerate(tqdm(sequence_ids)):
        for root, dirs, files in tqdm(os.walk(pjoin(self.data_root_youtube_talking, 'smplxflame_25'))):

            try:
                for file in files:
                    if not file.endswith('.npz'):
                        continue

                    # coef_path = os.path.join(root, file)
                    # clean_name = coef_path.replace(pjoin(self.data_root_youtube_talking, 'smplxflame_25'), '').replace('.npz', '')[1:]
                    # # Load FLAME coefficients for both participants in the conversation
                    # flame_coeffs = np.load(coef_path)

                    pose_path = os.path.join(root, file)
                    clean_name = pose_path.replace(pjoin(self.data_root_youtube_talking, 'smplxflame_25'), '').replace('.npz', '')[1:]
                    
                    pose_data = np.load(pose_path, allow_pickle=True)
                    poses = pose_data["poses"]
                    n, c = poses.shape[0], poses.shape[1]
                    pose_fps = pose_data["mocap_frame_rate"]

                    pose_processed = poses * JOINT_MASK_FULL
                    pose_processed = pose_processed[:, JOINT_MASK_FULL.astype(bool)]
                    # Extract and convert upper body pose data
                    tar_pose_upper = pose_processed[:, self.joint_mask_upper.astype(bool)].reshape(n, 13, 3)
                    tar_pose_upper_6d = axis_angle_to_6d_np(tar_pose_upper).reshape(n, 13 * 6)

                    # Calculate time segments
                    round_seconds_skeleton = tar_pose_upper_6d.shape[0] // pose_fps
                    if round_seconds_skeleton == 0:
                        round_seconds_skeleton = 1
                    clip_s_t, clip_e_t = 0, round_seconds_skeleton - 0
                    clip_s_f_pose, clip_e_f_pose = clip_s_t * pose_fps, clip_e_t * pose_fps
                
                    if self.split == 'test' or self.split == 'token':  # stride = length for test
                        cut_length = clip_e_f_pose - clip_s_f_pose
                        stride = cut_length
                        # self.max_length = cut_length
                    else:
                        stride = int(config.stride)
                        cut_length = int(self.ori_length)
                    
                    num_subdivision = math.floor((clip_e_f_pose - clip_s_f_pose - cut_length) / stride) + 1

                    # Create segments
                    for i in range(num_subdivision):
                        start_idx = clip_s_f_pose + i * stride
                        fin_idx = start_idx + cut_length
                        
                        # Skip if out of bounds
                        if fin_idx > tar_pose_upper_6d.shape[0]:
                            continue
                            
                        sample_upper = tar_pose_upper_6d[start_idx:fin_idx]

                        new_name = 'youtube_talking_upper_' + '%s_%d' % (clean_name, i)

                        # Store processed data
                        self.data_dict_youtube_talking_upper[new_name] = {
                            'upper': sample_upper,
                            'id': clean_name,
                            'dataset_name': 'YouTube_Talking',
                        }
                        self.metadata_youtube_talking_upper.append(new_name)

                    index += 1
                    if index >= self.maxdata:
                        break
            except Exception as e:
                print(f"Error processing sequence {new_name}: {e}")
                continue
        
        # Save processed data to cache
        self._save_to_cache(cache_path, self.data_dict_youtube_talking_upper, self.metadata_youtube_talking_upper, "YouTube_Talking_Upper")

    def _load_youtube_talking_synthetic(self, config):
        """
        Load the TFHP dataset based on the configuration.
        """
        # Get cache path for this specific dataset
        data_root = self.args["YouTube_Talking_Synthetic"].ROOT
        cache_path = self._get_cache_path(data_root, "YouTube_Talking_Synthetic")
                
        # Check if we should load from cache
        data_dict, metadata = self._load_from_cache(cache_path, "YouTube_Talking_Synthetic")
        if data_dict is not None:
            self.data_dict_youtube_talking = data_dict
            self.metadata_youtube_talking = metadata
            return
        
        print(f"Processing YouTube_Talking_Synthetic dataset...")
        
        self.data_root_youtube_talking = data_root

        self.data_dict_youtube_talking_synthetic = {}
        self.metadata_youtube_talking_synthetic = []
        index = 0
        # for index, sequence_id in enumerate(tqdm(sequence_ids)):
        for root, dirs, files in tqdm(os.walk(pjoin(self.data_root_youtube_talking, 'FLAME_coeffs_25'))):

            try:
                index += 1
                for file in files:
                    if not file.endswith('.npz'):
                        continue

                    coef_path = os.path.join(root, file)
                    clean_name = coef_path.replace(pjoin(self.data_root_youtube_talking, 'FLAME_coeffs_25'), '').replace('.npz', '')[1:]
                    # Load FLAME coefficients for both participants in the conversation
                    flame_coeffs = np.load(coef_path)

                    # Extract expression and pose parameters for both participants
                    expressions = flame_coeffs['exp']
                    pose = flame_coeffs['pose']

                    # Get sequence lengths
                    seq_length = expressions.shape[0]
                    
                    # Process head pose for participant 1
                    head_pose = pose[:, :3]
                    head_pose_6d = axis_angle_to_6d_np(head_pose).reshape(seq_length, 6)
                    
                    # Process jaw pose for participant 1
                    jaw_pose = pose[:, 3:]
                    # expressions_padded = np.zeros((seq_length, 100))
                    # expressions_padded[:, :50] = expressions[:,:50]
                    jaw_pose_6d = axis_angle_to_6d_np(jaw_pose).reshape(seq_length, 6)
                    
                    # Concatenate jaw pose and expressions for participant 1's face
                    face_features = np.concatenate([jaw_pose_6d, expressions], axis=1)
                    face_features_with_head = np.concatenate([head_pose_6d, jaw_pose_6d, expressions], axis=1)

                    # Store processed data for this sequence
                    self.data_dict_youtube_talking_synthetic[clean_name] = {
                        'face': face_features,
                        'face_with_head': face_features_with_head,
                        'id': clean_name,
                        'dataset_name': 'YouTube_Talking_Synthetic',
                    }
                    self.metadata_youtube_talking_synthetic.append(clean_name)
            except Exception as e:
                print(f"Error processing sequence {clean_name}: {e}")
                continue
            if index >= self.maxdata:
                break
        
        # Save processed data to cache
        self._save_to_cache(cache_path, self.data_dict_youtube_talking_synthetic, self.metadata_youtube_talking_synthetic, "YouTube_Talking_Synthetic")

    def _load_amass_h3d(self, config):
        """
        Load the AMASS dataset in HumanML3D format based on the configuration.
        
        Parameters:
        - config: Configuration dictionary for AMASS_H3D.
        """
        # Get cache path for this specific dataset
        data_root = config.get("data_root")
        cache_path = self._get_cache_path(data_root, "amass_h3d")
        
        # Check if we should load from cache
        data_dict, metadata = self._load_from_cache(cache_path, "amass_h3d")
        if data_dict is not None:
            self.data_dict_amass_h3d = data_dict
            self.metadata_amass_h3d = metadata
            return
        
        print(f"Processing AMASS_H3D dataset...")
        
        self.data_root_amass_h3d = data_root
        pose_fps_amass = config.pose_fps
        motion_unit = config.motion_unit
        split_file = pd.read_csv(pjoin(self.data_root_amass_h3d, f"new_{self.split}.csv"))
        # Calculate maximum lengths for data
        self.max_length = int(config.pose_length)
        self.ori_length = config.pose_length

        self.metadata_amass_h3d = []
        self.data_dict_amass_h3d = {}
        ##################  AMASS in Humanml3d Format ##################
        # Define lengths for data processing
        for index, file_name in tqdm(split_file.iterrows()):
            f_name = file_name["id"]
            pose_file = pjoin(self.data_root_amass_h3d, 'new_joint_vecs', f_name+'.npy')
            pose_each_file = np.load(pose_file, allow_pickle=True)
            # Normalization
            pose_each_file = (pose_each_file - self.h3d_mean) / self.h3d_std

            round_seconds_skeleton = pose_each_file.shape[0] // motion_unit
            if round_seconds_skeleton == 0:
                round_seconds_skeleton = 1
            clip_s_t, clip_e_t = 0, round_seconds_skeleton - 0  # assume [10, 90]s
            clip_s_f_pose, clip_e_f_pose = clip_s_t * pose_fps_amass, clip_e_t * motion_unit  # [150,90*15]

            
            if self.split == 'test':  # stride = length for test
                cut_length = clip_e_f_pose - clip_s_f_pose
                stride = cut_length
                self.max_length = cut_length
            else:
                stride = int(config.stride)
                cut_length = int(int(self.ori_length))
            num_subdivision = math.floor((clip_e_f_pose - clip_s_f_pose - cut_length) / stride) + 1

            for i in range(num_subdivision):  # cut into around 2s chip, (self npose)
                start_idx = clip_s_f_pose + i * config.stride
                fin_idx = start_idx + cut_length
                sample_pose = pose_each_file[start_idx:fin_idx]
                sample_trans = np.zeros(1)
                sample_shape = np.zeros(1)
                if sample_pose.any() and sample_pose.size != 0:
                    new_name = 'amass_' + '%s_%d' % (f_name,i)
                    self.data_dict_amass_h3d[new_name] = {
                        'id' : f_name,
                        'pose': sample_pose,
                        'shape': sample_shape,
                        'trans': sample_trans,
                    }
                    self.metadata_amass_h3d.append(new_name)
            # for fast debug
            if index >= self.maxdata:
                break
        
        # Save processed data to cache
        self._save_to_cache(cache_path, self.data_dict_amass_h3d, self.metadata_amass_h3d, "amass_h3d")

    def _load_amass(self, config):
        """
        Load the AMASS dataset based on the configuration.

        Parameters:
        - config: Configuration dictionary for AMASS.
        """
        # Get cache path for this specific dataset
        data_root = self.args["AMASS"].ROOT
        cache_path = self._get_cache_path(data_root, "AMASS")
        pose_rep = config.pose_rep
        # pose_rep_face = config.pose_rep_face
        foot_contact_path = config.foot_contact_path
        # Check if we should load from cache
        data_dict, metadata = self._load_from_cache(cache_path, "AMASS")
        if data_dict is not None:
            self.data_dict_amass = data_dict
            self.metadata_amass = metadata
            return
        
        # We need to process data from scratch, so initialize SMPLX if needed
        self._initialize_smplx_if_needed()
        
        print(f"Processing AMASS dataset...")
        
        self.data_root_amass = data_root
        pose_fps_amass = config.pose_fps
        # split_file = pd.read_csv(pjoin(self.data_root_amass, f"new_{self.split}.csv"))

        split_file_train = pjoin(self.data_root_amass, 'train.txt')
        # Data id list
        id_list_train = []
        with cs.open(split_file_train, "r") as f:
            for line in f.readlines():
                id_list_train.append(line.strip())

        split_file_test = pjoin(self.data_root_amass, 'test.txt')
        # Data id list
        id_list_test = []
        with cs.open(split_file_test, "r") as f:
            for line in f.readlines():
                id_list_test.append(line.strip())

        if self.split == 'train':
            id_list_amass = id_list_train
        elif self.split == 'test':
            id_list_amass = id_list_test
        else:
            id_list_amass = id_list_train + id_list_test

        self.ori_length = config.pose_length
        # Calculate maximum lengths for data
        # self.max_length = int(config.pose_length)

        self.metadata_amass = []
        self.data_dict_amass = {}

        ##################  AMASS  ##################
        # Process each file
        # for index, file_name in tqdm(split_file.iterrows()):
        for index, file_name in tqdm(enumerate(id_list_amass)):
            try:

                pose_file = pjoin(self.data_root_amass, pose_rep, file_name+'.npz')
                pose_data = np.load(pose_file, allow_pickle=True)

                # flame_path = pjoin(self.data_root_amass, pose_rep_face, file_name + '.npz')
                # flame_data = np.load(flame_path)
                # # Extract expression and pose parameters
                # face_expressions = flame_data['exp']  # Shape: [n_frames, 50]
                # face_pose = flame_data['pose']        # Shape: [n_frames, 6] (global + jaw)
                # face_shape = flame_data['shape']      # Shape: [n_frames, 100/300] (betas)
                # n_face, c_face = face_expressions.shape[0], face_expressions.shape[1]


                # Process pose data
                poses = pose_data["poses"]
                n, c = poses.shape[0], poses.shape[1]

                tar_trans = pose_data["trans"]

                padded_betas = np.zeros(300)
                padded_betas[:16] = pose_data["betas"]
                betas = np.repeat(padded_betas.reshape(1, 300), n, axis=0)
                
                # Process pose data
                pose_processed = poses * JOINT_MASK_FULL
                pose_processed = pose_processed[:, JOINT_MASK_FULL.astype(bool)]
                
                if self.motion_representation == 'rotation':
                    # Calculate foot contacts
                    foot_contacts_path = pjoin(self.data_root_amass, foot_contact_path, file_name + '.npy')
                    if os.path.exists(foot_contacts_path):
                        contacts = np.load(foot_contacts_path)
                    else:
                        contacts = self.comput_foot_contacts(pose_data)
                        os.makedirs(pjoin(self.data_root_amass, foot_contact_path), exist_ok=True)
                        np.save(foot_contacts_path, contacts)
                    pose_processed = np.concatenate([pose_processed, contacts], axis=1)
                
                tar_pose = pose_processed[:, :165]

                tar_contact = contacts

                # Extract and convert jaw pose data
                tar_pose_face = np.zeros((n, 106))
                tar_pose_face_with_head = np.zeros((n, 112))

                
                # # Process head pose (positions 0:3 in pose)
                # head_pose = face_pose[:, :3]  # Global head orientation in axis-angle
                # head_pose_6d = axis_angle_to_6d_np(head_pose).reshape(n_face, 6)
                # # Process jaw pose (positions 3:6 in pose)
                # jaw_pose = face_pose[:, 3:6]  # Jaw rotation in axis-angle
                # jaw_pose_6d = axis_angle_to_6d_np(jaw_pose).reshape(n_face, 6)
                
                # # Pad expressions to 100 dimensions as needed
                # if face_expressions.shape[1] < 100:
                #     padded_exps = np.zeros((n_face, 100), dtype=face_expressions.dtype)
                #     padded_exps[:, :face_expressions.shape[1]] = face_expressions
                #     face_expressions = padded_exps
                # elif face_expressions.shape[1] > 100:
                #     face_expressions = face_expressions[:, :100]
                
                # # Concatenate head pose, jaw pose and expressions for 112D face representation
                # tar_pose_face = np.concatenate([jaw_pose_6d, face_expressions], axis=1)
                # tar_pose_face_with_head = np.concatenate([head_pose_6d, jaw_pose_6d, face_expressions], axis=1)
                

                # Extract and convert hand pose data
                tar_pose_hands_6d = np.zeros((n, 180))
                # Extract and convert upper body pose data
                tar_pose_upper = tar_pose[:, self.joint_mask_upper.astype(bool)].reshape(n, 13, 3)
                tar_pose_upper_6d = axis_angle_to_6d_np(tar_pose_upper).reshape(n, 13 * 6)

                # Extract and convert lower body pose data
                tar_pose_leg = tar_pose[:, self.joint_mask_lower.astype(bool)].reshape(n, 9, 3)
                tar_pose_leg_6d = axis_angle_to_6d_np(tar_pose_leg).reshape(n, 9 * 6)
                
                # Combine lower body pose with translation and contacts
                tar_pose_lower = np.concatenate([tar_pose_leg_6d, tar_trans, tar_contact], axis=1)
                
                # # Convert full pose to 6D representation
                tar_pose_6d = axis_angle_to_6d_np(tar_pose.reshape(n, 55, 3)).reshape(n, 55, 6)

                # Calculate time segments
                round_seconds_skeleton = tar_pose_6d.shape[0] // pose_fps_amass
                if round_seconds_skeleton == 0:
                    round_seconds_skeleton = 1
                clip_s_t, clip_e_t = 0, round_seconds_skeleton - 0
                clip_s_f_pose, clip_e_f_pose = clip_s_t * pose_fps_amass, clip_e_t * pose_fps_amass
            
                if self.split == 'test' or self.split == 'token':  # stride = length for test
                    cut_length = clip_e_f_pose - clip_s_f_pose
                    stride = cut_length
                    # self.max_length = cut_length
                else:
                    stride = int(config.stride)
                    cut_length = int(self.ori_length)
                
                num_subdivision = math.floor((clip_e_f_pose - clip_s_f_pose - cut_length) / stride) + 1

                # Create segments
                for i in range(num_subdivision):
                    start_idx = clip_s_f_pose + i * stride
                    fin_idx = start_idx + cut_length
                    
                    # Skip if out of bounds
                    if fin_idx > tar_pose_6d.shape[0]:
                        continue
                        
                    sample_pose = tar_pose_6d[start_idx:fin_idx]
                    # sample_face = tar_pose_face[start_idx:fin_idx]  # Just jaw pose, no expressions
                    # sample_face_with_head = tar_pose_face_with_head[start_idx:fin_idx]
                    sample_hand = tar_pose_hands_6d[start_idx:fin_idx]
                    sample_upper = tar_pose_upper_6d[start_idx:fin_idx]
                    sample_lower = tar_pose_lower[start_idx:fin_idx]
                    sample_trans = tar_trans[start_idx:fin_idx]
                    sample_shape = betas[start_idx:fin_idx]

                    new_name = 'amass_' + '%s_%d' % (file_name, i)

                    # Store processed data
                    self.data_dict_amass[new_name] = {
                        # 'face': sample_face,
                        # 'face_with_head': sample_face_with_head,
                        'face': tar_pose_face,
                        'face_with_head': tar_pose_face_with_head,
                        'hand': sample_hand,
                        'upper': sample_upper,
                        'lower': sample_lower,
                        'pose': sample_pose,
                        'shape': sample_shape,
                        'trans': sample_trans,
                        'id': file_name,
                        'dataset_name': 'amass',
                    }
                    self.metadata_amass.append(new_name)
            except Exception as e:
                # print(f"Error processing file {f_name}: {str(e)}")
                continue

            # For fast debug
            if index >= self.maxdata:
                break
        
        # Save processed data to cache
        self._save_to_cache(cache_path, self.data_dict_amass, self.metadata_amass, "AMASS")

    def _load_amass_talking(self, config):
        """
        Load the AMASS dataset based on the configuration.

        Parameters:
        - config: Configuration dictionary for AMASS.
        """
        # Get cache path for this specific dataset
        data_root = self.args["AMASS_talking"].ROOT
        cache_path = self._get_cache_path(data_root, "AMASS_talking")
        pose_rep = config.pose_rep
        pose_rep_face = config.pose_rep_face
        foot_contact_path = config.foot_contact_path
        # Check if we should load from cache
        data_dict, metadata = self._load_from_cache(cache_path, "AMASS_talking")
        if data_dict is not None:
            self.data_dict_amass_talking = data_dict
            self.metadata_amass_talking = metadata
            return
        
        # We need to process data from scratch, so initialize SMPLX if needed
        self._initialize_smplx_if_needed()
        
        print(f"Processing AMASS dataset...")
        
        self.data_root_amass = data_root
        pose_fps_amass = config.pose_fps
        # split_file = pd.read_csv(pjoin(self.data_root_amass, f"new_{self.split}.csv"))

        split_file_train = pjoin(self.data_root_amass, 'train.txt')
        # Data id list
        id_list_train = []
        with cs.open(split_file_train, "r") as f:
            for line in f.readlines():
                id_list_train.append(line.strip())

        split_file_test = pjoin(self.data_root_amass, 'test.txt')
        # Data id list
        id_list_test = []
        with cs.open(split_file_test, "r") as f:
            for line in f.readlines():
                id_list_test.append(line.strip())

        if self.split == 'train':
            id_list_amass = id_list_train
        elif self.split == 'test':
            id_list_amass = id_list_test
        else:
            id_list_amass = id_list_train + id_list_test

        self.ori_length = config.pose_length
        # Calculate maximum lengths for data
        # self.max_length = int(config.pose_length)

        self.metadata_amass_talking = []
        self.data_dict_amass_talking = {}

        ##################  AMASS  ##################
        # Process each file
        # for index, file_name in tqdm(split_file.iterrows()):
        for index, file_name in tqdm(enumerate(id_list_amass)):
            try:

                pose_file = pjoin(self.data_root_amass, pose_rep, file_name+'.npz')
                pose_data = np.load(pose_file, allow_pickle=True)

                flame_path = pjoin(self.data_root_amass, pose_rep_face, file_name + '.npz')
                flame_data = np.load(flame_path)
                # Extract expression and pose parameters
                face_expressions = flame_data['exp']  # Shape: [n_frames, 50]
                face_pose = flame_data['pose']        # Shape: [n_frames, 6] (global + jaw)
                face_shape = flame_data['shape']      # Shape: [n_frames, 100/300] (betas)
                n_face, c_face = face_expressions.shape[0], face_expressions.shape[1]

                # Process pose data
                poses = pose_data["poses"]
                n, c = poses.shape[0], poses.shape[1]

                tar_trans = pose_data["trans"]

                padded_betas = np.zeros(300)
                padded_betas[:16] = pose_data["betas"]
                betas = np.repeat(padded_betas.reshape(1, 300), n, axis=0)
                
                # Process pose data
                pose_processed = poses * JOINT_MASK_FULL
                pose_processed = pose_processed[:, JOINT_MASK_FULL.astype(bool)]
                
                if self.motion_representation == 'rotation':
                    # Calculate foot contacts
                    foot_contacts_path = pjoin(self.data_root_amass, foot_contact_path, file_name + '.npy')
                    if os.path.exists(foot_contacts_path):
                        contacts = np.load(foot_contacts_path)
                    else:
                        contacts = self.comput_foot_contacts(pose_data)
                        os.makedirs(pjoin(self.data_root_amass, foot_contact_path), exist_ok=True)
                        np.save(foot_contacts_path, contacts)
                    pose_processed = np.concatenate([pose_processed, contacts], axis=1)
                
                tar_pose = pose_processed[:, :165]

                tar_contact = contacts

                # # Extract and convert jaw pose data
                # tar_pose_face = np.zeros((n, 106))
                # tar_pose_face_with_head = np.zeros((n, 112))

                # Process head pose (positions 0:3 in pose)
                head_pose = face_pose[:, :3]  # Global head orientation in axis-angle
                head_pose_6d = axis_angle_to_6d_np(head_pose).reshape(n_face, 6)
                # Process jaw pose (positions 3:6 in pose)
                jaw_pose = face_pose[:, 3:6]  # Jaw rotation in axis-angle
                jaw_pose_6d = axis_angle_to_6d_np(jaw_pose).reshape(n_face, 6)
                
                # Pad expressions to 100 dimensions as needed
                if face_expressions.shape[1] < 100:
                    padded_exps = np.zeros((n_face, 100), dtype=face_expressions.dtype)
                    padded_exps[:, :face_expressions.shape[1]] = face_expressions
                    face_expressions = padded_exps
                elif face_expressions.shape[1] > 100:
                    face_expressions = face_expressions[:, :100]
                
                # Concatenate head pose, jaw pose and expressions for 112D face representation
                tar_pose_face = np.concatenate([jaw_pose_6d, face_expressions], axis=1)
                tar_pose_face_with_head = np.concatenate([head_pose_6d, jaw_pose_6d, face_expressions], axis=1)
                
                # Extract and convert hand pose data
                tar_pose_hands_6d = np.zeros((n, 180))
                # Extract and convert upper body pose data
                tar_pose_upper = tar_pose[:, self.joint_mask_upper.astype(bool)].reshape(n, 13, 3)
                tar_pose_upper_6d = axis_angle_to_6d_np(tar_pose_upper).reshape(n, 13 * 6)

                # Extract and convert lower body pose data
                tar_pose_leg = tar_pose[:, self.joint_mask_lower.astype(bool)].reshape(n, 9, 3)
                tar_pose_leg_6d = axis_angle_to_6d_np(tar_pose_leg).reshape(n, 9 * 6)
                
                # Combine lower body pose with translation and contacts
                tar_pose_lower = np.concatenate([tar_pose_leg_6d, tar_trans, tar_contact], axis=1)
                
                # # Convert full pose to 6D representation
                tar_pose_6d = axis_angle_to_6d_np(tar_pose.reshape(n, 55, 3)).reshape(n, 55, 6)

                # Calculate time segments
                round_seconds_skeleton = tar_pose_6d.shape[0] // pose_fps_amass
                if round_seconds_skeleton == 0:
                    round_seconds_skeleton = 1
                clip_s_t, clip_e_t = 0, round_seconds_skeleton - 0
                clip_s_f_pose, clip_e_f_pose = clip_s_t * pose_fps_amass, clip_e_t * pose_fps_amass
            
                if self.split == 'test' or self.split == 'token':  # stride = length for test
                    cut_length = clip_e_f_pose - clip_s_f_pose
                    stride = cut_length
                    # Create segments                        
                    sample_pose = tar_pose_6d[:]
                    sample_hand = tar_pose_hands_6d[:]
                    sample_upper = tar_pose_upper_6d[:]
                    sample_lower = tar_pose_lower[:]
                    sample_trans = tar_trans[:]
                    sample_shape = betas[:]
                    new_name = 'amass_' + '%s' % (file_name)
                    # Store processed data
                    self.data_dict_amass_talking[new_name] = {
                        'face': tar_pose_face,
                        'face_with_head': tar_pose_face_with_head,
                        'hand': sample_hand,
                        'upper': sample_upper,
                        'lower': sample_lower,
                        'pose': sample_pose,
                        'shape': sample_shape,
                        'trans': sample_trans,
                        'id': file_name,
                        'dataset_name': 'amass_talking',
                    }
                    self.metadata_amass_talking.append(new_name)
                else:
                    stride = int(config.stride)
                    cut_length = int(self.ori_length)
                    
                    num_subdivision = math.floor((clip_e_f_pose - clip_s_f_pose - cut_length) / stride) + 1

                    # Create segments
                    for i in range(num_subdivision):
                        start_idx = clip_s_f_pose + i * stride
                        fin_idx = start_idx + cut_length
                        
                        # Skip if out of bounds
                        if fin_idx > tar_pose_6d.shape[0]:
                            continue
                            
                        sample_pose = tar_pose_6d[start_idx:fin_idx]
                        # sample_face = tar_pose_face[start_idx:fin_idx]  # Just jaw pose, no expressions
                        # sample_face_with_head = tar_pose_face_with_head[start_idx:fin_idx]
                        sample_hand = tar_pose_hands_6d[start_idx:fin_idx]
                        sample_upper = tar_pose_upper_6d[start_idx:fin_idx]
                        sample_lower = tar_pose_lower[start_idx:fin_idx]
                        sample_trans = tar_trans[start_idx:fin_idx]
                        sample_shape = betas[start_idx:fin_idx]

                        new_name = 'amass_' + '%s_%d' % (file_name, i)

                        # Store processed data
                        self.data_dict_amass_talking[new_name] = {
                            # 'face': sample_face,
                            # 'face_with_head': sample_face_with_head,
                            'face': tar_pose_face,
                            'face_with_head': tar_pose_face_with_head,
                            'hand': sample_hand,
                            'upper': sample_upper,
                            'lower': sample_lower,
                            'pose': sample_pose,
                            'shape': sample_shape,
                            'trans': sample_trans,
                            'id': file_name,
                            'dataset_name': 'amass_talking',
                        }
                        self.metadata_amass_talking.append(new_name)
            except Exception as e:
                # print(f"Error processing file {f_name}: {str(e)}")
                continue

            # For fast debug
            if index >= self.maxdata:
                break
        
        # Save processed data to cache
        self._save_to_cache(cache_path, self.data_dict_amass_talking, self.metadata_amass_talking, "AMASS_talking")


    def _load_amass_p2p(self, config):
        """
        Load the AMASS dataset based on the configuration.

        Parameters:
        - config: Configuration dictionary for AMASS.
        """
        # Get cache path for this specific dataset
        data_root = self.args["AMASS_P2P"].ROOT
        cache_path = self._get_cache_path(data_root, "AMASS_P2P")
        pose_rep = config.pose_rep
        # pose_rep_face = config.pose_rep_face
        foot_contact_path = config.foot_contact_path
        # Check if we should load from cache
        data_dict, metadata = self._load_from_cache(cache_path, "AMASS_P2P")
        if data_dict is not None:
            self.data_dict_amass_p2p = data_dict
            self.metadata_amass_p2p = metadata
            return
        
        # We need to process data from scratch, so initialize SMPLX if needed
        self._initialize_smplx_if_needed()
        
        print(f"Processing AMASS_P2P dataset...")
        
        self.data_root_amass_p2p = data_root
        pose_fps_amass_p2p = config.pose_fps
        # split_file = pd.read_csv(pjoin(self.data_root_amass, f"new_{self.split}.csv"))

        split_file_train = pjoin(self.data_root_amass_p2p, 'train.txt')
        # Data id list
        id_list_train = []
        with cs.open(split_file_train, "r") as f:
            for line in f.readlines():
                id_list_train.append(line.strip())

        split_file_test = pjoin(self.data_root_amass_p2p, 'test.txt')
        # Data id list
        id_list_test = []
        with cs.open(split_file_test, "r") as f:
            for line in f.readlines():
                id_list_test.append(line.strip())

        if self.split == 'train':
            id_list_amass_p2p = id_list_train
        elif self.split == 'test':
            id_list_amass_p2p = id_list_test
        else:
            id_list_amass_p2p = id_list_train + id_list_test

        self.ori_length = config.pose_length
        # Calculate maximum lengths for data
        # self.max_length = int(config.pose_length)

        self.metadata_amass_p2p = []
        self.data_dict_amass_p2p = {}

        ##################  AMASS  ##################
        # Process each file
        # for index, file_name in tqdm(split_file.iterrows()):
        for index, file_name in tqdm(enumerate(id_list_amass_p2p)):
            try:

                pose_file = pjoin(self.data_root_amass_p2p, pose_rep, file_name+'.npz')
                pose_data = np.load(pose_file, allow_pickle=True)

                # Process pose data
                poses = pose_data["poses"]
                n, c = poses.shape[0], poses.shape[1]

                tar_trans = pose_data["trans"]

                padded_betas = np.zeros(300)
                padded_betas[:16] = pose_data["betas"]
                betas = np.repeat(padded_betas.reshape(1, 300), n, axis=0)
                
                # Process pose data
                pose_processed = poses * JOINT_MASK_FULL
                pose_processed = pose_processed[:, JOINT_MASK_FULL.astype(bool)]
                
                if self.motion_representation == 'rotation':
                    # Calculate foot contacts
                    foot_contacts_path = pjoin(self.data_root_amass_p2p, foot_contact_path, file_name + '.npy')
                    if os.path.exists(foot_contacts_path):
                        contacts = np.load(foot_contacts_path)
                    else:
                        contacts = self.comput_foot_contacts(pose_data)
                        os.makedirs(pjoin(self.data_root_amass_p2p, foot_contact_path), exist_ok=True)
                        np.save(foot_contacts_path, contacts)
                    pose_processed = np.concatenate([pose_processed, contacts], axis=1)
                
                tar_pose = pose_processed[:, :165]

                tar_contact = contacts

                # Extract and convert jaw pose data
                tar_pose_face = np.zeros((n, 106))
                tar_pose_face_with_head = np.zeros((n, 112))

                # Extract and convert hand pose data
                tar_pose_hands_6d = np.zeros((n, 180))
                # Extract and convert upper body pose data
                tar_pose_upper = tar_pose[:, self.joint_mask_upper.astype(bool)].reshape(n, 13, 3)
                tar_pose_upper_6d = axis_angle_to_6d_np(tar_pose_upper).reshape(n, 13 * 6)

                # Extract and convert lower body pose data
                tar_pose_leg = tar_pose[:, self.joint_mask_lower.astype(bool)].reshape(n, 9, 3)
                tar_pose_leg_6d = axis_angle_to_6d_np(tar_pose_leg).reshape(n, 9 * 6)
                
                # Combine lower body pose with translation and contacts
                tar_pose_lower = np.concatenate([tar_pose_leg_6d, tar_trans, tar_contact], axis=1)
                
                # # Convert full pose to 6D representation
                tar_pose_6d = axis_angle_to_6d_np(tar_pose.reshape(n, 55, 3)).reshape(n, 55, 6)

                # Calculate time segments
                round_seconds_skeleton = tar_pose_6d.shape[0] // pose_fps_amass_p2p
                if round_seconds_skeleton == 0:
                    round_seconds_skeleton = 1
                clip_s_t, clip_e_t = 0, round_seconds_skeleton - 0
                clip_s_f_pose, clip_e_f_pose = clip_s_t * pose_fps_amass_p2p, clip_e_t * pose_fps_amass_p2p
            
                if self.split == 'test' or self.split == 'token':  # stride = length for test
                    cut_length = clip_e_f_pose - clip_s_f_pose
                    stride = cut_length
                    sample_pose = tar_pose_6d[:]
                    # sample_face = tar_pose_face[start_idx:fin_idx]  # Just jaw pose, no expressions
                    # sample_face_with_head = tar_pose_face_with_head[start_idx:fin_idx]
                    sample_hand = tar_pose_hands_6d[:]
                    sample_upper = tar_pose_upper_6d[:]
                    sample_lower = tar_pose_lower[:]
                    sample_trans = tar_trans[:]
                    sample_shape = betas[:]

                    new_name = 'amass_' + '%s' % (file_name)
                    # Store processed data
                    self.data_dict_amass_p2p[new_name] = {
                        # 'face': sample_face,
                        # 'face_with_head': sample_face_with_head,
                        'face': tar_pose_face,
                        'face_with_head': tar_pose_face_with_head,
                        'hand': sample_hand,
                        'upper': sample_upper,
                        'lower': sample_lower,
                        'pose': sample_pose,
                        'shape': sample_shape,
                        'trans': sample_trans,
                        'id': file_name,
                        'dataset_name': 'amass_p2p',
                    }
                    self.metadata_amass_p2p.append(new_name)
                    # self.max_length = cut_length
                else:
                    stride = int(config.stride)
                    cut_length = int(self.ori_length)
                    
                    num_subdivision = math.floor((clip_e_f_pose - clip_s_f_pose - cut_length) / stride) + 1

                    # Create segments
                    for i in range(num_subdivision):
                        start_idx = clip_s_f_pose + i * stride
                        fin_idx = start_idx + cut_length
                        
                        # Skip if out of bounds
                        if fin_idx > tar_pose_6d.shape[0]:
                            continue
                            
                        sample_pose = tar_pose_6d[start_idx:fin_idx]
                        # sample_face = tar_pose_face[start_idx:fin_idx]  # Just jaw pose, no expressions
                        # sample_face_with_head = tar_pose_face_with_head[start_idx:fin_idx]
                        sample_hand = tar_pose_hands_6d[start_idx:fin_idx]
                        sample_upper = tar_pose_upper_6d[start_idx:fin_idx]
                        sample_lower = tar_pose_lower[start_idx:fin_idx]
                        sample_trans = tar_trans[start_idx:fin_idx]
                        sample_shape = betas[start_idx:fin_idx]

                        new_name = 'amass_' + '%s_%d' % (file_name, i)

                        # Store processed data
                        self.data_dict_amass_p2p[new_name] = {
                            # 'face': sample_face,
                            # 'face_with_head': sample_face_with_head,
                            'face': tar_pose_face,
                            'face_with_head': tar_pose_face_with_head,
                            'hand': sample_hand,
                            'upper': sample_upper,
                            'lower': sample_lower,
                            'pose': sample_pose,
                            'shape': sample_shape,
                            'trans': sample_trans,
                            'id': file_name,
                            'dataset_name': 'amass_p2p',
                        }
                        self.metadata_amass_p2p.append(new_name)
            except Exception as e:
                # print(f"Error processing file {f_name}: {str(e)}")
                continue

            # For fast debug
            if index >= self.maxdata:
                break
        
        # Save processed data to cache
        self._save_to_cache(cache_path, self.data_dict_amass_p2p, self.metadata_amass_p2p, "AMASS_P2P")



    def comput_foot_contacts(self, m_data):
        """
        Compute foot contacts from motion data.
        This method requires SMPLX, so we ensure it's initialized.
        """
        # Make sure SMPLX is initialized
        self._initialize_smplx_if_needed()
        
        betas, poses, trans, exps = m_data["betas"], m_data["poses"], m_data["trans"], m_data["expressions"]
        n, c = poses.shape[0], poses.shape[1]
        
        # determine the dimension of betas
        beta_dim = betas.shape[-1]  # get the last dimension of betas

        if beta_dim == 16:
            padded_betas = np.zeros(300)
            padded_betas[:16] = betas
            exps = torch.zeros([n, 100], dtype=torch.float32).cuda()  # AMASS dataset
        else:
            padded_betas = betas
            exps = torch.from_numpy(m_data["expressions"]).cuda().float()  # BEAT2 dataset

        betas = padded_betas.reshape(1, 300)  # can be 16 or 300
        betas = np.tile(betas, (n, 1))

        betas = torch.from_numpy(betas).cuda().float()
        poses = torch.from_numpy(poses.reshape(n, c)).cuda().float()
        trans = torch.from_numpy(trans.reshape(n, 3)).cuda().float()
        max_length = 128
        s, r = n // max_length, n % max_length
        all_tensor = []
        
        for i in range(s):
            with torch.no_grad():
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
                )['joints'][:, (7, 8, 10, 11), :].reshape(max_length, 4, 3).cpu()
            all_tensor.append(joints)
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
        joints = torch.cat(all_tensor, axis=0)  # all, 4, 3
        feetv = torch.zeros(joints.shape[1], joints.shape[0])
        joints = joints.permute(1, 0, 2)
        feetv[:, :-1] = (joints[:, 1:] - joints[:, :-1]).norm(dim=-1)
        contacts = (feetv < 0.01).numpy().astype(float)
        contacts = contacts.transpose(1, 0)

        return contacts

    def __len__(self):
        """
        Returns the number of samples in the dataset.
        """
        return len(self.metadata)

    def __getitem__(self, item):
        """
        Retrieves a sample from the dataset based on the given index.
        Converts NumPy arrays to PyTorch tensors for model usage.

        Parameters:
        - item: Index of the sample to retrieve.

        Returns:
        - A dictionary containing data with tensors for model input.
        """
        dataset_name = self.metadata[item]
        data = self.data_dict[dataset_name]
        
        # Create a copy of the data dictionary to avoid modifying the original
        formatted_data = {}
        
        # Convert NumPy arrays to PyTorch tensors for model usage
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                if data['dataset_name'] == 'tfhp':
                    # Convert NumPy arrays to tensors
                    # formatted_data[key] = torch.from_numpy(value).float()
                    formatted_data[key]  = torch.FloatTensor(value)
                else:
                    # Convert NumPy arrays to tensors
                    formatted_data[key] = torch.from_numpy(value).float()
            else:
                # Keep other data types as is
                formatted_data[key] = value
                
        # # Get motion length from the pose data
        # motion_len_1 = formatted_data['face_p1'].shape[0] if 'face_p1' in formatted_data else 0
        # motion_len_2 = formatted_data['face_p2'].shape[0] if 'face_p2' in formatted_data else 0
        # Get motion length from the pose data
        if 'upper' in formatted_data:
            motion_len = formatted_data['upper'].shape[0]
        else:
            motion_len = formatted_data['face'].shape[0]
        # Add additional information
        formatted_data.update({
            "id_name": formatted_data.get('id', ""),
            "dataset_name": formatted_data.get('dataset_name', ""),
            "split_name": "vq",
            "select_part": "compositional",
            "motion_len": motion_len,
            # "motion_len_1": motion_len_1,
            # "motion_len_2": motion_len_2,
        })
        
        return formatted_data