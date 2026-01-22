import os
import numpy as np
import pytorch_lightning as pl
from pathlib import Path
from tqdm import tqdm
from multimodal_tokenizers.config import parse_args
from multimodal_tokenizers.data.build_data import build_data
from multimodal_tokenizers.models.build_model import build_model
from loguru import logger
from multimodal_tokenizers.utils.load_checkpoint import load_pretrained_vae_upper, load_pretrained_vae_compositional
from multimodal_tokenizers.utils.renderer_utils import RenderMesh
from multimodal_tokenizers.utils.utils_videos import write_video
from smplx import FLAME
import torch
import cv2
import torchaudio
from multimodal_tokenizers.utils.rotation_conversions import axis_angle_to_6d, axis_angle_to_matrix, rotation_6d_to_axis_angle, axis_angle_to_6d_np


def main():
    # parse options
    cfg = parse_args(phase="test")  # parse config file
    cfg.TRAIN.STAGE = "token"
    cfg.TRAIN.BATCH_SIZE = 1

    tokenize_cfg = getattr(cfg, "TOKENIZE", None)
    show_comparison = bool(getattr(tokenize_cfg, "VISUALIZE", False)) if tokenize_cfg else False
    vis_start = int(getattr(tokenize_cfg, "VIS_START", 0)) if tokenize_cfg else 0
    vis_end = int(getattr(tokenize_cfg, "VIS_END", 0)) if tokenize_cfg else 0
    vis_len = int(getattr(tokenize_cfg, "VIS_LEN", 500)) if tokenize_cfg else 500
    vis_batch = int(getattr(tokenize_cfg, "VIS_BATCH", 50)) if tokenize_cfg else 50

    device = torch.device(f"cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    if show_comparison:
        model_path = cfg.DATASET.FLAME_MODEL_DIR
        # Initialize FLAME model with the adjusted batch size and move to GPU
        flame_model = FLAME(str(model_path), num_expression_coeffs=100, ext='pkl', batch_size=vis_batch).to(device)

    # set seed
    pl.seed_everything(cfg.SEED_VALUE)

    # gpu setting
    if cfg.ACCELERATOR == "gpu":
        os.environ["PYTHONWARNINGS"] = "ignore"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # create dataset
    datasets = build_data(cfg, phase='token')
    print("datasets module initialized")

    dataset_key_map = {
        "amass": "AMASS",
        "amass_talking": "AMASS_talking",
        "amass_p2p": "AMASS_P2P",
        "beat2": "BEAT2",
        "candor": "CANDOR",
        "tfhp": "TFHP",
        "YouTube_Talking": "YouTube_Talking",
        "YouTube_Talking_Synthetic": "YouTube_Talking_Synthetic",
    }
    allowed_dataset_names = set()
    for config in cfg.DATASET.datasets:
        config_name = config.get("name")
        if config_name == "BEAT2":
            allowed_dataset_names.add("beat2")
        elif config_name == "AMASS":
            allowed_dataset_names.add("amass")
        elif config_name == "AMASS_talking":
            allowed_dataset_names.add("amass_talking")
        elif config_name == "AMASS_P2P":
            allowed_dataset_names.add("amass_p2p")
        elif config_name == "CANDOR":
            allowed_dataset_names.add("candor")
        elif config_name == "TFHP":
            allowed_dataset_names.add("tfhp")
        elif config_name == "YouTube_Talking":
            allowed_dataset_names.add("YouTube_Talking")
        elif config_name == "YouTube_Talking_Synthetic":
            allowed_dataset_names.add("YouTube_Talking_Synthetic")
    output_roots = {}
    dataset_modalities_map = {}
    code_path = cfg.DATASET.code_path
    saved_counts = {}
    modalities_cfg = getattr(cfg.DATASET, "MODALITIES", None)
    default_modalities = {"face", "upper", "lower", "hand"}
    warned_modalities = set()
    warned_missing_parts = set()

    def ensure_output_dirs(base_dir, modalities):
        for part in modalities:
            os.makedirs(os.path.join(base_dir, part), exist_ok=True)

    def normalize_modalities(value):
        if not value:
            return None
        return {str(modality).lower() for modality in value}

    def get_modalities(dataset_name):
        dataset_key = dataset_key_map.get(dataset_name)
        if dataset_key is None or modalities_cfg is None:
            return default_modalities
        dataset_modalities = normalize_modalities(modalities_cfg.get(dataset_key))
        if not dataset_modalities:
            if dataset_name not in warned_modalities:
                logger.warning(
                    f"No modalities configured for {dataset_name}; defaulting to {sorted(default_modalities)}."
                )
                warned_modalities.add(dataset_name)
            return default_modalities
        return dataset_modalities

    def normalize_face_tensor(face_tensor, target_dim):
        current_dim = face_tensor.shape[-1]
        if current_dim == target_dim:
            return face_tensor
        if current_dim > target_dim:
            return face_tensor[..., :target_dim]
        pad_dim = target_dim - current_dim
        pad_shape = list(face_tensor.shape[:-1]) + [pad_dim]
        pad = torch.zeros(pad_shape, device=face_tensor.device, dtype=face_tensor.dtype)
        return torch.cat([pad, face_tensor], dim=-1)
    # Model
    model = build_model(cfg)
    logger.info("model {} loaded".format(cfg.model.target))

    # load_pretrained_vae_face(cfg, model, logger, phase="token")
    # load_pretrained_vae_upper(cfg, model, logger, phase="token")
    load_pretrained_vae_compositional(cfg, model, logger, phase="token")

    if cfg.ACCELERATOR == "gpu":
        model.vae_face.to(device)
        model.vae_upper.to(device)
        model.vae_lower.to(device)
        model.vae_hand.to(device)

    model.vae_face.eval()
    model.vae_upper.eval()
    model.vae_lower.eval()
    model.vae_hand.eval()
    logger.info("model loaded")

    for batch in tqdm(datasets.token_dataloader(), desc=f'compositional motion tokenize'):
        seq_name = batch["id_name"][0]
        dataset_name = batch["dataset_name"][0]
        if allowed_dataset_names and dataset_name not in allowed_dataset_names:
            logger.warning(f"Skipping {seq_name} from {dataset_name}: not enabled in config datasets.")
            continue
        dataset_key = dataset_key_map.get(dataset_name)
        if dataset_key is None:
            logger.warning(f"Skipping {seq_name}: unknown dataset '{dataset_name}'.")
            continue
        dataset_cfg = cfg.DATASET.get(dataset_key)
        if dataset_cfg is None:
            logger.warning(f"Skipping {seq_name}: missing DATASET.{dataset_key} config.")
            continue

        dataset_modalities = dataset_modalities_map.get(dataset_name)
        if dataset_modalities is None:
            dataset_modalities = get_modalities(dataset_name)
            dataset_modalities_map[dataset_name] = dataset_modalities

        output_root = output_roots.get(dataset_name)
        if output_root is None:
            output_root = os.path.join(dataset_cfg.ROOT, code_path)
            ensure_output_dirs(output_root, dataset_modalities)
            output_roots[dataset_name] = output_root

        saved_any = False
        vis_face_tokens = None
        vis_face_gt = None
        if "upper" in dataset_modalities:
            if "upper" not in batch:
                missing_key = (dataset_name, "upper")
                if missing_key not in warned_missing_parts:
                    logger.warning(f"Missing upper in {dataset_name} sample {seq_name}.")
                    warned_missing_parts.add(missing_key)
            else:
                upper = batch["upper"].to(device)
                tar_index_value_upper_top = model.vae_upper.map2index(upper)  # bs*n/4
                target_path_upper = os.path.join(output_root, 'upper', seq_name + '.npy')
                Path(target_path_upper).parent.mkdir(parents=True, exist_ok=True)
                np.save(target_path_upper, tar_index_value_upper_top.to('cpu').numpy())
                saved_any = True
        if "lower" in dataset_modalities:
            if "lower" not in batch:
                missing_key = (dataset_name, "lower")
                if missing_key not in warned_missing_parts:
                    logger.warning(f"Missing lower in {dataset_name} sample {seq_name}.")
                    warned_missing_parts.add(missing_key)
            else:
                lower = batch["lower"].to(device)
                tar_index_value_lower_top = model.vae_lower.map2index(lower[:, :, :54])  # bs*n/4
                target_path_lower = os.path.join(output_root, 'lower', seq_name + '.npy')
                Path(target_path_lower).parent.mkdir(parents=True, exist_ok=True)
                np.save(target_path_lower, tar_index_value_lower_top.to('cpu').numpy())
                saved_any = True
        if "hand" in dataset_modalities:
            if "hand" not in batch:
                missing_key = (dataset_name, "hand")
                if missing_key not in warned_missing_parts:
                    logger.warning(f"Missing hand in {dataset_name} sample {seq_name}.")
                    warned_missing_parts.add(missing_key)
            else:
                hand = batch["hand"].to(device)
                tar_index_value_hand_top = model.vae_hand.map2index(hand)  # bs*n/4
                target_path_hand = os.path.join(output_root, 'hand', seq_name + '.npy')
                Path(target_path_hand).parent.mkdir(parents=True, exist_ok=True)
                np.save(target_path_hand, tar_index_value_hand_top.to('cpu').numpy())
                saved_any = True
        if "face" in dataset_modalities:
            face_entries = []
            if "face_with_head" in batch:
                face_entries.append(("face", batch["face_with_head"], None))
            elif "face" in batch:
                face_entries.append(("face", batch["face"], None))
            else:
                if "face_p1" in batch:
                    name_suffix = batch.get("p1_name")
                    if isinstance(name_suffix, (list, tuple)):
                        name_suffix = name_suffix[0]
                    face_entries.append(("face_p1", batch["face_p1"], name_suffix))
                if "face_p2" in batch:
                    name_suffix = batch.get("p2_name")
                    if isinstance(name_suffix, (list, tuple)):
                        name_suffix = name_suffix[0]
                    face_entries.append(("face_p2", batch["face_p2"], name_suffix))
            if not face_entries:
                missing_key = (dataset_name, "face")
                if missing_key not in warned_missing_parts:
                    logger.warning(f"Missing face in {dataset_name} sample {seq_name}.")
                    warned_missing_parts.add(missing_key)
            else:
                for _, face_tensor, name_suffix in face_entries:
                    face_tensor = face_tensor.to(device)
                    face_tensor = normalize_face_tensor(face_tensor, 112)
                    tar_index_value_face_top = model.vae_face.map2index(face_tensor)  # bs*n/4
                    target_name = seq_name
                    if name_suffix:
                        target_name = f"{seq_name}_{name_suffix}"
                    target_path_face = os.path.join(output_root, 'face', target_name + '.npy')
                    Path(target_path_face).parent.mkdir(parents=True, exist_ok=True)
                    np.save(target_path_face, tar_index_value_face_top.to('cpu').numpy())
                    saved_any = True
                    if show_comparison and vis_face_tokens is None:
                        vis_face_tokens = tar_index_value_face_top
                        vis_face_gt = face_tensor
        if not saved_any:
            logger.warning(f"Skipping {seq_name} from {dataset_name}: no tokens saved.")
            continue
        rec_face_test = None
        if show_comparison and vis_face_tokens is not None:
            rec_face_test = model.vae_face.decode(vis_face_tokens.int())
        # # Extract components similar to the original FLAME format
        # rec_head_pose = rotation_6d_to_axis_angle(rec_upper_test[0, :, :6]).detach().cpu().numpy()
        # rec_jaw_pose = rotation_6d_to_axis_angle(rec_upper_test[0, :, 6:12]).detach().cpu().numpy()
        # rec_exp = rec_upper_test[0, :, 12:62].detach().cpu().numpy()  # First 50 dims of expression
        

        if show_comparison:
            if rec_face_test is None:
                logger.warning(f"Skip visualization for {seq_name}: face data unavailable.")
                continue
            faces = torch.LongTensor(flame_model.faces.astype(np.int64))
            mesh_renderer = RenderMesh(image_size=256, faces=faces, scale=1.0)
            audio_clip = None
            if dataset_name == "YouTube_Talking":
                audio_path = os.path.join(
                    cfg.DATASET["YouTube_Talking"].ROOT,
                    "audios_original",
                    f"{seq_name}.wav",
                )
                if os.path.exists(audio_path):
                    audio, sr = torchaudio.load(audio_path)
                    audio = torchaudio.transforms.Resample(sr, 16000)(audio).mean(dim=0)
                    audio_clip = audio
            rec_face = rec_face_test[0]
            gt_face = vis_face_gt[0]
            rec_face_head = rotation_6d_to_axis_angle(rec_face[:, :6])
            rec_face_jaw = rotation_6d_to_axis_angle(rec_face[:, 6:12])
            rec_exp = rec_face[:, 12:]
            gt_face_head = rotation_6d_to_axis_angle(gt_face[:, :6])
            gt_face_jaw = rotation_6d_to_axis_angle(gt_face[:, 6:12])
            gt_exp = gt_face[:, 12:]

            pred_images = []
            seq_len = rec_face.shape[0]
            start_frame = max(0, min(vis_start, seq_len))
            end_frame = vis_end if vis_end > 0 else start_frame + vis_len
            end_frame = min(end_frame, seq_len)
            if end_frame <= start_frame:
                logger.warning(f"Skip visualization for {seq_name}: invalid frame range.")
                continue

            for i in range(start_frame, end_frame, vis_batch):
                actual_visualize_batch_size = min(vis_batch, end_frame - i)

                # Run FLAME model for both reconstructed and ground truth
                with torch.no_grad():
                    flame_out = flame_model(
                        global_orient=rec_face_head[i:i+actual_visualize_batch_size, :],
                        expression=rec_exp[i:i+actual_visualize_batch_size, :],
                        jaw_pose=rec_face_jaw[i:i+actual_visualize_batch_size, :],
                        shape=torch.zeros(actual_visualize_batch_size, 100).to(device),
                    )
                    flame_out_gt = flame_model(
                        global_orient=gt_face_head[i:i+actual_visualize_batch_size, :],
                        expression=gt_exp[i:i+actual_visualize_batch_size, :],
                        jaw_pose=gt_face_jaw[i:i+actual_visualize_batch_size, :],
                        shape=torch.zeros(actual_visualize_batch_size, 100).to(device),
                    )
                
                verts = flame_out['vertices'].detach()
                verts_gt = flame_out_gt['vertices'].detach()

                # Render meshes
                render_output = mesh_renderer(verts)
                render_output_gt = mesh_renderer(verts_gt)
                
                image_renders = render_output[0] / 255.0
                image_renders_gt = render_output_gt[0] / 255.0
                
                # Process each frame in the batch
                for b in range(actual_visualize_batch_size):
                    frame_idx = i + b
                    
                    # Get current frame's rendered images
                    image_render = image_renders[b]
                    image_render_gt = image_renders_gt[b]
                    
                    # Convert tensors to numpy and rearrange dimensions
                    image_render_np = image_render.cpu().numpy()
                    image_render_np = np.transpose(image_render_np, (1, 2, 0))  # Convert from [C,H,W] to [H,W,C]
                    
                    image_render_gt_np = image_render_gt.cpu().numpy()
                    image_render_gt_np = np.transpose(image_render_gt_np, (1, 2, 0))  # Convert from [C,H,W] to [H,W,C]
                    
                    # Get dimensions for resizing
                    render_height = image_render_np.shape[0]
                    render_width = image_render_np.shape[1]
                    
                    # Resize the rendered images to the same height if needed (here assumed already same)
                    # Create a combined image (side by side: reconstructed, ground truth)
                    combined_width = render_width * 2
                    combined_image = np.zeros((render_height, combined_width, 3))
                    combined_image[:, :render_width, :] = image_render_np
                    combined_image[:, render_width:, :] = image_render_gt_np
                    
                    # Convert to tensor and append to results
                    combined_image = np.asarray(combined_image).astype(np.float32)
                    combined_image = torch.FloatTensor(combined_image)

                    # combined_image = torch.from_numpy(combined_image).float()
                    combined_image = combined_image.permute(2, 0, 1)  # [H,W,C] -> [C,H,W]
                    pred_images.append(combined_image.unsqueeze(0))  # Add batch dimension



            if not pred_images:
                logger.warning(f"Skip visualization for {seq_name}: no frames rendered.")
                continue
            pred_images_tensor = torch.cat(pred_images, dim=0).cpu()
            # Use sequence and video name in the output filename
            dataset_root = cfg.DATASET[dataset_key_map[dataset_name]].ROOT
            output_dir_video = os.path.join(dataset_root, 'reconstructed_vs_gt')
            os.makedirs(output_dir_video, exist_ok=True)
            dump_path = os.path.join(output_dir_video, f"{seq_name.replace('/', '_')}.mp4")
            
            print(f"Saving video to: {dump_path}")
            if audio_clip is not None:
                audio_clip = audio_clip[int(start_frame/25.0*16000):int(end_frame/25.0*16000)]
                write_video(pred_images_tensor*255.0, dump_path, 25, audio_clip, 16000, "aac")
            else:
                write_video(pred_images_tensor*255.0, dump_path, 25)
            print("Video saved successfully!")

        saved_counts[dataset_name] = saved_counts.get(dataset_name, 0) + 1


    if output_roots:
        output_summary = ", ".join(f"{name}:{path}" for name, path in output_roots.items())
        print(f"Motion tokenization done, outputs: {output_summary}")
    if saved_counts:
        count_summary = ", ".join(f"{name}:{count}" for name, count in saved_counts.items())
        print(f"Tokenized samples per dataset: {count_summary}")


if __name__ == "__main__":
    main()
