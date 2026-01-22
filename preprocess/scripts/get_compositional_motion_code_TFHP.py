import os
import numpy as np
import pytorch_lightning as pl
from pathlib import Path
from tqdm import tqdm
from conver_agent.config import parse_args
from conver_agent.data.build_data import build_data
from conver_agent.models.build_model import build_model
from loguru import logger
from conver_agent.utils.load_checkpoint import load_pretrained_vae_face
from conver_agent.utils.renderer_utils import RenderMesh
from conver_agent.utils.utils_videos import write_video
from smplx import FLAME
import torch
import cv2
import torchaudio
from conver_agent.utils.rotation_conversions import axis_angle_to_6d, axis_angle_to_matrix, rotation_6d_to_axis_angle, axis_angle_to_6d_np


def load_video_frames(video_path, start_frame, end_frame, target_height=256, target_width=256):
    """
    Load video frames from the specified video file while preserving aspect ratio
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Warning: Could not open video file {video_path}")
        return None
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video info: {total_frames} frames, {fps} FPS, resolution: {original_width}x{original_height}")
    
    # Calculate aspect ratio preserving dimensions
    aspect_ratio = original_width / original_height
    
    if aspect_ratio > 1:  # Landscape video
        # Scale based on height, but ensure width doesn't exceed target
        new_height = target_height
        new_width = int(target_height * aspect_ratio)
        
        # If width exceeds target, scale based on width instead
        if new_width > target_width:
            new_width = target_width
            new_height = int(target_width / aspect_ratio)
    else:  # Portrait or square video
        # Scale based on width, but ensure height doesn't exceed target
        new_width = target_width
        new_height = int(target_width / aspect_ratio)
        
        # If height exceeds target, scale based on height instead
        if new_height > target_height:
            new_height = target_height
            new_width = int(target_height * aspect_ratio)
    
    print(f"Scaling to: {new_width}x{new_height} (aspect ratio: {aspect_ratio:.2f})")
    
    frames = []
    for frame_idx in range(start_frame, min(end_frame, total_frames)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Resize preserving aspect ratio
            frame = cv2.resize(frame, (new_width, new_height))
            
            # Create a canvas with target dimensions and place the frame in the center
            canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
            
            # Calculate position to center the frame
            y_offset = max(0, (target_height - new_height) // 2)
            x_offset = max(0, (target_width - new_width) // 2)
            
            # Ensure the frame fits within the canvas
            y_end = min(y_offset + new_height, target_height)
            x_end = min(x_offset + new_width, target_width)
            frame_h = y_end - y_offset
            frame_w = x_end - x_offset
            
            # Ensure we don't try to access frame areas that don't exist
            frame_y_end = min(new_height, frame_h)
            frame_x_end = min(new_width, frame_w)
            
            canvas[y_offset:y_offset+frame_h, x_offset:x_offset+frame_w] = frame[:frame_y_end, :frame_x_end]
            
            # Normalize to [0, 1]
            frame = canvas.astype(np.float32) / 255.0
            frames.append(frame)
        else:
            # If frame reading fails, create a black frame
            black_frame = np.zeros((target_height, target_width, 3), dtype=np.float32)
            frames.append(black_frame)
    
    cap.release()
    return np.array(frames)


def main():
    # parse options
    cfg = parse_args(phase="test")  # parse config file
    cfg.TRAIN.STAGE = "token"
    cfg.TRAIN.BATCH_SIZE = 1

    show_comparison = False

    device = torch.device(f"cuda:0") if torch.cuda.is_available() else torch.device("cpu")


    if show_comparison:
        model_path = "./model_files/FLAME2020/"
        batch_size_visualize = 50
        # We'll initialize FLAME model dynamically based on actual batch size needed
        flame_model = None
        start_frame = 0
        end_frame = 100

    # set seed
    pl.seed_everything(cfg.SEED_VALUE)

    # gpu setting
    if cfg.ACCELERATOR == "gpu":
        os.environ["PYTHONWARNINGS"] = "ignore"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # create dataset
    datasets = build_data(cfg, phase='token')
    print("datasets module initialized")

    # Load each dataset based on its type from the configuration
    for config in cfg.DATASET.datasets:
        dataset_name = config.get("name")
        code_path = config.get("code_path")
        if dataset_name == "AMASS":
            data_root_amass = cfg.DATASET["AMASS"].ROOT 
            output_dir_amass = os.path.join(data_root_amass, code_path)
            os.makedirs(output_dir_amass, exist_ok=True)
        if dataset_name == "BEAT2":
            data_root_beat2 = cfg.DATASET["BEAT2"].ROOT
            output_dir_beat2 = os.path.join(data_root_beat2, code_path)
            os.makedirs(output_dir_beat2, exist_ok=True)
        if dataset_name == "CANDOR":
            data_root_candor = cfg.DATASET["CANDOR"].ROOT
            output_root_candor = os.path.join(data_root_candor, code_path)
            os.makedirs(output_root_candor, exist_ok=True)
            output_root_reconstructed = os.path.join(data_root_candor, 'reconstructed')
            os.makedirs(output_root_reconstructed, exist_ok=True)
        if dataset_name == "TFHP":
            data_root_tfhp = cfg.DATASET["TFHP"].ROOT
            output_root_tfhp = os.path.join(data_root_tfhp, code_path)
            os.makedirs(output_root_tfhp, exist_ok=True)
            output_root_reconstructed = os.path.join(data_root_tfhp, 'reconstructed')
            os.makedirs(output_root_reconstructed, exist_ok=True)
    # Model
    # model = build_model(cfg, datasets)
    model = build_model(cfg)
    logger.info("model {} loaded".format(cfg.model.target))

    load_pretrained_vae_face(cfg, model, logger, phase="token")

    if cfg.ACCELERATOR == "gpu":
        model.vae_face.to(device)

    model.vae_face.eval()

    logger.info("model loaded")

    for batch in tqdm(datasets.token_dataloader(), desc=f'compositional motion tokenize'):

        seq_name =  batch["id_name"][0]
        # if seq_name != '2c81c2ee-4b36-4de3-9463-c1fb6dd33899':
        #     continue

        dataset_name =  batch["dataset_name"][0]        

        if dataset_name == 'amass':
            output_dir = output_dir_amass
        elif dataset_name == 'candor':
            output_dir = output_root_candor
            p1_name = batch["p1_name"][0]
            p2_name = batch["p2_name"][0]
            output_dir_candor = os.path.join(output_dir, seq_name)
            os.makedirs(output_dir_candor, exist_ok=True)
            output_dir_reconstructed = os.path.join(output_root_reconstructed, seq_name)
            os.makedirs(output_dir_reconstructed, exist_ok=True)
        elif dataset_name == 'tfhp':
            output_dir = output_root_tfhp
            output_dir_tfhp = os.path.join(output_dir, seq_name)
            os.makedirs(output_dir_tfhp, exist_ok=True)
            output_dir_reconstructed = os.path.join(output_root_reconstructed, seq_name)
            os.makedirs(output_dir_reconstructed, exist_ok=True)
        else:
            output_dir = output_dir_beat2

        # face_p1 = [
        #     batch[key].to(device) for key in ["face_p1"]
        # ]
        face = batch["face"].to(device)
        face_with_head = batch["face_with_head"].to(device)
        # Get the actual sequence length from the face data
        actual_sequence_length = face_with_head.shape[1]
        
        tar_index_value_face_top_p1 = model.vae_face.map2index(face_with_head)  # bs*n/4
        rec_face_test_p1 = model.vae_face.decode(tar_index_value_face_top_p1.int())

        
        # Extract components similar to the original FLAME format
        rec_head_pose_p1 = rotation_6d_to_axis_angle(rec_face_test_p1[0, :, :6]).detach().cpu().numpy()
        rec_jaw_pose_p1 = rotation_6d_to_axis_angle(rec_face_test_p1[0, :, 6:12]).detach().cpu().numpy()
        rec_exp_p1 = rec_face_test_p1[0, :, 12:62].detach().cpu().numpy()  # First 50 dims of expression
        
        # # Save in the same format as the example
        # np.savez(output_dir_reconstructed + ".npz", 
        #             pose=np.concatenate([rec_head_pose_p1, rec_jaw_pose_p1], axis=1),
        #             exp=rec_exp_p1,
        #             shape=np.zeros((rec_exp_p1.shape[0], 100)))  # Placeholder for shape
        

        if show_comparison:
            # Parse seq_name to get correct video path and frame range
            # seq_name format: TH_00046/000/013
            seq_parts = seq_name.split('/')
            if len(seq_parts) == 3:
                video_dir = seq_parts[0]  # TH_00046
                video_id = seq_parts[1]   # 000
                segment_id = int(seq_parts[2])  # 013
                
                # Construct the original video path
                video_path = f"/simurgh/u/juze/datasets/TFHP/data/{video_dir}/{video_id}.mp4"
                
                # Calculate the frame range for this segment
                segment_start_frame = segment_id * 100
                
                # Adjust start_frame and end_frame to be relative to the segment
                actual_start_frame = segment_start_frame + start_frame
                # Use the actual sequence length to determine the correct end frame
                actual_end_frame = segment_start_frame + min(end_frame, actual_sequence_length)
                
                print(f"Processing {seq_name}: video={video_path}, segment={segment_id}, actual_seq_len={actual_sequence_length}, frames={actual_start_frame}-{actual_end_frame}")
            else:
                # Fallback to original behavior if seq_name format is unexpected
                video_path = f"/simurgh/u/juze/datasets/TFHP/data/{seq_name}.mp4"
                actual_start_frame = start_frame
                actual_end_frame = end_frame
                print(f"Warning: Unexpected seq_name format: {seq_name}, using fallback path")
            
            # Check if original video exists, skip if not
            if not os.path.exists(video_path):
                print(f"Warning: Video file {video_path} does not exist. Skipping visualization for {seq_name}")
                continue
            
            audio_path = f"/simurgh/u/juze/datasets/TFHP/audios/{seq_name}.wav"
            audio, sr = torchaudio.load(audio_path)
            audio = torchaudio.transforms.Resample(sr, 16000)(audio).mean(dim=0)
            
            # Load original video frames with correct frame range
            original_frames = load_video_frames(video_path, actual_start_frame, actual_end_frame, target_height=256, target_width=256)
            
            rec_face_head_p1 = rotation_6d_to_axis_angle(rec_face_test_p1[0, :,:6])  # head pose
            rec_face_yaw_p1 = rotation_6d_to_axis_angle(rec_face_test_p1[0, :,6:12])
            rec_exp_p1 = rec_face_test_p1[0, :, 12:]
            gt_face_head_p1 = rotation_6d_to_axis_angle(face_with_head[0, :,:6])
            gt_face_yaw_p1 = rotation_6d_to_axis_angle(face_with_head[0, :,6:12])
            gt_face_exp_p1 = face_with_head[0, :, 12:]

            pred_images = []

            # Use the actual sequence length to limit the visualization range
            visualization_end_frame = min(end_frame, actual_sequence_length)
            
            # Initialize FLAME model with the correct batch size for this sequence
            current_batch_size = min(batch_size_visualize, visualization_end_frame - start_frame)
            if flame_model is None or flame_model.batch_size != current_batch_size:
                print(f"Initializing FLAME model with batch_size={current_batch_size}")
                flame_model = FLAME(model_path, num_expression_coeffs=100, ext='pkl', batch_size=current_batch_size).to(device)
                faces = torch.LongTensor(flame_model.faces.astype(np.int64))
                mesh_renderer = RenderMesh(image_size=256, faces=faces, scale=1.0)
            
            # Process in batches for efficiency
            for i in range(start_frame, visualization_end_frame, batch_size_visualize):
                actual_visualize_batch_size = min(batch_size_visualize, visualization_end_frame - i)
                
                # If the actual batch size is different from current FLAME model batch size, reinitialize
                if flame_model.batch_size != actual_visualize_batch_size:
                    print(f"Reinitializing FLAME model with batch_size={actual_visualize_batch_size}")
                    flame_model = FLAME(model_path, num_expression_coeffs=100, ext='pkl', batch_size=actual_visualize_batch_size).to(device)
                    faces = torch.LongTensor(flame_model.faces.astype(np.int64))
                    mesh_renderer = RenderMesh(image_size=256, faces=faces, scale=1.0)
                
                # Prepare batch data
                batch_rec_head = rec_face_head_p1[i:i+actual_visualize_batch_size, :]
                batch_rec_exp = rec_exp_p1[i:i+actual_visualize_batch_size, :]
                batch_rec_jaw = rec_face_yaw_p1[i:i+actual_visualize_batch_size, :]
                
                batch_gt_head = gt_face_head_p1[i:i+actual_visualize_batch_size, :]
                batch_gt_exp = gt_face_exp_p1[i:i+actual_visualize_batch_size, :]
                batch_gt_jaw = gt_face_yaw_p1[i:i+actual_visualize_batch_size, :]
                
                # Run FLAME model for both reconstructed and ground truth
                with torch.no_grad():
                    try:
                        flame_out = flame_model(
                            global_orient=batch_rec_head,
                            expression=batch_rec_exp,
                            jaw_pose=batch_rec_jaw,
                            shape=torch.zeros(actual_visualize_batch_size, 100).to(device),
                        )
                        flame_out_gt = flame_model(
                            global_orient=batch_gt_head,
                            expression=batch_gt_exp,
                            jaw_pose=batch_gt_jaw,
                            shape=torch.zeros(actual_visualize_batch_size, 100).to(device),
                        )
                    except RuntimeError as e:
                        print(f"Error in FLAME model with batch_size {actual_visualize_batch_size}: {e}")
                        print(f"Tensor shapes - rec_head: {batch_rec_head.shape}, rec_exp: {batch_rec_exp.shape}, rec_jaw: {batch_rec_jaw.shape}")
                        print(f"Tensor shapes - gt_head: {batch_gt_head.shape}, gt_exp: {batch_gt_exp.shape}, gt_jaw: {batch_gt_jaw.shape}")
                        # Skip this batch and continue with the next one
                        continue
                
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
                    
                    # Get original video frame
                    if original_frames is not None and frame_idx < len(original_frames):
                        original_frame = original_frames[frame_idx]
                    else:
                        # Create a black frame if original video is not available
                        original_frame = np.zeros((256, 256, 3), dtype=np.float32)
                    
                    # Get dimensions for resizing
                    render_height = image_render_np.shape[0]
                    render_width = image_render_np.shape[1]
                    
                    # Create a combined image (side by side: reconstructed, ground truth, original video)
                    combined_width = render_width * 3  # Changed from 2 to 3 for three images
                    combined_image = np.zeros((render_height, combined_width, 3))
                    combined_image[:, :render_width, :] = image_render_np  # Reconstructed
                    combined_image[:, render_width:render_width*2, :] = image_render_gt_np  # Ground truth
                    combined_image[:, render_width*2:, :] = original_frame  # Original video
                    
                    # Convert to tensor and append to results
                    combined_image = np.asarray(combined_image).astype(np.float32)
                    combined_image = torch.FloatTensor(combined_image)

                    # combined_image = torch.from_numpy(combined_image).float()
                    combined_image = combined_image.permute(2, 0, 1)  # [H,W,C] -> [C,H,W]
                    pred_images.append(combined_image.unsqueeze(0))  # Add batch dimension


            pred_images_tensor = torch.cat(pred_images, dim=0).cpu()
            # Use sequence and video name in the output filename
            output_dir_video = os.path.join(data_root_tfhp, 'reconstructed_vs_gt_512_512_DS1_wo_meshloss')  # Updated directory name
            # output_dir_video = os.path.join(data_root_tfhp, 'reconstructed_vs_gt_512_512_DS1')  # Updated directory name
            # output_dir_video = os.path.join(data_root_tfhp, 'reconstructed_vs_gt_256_256_DS4')  # Updated directory name
            os.makedirs(output_dir_video, exist_ok=True)
            dump_path = os.path.join(output_dir_video, f"{seq_name.replace('/', '_')}.mp4")
            if os.path.exists(dump_path):
                continue
            
            print(f"Saving video to: {dump_path}")
            # Use the actual visualization range for audio clipping
            audio_clip = audio[int(start_frame/25.0*16000):int(visualization_end_frame/25.0*16000)]
            write_video(pred_images_tensor*255.0, dump_path, 25, audio_clip, 16000, "aac")
            print("Video saved successfully!")

        target_path_face_p1 = os.path.join(output_dir_tfhp + '.npy')
        Path(target_path_face_p1).parent.mkdir(parents=True, exist_ok=True)
        np.save(target_path_face_p1, tar_index_value_face_top_p1.to('cpu').numpy())


    print(
        f'Motion tokenization done, the motion tokens are saved to {output_dir}'
    )


if __name__ == "__main__":
    main()
