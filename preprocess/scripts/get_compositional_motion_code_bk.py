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


def main():
    # parse options
    cfg = parse_args(phase="test")  # parse config file
    cfg.TRAIN.STAGE = "token"
    cfg.TRAIN.BATCH_SIZE = 1

    show_comparison = True

    device = torch.device(f"cuda:2") if torch.cuda.is_available() else torch.device("cpu")


    if show_comparison:
        model_path = "./model_files/FLAME2020/"
        batch_size_visualize = 100
        # Initialize FLAME model with the adjusted batch size and move to GPU
        flame_model = FLAME(model_path, num_expression_coeffs=100, ext='pkl', batch_size=batch_size_visualize).to(device)
        start_frame = 40000
        end_frame = 40100

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
            output_dir_candor = os.path.join(data_root_candor, code_path)
            os.makedirs(output_dir_candor, exist_ok=True)

    # Model
    model = build_model(cfg, datasets)
    logger.info("model {} loaded".format(cfg.model.target))

    load_pretrained_vae_face(cfg, model, logger, phase="token")

    if cfg.ACCELERATOR == "gpu":
        model.vae_face.to(device)

    model.vae_face.eval()

    logger.info("model loaded")

    for batch in tqdm(datasets.token_dataloader(), desc=f'compositional motion tokenize'):

        seq_name =  batch["id_name"][0]
        dataset_name =  batch["dataset_name"][0]        

        if dataset_name == 'amass':
            output_dir = output_dir_amass
        elif dataset_name == 'candor':
            output_dir = output_dir_candor
            p1_name = batch["p1_name"][0]
            p2_name = batch["p2_name"][0]
            output_dir_candor = os.path.join(output_dir, seq_name)
            os.makedirs(output_dir_candor, exist_ok=True)
        else:
            output_dir = output_dir_beat2

        face_p1, face_p2 = [
            batch[key].to(device) for key in ["face_p1","face_p2"]
        ]

        tar_index_value_face_top_p1 = model.vae_face.map2index(face_p1)  # bs*n/4
        tar_index_value_face_top_p2 = model.vae_face.map2index(face_p2)  # bs*n/4
        rec_face_test_p1 = model.vae_face.decode(tar_index_value_face_top_p1.int())
        rec_face_test_p2 = model.vae_face.decode(tar_index_value_face_top_p2.int())

        if show_comparison:

            image_folder = f"/simurgh/group/yuheng/CANDOR_spectre_mica/SPECTRE_coeffs/{seq_name}/{p1_name}"
            mesh_renderer = RenderMesh(image_size=256, faces=flame_model.faces, scale=1.0)
            audio_path = f"/simurgh/group/yuheng/CANDOR_processed/{seq_name}/{p1_name}.mp3"
            audio, sr = torchaudio.load(audio_path)
            audio = torchaudio.transforms.Resample(sr, 16000)(audio).mean(dim=0)
            rec_face_head_p1 = rotation_6d_to_axis_angle(rec_face_test_p1[0, :,:6])
            rec_face_yaw_p1 = rotation_6d_to_axis_angle(rec_face_test_p1[0, :,6:12])
            rec_exp_p1 = rec_face_test_p1[0, :, 12:]
            gt_face_head_p1 = rotation_6d_to_axis_angle(face_p1[0, :,:6])
            gt_face_yaw_p1 = rotation_6d_to_axis_angle(face_p1[0, :,6:12])
            gt_face_exp_p1 = face_p1[0, :, 12:]

            pred_images = []

            for i in range(start_frame, end_frame, batch_size_visualize):
                actual_visualize_batch_size = min(batch_size_visualize, end_frame - i)

                # Run FLAME model for both reconstructed and ground truth
                with torch.no_grad():
                    flame_out = flame_model(
                        global_orient=rec_face_head_p1[i:i+actual_visualize_batch_size, :],
                        expression=rec_exp_p1[i:i+actual_visualize_batch_size, :],
                        jaw_pose=rec_face_yaw_p1[i:i+actual_visualize_batch_size, :],
                        shape=torch.zeros(actual_visualize_batch_size, 100).to(device),
                    )
                    flame_out_gt = flame_model(
                        global_orient=gt_face_head_p1[i:i+actual_visualize_batch_size, :],
                        expression=gt_face_exp_p1[i:i+actual_visualize_batch_size, :],
                        jaw_pose=gt_face_yaw_p1[i:i+actual_visualize_batch_size, :],
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
                    
                    # Get the original image
                    image_path = os.path.join(image_folder, f'{frame_idx:06d}.jpg')
                    
                    # Check if the image file exists
                    if not os.path.exists(image_path):
                        image_original = np.zeros((256, 256, 3), dtype=np.uint8)
                    else:
                        image_original = cv2.imread(image_path)
                        if image_original is None:
                            image_original = np.zeros((256, 256, 3), dtype=np.uint8)
                    
                    # Get current frame's rendered images
                    image_render = image_renders[b]
                    image_render_gt = image_renders_gt[b]
                    
                    # Convert tensors to numpy and rearrange dimensions
                    image_render_np = image_render.cpu().numpy()
                    image_render_np = np.transpose(image_render_np, (1, 2, 0))  # Convert from [C,H,W] to [H,W,C]
                    
                    image_render_gt_np = image_render_gt.cpu().numpy()
                    image_render_gt_np = np.transpose(image_render_gt_np, (1, 2, 0))  # Convert from [C,H,W] to [H,W,C]
                    
                    # Get dimensions for resizing
                    original_height = image_original.shape[0]
                    render_height = image_render_np.shape[0]
                    
                    # Calculate new width to maintain aspect ratio
                    new_width = int((original_height / render_height) * image_render_np.shape[1])
                    if new_width <= 0:
                        new_width = original_height
                    
                    # Resize the rendered images
                    image_render_resized = cv2.resize(image_render_np, (new_width, original_height))
                    image_render_gt_resized = cv2.resize(image_render_gt_np, (new_width, original_height))
                    
                    # Convert original image to RGB
                    image_original_rgb = cv2.cvtColor(image_original, cv2.COLOR_BGR2RGB) / 255.0
                    
                    # Create a combined image (side by side: original, reconstructed, ground truth)
                    combined_width = image_original_rgb.shape[1] + new_width * 2
                    combined_image = np.zeros((original_height, combined_width, 3))
                    combined_image[:, :image_original_rgb.shape[1], :] = image_original_rgb
                    combined_image[:, image_original_rgb.shape[1]:image_original_rgb.shape[1]+new_width, :] = image_render_resized
                    combined_image[:, image_original_rgb.shape[1]+new_width:, :] = image_render_gt_resized
                    
                    # Convert to tensor and append to results
                    combined_image = torch.from_numpy(combined_image).float()
                    combined_image = combined_image.permute(2, 0, 1)  # [H,W,C] -> [C,H,W]
                    pred_images.append(combined_image.unsqueeze(0))  # Add batch dimension

                # # Stack frames for this batch
                # batch_tensor = torch.cat(pred_images[-actual_visualize_batch_size:], dim=0)

            pred_images_tensor = torch.cat(pred_images, dim=0).cpu()
            # Use sequence and video name in the output filename
            output_dir_video = os.path.join(data_root_candor, 'reconstructed_vs_gt', seq_name)
            os.makedirs(output_dir_video, exist_ok=True)
            dump_path = os.path.join(output_dir_video, f"{p1_name}_{start_frame}_{end_frame}.mp4")
            
            print(f"Saving video to: {dump_path}")
            audio_clip = audio[int(start_frame/25.0*16000):int(end_frame/25.0*16000)]
            write_video(pred_images_tensor*255.0, dump_path, 25, audio_clip, 16000, "aac")
            print("Video saved successfully!")


        target_path_face_p1 = os.path.join(output_dir_candor, p1_name + '.npy')
        Path(target_path_face_p1).parent.mkdir(parents=True, exist_ok=True)
        np.save(target_path_face_p1, tar_index_value_face_top_p1.to('cpu').numpy())

        target_path_face_p2 = os.path.join(output_dir_candor, p2_name + '.npy')
        Path(target_path_face_p2).parent.mkdir(parents=True, exist_ok=True)
        np.save(target_path_face_p2, tar_index_value_face_top_p2.to('cpu').numpy())



    print(
        f'Motion tokenization done, the motion tokens are saved to {output_dir}'
    )


if __name__ == "__main__":
    main()
