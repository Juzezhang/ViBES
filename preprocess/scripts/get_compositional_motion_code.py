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


def get_video_frame_count(video_path):
    """
    获取视频的总帧数
    
    Args:
        video_path: 视频文件路径
        
    Returns:
        int: 视频总帧数，如果失败返回0
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return 0
        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return frame_count
    except Exception as e:
        print(f"Error getting frame count from {video_path}: {e}")
        return 0


def extract_frame_from_video(video_path, frame_idx, default_size=(256, 256)):
    """
    从视频文件中抽取指定帧
    
    Args:
        video_path: 视频文件路径
        frame_idx: 帧索引（从0开始）
        default_size: 如果抽帧失败时返回的默认图片尺寸
        
    Returns:
        numpy array: 抽取的帧 (H, W, C) 格式，BGR颜色空间
    """
    try:
        # 打开视频文件
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            # 视频文件无法打开，返回默认黑色图片
            return np.zeros((*default_size, 3), dtype=np.uint8)
        
        # 跳转到指定帧
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        
        # 读取帧
        ret, frame = cap.read()
        cap.release()
        
        if ret and frame is not None:
            # 如果需要，调整图片尺寸到默认大小
            if frame.shape[:2] != default_size:
                frame = cv2.resize(frame, default_size)
            return frame
        else:
            # 读取失败，返回默认黑色图片
            return np.zeros((*default_size, 3), dtype=np.uint8)
            
    except Exception as e:
        print(f"Error extracting frame {frame_idx} from {video_path}: {e}")
        return np.zeros((*default_size, 3), dtype=np.uint8)


def extract_frames_batch(video_path, frame_indices, default_size=(256, 256)):
    """
    从视频文件中批量抽取指定帧，性能更好的版本
    
    Args:
        video_path: 视频文件路径
        frame_indices: 帧索引列表
        default_size: 如果抽帧失败时返回的默认图片尺寸
        
    Returns:
        dict: {frame_idx: frame_array} 的字典
    """
    frames_dict = {}
    
    try:
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            # 视频文件无法打开，为所有帧返回默认黑色图片
            default_frame = np.zeros((*default_size, 3), dtype=np.uint8)
            return {idx: default_frame.copy() for idx in frame_indices}
        
        # 排序帧索引以便顺序读取
        sorted_indices = sorted(frame_indices)
        current_frame_idx = 0
        
        for target_idx in sorted_indices:
            # 跳转到目标帧
            if target_idx >= current_frame_idx:
                # 如果目标帧在当前位置之后，逐帧读取
                while current_frame_idx < target_idx:
                    ret, _ = cap.read()
                    if not ret:
                        break
                    current_frame_idx += 1
            else:
                # 如果目标帧在当前位置之前，重新定位
                cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
                current_frame_idx = target_idx
            
            # 读取目标帧
            ret, frame = cap.read()
            
            if ret and frame is not None:
                if frame.shape[:2] != default_size:
                    frame = cv2.resize(frame, default_size)
                frames_dict[target_idx] = frame
                current_frame_idx += 1
            else:
                frames_dict[target_idx] = np.zeros((*default_size, 3), dtype=np.uint8)
        
        cap.release()
        
        # 为未成功读取的帧添加默认图片
        default_frame = np.zeros((*default_size, 3), dtype=np.uint8)
        for idx in frame_indices:
            if idx not in frames_dict:
                frames_dict[idx] = default_frame.copy()
                
    except Exception as e:
        print(f"Error extracting frames from {video_path}: {e}")
        default_frame = np.zeros((*default_size, 3), dtype=np.uint8)
        frames_dict = {idx: default_frame.copy() for idx in frame_indices}
    
    return frames_dict


def find_video_path(video_base_path, seq_name, person_name):
    """
    查找指定人员的视频文件路径
    
    Args:
        video_base_path: 视频基础路径
        seq_name: 序列名称
        person_name: 人员名称
        
    Returns:
        str or None: 找到的视频路径，如果未找到返回None
    """
    possible_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
    
    # 尝试不同的路径结构和文件扩展名
    for ext in possible_extensions:
        path_patterns = [
            os.path.join(video_base_path, seq_name, f"{person_name}{ext}"),
            os.path.join(video_base_path, f"{seq_name}_{person_name}{ext}"),
            os.path.join(video_base_path, seq_name, f"{seq_name}_{person_name}{ext}")
        ]
        
        for potential_path in path_patterns:
            if os.path.exists(potential_path):
                return potential_path
    
    return None


def main():
    # parse options
    cfg = parse_args(phase="test")  # parse config file
    cfg.TRAIN.STAGE = "token"
    cfg.TRAIN.BATCH_SIZE = 1

    show_comparison = True

    device = torch.device(f"cuda:0") if torch.cuda.is_available() else torch.device("cpu")


    if show_comparison:
        model_path = "./model_files/FLAME2020/"
        batch_size_visualize = 100
        # Initialize FLAME model with the adjusted batch size and move to GPU
        flame_model = FLAME(model_path, num_expression_coeffs=100, ext='pkl', batch_size=batch_size_visualize).to(device)
        start_frame = 10000
        end_frame = 15000
        # start_frame = 0
        # end_frame = 1e10
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
        else:
            output_dir = output_dir_beat2

        face_p1, face_p2 = [
            batch[key].to(device) for key in ["face_p1","face_p2"]
        ]

        tar_index_value_face_top_p1 = model.vae_face.map2index(face_p1)  # bs*n/4
        tar_index_value_face_top_p2 = model.vae_face.map2index(face_p2)  # bs*n/4
        rec_face_test_p1 = model.vae_face.decode(tar_index_value_face_top_p1.int())
        rec_face_test_p2 = model.vae_face.decode(tar_index_value_face_top_p2.int())

        
        # Extract components similar to the original FLAME format
        rec_head_pose_p1 = rotation_6d_to_axis_angle(rec_face_test_p1[0, :, :6]).detach().cpu().numpy()
        rec_jaw_pose_p1 = rotation_6d_to_axis_angle(rec_face_test_p1[0, :, 6:12]).detach().cpu().numpy()
        rec_exp_p1 = rec_face_test_p1[0, :, 12:62].detach().cpu().numpy()  # First 50 dims of expression
        
        rec_head_pose_p2 = rotation_6d_to_axis_angle(rec_face_test_p2[0, :, :6]).detach().cpu().numpy()
        rec_jaw_pose_p2 = rotation_6d_to_axis_angle(rec_face_test_p2[0, :, 6:12]).detach().cpu().numpy()
        rec_exp_p2 = rec_face_test_p2[0, :, 12:62].detach().cpu().numpy()  # First 50 dims of expression
        
        # Save in the same format as the example
        np.savez(os.path.join(output_dir_reconstructed, f"{p1_name}.npz"), 
                    pose=np.concatenate([rec_head_pose_p1, rec_jaw_pose_p1], axis=1),
                    exp=rec_exp_p1,
                    shape=np.zeros((rec_exp_p1.shape[0], 100)))  # Placeholder for shape
        
        np.savez(os.path.join(output_dir_reconstructed, f"{p2_name}.npz"), 
                    pose=np.concatenate([rec_head_pose_p2, rec_jaw_pose_p2], axis=1),
                    exp=rec_exp_p2,
                    shape=np.zeros((rec_exp_p2.shape[0], 100)))  # Placeholder for shape

        if show_comparison:

            # 修改: 从视频路径读取而不是图片文件夹
            video_base_path = "/simurgh/u/juze/datasets/CANDOR/videos"
            
            # 查找p1和p2的视频路径
            video_path_p1 = find_video_path(video_base_path, seq_name, p1_name)
            video_path_p2 = find_video_path(video_base_path, seq_name, p2_name)
            
            if video_path_p1 is None:
                print(f"Warning: No video found for {seq_name}/{p1_name} in {video_base_path}")
            else:
                print(f"Found video for p1: {video_path_p1}")
                
            if video_path_p2 is None:
                print(f"Warning: No video found for {seq_name}/{p2_name} in {video_base_path}")
            else:
                print(f"Found video for p2: {video_path_p2}")
            
            # 获取视频帧数并调整start_frame和end_frame
            max_frames_p1 = get_video_frame_count(video_path_p1) if video_path_p1 else 0
            max_frames_p2 = get_video_frame_count(video_path_p2) if video_path_p2 else 0
            
            # 创建局部变量避免修改全局变量
            local_start_frame = start_frame
            local_end_frame = end_frame
            
            # 取两个视频中较小的帧数，或者如果没有视频则使用默认值
            if max_frames_p1 > 0 or max_frames_p2 > 0:
                max_frames = min(max_frames_p1, max_frames_p2) if max_frames_p1 > 0 and max_frames_p2 > 0 else max(max_frames_p1, max_frames_p2)
                original_start = local_start_frame
                original_end = local_end_frame
                
                # 调整帧范围
                local_start_frame = min(local_start_frame, max_frames - 1)
                local_end_frame = min(local_end_frame, max_frames)
                
                if local_start_frame < 0:
                    local_start_frame = 0
                if local_end_frame <= local_start_frame:
                    local_end_frame = min(local_start_frame + 1000, max_frames)  # 默认1000帧
                
                if original_start != local_start_frame or original_end != local_end_frame:
                    print(f"Adjusted frame range from [{original_start}, {original_end}] to [{local_start_frame}, {local_end_frame}] (max frames: {max_frames})")
            
            mesh_renderer = RenderMesh(image_size=256, faces=flame_model.faces, scale=1.0)
            
            # 分别处理p1和p2的音频
            audio_path_p1 = f"/simurgh/group/yuheng/CANDOR_processed/{seq_name}/{p1_name}.mp3"
            audio_path_p2 = f"/simurgh/group/yuheng/CANDOR_processed/{seq_name}/{p2_name}.mp3"
            
            # 加载p1音频
            if os.path.exists(audio_path_p1):
                audio_p1, sr = torchaudio.load(audio_path_p1)
                audio_p1 = torchaudio.transforms.Resample(sr, 16000)(audio_p1).mean(dim=0)
                print(f"Loaded audio for p1: {audio_path_p1}")
            else:
                print(f"Warning: No audio found for p1: {audio_path_p1}")
                # 创建静音音频
                audio_length = int((local_end_frame - local_start_frame) / 25.0 * 16000)
                audio_p1 = torch.zeros(audio_length)
            
            # 加载p2音频
            if os.path.exists(audio_path_p2):
                audio_p2, sr = torchaudio.load(audio_path_p2)
                audio_p2 = torchaudio.transforms.Resample(sr, 16000)(audio_p2).mean(dim=0)
                print(f"Loaded audio for p2: {audio_path_p2}")
            else:
                print(f"Warning: No audio found for p2: {audio_path_p2}")
                # 创建静音音频
                audio_length = int((local_end_frame - local_start_frame) / 25.0 * 16000)
                audio_p2 = torch.zeros(audio_length)
            
            # 处理p1的面部数据
            rec_face_head_p1 = rotation_6d_to_axis_angle(rec_face_test_p1[0, :,:6])
            rec_face_yaw_p1 = rotation_6d_to_axis_angle(rec_face_test_p1[0, :,6:12])
            rec_exp_p1 = rec_face_test_p1[0, :, 12:]
            gt_face_head_p1 = rotation_6d_to_axis_angle(face_p1[0, :,:6])
            gt_face_yaw_p1 = rotation_6d_to_axis_angle(face_p1[0, :,6:12])
            gt_face_exp_p1 = face_p1[0, :, 12:]
            
            # 处理p2的面部数据
            rec_face_head_p2 = rotation_6d_to_axis_angle(rec_face_test_p2[0, :,:6])
            rec_face_yaw_p2 = rotation_6d_to_axis_angle(rec_face_test_p2[0, :,6:12])
            rec_exp_p2 = rec_face_test_p2[0, :, 12:]
            gt_face_head_p2 = rotation_6d_to_axis_angle(face_p2[0, :,:6])
            gt_face_yaw_p2 = rotation_6d_to_axis_angle(face_p2[0, :,6:12])
            gt_face_exp_p2 = face_p2[0, :, 12:]

            # 批量提取需要的视频帧，提升性能
            frame_indices = list(range(local_start_frame, local_end_frame))
            
            # 提取p1的视频帧
            if video_path_p1:
                video_frames_p1 = extract_frames_batch(video_path_p1, frame_indices, default_size=(256, 256))
            else:
                default_frame = np.zeros((256, 256, 3), dtype=np.uint8)
                video_frames_p1 = {idx: default_frame.copy() for idx in frame_indices}
                
            # 提取p2的视频帧
            if video_path_p2:
                video_frames_p2 = extract_frames_batch(video_path_p2, frame_indices, default_size=(256, 256))
            else:
                default_frame = np.zeros((256, 256, 3), dtype=np.uint8)
                video_frames_p2 = {idx: default_frame.copy() for idx in frame_indices}

            pred_images_p1 = []
            pred_images_p2 = []

            for i in range(local_start_frame, local_end_frame, batch_size_visualize):
                actual_visualize_batch_size = min(batch_size_visualize, local_end_frame - i)

                # Run FLAME model for both p1 and p2 (reconstructed and ground truth)
                with torch.no_grad():
                    # P1 reconstructed
                    flame_out_p1_rec = flame_model(
                        global_orient=rec_face_head_p1[i:i+actual_visualize_batch_size, :],
                        expression=rec_exp_p1[i:i+actual_visualize_batch_size, :],
                        jaw_pose=rec_face_yaw_p1[i:i+actual_visualize_batch_size, :],
                        shape=torch.zeros(actual_visualize_batch_size, 100).to(device),
                    )
                    # P1 ground truth
                    flame_out_p1_gt = flame_model(
                        global_orient=gt_face_head_p1[i:i+actual_visualize_batch_size, :],
                        expression=gt_face_exp_p1[i:i+actual_visualize_batch_size, :],
                        jaw_pose=gt_face_yaw_p1[i:i+actual_visualize_batch_size, :],
                        shape=torch.zeros(actual_visualize_batch_size, 100).to(device),
                    )
                    # P2 reconstructed
                    flame_out_p2_rec = flame_model(
                        global_orient=rec_face_head_p2[i:i+actual_visualize_batch_size, :],
                        expression=rec_exp_p2[i:i+actual_visualize_batch_size, :],
                        jaw_pose=rec_face_yaw_p2[i:i+actual_visualize_batch_size, :],
                        shape=torch.zeros(actual_visualize_batch_size, 100).to(device),
                    )
                    # P2 ground truth
                    flame_out_p2_gt = flame_model(
                        global_orient=gt_face_head_p2[i:i+actual_visualize_batch_size, :],
                        expression=gt_face_exp_p2[i:i+actual_visualize_batch_size, :],
                        jaw_pose=gt_face_yaw_p2[i:i+actual_visualize_batch_size, :],
                        shape=torch.zeros(actual_visualize_batch_size, 100).to(device),
                    )
                
                # Get vertices for all models
                verts_p1_rec = flame_out_p1_rec['vertices'].detach()
                verts_p1_gt = flame_out_p1_gt['vertices'].detach()
                verts_p2_rec = flame_out_p2_rec['vertices'].detach()
                verts_p2_gt = flame_out_p2_gt['vertices'].detach()

                # Render all meshes
                render_p1_rec = mesh_renderer(verts_p1_rec)
                render_p1_gt = mesh_renderer(verts_p1_gt)
                render_p2_rec = mesh_renderer(verts_p2_rec)
                render_p2_gt = mesh_renderer(verts_p2_gt)
                
                # Normalize rendered images
                images_p1_rec = render_p1_rec[0] / 255.0
                images_p1_gt = render_p1_gt[0] / 255.0
                images_p2_rec = render_p2_rec[0] / 255.0
                images_p2_gt = render_p2_gt[0] / 255.0
                
                # Process each frame in the batch
                for b in range(actual_visualize_batch_size):
                    frame_idx = i + b
                    
                    # 获取原始图片
                    image_original_p1 = video_frames_p1.get(frame_idx, np.zeros((256, 256, 3), dtype=np.uint8))
                    image_original_p2 = video_frames_p2.get(frame_idx, np.zeros((256, 256, 3), dtype=np.uint8))
                    
                    # 获取当前帧的渲染图片
                    image_p1_rec = images_p1_rec[b]
                    image_p1_gt = images_p1_gt[b]
                    image_p2_rec = images_p2_rec[b]
                    image_p2_gt = images_p2_gt[b]
                    
                    # 转换tensor到numpy并调整维度
                    def tensor_to_numpy(tensor_img):
                        img_np = tensor_img.cpu().numpy()
                        return np.transpose(img_np, (1, 2, 0))  # Convert from [C,H,W] to [H,W,C]
                    
                    image_p1_rec_np = tensor_to_numpy(image_p1_rec)
                    image_p1_gt_np = tensor_to_numpy(image_p1_gt)
                    image_p2_rec_np = tensor_to_numpy(image_p2_rec)
                    image_p2_gt_np = tensor_to_numpy(image_p2_gt)
                    
                    # 转换原始图片到RGB
                    image_original_p1_rgb = cv2.cvtColor(image_original_p1, cv2.COLOR_BGR2RGB) / 255.0
                    image_original_p2_rgb = cv2.cvtColor(image_original_p2, cv2.COLOR_BGR2RGB) / 255.0
                    
                    # 获取尺寸信息
                    height = image_original_p1_rgb.shape[0]
                    width = image_original_p1_rgb.shape[1]
                    
                    # 调整渲染图片大小以匹配原始图片
                    image_p1_rec_resized = cv2.resize(image_p1_rec_np, (width, height))
                    image_p1_gt_resized = cv2.resize(image_p1_gt_np, (width, height))
                    image_p2_rec_resized = cv2.resize(image_p2_rec_np, (width, height))
                    image_p2_gt_resized = cv2.resize(image_p2_gt_np, (width, height))
                    
                    # 创建P1的图片 (1行3列: Original | Reconstructed | Ground Truth)
                    combined_image_p1 = np.concatenate([image_original_p1_rgb, image_p1_rec_resized, image_p1_gt_resized], axis=1)
                    
                    # 创建P2的图片 (1行3列: Original | Reconstructed | Ground Truth)
                    combined_image_p2 = np.concatenate([image_original_p2_rgb, image_p2_rec_resized, image_p2_gt_resized], axis=1)
                    
                    # 添加标签
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.7
                    font_color = (255, 255, 255)
                    thickness = 2
                    
                    # 为P1图片添加标签
                    combined_image_p1_uint8 = (combined_image_p1 * 255).astype(np.uint8)
                    cv2.putText(combined_image_p1_uint8, f'P1({p1_name})', (10, 30), font, font_scale, font_color, thickness)
                    cv2.putText(combined_image_p1_uint8, 'Original', (10, height - 10), font, font_scale, font_color, thickness)
                    cv2.putText(combined_image_p1_uint8, 'Reconstructed', (width + 10, height - 10), font, font_scale, font_color, thickness)
                    cv2.putText(combined_image_p1_uint8, 'Ground Truth', (2*width + 10, height - 10), font, font_scale, font_color, thickness)
                    combined_image_p1 = combined_image_p1_uint8.astype(np.float32) / 255.0
                    
                    # 为P2图片添加标签
                    combined_image_p2_uint8 = (combined_image_p2 * 255).astype(np.uint8)
                    cv2.putText(combined_image_p2_uint8, f'P2({p2_name})', (10, 30), font, font_scale, font_color, thickness)
                    cv2.putText(combined_image_p2_uint8, 'Original', (10, height - 10), font, font_scale, font_color, thickness)
                    cv2.putText(combined_image_p2_uint8, 'Reconstructed', (width + 10, height - 10), font, font_scale, font_color, thickness)
                    cv2.putText(combined_image_p2_uint8, 'Ground Truth', (2*width + 10, height - 10), font, font_scale, font_color, thickness)
                    combined_image_p2 = combined_image_p2_uint8.astype(np.float32) / 255.0
                    
                    # 转换为tensor并添加到对应的结果列表
                    combined_image_p1_tensor = torch.from_numpy(combined_image_p1).float()
                    combined_image_p1_tensor = combined_image_p1_tensor.permute(2, 0, 1)  # [H,W,C] -> [C,H,W]
                    pred_images_p1.append(combined_image_p1_tensor.unsqueeze(0))  # Add batch dimension
                    
                    combined_image_p2_tensor = torch.from_numpy(combined_image_p2).float()
                    combined_image_p2_tensor = combined_image_p2_tensor.permute(2, 0, 1)  # [H,W,C] -> [C,H,W]
                    pred_images_p2.append(combined_image_p2_tensor.unsqueeze(0))  # Add batch dimension

            # 分别保存P1和P2的视频
            output_dir_video = os.path.join(data_root_candor, 'reconstructed_vs_gt', seq_name)
            os.makedirs(output_dir_video, exist_ok=True)
            
            # 保存P1视频
            pred_images_p1_tensor = torch.cat(pred_images_p1, dim=0).cpu()
            dump_path_p1 = os.path.join(output_dir_video, f"{p1_name}_{local_start_frame}_{local_end_frame}.mp4")
            print(f"Saving P1 video to: {dump_path_p1}")
            audio_clip_p1 = audio_p1[int(local_start_frame/25.0*16000):int(local_end_frame/25.0*16000)]
            write_video(pred_images_p1_tensor*255.0, dump_path_p1, 25, audio_clip_p1, 16000, "aac")
            print("P1 video saved successfully!")
            
            # 保存P2视频
            pred_images_p2_tensor = torch.cat(pred_images_p2, dim=0).cpu()
            dump_path_p2 = os.path.join(output_dir_video, f"{p2_name}_{local_start_frame}_{local_end_frame}.mp4")
            print(f"Saving P2 video to: {dump_path_p2}")
            audio_clip_p2 = audio_p2[int(local_start_frame/25.0*16000):int(local_end_frame/25.0*16000)]
            write_video(pred_images_p2_tensor*255.0, dump_path_p2, 25, audio_clip_p2, 16000, "aac")
            print("P2 video saved successfully!")

        # target_path_face_p1 = os.path.join(output_dir_candor, p1_name + '.npy')
        # Path(target_path_face_p1).parent.mkdir(parents=True, exist_ok=True)
        # np.save(target_path_face_p1, tar_index_value_face_top_p1.to('cpu').numpy())

        # target_path_face_p2 = os.path.join(output_dir_candor, p2_name + '.npy')
        # Path(target_path_face_p2).parent.mkdir(parents=True, exist_ok=True)
        # np.save(target_path_face_p2, tar_index_value_face_top_p2.to('cpu').numpy())

    print(
        f'Motion tokenization done, the motion tokens are saved to {output_dir}'
    )


if __name__ == "__main__":
    main()
