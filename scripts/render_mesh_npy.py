"""
Render mesh .npy files to videos using the GENMO rendering pipeline.

Supports both SMPL (6890 vertices) and SMPLX (10475 vertices) meshes.

Usage:
    python scripts/render_mesh_npy.py \
        --input_dir paper_result/question2motion/motiongpt \
        --output_dir paper_result/question2motion/motiongpt/videos \
        --fps 30
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.genmo.camera import create_camera_sensor
from utils.genmo.video_io_utils import save_video
from utils.genmo.vis.renderer import (
    Renderer,
    get_global_cameras_static,
    get_ground_params_from_points,
)

GVHMR_ASSET_DIR = Path("model_files/gvhmr")
SMPLX_MODEL_DIR = Path("model_files/smplx_models")

# Vertex counts for auto-detection
SMPL_VERTS = 6890
SMPLX_VERTS = 10475


def load_faces(num_verts):
    """Load face topology matching the vertex count."""
    if num_verts == SMPL_VERTS:
        faces = np.load(GVHMR_ASSET_DIR / "smpl_faces.npy")
        print(f"Detected SMPL mesh ({num_verts} verts, {faces.shape[0]} faces)")
    elif num_verts == SMPLX_VERTS:
        import smplx
        model = smplx.create(str(SMPLX_MODEL_DIR), model_type="smplx", gender="neutral")
        faces = model.faces.astype(np.int64)
        print(f"Detected SMPLX mesh ({num_verts} verts, {faces.shape[0]} faces)")
    else:
        raise ValueError(
            f"Unknown vertex count {num_verts}. Expected {SMPL_VERTS} (SMPL) or {SMPLX_VERTS} (SMPLX)."
        )
    return faces


def get_fixed_camera(num_frames, beta=2.5, cam_height_degree=30,
                     target_center_height=1.0, vec_rot=45, device="cuda",
                     front_view=False):
    """Compute a fixed camera position independent of mesh data.

    Uses a fixed distance from the origin so all sequences share the same
    viewpoint.  The ``beta`` parameter directly controls distance (metres).

    Returns:
        cam_R: (L, 3, 3) rotation matrices
        cam_T: (L, 3) translation vectors
        lights: PointLights
    """
    from pytorch3d.renderer import PointLights
    from pytorch3d.renderer.cameras import look_at_rotation

    if front_view:
        vec_rot = 0
        cam_height_degree = 0

    vec_rad = vec_rot / 180 * np.pi
    vec = torch.tensor([np.sin(vec_rad), 0, np.cos(vec_rad)]).float()
    vec = vec / torch.norm(vec)

    target_center = torch.zeros(3)
    position = target_center + vec * beta
    position[1] = beta * np.tan(np.pi * cam_height_degree / 180) + target_center_height

    positions = position.unsqueeze(0).repeat(num_frames, 1)
    target_centers = target_center.unsqueeze(0).repeat(num_frames, 1)
    target_centers[:, 1] = target_center_height
    rotation = look_at_rotation(positions, target_centers).mT
    translation = -(rotation @ positions.unsqueeze(-1)).squeeze(-1)

    lights = PointLights(device=device, location=[position.tolist()])
    return rotation, translation, lights


def render_mesh_sequence(verts, faces, renderer, color, cam_beta=2.5,
                         fixed_camera=False, front_view=False):
    """
    Render a sequence of meshes to a list of frames.

    Args:
        verts: (T, V, 3) tensor of vertex positions
        faces: (F, 3) numpy array of face indices
        renderer: Renderer instance
        color: (3,) tensor of mesh color
        cam_beta: camera distance multiplier (lower = closer)
        fixed_camera: if True, use a fixed camera position at origin instead
                      of computing from per-sequence mesh trajectory
        front_view: if True, place camera facing the front of the SMPL body
    Returns:
        frames: (T, H, W, 3) numpy array of RGB frames
    """
    device = verts.device
    T = verts.shape[0]

    # Front-view camera overrides
    vec_rot = 0 if front_view else 45
    cam_height_degree = 0 if front_view else 30

    # Compute camera and ground params
    root_points = verts.mean(1)  # (T, 3)
    scale, cx, cz = get_ground_params_from_points(root_points, verts)
    renderer.set_ground(max(scale, 3) * 1.5, cx, cz)

    if fixed_camera:
        cam_R, cam_T, lights = get_fixed_camera(
            T, beta=cam_beta, device=str(device), front_view=front_view)
    else:
        cam_R, cam_T, lights = get_global_cameras_static(
            verts.cpu(), beta=cam_beta, vec_rot=vec_rot,
            cam_height_degree=cam_height_degree)

    frames = []
    for i in range(T):
        with torch.no_grad():
            cams = renderer.create_camera(cam_R[i], cam_T[i])
            img = renderer.render_with_ground(verts[[i]], color[None], cams, lights)
            frames.append(img)

    return np.array(frames)


def main():
    parser = argparse.ArgumentParser(description="Render mesh .npy files to videos")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing *_mesh.npy files")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for videos (default: input_dir/videos)")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--color", type=float, nargs=3, default=[0.69, 0.39, 0.96],
                        help="Mesh RGB color (0-1)")
    parser.add_argument("--crf", type=int, default=23, help="Video compression quality")
    parser.add_argument("--cam_beta", type=float, default=2.5,
                        help="Camera distance multiplier (lower = closer, default 2.5)")
    parser.add_argument("--fixed_camera", action="store_true",
                        help="Use a fixed camera position (same viewpoint for all sequences)")
    parser.add_argument("--front_view", action="store_true",
                        help="Camera faces the front of the SMPL body (eye-level, no elevation)")
    parser.add_argument("--pattern", type=str, default="*_mesh.npy",
                        help="Glob pattern for mesh files")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir / "videos"
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Find all mesh files
    mesh_files = sorted(input_dir.glob(args.pattern))
    if not mesh_files:
        print(f"No files matching '{args.pattern}' found in {input_dir}")
        return

    print(f"Found {len(mesh_files)} mesh files in {input_dir}")

    # Load first file to detect mesh type and initialize renderer
    sample = np.load(mesh_files[0])
    num_verts = sample.shape[1]
    faces = load_faces(num_verts)

    # Initialize renderer
    _, _, K = create_camera_sensor(args.width, args.height, 24)
    renderer = Renderer(args.width, args.height, device=str(device), faces=faces, K=K, bin_size=0)
    color = torch.tensor(args.color, dtype=torch.float32).to(device)

    print(f"Rendering {len(mesh_files)} sequences at {args.width}x{args.height}, {args.fps} fps")

    for mesh_file in mesh_files:
        name = mesh_file.stem.replace("_mesh", "")
        out_path = output_dir / f"{name}.mp4"

        verts_np = np.load(mesh_file)  # (T, V, 3)
        T, V, _ = verts_np.shape

        if T < 2:
            print(f"Skipping {mesh_file.name}: too few frames ({T})")
            continue

        verts = torch.from_numpy(verts_np).float().to(device)

        print(f"Rendering {mesh_file.name} ({T} frames, {V} verts) -> {out_path.name}")
        frames = render_mesh_sequence(verts, faces, renderer, color,
                                      cam_beta=args.cam_beta,
                                      fixed_camera=args.fixed_camera,
                                      front_view=args.front_view)

        save_video(frames, str(out_path), fps=args.fps, crf=args.crf)

    print(f"Done. Videos saved to {output_dir}")


if __name__ == "__main__":
    main()
