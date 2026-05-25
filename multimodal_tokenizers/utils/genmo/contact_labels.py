import torch
import numpy as np

def get_joint_velocities(joints_3d, fps=30.0):
    """
    Compute velocities for 3D joints.
    Args:
        joints_3d: (L, J, 3) tensor or array
        fps: frame rate
    Returns:
        velocities: (L, J) magnitudes
    """
    if isinstance(joints_3d, np.ndarray):
        joints_3d = torch.from_numpy(joints_3d).float()

    # Compute frame-to-frame distance
    diff = joints_3d[1:] - joints_3d[:-1]
    velocity = torch.norm(diff, dim=-1) * fps # distance per second

    # Repeat last frame to keep length L
    velocity = torch.cat([velocity, velocity[-1:]], dim=0)
    return velocity

def get_contact_labels(joints_3d, vel_thr=0.15, fps=30.0):
    """
    Compute contact labels based on velocity threshold.
    Args:
        joints_3d: (L, J, 3)
        vel_thr: velocity threshold in m/s (default 0.15m/s as in HuMoR/GVHMR)
        fps: frame rate
    Returns:
        contact_labels: (L, J) boolean tensor
    """
    velocities = get_joint_velocities(joints_3d, fps)
    return velocities < vel_thr

def get_contact_joint_ids():
    """
    Returns the joint IDs used for contact monitoring in GENMO.
    7: L_Ankle, 10: L_Foot, 8: R_Ankle, 11: R_Foot, 20: L_Wrist, 21: R_Wrist
    """
    return [7, 10, 8, 11, 20, 21]
