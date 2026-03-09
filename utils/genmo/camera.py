"""Camera utilities vendored from GVHMR for self-contained rendering."""

import numpy as np
import torch


def create_camera_sensor(width=None, height=None, f_fullframe=None):
    """Create camera intrinsic matrix from image dimensions and focal length.

    Args:
        width: Image width in pixels. If None, randomly chosen (1200 or 1600).
        height: Image height in pixels. If None, randomly chosen.
        f_fullframe: Focal length in 35mm full-frame equivalent (mm).
            If None, randomly sampled from common lens options.

    Returns:
        (width, height, K) where K is a 3x3 intrinsic matrix.
    """
    if width is None or height is None:
        if np.random.rand() < 0.5:
            width, height = 1200, 1600
        else:
            width, height = 1600, 1200

    if f_fullframe is None:
        f_fullframe_options = [24, 26, 28, 30, 35, 40, 50, 60, 70]
        f_fullframe = np.random.choice(f_fullframe_options)

    # Map focal length using sensor diagonal ratio
    diag_fullframe = (24**2 + 36**2) ** 0.5
    diag_img = (width**2 + height**2) ** 0.5
    focal_length = diag_img / diag_fullframe * f_fullframe

    K = torch.eye(3)
    K[0, 0] = focal_length
    K[1, 1] = focal_length
    K[0, 2] = width / 2
    K[1, 2] = height / 2

    return width, height, K
