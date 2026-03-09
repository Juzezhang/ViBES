"""Thin wrappers around multimodal_tokenizers rotation conversions."""
# Derived from: /simurgh/u/askhan1/winter26/Video-as-Action-Prompt/genmo/utils/rotation_conversions.py
# Here we forward to local rotation conversion utilities.
from multimodal_tokenizers.utils.rotation_conversions import (  # noqa: F401
    axis_angle_to_matrix,
    matrix_to_axis_angle,
    rotation_6d_to_matrix,
    matrix_to_rotation_6d,
    euler_angles_to_matrix,
    quaternion_to_axis_angle,
    matrix_to_quaternion,
)
