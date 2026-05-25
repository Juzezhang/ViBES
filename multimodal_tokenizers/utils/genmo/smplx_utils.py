"""SMPL-X model factory vendored from GVHMR for self-contained rendering.

Only supports the model types used in ViBES: "supermotion" and "smpl".
"""

from pathlib import Path

import numpy as np

from .body_model import BodyModelSMPLX, BodyModelSMPLH

_ROOT_DIR = Path(__file__).resolve().parents[3]  # ViBES project root
_BODY_MODEL_DIR = _ROOT_DIR / "model_files" / "smplx_models"
_GVHMR_ASSET_DIR = _ROOT_DIR / "model_files" / "gvhmr"


def make_smplx(type="supermotion", **kwargs):
    """Create an SMPL-X or SMPL body model.

    Args:
        type: Model type. Supported: "supermotion", "smpl".
        **kwargs: Extra keyword arguments forwarded to the body model constructor.

    Returns:
        Body model instance.
    """
    if type == "supermotion":
        bm_kwargs = {
            "model_type": "smplx",
            "gender": "neutral",
            "num_pca_comps": 12,
            "flat_hand_mean": False,
        }
        bm_kwargs.update(kwargs)
        return BodyModelSMPLX(model_path=str(_BODY_MODEL_DIR), **bm_kwargs)

    elif type == "smpl":
        faces_path = _GVHMR_ASSET_DIR / "smpl_faces.npy"
        if faces_path.exists():
            faces = np.load(str(faces_path))

            class _SmplFaces:
                pass

            obj = _SmplFaces()
            obj.faces = faces
            return obj

        bm_kwargs = {
            "model_type": "smpl",
            "gender": "neutral",
            "num_betas": 10,
            "create_body_pose": False,
            "create_betas": False,
            "create_global_orient": False,
            "create_transl": False,
        }
        bm_kwargs.update(kwargs)
        return BodyModelSMPLH(model_path=str(_BODY_MODEL_DIR), **bm_kwargs)

    else:
        raise NotImplementedError(
            f"make_smplx type '{type}' not supported. Use 'supermotion' or 'smpl'."
        )
