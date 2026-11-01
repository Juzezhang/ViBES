"""SMPL-X model factory vendored from GVHMR for self-contained rendering.

Only supports the model type used in ViBES: "supermotion".
"""

from pathlib import Path

from .body_model import BodyModelSMPLX

_ROOT_DIR = Path(__file__).resolve().parents[3]  # ViBES project root
_BODY_MODEL_DIR = _ROOT_DIR / "model_files" / "smplx_models"


def make_smplx(type="supermotion", **kwargs):
    """Create an SMPL-X body model.

    Args:
        type: Model type. Supported: "supermotion".
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

    else:
        raise NotImplementedError(
            f"make_smplx type '{type}' not supported. Use 'supermotion'."
        )
