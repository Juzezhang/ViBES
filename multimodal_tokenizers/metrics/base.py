from torch import Tensor, nn
from os.path import join as pjoin
from .rotation_metric import RotationMetrics
from .h3d_metric import H3DMetrics
from .co_speech import CoSpeechMetrics
from .body_metric import BodyMetrics
from .global_metric import GlobalMetrics
class BaseMetrics(nn.Module):
    # def __init__(self, cfg, datamodule, debug, **kwargs) -> None:
    def __init__(self, cfg, debug, **kwargs) -> None:
        super().__init__()

        for metric in cfg.METRIC.TYPE:
            setattr(self, metric, globals()[metric](
                    cfg=cfg,
                    dist_sync_on_step=cfg.METRIC.DIST_SYNC_ON_STEP,
                ))
