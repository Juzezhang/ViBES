from typing import List
import torch
from torch import Tensor
from torchmetrics import Metric
import pickle
import numpy as np

class FaceMetrics(Metric):
    def __init__(self,
                 cfg,
                 dist_sync_on_step=True,
                 **kwargs):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.cfg = cfg
        self.name = "Face metrics"
        self.region_path = cfg.METRIC.REGION_PATH
        with open(self.region_path, 'rb') as f:
            masks = pickle.load(f, encoding='latin1')
        self.template_path = cfg.METRIC.TEMPLATE_PATH
        self.template = torch.from_numpy(np.load(self.template_path)['mean_vertices']).float()

        # Common key names might be 'lips'/'mouth', 'upper_face' or need to combine 'eyes','brows','forehead' etc.
        self.lip_idx = masks['lips']
        self.upper_face_idx = np.concatenate([masks['eye_region'], masks['forehead']])
        
        # Add states for accumulating metrics
        self.add_state("LVE", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("FFD", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("MPVPE_FACE", default=torch.tensor(0.0), dist_reduce_fx="sum") 
        self.add_state("MOD", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    # def compute_lve(self, vertices_pred, vertices_gt, lip_idx):
    #     """
    #     Mean per-joint position error (i.e. mean Euclidean distance)
    #     often referred to as "Protocol #1" in many papers.
    #     """
    #     L2_dis_mouth_max = torch.stack([torch.square(vertices_gt[:,v, :]-vertices_pred[:,v,:]) for v in lip_idx])
    #     L2_dis_mouth_max = torch.transpose(L2_dis_mouth_max, 0, 1)
    #     L2_dis_mouth_max = torch.sum(L2_dis_mouth_max, dim=2)
    #     L2_dis_mouth_max = torch.max(L2_dis_mouth_max, dim=1).values
    #     L2_dis_mouth_max = L2_dis_mouth_max.mean()
    #     return L2_dis_mouth_max

    # def compute_lve(self, vertices_pred, vertices_gt, lip_idx):
    #     """
    #     Lip Vertex Error (LVE): mean Euclidean distance between predicted and GT lip vertices.
    #     """
    #     # Extract lip vertices
    #     pred_lip = vertices_pred[:, lip_idx, :]  # [T, N_lip, 3]
    #     gt_lip = vertices_gt[:, lip_idx, :]      # [T, N_lip, 3]

    #     # L2 distance per vertex
    #     l2_dis = torch.norm(pred_lip - gt_lip, dim=2)  # [T, N_lip]

    #     # Mean
    #     lve = l2_dis.mean()  # scalar
    #     return lve

    def compute_lve(self, vertices_pred, vertices_gt, lip_idx):
        """
        LVE (per your definition):
        per-frame max L2 over lip vertices, then average over frames.
        vertices_*: [T, V, 3]
        returns: scalar with same unit as vertices (e.g., meters)
        """
        idx = torch.as_tensor(lip_idx, device=vertices_pred.device, dtype=torch.long)
        diffs = vertices_pred.index_select(1, idx) - vertices_gt.index_select(1, idx)  # [T, |lip|, 3]
        per_vert_l2 = torch.linalg.norm(diffs, dim=2)                                   # [T, |lip|]
        # per_vert_l2 = (diffs ** 2).sum(dim=2)           # [T, |lip|], m^2

        per_frame_max = per_vert_l2.max(dim=1).values                                    # [T]
        # return per_frame_max.mean()
        return per_frame_max.sum()


    # def compute_fdd(self, vertices_pred, vertices_gt, upper_face_idx):
    #     """
    #     FDD
    #     """
    #     # Check if the template is on the same device as the vertices
    #     if self.template.device != vertices_pred.device:
    #         self.template = self.template.to(vertices_pred.device)

    #     # Calculate the motion of the predicted and ground truth vertices
    #     motion_pred = vertices_pred - self.template.reshape(1,-1,3)
    #     motion_gt = vertices_gt - self.template.reshape(1,-1,3)
    #     L2_dis_upper = torch.stack([torch.square(motion_gt[:,v, :]) for v in upper_face_idx])
    #     L2_dis_upper = torch.transpose(L2_dis_upper, 0, 1)
    #     L2_dis_upper = torch.sum(L2_dis_upper, dim=2)
    #     L2_dis_upper = torch.std(L2_dis_upper, dim=0)
    #     gt_motion_std = torch.mean(L2_dis_upper)

    #     # Calculate the motion of the predicted vertices
    #     L2_dis_upper = torch.stack([torch.square(motion_pred[:,v, :]) for v in upper_face_idx])
    #     L2_dis_upper = torch.transpose(L2_dis_upper, 0, 1)
    #     L2_dis_upper = torch.sum(L2_dis_upper, dim=2)
    #     L2_dis_upper = torch.std(L2_dis_upper, dim=0)
    #     pred_motion_std = torch.mean(L2_dis_upper)

    #     # Calculate the FFD
    #     FFD = gt_motion_std - pred_motion_std
        
    #     return FFD
    
    def compute_fdd(self, vertices_pred, vertices_gt, upper_face_idx):
        """
        FDD (Upper-face Dynamics Deviation)
        Matches CodeTalker's evaluation definition and values (np.std with ddof=0)
        Args:
            vertices_pred: [T, V, 3]
            vertices_gt:   [T, V, 3]
            upper_face_idx: list[int] or LongTensor, length = U
        Return:
            fdd: scalar (mean GT dynamic magnitude - mean Pred dynamic magnitude)
        """
        # 1) Align template device/dtype and ensure shape is [1, V, 3]
        template = self.template
        if template.dim() == 2:
            template = template.unsqueeze(0)          # [1, V, 3]
        template = template.to(vertices_pred.device, dtype=vertices_pred.dtype)

        # 2) Displacement relative to the template (remove static shape)
        motion_pred = vertices_pred - template       # [T, V, 3]
        motion_gt   = vertices_gt   - template       # [T, V, 3]

        # 3) Select upper-face vertices and compute squared displacement norms (no sqrt; matches official script)
        #    Shape: [T, U, 3] -> sum over dim=2 -> [T, U]
        motion_pred_u_msq = (motion_pred[:, upper_face_idx, :] ** 2).sum(dim=2)  # [T, U]
        motion_gt_u_msq   = (motion_gt[:,   upper_face_idx, :] ** 2).sum(dim=2)  # [T, U]

        # 4) Std over time (ddof=0 -> unbiased=False), then mean over vertices
        pred_std_per_v = motion_pred_u_msq.std(dim=0, unbiased=False)  # [U]
        gt_std_per_v   = motion_gt_u_msq.std(dim=0,   unbiased=False)  # [U]

        pred_motion_std = pred_std_per_v.mean()  # scalar
        gt_motion_std   = gt_std_per_v.mean()    # scalar

        # 5) FDD (signed difference: GT - Pred)
        fdd = gt_motion_std - pred_motion_std
        return fdd


    # def compute_mod(self, vertices_pred, vertices_gt, lip_idx):
    #     """
    #     Mouth Opening Difference (MOD)
    #     - Inputs:
    #         vertices_pred: [T, V, 3]  predicted vertices (per frame)
    #         vertices_gt:   [T, V, 3]  ground-truth vertices (per frame)
    #         lip_idx:       list[int]  FLAME lip-region vertex indices (e.g., the provided 254 points)
    #     - Output:
    #         A scalar representing the average error within the lip region (across frames and vertices).
    #     - Notes:
    #         Consistent with your LVE implementation, this uses squared Euclidean distance (no square root),
    #         but replaces "max over lip vertices" with "mean over lip vertices".
    #         If you want millimeters and Euclidean distance, see the commented variant below.
    #     """
    #     # Move indices to the same device
    #     idx = torch.as_tensor(lip_idx, device=vertices_pred.device, dtype=torch.long)

    #     # Select lip-region points and compute per-frame, per-vertex errors
    #     # shape: [T, |lip|, 3]
    #     diffs = vertices_gt.index_select(1, idx) - vertices_pred.index_select(1, idx)

    #     # Same as LVE: use squared Euclidean distance, without square root
    #     # shape: [T, |lip|]
    #     dist_sq = (diffs ** 2).sum(dim=2)

    #     # MOD: take mean over lip vertices (not max as in LVE), then mean over time
    #     # scalar
    #     mod = dist_sq.mean()

    #     return mod


    # def compute_mod(self, vertices_pred, vertices_gt, lip_idx, to_mm=False):
    #     """
    #     MOD: avg over time & lip vertices of Euclidean distance.
    #     """
    #     idx = torch.as_tensor(lip_idx, device=vertices_pred.device, dtype=torch.long)
    #     idx = torch.unique(idx)
    #     diffs = vertices_gt[:, idx, :] - vertices_pred[:, idx, :]   # [T, |lip|, 3]
    #     dist  = torch.linalg.norm(diffs, dim=2)                     # [T, |lip|]
    #     mod   = dist.mean()                                         # 标量
    #     return mod * 1000.0 if to_mm else mod


    def compute_mod(self, vertices_pred, vertices_gt, lip_idx):
        """
        LVE (per your definition):
        per-frame max L2 over lip vertices, then average over frames.
        vertices_*: [T, V, 3]
        returns: scalar with same unit as vertices (e.g., meters)
        """
        idx = torch.as_tensor(lip_idx, device=vertices_pred.device, dtype=torch.long)
        diffs = vertices_pred.index_select(1, idx) - vertices_gt.index_select(1, idx)  # [T, |lip|, 3]
        per_vert_l2 = torch.linalg.norm(diffs, dim=2)                                   # [T, |lip|]
        per_frame_mean = per_vert_l2.mean(dim=1)                                    # [T]
        # return per_frame_max.mean()
        return per_frame_mean.sum()



    def compute_mpvpe(self, vertices_pred, vertices_gt):
        """
        Mean per-joint position error (i.e. mean Euclidean distance)
        often referred to as "Protocol #1" in many papers.
        """
        # Calculate the motion of the predicted and ground truth vertices
        mpvpe = torch.norm(vertices_pred - vertices_gt, p=2, dim=-1).mean(-1)
        return mpvpe
    
    @torch.no_grad()
    def update(self, rec_vertices: Tensor, tar_vertices: Tensor, lengths: Tensor = None):
        bs, n = rec_vertices.shape[0], rec_vertices.shape[1]
        # self.count += bs
        # Calculate metrics
        for bs_idx in range(bs):
            length = lengths[bs_idx]
            self.count += length
            self.LVE += self.compute_lve(1000*rec_vertices[bs_idx, :length], 1000*tar_vertices[bs_idx, :length], self.lip_idx)
            self.FFD += self.compute_fdd(1000*rec_vertices[bs_idx, :length], 1000*tar_vertices[bs_idx, :length], self.upper_face_idx)
            self.MPVPE_FACE += self.compute_mpvpe(1000*rec_vertices[bs_idx, :length], 1000*tar_vertices[bs_idx, :length]).mean()
            self.MOD += self.compute_mod(1000*rec_vertices[bs_idx, :length], 1000*tar_vertices[bs_idx, :length], self.lip_idx)

    def compute(self, sanity_flag=False):
        """Compute final metrics.
        Args:
            sanity_flag: Flag for sanity check, ignored in computation
        """
        metrics = {
            "LVE": self.LVE / self.count,
            "FFD": self.FFD / self.count,
            "MPVPE_FACE": self.MPVPE_FACE / self.count,
            "MOD": self.MOD / self.count,
        }
        return metrics 