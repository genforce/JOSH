from dataclasses import dataclass, field
from typing import Optional, TypedDict

import numpy as np
import torch
import trimesh


@dataclass
class JOSHConfig:
    input_folder: str = "data/demo1"
    output_folder: str = "josh"
    optimize_smpl: bool = True
    optimize_focal: bool = True
    init_focal: Optional[float] = None
    optimize_depth: bool = False
    scale_loss_weight: float = 1.0
    prior_loss_weight: float = 100.0
    static_loss_weight: float = 0.1
    smooth_loss_weight: float = 0.1
    scene_graph: str = "window-10"
    opt_interval: int = 5
    start_frame: int = 0
    max_frames: int = 21
    num_frames: int = 0
    img_idx: list = field(default_factory=list)
    conf_thres: float = 0.1
    use_depth_model: bool = True
    update_correspondences: bool = False
    depth_filter_ratio: float = 1.01
    visualize_results: bool = True


class OptimizedFrameResult(TypedDict):
    frame_idx: int
    conf_hw: np.ndarray
    rgb_hw3: np.ndarray
    depth_hw: np.ndarray
    mask_hw: np.ndarray
    pred_smpl: trimesh.Trimesh | None
    pred_cam: np.ndarray


@dataclass
class OptimizedResult:
    point_cloud: trimesh.PointCloud
    mesh: trimesh.Trimesh
    intrinsics: np.ndarray
    img_size: tuple[int, int]
    frame_result: list[OptimizedFrameResult]
    eval_metrics: dict


class ImageDict(TypedDict):
    img: torch.Tensor
    true_shape: tuple[int, int] | torch.Tensor
    idx: int | list[int]
    instance: str | list[str]
