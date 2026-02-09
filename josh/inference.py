import os
import sys
import time
from argparse import ArgumentParser
from glob import glob
from pathlib import Path

import rerun as rr
import rerun.blueprint as rrb

sys.path.append("./third_party/pi3")
sys.path.append("./third_party/mast3r")
import glob

import joblib
import numpy as np
import rerun as rr
import torch
from mast3r.cloud_opt.utils.losses import gamma_loss
from mast3r.model import AsymmetricMASt3R
from pi3.models.pi3x import Pi3X
from PIL import Image
from tqdm import tqdm

from josh.config import ImageDict, JOSHConfig, OptimizedResult
from josh.joint_opt import joint_opt, scene_to_results_josh
from josh.utils.focal_utils import focal_to_intrinsics, recover_focal_shift
from josh.utils.image_utils import (generate_colors, load_images_and_masks, load_images_as_tensor, make_pairs)


def log_optimized_result(optimized_result: OptimizedResult, parent_log_path: Path, cfg: JOSHConfig) -> None:
    colors = generate_colors(n_colors=128, n_samples=5000)
    rr.log(f"{parent_log_path}", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)
    # log pointcloud
    rr.log(
        f"{parent_log_path}/pointcloud",
        rr.Points3D(
            positions=optimized_result.point_cloud.vertices,
            colors=optimized_result.point_cloud.colors,
        ),
        static=True,
    )

    mesh = optimized_result.mesh
    if mesh is not None:
        rr.log(
            f"{parent_log_path}/mesh",
            rr.Mesh3D(
                vertex_positions=mesh.vertices,
                vertex_colors=mesh.visual.vertex_colors,
                triangle_indices=mesh.faces,
            ),
            static=True,
        )

    intrinsics = optimized_result.intrinsics

    pbar = tqdm(
        optimized_result.frame_result,
        total=len(optimized_result.frame_result),
    )
    for i, frame_result in enumerate(pbar):
        pred_log_path = f"{parent_log_path}/pred_camera"
        rr.set_time("frame_idx", sequence=i)

        rr.log(
            f"{pred_log_path}",
            rr.Transform3D(
                translation=frame_result["pred_cam"][:3, 3],
                mat3x3=frame_result["pred_cam"][:3, :3],
                from_parent=False,
            ),
        )

        rr.log(
            f"{pred_log_path}/pinhole",
            rr.Pinhole(
                image_from_camera=intrinsics,
                height=optimized_result.img_size[1],
                width=optimized_result.img_size[0],
                camera_xyz=rr.ViewCoordinates.RDF,
            ),
        )
        rr.log(f"{parent_log_path}/pred_smpl_mesh", rr.Clear(recursive=True))

        if len(frame_result["pred_smpl"]) > 0:

            pred_smpls = frame_result["pred_smpl"]

            for idx, pred_smpl_mesh, pred_contact in pred_smpls:
                rr.log(
                    f"{parent_log_path}/pred_smpl_mesh/{idx}",
                    rr.Mesh3D(
                        vertex_positions=pred_smpl_mesh.vertices,
                        triangle_indices=pred_smpl_mesh.faces,
                        vertex_normals=pred_smpl_mesh.vertex_normals,
                        albedo_factor=colors[idx],
                    ),
                )

                valid_vertex_ids = torch.where(pred_contact == 1)[0]

                rr.log(f"{parent_log_path}/pred_smpl_contact/{idx}",
                       rr.Points3D(
                           positions=pred_smpl_mesh.vertices[valid_vertex_ids],
                           colors=np.array([1.0, 1.0, 0.0, 1.0]),
                       ))

        if "rgb_hw3" in frame_result:

            rgb_hw3 = frame_result["rgb_hw3"]
            rr.log(
                f"{pred_log_path}/pinhole/rgb",
                rr.Image(rgb_hw3),
            )


def inference(scene_model, device, cfg: JOSHConfig) -> OptimizedResult:
    """
    Perform inference using the Dust3r algorithm.

    Args:
        image_dir_or_list (Union[Path, List[Path]]): Path to the directory containing images or a list of image paths.
        model (AsymmetricMASt3R): The Dust3r model to use for inference.
        device (Literal["cpu", "cuda", "mps"]): The device to use for inference ("cpu", "cuda", or "mps").
        batch_size (int, optional): The batch size for inference. Defaults to 1.
        image_size (Literal[224, 512], optional): The size of the input images. Defaults to 512.
        niter (int, optional): The number of iterations for the global alignment optimization. Defaults to 100.
        schedule (Literal["linear", "cosine"], optional): The learning rate schedule for the global alignment optimization. Defaults to "linear".
        min_conf_thr (float, optional): The minimum confidence threshold for the optimized result. Defaults to 10.

    Returns:
        OptimizedResult: The optimized result containing the RGB, depth, and confidence images.

    Raises:
        ValueError: If `image_dir_or_list` is neither a list of paths nor a path.
    """

    img_files = sorted(glob.glob(os.path.join(cfg.input_folder, "rgb", "*.jpg")))
    mask_files = sorted(glob.glob(os.path.join(cfg.input_folder, "mask", "*.png")))
    import tempfile

    # cache_dir = tempfile.mkdtemp()
    cache_dir = "data/mast3r_results"
    assert len(img_files) == len(mask_files)
    imgs, masks, num_frames, img_idx, new_file_list = load_images_and_masks(
        img_files,
        mask_files,
        start_frame=cfg.start_frame,
        interval=cfg.opt_interval,
        max_num=cfg.max_frames,
    )

    H, W = imgs[0]['img'].shape[-2:]

    cfg.num_frames = num_frames
    cfg.img_idx = img_idx

    cfg.output_folder = f"josh_{cfg.start_frame}-{cfg.start_frame+cfg.num_frames-1}"
    os.makedirs(os.path.join(cfg.input_folder, cfg.output_folder), exist_ok=True)

    img_pairs: list[tuple[ImageDict, ImageDict]] = make_pairs(imgs, scene_graph=cfg.scene_graph)
    mask_pairs: list[tuple[ImageDict, ImageDict]] = make_pairs(masks, scene_graph=cfg.scene_graph)

    W_OG, H_OG = Image.open(img_files[0]).size
    old_focal = (W_OG**2 + H_OG**2)**0.5 * W / W_OG  # hardcoded as the default human focal from VIMO

    if cfg.use_depth_model:
        # running depth model
        print("Running depth model using PI3...")
        depth_model = Pi3X.from_pretrained("yyfz233/Pi3X").to(device).eval()
        imgs_pi3 = load_images_as_tensor(new_file_list, interval=1).to(device)
        T, _, H_PI3, W_PI3 = imgs_pi3.shape

        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16

        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=dtype):
                results = depth_model(imgs_pi3[None])

        local_points = results["local_points"]
        conf_masks = torch.sigmoid(results["conf"][..., 0]) > cfg.conf_thres
        # use recover_focal_shift function from MoGe
        focal, shift = recover_focal_shift(local_points, conf_masks)
        focal = torch.mean(focal, dim=1).unsqueeze(1).repeat(1, T)  # Use mean focal length across batch
        intrinsics = focal_to_intrinsics(focal, height=H_PI3, width=W_PI3)
        intrinsics = intrinsics[0, 0]
        intrinsics[0, :] = intrinsics[0, :] * W / W_PI3
        intrinsics[1, :] = intrinsics[1, :] * H / H_PI3

        pred_points = results['local_points'][0]
        pred_conf = torch.sigmoid(results['conf'][0][:, :, :, 0])
        pred_points = torch.nn.functional.interpolate(pred_points.permute(0, 3, 1, 2), size=(H, W), mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
        pred_conf = torch.nn.functional.interpolate(pred_conf.unsqueeze(1), size=(H, W), mode='bilinear', align_corners=False).squeeze(1)
    else:
        pred_conf, pred_points = None, None
        intrinsics = torch.tensor([[old_focal, 0, W / 2], [0, old_focal, H / 2], [0, 0, 1]], device=device)
    scene, all_human_img_idx = joint_opt(new_file_list,
                                         img_pairs,
                                         mask_pairs,
                                         cache_dir,
                                         scene_model,
                                         pred_points=pred_points,
                                         pred_conf=pred_conf,
                                         lr1=0.07,
                                         niter1=500,
                                         lr2=0.014,
                                         niter2=200,
                                         device=device,
                                         shared_intrinsics=True,
                                         init_K=intrinsics,
                                         matching_conf_thr=0.,
                                         loss_dust3r_w=0.,
                                         old_focal=old_focal,
                                         cfg=cfg)

    # get the optimized result from the scene
    optimized_result: OptimizedResult = scene_to_results_josh(scene, old_focal, all_human_img_idx, cfg)

    return optimized_result


if __name__ == "__main__":
    cfg = JOSHConfig()
    parser = ArgumentParser()
    parser.add_argument("--input_folder", type=str, default=cfg.input_folder)
    parser.add_argument("--start_frame", type=int, default=cfg.start_frame)
    parser.add_argument("--num_frames", type=int, default=cfg.num_frames)
    parser.add_argument("--visualize", action="store_true")

    args = parser.parse_args()
    cfg.input_folder = args.input_folder
    cfg.start_frame = args.start_frame
    cfg.num_frames = args.num_frames
    cfg.visualize_results = args.visualize
    device = "cuda" if torch.cuda.is_available() else "cpu"
    scene_model = AsymmetricMASt3R.from_pretrained("naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric").to(device).eval()

    result = inference(scene_model=scene_model, device=device, cfg=cfg)
    save_result = {}
    save_result["eval_metrics"] = result.eval_metrics
    save_result["pred_cam"] = np.stack([x["pred_cam"] for x in result.frame_result], axis=0)
    save_result["depth_hw"] = np.stack([x["depth_hw"] for x in result.frame_result if "depth_hw" in x], axis=0)
    save_result["rgb_hw3"] = np.stack([x["rgb_hw3"] for x in result.frame_result if "rgb_hw3" in x], axis=0)
    save_result["conf_hw"] = np.stack([x["conf_hw"] for x in result.frame_result if "conf_hw" in x], axis=0)
    save_result["intrinsics"] = result.intrinsics
    save_result["img_idx"] = cfg.img_idx

    result_file_name = os.path.join(cfg.input_folder, cfg.output_folder, "scene.pkl")
    joblib.dump(save_result, result_file_name)

    if cfg.visualize_results:
        rr.init("my_app")
        rr.connect_grpc(url='rerun+http://127.0.0.1:9876/proxy')
        blueprint = rrb.Blueprint(
            rrb.Horizontal(rrb.Spatial3DView(origin=f"test"),),
            collapse_panels=True,
        )
        rr.send_blueprint(blueprint)
        log_optimized_result(result, "test", cfg)
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("Ctrl-C received. Exiting.")
