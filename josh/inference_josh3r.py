import os
import sys
import time
from argparse import ArgumentParser
from glob import glob
from pathlib import Path

sys.path.append("./third_party/mast3r")

import joblib
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from tqdm import tqdm

from josh.config import JOSHConfig, OptimizedResult
from josh.josh3r.model import JOSH3R
from josh.utils.image_utils import generate_colors
from josh.utils.rot_utils import (axis_angle_to_matrix, matrix_to_axis_angle, rotation_6d_to_matrix)
from smplx import SMPL


def preprocess_image(img_path, long_edge_size=512):
    img = Image.open(img_path).convert("RGB")
    S = max(img.size)
    interp = Image.LANCZOS if S > long_edge_size else Image.BICUBIC
    new_size = tuple(int(round(x * long_edge_size / S)) // 16 * 16 for x in img.size)
    img = img.resize(new_size, interp)
    true_shape = torch.tensor(np.int32(img.size[::-1])).long()
    img_tensor = TF.to_tensor(img)
    return img_tensor, true_shape, long_edge_size / S


def center_crop_image(img, bbox, scale):
    _, H, W = img.shape
    new_W = int(W * scale)
    new_H = int(H * scale)
    crop_x1 = (W - new_W) // 2
    crop_y1 = (H - new_H) // 2
    x1, y1, x2, y2 = bbox
    cropped_img = TF.center_crop(img, [new_H, new_W])
    cropped_img = TF.resize(cropped_img, [H, W])
    cropped_bbox = torch.tensor([
        (max(x1, 0) - crop_x1) / scale,
        (max(y1, 0) - crop_y1) / scale,
        (min(x2, W) - crop_x1) / scale,
        (min(y2, H) - crop_y1) / scale,
    ])
    return cropped_img, cropped_bbox


def interpolate_se3(poses, times, query_times):
    translations = np.array([p[:3, 3] for p in poses])
    rotations = R.from_matrix([p[:3, :3] for p in poses])
    interp_t = interp1d(times, translations, axis=0)
    slerp = Slerp(times, rotations)
    out = np.zeros((len(query_times), 4, 4))
    out[:, 3, 3] = 1
    out[:, :3, 3] = interp_t(query_times)
    out[:, :3, :3] = slerp(query_times).as_matrix()
    return out


def log_optimized_result(optimized_result: OptimizedResult, parent_log_path: str, cfg: JOSHConfig):
    colors = generate_colors(n_colors=128, n_samples=5000)
    rr.log(f"{parent_log_path}", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)
    rr.log(
        f"{parent_log_path}/pointcloud",
        rr.Points3D(
            positions=optimized_result.point_cloud.vertices,
            colors=optimized_result.point_cloud.colors,
        ),
        static=True,
    )
    if optimized_result.mesh is not None:
        rr.log(
            f"{parent_log_path}/mesh",
            rr.Mesh3D(
                vertex_positions=optimized_result.mesh.vertices,
                vertex_colors=optimized_result.mesh.visual.vertex_colors,
                triangle_indices=optimized_result.mesh.faces,
            ),
            static=True,
        )
    intrinsics = optimized_result.intrinsics
    for i, frame_result in enumerate(tqdm(optimized_result.frame_result, desc="Logging")):
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
            for idx, pred_smpl_mesh, pred_contact in frame_result["pred_smpl"]:
                rr.log(
                    f"{parent_log_path}/pred_smpl_mesh/{idx}",
                    rr.Mesh3D(
                        vertex_positions=pred_smpl_mesh.vertices,
                        triangle_indices=pred_smpl_mesh.faces,
                        vertex_normals=pred_smpl_mesh.vertex_normals,
                        albedo_factor=colors[idx],
                    ),
                )
        if "rgb_hw3" in frame_result:
            rr.log(f"{pred_log_path}/pinhole/rgb", rr.Image(frame_result["rgb_hw3"]))


@torch.no_grad()
def inference_josh3r(model, smpl_model, device, cfg: JOSHConfig) -> OptimizedResult:
    import trimesh

    input_folder = cfg.input_folder
    img_files = sorted(glob(os.path.join(input_folder, "rgb", "*.jpg")))

    # Load HPS predictions (TRAM) — only frame 0 used as anchor
    hps_folder = os.path.join(input_folder, "tram")
    hps_file = sorted(glob(os.path.join(hps_folder, "*.npy")))[0]
    pred_smpl = np.load(hps_file, allow_pickle=True).item()
    pred_rotmat = pred_smpl['pred_rotmat']  # (N, 24, 3, 3)
    pred_shape = pred_smpl['pred_shape']  # (N, 10)
    pred_trans = pred_smpl['pred_trans'].squeeze()  # (N, 3) smpl-to-camera
    frame = pred_smpl['frame'].numpy().astype(int)
    bboxes = pred_smpl['bbox'][:, :4].numpy() if 'bbox' in pred_smpl else None
    body_pose = pred_rotmat[:, 1:]  # (N, 23, 3, 3)

    mean_shape = pred_shape.mean(dim=0, keepdim=True).repeat(len(pred_shape), 1)

    W_OG, H_OG = Image.open(img_files[0]).size
    _, H_proc, W_proc = preprocess_image(img_files[0])[0].shape

    # Frame-0 anchor for initialization
    smpl_to_c0 = torch.eye(4)
    smpl_to_c0[:3, :3] = pred_rotmat[0, 0]
    smpl_to_c0[:3, 3] = pred_trans[0]

    # --- Sliding-window: accumulate rel_trans/rel_rot in SMPL (human) frame ---
    # Accumulated result = smpl_to_world (human root trajectory in world)
    pred_global_trans = torch.zeros((1, 3))
    pred_global_rotmat = torch.eye(3).unsqueeze(0)
    current_global_rotmat = torch.eye(3)

    head = 0
    tail = 0
    pred_frame = [frame[0]]
    pbar = tqdm(total=len(frame) - 1, desc="JOSH3R inference")

    keyframe_data = []  # (X11_scaled, img_tensor, step_idx) for pointcloud

    while head < len(frame) - 1:
        while tail < len(frame) - 1 and frame[tail] - frame[head] < 3:
            tail += 1

        head_frame = frame[head]
        tail_frame = frame[tail]

        img_1, true_shape_1, sf = preprocess_image(img_files[head_frame])
        img_2, true_shape_2, _ = preprocess_image(img_files[tail_frame])

        img_1_norm = TF.normalize(img_1, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        img_2_norm = TF.normalize(img_2, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

        batch_dict = {
            "imgs": torch.stack([img_1_norm, img_2_norm], dim=0).unsqueeze(0).to(device),
            "img_shapes": torch.stack([true_shape_1, true_shape_2], dim=0).unsqueeze(0).to(device),
        }

        if bboxes is not None:
            bbox_pair = torch.tensor([bboxes[head], bboxes[tail]], dtype=torch.float32) * sf
        else:
            bbox_pair = torch.tensor([[0, 0, W_proc, H_proc], [0, 0, W_proc, H_proc]], dtype=torch.float32)
        batch_dict["bbox"] = bbox_pair.unsqueeze(0).to(device)

        img1_c, bbox1_c = center_crop_image(batch_dict['imgs'][0, 0], batch_dict["bbox"][0, 0], 0.5)
        img2_c, bbox2_c = center_crop_image(batch_dict['imgs'][0, 1], batch_dict["bbox"][0, 1], 0.5)
        batch_dict["imgs"] = torch.stack([img1_c, img2_c], dim=0).unsqueeze(0).to(device)
        batch_dict["bbox"] = torch.stack([bbox1_c, bbox2_c], dim=0).unsqueeze(0).to(device)

        out = model(batch_dict)
        out = [x.cpu() for x in out]
        rel_trans_pred, rel_rot_pred = out[0], out[1]
        X11, X21, scale1, scale2 = out[4], out[5], out[6], out[7]

        # Accumulate in SMPL/human frame
        rel_rot_mat = rotation_6d_to_matrix(rel_rot_pred)
        rel_trans_pred = torch.einsum("ij,bj->bi", current_global_rotmat, rel_trans_pred)
        rel_rot_mat = torch.einsum("ij,bjk->bik", current_global_rotmat, rel_rot_mat)

        pred_global_trans = torch.cat([pred_global_trans, rel_trans_pred + pred_global_trans[[-1]]], dim=0)
        pred_global_rotmat = torch.cat([pred_global_rotmat, rel_rot_mat], dim=0)
        current_global_rotmat = rel_rot_mat.squeeze()

        # Store X11 every 10 keyframes for pointcloud
        step_idx = len(pred_frame) - 1
        if step_idx % 10 == 0:
            keyframe_data.append((
                X11[0].numpy() * scale1[0].numpy(),
                img_1,
                step_idx,
            ))

        pbar.update(tail - head)
        pred_frame.append(tail_frame)
        head = tail

    pbar.close()

    # --- Interpolate smpl_to_world to all HPS frames ---
    all_smpl_to_w = np.eye(4)[None].repeat(len(pred_frame), 0)
    all_smpl_to_w[:, :3, :3] = pred_global_rotmat.numpy()
    all_smpl_to_w[:, :3, 3] = pred_global_trans.numpy()
    all_smpl_to_w = interpolate_se3(all_smpl_to_w, np.array(pred_frame), frame)
    smpl_to_w = torch.from_numpy(all_smpl_to_w).float()

    # --- Derive camera-to-world per frame: c_to_w[i] = smpl_to_w[i] @ inv(smpl_to_c[i]) ---
    # Per-frame smpl_to_c from TRAM
    N = len(frame)
    smpl_to_c = torch.eye(4).unsqueeze(0).repeat(N, 1, 1)
    smpl_to_c[:, :3, :3] = pred_rotmat[:, 0]  # per-frame global orient
    smpl_to_c[:, :3, 3] = pred_trans  # per-frame translation
    c_to_w = torch.einsum("bij,bjk->bik", smpl_to_w, torch.inverse(smpl_to_c))

    # --- Build SMPL meshes directly in world frame ---
    # global_orient = smpl_to_w rotation, transl = smpl_to_w translation
    # body_pose from TRAM (per-frame joint rotations)
    N = len(frame)
    smpl_output = smpl_model(
        body_pose=matrix_to_axis_angle(body_pose).reshape(N, 23 * 3),
        global_orient=matrix_to_axis_angle(smpl_to_w[:, :3, :3].unsqueeze(1)).reshape(N, 3),
        betas=mean_shape,
        transl=smpl_to_w[:, :3, 3],
    )
    verts_world = smpl_output.vertices  # (N, V, 3) in world frame

    # --- Fuse keyframe pointclouds into world frame ---
    # X11 is in camera space of that keyframe -> transform via c_to_w
    all_pts = []
    all_colors = []
    for X_local, img_tensor, step_idx in keyframe_data:
        kf_frame = pred_frame[step_idx]
        kf_idx = np.searchsorted(frame, kf_frame)
        c2w = c_to_w[kf_idx].numpy()
        pts = X_local.reshape(-1, 3)
        pts_h = np.concatenate([pts, np.ones((pts.shape[0], 1))], axis=-1)
        pts_world = (c2w @ pts_h.T).T[:, :3]
        colors = (img_tensor.permute(1, 2, 0).numpy().reshape(-1, 3) * 255).astype(np.uint8)
        all_pts.append(pts_world)
        all_colors.append(colors)

    all_pts = np.concatenate(all_pts, axis=0)
    all_colors = np.concatenate(all_colors, axis=0)
    point_cloud = trimesh.PointCloud(vertices=all_pts, colors=all_colors)

    # --- Intrinsics ---
    old_focal = (W_OG**2 + H_OG**2)**0.5 * W_proc / W_OG
    intrinsics = np.array([[old_focal, 0, W_proc / 2], [0, old_focal, H_proc / 2], [0, 0, 1]])

    # --- Build frame results ---
    smpl_faces = smpl_model.faces
    frame_results = []
    for i, fi in enumerate(frame):
        verts_w_i = verts_world[i].numpy()
        smpl_mesh = trimesh.Trimesh(vertices=verts_w_i, faces=smpl_faces, process=False)

        fr = {
            "pred_cam": c_to_w[i].numpy(),  # camera_i -> world
            "pred_smpl": [(0, smpl_mesh, torch.zeros(verts_w_i.shape[0]))],
        }
        if fi < len(img_files):
            rgb = np.array(Image.open(img_files[fi]).convert("RGB").resize((W_proc, H_proc)))
            fr["rgb_hw3"] = rgb

        frame_results.append(fr)

    return OptimizedResult(
        point_cloud=point_cloud,
        mesh=None,
        intrinsics=intrinsics,
        img_size=(W_proc, H_proc),
        frame_result=frame_results,
        eval_metrics={},
    )


if __name__ == "__main__":
    cfg = JOSHConfig()
    parser = ArgumentParser()
    parser.add_argument("--input_folder", type=str, default=cfg.input_folder)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--visualize", action="store_true")
    args = parser.parse_args()

    cfg.input_folder = args.input_folder
    cfg.visualize_results = args.visualize

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load JOSH3R
    model = JOSH3R().to(device)
    ckpt = torch.load(args.ckpt_path, map_location=device)
    model.load_state_dict(ckpt if "state_dict" not in ckpt else ckpt["state_dict"])
    model.eval()

    smpl_model = SMPL(model_path="data/smpl", gender="neutral")

    result = inference_josh3r(model, smpl_model, device, cfg)

    # Save
    cfg.output_folder = f"josh3r_{cfg.start_frame}"
    os.makedirs(os.path.join(cfg.input_folder, cfg.output_folder), exist_ok=True)

    save_result = {
        "pred_cam": np.stack([x["pred_cam"] for x in result.frame_result], axis=0),
        "intrinsics": result.intrinsics,
    }
    if any("rgb_hw3" in x for x in result.frame_result):
        save_result["rgb_hw3"] = np.stack([x["rgb_hw3"] for x in result.frame_result if "rgb_hw3" in x], axis=0)

    joblib.dump(save_result, os.path.join(cfg.input_folder, cfg.output_folder, "scene.pkl"))

    if cfg.visualize_results:
        rr.init("josh3r_vis")
        rr.connect_grpc(url='rerun+http://127.0.0.1:9876/proxy')
        blueprint = rrb.Blueprint(
            rrb.Horizontal(rrb.Spatial3DView(origin="test")),
            collapse_panels=True,
        )
        rr.send_blueprint(blueprint)
        log_optimized_result(result, "test", cfg)
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("Ctrl-C received. Exiting.")
