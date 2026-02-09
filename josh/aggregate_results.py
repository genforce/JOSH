import os
import re
import time
from argparse import ArgumentParser
from glob import glob

import joblib
import numpy as np
import open3d
import rerun as rr
import rerun.blueprint as rrb
import torch
import trimesh
from smplx import SMPL

from josh.config import JOSHConfig, OptimizedFrameResult, OptimizedResult
from josh.inference import log_optimized_result
from josh.utils.rot_utils import (matrix_to_rotation_6d, pixel_to_world, rotation_6d_to_matrix)


def aggregate_results(input_folder, cfg: JOSHConfig) -> OptimizedResult:
    josh_folders = sorted(glob(os.path.join(input_folder, "josh_*")))
    folder_info = []
    for folder in josh_folders:
        folder_name = os.path.basename(folder)
        # Extract start and end frame from folder name like "josh_0-20"
        match = re.match(r"josh_(\d+)-(\d+)", folder_name)
        if match:
            start_frame = int(match.group(1))
            end_frame = int(match.group(2))
            scene_file = os.path.join(folder, "scene.pkl")
            if os.path.exists(scene_file):
                folder_info.append({'folder': folder, 'start_frame': start_frame, 'end_frame': end_frame, 'scene_file': scene_file})

    folder_info = sorted(folder_info, key=lambda x: x['start_frame'])
    print(f"Processing {len(folder_info)} chunks:")
    for info in folder_info:
        print(f"  - {os.path.basename(info['folder'])}: frames {info['start_frame']}-{info['end_frame']}")

    pred_cams = []
    all_cams = []
    all_intrinsics = []
    all_depths = []
    all_confs = []
    all_colors = []
    all_idx = []
    for file_info in folder_info:
        ff = joblib.load(file_info['scene_file'])
        pred_cam = ff["pred_cam"]
        if len(pred_cams) > 0:
            pred_cam = np.einsum("ij, bjk->bik", pred_cams[-1][-1], pred_cam)
            pred_cams.append(pred_cam[1:])
            current_depth = ff["depth_hw"][1:]
            conf = ff["conf_hw"][1:]
            img_idx = np.array(ff["img_idx"][1:])
            colors = ff["rgb_hw3"][1:]
            cam_pose = pred_cam[img_idx]
            img_idx = img_idx + file_info['start_frame']
        else:
            pred_cams.append(pred_cam)
            current_depth = ff["depth_hw"]
            conf = ff["conf_hw"]
            img_idx = np.array(ff["img_idx"])
            colors = ff["rgb_hw3"]
            cam_pose = pred_cam[img_idx]
        all_depths.append(current_depth)
        all_confs.append(conf)
        all_colors.append(colors)
        all_cams.append(cam_pose)
        all_idx.append(img_idx)
        all_intrinsics.append(ff["intrinsics"])

    all_depths = np.concatenate(all_depths, axis=0)
    all_confs = np.concatenate(all_confs, axis=0)
    all_colors = np.concatenate(all_colors, axis=0)
    all_cams = np.concatenate(all_cams, axis=0)
    all_intrinsics = np.stack(all_intrinsics, axis=0).mean(axis=0)
    all_idx = np.concatenate(all_idx, axis=0).tolist()
    pred_cams = np.concatenate(pred_cams, axis=0)

    points = pixel_to_world(torch.tensor(all_depths).float(), torch.tensor(all_intrinsics).float(), torch.tensor(all_cams).float())
    conf = all_confs.flatten()
    conf = np.logical_and(conf > cfg.conf_thres, all_depths.flatten() > 0)
    points = points[conf]
    colors = all_colors.reshape(-1, 3)[conf]

    pred_pcd = open3d.geometry.PointCloud()
    pred_pcd.points = open3d.utility.Vector3dVector(points)
    pred_pcd.colors = open3d.utility.Vector3dVector(colors)
    down_pred_pcd = pred_pcd.voxel_down_sample(voxel_size=0.01)  # downsampling
    point_cloud = trimesh.PointCloud(np.asarray(down_pred_pcd.points), np.asarray(down_pred_pcd.colors))

    # aggregate smpl results
    pred_cam = pred_cams
    smpl_files = sorted(glob(os.path.join(input_folder, "tram", "*.npy")))
    pred_smpls = []
    pred_contacts = []
    pred_frames = []
    pred_ids = []
    for i, smpl_file in enumerate(smpl_files):
        pred_smpl_dict = np.load(smpl_file, allow_pickle=True).item()
        pred_ids.append(pred_smpl_dict['id'])
        pred_contacts.append(torch.tensor(np.load(smpl_file.replace('tram', 'deco'))))
        pred_rotmat = pred_smpl_dict['pred_rotmat']
        pred_shape = pred_smpl_dict['pred_shape']
        pred_trans = pred_smpl_dict['pred_trans'].squeeze(1)
        pred_frame = pred_smpl_dict['frame'].tolist()
        for file_info in folder_info:
            if not os.path.exists(os.path.join(file_info['folder'], smpl_file.split("/")[-1])):
                continue
            josh_smpl_dict = np.load(os.path.join(file_info['folder'], smpl_file.split("/")[-1]), allow_pickle=True).item()
            frame_mask = np.isin(pred_frame, josh_smpl_dict['frame'])
            pred_rotmat[frame_mask] = josh_smpl_dict['pred_rotmat']
            pred_shape[frame_mask] = josh_smpl_dict['pred_shape']
            pred_trans[frame_mask] = josh_smpl_dict['pred_trans'].squeeze(1)

        pred_smpl_dict['pred_rotmat'] = pred_rotmat
        pred_smpl_dict['pred_shape'] = pred_shape
        pred_smpl_dict['pred_trans'] = pred_trans.unsqueeze(1)
        np.save(smpl_file.replace('tram', 'josh'), pred_smpl_dict)

        tt = lambda x: torch.Tensor(x).float()
        smpl = SMPL(model_path="data/smpl")
        pred_smpl = smpl(body_pose=pred_rotmat[:, 1:],
                         global_orient=pred_rotmat[:, [0]],
                         betas=pred_shape,
                         transl=pred_trans,
                         pose2rot=False,
                         default_smpl=True)

        smpl_temp = smpl(betas=pred_shape.mean(0, keepdim=True))
        smpl_offset = smpl_temp.joints[0, 0]

        smpl_offset_transform = torch.eye(4)
        smpl_offset_transform[:3, 3] = smpl_offset
        pred_cam_torch = torch.tensor(pred_cam).float()
        pred_cam_torch = torch.inverse(smpl_offset_transform) @ pred_cam_torch @ smpl_offset_transform
        smpl_local_trans = torch.eye(4).unsqueeze(0).repeat(pred_trans.shape[0], 1, 1)
        smpl_local_trans[:, :3, :3] = pred_rotmat[:, 0]
        smpl_local_trans[:, :3, 3] = pred_trans

        smpl_global_trans = torch.einsum("bij, bjk->bik",
                                         torch.tensor(
                                             pred_cam_torch[pred_smpl_dict['frame'] - cfg.start_frame],
                                             device=smpl_local_trans.device,
                                         ), smpl_local_trans)

        init_trans = smpl_global_trans[0, :3, 3]
        delta_trans = smpl_global_trans[1:, :3, 3] - smpl_global_trans[:-1, :3, 3]

        delta_trans = torch.concat([torch.zeros(1, 3), delta_trans], dim=0)

        all_body_pose = torch.concat(
            [matrix_to_rotation_6d(pred_rotmat[:, 1:]).reshape(-1, 23 * 6),
             matrix_to_rotation_6d(smpl_global_trans[:, :3, :3]), delta_trans], dim=-1)

        delta_trans = all_body_pose[:, -3:]
        delta_trans = torch.cumsum(delta_trans, dim=0)
        delta_trans += init_trans.unsqueeze(0)
        all_body_pose[:, -3:] = delta_trans
        pred_glob = smpl(body_pose=rotation_6d_to_matrix(all_body_pose[:, :23 * 6].reshape(-1, 23, 6)),
                         global_orient=rotation_6d_to_matrix(all_body_pose[:, 23 * 6:23 * 6 + 6]).reshape(-1, 1, 3, 3),
                         betas=pred_shape,
                         transl=all_body_pose[:, -3:],
                         pose2rot=False,
                         default_smpl=True)

        pred_smpls.append(pred_glob)
        pred_frames.append(pred_frame)

    # add smpl mesh in global
    pred_smpl_mesh = {x: [] for x in range(len(pred_cams))}
    pred_joints_all = []

    for pred_id, pred_smpl, pred_frame, pred_contact in zip(pred_ids, pred_smpls, pred_frames, pred_contacts):
        for i in range(len(pred_cams)):
            current_idx = i + cfg.start_frame
            if current_idx in pred_frame:
                frame = pred_frame.index(current_idx)
                pred_vertices = pred_smpl.vertices[frame]
                pred_joints = pred_smpl.joints[frame, :24]

                pred_smpl_mesh[i].append([pred_id, trimesh.Trimesh(vertices=pred_vertices[:, :3], faces=smpl.faces), pred_contact[frame]])
                pred_joints_all.append(pred_joints[:, :3])

    frame_results = []

    for i in range(len(pred_cams)):
        frame_result = OptimizedFrameResult(frame_idx=i, pred_cam=pred_cams[i])
        frame_result["pred_smpl"] = pred_smpl_mesh[i]

        if i in all_idx:
            idx = all_idx.index(i)
            frame_result["rgb_hw3"] = all_colors[idx]
            frame_result["depth_hw"] = all_depths[idx]
            frame_result["conf_hw"] = all_confs[idx]

        frame_results.append(frame_result)

    optimised_result = OptimizedResult(
        point_cloud=point_cloud,
        mesh=None,
        intrinsics=all_intrinsics,  # intrinsics is fixed
        img_size=(all_colors[0].shape[1], all_colors[0].shape[0]),
        frame_result=frame_results,
        eval_metrics={
            "w_jpe": 0.,
            "wa_jpe": 0.
        })

    return optimised_result, all_idx


if __name__ == "__main__":
    cfg = JOSHConfig()
    parser = ArgumentParser()
    parser.add_argument("--input_folder", type=str, default=cfg.input_folder)
    parser.add_argument("--visualize", action="store_true")
    args = parser.parse_args()
    cfg.input_folder = args.input_folder
    cfg.visualize_results = args.visualize

    result, all_idx = aggregate_results(cfg.input_folder, cfg)
    os.makedirs(os.path.join(cfg.input_folder, cfg.output_folder), exist_ok=True)
    save_result = {}
    save_result["eval_metrics"] = result.eval_metrics
    save_result["pred_cam"] = np.stack([x["pred_cam"] for x in result.frame_result], axis=0)
    save_result["depth_hw"] = np.stack([x["depth_hw"] for x in result.frame_result if "depth_hw" in x], axis=0)
    save_result["rgb_hw3"] = np.stack([x["rgb_hw3"] for x in result.frame_result if "rgb_hw3" in x], axis=0)
    save_result["conf_hw"] = np.stack([x["conf_hw"] for x in result.frame_result if "conf_hw" in x], axis=0)
    save_result["intrinsics"] = result.intrinsics
    save_result["img_idx"] = all_idx

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
