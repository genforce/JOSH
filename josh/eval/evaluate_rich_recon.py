import argparse
import os
import os.path as osp
import pickle
from collections import defaultdict
from glob import glob

import joblib
import numpy as np
import open3d
import sklearn.neighbors as skln
import torch
from loguru import logger

from josh.utils.rot_utils import axis_angle_to_matrix
from smplx import SMPL
"""
Evaluate JOSH reconstruction quality on RICH dataset.
Expects: data/RICH/{seq}/ with per-frame GT SMPL pkl, images/
         data/RICH/{seq}/josh/scene.pkl and josh/*.npy from JOSH pipeline.
         data/scan_calibration/{scene}/scan_camcoord.ply for GT scan.
Metrics: Chamfer Distance (accuracy, completeness).
"""


def pixel_to_world(depth_map, K, Tcw):
    b, h, w = depth_map.shape
    K_inv = torch.inverse(K)

    # Create a meshgrid for pixel coordinates
    u, v = torch.meshgrid(torch.arange(w), torch.arange(h), indexing="xy")
    u = u.flatten()[None, :].repeat(b, 1).cuda().float()
    v = v.flatten()[None, :].repeat(b, 1).cuda().float()
    depth = depth_map.reshape(b, -1)

    # Prepare homogeneous coordinates for pixels
    pixels_hom = torch.stack((u, v, torch.ones_like(u)), dim=-1)
    # Backproject to camera coordinates
    cam_coords = torch.einsum("ij, bnj -> bni", K_inv, pixels_hom) * depth[..., None]  # b, n, 3

    # Add homogeneous coordinate for transformation
    cam_coords_hom = torch.concatenate((cam_coords, torch.ones((cam_coords.shape[0], cam_coords.shape[1], 1)).cuda()), axis=-1)

    # Transform to world coordinates
    world_coords_hom = torch.einsum('bij, bnj-> bni', Tcw, cam_coords_hom)

    # Extract world coordinates
    world_coords = world_coords_hom[:, :, :3].reshape(b * h * w, -1)

    return world_coords


m2mm = 1e3


@torch.no_grad()
def main(args):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    logger.info(f'GPU name -> {torch.cuda.get_device_name()}')
    logger.info(f'GPU feat -> {torch.cuda.get_device_properties("cuda")}')

    accumulator = defaultdict(list)
    tt = lambda x: torch.from_numpy(x).float().cuda()
    smpl = {k: SMPL(model_path="data/smpl", gender=k).cuda() for k in ['male', 'female', 'neutral']}
    scene_ids = set()
    for folder_ in sorted(os.listdir("data/RICH")):
        folder = os.path.join("data/RICH", folder_)
        scene_id = folder.split("/")[-1].split("_")[0]
        if scene_id in scene_ids:
            continue
        scene_ids.add(scene_id)
        seq_id = folder.split("/")[-1].split("_")
        seq_id = "".join(seq_id)
        image_folder = os.path.join(folder, "images")
        root = os.path.dirname(image_folder)
        hps_folder = f'{root}/josh'
        hps_files = sorted(glob(f'{hps_folder}/*.npy'))

        gt_frame = []
        poses_body = []
        poses_root = []
        betas = []
        trans = []

        for idx, image_name in enumerate(sorted(os.listdir(os.path.join(folder, "images")))):
            image_id = image_name.split("_")[0]
            if os.path.exists(os.path.join(folder, image_id)):
                gt_frame.append(idx)
                gt_file = glob(osp.join(folder, image_id, '*_smpl.pkl'))[0]
                with open(gt_file, "rb") as f:
                    gt_smpl = pickle.load(f)
                    trans.append(tt(gt_smpl["transl"]))
                    betas.append(tt(gt_smpl['betas']))
                    poses_body.append(tt(gt_smpl['body_pose']))
                    poses_root.append(tt(gt_smpl["global_orient"]))

        poses_body = torch.cat(poses_body).reshape(len(gt_frame), -1)
        betas = torch.cat(betas)
        poses_root = torch.cat(poses_root).reshape(len(gt_frame), -1)
        trans = torch.cat(trans)

        hps_file = hps_files[0]
        pred_smpl = np.load(hps_file, allow_pickle=True).item()
        pred_rotmat = pred_smpl['pred_rotmat'].cuda()
        pred_shape = pred_smpl['pred_shape'].cuda()
        pred_trans = pred_smpl['pred_trans'].squeeze().cuda()
        frame = pred_smpl['frame'].cpu().numpy().tolist()

        mask1 = [i for i, item in enumerate(frame) if item in gt_frame]
        mask2 = [i for i, item in enumerate(gt_frame) if item in frame]
        assert len(mask1) == len(mask2)
        mask1 = tt(np.array(mask1)).long()
        mask2 = tt(np.array(mask2)).long()

        mask1 = mask1[:]
        mask2 = mask2[:]
        pred_poses_body = pred_rotmat[mask1, 1:]
        pred_poses_root = pred_rotmat[mask1, 0]
        pred_shape = pred_shape[mask1]
        pred_trans = pred_trans[mask1]

        # Groundtruth global motion

        poses_body = poses_body[mask2]
        betas = betas[mask2]
        poses_root = poses_root[mask2]
        trans = trans[mask2]
        target_glob = smpl["neutral"](body_pose=poses_body, global_orient=poses_root, betas=betas, transl=trans)
        trans_eval = trans
        target_j3d_glob = target_glob.joints[:, :24]

        # =======>

        mean_shape = pred_shape.mean(dim=0, keepdim=True)
        pred_shape = mean_shape.repeat(len(pred_shape), 1)

        pred_init_transfrom = torch.eye(4)  # smpl0 to pred
        pred_init_transfrom[:3, 3] = pred_trans[0]
        pred_init_transfrom[:3, :3] = pred_poses_root[0]

        gt_init_transfrom = torch.eye(4)  # smpl0 to gt
        gt_init_transfrom[:3, 3] = trans[0]
        gt_init_transfrom[:3, :3] = axis_angle_to_matrix(poses_root[0])

        pred_to_gt = torch.einsum("ij,jk->ik", gt_init_transfrom, torch.inverse(pred_init_transfrom))  # pred to gt

        smpl_temp = smpl["neutral"](betas=pred_shape[[0]])
        smpl_offset = smpl_temp.joints[0, 0]

        smpl_offset_transform = torch.eye(4)
        smpl_offset_transform[:3, 3] = smpl_offset
        pred_to_gt = smpl_offset_transform @ pred_to_gt @ torch.inverse(smpl_offset_transform)

        pred = smpl["neutral"](body_pose=pred_poses_body,
                               global_orient=pred_poses_root.unsqueeze(1),
                               betas=pred_shape,
                               transl=pred_trans,
                               pose2rot=False,
                               default_smpl=True)
        pred_verts_cam = pred.vertices
        pred_j3d_cam = pred.joints[:, :24]
        # pred_trans = trans_cam_eval[masks]

        # get pred cam for seqs

        #---------------------------------mast3r-------------------------------------
        scene_file = f"{hps_folder}/scene.pkl"
        scene_pred = joblib.load(scene_file)
        pred_cam = tt(scene_pred["pred_cam"])

        all_cams = pred_cam[scene_pred["img_idx"]]
        all_depths = tt(scene_pred["depth_hw"])
        all_confs = tt(scene_pred["conf_hw"])
        all_colors = tt(scene_pred["rgb_hw3"])
        intrinsics = tt(scene_pred["intrinsics"][0])
        pred_cam = pred_cam[frame][mask1]

        all_cams = torch.einsum('ij,bjk->bik', torch.inverse(pred_cam[0]), all_cams)
        all_cams = torch.einsum('ij,bjk->bik', pred_to_gt.cuda(), all_cams)
        points = pixel_to_world(all_depths, intrinsics, all_cams)
        conf = torch.logical_and(all_confs > 0.1, all_depths.flatten() > 0)
        conf = conf.flatten()
        points = points[conf]
        colors = all_colors.reshape(-1, 3)[conf]
        pred_points = points
        pred_colors = colors
        gt_pcd = open3d.io.read_point_cloud(f"data/scan_calibration/{scene_id}/scan_camcoord.ply")

        pred_pcd = open3d.geometry.PointCloud()
        pred_pcd.points = open3d.utility.Vector3dVector(pred_points.cpu().numpy())
        pred_pcd.colors = open3d.utility.Vector3dVector(pred_colors.cpu().numpy())

        down_pred_pcd = pred_pcd.voxel_down_sample(voxel_size=0.3)
        down_gt_pcd = gt_pcd.voxel_down_sample(voxel_size=0.3)
        pred_points_down = np.asarray(down_pred_pcd.points)
        gt_points_down = np.asarray(down_gt_pcd.points)

        nn_engine = skln.NearestNeighbors(n_neighbors=1, radius=1.0, algorithm='kd_tree', n_jobs=-1)
        nn_engine.fit(gt_points_down)
        dist_d2s, idx_d2s = nn_engine.kneighbors(pred_points_down, n_neighbors=1, return_distance=True)
        max_dist = 1000.0
        mean_d2s = dist_d2s[dist_d2s < max_dist].mean()
        nn_engine.fit(pred_points_down)

        dist_s2d, idx_s2d = nn_engine.kneighbors(gt_points_down, n_neighbors=1, return_distance=True)
        mean_s2d = dist_s2d[dist_s2d < max_dist].mean()

        accuracy = mean_d2s
        accuracy_percent = np.sum(dist_d2s < max_dist) / dist_d2s.shape[0] * 1e2
        completeness = mean_s2d
        completeness_percent = np.sum(dist_s2d < max_dist) / dist_s2d.shape[0] * 1e2
        # Compute the Chamfer distance
        chamfer_dist = (accuracy + completeness) / 2

        print(folder)
        print(f"chamfer_dist: {chamfer_dist}")
        print(f"accuracy: {accuracy}")
        print(f"completeness: {completeness}")
        print(f"accuracy percent: {accuracy_percent}")
        print(f"completeness percent: {completeness_percent}")
        accumulator['chamfer_dist'].append(chamfer_dist)
        accumulator['accuracy'].append(accuracy)
        accumulator['completeness'].append(completeness)
        accumulator['accuracy_percent'].append(accuracy_percent)
        accumulator['completeness_percent'].append(completeness_percent)

    for k, v in accumulator.items():
        accumulator[k] = np.array(v).mean()

    print('')
    log_str = f'Evaluation on RICH, '
    log_str += ' '.join([f'{k.upper()}: {v:.4f},' for k, v in accumulator.items()])
    logger.info(log_str)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default='cpu')
    args = parser.parse_args()

    main(args)
