import argparse
import os
from collections import defaultdict
from glob import glob

import joblib
import numpy as np
import open3d
import sklearn.neighbors as skln
import torch
import trimesh
from loguru import logger
from smplx import SMPL

from josh.utils.rot_utils import matrix_to_rotation_6d, rotation_6d_to_matrix
from josh.utils.sloper4d_dataset import SLOPER4D_Dataset, camera_to_pixel

"""
Evaluate JOSH reconstruction quality on SLOPER4D dataset.
Expects: data/SLOPER4D/{seq}/ with *_labels.pkl, images/, lidar_data/
         data/SLOPER4D/{seq}/josh/scene.pkl and josh/*.npy from JOSH pipeline.
Metrics: Chamfer Distance, Depth metrics, Foot Floating Rate, Collision Rate.
"""

tt_cpu = lambda x: torch.from_numpy(x).float()
tt_cuda = lambda x: torch.from_numpy(x).float().cuda()


# ======================== Utility functions ========================

def pixel_to_world(depth_map, K, Tcw, use_cuda=False):
    b, h, w = depth_map.shape
    K_inv = torch.inverse(K)
    u, v = torch.meshgrid(torch.arange(w), torch.arange(h), indexing="xy")
    if use_cuda:
        u = u.flatten()[None, :].repeat(b, 1).cuda().float()
        v = v.flatten()[None, :].repeat(b, 1).cuda().float()
    else:
        u = u.flatten()[None, :].repeat(b, 1).float()
        v = v.flatten()[None, :].repeat(b, 1).float()
    depth = depth_map.reshape(b, -1)
    pixels_hom = torch.stack((u, v, torch.ones_like(u)), dim=-1)
    cam_coords = torch.einsum("ij,bnj->bni", K_inv, pixels_hom) * depth[..., None]
    ones = torch.ones((b, cam_coords.shape[1], 1))
    if use_cuda:
        ones = ones.cuda()
    cam_coords_hom = torch.cat((cam_coords, ones), dim=-1)
    world_coords_hom = torch.einsum('bij,bnj->bni', Tcw, cam_coords_hom)
    return world_coords_hom[:, :, :3].reshape(b, h, w, -1)


def project_depth_to_image(point_cloud, extrinsics, dataset, image_height, image_width):
    ones = torch.ones((point_cloud.shape[0], 1)).cuda()
    homogeneous_points = torch.hstack((point_cloud, ones))
    camera_coordinates = (extrinsics @ homogeneous_points.T).T
    valid_mask = camera_coordinates[:, 2] > 0
    camera_coordinates = camera_coordinates[valid_mask]
    image_coordinates = camera_to_pixel(camera_coordinates[:, :3].cpu().numpy(), dataset.cam['intrinsics'], dataset.cam['dist'])
    image_coordinates = tt_cuda(image_coordinates) * image_height / dataset.cam['height']
    pixel_coordinates = torch.round(image_coordinates[:, :2]).long()
    valid_pixels = (pixel_coordinates[:, 0] >= 0) & (pixel_coordinates[:, 0] < image_width) & \
                   (pixel_coordinates[:, 1] >= 0) & (pixel_coordinates[:, 1] < image_height)
    mask2 = torch.zeros((image_height, image_width), dtype=torch.float).cuda()
    mask2[...] = torch.inf
    mask2[pixel_coordinates[valid_pixels, 1], pixel_coordinates[valid_pixels, 0]] = camera_coordinates[valid_pixels, 2]
    return mask2


def project_points_to_image(point_cloud, extrinsics, dataset, image_height, image_width):
    ones = torch.ones((point_cloud.shape[0], 1))
    homogeneous_points = torch.hstack((point_cloud, ones))
    camera_coordinates = (extrinsics @ homogeneous_points.T).T
    valid_mask = camera_coordinates[:, 2] > 0
    camera_coordinates = camera_coordinates[valid_mask]
    image_coordinates = camera_to_pixel(camera_coordinates[:, :3].cpu().numpy(), dataset.cam['intrinsics'], dataset.cam['dist'])
    image_coordinates = tt_cpu(image_coordinates) * image_height / dataset.cam['height']
    pixel_coordinates = torch.round(image_coordinates[:, :2]).long()
    valid_pixels = (pixel_coordinates[:, 0] >= 0) & (pixel_coordinates[:, 0] < image_width) & \
                   (pixel_coordinates[:, 1] >= 0) & (pixel_coordinates[:, 1] < image_height)
    mask = torch.zeros(point_cloud.shape[0], dtype=torch.bool)
    mask[torch.where(valid_mask)[0][valid_pixels]] = True
    return mask.cpu().numpy()


def compute_depth_metrics(gt_depth, pred_depth):
    assert gt_depth.shape == pred_depth.shape
    gt_depth = gt_depth.flatten()
    pred_depth = pred_depth.flatten()
    valid_mask = (gt_depth < np.inf) & (gt_depth > 0) & (pred_depth > 0)
    if np.sum(valid_mask) == 0:
        return None
    gt, pred = gt_depth[valid_mask], pred_depth[valid_mask]
    abs_rel = np.mean(np.abs(gt - pred) / gt)
    rmse = np.sqrt(np.mean((gt - pred)**2))
    log_rmse = np.sqrt(np.mean((np.log(gt) - np.log(pred))**2))
    log_diff = np.log(gt) - np.log(pred)
    si_log_rmse = np.sqrt(np.mean(log_diff**2) - np.mean(log_diff)**2)
    thresh = np.maximum(gt / pred, pred / gt)
    return {
        'Abs Rel': abs_rel, 'RMSE': rmse, 'Log RMSE': log_rmse,
        'SI Log RMSE': si_log_rmse,
        'Delta 1': np.mean(thresh < 1.25),
        'Delta 2': np.mean(thresh < 1.25**2),
        'Delta 3': np.mean(thresh < 1.25**3),
    }


def compute_chamfer(pred_points_down, gt_points_down, max_dist=1000.0):
    nn_engine = skln.NearestNeighbors(n_neighbors=1, algorithm='kd_tree', n_jobs=-1)
    nn_engine.fit(gt_points_down)
    dist_d2s, _ = nn_engine.kneighbors(pred_points_down, n_neighbors=1, return_distance=True)
    mean_d2s = dist_d2s[dist_d2s < max_dist].mean()
    nn_engine.fit(pred_points_down)
    dist_s2d, _ = nn_engine.kneighbors(gt_points_down, n_neighbors=1, return_distance=True)
    mean_s2d = dist_s2d[dist_s2d < max_dist].mean()
    return {
        'chamfer_dist': (mean_d2s + mean_s2d) / 2,
        'accuracy': mean_d2s,
        'completeness': mean_s2d,
        'accuracy_pct': np.sum(dist_d2s < max_dist) / dist_d2s.shape[0] * 1e2,
        'completeness_pct': np.sum(dist_s2d < max_dist) / dist_s2d.shape[0] * 1e2,
    }


# ======================== Main ========================

@torch.no_grad()
def main(args):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    logger.info(f'GPU name -> {torch.cuda.get_device_name()}')
    logger.info(f'GPU feat -> {torch.cuda.get_device_properties("cuda")}')

    accumulator = defaultdict(list)
    cr_thresholds = [4, 5, 6, 7, 8, 9, 10]

    for folder in sorted(os.listdir("data/SLOPER4D")):
        if "seq009" in folder:
            continue
        seq_id = folder.split("_")[0]
        root = os.path.join("data/SLOPER4D", folder)
        dataset = SLOPER4D_Dataset(glob(f'{root}/*_labels.pkl')[0], return_torch=False, fix_pts_num=True)
        extrinsics = tt_cpu(np.array(dataset.cam_pose))
        init_trans = torch.inverse(extrinsics[0])  # c0 to w

        image_folder = os.path.join(root, "images")
        hps_folder = os.path.join(root, "josh")
        scene_file = os.path.join(hps_folder, "scene.pkl")
        scene_pred = joblib.load(scene_file)

        pred_cam = tt_cpu(scene_pred["pred_cam"])
        all_cams = pred_cam[scene_pred["img_idx"]]
        all_depths = tt_cpu(scene_pred["depth_hw"])
        all_confs = tt_cpu(scene_pred["conf_hw"])
        all_colors = tt_cpu(scene_pred["rgb_hw3"])
        intrinsics = tt_cpu(scene_pred["intrinsics"][0])
        all_img_idx = tt_cpu(np.array(scene_pred["img_idx"]))

        # Reconstruct pointcloud from JOSH depth maps
        points = pixel_to_world(all_depths, intrinsics, all_cams)
        conf = torch.logical_and(all_confs > 0.1, all_depths > 0.0)

        # ==================== Chamfer Distance ====================
        conf_flat = conf.flatten()
        pred_pts_flat = points.reshape(-1, 3)[conf_flat]
        pred_colors_flat = all_colors.reshape(-1, 3)[conf_flat]

        # Transform to world frame via first GT extrinsic
        pred_pts_h = torch.cat([pred_pts_flat, torch.ones(pred_pts_flat.shape[0], 1)], dim=-1)
        pred_pts_world = torch.einsum("ij,bj->bi", init_trans, pred_pts_h)[:, :3]

        lidar_file = glob(os.path.join(root, "lidar_data", "*frames.ply"))[0]
        gt_pcd = open3d.io.read_point_cloud(lidar_file)

        pred_pcd = open3d.geometry.PointCloud()
        pred_pcd.points = open3d.utility.Vector3dVector(pred_pts_world.numpy())
        pred_pcd.colors = open3d.utility.Vector3dVector(pred_colors_flat.numpy())

        down_pred_pcd = pred_pcd.voxel_down_sample(voxel_size=0.3)
        down_gt_pcd = gt_pcd.voxel_down_sample(voxel_size=0.3)

        pred_pts_down = np.asarray(down_pred_pcd.points)
        gt_pts_down = np.asarray(down_gt_pcd.points)

        # Filter GT by camera FOV
        gt_fov_mask = np.zeros(gt_pts_down.shape[0], dtype=bool)
        for ext in extrinsics:
            gt_fov_mask |= project_points_to_image(tt_cpu(gt_pts_down), ext, dataset, 288, 512)
        gt_pts_down = gt_pts_down[gt_fov_mask]

        cd_metrics = compute_chamfer(pred_pts_down, gt_pts_down)
        for k, v in cd_metrics.items():
            accumulator[k].append(v)
        print(f"{folder} CD: {cd_metrics['chamfer_dist']:.4f}")

        # ==================== Per-frame Depth Metrics ====================
        extrinsics_cuda = extrinsics.cuda()
        for i, idx in enumerate(scene_pred["img_idx"]):
            pcd_name = f"{dataset.lidar_tstamps[idx]:.03f}".replace('.', '_') + '.pcd'
            pcd_path = os.path.join(root, 'lidar_data', 'lidar_frames_rot', pcd_name)
            if not os.path.exists(pcd_path):
                continue
            gt_depth_pcd = open3d.io.read_point_cloud(pcd_path)
            gt_depth = tt_cuda(np.asarray(gt_depth_pcd.points))
            point_depth = project_depth_to_image(gt_depth, extrinsics_cuda[idx], dataset, all_confs.shape[1], all_confs.shape[2])
            metric_dict = compute_depth_metrics(point_depth.cpu().numpy(), all_depths[i].numpy())
            if metric_dict is not None:
                for k, v in metric_dict.items():
                    accumulator[f'depth_{k}'].append(v)

        # ==================== FFR / CR ====================
        smpl_model = SMPL(model_path="data/smpl", gender="neutral").to(args.device)

        hps_files = sorted(glob(os.path.join(hps_folder, "*.npy")))
        hps_file = hps_files[0]
        pred_smpl = np.load(hps_file, allow_pickle=True).item()
        pred_rotmat = pred_smpl['pred_rotmat']
        pred_shape = pred_smpl['pred_shape']
        pred_trans = pred_smpl['pred_trans'].squeeze()
        frame = pred_smpl['frame']
        masks = torch.tensor(frame)

        mean_shape = pred_shape.mean(dim=0, keepdim=True).repeat(len(pred_shape), 1)

        # Build global SMPL via JOSH cameras
        smpl_temp = smpl_model(betas=mean_shape[[0]])
        smpl_offset = smpl_temp.joints[0, 0]
        smpl_offset_transform = torch.eye(4)
        smpl_offset_transform[:3, 3] = smpl_offset

        pred_cam_rel = torch.einsum('ij,bjk->bik', torch.inverse(pred_cam[0]), pred_cam)
        pred_cam_rel = torch.inverse(smpl_offset_transform) @ pred_cam_rel @ smpl_offset_transform

        smpl_local_trans = torch.eye(4).unsqueeze(0).repeat(masks.shape[0], 1, 1)
        smpl_local_trans[:, :3, :3] = pred_rotmat[:, 0]
        smpl_local_trans[:, :3, 3] = pred_trans
        smpl_global_trans = torch.einsum("bij,bjk->bik", pred_cam_rel[masks], smpl_local_trans)

        init_t = smpl_global_trans[0, :3, 3]
        delta_t = torch.cat([torch.zeros(1, 3), smpl_global_trans[1:, :3, 3] - smpl_global_trans[:-1, :3, 3]], dim=0)
        all_body_pose = torch.cat([
            matrix_to_rotation_6d(pred_rotmat[:, 1:]).reshape(-1, 23 * 6),
            matrix_to_rotation_6d(smpl_global_trans[:, :3, :3]),
            delta_t,
        ], dim=-1)
        cum_t = torch.cumsum(all_body_pose[:, -3:], dim=0) + init_t.unsqueeze(0)
        all_body_pose[:, -3:] = cum_t

        pred_glob = smpl_model(
            body_pose=rotation_6d_to_matrix(all_body_pose[:, :23 * 6].reshape(-1, 23, 6)),
            global_orient=rotation_6d_to_matrix(all_body_pose[:, 23 * 6:23 * 6 + 6]).reshape(-1, 1, 3, 3),
            betas=mean_shape, transl=all_body_pose[:, -3:],
            pose2rot=False, default_smpl=True)
        pred_verts_glob = pred_glob.vertices

        all_floating = []
        all_cr = {x: [] for x in cr_thresholds}
        foot_idxs = [3216, 3387, 6617, 6787]

        for start in range(0, masks[-1], 100):
            end = min(masks[-1].item(), start + 100)
            frame_mask = torch.logical_and(masks >= start, masks < end)
            verts_chunk = pred_verts_glob[frame_mask]
            img_mask = torch.logical_and(all_img_idx >= start, all_img_idx < end)
            pts_chunk = points[img_mask][conf[img_mask]].cuda()

            # FFR
            foot_verts = verts_chunk[:, foot_idxs].reshape(-1, 3).cuda()
            dist = torch.norm(foot_verts.unsqueeze(1) - pts_chunk.unsqueeze(0), dim=-1)
            torch.cuda.empty_cache()
            dist = dist.reshape(verts_chunk.shape[0], 4, -1)
            all_floating.append(dist.amin(dim=-1).amin(dim=-1) > 0.2)

            # CR
            pts_sparse = points[img_mask, ::8, ::8][conf[img_mask, ::8, ::8]]
            for i in range(0, verts_chunk.shape[0], 20):
                mesh = trimesh.Trimesh(vertices=verts_chunk[i, :, :3].numpy(), faces=smpl_model.faces)
                n_inside = mesh.contains(pts_sparse.numpy()).sum()
                for x in cr_thresholds:
                    all_cr[x].append(n_inside > x)

        all_floating = torch.cat(all_floating)
        ffr = (all_floating.sum() / all_floating.numel()).item()
        accumulator['ffr'].append(ffr)
        print(f"{folder} FFR: {ffr:.4f}")

        for x in cr_thresholds:
            cr = np.mean(all_cr[x])
            accumulator[f'cr_{x}'].append(cr)
            print(f"  CR@{x}: {cr:.4f}")

    # ==================== Summary ====================
    for k, v in accumulator.items():
        accumulator[k] = np.array(v).mean()

    log_str = 'Reconstruction Evaluation on SLOPER4D:\n'
    log_str += ' '.join([f'{k.upper()}: {v:.4f},' for k, v in accumulator.items()])
    logger.info(log_str)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default='tram')
    parser.add_argument("--device", default='cpu')
    args = parser.parse_args()
    main(args)
