import os
from glob import glob

import numpy as np
import torch
import trimesh
from mast3r.cloud_opt.utils.losses import gamma_loss
from smplx import SMPL
from smplx.lbs import vertices2joints
from tqdm import tqdm

from josh.config import JOSHConfig, OptimizedFrameResult, OptimizedResult
from josh.utils.josh_utils import (SparseGA, compute_min_spanning_tree, condense_data, convert_dust3r_pairs_naming, forward_mast3r_w_mask,
                                   prepare_canonical_data, sparse_scene_optimizer)
from josh.utils.mesh_utils import cat_meshes, pts3d_to_trimesh
from josh.utils.rot_utils import (interpolate_se3, interpolate_so3, matrix_to_quaternion, matrix_to_rotation_6d, quaternion_to_matrix, rotation_6d_to_matrix)


def scene_to_results_josh(scene: SparseGA, old_focal, all_smpl_img_idx, cfg: JOSHConfig) -> OptimizedResult:
    ### get camera parameters K and T

    world_T_cam_b44: np.ndarray = scene.get_im_poses().numpy(force=True)

    K_b33: np.ndarray = scene.intrinsics.numpy(force=True)
    ### image, confidence, depths
    rgb_hw3_list: np.ndarray = scene.imgs

    pts3d_list, depth_hw_list, conf_hw_list = scene.get_dense_pts3d()

    depth_hw_list = [depth.reshape(conf_hw_list[0].shape[0], conf_hw_list[0].shape[1]).numpy(force=True) for depth in depth_hw_list]
    pts3d_list = [pts3d.reshape(conf_hw_list[0].shape[0], conf_hw_list[0].shape[1], 3).numpy(force=True) for pts3d in pts3d_list]
    conf_hw_list = [conf.numpy(force=True) for conf in conf_hw_list]

    # set the minimum confidence threshold
    masks_list: np.ndarray = [conf > cfg.conf_thres for conf in conf_hw_list]
    # normalize the point cloud and apply the scale
    normalize_transform = np.linalg.inv(world_T_cam_b44[0])

    world_T_cam_b44 = np.einsum('ij,bjk->bik', normalize_transform, world_T_cam_b44)

    for i, pts3d in enumerate(pts3d_list):
        pts3d = np.concatenate([pts3d, np.ones([pts3d.shape[0], pts3d.shape[1], 1])], axis=-1)
        pts3d = np.einsum('ij,hwj->hwi', normalize_transform, pts3d)
        pts3d_list[i] = pts3d[..., :3]

    point_cloud: np.ndarray = np.concatenate([p[m] for p, m in zip(pts3d_list, masks_list)])
    colors: np.ndarray = np.concatenate([p[m] for p, m in zip(rgb_hw3_list, masks_list)])
    point_cloud = trimesh.PointCloud(point_cloud.reshape(-1, 3), colors=colors.reshape(-1, 3))

    meshes = []
    pbar = tqdm(zip(rgb_hw3_list, pts3d_list, masks_list), total=len(rgb_hw3_list))
    for rgb_hw3, pts3d, mask in pbar:
        meshes.append(pts3d_to_trimesh(rgb_hw3, pts3d, mask))

    mesh = trimesh.Trimesh(**cat_meshes(meshes))

    pred_cam = interpolate_se3(world_T_cam_b44, times=np.array(cfg.img_idx), query_times=np.arange(cfg.num_frames))

    # add smpl paramters to optimized results

    smpl_files = sorted(glob(os.path.join(cfg.input_folder, "tram", "*.npy")))
    pred_smpls = []
    pred_contacts = []
    pred_frames = []
    pred_ids = []
    for i, smpl_file in enumerate(smpl_files):
        pred_smpl_dict = np.load(smpl_file, allow_pickle=True).item()
        pred_ids.append(pred_smpl_dict['id'])
        pred_rotmat = pred_smpl_dict['pred_rotmat']
        pred_shape = pred_smpl_dict['pred_shape']
        pred_trans = pred_smpl_dict['pred_trans'].squeeze(1)
        pred_frame = pred_smpl_dict['frame'].tolist()
        pred_trans[:, 2] = pred_trans[:, 2] * K_b33[0][0][0] / old_focal

        pred_contacts.append(torch.tensor(np.load(smpl_file.replace('tram', 'deco'))))

        smpl_img_idx = all_smpl_img_idx[i]
        if scene.smpl_transforms is not None and scene.smpl_transforms[i][0].shape[0] > 0:
            final_orient, final_trans, final_body_pose, final_body_shape = scene.smpl_transforms[i]
            final_orient = quaternion_to_matrix(final_orient).squeeze(1)
            final_orient = final_orient.detach().cpu().numpy()
            final_trans = final_trans.detach().cpu().numpy()
            final_body_pose = final_body_pose.detach().cpu().numpy()
            final_pose = np.concatenate([final_orient, final_trans[..., np.newaxis]], axis=-1)
            vec = np.repeat(np.array([[[0, 0, 0, 1]]]), final_pose.shape[0], axis=0)
            final_pose = np.hstack([final_pose, vec])
            interp_idxs = [i for i in cfg.img_idx if i + cfg.start_frame in pred_frame]
            start_idx = interp_idxs[0]
            end_idx = min(cfg.num_frames, interp_idxs[-1] + 1)
            if len(final_pose) > 1:
                final_pose = interpolate_se3(
                    final_pose,
                    #  np.array(IMG_IDX),
                    times=np.array(interp_idxs),
                    query_times=np.arange(start_idx, end_idx))

                # Interpolate body pose
                final_body_pose_interp = []
                for i in range(final_body_pose.shape[1]):
                    final_body_pose_interp.append(interpolate_so3(final_body_pose[:, i], times=np.array(interp_idxs), query_times=np.arange(start_idx,
                                                                                                                                            end_idx)))
                final_body_pose_interp = np.stack(final_body_pose_interp, axis=1)
            final_pose = torch.tensor(final_pose, device=pred_rotmat.device).float()
            final_pose = final_pose.unsqueeze(1)
            final_orient = final_pose[..., :3, :3]
            final_trans = final_pose[..., :3, 3].squeeze(1)
            final_body_pose = torch.tensor(final_body_pose_interp, device=pred_rotmat.device).float()
            # replace the predicted smpl with optimized smpl
            for i in range(start_idx, end_idx):
                if i + cfg.start_frame in pred_frame:
                    frame = pred_frame.index(i + cfg.start_frame)
                    pred_rotmat[frame, 0] = final_orient[i - start_idx]
                    pred_rotmat[frame, 1:] = final_body_pose[i - start_idx]
                    pred_trans[frame] = final_trans[i - start_idx]
            pred_shape = final_body_shape.detach().cpu().unsqueeze(0).repeat(pred_shape.shape[0], 1)

        frame_mask = torch.logical_and(pred_smpl_dict['frame'] >= cfg.start_frame, pred_smpl_dict['frame'] < cfg.start_frame + cfg.num_frames)
        pred_rotmat = pred_rotmat[frame_mask]
        pred_shape = pred_shape[frame_mask]
        pred_trans = pred_trans[frame_mask]
        pred_frame = torch.tensor(pred_frame)[frame_mask]
        pred_smpl_dict['pred_rotmat'] = pred_rotmat
        pred_smpl_dict['pred_shape'] = pred_shape
        pred_smpl_dict['pred_trans'] = pred_trans.unsqueeze(1)
        pred_smpl_dict['frame'] = pred_frame
        pred_frame = pred_frame.tolist()
        np.save(smpl_file.replace("tram", cfg.output_folder), pred_smpl_dict)

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
    pred_smpl_mesh = {x: [] for x in range(cfg.num_frames)}
    pred_joints_all = []

    for pred_id, pred_smpl, pred_frame, pred_contact in zip(pred_ids, pred_smpls, pred_frames, pred_contacts):
        for i in range(cfg.num_frames):
            current_idx = i + cfg.start_frame
            if current_idx in pred_frame:
                frame = pred_frame.index(current_idx)
                pred_vertices = pred_smpl.vertices[frame]
                pred_joints = pred_smpl.joints[frame, :24]

                pred_smpl_mesh[i].append([pred_id, trimesh.Trimesh(vertices=pred_vertices[:, :3], faces=smpl.faces), pred_contact[frame]])
                pred_joints_all.append(pred_joints[:, :3])

    w_jpe = 0.
    wa_jpe = 0.

    frame_result = []
    for i in range(cfg.num_frames):
        per_frame_result = OptimizedFrameResult(frame_idx=i, pred_cam=pred_cam[i])

        per_frame_result["pred_smpl"] = pred_smpl_mesh[i]
        per_frame_result["pred_cam"] = pred_cam[i]

        if i in cfg.img_idx:
            idx = cfg.img_idx.index(i)
            per_frame_result["rgb_hw3"] = rgb_hw3_list[idx]
            per_frame_result["depth_hw"] = depth_hw_list[idx]
            per_frame_result["conf_hw"] = conf_hw_list[idx]
            per_frame_result["mask_hw"] = masks_list[idx]
        frame_result.append(per_frame_result)

    optimised_result = OptimizedResult(
        point_cloud=point_cloud,
        mesh=mesh,
        intrinsics=K_b33[0],  # intrinsics is fixed
        img_size=(rgb_hw3_list[0].shape[1], rgb_hw3_list[0].shape[0]),
        frame_result=frame_result,
        eval_metrics={
            "w_jpe": w_jpe,
            "wa_jpe": wa_jpe
        })

    return optimised_result


def find_closest_valid(mask, vv, uu):
    """Find closest point in mask with value True to (vv, uu)."""
    points = torch.argwhere(mask)  # (N, 2) in (row, col) order
    if len(points) == 0:
        return None
    distances = (points[:, 0].float() - vv.float())**2 + (points[:, 1].float() - uu.float())**2
    min_idx = torch.argmin(distances)
    return points[min_idx][1].item(), points[min_idx][0].item()  # (px_x, px_y)


def joint_opt(imgs,
              pairs_in,
              mask_pairs_in,
              cache_path,
              model,
              pred_points,
              pred_conf,
              init_K,
              old_focal,
              cfg: JOSHConfig,
              subsample=8,
              desc_conf='desc_conf',
              device='cuda',
              dtype=torch.float32,
              shared_intrinsics=False,
              **kw):
    """ Sparse alignment with MASt3R
        imgs: list of image paths
        cache_path: path where to dump temporary files (str)

        lr1, niter1: learning rate and #iterations for coarse global alignment (3D matching)
        lr2, niter2: learning rate and #iterations for refinement (2D reproj error)

        lora_depth: smart dimensionality reduction with depthmaps
    """
    # Convert pair naming convention from dust3r to mast3r
    pairs_in = convert_dust3r_pairs_naming(imgs, pairs_in)
    mask_pairs_in = convert_dust3r_pairs_naming(imgs, mask_pairs_in)
    # forward pass
    pairs, cache_path = forward_mast3r_w_mask(pairs_in, mask_pairs_in, model, cache_path=cache_path, subsample=subsample, desc_conf=desc_conf, device=device)

    # extract canonical pointmaps
    tmp_pairs, pairwise_scores, canonical_views, canonical_paths, preds_21 = \
        prepare_canonical_data(imgs, pairs, subsample, pred_points, pred_conf, cache_path=cache_path, mode='avg-angle', device=device)

    # compute minimal spanning tree
    mst = compute_min_spanning_tree(pairwise_scores)

    # smartly combine all usefull data
    imsizes, pps, base_focals, core_depth, anchors, corres, corres2d, preds_21 = \
        condense_data(imgs, tmp_pairs, canonical_views, preds_21, dtype)

    confs = torch.stack([torch.load(pth)[0][2].mean() for pth in canonical_paths]).to(pps)
    weighting = confs / confs.sum()

    if cfg.init_focal is not None:
        new_focal = cfg.init_focal
    else:
        new_focal = (weighting @ base_focals).item()
    init_K[0][0] = new_focal
    init_K[1][1] = new_focal

    smpl_scale_infos = []
    smpl_static_infos = []
    all_smpl_img_idx = []
    all_smpl_body_pose = []
    all_smpl_quat = []
    all_smpl_trans = []
    all_smpl_body_shape = []

    canon_confs = []
    canon_depths = []
    canon_masks = []
    canon_core_depths = []
    for canon_path in canonical_paths:
        (canon, canon_depth, conf), _ = torch.load(canon_path, map_location=device)
        canon_confs.append(conf)
        canon_depths.append(canon_depth)
        canon_masks.append(conf > cfg.conf_thres)
        H1, W1 = conf.shape
        yx = np.mgrid[subsample // 2:H1:subsample, subsample // 2:W1:subsample]
        H2, W2 = yx.shape[1:]
        cy, cx = yx.reshape(2, -1)
        depth = canon_depth[cy, cx]
        assert (depth > 0).all()
        canon_core_depths.append(depth)

    smpl_files = sorted(glob(os.path.join(cfg.input_folder, "tram", "*.npy")))
    for smpl_file in smpl_files:
        pred_smpl = np.load(smpl_file, allow_pickle=True).item()

        pred_rotmat = pred_smpl['pred_rotmat']
        pred_shape = pred_smpl['pred_shape']

        pred_trans = pred_smpl['pred_trans'].squeeze(1)
        pred_trans[:, 2] = pred_trans[:, 2] * new_focal / old_focal
        pred_frame = pred_smpl['frame'].tolist()

        smpl = SMPL(model_path="data/smpl")
        tt = lambda x: torch.Tensor(x).float()
        smpl_scale_info = []
        smpl_static_info = []
        idxs = [pred_frame.index(i + cfg.start_frame) for i in range(cfg.num_frames) if i + cfg.start_frame in pred_frame and i in cfg.img_idx]

        pred_smpl = smpl(body_pose=pred_rotmat[:, 1:],
                         global_orient=pred_rotmat[:, [0]],
                         betas=pred_shape,
                         transl=pred_trans,
                         pose2rot=False,
                         default_smpl=True)
        joints_all = vertices2joints(smpl.J_regressor, pred_smpl.vertices)
        contact = torch.tensor(np.load(smpl_file.replace('tram', 'deco')))

        assert len(canonical_paths) == len(cfg.img_idx)
        smpl_idx = 0
        smpl_img_idx = []
        last_contact_vertex_ids = []
        for i in range(cfg.num_frames):
            current_idx = i + cfg.start_frame
            if current_idx in pred_frame and i in cfg.img_idx:
                frame = pred_frame.index(current_idx)
                image_frame = cfg.img_idx.index(i)
                smpl_img_idx.append(image_frame)

                # change frame indexing to smpl_idx
                curr_vertices = pred_smpl.vertices[frame].cpu()
                valid_vertex_ids = torch.where(contact[frame] == 1)[0]
                points = curr_vertices[valid_vertex_ids]
                joints = []
                joint_idx = []
                vertex_ids = []
                for ji, joint in enumerate(joints_all[frame]):
                    dist, idx = torch.min(torch.linalg.norm(joint - points, dim=-1), dim=-1)
                    if dist < 0.05:
                        joints.append(points[idx])
                        joint_idx.append(ji)
                        vertex_ids.append(valid_vertex_ids[idx].item())

                if len(joints) == 0:
                    continue
                points = torch.stack(joints, axis=0)

                points_2d = torch.einsum("ij,nj->ni", init_K.cpu(), points)
                points_2d = points_2d / points_2d[:, [2]]
                points_2d = points_2d[:, :2]

                canon_depth = canon_depths[image_frame]
                conf = canon_confs[image_frame].cpu().numpy()
                mask = conf > cfg.conf_thres
                H1, W1 = conf.shape
                yx = np.mgrid[subsample // 2:H1:subsample, subsample // 2:W1:subsample]
                H2, W2 = yx.shape[1:]
                cy, cx = yx.reshape(2, -1)
                depth = canon_depth[cy, cx]
                assert (depth > 0).all()
                current_contact_vertex_ids = []
                for point_idx in range(points.shape[0]):
                    current_contact_vertex_ids.append(vertex_ids[point_idx])
                    if vertex_ids[point_idx] in last_contact_vertex_ids:
                        smpl_static_info.append(torch.tensor([smpl_idx, vertex_ids[point_idx]]))

                    # ensure the joints are projected on the image:
                    pred_points_2d = points_2d[point_idx]
                    if pred_points_2d[0] >= 0 and pred_points_2d[0] < W1 and pred_points_2d[1] >= 0 and pred_points_2d[1] < H1 and mask.sum() > 0 and mask[
                            pred_points_2d[1].long(), pred_points_2d[0].long()] == False:
                        pred_points_3d = points[point_idx]
                        valid_points_2d = find_closest_valid(torch.tensor(mask, device=pred_points_2d.device), pred_points_2d[1], pred_points_2d[0])
                        px, py = valid_points_2d

                        core_idx = (py // subsample) * W2 + (px // subsample)

                        # compute relative depth offsets w.r.t. anchors
                        ref_z = depth[core_idx]
                        pts_z = canon_depth[py, px]
                        human_z = canon_depth[pred_points_2d[1].long(), pred_points_2d[0].long()]
                        if max(human_z / pts_z, pts_z / human_z) > cfg.depth_filter_ratio:
                            continue
                        conf_z = conf[py, px]
                        offset = pts_z / ref_z
                        if cfg.update_correspondences:
                            smpl_scale_info.append({
                                'image_frame': image_frame,
                                'smpl_idx': smpl_idx,
                                'vertex_id': vertex_ids[point_idx],
                            })
                        else:
                            smpl_scale_info.append(torch.tensor([image_frame, core_idx, offset, conf_z, smpl_idx, vertex_ids[point_idx]]))
                last_contact_vertex_ids = current_contact_vertex_ids
                smpl_idx += 1
        smpl_body_pose = pred_rotmat[idxs, 1:].clone().to(device=device)
        smpl_quat = matrix_to_quaternion(pred_rotmat[idxs, 0:1].clone().to(device=device))
        smpl_body_shape = pred_shape[idxs].clone().to(device=device)
        smpl_trans = pred_trans[idxs].clone().to(device=device)
        smpl_scale_infos.append(smpl_scale_info)
        smpl_static_infos.append(smpl_static_info)
        all_smpl_img_idx.append(smpl_img_idx)
        all_smpl_body_pose.append(smpl_body_pose)
        all_smpl_quat.append(smpl_quat)
        all_smpl_trans.append(smpl_trans)
        all_smpl_body_shape.append(smpl_body_shape)

    imgs, res_coarse, res_fine, smpl_transforms = sparse_scene_optimizer(imgs,
                                                                         subsample,
                                                                         imsizes,
                                                                         pps,
                                                                         base_focals,
                                                                         core_depth,
                                                                         anchors,
                                                                         corres,
                                                                         corres2d,
                                                                         preds_21,
                                                                         canonical_paths,
                                                                         mst,
                                                                         smpl_scale_infos,
                                                                         smpl_static_infos,
                                                                         canon_confs=canon_confs,
                                                                         canon_depths=canon_depths,
                                                                         canon_masks=canon_masks,
                                                                         canon_core_depths=canon_core_depths,
                                                                         conf_thres=cfg.conf_thres,
                                                                         smpl_img_idx=all_smpl_img_idx,
                                                                         smpl_body_pose=all_smpl_body_pose,
                                                                         smpl_quat=all_smpl_quat,
                                                                         smpl_body_shape=all_smpl_body_shape,
                                                                         smpl_trans=all_smpl_trans,
                                                                         shared_intrinsics=shared_intrinsics,
                                                                         cache_path=cache_path,
                                                                         device=device,
                                                                         dtype=dtype,
                                                                         init_K=init_K,
                                                                         loss_s=gamma_loss(1.1),
                                                                         loss_scale_w=cfg.scale_loss_weight,
                                                                         loss_prior_w=cfg.prior_loss_weight,
                                                                         loss_smooth_w=cfg.smooth_loss_weight,
                                                                         loss_static_w=cfg.static_loss_weight,
                                                                         optimize_smpl=cfg.optimize_smpl,
                                                                         opt_pp=cfg.optimize_focal,
                                                                         opt_depth=cfg.optimize_depth,
                                                                         opt_corres=cfg.update_correspondences,
                                                                         **kw)

    return SparseGA(imgs, pairs_in, res_fine or res_coarse, anchors, canonical_paths, smpl_transforms), all_smpl_img_idx
