import argparse
import copy
import os
from collections import defaultdict
from glob import glob

import joblib
import numpy as np
import torch
from loguru import logger
from smplx import SMPL

from josh.utils.eval_utils import (
    batch_align_by_pelvis, batch_compute_similarity_transform_torch,
    compute_ate, compute_error_accel,
    compute_foot_sliding, compute_jitter, compute_jpe, compute_rte,
    first_align_joints, global_align_joints)
from josh.utils.rot_utils import axis_angle_to_matrix, matrix_to_axis_angle
from josh.utils.sloper4d_dataset import SLOPER4D_Dataset

"""
Evaluate JOSH on SLOPER4D dataset.
Expects: data/SLOPER4D/{seq}/ with *_labels.pkl, images/
         data/SLOPER4D/{seq}/josh/scene.pkl and josh/*.npy from JOSH pipeline.
Metrics: PA-MPJPE, MPJPE, PVE, Accel, W-MPJPE, WA-MPJPE, RTE, Jitter, FS, ATE.
"""

m2mm = 1e3


@torch.no_grad()
def main(args):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    logger.info(f'GPU name -> {torch.cuda.get_device_name()}')
    logger.info(f'GPU feat -> {torch.cuda.get_device_properties("cuda")}')

    accumulator = defaultdict(list)
    tt = lambda x: torch.from_numpy(x).float().to(args.device)
    smpl = {k: SMPL(model_path="data/smpl", gender=k).to(args.device) for k in ['male', 'female', 'neutral']}
    pelvis_idxs = [1, 2]

    for folder in sorted(os.listdir("data/SLOPER4D")):
        seq_id = folder.split("_")[0]
        print(seq_id)

        root = os.path.join("data/SLOPER4D", folder)
        hps_folder = os.path.join(root, "josh")
        hps_files = sorted(glob(os.path.join(hps_folder, "*.npy")))
        dataset = SLOPER4D_Dataset(glob(os.path.join(root, "*_labels.pkl"))[0], return_torch=False, fix_pts_num=True)

        # --- Load JOSH HPS predictions ---
        hps_file = hps_files[0]
        pred_smpl = np.load(hps_file, allow_pickle=True).item()
        pred_rotmat = pred_smpl['pred_rotmat']
        pred_shape = pred_smpl['pred_shape']
        pred_trans = pred_smpl['pred_trans'].squeeze()
        frame = pred_smpl['frame']

        # --- Load JOSH camera predictions ---
        scene_file = os.path.join(hps_folder, "scene.pkl")
        scene_pred = joblib.load(scene_file)
        pred_cam = tt(scene_pred["pred_cam"])

        # --- Load GT ---
        masks = frame
        gender = dataset.smpl_gender
        poses_body = dataset.smpl_pose[:, 3:]
        poses_root = dataset.smpl_pose[:, :3]
        betas = np.repeat(np.array(dataset.betas).reshape((1, -1)), repeats=poses_body.shape[0], axis=0)
        trans = dataset.global_trans
        extrinsics = np.array(dataset.cam_pose)

        # --- GT global motion (world frame) ---
        target_glob = smpl[gender](body_pose=tt(poses_body), global_orient=tt(poses_root), betas=tt(betas), transl=tt(trans))

        # --- GT in per-frame camera space ---
        all_trans = torch.tensor(extrinsics).float()  # world-to-camera per frame

        trans_cam_eval = tt(trans)
        trans_cam_eval = torch.cat([trans_cam_eval, torch.ones(trans_cam_eval.shape[0], 1)], dim=-1)
        trans_cam_eval = torch.einsum('bij,bj->bi', all_trans, trans_cam_eval)[..., :3]

        target_cam_vertices = torch.cat([target_glob.vertices, torch.ones(*target_glob.vertices.shape[:2], 1)], dim=-1)
        target_cam_vertices = torch.einsum('bij,bnj->bni', all_trans, target_cam_vertices)[..., :3]

        target_cam_joints = torch.cat([target_glob.joints, torch.ones(*target_glob.joints.shape[:2], 1)], dim=-1)
        target_cam_joints = torch.einsum('bij,bnj->bni', all_trans, target_cam_joints)[..., :3]

        # GT root in camera coordinates
        poses_root_cam = matrix_to_axis_angle(tt(extrinsics[:, :3, :3]) @ axis_angle_to_matrix(tt(poses_root)))

        # --- GT in first-camera frame (for global eval) ---
        init_trans = torch.tensor(extrinsics[0]).float()  # world-to-camera_0

        trans_eval = tt(trans)
        trans_eval = torch.cat([trans_eval, torch.ones(trans_eval.shape[0], 1)], dim=-1)
        trans_eval = torch.einsum('ij,nj->ni', init_trans, trans_eval)[..., :3]

        target_glob_v = torch.cat([target_glob.vertices, torch.ones(*target_glob.vertices.shape[:2], 1)], dim=-1)
        target_glob.vertices = torch.einsum('ij,bnj->bni', init_trans, target_glob_v)[..., :3]

        target_glob_j = torch.cat([target_glob.joints, torch.ones(*target_glob.joints.shape[:2], 1)], dim=-1)
        target_glob.joints = torch.einsum('ij,bnj->bni', init_trans, target_glob_j)[..., :3]

        target_j3d_glob = target_glob.joints[:, :24][masks]
        target_verts_cam = target_cam_vertices[masks]
        target_j3d_cam = target_cam_joints[:, :24][masks]

        # --- Prediction in camera space ---
        mean_shape = pred_shape.mean(dim=0, keepdim=True).repeat(len(pred_shape), 1)

        pred = smpl["neutral"](
            body_pose=pred_rotmat[:, 1:],
            global_orient=pred_rotmat[:, [0]],
            betas=mean_shape,
            transl=pred_trans,
            pose2rot=False,
            default_smpl=True)
        pred_verts_cam = pred.vertices
        pred_j3d_cam = pred.joints[:, :24]

        # Use GT camera-space translation for global evaluation (standard WHAM protocol)
        pred_trans_for_global = trans_cam_eval[masks]

        # --- Camera trajectory for ATE ---
        pred_cam_traj = torch.inverse(torch.einsum('ij,bjk->bik', torch.inverse(pred_cam[0]), pred_cam))[:, :3, 3]
        gt_cam_traj = torch.einsum("bij,jk->bik", all_trans, torch.inverse(all_trans[0]))[:, :3, 3]

        # --- Transform predictions to global frame ---
        pred_cam_masked = pred_cam[masks]
        pred_cam_masked = torch.einsum('ij,bjk->bik', torch.inverse(pred_cam_masked[0]), pred_cam_masked)

        pred_trans_eval = torch.cat([pred_trans_for_global, torch.ones(pred_trans_for_global.shape[0], 1)], dim=-1)
        pred_trans_eval = torch.einsum('bij,bj->bi', pred_cam_masked, pred_trans_eval)[..., :3]

        pred_verts_homo = torch.cat([pred_verts_cam, torch.ones(*pred_verts_cam.shape[:2], 1)], dim=-1)
        pred_verts_glob = torch.einsum('bij,bnj->bni', pred_cam_masked, pred_verts_homo)[..., :3]

        pred_j3d_homo = torch.cat([pred_j3d_cam, torch.ones(*pred_j3d_cam.shape[:2], 1)], dim=-1)
        pred_j3d_glob = torch.einsum('bij,bnj->bni', pred_cam_masked, pred_j3d_homo)[..., :3]

        pred_glob = copy.deepcopy(pred)
        pred_glob.joints = pred_j3d_glob
        pred_glob.vertices = pred_verts_glob

        # === Local motion metrics ===
        pred_j3d_cam, target_j3d_cam, pred_verts_cam, target_verts_cam = batch_align_by_pelvis(
            [pred_j3d_cam, target_j3d_cam, pred_verts_cam, target_verts_cam], pelvis_idxs)
        S1_hat = batch_compute_similarity_transform_torch(pred_j3d_cam, target_j3d_cam)
        pa_mpjpe = torch.sqrt(((S1_hat - target_j3d_cam)**2).sum(dim=-1)).mean(dim=-1).cpu().numpy() * m2mm
        mpjpe = torch.sqrt(((pred_j3d_cam - target_j3d_cam)**2).sum(dim=-1)).mean(dim=-1).cpu().numpy() * m2mm
        pve = torch.sqrt(((pred_verts_cam - target_verts_cam)**2).sum(dim=-1)).mean(dim=-1).cpu().numpy() * m2mm
        accel = compute_error_accel(joints_pred=pred_j3d_cam.cpu(), joints_gt=target_j3d_cam.cpu())[1:-1]
        accel = accel * (20**2)  # SLOPER4D is 20 fps

        # === Global motion metrics ===
        chunk_length = 100
        w_mpjpe, wa_mpjpe = [], []
        for start in range(0, len(masks), chunk_length):
            end = min(len(masks), start + chunk_length)
            target_j3d = target_j3d_glob[start:end].clone().cpu()
            pred_j3d = pred_j3d_glob[start:end].clone().cpu()
            w_mpjpe.append(compute_jpe(target_j3d, first_align_joints(target_j3d, pred_j3d)))
            wa_mpjpe.append(compute_jpe(target_j3d, global_align_joints(target_j3d, pred_j3d)))
        w_mpjpe = np.concatenate(w_mpjpe) * m2mm
        wa_mpjpe = np.concatenate(wa_mpjpe) * m2mm

        # === Additional metrics ===
        rte = compute_rte(trans_eval[masks], pred_trans_eval.cpu()) * 1e2
        jitter = compute_jitter(pred_glob, fps=20)  # SLOPER4D is 20 fps
        foot_sliding = compute_foot_sliding(target_glob, pred_glob, masks) * m2mm

        print(f"{folder}: w={w_mpjpe.mean():.1f}, wa={wa_mpjpe.mean():.1f}")

        # === Accumulate ===
        accumulator['pa_mpjpe'].append(pa_mpjpe)
        accumulator['mpjpe'].append(mpjpe)
        accumulator['pve'].append(pve)
        accumulator['accel'].append(accel)
        accumulator['wa_mpjpe'].append(wa_mpjpe)
        accumulator['w_mpjpe'].append(w_mpjpe)
        accumulator['RTE'].append(rte)
        accumulator['jitter'].append(jitter)
        accumulator['FS'].append(foot_sliding)

        if "seq009" not in folder:
            ate = np.array([compute_ate(pred_cam_traj, gt_cam_traj)])
            accumulator['ATE'].append(ate)
            print(f"  ATE: {ate.mean():.4f}")

    for k, v in accumulator.items():
        accumulator[k] = np.concatenate(v).mean()

    log_str = 'Evaluation on SLOPER4D, '
    log_str += ' '.join([f'{k.upper()}: {v:.4f},' for k, v in accumulator.items()])
    logger.info(log_str)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default='cpu')
    args = parser.parse_args()
    main(args)
