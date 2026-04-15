import argparse
import copy
import os
import os.path as osp
import pickle
from collections import defaultdict
from glob import glob

import joblib
import numpy as np
import torch
from loguru import logger
from smplx import SMPL

from josh.utils.eval_utils import (
    compute_error_accel, compute_foot_sliding, compute_jitter,
    compute_jpe, compute_rte, first_align_joints,
    global_align_joints)

"""
Evaluate JOSH on RICH dataset.
Expects: data/RICH/{seq}/ with per-frame GT SMPL pkl, images/
         data/RICH/{seq}/josh/scene.pkl and josh/*.npy from JOSH pipeline.
Metrics: W-MPJPE, WA-MPJPE, RTE, Accel, Jitter, FS.
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

    for folder_ in sorted(os.listdir("data/RICH")):
        folder = os.path.join("data/RICH", folder_)
        seq_id = "".join(folder.split("/")[-1].split("_"))
        root = os.path.dirname(os.path.join(folder, "images"))
        hps_folder = os.path.join(root, "josh")
        hps_files = sorted(glob(os.path.join(hps_folder, "*.npy")))

        # --- Load GT SMPL per frame ---
        gt_frame = []
        poses_body, poses_root, betas_list, trans_list = [], [], [], []
        for idx, image_name in enumerate(sorted(os.listdir(os.path.join(folder, "images")))):
            image_id = image_name.split("_")[0]
            if os.path.exists(os.path.join(folder, image_id)):
                gt_frame.append(idx)
                gt_file = glob(osp.join(folder, image_id, '*_smpl.pkl'))[0]
                with open(gt_file, "rb") as f:
                    gt_smpl = pickle.load(f)
                    trans_list.append(tt(gt_smpl["transl"]))
                    betas_list.append(tt(gt_smpl['betas']))
                    poses_body.append(tt(gt_smpl['body_pose']))
                    poses_root.append(tt(gt_smpl["global_orient"]))

        poses_body = torch.cat(poses_body).reshape(len(gt_frame), -1)
        betas = torch.cat(betas_list)
        poses_root = torch.cat(poses_root).reshape(len(gt_frame), -1)
        trans = torch.cat(trans_list)

        # --- Load JOSH HPS predictions ---
        hps_file = hps_files[0]
        pred_smpl = np.load(hps_file, allow_pickle=True).item()
        pred_rotmat = pred_smpl['pred_rotmat']
        pred_shape = pred_smpl['pred_shape']
        pred_trans = pred_smpl['pred_trans'].squeeze()
        frame = pred_smpl['frame']
        if isinstance(frame, torch.Tensor):
            frame = frame.cpu().numpy().astype(int).tolist()

        # --- Load JOSH camera predictions ---
        scene_file = os.path.join(hps_folder, "scene.pkl")
        scene_pred = joblib.load(scene_file)
        pred_cam = tt(scene_pred["pred_cam"])

        # --- Align GT and prediction frames ---
        mask1 = [i for i, item in enumerate(frame) if item in gt_frame]
        mask2 = [i for i, item in enumerate(gt_frame) if item in frame]
        assert len(mask1) == len(mask2)
        mask1 = torch.tensor(np.array(mask1)).long()
        mask2 = torch.tensor(np.array(mask2)).long()

        pred_rotmat = pred_rotmat[mask1]
        pred_shape = pred_shape[mask1]
        pred_trans = pred_trans[mask1]

        # --- GT global motion ---
        target_glob = smpl["neutral"](body_pose=poses_body, global_orient=poses_root, betas=betas, transl=trans)
        trans_eval = trans[mask2]
        target_j3d_glob = target_glob.joints[:, :24][mask2]
        target_verts_glob = target_glob.vertices[mask2]

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

        # --- Transform predictions to global using JOSH cameras ---
        pred_cam_sel = pred_cam[frame][mask1]
        pred_cam_sel = torch.einsum('ij,bjk->bik', torch.inverse(pred_cam_sel[0]), pred_cam_sel)

        pred_trans_eval = torch.cat([pred_trans, torch.ones(pred_trans.shape[0], 1)], dim=-1)
        pred_trans_eval = torch.einsum('bij,bj->bi', pred_cam_sel, pred_trans_eval)[..., :3]

        pred_verts_homo = torch.cat([pred_verts_cam, torch.ones(*pred_verts_cam.shape[:2], 1)], dim=-1)
        pred_verts_glob = torch.einsum('bij,bnj->bni', pred_cam_sel, pred_verts_homo)[..., :3]

        pred_j3d_homo = torch.cat([pred_j3d_cam, torch.ones(*pred_j3d_cam.shape[:2], 1)], dim=-1)
        pred_j3d_glob = torch.einsum('bij,bnj->bni', pred_cam_sel, pred_j3d_homo)[..., :3]

        pred_glob = copy.deepcopy(pred)
        pred_glob.joints = pred_j3d_glob
        pred_glob.vertices = pred_verts_glob

        # === Global motion metrics (chunked alignment) ===
        chunk_length = 100
        w_mpjpe, wa_mpjpe = [], []
        for start in range(0, len(mask1), chunk_length):
            end = min(len(mask1), start + chunk_length)
            target_j3d = target_j3d_glob[start:end].clone().cpu()
            pred_j3d = pred_j3d_glob[start:end].clone().cpu()
            w_mpjpe.append(compute_jpe(target_j3d, first_align_joints(target_j3d, pred_j3d)))
            wa_mpjpe.append(compute_jpe(target_j3d, global_align_joints(target_j3d, pred_j3d)))
        w_mpjpe = np.concatenate(w_mpjpe) * m2mm
        wa_mpjpe = np.concatenate(wa_mpjpe) * m2mm

        # === Additional metrics ===
        rte = compute_rte(trans_eval, pred_trans_eval.cpu()) * 1e2
        jitter = compute_jitter(pred_glob, fps=30)
        accel = compute_error_accel(joints_pred=pred_j3d_glob.cpu(), joints_gt=target_j3d_glob.cpu())[1:-1]
        accel = accel * (30**2)
        foot_sliding = compute_foot_sliding(target_glob, pred_glob, mask2) * m2mm

        print(f"{folder}: w_mpjpe={w_mpjpe.mean():.2f}, wa_mpjpe={wa_mpjpe.mean():.2f}, RTE={rte.mean():.4f}")

        # === Accumulate ===
        accumulator['wa_mpjpe'].append(wa_mpjpe)
        accumulator['w_mpjpe'].append(w_mpjpe)
        accumulator['RTE'].append(rte)
        accumulator['jitter'].append(jitter)
        accumulator['accel'].append(accel)
        accumulator['FS'].append(foot_sliding)

    for k, v in accumulator.items():
        accumulator[k] = np.concatenate(v).mean()

    log_str = 'Evaluation on RICH, '
    log_str += ' '.join([f'{k.upper()}: {v:.4f},' for k, v in accumulator.items()])
    logger.info(log_str)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default='cpu')
    args = parser.parse_args()
    main(args)
