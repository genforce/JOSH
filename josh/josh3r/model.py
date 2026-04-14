import sys
from typing import Dict

sys.path.append("./third_party/mast3r")
import torch
import torch.nn as nn
import torch.nn.functional as F
from mast3r.model import AsymmetricMASt3R

from josh.josh3r.blocks import (DecoderBlock, MLPHead, ROIFeatureRegressor, RoPE2D)
from josh.utils.rot_utils import axis_angle_to_matrix, rotation_6d_to_matrix
from smplx import SMPLLayer


class JOSH3R(nn.Module):

    def __init__(self):
        super().__init__()
        self.mast3r = AsymmetricMASt3R.from_pretrained("naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric")
        self.mast3r_dim = 768
        self.human_feat_extractor_1 = ROIFeatureRegressor(self.mast3r_dim, self.mast3r_dim)
        self.human_feat_extractor_2 = ROIFeatureRegressor(self.mast3r_dim, self.mast3r_dim)
        self.human_head_1 = MLPHead(self.mast3r_dim, 3)
        self.human_head_2 = MLPHead(self.mast3r_dim, 3)
        self.rel_trans_head = MLPHead(4 * self.mast3r_dim, 3, num_channels=self.mast3r_dim, zero_init=True)
        self.rel_rot_head = MLPHead(4 * self.mast3r_dim, 6, num_channels=self.mast3r_dim)
        self.scene_head_1 = MLPHead(self.mast3r_dim, 1, zero_init=True)
        self.scene_head_2 = MLPHead(self.mast3r_dim, 1, zero_init=True)

        self.smpl = SMPLLayer(model_path="data/smpl", gender='neutral')
        for param in self.smpl.parameters():
            param.requires_grad = False
        self.smpl.eval()
        self.criterion = F.mse_loss
        self.decoder_block_1 = DecoderBlock(dim=768, num_heads=12, mlp_ratio=4, qkv_bias=True, rope=RoPE2D())
        self.decoder_block_2 = DecoderBlock(dim=768, num_heads=12, mlp_ratio=4, qkv_bias=True, rope=RoPE2D())
        self.dec_norm = nn.LayerNorm(self.mast3r_dim)

    def forward(self, batch: Dict):
        b, _, c, h, w = batch["imgs"].shape
        shape1 = batch["img_shapes"][:, 1]
        shape2 = batch["img_shapes"][:, 1]

        feat1, feat2, pos1, pos2 = self.mast3r._encode_image_pairs(batch["imgs"][:, 0], batch["imgs"][:, 1], shape1, shape2)
        dec1, dec2 = self.mast3r._decoder(feat1, pos1, feat2, pos2)

        res1 = self.mast3r._downstream_head(1, [tok.float() for tok in dec1], shape1)
        res2 = self.mast3r._downstream_head(2, [tok.float() for tok in dec2], shape2)

        X11, C11 = res1['pts3d'], res1['conf']
        X21, C21 = res2['pts3d'], res2['conf']

        X11 = X11.reshape(b, -1, 3)
        X21 = X21.reshape(b, -1, 3)

        dec1 = dec1[-1]
        dec2 = dec2[-1]

        assert dec1.shape[-1] == self.mast3r_dim
        dec1_2d = dec1.permute(0, 2, 1).reshape(b, -1, h // 16, w // 16)
        dec2_2d = dec2.permute(0, 2, 1).reshape(b, -1, h // 16, w // 16)

        spatial_scale = torch.tensor([w, h, w, h], dtype=torch.float32).to(dec1.device)

        # Normalize bbox coordinates to the range [-1, 1] relative to the feature map size
        bboxes_1 = (batch["bbox"][:, 0] / spatial_scale) * 2 - 1
        bboxes_2 = (batch["bbox"][:, 1] / spatial_scale) * 2 - 1

        # Grid for sampling

        box_feat_1 = self.sample_roi(bboxes_1, dec1_2d)
        box_feat_2 = self.sample_roi(bboxes_2, dec2_2d)

        human_feat_1 = self.human_feat_extractor_1(box_feat_1)  # B, C
        human_feat_2 = self.human_feat_extractor_2(box_feat_2)  # B, C

        dec1 = torch.cat([dec1, human_feat_1.unsqueeze(1)], dim=1)  # B, N+1, C
        dec2 = torch.cat([dec2, human_feat_2.unsqueeze(1)], dim=1)  # B, N+1, C

        pos1 = torch.cat([pos1, -torch.ones(b, 1, 2, dtype=torch.long, device=pos1.device)], dim=1)  # B, N+1, 2
        pos2 = torch.cat([pos2, -torch.ones(b, 1, 2, dtype=torch.long, device=pos2.device)], dim=1)  # B, N+1, 2

        dec1_new = self.dec_norm(self.decoder_block_1(dec1, dec2, pos1, pos2)[0])
        # img2 side
        dec2_new = self.dec_norm(self.decoder_block_2(dec2, dec1, pos2, pos1)[0])

        scene_token_1 = torch.amax(dec1_new[:, :-1], dim=1)
        scene_token_2 = torch.amax(dec2_new[:, :-1], dim=1)
        human_token_1 = dec1_new[:, -1]
        human_token_2 = dec2_new[:, -1]

        trans1 = self.human_head_1(human_token_1)
        trans2 = self.human_head_2(human_token_2)
        scale1 = self.scene_head_1(scene_token_1)
        scale2 = self.scene_head_2(scene_token_2)

        all_tokens = torch.cat([human_token_1, human_token_2, scene_token_1, scene_token_2], dim=-1)

        rel_trans = self.rel_trans_head(all_tokens)
        rel_rot = self.rel_rot_head(all_tokens)

        return rel_trans, rel_rot, trans1, trans2, X11, X21, torch.exp(scale1), torch.exp(scale2)

    def sample_roi(self, bboxes, feature_map, output_size=(7, 7)):
        B = feature_map.shape[0]
        grid = []
        for i in range(B):
            x1, y1, x2, y2 = bboxes[i]
            grid_x = torch.linspace(x1, x2, output_size[0], device=feature_map.device)
            grid_y = torch.linspace(y1, y2, output_size[1], device=feature_map.device)
            grid_y, grid_x = torch.meshgrid(grid_y, grid_x)
            grid_xy = torch.stack([grid_x, grid_y], dim=-1)
            grid.append(grid_xy)

        # Convert grid to tensor of shape (B, output_h, output_w, 2)
        grid = torch.stack(grid, dim=0)
        # Perform bilinear sampling using grid_sample
        roi_feat = F.grid_sample(feature_map, grid, align_corners=False, padding_mode='border')

        return roi_feat

    def get_loss(self, batch: Dict, ret):
        b, _, c, h, w = batch["imgs"].shape
        rel_trans, rel_rot, trans1, trans2, X11, X21, scale1, scale2 = ret
        loss_rel_trans = torch.norm(batch["trans"] - rel_trans, dim=-1).mean()
        loss_rel_rot = self.criterion(batch["rot"], rel_rot)
        loss_trans1 = torch.norm(batch["trans1"] - trans1, dim=-1) / (torch.norm(batch["trans1"], dim=-1) + 1e-3)
        loss_trans1 = loss_trans1.mean()
        loss_trans2 = torch.norm(batch["trans2"] - trans2, dim=-1) / (torch.norm(batch["trans2"], dim=-1) + 1e-3)
        loss_trans2 = loss_trans2.mean()

        # contact scale loss
        smpl1 = self.smpl(transl=trans1,
                          betas=batch["beta"],
                          global_orient=rotation_6d_to_matrix(batch["rot1"]),
                          body_pose=axis_angle_to_matrix(batch["pose"][:, [0]]))
        smpl2 = self.smpl(transl=trans2,
                          betas=batch["beta"],
                          global_orient=rotation_6d_to_matrix(batch["rot2"]),
                          body_pose=axis_angle_to_matrix(batch["pose"][:, [1]]))
        smpl_foot1 = smpl1.joints[:, [10, 11, 7, 8]]  # B, 4, 3
        smpl_foot2 = smpl2.joints[:, [10, 11, 7, 8]]  # B, 4, 3

        X11_detach = X11.clone().detach() * scale1.unsqueeze(-1)
        X21_detach = X21.clone().detach() * scale2.unsqueeze(-1)

        loss_contact_scale_1 = torch.min(torch.norm(smpl_foot1.unsqueeze(2) - X11_detach.unsqueeze(1), dim=-1), dim=-1)[0].mean()
        loss_contact_scale_2 = torch.min(torch.norm(smpl_foot2.unsqueeze(2) - X21_detach.unsqueeze(1), dim=-1), dim=-1)[0].mean()

        loss_contact_scale = loss_contact_scale_1 + loss_contact_scale_2

        # contact static loss
        smpllocal1 = self.smpl(transl=torch.zeros([b, 3]).cuda(),
                               betas=batch["beta"],
                               global_orient=torch.eye(3).unsqueeze(0).repeat(b, 1, 1).cuda(),
                               body_pose=axis_angle_to_matrix(batch["pose"][:, [0]]))
        smpllocal2 = self.smpl(transl=rel_trans,
                               betas=batch["beta"],
                               global_orient=rotation_6d_to_matrix(rel_rot),
                               body_pose=axis_angle_to_matrix(batch["pose"][:, [1]]))
        smpl_foot_right_1 = smpllocal1.joints[:, [11, 8]]  # B, 2, 3
        smpl_foot_right_2 = smpllocal2.joints[:, [11, 8]]  # B, 2, 3
        smpl_foot_left_1 = smpllocal1.joints[:, [10, 7]]  # B, 2, 3
        smpl_foot_left_2 = smpllocal2.joints[:, [10, 7]]  # B, 2, 3

        loss_contact_static = torch.minimum(torch.norm(smpl_foot_right_1 - smpl_foot_right_2, dim=-1), torch.norm(smpl_foot_left_1 - smpl_foot_left_2,
                                                                                                                  dim=-1)).mean()

        if not torch.any(batch["depth_mask"]):
            loss_depth = 0
        else:
            gt_depth_1 = batch["depth"][:, 0].reshape(b, -1, 3)
            gt_depth_2 = batch["depth"][:, 1].reshape(b, -1, 3)
            loss_depth = torch.stack([
                torch.norm((X11 * scale1.unsqueeze(-1) - gt_depth_1), dim=-1) / (torch.norm(gt_depth_1, dim=-1) + 1e-3),
                torch.norm((X21 * scale2.unsqueeze(-1) - gt_depth_2), dim=-1) / (torch.norm(gt_depth_2, dim=-1) + 1e-3),
            ],
                                     dim=1)  # B, 2, N
            loss_depth = loss_depth[batch["depth_mask"]].mean()

        loss_dict = {
            "loss_rel_tran": loss_rel_trans,
            "loss_rel_rot": loss_rel_rot,
            "loss_trans1": loss_trans1,
            "loss_trans2": loss_trans2,
            "loss_contact_scale": loss_contact_scale,
            "loss_contact_static": loss_contact_static,
            "loss_depth": loss_depth,
            "loss": loss_rel_trans + loss_rel_rot + loss_trans1 + loss_trans2 + loss_contact_scale + loss_contact_static + loss_depth
        }
        return loss_dict
