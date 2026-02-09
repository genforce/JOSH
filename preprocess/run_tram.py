import json
import os
import sys

import cv2

sys.path.insert(0, './third_party/tram')

import argparse
from glob import glob

import numpy as np
import smplx
from lib.models import get_hmr_vimo
from PIL import Image

from josh.utils.mesh_utils import render_mesh

parser = argparse.ArgumentParser()
parser.add_argument('--input_folder', type=str, default='input')
parser.add_argument('--human_thres', type=float, default=0.7)
parser.add_argument('--visualize', action='store_true')

args = parser.parse_args()

img_files = sorted(glob(os.path.join(args.input_folder, "rgb", "*.jpg")))
img_size = Image.open(img_files[0]).size
img_focal = img_size[0]

bbox_files = sorted(glob(os.path.join(args.input_folder, "mask", "*.json")))
output_folder = os.path.join(args.input_folder, "tram")
os.makedirs(output_folder, exist_ok=True)
tracks = {}

for i, bbox_file in enumerate(bbox_files):
    with open(bbox_file, 'r') as f:
        bbox_data = json.load(f)

    for bbox_info in bbox_data:
        if bbox_info['label'] != 'person' or bbox_info['score'] < args.human_thres:
            continue
        if bbox_info['id'] not in tracks:
            tracks[bbox_info['id']] = {'frame': [], 'det': [], 'det_box': []}
        tracks[bbox_info['id']]['frame'].append(i)
        tracks[bbox_info['id']]['det'].append(True)
        tracks[bbox_info['id']]['det_box'].append(bbox_info['xyxy'])

##### Run HPS (here we use tram) #####
print('Estimate HPS ...')
model = get_hmr_vimo(checkpoint='data/checkpoints/vimo_checkpoint.pth.tar')
smpl_model = smplx.SMPL('data/smpl', gender='neutral')

all_results = []
for k, trk in tracks.items():
    valid = np.array(trk['det'])
    boxes = np.array(trk['det_box'])
    frame = np.array(trk['frame'])
    results = model.inference(img_files, boxes, valid=valid, frame=frame, img_focal=None, img_center=None)
    if results is not None:
        results['boxes'] = boxes
        results['id'] = k
        pred_rotmat = results['pred_rotmat']
        pred_shape = results['pred_shape']
        pred_trans = results['pred_trans'].squeeze(1)
        pred_frame = results['frame'].tolist()
        pred_smpl = smpl_model(body_pose=pred_rotmat[:, 1:],
                               global_orient=pred_rotmat[:, [0]],
                               betas=pred_shape,
                               transl=pred_trans,
                               pose2rot=False,
                               default_smpl=True)
        all_results.append({'frame': pred_frame, 'vertices': pred_smpl.vertices.detach().cpu().numpy()})
        if results is not None:
            np.save(f'{output_folder}/hps_track_{k}.npy', results)

if args.visualize:
    # visualize
    output_vis_folder = os.path.join(args.input_folder, "tram_vis")
    os.makedirs(output_vis_folder, exist_ok=True)

    for i, img_path in enumerate(img_files):
        with open(bbox_files[i], 'r') as f:
            bbox_data = json.load(f)
        img_bgr = cv2.imread(img_path)
        vis_img = img_bgr.copy()
        focal = (img_bgr.shape[1]**2 + img_bgr.shape[0]**2)**0.5
        all_vertices = []
        for results in all_results:
            if i in results['frame']:
                vertices = results['vertices'][results['frame'].index(i)]
                all_vertices.append(vertices)
        vis_img, mask = render_mesh(vis_img,
                                    all_vertices,
                                    smpl_model.faces, {
                                        'focal': [focal, focal],
                                        'princpt': [img_bgr.shape[1] / 2, img_bgr.shape[0] / 2]
                                    },
                                    mesh_as_vertices=False)
        frame_name = os.path.basename(img_path)
        cv2.imwrite(os.path.join(output_vis_folder, frame_name), vis_img.astype(np.uint8))
