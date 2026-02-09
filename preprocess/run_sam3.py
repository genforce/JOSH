import sys

sys.path.insert(0, './third_party/sam3')

import argparse
import glob
import json
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pycocotools import mask as maskutils
from sam3.visualization_utils import (load_frame, prepare_masks_for_visualization, render_masklet_frame, visualize_formatted_frame_output)


def propagate_in_video(predictor, session_id):
    # we will just propagate from frame 0 to the end of the video
    outputs_per_frame = {}
    for response in predictor.handle_stream_request(request=dict(
            type="propagate_in_video",
            session_id=session_id,
    )):
        outputs_per_frame[response["frame_index"]] = response["outputs"]

    return outputs_per_frame


from sam3.model_builder import build_sam3_video_predictor

parser = argparse.ArgumentParser()
parser.add_argument('--input_folder', type=str, default='input')
args = parser.parse_args()
video_predictor = build_sam3_video_predictor()
video_path = os.path.join(args.input_folder, "rgb")  # a JPEG folder or an MP4 video file
video_frames_for_vis = sorted(glob.glob(os.path.join(video_path, "*.jpg")))

# Start a session
response = video_predictor.handle_request(request=dict(
    type="start_session",
    resource_path=video_path,
))
session_id = response["session_id"]

response = video_predictor.handle_request(request=dict(
    type="add_prompt",
    session_id=session_id,
    frame_index=0,
    text="person",
))

MASK_FOLDER = os.path.join(args.input_folder, "mask")
VIS_FOLDER = os.path.join(args.input_folder, "vis")
os.makedirs(MASK_FOLDER, exist_ok=True)
os.makedirs(VIS_FOLDER, exist_ok=True)
# now we propagate the outputs from frame 0 to the end of the video and collect all outputs
outputs_per_frame = propagate_in_video(video_predictor, session_id)

for i, img_path in enumerate(video_frames_for_vis):
    img_base = os.path.basename(img_path)
    img = load_frame(video_frames_for_vis[i])
    frame_result = []
    masked_img = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    for idx in range(len(outputs_per_frame[i]['out_obj_ids'])):
        instance_result = {}
        instance_result["id"] = int(outputs_per_frame[i]['out_obj_ids'][idx])
        x1, y1, w, h = outputs_per_frame[i]['out_boxes_xywh'][idx]
        instance_result["xyxy"] = [float(x1) * img.shape[1], float(y1) * img.shape[0], float((x1 + w) * img.shape[1]), float((y1 + h) * img.shape[0])]
        instance_result["label"] = "person"
        instance_result["score"] = float(outputs_per_frame[i]['out_probs'][idx])
        rle = maskutils.encode(np.asfortranarray(outputs_per_frame[i]['out_binary_masks'][idx].astype(np.uint8)))
        if isinstance(rle["counts"], bytes):
            rle["counts"] = rle["counts"].decode("ascii")
        instance_result["mask"] = rle
        frame_result.append(instance_result)
        masked_img[outputs_per_frame[i]['out_binary_masks'][idx]] = 255
    with open(os.path.join(MASK_FOLDER, f"{img_base[:-4]}.json"), "w") as f:
        json.dump(frame_result, f)

    frame_mask = render_masklet_frame(img, outputs_per_frame[i])
    Image.fromarray(frame_mask).save(os.path.join(VIS_FOLDER, f"{img_base[:-4]}.jpg"))
    Image.fromarray(masked_img).save(os.path.join(MASK_FOLDER, f"{img_base[:-4]}.png"))
