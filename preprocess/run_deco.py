import argparse
import glob
import os
import sys

sys.path.insert(0, './third_party/deco')

import cv2
import numpy as np
import PIL.Image as pil_img
import pyrender
import torch
import trimesh
from common import constants
from loguru import logger
from models.deco import DECO
from tqdm import tqdm

os.environ['PYOPENGL_PLATFORM'] = 'egl'

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


def initiate_model(args):
    deco_model = DECO('hrnet', True, device)

    logger.info(f'Loading weights from {args.model_path}')
    checkpoint = torch.load(args.model_path, weights_only=False)
    deco_model.load_state_dict(checkpoint['deco'], strict=True)

    deco_model.eval()

    return deco_model


def render_image(scene, img_res, img=None, viewer=False):
    '''
    Render the given pyrender scene and return the image. Can also overlay the mesh on an image.
    '''
    if viewer:
        pyrender.Viewer(scene, use_raymond_lighting=True)
        return 0
    else:
        r = pyrender.OffscreenRenderer(viewport_width=img_res, viewport_height=img_res, point_size=1.0)
        color, _ = r.render(scene, flags=pyrender.RenderFlags.RGBA)
        color = color.astype(np.float32) / 255.0

        if img is not None:
            valid_mask = (color[:, :, -1] > 0)[:, :, np.newaxis]
            input_img = img.detach().cpu().numpy()
            output_img = (color[:, :, :-1] * valid_mask + (1 - valid_mask) * input_img)
        else:
            output_img = color
        return output_img


def create_scene(mesh, img, focal_length=500, camera_center=250, img_res=500):
    # Setup the scene
    scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 1.0], ambient_light=(0.3, 0.3, 0.3))
    # add mesh for camera
    camera_pose = np.eye(4)
    camera_rotation = np.eye(3, 3)
    camera_translation = np.array([0., 0, 2.5])
    camera_pose[:3, :3] = camera_rotation
    camera_pose[:3, 3] = camera_rotation @ camera_translation
    pyrencamera = pyrender.camera.IntrinsicsCamera(fx=focal_length, fy=focal_length, cx=camera_center, cy=camera_center)
    scene.add(pyrencamera, pose=camera_pose)
    # create and add light
    light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=1)
    light_pose = np.eye(4)
    for lp in [[1, 1, 1], [-1, 1, 1], [1, -1, 1], [-1, -1, 1]]:
        light_pose[:3, 3] = mesh.vertices.mean(0) + np.array(lp)
        # out_mesh.vertices.mean(0) + np.array(lp)
        scene.add(light, pose=light_pose)
    # add body mesh
    material = pyrender.MetallicRoughnessMaterial(metallicFactor=0.0, alphaMode='OPAQUE', baseColorFactor=(1.0, 1.0, 0.9, 1.0))
    mesh_images = []

    # resize input image to fit the mesh image height
    img_height = img_res
    img_width = int(img_height * img.shape[1] / img.shape[0])
    img = cv2.resize(img, (img_width, img_height))
    mesh_images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    for sideview_angle in [0, 90, 180, 270]:
        out_mesh = mesh.copy()
        rot = trimesh.transformations.rotation_matrix(np.radians(sideview_angle), [0, 1, 0])
        out_mesh.apply_transform(rot)
        out_mesh = pyrender.Mesh.from_trimesh(out_mesh, material=material)
        mesh_pose = np.eye(4)
        scene.add(out_mesh, pose=mesh_pose, name='mesh')
        output_img = render_image(scene, img_res)
        output_img = pil_img.fromarray((output_img * 255).astype(np.uint8))
        output_img = np.asarray(output_img)[:, :, :3]
        mesh_images.append(output_img)
        # delete the previous mesh
        prev_mesh = scene.get_nodes(name='mesh').pop()
        scene.remove_node(prev_mesh)

    # show upside down view
    for topview_angle in [90, 270]:
        out_mesh = mesh.copy()
        rot = trimesh.transformations.rotation_matrix(np.radians(topview_angle), [1, 0, 0])
        out_mesh.apply_transform(rot)
        out_mesh = pyrender.Mesh.from_trimesh(out_mesh, material=material)
        mesh_pose = np.eye(4)
        scene.add(out_mesh, pose=mesh_pose, name='mesh')
        output_img = render_image(scene, img_res)
        output_img = pil_img.fromarray((output_img * 255).astype(np.uint8))
        output_img = np.asarray(output_img)[:, :, :3]
        mesh_images.append(output_img)
        # delete the previous mesh
        prev_mesh = scene.get_nodes(name='mesh').pop()
        scene.remove_node(prev_mesh)

    # stack images
    IMG = np.hstack(mesh_images)
    IMG = pil_img.fromarray(IMG)
    IMG.thumbnail((3000, 3000))
    return IMG


def main(args):

    images = sorted(glob.iglob(args.input_folder + '/rgb/*', recursive=True))
    tram_results = sorted(glob.iglob(args.input_folder + '/tram/*.npy', recursive=True))
    output_folder = os.path.join(args.input_folder, 'deco')
    os.makedirs(output_folder, exist_ok=True)

    deco_model = initiate_model(args)

    all_imgs = []

    for img_name in images:
        img = cv2.imread(img_name)
        all_imgs.append(img)

    for tram_result in tram_results:
        all_contacts = []
        pred_smpl = np.load(tram_result, allow_pickle=True).item()
        for i, frame in tqdm(enumerate(pred_smpl['frame'])):
            full_img = all_imgs[frame]
            bbox = pred_smpl['boxes'][i]
            x1, y1, x2, y2 = bbox
            center = [(x1 + x2) / 2, (y1 + y2) / 2]
            size = max(x2 - x1, y2 - y1) * 1.2
            x1, y1, x2, y2 = max(int(center[0] - size / 2), 0), max(int(center[1] - size / 2),
                                                                    0), min(int(center[0] + size / 2),
                                                                            full_img.shape[1]), min(int(center[1] + size / 2), full_img.shape[0])
            img = full_img[y1:y2, x1:x2]
            img = cv2.resize(img, (256, 256), cv2.INTER_CUBIC)
            img = img.transpose(2, 0, 1) / 255.0
            img = img[np.newaxis, :, :, :]
            img = torch.tensor(img, dtype=torch.float32).to(device)

            cont, _, _ = deco_model(img)
            cont = cont.detach().cpu().numpy().squeeze()
            cont_smpl = []
            for indx, i in enumerate(cont):
                if i >= 0.5:
                    cont_smpl.append(indx)

            img = img.detach().cpu().numpy()
            img = np.transpose(img[0], (1, 2, 0))
            img = img * 255
            img = img.astype(np.uint8)

            contact_smpl = np.zeros(6890)
            contact_smpl[cont_smpl] = 1
            all_contacts.append(contact_smpl)
        all_contacts = np.stack(all_contacts, axis=0)
        np.save(tram_result.replace('tram', 'deco'), all_contacts)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', help='Source of image(s). Can be file or directory', default='./demo_out', type=str)
    parser.add_argument('--model_path', help='Path to best model weights', default='./data/checkpoints/deco_best.pth', type=str)
    args = parser.parse_args()

    main(args)
