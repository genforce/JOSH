import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

os.environ['PYOPENGL_PLATFORM'] = 'egl'
import pyrender
import trimesh


def vis_keypoints_with_skeleton(img, kps, kps_lines, kp_thresh=0.4, alpha=1):
    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(kps_lines) + 2)]
    colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]

    # Perform the drawing on a copy of the image, to allow for blending.
    kp_mask = np.copy(img)

    # Draw the keypoints.
    for l in range(len(kps_lines)):
        i1 = kps_lines[l][0]
        i2 = kps_lines[l][1]
        p1 = kps[0, i1].astype(np.int32), kps[1, i1].astype(np.int32)
        p2 = kps[0, i2].astype(np.int32), kps[1, i2].astype(np.int32)
        if kps[2, i1] > kp_thresh and kps[2, i2] > kp_thresh:
            cv2.line(kp_mask, p1, p2, color=colors[l], thickness=2, lineType=cv2.LINE_AA)
        if kps[2, i1] > kp_thresh:
            cv2.circle(kp_mask, p1, radius=3, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)
        if kps[2, i2] > kp_thresh:
            cv2.circle(kp_mask, p2, radius=3, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)

    # Blend the keypoints.
    return cv2.addWeighted(img, 1.0 - alpha, kp_mask, alpha, 0)


def vis_keypoints(img, kps, alpha=1, radius=3, color=None):
    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    if color is None:
        colors = [cmap(i) for i in np.linspace(0, 1, len(kps) + 2)]
        colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]

    # Perform the drawing on a copy of the image, to allow for blending.
    kp_mask = np.copy(img)

    # Draw the keypoints.
    for i in range(len(kps)):
        p = kps[i][0].astype(np.int32), kps[i][1].astype(np.int32)
        if color is None:
            cv2.circle(kp_mask, p, radius=radius, color=colors[i], thickness=-1, lineType=cv2.LINE_AA)
        else:
            cv2.circle(kp_mask, p, radius=radius, color=color, thickness=-1, lineType=cv2.LINE_AA)

    # Blend the keypoints.
    return cv2.addWeighted(img, 1.0 - alpha, kp_mask, alpha, 0)


def vis_mesh(img, mesh_vertex, alpha=0.5):
    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(mesh_vertex))]
    colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]

    # Perform the drawing on a copy of the image, to allow for blending.
    mask = np.copy(img)

    # Draw the mesh
    for i in range(len(mesh_vertex)):
        p = mesh_vertex[i][0].astype(np.int32), mesh_vertex[i][1].astype(np.int32)
        cv2.circle(mask, p, radius=1, color=colors[i], thickness=-1, lineType=cv2.LINE_AA)

    # Blend the keypoints.
    return cv2.addWeighted(img, 1.0 - alpha, mask, alpha, 0)


def vis_3d_skeleton(kpt_3d, kpt_3d_vis, kps_lines, filename=None):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(kps_lines) + 2)]
    colors = [np.array((c[2], c[1], c[0])) for c in colors]

    for l in range(len(kps_lines)):
        i1 = kps_lines[l][0]
        i2 = kps_lines[l][1]
        x = np.array([kpt_3d[i1, 0], kpt_3d[i2, 0]])
        y = np.array([kpt_3d[i1, 1], kpt_3d[i2, 1]])
        z = np.array([kpt_3d[i1, 2], kpt_3d[i2, 2]])

        if kpt_3d_vis[i1, 0] > 0 and kpt_3d_vis[i2, 0] > 0:
            ax.plot(x, z, -y, c=colors[l], linewidth=2)
        if kpt_3d_vis[i1, 0] > 0:
            ax.scatter(kpt_3d[i1, 0], kpt_3d[i1, 2], -kpt_3d[i1, 1], c=colors[l], marker='o')
        if kpt_3d_vis[i2, 0] > 0:
            ax.scatter(kpt_3d[i2, 0], kpt_3d[i2, 2], -kpt_3d[i2, 1], c=colors[l], marker='o')

    x_r = np.array([0, cfg.input_shape[1]], dtype=np.float32)
    y_r = np.array([0, cfg.input_shape[0]], dtype=np.float32)
    z_r = np.array([0, 1], dtype=np.float32)

    if filename is None:
        ax.set_title('3D vis')
    else:
        ax.set_title(filename)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Z Label')
    ax.set_zlabel('Y Label')
    ax.legend()

    plt.show()
    cv2.waitKey(0)


def pts3d_to_trimesh(img, pts3d, valid=None):
    H, W, THREE = img.shape
    assert THREE == 3
    assert img.shape == pts3d.shape

    vertices = pts3d.reshape(-1, 3)

    # make squares: each pixel == 2 triangles
    idx = np.arange(len(vertices)).reshape(H, W)
    idx1 = idx[:-1, :-1].ravel()  # top-left corner
    idx2 = idx[:-1, +1:].ravel()  # right-left corner
    idx3 = idx[+1:, :-1].ravel()  # bottom-left corner
    idx4 = idx[+1:, +1:].ravel()  # bottom-right corner
    faces = np.concatenate(
        (
            np.c_[idx1, idx2, idx3],
            np.c_[idx3, idx2, idx1],  # same triangle, but backward (cheap solution to cancel face culling)
            np.c_[idx2, idx3, idx4],
            np.c_[idx4, idx3, idx2],  # same triangle, but backward (cheap solution to cancel face culling)
        ),
        axis=0)

    # prepare triangle colors
    face_colors = np.concatenate((img[:-1, :-1].reshape(-1, 3), img[:-1, :-1].reshape(-1, 3), img[+1:, +1:].reshape(-1, 3), img[+1:, +1:].reshape(-1, 3)),
                                 axis=0)

    # remove invalid faces
    if valid is not None:
        assert valid.shape == (H, W)
        valid_idxs = valid.ravel()
        valid_faces = valid_idxs[faces].all(axis=-1)
        faces = faces[valid_faces]
        face_colors = face_colors[valid_faces]

    assert len(faces) == len(face_colors)
    return dict(vertices=vertices, face_colors=face_colors, faces=faces)


def cat_meshes(meshes):
    vertices, faces, colors = zip(*[(m['vertices'], m['faces'], m['face_colors']) for m in meshes])
    n_vertices = np.cumsum([0] + [len(v) for v in vertices])
    for i in range(len(faces)):
        faces[i][:] += n_vertices[i]

    vertices = np.concatenate(vertices)
    colors = np.concatenate(colors)
    faces = np.concatenate(faces)
    return dict(vertices=vertices, face_colors=colors, faces=faces)


def pts3d_to_trimesh(img, pts3d, valid=None):
    H, W, THREE = img.shape
    assert THREE == 3
    assert img.shape == pts3d.shape

    vertices = pts3d.reshape(-1, 3)

    # make squares: each pixel == 2 triangles
    idx = np.arange(len(vertices)).reshape(H, W)
    idx1 = idx[:-1, :-1].ravel()  # top-left corner
    idx2 = idx[:-1, +1:].ravel()  # right-left corner
    idx3 = idx[+1:, :-1].ravel()  # bottom-left corner
    idx4 = idx[+1:, +1:].ravel()  # bottom-right corner
    faces = np.concatenate(
        (
            np.c_[idx1, idx2, idx3],
            np.c_[idx3, idx2, idx1],  # same triangle, but backward (cheap solution to cancel face culling)
            np.c_[idx2, idx3, idx4],
            np.c_[idx4, idx3, idx2],  # same triangle, but backward (cheap solution to cancel face culling)
        ),
        axis=0)

    # prepare triangle colors
    face_colors = np.concatenate((img[:-1, :-1].reshape(-1, 3), img[:-1, :-1].reshape(-1, 3), img[+1:, +1:].reshape(-1, 3), img[+1:, +1:].reshape(-1, 3)),
                                 axis=0)

    # remove invalid faces
    if valid is not None:
        assert valid.shape == (H, W)
        valid_idxs = valid.ravel()
        valid_faces = valid_idxs[faces].all(axis=-1)
        faces = faces[valid_faces]
        face_colors = face_colors[valid_faces]

    assert len(faces) == len(face_colors)
    return dict(vertices=vertices, face_colors=face_colors, faces=faces)


def cat_meshes(meshes):
    vertices, faces, colors = zip(*[(m['vertices'], m['faces'], m['face_colors']) for m in meshes])
    n_vertices = np.cumsum([0] + [len(v) for v in vertices])
    for i in range(len(faces)):
        faces[i][:] += n_vertices[i]

    vertices = np.concatenate(vertices)
    colors = np.concatenate(colors)
    faces = np.concatenate(faces)
    return dict(vertices=vertices, face_colors=colors, faces=faces)


def save_obj(v, f, file_name='output.obj'):
    obj_file = open(file_name, 'w')
    for i in range(len(v)):
        obj_file.write('v ' + str(v[i][0]) + ' ' + str(v[i][1]) + ' ' + str(v[i][2]) + '\n')
    for i in range(len(f)):
        obj_file.write('f ' + str(f[i][0] + 1) + ' ' + str(f[i][1] + 1) + ' ' + str(f[i][2] + 1) + '\n')
    obj_file.close()


def perspective_projection(vertices, cam_param):
    # vertices: [N, 3]
    # cam_param: [3]
    fx, fy = cam_param['focal']
    cx, cy = cam_param['princpt']
    vertices[:, 0] = vertices[:, 0] * fx / vertices[:, 2] + cx
    vertices[:, 1] = vertices[:, 1] * fy / vertices[:, 2] + cy
    return vertices


def render_mesh(img, vertices, faces, cam_param, mesh_as_vertices=False, return_visibility=False):
    if mesh_as_vertices:
        vertices_2d = perspective_projection(vertices.copy(), cam_param)
        img = vis_keypoints(img, vertices_2d, alpha=0.8, radius=2, color=(0, 0, 255))
        original_mask = None
        visible_vertices = np.ones(len(vertices), dtype=bool) if return_visibility else None
    else:
        focal, princpt = cam_param['focal'], cam_param['princpt']
        camera = pyrender.IntrinsicsCamera(fx=focal[0], fy=focal[1], cx=princpt[0], cy=princpt[1])

        pyrender2opencv = np.array([[1.0, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

        material = pyrender.MetallicRoughnessMaterial(metallicFactor=0.1,
                                                      roughnessFactor=0.4,
                                                      alphaMode='OPAQUE',
                                                      emissiveFactor=(0.2, 0.2, 0.2),
                                                      baseColorFactor=(0.7, 0.7, 0.7, 1))

        light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
        cam_pose = pyrender2opencv @ np.eye(4)

        scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=(0.3, 0.3, 0.3))
        scene.add(camera, pose=cam_pose)
        scene.add(light, pose=cam_pose)
        for i, current_vertices in enumerate(vertices):
            body_trimesh = trimesh.Trimesh(current_vertices, faces, process=False)
            body_mesh = pyrender.Mesh.from_trimesh(body_trimesh, material=material)
            scene.add(body_mesh, f'mesh_{i}')

        r = pyrender.OffscreenRenderer(viewport_width=img.shape[1], viewport_height=img.shape[0], point_size=1.0)

        color, depth = r.render(scene, flags=pyrender.RenderFlags.RGBA)
        color = color.astype(np.float32) / 255.0
        alpha = 0.8
        valid_mask = (color[:, :, -1] > 0)[:, :, np.newaxis]
        original_mask = valid_mask.copy()
        valid_mask = valid_mask * alpha
        img = img / 255.0
        color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        output_img = (color * valid_mask + (1 - valid_mask) * img)
        img = (output_img * 255).astype(np.uint8)

        visible_vertices = None
        if return_visibility:
            # Compute vertex visibility
            verts_cam = vertices
            # 2. Project to pixel coords (same intrinsics used above)
            z = verts_cam[:, 2]
            eps = 1e-8
            x_pix = verts_cam[:, 0] * focal[0] / (z + eps) + princpt[0]
            y_pix = verts_cam[:, 1] * focal[1] / (z + eps) + princpt[1]

            # 3. In-frame mask
            H, W = depth.shape
            x_i = np.round(x_pix).astype(int)
            y_i = np.round(y_pix).astype(int)
            in_frame = (x_i >= 0) & (x_i < W) & (y_i >= 0) & (y_i < H) & (z > 0)

            # 4. Depth test (vertex slightly in front of depth buffer)
            depth_sample = np.zeros_like(z) + 1e9
            valid_idx = np.where(in_frame)[0]
            depth_sample[valid_idx] = depth[y_i[valid_idx], x_i[valid_idx]]

            depth_ok = np.abs(z - depth_sample) < 1e-2  # tolerance
            in_front = z <= depth_sample + 1e-4

            # 5. Normal facing test
            face_normals = body_trimesh.face_normals
            # accumulate normals to vertices
            vert_normals = np.zeros_like(vertices)
            for f_i, f in enumerate(faces):
                vert_normals[f] += face_normals[f_i]
            # normalize
            n_norm = np.linalg.norm(vert_normals, axis=1) + 1e-8
            vert_normals /= n_norm[:, None]
            # view direction from vertex toward camera (camera at origin in cam space)
            view_dir = -verts_cam / (np.linalg.norm(verts_cam, axis=1, keepdims=True) + 1e-8)
            facing = (np.sum(vert_normals * view_dir, axis=1) > 0)
            visible_vertices = in_frame & depth_ok

    if return_visibility:
        return img, original_mask, visible_vertices
    return img, original_mask
