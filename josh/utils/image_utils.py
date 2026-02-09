import math
import os
import os.path as osp

import cv2
import numpy as np
import torch
import torchvision.transforms as tvf
from PIL import Image
from skimage.color import lab2rgb, rgb2lab
from sklearn.cluster import KMeans

from josh.config import ImageDict

ImgNorm = tvf.Compose([tvf.ToTensor(), tvf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


def _filter_edges_seq(edges, seq_dis_thr, cyclic=False):
    # number of images
    n = max(max(e) for e in edges) + 1

    kept = []
    for e, (i, j) in enumerate(edges):
        dis = abs(i - j)
        if cyclic:
            dis = min(dis, abs(i + n - j), abs(i - n - j))
        if dis <= seq_dis_thr:
            kept.append(e)
    return kept


def generate_colors(n_colors=256, n_samples=5000):
    # Step 1: Random RGB samples
    np.random.seed(42)
    rgb = np.random.rand(n_samples, 3)
    # Step 2: Convert to LAB for perceptual uniformity
    # print(f"Converting {n_samples} RGB samples to LAB color space...")
    lab = rgb2lab(rgb.reshape(1, -1, 3)).reshape(-1, 3)
    # print("Conversion to LAB complete.")
    # Step 3: k-means clustering in LAB
    kmeans = KMeans(n_clusters=n_colors, n_init=10)
    # print(f"Fitting KMeans with {n_colors} clusters on {n_samples} samples...")
    kmeans.fit(lab)
    # print("KMeans fitting complete.")
    centers_lab = kmeans.cluster_centers_
    # Step 4: Convert LAB back to RGB
    colors_rgb = lab2rgb(centers_lab.reshape(1, -1, 3)).reshape(-1, 3)
    colors_rgb = np.clip(colors_rgb, 0, 1)
    return colors_rgb


def filter_pairs_seq(pairs, seq_dis_thr, cyclic=False):
    edges = [(img1["idx"], img2["idx"]) for img1, img2 in pairs]
    kept = _filter_edges_seq(edges, seq_dis_thr, cyclic=cyclic)
    return [pairs[i] for i in kept]


def make_pairs(
    imgs: list[ImageDict],
    scene_graph: str = "complete",
    prefilter=None,
    symmetrize=True,
) -> list[tuple[ImageDict, ImageDict]]:
    pairs = []
    if scene_graph == "complete":  # complete graph
        for i in range(len(imgs)):
            for j in range(i):
                pairs.append((imgs[i], imgs[j]))
    elif scene_graph.startswith("swin"):
        winsize = int(scene_graph.split("-")[1]) if "-" in scene_graph else 3
        pairsid = set()
        for i in range(len(imgs)):
            for j in range(1, winsize + 1):
                idx = (i + j) % len(imgs)  # explicit loop closure
                pairsid.add((i, idx) if i < idx else (idx, i))
        for i, j in pairsid:
            pairs.append((imgs[i], imgs[j]))

    elif scene_graph.startswith("window"):
        winsize = int(scene_graph.split("-")[1]) if "-" in scene_graph else 3
        pairsid = set()
        for i in range(len(imgs)):
            for j in range(1, winsize + 1):
                idx = i + j
                if idx < len(imgs):
                    pairsid.add((i, idx) if i < idx else (idx, i))
        for i, j in pairsid:
            pairs.append((imgs[i], imgs[j]))
    elif scene_graph.startswith("oneref"):
        refid = int(scene_graph.split("-")[1]) if "-" in scene_graph else 0
        for j in range(len(imgs)):
            if j != refid:
                pairs.append((imgs[refid], imgs[j]))
    if symmetrize:
        pairs += [(img2, img1) for img1, img2 in pairs]

    # now, remove edges
    if isinstance(prefilter, str) and prefilter.startswith("seq"):
        pairs = filter_pairs_seq(pairs, int(prefilter[3:]))

    if isinstance(prefilter, str) and prefilter.startswith("cyc"):
        pairs = filter_pairs_seq(pairs, int(prefilter[3:]), cyclic=True)

    return pairs


def _resize_pil_image(img, long_edge_size):
    S = max(img.size)
    if S > long_edge_size:
        interp = Image.LANCZOS
    elif S <= long_edge_size:
        interp = Image.BICUBIC
    new_size = tuple(int(round(x * long_edge_size / S)) for x in img.size)
    return img.resize(new_size, interp)


def _resize_mask(img, long_edge_size):
    S = max(img.size)
    interp = Image.NEAREST
    new_size = tuple(int(round(x * long_edge_size / S)) for x in img.size)
    return img.resize(new_size, interp)


def load_images_and_masks(
    image_files,
    mask_files,
    interval=1,
    start_frame=0,
    max_num=10000,
):
    imgs = []
    masks = []
    img_idx = []
    i = start_frame

    remaining_num = len(image_files) - start_frame

    # Hardcoded
    if remaining_num < 1 + interval * (max_num - 1) * 2:
        num_frames = remaining_num
    else:
        num_frames = 1 + interval * (max_num - 1)

    new_filelist = []

    while i < len(image_files):

        img = Image.open(image_files[i]).convert("RGB")
        W1, H1 = img.size
        img = _resize_pil_image(img, 512)
        W, H = img.size
        print(f" - adding {image_files[i]} with resolution {W1}x{H1} --> {W}x{H}")
        imgs.append(dict(
            img=ImgNorm(img)[None],
            true_shape=np.int32([img.size[::-1]]),
            idx=len(imgs),
            instance=str(len(imgs)),
        ))

        mask = Image.open(mask_files[i]).convert('L')
        mask = _resize_mask(mask, 512)
        masks.append(dict(
            img=tvf.ToTensor()(mask)[None],
            true_shape=np.int32([mask.size[::-1]]),
            idx=len(masks),
            instance=str(len(masks)),
        ))

        img_idx.append(i - start_frame)

        new_filelist.append(image_files[i])

        if i - start_frame + 1 == num_frames:
            break
        elif i - start_frame + 1 + interval > num_frames:
            i = num_frames + start_frame - 1
        else:
            i += interval

    return imgs, masks, num_frames, img_idx, new_filelist


def load_images_as_tensor(path='data/truck', interval=1, PIXEL_LIMIT=255000, target_size=None):
    """
    Loads images from a directory or video, resizes them to a uniform size,
    then converts and stacks them into a single [N, 3, H, W] PyTorch tensor.
    """
    sources = []

    # --- 1. Load image paths or video frames ---
    if type(path) == list:
        print(f"Loading images from a list of path")
        filenames = path
        for i in range(0, len(filenames), interval):
            try:
                sources.append(Image.open(filenames[i]).convert('RGB'))
            except Exception as e:
                print(f"Could not load image {filenames[i]}: {e}")
    elif osp.isdir(path):
        print(f"Loading images from directory: {path}")
        filenames = sorted([x for x in os.listdir(path) if x.lower().endswith(('.png', '.jpg', '.jpeg', '.JPG'))])
        for i in range(0, len(filenames), interval):
            img_path = osp.join(path, filenames[i])
            try:
                sources.append(Image.open(img_path).convert('RGB'))
            except Exception as e:
                print(f"Could not load image {filenames[i]}: {e}")
    elif path.lower().endswith('.mp4'):
        print(f"Loading frames from video: {path}")
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video file: {path}")
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % interval == 0:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                sources.append(Image.fromarray(rgb_frame))
            frame_idx += 1
        cap.release()

    else:
        raise ValueError(f"Unsupported path. Must be a directory or a .mp4 file: {path}")

    if not sources:
        print("No images found or loaded.")
        return torch.empty(0)

    print(f"Found {len(sources)} images/frames. Processing...")

    # --- 2. Determine a uniform target size for all images based on the first image ---
    # This is necessary to ensure all tensors have the same dimensions for stacking.
    if target_size is not None:
        TARGET_W, TARGET_H = target_size
        print(f"Using provided target size: ({TARGET_W}, {TARGET_H})")
    else:
        first_img = sources[0]
        W_orig, H_orig = first_img.size
        scale = math.sqrt(PIXEL_LIMIT / (W_orig * H_orig)) if W_orig * H_orig > 0 else 1
        W_target, H_target = W_orig * scale, H_orig * scale
        k, m = round(W_target / 14), round(H_target / 14)
        while (k * 14) * (m * 14) > PIXEL_LIMIT:
            if k / m > W_target / H_target:
                k -= 1
            else:
                m -= 1
        TARGET_W, TARGET_H = max(1, k) * 14, max(1, m) * 14
    print(f"All images will be resized to a uniform size: ({TARGET_W}, {TARGET_H})")

    # --- 3. Resize images and convert them to tensors in the [0, 1] range ---
    tensor_list = []
    # Define a transform to convert a PIL Image to a CxHxW tensor and normalize to [0,1]
    to_tensor_transform = tvf.ToTensor()

    for img_pil in sources:
        try:
            # Resize to the uniform target size
            resized_img = img_pil.resize((TARGET_W, TARGET_H), Image.Resampling.LANCZOS)
            # Convert to tensor
            img_tensor = to_tensor_transform(resized_img)
            tensor_list.append(img_tensor)
        except Exception as e:
            print(f"Error processing an image: {e}")

    if not tensor_list:
        print("No images were successfully processed.")
        return torch.empty(0)

    # --- 4. Stack the list of tensors into a single [N, C, H, W] batch tensor ---
    return torch.stack(tensor_list, dim=0)
