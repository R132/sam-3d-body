# Copyright (c) Meta Platforms, Inc. and affiliates.
import argparse
import os
from glob import glob

import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml", ".sl"],
    pythonpath=True,
    dotenv=True,
)

import cv2
import numpy as np
import torch
from sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator
from sam_3d_body.metadata.mhr70 import pose_info as mhr70_pose_info
from notebook.utils import save_mesh_results
from pose_optimization import test_visualize_sapiens, load_sapiens_predictions, optimize_mhr_pose
from tools.vis_utils import visualize_sample, visualize_sample_together
from tqdm import tqdm
import json


def parse_colmap_cameras(cameras_txt_path):
    cams = {}
    if not os.path.exists(cameras_txt_path):
        return cams
    with open(cameras_txt_path, "r") as f:
        for line in f:
            if line.startswith("#") or len(line.strip()) == 0:
                continue
            parts = line.strip().split()
            # COLMAP cameras.txt format: cam_id, model, width, height, params...
            cam_id = int(parts[0])
            model = parts[1]
            params = [float(x) for x in parts[4:]]
            # For PINHOLE / SIMPLE_PINHOLE / OPENCV, params typically contain fx,fy,cx,cy or fx,cx,fy,cy variants.
            if len(params) >= 4:
                fx = params[0]
                fy = params[1]
                cx = params[2]
                cy = params[3]
            elif len(params) == 3:
                fx = params[0]
                fy = params[0]
                cx = params[1]
                cy = params[2]
            else:
                continue
            K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
            cams[cam_id] = {"model": model, "K": K}
    return cams


def parse_colmap_images(images_txt_path):
    imgs = {}
    if not os.path.exists(images_txt_path):
        return imgs
    with open(images_txt_path, "r") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if line.startswith("#") or len(line) == 0:
            continue

        parts = line.split()
        # COLMAP images.txt: IMAGE_ID QW QX QY QZ TX TY TZ CAM_ID NAME
        # Next line is 2D keypoints (X, Y, POINT3D_ID) — skip it
        if i < len(lines):
            i += 1  # skip keypoints line

        if len(parts) < 10:
            continue

        try:
            image_id = int(parts[0])
        except ValueError:
            continue  # not a header line, skip

        qw, qx, qy, qz = [float(x) for x in parts[1:5]]
        tx, ty, tz = [float(x) for x in parts[5:8]]
        cam_id = int(parts[8])
        name = parts[9]
        entry = {
            "image_id": image_id,
            "qvec": [qw, qx, qy, qz],
            "tvec": [tx, ty, tz],
            "cam_id": cam_id,
        }
        imgs[name] = entry
        imgs[name.lower()] = entry
        imgs[os.path.splitext(name)[0]] = entry
        imgs[os.path.splitext(name.lower())[0]] = entry

    return imgs


def read_sequence(input_root):
    """Read sequence structured as:
    input_root/images/*.png
    input_root/masks/*.png
    input_root/sparse/0/cameras.txt and images.txt (COLMAP-style)
    Returns a list of dicts with image_path, mask_path, cam_int (torch.tensor)
    """
    images_dir = os.path.join(input_root, "images")
    masks_dir = os.path.join(input_root, "masks")
    sparse_dir = os.path.join(input_root, "sparse", "0")

    cameras = parse_colmap_cameras(os.path.join(sparse_dir, "cameras.txt"))
    images = parse_colmap_images(os.path.join(sparse_dir, "images.txt"))

    image_paths = sorted(
        [os.path.join(images_dir, f) for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'))]
    )

    seq = []
    for img_path in image_paths:
        name = os.path.basename(img_path)
        mask_path = os.path.join(masks_dir, name)
        cam_int = None
        key = name
        if key not in images:
            key = name.lower()
        if key not in images:
            key = os.path.splitext(name)[0]
        if key not in images:
            key = os.path.splitext(name.lower())[0]
        if key in images:
            cam_id = images[key]["cam_id"]
            if cam_id in cameras:
                cam_int = torch.from_numpy(cameras[cam_id]["K"]).float().unsqueeze(0)
        else:
            print(f"Warning: no COLMAP image entry found for {name}")
        seq.append({
            "image": img_path,
            "mask": mask_path if os.path.exists(mask_path) else None,
            "cam_int": cam_int,
        })
    return seq


def save_keypoint_comparison(
    img: np.ndarray,
    outputs: list,
    sapiens_kps2d: np.ndarray,  # (70, 2) or (70, 3)
    cam_K: np.ndarray,          # (3, 3)
    out_path: str,
):
    """Visualize Sapiens 2D keypoints (green) and projected MHR 3D keypoints (red) on the same image."""
    if len(outputs) == 0:
        return

    first = outputs[0]
    pred_k3d = np.array(first["pred_keypoints_3d"])  # (70, 3)
    cam_t = np.array(first["pred_cam_t"])            # (3,)
    focal = float(first.get("focal_length", 1.0))
    h, w = img.shape[:2]

    # Match model's weak perspective projection (sam3d_body.py:1628-1645)
    # 1. Add camera translation
    kps_cam = pred_k3d + cam_t  # (70, 3)
    # 2. Apply focal length to X, Y
    kps_cam[:, 0] *= focal
    kps_cam[:, 1] *= focal
    # 3. Add image center offset * Z
    cx = cam_K[0, 2]
    cy = cam_K[1, 2]
    kps_cam[:, 0] += cx * kps_cam[:, 2]
    kps_cam[:, 1] += cy * kps_cam[:, 2]
    # 4. Divide by Z
    kps_cam[:, 0] /= kps_cam[:, 2]
    kps_cam[:, 1] /= kps_cam[:, 2]
    proj_2d = kps_cam[:, :2]  # (70, 2)

    vis = img.copy()

    # Draw Sapiens 2D keypoints (green)
    kps = sapiens_kps2d.copy()
    if kps.ndim == 1:
        if kps.size % 3 == 0:
            kps = kps.reshape(-1, 3)
        elif kps.size % 2 == 0:
            kps = kps.reshape(-1, 2)
    pts2d = kps[:, :2]
    conf = kps[:, 2] if kps.shape[1] >= 3 else np.ones(len(pts2d))
    for (x, y), c in zip(pts2d, conf):
        if np.isnan(x) or np.isnan(y) or c < 0.05:
            continue
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(vis, (int(x), int(y)), 4, (0, 255, 0), -1)

    # Draw projected MHR 2D keypoints (red)
    for i in range(len(proj_2d)):
        px, py = proj_2d[i]
        if 0 <= px < w and 0 <= py < h:
            cv2.circle(vis, (int(px), int(py)), 3, (0, 0, 255), -1)

    # Add legend
    cv2.circle(vis, (20, 30), 5, (0, 255, 0), -1)
    cv2.putText(vis, "Sapiens 2D", (30, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    cv2.circle(vis, (20, 55), 5, (0, 0, 255), -1)
    cv2.putText(vis, "MHR 2D proj", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)

    cv2.imwrite(out_path, vis)
    print(f"Saved keypoint comparison: {out_path}")


def load_keypoints2d_for_image(keypoints_folder, image_name, exts=(".npy", ".npz", ".json")):
    base = os.path.splitext(image_name)[0]
    for ext in exts:
        cand = os.path.join(keypoints_folder, base + ext)
        if os.path.exists(cand):
            if ext == ".npy":
                return np.load(cand)
            if ext == ".npz":
                return np.load(cand)["arr_0"]
            if ext == ".json":
                with open(cand, "r") as f:
                    return np.array(json.load(f))
    return None


def reprojection_loss(pred_keypoints_3d, cam_K, keypoints2d, vis_thresh=0.0):
    """Project 3D keypoints (N,3) using intrinsics cam_K (3x3) and compute L2 loss to keypoints2d (N,2).
    pred_keypoints_3d: numpy array (N,3) in camera coordinates
    cam_K: numpy array (3,3)
    keypoints2d: numpy array (N,2) or (N,3) where third dim may be visibility/confidence
    Returns mean squared reprojection error and per-joint errors.
    """
    if keypoints2d is None:
        return None, None
    pts3 = pred_keypoints_3d.copy()
    eps = 1e-6
    Z = pts3[:, 2:3] + eps
    fx = cam_K[0, 0]
    fy = cam_K[1, 1]
    cx = cam_K[0, 2]
    cy = cam_K[1, 2]
    proj_x = fx * (pts3[:, 0:1] / Z) + cx
    proj_y = fy * (pts3[:, 1:2] / Z) + cy
    proj = np.concatenate([proj_x, proj_y], axis=1)
    if keypoints2d.shape[1] >= 3:
        vis = keypoints2d[:, 2] > vis_thresh
    else:
        vis = ~np.isnan(keypoints2d[:, 0])
    dif = proj - keypoints2d[:, :2]
    squared = (dif**2).sum(axis=1)
    if vis.sum() == 0:
        return None, None
    mse = squared[vis].mean()
    per_joint = np.sqrt(squared)
    return mse, per_joint



def main(args):
    # Require CUDA GPU
    if not torch.cuda.is_available():
        print("ERROR: CUDA GPU is required but not available. "
              "This program must run on a GPU for MHR inference and pose optimization.")
        print("Check: nvidia-smi, torch.cuda.is_available()")
        sys.exit(1)

    # Determine output folder
    if args.output_folder == "":
        output_folder = os.path.join(args.input_dir, "sam3D")
    else:
        output_folder = args.output_folder

    os.makedirs(output_folder, exist_ok=True)

    # Load Sapiens predictions if available (for pose optimization)
    sapiens_kps_dict = None
    input_root = args.input_dir
    sapiens_json_path = os.path.join(input_root, "sapiens_pose", "sapiens_pose_predictions.json")
    if os.path.exists(sapiens_json_path):
        # Images are in <root>/images/
        sapiens_images_dir = input_root

        # Visualize first frame if requested
        vis_out = os.path.join(output_folder, "sapiens_kps_vis.png")
        if args.visualize_sapiens:
            test_visualize_sapiens(sapiens_json_path, sapiens_images_dir, vis_out)
            print("Saved sapiens visualization to", vis_out)

        # Load all predictions into a dict keyed by image filename
        try:
            raw_dict = load_sapiens_predictions(sapiens_json_path)
            # Build a lookup with multiple key formats for robust matching
            sapiens_kps_dict = {}
            for k, v in raw_dict.items():
                sapiens_kps_dict[k] = v
                sapiens_kps_dict[os.path.splitext(k)[0]] = v
                sapiens_kps_dict[k.lower()] = v
                sapiens_kps_dict[os.path.splitext(k.lower())[0]] = v
            print(f"Loaded {len(raw_dict)} Sapiens pose predictions from {sapiens_json_path}")
        except Exception as e:
            print(f"Warning: could not load Sapiens predictions: {e}")

    # Use command-line args or environment variables
    mhr_path = args.mhr_path or os.environ.get("SAM3D_MHR_PATH", "")
    detector_path = args.detector_path or os.environ.get("SAM3D_DETECTOR_PATH", "")
    segmentor_path = args.segmentor_path or os.environ.get("SAM3D_SEGMENTOR_PATH", "")
    fov_path = args.fov_path or os.environ.get("SAM3D_FOV_PATH", "")

    # Initialize sam-3d-body model and other optional modules
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required but not available.")
    device = torch.device("cuda")

    model, model_cfg = load_sam_3d_body(args.checkpoint_path, device=device, mhr_path=mhr_path)

    human_detector, human_segmentor, fov_estimator = None, None, None
    if args.detector_name:
        from tools.build_detector import HumanDetector
        human_detector = HumanDetector(name=args.detector_name, device=device, path=detector_path)

    if (args.segmentor_name == "sam2" and len(segmentor_path)) or args.segmentor_name != "sam2":
        from tools.build_sam import HumanSegmentor

        human_segmentor = HumanSegmentor(name=args.segmentor_name, device=device, path=segmentor_path)

    seq = None
    seq = read_sequence(args.input_dir)
    if len(seq) == 0:
        raise ValueError(f"No images found in input_dir: {args.input_dir}")
    
    # TODO: DEBUG
    seq = seq[:1]

    cam_count = sum(frame["cam_int"] is not None for frame in seq)
    print(
        f"Using structured input_dir; reading COLMAP intrinsics from {args.input_dir}/sparse/0 "
        f"({cam_count}/{len(seq)} frames found)."
    )

    estimator = SAM3DBodyEstimator(
        sam_3d_body_model=model,
        model_cfg=model_cfg,
        human_detector=human_detector,
        human_segmentor=human_segmentor,
        fov_estimator=fov_estimator,
    )

    losses = []

    for idx, frame in enumerate(tqdm(seq)):
        img_path = frame["image"]
        mask_path = frame["mask"]
        cam_int = frame["cam_int"]

        masks = None
        if mask_path is not None and os.path.exists(mask_path):
            mask_img = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
            if mask_img is not None:
                # ensure single channel binary mask
                if mask_img.ndim == 3:
                    mask_img = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
                masks = mask_img.astype(np.uint8)[None, ...]

        outputs = estimator.process_one_image(
            img_path,
            bboxes=None,
            masks=masks,
            cam_int=cam_int,
            bbox_thr=args.bbox_thresh,
            use_mask=bool(masks is not None),
        )

        img = cv2.imread(img_path)
        rend_img = visualize_sample_together(img, outputs, estimator.faces)
        out_name = os.path.basename(img_path)
        cv2.imwrite(os.path.join(output_folder, out_name), rend_img.astype(np.uint8))
        print(f"Frame {idx}: saved {out_name}")

        if args.save_mesh and idx % args.mesh_save_interval == 0:
            save_mesh_results(img, outputs, estimator.faces, output_folder, os.path.splitext(out_name)[0])

        # Load keypoints for this frame: check keypoints_folder first, then sapiens dict
        kp2d = None
        kp2d_full = None  # original 308 keypoints from Sapiens
        if args.keypoints_folder:
            kp2d_full = load_keypoints2d_for_image(args.keypoints_folder, out_name)
            if kp2d_full is not None:
                print(f"Frame {idx}: loaded keypoints from {args.keypoints_folder}, shape={kp2d_full.shape}")
        if kp2d_full is None and sapiens_kps_dict is not None:
            stem = os.path.splitext(out_name)[0]
            if stem in sapiens_kps_dict:
                kp2d_full = sapiens_kps_dict[stem]
                print(f"Frame {idx}: loaded sapiens keypoints, shape={kp2d_full.shape}")
            elif out_name in sapiens_kps_dict:
                kp2d_full = sapiens_kps_dict[out_name]
            elif out_name.lower() in sapiens_kps_dict:
                kp2d_full = sapiens_kps_dict[out_name.lower()]

        # Map 308 Sapiens keypoints → 70 MHR keypoints
        if kp2d_full is not None:
            mhr70_idxs = list(mhr70_pose_info["original_keypoint_info"].keys())
            kp2d = kp2d_full[mhr70_idxs]  # (70, 2) or (70, 3)
            print(f"Frame {idx}: mapped to MHR70 keypoints, shape={kp2d.shape}")

        # Visualize 2D keypoints (before optimization)
        if kp2d_full is not None:
            kps_vis = img.copy()
            kps = kp2d_full.copy()
            if kps.ndim == 1:
                if kps.size % 3 == 0:
                    kps = kps.reshape(-1, 3)
                elif kps.size % 2 == 0:
                    kps = kps.reshape(-1, 2)
            pts = kps[:, :2]
            conf = kps[:, 2] if kps.shape[1] >= 3 else np.ones(len(pts))
            for (x, y), c in zip(pts, conf):
                if np.isnan(x) or np.isnan(y) or c < 0.05:
                    continue
                cv2.circle(kps_vis, (int(x), int(y)), 3, (0, 255, 0), -1)
            kp_out_path = os.path.join(output_folder, f"kps_{os.path.splitext(out_name)[0]}.jpg")
            cv2.imwrite(kp_out_path, kps_vis)

        # Save keypoint comparison (Sapiens 2D + MHR 3D projected)
        if len(outputs) > 0 and kp2d is not None:
            if cam_int is not None:
                comp_cam_K = cam_int.squeeze(0).cpu().numpy()
            else:
                hf, wf = img.shape[:2]
                f = float(outputs[0].get("focal_length", 1.0))
                comp_cam_K = np.array([[f, 0.0, wf / 2.0], [0.0, f, hf / 2.0], [0.0, 0.0, 1.0]], dtype=np.float32)
            comp_out_path = os.path.join(output_folder, f"kps_cmp_{os.path.splitext(out_name)[0]}.jpg")
            save_keypoint_comparison(img, outputs, kp2d, comp_cam_K, comp_out_path)

        # Run MHR pose optimization if sapiens keypoints are available
        if len(outputs) > 0 and kp2d is not None:
            first = outputs[0]
            init_params = {
                'global_rot': first.get('global_rot'),
                'body_pose_params': first.get('body_pose_params'),
                'hand_pose_params': first.get('hand_pose_params'),
                'scale_params': first.get('scale_params'),
                'shape_params': first.get('shape_params'),
                'expr_params': first.get('expr_params'),
                'pred_cam_t': first.get('pred_cam_t'),
                'focal_length': float(first.get('focal_length', 1.0)),
            }
            # Debug: print init_params
            print(f"\n[OPT] init_params keys: {list(init_params.keys())}")
            for k, v in init_params.items():
                if v is not None:
                    if hasattr(v, 'shape'):
                        print(f"  {k}: shape={v.shape}")
                    else:
                        print(f"  {k}: {v}")
                else:
                    print(f"  {k}: None")

            # Build camera intrinsics matrix
            if cam_int is not None:
                opt_cam_K = cam_int.squeeze(0).cpu().numpy()
            else:
                hf, wf = img.shape[:2]
                f = float(first.get('focal_length', 1.0))
                opt_cam_K = np.array([[f, 0.0, wf / 2.0], [0.0, f, hf / 2.0], [0.0, 0.0, 1.0]], dtype=np.float32)

            opt_out_prefix = os.path.join(output_folder, f"opt_{os.path.splitext(out_name)[0]}")
            print(f"\n[OPT] Frame {idx}: Running pose optimization, iters={args.opt_iters}")
            opt_result = optimize_mhr_pose(
                estimator.model.head_pose,
                init_params,
                kp2d,
                opt_cam_K,
                img,
                estimator.faces,
                opt_out_prefix,
                device=device,
                num_iters=args.opt_iters,
                lr=args.opt_lr,
            )
            print(f"[OPT] Frame {idx}: init_loss={opt_result['init_loss']:.1f}, final_loss={opt_result['final_loss']:.1f}, improvement={100*(1-opt_result['final_loss']/opt_result['init_loss']):.1f}%")
            losses.append({
                "frame": out_name,
                "init_loss": float(opt_result['init_loss']),
                "final_loss": float(opt_result['final_loss']),
            })

        # Fallback: compute simple reprojection loss if no sapiens keypoints
        elif len(outputs) > 0 and kp2d is not None and "pred_keypoints_3d" in outputs[0]:
            first = outputs[0]
            pred_k3d = np.array(first["pred_keypoints_3d"])
            if cam_int is not None:
                K = cam_int.squeeze(0).cpu().numpy()
            elif "focal_length" in first:
                hf, wf = img.shape[:2]
                f = first.get("focal_length", 1.0)
                K = np.array([[f, 0.0, wf / 2.0], [0.0, f, hf / 2.0], [0.0, 0.0, 1.0]], dtype=np.float32)
            else:
                K = np.array([[1.0, 0.0, img.shape[1] / 2.0], [0.0, 1.0, img.shape[0] / 2.0], [0.0, 0.0, 1.0]], dtype=np.float32)
            mse, _ = reprojection_loss(pred_k3d, K, kp2d)
            if mse is not None:
                losses.append({"frame": out_name, "mse": float(mse)})

    # Save losses summary if any
    if len(losses) > 0:
        with open(os.path.join(output_folder, "reproj_losses.json"), "w") as f:
            json.dump(losses, f, indent=2)
        print("Saved reprojection losses to", os.path.join(output_folder, "reproj_losses.json"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SAM 3D Body Demo - Single Image Human Mesh Recovery",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
                Examples:
                python demo.py --input_dir ./data/sequence_01 --checkpoint_path ./checkpoints/model.ckpt

                Environment Variables:
                SAM3D_MHR_PATH: Path to MHR asset
                SAM3D_DETECTOR_PATH: Path to human detection model folder
                SAM3D_SEGMENTOR_PATH: Path to human segmentation model folder
                SAM3D_FOV_PATH: Path to fov estimation model folder
                """,
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        type=str,
        help="Path to structured input directory containing images/, masks/, sparse/0/",
    )
    parser.add_argument(
        "--output_folder",
        default="",
        type=str,
        help="Path to output folder (default: <input_dir>/sam3D)",
    )
    parser.add_argument(
        "--checkpoint_path",
        default="/home/gj/Projects/2026.05.07-SMA3/checkpoints/sam-3d-body/sam-3d-body-vith/model.ckpt",
        type=str,
        help="Path to SAM 3D Body model checkpoint",
    )
    parser.add_argument(
        "--detector_name",
        default="",
        type=str,
        help="Human detection model for demo. Leave empty to disable detector and use full-image inference.",
    )
    parser.add_argument(
        "--segmentor_name",
        default="sam2",
        type=str,
        help="Human segmentation model for demo (Default `sam2`, add your favorite segmentor if needed).",
    )
    parser.add_argument(
        "--fov_name",
        default="moge2",
        type=str,
        help="FOV estimation model for demo (Default `moge2`, add your favorite fov estimator if needed).",
    )
    parser.add_argument(
        "--detector_path",
        default="",
        type=str,
        help="Path to human detection model folder (or set SAM3D_DETECTOR_PATH)",
    )
    parser.add_argument(
        "--segmentor_path",
        default="",
        type=str,
        help="Path to human segmentation model folder (or set SAM3D_SEGMENTOR_PATH)",
    )
    parser.add_argument(
        "--fov_path",
        default="",
        type=str,
        help="Path to fov estimation model folder (or set SAM3D_FOV_PATH)",
    )
    parser.add_argument(
        "--mhr_path",
        default="/home/gj/Projects/2026.05.07-SMA3/checkpoints/MHR/assets/mhr_model.pt",
        type=str,
        help="Path to MoHR/assets folder (or set SAM3D_mhr_path)",
    )
    parser.add_argument(
        "--bbox_thresh",
        default=0.8,
        type=float,
        help="Bounding box detection threshold",
    )
    parser.add_argument(
        "--use_mask",
        action="store_true",
        default=False,  # TODO:
        help="Use mask-conditioned prediction (segmentation mask is automatically generated from bbox)",
    )
    parser.add_argument(
        "--save_mesh",
        type=lambda x: x.lower() == 'true',
        default=True,
        help="Save mesh output files for selected frames (default: True, use --save_mesh False to disable)",
    )
    parser.add_argument(
        "--mesh_save_interval",
        default=10,
        type=int,
        help="Save mesh every N frames when mesh saving is enabled",
    )
    parser.add_argument(
        "--keypoints_folder",
        default="",
        type=str,
        help="Folder containing 2D keypoints files matching image basenames (.npy/.npz/.json)",
    )
    parser.add_argument(
        "--visualize_sapiens",
        action="store_true",
        default=False,
        help="Visualize sapiens 2D keypoints and save to output folder immediately",
    )
    parser.add_argument(
        "--opt_iters",
        default=200,
        type=int,
        help="Number of optimization iterations for MHR pose fitting (default: 200)",
    )
    parser.add_argument(
        "--opt_lr",
        default=1e-2,
        type=float,
        help="Learning rate for MHR pose optimization (default: 0.01)",
    )
    args = parser.parse_args()

    main(args)
