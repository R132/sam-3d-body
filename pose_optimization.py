"""
pose_optimization.py

Utilities to read Sapiens 2D predictions, visualize them, and optimize MHR pose
parameters by minimizing 2D reprojection error.

This module expects to be used from the same environment where the project's
models are available (so you can pass `estimator.model.head_pose` as `mhr_head`).

Functions:
- load_sapiens_predictions(json_path)
- visualize_sapiens_keypoints(img, keypoints, out_path)
- optimize_mhr_pose(mhr_head, init_params, sapiens_kps2d, cam_K, img, faces, out_prefix, device='cuda', num_iters=200, lr=1e-2)
- test_visualize_sapiens(json_path, images_dir, save_path)

Notes:
- `init_params` should be a dict containing keys returned by the estimator for a
  single person: `global_rot`, `body_pose_params`, `hand_pose_params`,
  `scale_params`, `shape_params`, `expr_params`, and `pred_cam_t` (camera translation).
- `mhr_head` should be an instance of the project's MHR head (e.g. `model.head_pose`).

"""

import json
import os
from typing import Dict, Any, Optional

import cv2
import numpy as np
import torch


def load_sapiens_predictions(json_path: str) -> Dict[str, np.ndarray]:
    """Load Sapiens 2D predictions JSON and return mapping image_name -> keypoints array.

    The loader tries to accept multiple common formats:
    - A dict mapping image filenames to {'keypoints': [...]} or list of floats.
    - A list of entries with 'image' and 'keypoints' fields.

    Returned keypoints arrays have shape (N,2) or (N,3) if confidence provided.
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    out = {}

    if isinstance(data, dict):
        # mapping style
        # Check for standard keypoint dict or "frames" format (Sapiens detection output)
        if "frames" in data and isinstance(data["frames"], list):
            # Sapiens format: {"video": ..., "frames": [{"image_name": ..., "instances": [{"keypoints": [...], "keypoint_scores": [...]}]}]}
            for frame in data["frames"]:
                if not isinstance(frame, dict):
                    continue
                image_name = frame.get("image_name")
                if image_name is None:
                    continue
                instances = frame.get("instances", [])
                if not instances:
                    continue
                # Take first instance
                inst = instances[0]
                kps_list = inst.get("keypoints")
                scores_list = inst.get("keypoint_scores")
                if kps_list is None:
                    continue
                kps = np.array(kps_list, dtype=np.float32)  # (N, 2)
                if scores_list is not None:
                    scores = np.array(scores_list, dtype=np.float32)  # (N,)
                    # Append confidence as third column
                    kps = np.concatenate([kps, scores[:, None]], axis=1)
                out[os.path.basename(image_name)] = kps
        else:
            for k, v in data.items():
                if isinstance(v, dict) and "keypoints" in v:
                    kps = np.array(v["keypoints"]) 
                else:
                    # assume the value is the flat list
                    try:
                        kps = np.array(v)
                    except Exception:
                        continue
                if kps.ndim == 1:
                    if kps.size % 3 == 0:
                        kps = kps.reshape(-1, 3)
                    elif kps.size % 2 == 0:
                        kps = kps.reshape(-1, 2)
                out[os.path.basename(k)] = kps
    elif isinstance(data, list):
        for entry in data:
            if not isinstance(entry, dict):
                continue
            image = entry.get("image") or entry.get("file_name") or entry.get("image_id") or entry.get("image_name")
            if image is None:
                continue
            kps = None
            if "keypoints" in entry:
                kps = np.array(entry["keypoints"])
            elif "pred_keypoints_2d" in entry:
                kps = np.array(entry["pred_keypoints_2d"])
            if kps is None:
                continue
            if kps.ndim == 1:
                if kps.size % 3 == 0:
                    kps = kps.reshape(-1, 3)
                elif kps.size % 2 == 0:
                    kps = kps.reshape(-1, 2)
            out[os.path.basename(image)] = kps
    else:
        raise ValueError("Unsupported Sapiens JSON format")

    return out


def visualize_sapiens_keypoints(img: np.ndarray, keypoints: np.ndarray, out_path: str):
    """Draw keypoints on `img` and save to `out_path`.

    keypoints: (N,2) or (N,3) - if third column exists it's treated as confidence.
    """
    vis = img.copy()
    if keypoints is None:
        raise ValueError("keypoints is None")

    kps = keypoints.copy()
    if kps.ndim == 1:
        if kps.size % 3 == 0:
            kps = kps.reshape(-1, 3)
        elif kps.size % 2 == 0:
            kps = kps.reshape(-1, 2)

    if kps.shape[1] == 3:
        pts = kps[:, :2]
        conf = kps[:, 2]
    else:
        pts = kps[:, :2]
        conf = np.ones(len(pts))

    for (x, y), c in zip(pts, conf):
        if np.isnan(x) or np.isnan(y):
            continue
        if c < 0.05:
            continue
        cv2.circle(vis, (int(x), int(y)), 3, (0, 255, 0), -1)

    cv2.imwrite(out_path, vis)


def _project_points(pts3: torch.Tensor, cam_K: np.ndarray, cam_t: torch.Tensor, focal_length: float = 1.0):
    """Project 3D points (N,3) to image plane, matching model's pred_keypoints_2d.

    Model pipeline (mhr_head.py forward() + sam3d_body.py projection):
      1. mhr_forward returns keypoints in Y-up model coords
      2. forward() flips Y/Z: j3d[..., [1, 2]] *= -1
      3. Projection: (j3d_flipped + cam_t) -> weak perspective

    pts3: torch tensor (N,3) - MHR 3D keypoints from mhr_forward (Y-up)
    cam_K: numpy 3x3 camera intrinsics
    cam_t: torch tensor (3,) - camera translation (pred_cam_t, defined in flipped coords)
    focal_length: float - model's focal length

    Returns: torch tensor (N,2) projected 2D coordinates
    """
    if isinstance(pts3, np.ndarray):
        pts3 = torch.from_numpy(pts3).float()
    if pts3.dim() == 3 and pts3.shape[0] == 1:
        pts3 = pts3[0]

    # Flip Y/Z to match model's forward() output (camera system difference)
    pts3_flipped = pts3.clone()
    pts3_flipped[:, 1] *= -1
    pts3_flipped[:, 2] *= -1

    # Add camera translation
    kps_cam = pts3_flipped + cam_t  # (N,3)

    # Weak Perspective projection (matching sam3d_body.py)
    cx = float(cam_K[0, 2])
    cy = float(cam_K[1, 2])
    eps = 1e-6

    X = kps_cam[:, 0]
    Y = kps_cam[:, 1]
    Z = kps_cam[:, 2]

    proj_u = X * focal_length / (Z + eps) + cx
    proj_v = Y * focal_length / (Z + eps) + cy

    return torch.stack([proj_u, proj_v], dim=1)


def optimize_mhr_pose(
    mhr_head,
    init_params: Dict[str, Any],
    sapiens_kps2d: np.ndarray,
    cam_K: np.ndarray,
    img: np.ndarray,
    faces: np.ndarray,
    out_prefix: str,
    device: str = "cuda",
    num_iters: int = 200,
    lr: float = 1e-2,
):
    """Optimize MHR pose parameters (differentiable) to fit 2D Sapiens keypoints.

    Args:
        mhr_head: MHR head (e.g. model.head_pose) which provides `mhr_forward`.
        init_params: dict with keys: `global_rot`, `body_pose_params`, `hand_pose_params`,
            `scale_params`, `shape_params`, `expr_params`, `pred_cam_t`, `focal_length` (optional)
        sapiens_kps2d: (N,2) or (N,3) numpy array of 2D keypoints
        cam_K: 3x3 numpy intrinsics
        img: original image (H,W,3)
        faces: mesh faces array for saving mesh
        out_prefix: path prefix for saved results (without extension)
    Returns:
        dict with 'init_loss', 'final_loss', and 'optimized_params'. Also saves images and mesh.
    """
    use_cuda = torch.cuda.is_available() and str(device).startswith("cuda")
    dev = torch.device("cuda" if use_cuda else "cpu")

    # Prepare initial tensors
    # Some init fields may be numpy arrays
    def to_t(x):
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x).to(dev).float()
        elif isinstance(x, torch.Tensor):
            return x.to(dev).float()
        elif x is None:
            return None
        else:
            return torch.tensor(x, dtype=torch.float32, device=dev)

    global_rot = to_t(init_params.get("global_rot"))
    def _pick_param(a, b):
        v = init_params.get(a)
        if v is None:
            v = init_params.get(b)
        return to_t(v)

    body_pose = _pick_param("body_pose_params", "body_pose")
    hand_pose = _pick_param("hand_pose_params", "hand")
    scale_params = _pick_param("scale_params", "scale")
    shape_params = _pick_param("shape_params", "shape")
    expr_params = _pick_param("expr_params", "expr")
    cam_t = to_t(init_params.get("pred_cam_t"))
    focal = float(init_params.get("focal_length", 1.0))

    # Unsqueeze batch dim
    def ensure_batch(x):
        if x is None:
            return None
        if x.dim() == 1:
            return x.unsqueeze(0)
        return x

    global_rot = ensure_batch(global_rot)
    body_pose = ensure_batch(body_pose)
    hand_pose = ensure_batch(hand_pose) if hand_pose is not None else None
    scale_params = ensure_batch(scale_params)
    shape_params = ensure_batch(shape_params) if shape_params is not None else None
    expr_params = ensure_batch(expr_params) if expr_params is not None else None

    # Ensure tensors expected by mhr_forward are present (use zeros defaults)
    if scale_params is None:
        scale_params = torch.zeros(1, getattr(mhr_head, 'num_scale_comps', 28), device=dev)
    if shape_params is None:
        shape_params = torch.zeros(1, getattr(mhr_head, 'num_shape_comps', 45), device=dev)
    if expr_params is None:
        expr_params = torch.zeros(1, getattr(mhr_head, 'num_face_comps', 72), device=dev)

    # Make parameters require gradients
    optim_vars = []
    # We'll optimize body_pose and global_rot and optionally shape/scale
    if global_rot is None:
        global_rot = torch.zeros(1, 3, device=dev)
    global_rot = global_rot.clone().detach()
    global_rot.requires_grad = True
    optim_vars.append(global_rot)

    if body_pose is None:
        body_pose = torch.zeros(1, 130, device=dev)
    body_pose = body_pose.clone().detach()
    body_pose.requires_grad = True
    optim_vars.append(body_pose)

    if hand_pose is not None:
        hand_pose = hand_pose.clone().detach()
        hand_pose.requires_grad = True
        optim_vars.append(hand_pose)

    if scale_params is not None:
        scale_params = scale_params.clone().detach()
        scale_params.requires_grad = True
        optim_vars.append(scale_params)

    if shape_params is not None:
        shape_params = shape_params.clone().detach()
        shape_params.requires_grad = True
        optim_vars.append(shape_params)

    if expr_params is not None:
        expr_params = expr_params.clone().detach()
        expr_params.requires_grad = True
        optim_vars.append(expr_params)

    # camera translation - now also optimizable (crucial for aligning 3D to 2D)
    if cam_t is None:
        cam_t = torch.zeros(1, 3, device=dev)
    else:
        cam_t = cam_t.to(dev).float()
        if cam_t.dim() == 1:
            cam_t = cam_t.unsqueeze(0)
        elif cam_t.dim() > 2:
            cam_t = cam_t.view(1, -1)[:,:3]
    cam_t = cam_t.clone().detach()
    cam_t.requires_grad = True
    optim_vars.append(cam_t)

    sapiens_kps = sapiens_kps2d.copy()
    if sapiens_kps.ndim == 1:
        if sapiens_kps.size % 3 == 0:
            sapiens_kps = sapiens_kps.reshape(-1, 3)
        elif sapiens_kps.size % 2 == 0:
            sapiens_kps = sapiens_kps.reshape(-1, 2)
    if sapiens_kps.shape[1] == 3:
        vis_mask = ~np.isnan(sapiens_kps[:, 0]) & (sapiens_kps[:, 2] > 0.01)
        target_kps = torch.from_numpy(sapiens_kps[:, :2]).to(dev).float()
    else:
        vis_mask = ~np.isnan(sapiens_kps[:, 0])
        target_kps = torch.from_numpy(sapiens_kps[:, :2]).to(dev).float()

    vis_mask = torch.from_numpy(vis_mask.astype(np.bool_)).to(dev)

    optimizer = torch.optim.Adam(optim_vars, lr=lr)

    def forward_and_project():
        # call mhr_forward with current params
        out = mhr_head.mhr_forward(
            global_trans=torch.zeros(1, 3, device=dev),
            global_rot=global_rot,
            body_pose_params=body_pose,
            hand_pose_params=hand_pose,
            scale_params=scale_params if scale_params is not None else None,
            shape_params=shape_params if shape_params is not None else None,
            expr_params=expr_params if expr_params is not None else None,
            return_keypoints=True,
        )
        if isinstance(out, tuple) or isinstance(out, list):
            verts = out[0]
            kps3d = out[1]
        else:
            verts = out
            kps3d = None

        if kps3d is None:
            raise RuntimeError("MHR forward did not return keypoints; ensure return_keypoints=True")

        # kps3d shape: B x K x 3, K=308 for MHR, take first 70
        kps3d = kps3d.squeeze(0)[:70]  # (70, 3)
        # Match model's weak perspective projection
        proj = _project_points(kps3d, cam_K, cam_t.squeeze(0), focal_length=focal)
        return proj, verts

    # compute initial loss
    with torch.no_grad():
        proj_init, verts_init = forward_and_project()
        vis = vis_mask
        init_loss = ((proj_init[vis] - target_kps[vis]) ** 2).sum(dim=1).mean().item()

    def _snapshot_params():
        """Capture current state of all optimizable parameters."""
        snap = {}
        snap["global_rot"] = global_rot.detach().cpu().numpy()
        snap["body_pose"] = body_pose.detach().cpu().numpy()
        snap["cam_t"] = cam_t.detach().cpu().numpy()
        if hand_pose is not None and hand_pose.requires_grad:
            snap["hand_pose"] = hand_pose.detach().cpu().numpy()
        if scale_params is not None and scale_params.requires_grad:
            snap["scale_params"] = scale_params.detach().cpu().numpy()
        if shape_params is not None and shape_params.requires_grad:
            snap["shape_params"] = shape_params.detach().cpu().numpy()
        if expr_params is not None and expr_params.requires_grad:
            snap["expr_params"] = expr_params.detach().cpu().numpy()
        return snap

    best_loss = init_loss
    best_params = _snapshot_params()

    for it in range(num_iters):
        optimizer.zero_grad()
        proj, verts = forward_and_project()
        if vis_mask.sum().item() == 0:
            loss = torch.tensor(0.0, device=dev)
        else:
            loss = ((proj[vis_mask] - target_kps[vis_mask]) ** 2).sum(dim=1).mean()
        loss_value = loss.item() if not torch.isnan(loss).any().item() else float('nan')
        loss.backward()
        optimizer.step()

        if loss_value < best_loss:
            best_loss = loss_value
            best_params = _snapshot_params()

        if it % 10 == 0 or it == num_iters - 1:
            print(f"iter {it}/{num_iters} loss: {loss_value:.6f}")

    # After optimization, get final verts and proj
    with torch.no_grad():
        proj_final, verts_final = forward_and_project()

    # Clean up GPU memory
    if dev.type == "cuda":
        torch.cuda.empty_cache()

    return {"init_loss": init_loss, "final_loss": best_loss, "optimized_params": best_params}


# ----------------------- Test Helpers -----------------------
def test_visualize_sapiens(json_path: str, images_dir: str, save_path: str):
    """Simple test that reads the Sapiens JSON and visualizes the first entry.

    Saves the visualization to `save_path`.
    """
    preds = load_sapiens_predictions(json_path)
    if len(preds) == 0:
        raise ValueError("No predictions found in " + json_path)
    image_name = list(preds.keys())[0]
    kps = preds[image_name]
    # Try to find image file in images_dir
    cand = os.path.join(images_dir, image_name)
    if not os.path.exists(cand):
        # try common extensions
        stem = os.path.splitext(image_name)[0]
        found = None
        for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
            p = os.path.join(images_dir, stem + ext)
            if os.path.exists(p):
                found = p
                break
        if found is None:
            raise FileNotFoundError(f"Image {image_name} not found in {images_dir}")
        cand = found
    img = cv2.imread(cand)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    visualize_sapiens_keypoints(img, kps, save_path)
    print("Saved visualization to", save_path)


if __name__ == "__main__":
    print("pose_optimization module loaded. Use the functions from your scripts.")
