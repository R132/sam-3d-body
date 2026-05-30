import torch
import numpy as np
import roma
import os
import cv2
import json

from tqdm import tqdm
import torch.nn.functional as F


def numpy_to_torch(array, device=None):
    """Convert a numpy array to a 2D float torch tensor on the specified device."""
    if isinstance(array, torch.Tensor):
        t = array.float()
    else:
        # Use from_numpy to keep shape, then convert
        t = torch.from_numpy(array).float()
    
    if device:
        t = t.to(device)
    
    # Ensure it's 2D (1, N) if it was 1D (N,)
    if t.dim() == 1:
        t = t.unsqueeze(0)
    
    return t


class MHRUtils:
    def __init__(self, MHR_model_path):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('Loading MHR model on', self.device)
        self.mhr_model = torch.jit.load(MHR_model_path, map_location=self.device)

        # Load keypoint mapping
        mhr_dir = os.path.dirname(MHR_model_path)
        mapping_path = os.path.join(mhr_dir, "keypoint_mapping.pt")
        if os.path.exists(mapping_path):
            self.keypoint_mapping = torch.load(mapping_path, map_location=self.device)
            print(f"[INFO] Loaded keypoint_mapping from {mapping_path}")
        else:
            print(f"[WARNING] keypoint_mapping.pt not found at {mapping_path}")
            self.keypoint_mapping = None

    def inference(self, outputs):
        """Run inference with MHR from SAM3D outputs"""
        # Convert inputs to tensors on the correct device
        identity_coeffs = numpy_to_torch(outputs['shape_params'], self.device)
        model_parameters = numpy_to_torch(outputs['mhr_model_params'], self.device)
        face_expr_coeffs = numpy_to_torch(outputs['expr_params'], self.device)
        
        vertices, skeleton_state = self.mhr_model(
            identity_coeffs=identity_coeffs,
            model_parameters=model_parameters,
            face_expr_coeffs=face_expr_coeffs,
        )

        vertices = vertices / 100.
        vertices = vertices * torch.tensor([1, -1, -1], device=vertices.device)

        # Add camera translation
        pred_cam_t = numpy_to_torch(outputs.get('pred_cam_t'), self.device)
        vertices = vertices + pred_cam_t
        vertices = vertices.squeeze(0)
        R = torch.tensor(
            [[1, 0, 0], [0, -1, 0], [0, 0, -1]], device=vertices.device
        ).float()

        vertices = (R @ vertices.T).T

        if "camera_c2w" in outputs:
            vertices_homo = torch.concatenate((vertices, torch.ones_like(vertices[:, :1], device=vertices.device)), dim=1)
            vertices = (outputs["camera_c2w"] @ vertices_homo.T).T[:, :3]

        # TODO: Process skeleton_state

        return vertices

    def get_joints_3d(self, outputs):
        identity_coeffs = numpy_to_torch(outputs['shape_params'], self.device)
        model_parameters = numpy_to_torch(outputs['mhr_model_params'], self.device)
        face_expr_coeffs = numpy_to_torch(outputs['expr_params'], self.device)

        vertices, skeleton_state = self.mhr_model(
            identity_coeffs=identity_coeffs,
            model_parameters=model_parameters,
            face_expr_coeffs=face_expr_coeffs,
        )

        # Split skeleton state to get joint coordinates
        joint_coords, joint_quats, _ = torch.split(skeleton_state, [3, 4, 1], dim=2)

        vertices = vertices / 100.
        joint_coords = joint_coords / 100.

        model_vert_joints = torch.cat(
                [vertices, joint_coords], dim=1
            )  # B x (num_verts + 127) x 3

        j3d = (
            (
                self.keypoint_mapping
                @ model_vert_joints.permute(1, 0, 2).flatten(1, 2)
            )
            .reshape(-1, model_vert_joints.shape[0], 3)
            .permute(1, 0, 2)
        )
        j3d = j3d[:, :70]  # 308 --> 70 keypoints
        j3d[..., [1, 2]] *= -1  # Camera system difference

        return j3d.squeeze(0)
    
    def get_joints_3d_diff(self, mhr_paras):
        identity_coeffs = mhr_paras['identity_coeffs']
        model_parameters = mhr_paras['model_parameters']
        face_expr_coeffs = mhr_paras['face_expr_coeffs']

        vertices, skeleton_state = self.mhr_model(
            identity_coeffs=identity_coeffs,
            model_parameters=model_parameters,
            face_expr_coeffs=face_expr_coeffs,
        )

        # Split skeleton state to get joint coordinates
        joint_coords, joint_quats, _ = torch.split(skeleton_state, [3, 4, 1], dim=2)

        vertices = vertices / 100.
        joint_coords = joint_coords / 100.

        model_vert_joints = torch.cat(
                [vertices, joint_coords], dim=1
            )  # B x (num_verts + 127) x 3

        j3d = (
            (
                self.keypoint_mapping
                @ model_vert_joints.permute(1, 0, 2).flatten(1, 2)
            )
            .reshape(-1, model_vert_joints.shape[0], 3)
            .permute(1, 0, 2)
        )
        j3d = j3d[:, :70]  # 308 --> 70 keypoints
        j3d[..., [1, 2]] *= -1  # Camera system difference

        return j3d
    
    def get_j2d_diff(self, mhr_paras):
        height = mhr_paras["height"]
        width = mhr_paras["width"]
        focal_length = mhr_paras["focal_length"]
        pred_cam_t = mhr_paras["pred_cam_t"]

        j3d = self.get_joints_3d_diff(mhr_paras)

        pred_keypoints_3d_proj = j3d + pred_cam_t
        # 直接相乘，因为维度已经对齐
        pred_keypoints_3d_proj[:, :, [0, 1]] *= focal_length
        
        # 图像中心偏移
        pred_keypoints_3d_proj[:, :, [0, 1]] = (
            pred_keypoints_3d_proj[:, :, [0, 1]]
            + torch.FloatTensor([width / 2, height / 2]).to(pred_keypoints_3d_proj)[None, None, :]
            * pred_keypoints_3d_proj[:, :, [2]]
        )
        
        # 透视除法
        pred_keypoints_3d_proj[:, :, :2] = (
            pred_keypoints_3d_proj[:, :, :2] / pred_keypoints_3d_proj[:, :, [2]]
        )
        
        pred_keypoints_2d = pred_keypoints_3d_proj[:, :, :2]

        return pred_keypoints_2d

    def j2d_loss(self, pred_kps2d, target_kps2d):
        return F.mse_loss(pred_kps2d, target_kps2d)

    def project_joints_to_2d(self, outputs, img, output_image_path=None):
        j3d = self.get_joints_3d(outputs)
        height, width = img.shape[:2]

        focal_length = outputs.get('focal_length', 1.0)
        pred_cam_t = outputs.get('pred_cam_t')

        # 确保维度对齐以进行广播计算
        # j3d is (1, 70, 3)
        if j3d.dim() == 2:
            j3d = j3d.unsqueeze(0)

        focal_length = torch.tensor(focal_length, device=j3d.device).float()
        if focal_length.dim() == 0:
            focal_length = focal_length.view(1, 1, 1)  # Scalar -> (1, 1, 1)

        pred_cam_t = torch.tensor(pred_cam_t, device=j3d.device).float()
        if pred_cam_t.dim() == 1:
            pred_cam_t = pred_cam_t.unsqueeze(0).unsqueeze(0)  # (3,) -> (1, 1, 3)
        elif pred_cam_t.dim() == 2:
            pred_cam_t = pred_cam_t.unsqueeze(1)  # (1, 3) -> (1, 1, 3)

        pred_keypoints_3d_proj = j3d + pred_cam_t
        # 直接相乘，因为维度已经对齐
        pred_keypoints_3d_proj[:, :, [0, 1]] *= focal_length
        
        # 图像中心偏移
        pred_keypoints_3d_proj[:, :, [0, 1]] = (
            pred_keypoints_3d_proj[:, :, [0, 1]]
            + torch.FloatTensor([width / 2, height / 2]).to(pred_keypoints_3d_proj)[None, None, :]
            * pred_keypoints_3d_proj[:, :, [2]]
        )
        
        # 透视除法
        pred_keypoints_3d_proj[:, :, :2] = (
            pred_keypoints_3d_proj[:, :, :2] / pred_keypoints_3d_proj[:, :, [2]]
        )
        
        pred_keypoints_2d = pred_keypoints_3d_proj[:, :, :2]

        if output_image_path is not None:
            # Draw keypoints on image
            vis_img = img.copy()
            kps_2d = pred_keypoints_2d.squeeze(0).cpu().numpy() # Shape: (70, 2)
            for i, (x, y) in enumerate(kps_2d):
                x, y = int(x), int(y)
                if 0 <= x < width and 0 <= y < height:
                    cv2.circle(vis_img, (x, y), 3, (0, 255, 0), -1)
            
            out_dir = os.path.dirname(output_image_path)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            cv2.imwrite(output_image_path, vis_img)
            print(f"[INFO] Saved projected keypoints image to {output_image_path}")

        return pred_keypoints_2d
    
    def opt_pose_by_kps2d(self, outputs, target_kps2d, img, num_iters=100, lr=0.01, output_image_path=None):
        """
        Optimize MHR pose parameters (body_pose, global_rot, cam_t) to fit target 2D keypoints.
        """
        device = self.device
        
        # 0. 准备 Target (转为 Tensor, shape: 1, 70, 2)
        if isinstance(target_kps2d, np.ndarray):
            target_kps2d = torch.from_numpy(target_kps2d).float().to(device)
        if target_kps2d.dim() == 2:
            target_kps2d = target_kps2d.unsqueeze(0)

        # 获取图像尺寸用于投影
        height, width = img.shape[:2]

        # 1. 初始化参数 (提取 outputs 中的数据并包装为可优化变量)
        def to_opt_var(outputs, key, shape=None, requires_grad=True):
            val = outputs.get(key)
            if val is None:
                raise ValueError(f"Key '{key}' not found in outputs for optimization.")
            if isinstance(val, np.ndarray):
                t = torch.from_numpy(val).float().to(device)
            else:
                t = val.float().to(device)
            # 确保维度为 (1, N)
            if t.dim() == 1: t = t.unsqueeze(0)
            t.requires_grad_(requires_grad)
            return t

        # 提取并设置优化器管理的参数
        mhr_paras ={}
        identity_coeffs = to_opt_var(outputs, 'shape_params')
        model_parameters = to_opt_var(outputs, 'mhr_model_params')
        face_expr_coeffs = to_opt_var(outputs, 'expr_params')

        global_rot = to_opt_var(outputs, 'global_rot', shape=(1, 3))
        pred_cam_t = to_opt_var(outputs, 'pred_cam_t', shape=(1, 3))

        # 2. 初始化优化器
        optimizer = torch.optim.Adam([identity_coeffs, model_parameters, face_expr_coeffs, global_rot, pred_cam_t], lr=lr)

        mhr_paras = {
            "identity_coeffs": identity_coeffs,
            "model_parameters": model_parameters,
            "face_expr_coeffs": face_expr_coeffs,
            "global_rot": global_rot,
            "pred_cam_t": pred_cam_t,
            "height": height,
            "width": width,
            "focal_length": outputs['focal_length'],
        }

        loop = tqdm(range(num_iters), desc="Optimizing MHR pose")
        for iter in loop:
            optimizer.zero_grad()
            
            # --- 前向传播 (Differentiable MHR Inference) ---
            pred_j2d = self.get_j2d_diff(mhr_paras)  # Shape: (1, 70, 2)

            # 计算 Loss
            loss = self.j2d_loss(pred_j2d, target_kps2d)

            # 反向传播
            loss.backward()
            optimizer.step()
            if iter % 10 == 0:
                print(f"Iter {iter}: Loss {loss.item():.6f}")
                loop.set_description(f"Iter {iter}: Loss {loss.item():.6f}")
                # 可视化
                if output_image_path is not None:
                    # 绘制优化结果
                    vis_img = img.copy()
                    pred_j2d_np = pred_j2d.squeeze(0).cpu().numpy()
                    target_kps2d_np = target_kps2d.squeeze(0).cpu().numpy()
                    visualize_dual_keypoints_on_image(vis_img, target_kps2d_np, pred_j2d_np, output_image_path.replace('.png', '_iter{}.png'.format(iter)))

        # # 5. 回填数据到 outputs
        # outputs['body_pose_params'] = body_pose_params.detach().cpu().numpy().squeeze(0)
        # outputs['global_rot'] = global_rot.detach().cpu().numpy().squeeze(0)
        # outputs['pred_cam_t'] = pred_cam_t.detach().cpu().numpy().squeeze(0)
        
        # # 注意：更新 outputs 参数后，pred_keypoints_3d/2d 等字段并未自动更新，
        # # 需要调用 estimator.process_one_image 或重新计算。
        # # 这里为了演示，仅更新了基础参数。

        return outputs

    def _inference_wrt_rt(self, outputs):
        """Run inference with MHR from SAM3D outputs"""
        # Convert inputs to tensors on the correct device
        identity_coeffs = numpy_to_torch(outputs['shape_params'], self.device)
        model_parameters = numpy_to_torch(outputs['mhr_model_params'], self.device)
        face_expr_coeffs = numpy_to_torch(outputs['expr_params'], self.device)
        
        vertices, skeleton_state = self.mhr_model(
            identity_coeffs=identity_coeffs,
            model_parameters=model_parameters,
            face_expr_coeffs=face_expr_coeffs,
        )

        vertices = vertices / 100.
        vertices = vertices * torch.tensor([1, -1, -1], device=vertices.device)

        joint_coords, joint_quats, _ = torch.split(skeleton_state, split_size_or_sections=[3, 4, 1], dim=2)
        joint_coords = joint_coords / 100.
        joint_coords = joint_coords * torch.tensor([1, -1, -1], device=vertices.device)
        joint_rots = roma.unitquat_to_rotmat(joint_quats)

        return vertices, joint_coords, joint_rots

    def _test_mhr_inference(self, outputs):
        """Compare the MHR inference results with vertices in outputs."""
        # Run MHR inference
        mhr_vertices, joint_coords, joint_rots = self._inference_wrt_rt(outputs)
        
        # vertices test : PASSED
        # Get original vertices from outputs
        print("  [TEST] vertices in outputs and MHR inference")
        pred_vertices = outputs.get('pred_vertices')
        if pred_vertices is None:
            print("  [TEST] No pred_vertices found in outputs")
            return None, None
        
        # Convert to numpy if needed
        if isinstance(pred_vertices, torch.Tensor):
            pred_vertices = pred_vertices.cpu().numpy()
        if isinstance(mhr_vertices, torch.Tensor):
            mhr_vertices = mhr_vertices.cpu().numpy()
        
        # Ensure same shape
        if pred_vertices.ndim == 3:
            pred_vertices = pred_vertices.squeeze(0)
        if mhr_vertices.ndim == 3:
            mhr_vertices = mhr_vertices.squeeze(0)
        
        # Compute MSE
        mse = ((pred_vertices - mhr_vertices) ** 2).mean()
        
        # Compute per-vertex errors (Euclidean distance)
        per_vertex_errors = np.sqrt(((pred_vertices - mhr_vertices) ** 2).sum(axis=1))
        
        print(f"  [TEST] Vertices MSE: {mse:.6f}")
        print(f"  [TEST] Vertices mean error: {per_vertex_errors.mean():.6f}")
        print(f"  [TEST] Vertices max error: {per_vertex_errors.max():.6f}")
        print(f"  [TEST] Pred vertices shape: {pred_vertices.shape}")
        print(f"  [TEST] MHR vertices shape: {mhr_vertices.shape}")

        """
        [TEST] Vertices MSE: 0.000000
        [TEST] Vertices mean error: 0.000000
        [TEST] Vertices max error: 0.000000
        [TEST] Pred vertices shape: (18439, 3)
        [TEST] MHR vertices shape: (18439, 3)
        """

        print("")
        print("  [TEST] pred_joint_coords in outputs and MHR inference")

        pred_joint_coords = outputs.get('pred_joint_coords')  # (127, 3）
        joint_coords = joint_coords.squeeze(0).cpu().numpy()

        mse = ((pred_joint_coords - joint_coords) ** 2).mean()
        per_vertex_errors = np.sqrt(((pred_joint_coords - joint_coords) ** 2).sum(axis=1))
        
        print(f"  [TEST] Joints MSE: {mse:.6f}")
        print(f"  [TEST] Joints mean error: {per_vertex_errors.mean():.6f}")
        print(f"  [TEST] Joints max error: {per_vertex_errors.max():.6f}")
        print(f"  [TEST] Pred joints shape: {pred_joint_coords.shape}")
        print(f"  [TEST] MHR joints shape: {joint_coords.shape}")

        """
        [TEST] pred_joint_coords in outputs and MHR inference
        [TEST] Joints MSE: 0.000000
        [TEST] Joints mean error: 0.000000
        [TEST] Joints max error: 0.000000
        [TEST] Pred joints shape: (127, 3)
        [TEST] MHR joints shape: (127, 3)
        """

        # Run MHR inference
        mhr_vertices, joint_coords, joint_rots = self._inference_wrt_rt(outputs)
        mhr_vertices = mhr_vertices * torch.tensor([1, -1, -1], device=self.device)
        joint_coords = joint_coords * torch.tensor([1, -1, -1], device=self.device)

        model_vert_joints = torch.cat(
                [mhr_vertices, joint_coords], dim=1
            )  # B x (num_verts + 127) x 3

        j3d = (
            (
                self.keypoint_mapping
                @ model_vert_joints.permute(1, 0, 2).flatten(1, 2)
            )
            .reshape(-1, model_vert_joints.shape[0], 3)
            .permute(1, 0, 2)
        )
        j3d = j3d[:, :70]  # 308 --> 70 keypoints
        j3d[..., [1, 2]] *= -1  # Camera system difference

        print("")
        print("  [TEST] j3d in outputs and MHR inference")

        pred_j3d = outputs.get('pred_keypoints_3d')  # (70, 3）
        j3d = j3d.squeeze(0).cpu().numpy()

        mse = ((pred_j3d - j3d) ** 2).mean()
        per_vertex_errors = np.sqrt(((pred_j3d - j3d) ** 2).sum(axis=1))
        
        print(f"  [TEST] Joints MSE: {mse:.6f}")
        print(f"  [TEST] Joints mean error: {per_vertex_errors.mean():.6f}")
        print(f"  [TEST] Joints max error: {per_vertex_errors.max():.6f}")
        print(f"  [TEST] Pred joints shape: {pred_j3d.shape}")
        print(f"  [TEST] MHR joints shape: {j3d.shape}")

        """
        [TEST] j3d in outputs and MHR inference
        [TEST] Joints MSE: 0.000000
        [TEST] Joints mean error: 0.000000
        [TEST] Joints max error: 0.000000
        [TEST] Pred joints shape: (70, 3)
        [TEST] MHR joints shape: (70, 3)

        """


        return mse, per_vertex_errors
    

def load_sapiens_kps(json_path: str) -> dict:
    """Load Sapiens 2D predictions JSON and return mapping image_name -> keypoints array.

    The loader tries to accept multiple common formats:
    - A dict mapping image filenames to {'keypoints': [...]} or list of floats.
    - A list of entries with 'image' and 'keypoints' fields.
    - A "frames" format (Sapiens detection output).

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


def visualize_keypoints_on_image(img, keypoints, output_path=None):
    """
    Visualize 2D keypoints on an image.

    Args:
        img: Input image (numpy array, BGR format).
        keypoints: Keypoints array of shape (N, 2) or (N, 3) where 3rd dim is confidence.
        output_path: If provided, save the image to this path.

    Returns:
        The image with keypoints drawn.
    """
    vis_img = img.copy()
    h, w = img.shape[:2]

    # Ensure keypoints are numpy array
    if hasattr(keypoints, 'cpu'):
        keypoints = keypoints.cpu().numpy()
    
    # Handle shape
    if keypoints.ndim == 1:
        if keypoints.size % 3 == 0:
            keypoints = keypoints.reshape(-1, 3)
        elif keypoints.size % 2 == 0:
            keypoints = keypoints.reshape(-1, 2)
    
    pts = keypoints[:, :2]
    conf = keypoints[:, 2] if keypoints.shape[1] >= 3 else np.ones(len(pts))

    for i, (x, y) in enumerate(pts):
        c = conf[i]
        # Filter out invalid or low confidence points
        if np.isnan(x) or np.isnan(y) or c < 0.1:
            continue
        
        x, y = int(x), int(y)
        if 0 <= x < w and 0 <= y < h:
            # Draw circle for keypoint
            cv2.circle(vis_img, (x, y), 4, (0, 255, 0), -1)
            # Optional: Draw index number
            # cv2.putText(vis_img, str(i), (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

    if output_path is not None:
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        cv2.imwrite(output_path, vis_img)
        print(f"[INFO] Saved keypoints visualization to {output_path}")

    return vis_img


def visualize_dual_keypoints_on_image(img, keypoints1, keypoints2, output_path=None):
    """
    Visualize two sets of keypoints on an image.
    keypoints1 -> Green
    keypoints2 -> Red
    """
    vis_img = img.copy()
    h, w = img.shape[:2]

    def draw_kps(kps, color):
        if kps is None: return
        if hasattr(kps, 'cpu'): kps = kps.cpu().numpy()
        
        # 1. 处理 Batch 维度: (1, N, 2) -> (N, 2)
        if kps.ndim == 3 and kps.shape[0] == 1:
            kps = kps[0]
        # 2. 处理 1D 展平数据
        elif kps.ndim == 1:
            if kps.size % 3 == 0: kps = kps.reshape(-1, 3)
            elif kps.size % 2 == 0: kps = kps.reshape(-1, 2)

        # 确保是 2D 数组 (N, 2) 或 (N, 3)
        if kps.ndim != 2: return

        pts = kps[:, :2]
        conf = kps[:, 2] if kps.shape[1] >= 3 else np.ones(len(pts))

        for i in range(len(pts)):
            x, y = float(pts[i, 0]), float(pts[i, 1])
            c = float(conf[i])
            if np.isnan(x) or np.isnan(y) or c < 0.1: continue
            
            x, y = int(x), int(y)
            if 0 <= x < w and 0 <= y < h:
                cv2.circle(vis_img, (x, y), 4, color, -1)

    # 1. Draw keypoints1 in GREEN
    draw_kps(keypoints1, (0, 255, 0))

    # 2. Draw keypoints2 in RED
    draw_kps(keypoints2, (0, 0, 255))

    if output_path is not None:
        out_dir = os.path.dirname(output_path)
        if out_dir: os.makedirs(out_dir, exist_ok=True)
        cv2.imwrite(output_path, vis_img)
        print(f"[INFO] Saved dual keypoints visualization to {output_path}")

    return vis_img