import torch
import numpy as np
import roma
import os
import cv2


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