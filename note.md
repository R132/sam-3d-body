# Code Notes

## MHR Inference Logic: Skeleton State Parsing

**File Path:**
`sam_3d_body/models/heads/mhr_head.py`

**Function:**
`mhr_forward` (Lines ~226-233)

**Code Snippet:**
```python
curr_skinned_verts, curr_skel_state = self.mhr(
    shape_params, model_params, expr_params
)
curr_joint_coords, curr_joint_quats, _ = torch.split(
    curr_skel_state, [3, 4, 1], dim=2
)
curr_skinned_verts = curr_skinned_verts / 100
curr_joint_coords = curr_joint_coords / 100
curr_joint_rots = roma.unitquat_to_rotmat(curr_joint_quats)
```

**Description:**
该代码块调用核心 MHR 模型生成网格顶点和骨架状态，并解析骨架数据：
1.  **拆分骨架状态**：使用 `torch.split` 按维度 `dim=2` 将骨架数据拆解为：
    *   `[3]`：关节的 3D 坐标 (x, y, z)。
    *   `[4]`：关节的旋转四元数 (w, x, y, z)。
    *   `[1]`：忽略剩余的 1 维数据（可能是掩码）。
2.  **单位换算**：除以 100，将模型输出的坐标单位（通常为厘米）转换为米。
3.  **旋转转换**：使用 `roma.unitquat_to_rotmat` 将四元数转换为旋转矩阵，便于后续几何计算。

---

## MHR Inference Logic: Keypoints Extraction

**File Path:**
`sam_3d_body/models/heads/mhr_head.py`

**Function:**
`mhr_forward` (Lines ~239-251)

**Code Snippet:**
```python
if return_keypoints:
    # Get sapiens 308 keypoints
    model_vert_joints = torch.cat(
        [curr_skinned_verts, curr_joint_coords], dim=1
    )  # B x (num_verts + 127) x 3
    model_keypoints_pred = (
        (
            self.keypoint_mapping
            @ model_vert_joints.permute(1, 0, 2).flatten(1, 2)
        )
        .reshape(-1, model_vert_joints.shape[0], 3)
        .permute(1, 0, 2)
    )
```

**Description:**
该代码块通过矩阵映射从密集几何点中提取 308 个 Sapiens 关键点：
1.  **融合几何数据**：将蒙皮顶点 (`curr_skinned_verts`) 和关节坐标 (`curr_joint_coords`) 拼接，形成包含所有几何信息的张量。
2.  **维度变换**：将 Batch 维移到最后并展平坐标，使其符合矩阵乘法要求。
3.  **关键点采样**：通过预训练的权重矩阵 `self.keypoint_mapping` 进行线性变换。这实际上是基于皮肤权重，根据顶点和关节的位置插值计算关键点坐标，而不是直接预测。
4.  **形状恢复**：将计算结果重新变换回 `(Batch, 308, 3)` 的标准格式。