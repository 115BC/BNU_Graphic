import torch
import numpy as np

def batch_rodrigues(rot_vecs):
    """
    将轴角表示转换为旋转矩阵
    Args:
        rot_vecs: (batch_size, num_joints, 3) - 轴角向量
    Returns:
        rot_mats: (batch_size, num_joints, 3, 3) - 旋转矩阵
    """
    batch_size = rot_vecs.shape[0]
    num_joints = rot_vecs.shape[1]
    
    angle = torch.norm(rot_vecs, dim=-1, keepdim=True)
    axis = rot_vecs / (angle + 1e-8)
    
    cos_angle = torch.cos(angle)
    sin_angle = torch.sin(angle)
    
    # 构造旋转矩阵
    identity = torch.eye(3, device=rot_vecs.device).unsqueeze(0).unsqueeze(0)
    outer = torch.matmul(axis.unsqueeze(-1), axis.unsqueeze(-2))
    
    axis_skew = torch.zeros((batch_size, num_joints, 3, 3), device=rot_vecs.device)
    axis_skew[:, :, 0, 1] = -axis[:, :, 2]
    axis_skew[:, :, 1, 0] = axis[:, :, 2]
    axis_skew[:, :, 0, 2] = axis[:, :, 1]
    axis_skew[:, :, 2, 0] = -axis[:, :, 1]
    axis_skew[:, :, 1, 2] = -axis[:, :, 0]
    axis_skew[:, :, 2, 1] = axis[:, :, 0]
    
    rot_mats = cos_angle.unsqueeze(-1) * identity + \
               (1 - cos_angle).unsqueeze(-1) * outer + \
               sin_angle.unsqueeze(-1) * axis_skew
    
    return rot_mats

def batch_rodrigues(rot_vecs):
    """
    将轴角表示转换为旋转矩阵
    Args:
        rot_vecs: (batch_size, num_joints, 3) - 轴角向量
    Returns:
        rot_mats: (batch_size, num_joints, 3, 3) - 旋转矩阵
    """
    batch_size = rot_vecs.shape[0]
    num_joints = rot_vecs.shape[1]
    
    angle = torch.norm(rot_vecs, dim=-1, keepdim=True)
    axis = rot_vecs / (angle + 1e-8)
    
    cos_angle = torch.cos(angle)
    sin_angle = torch.sin(angle)
    
    # 构造旋转矩阵
    identity = torch.eye(3, device=rot_vecs.device).unsqueeze(0).unsqueeze(0)
    outer = torch.matmul(axis.unsqueeze(-1), axis.unsqueeze(-2))
    
    axis_skew = torch.zeros((batch_size, num_joints, 3, 3), device=rot_vecs.device)
    axis_skew[:, :, 0, 1] = -axis[:, :, 2]
    axis_skew[:, :, 1, 0] = axis[:, :, 2]
    axis_skew[:, :, 0, 2] = axis[:, :, 1]
    axis_skew[:, :, 2, 0] = -axis[:, :, 1]
    axis_skew[:, :, 1, 2] = -axis[:, :, 0]
    axis_skew[:, :, 2, 1] = axis[:, :, 0]
    
    rot_mats = cos_angle.unsqueeze(-1) * identity + \
               (1 - cos_angle).unsqueeze(-1) * outer + \
               sin_angle.unsqueeze(-1) * axis_skew
    
    return rot_mats

def vertices2joints(J_regressor, vertices):
    """
    从顶点回归关节位置
    Args:
        J_regressor: (num_joints, num_vertices) - 关节回归器
        vertices: (batch_size, num_vertices, 3) - 顶点位置
    Returns:
        J: (batch_size, num_joints, 3) - 关节位置
    """
    J = torch.matmul(J_regressor, vertices)
    return J

def blend_shapes(betas, shapedirs):
    """
    计算形状形变
    Args:
        betas: (batch_size, num_betas) - 形状参数
        shapedirs: (num_vertices, 3, num_betas) - 形状形变基
    Returns:
        blend_shape: (batch_size, num_vertices, 3) - 形状形变
    """
    batch_size = betas.shape[0]
    num_vertices = shapedirs.shape[0]
    
    # (batch_size, num_betas) x (num_vertices, 3, num_betas) -> (batch_size, num_vertices, 3)
    blend_shape = torch.einsum('bi,vji->bvi', betas, shapedirs)
    return blend_shape

def batch_rigid_transform(rot_mats, joints, parents):
    """
    计算每个关节的全局刚体变换
    Args:
        rot_mats: (batch_size, num_joints, 3, 3) - 旋转矩阵
        joints: (batch_size, num_joints, 3) 或 (num_joints, 3) - 关节位置
        parents: (num_joints,) - 父关节索引
    Returns:
        J_transformed: (batch_size, num_joints, 3) - 变换后的关节位置
        A: (batch_size, num_joints, 4, 4) - 全局变换矩阵
    """
    batch_size = rot_mats.shape[0]
    num_joints = rot_mats.shape[1]
    
    # 确保 parents 是 numpy 数组
    if isinstance(parents, torch.Tensor):
        parents = parents.cpu().numpy().astype(int)
    
    # 确保 joints 有 batch 维度
    if joints.dim() == 2:
        joints = joints.unsqueeze(0)
    
    # 创建 4x4 变换矩阵
    A = torch.zeros((batch_size, num_joints, 4, 4), device=rot_mats.device)
    A[:, :, 3, 3] = 1.0
    
    # 填充旋转部分
    A[:, :, :3, :3] = rot_mats
    
    J_transformed = torch.zeros((batch_size, num_joints, 3), device=rot_mats.device)
    
    for i in range(num_joints):
        if parents[i] == -1:
            # 根关节
            A[:, i, :3, 3] = joints[:, i]
            J_transformed[:, i] = joints[:, i]
        else:
            # 子关节：先变换到父关节坐标系
            parent_idx = parents[i]
            
            # 使用 bmm 进行批量矩阵乘法
            diff = joints[:, i] - joints[:, parent_idx]
            matmul_result = torch.bmm(A[:, parent_idx, :3, :3], diff.unsqueeze(-1)).squeeze(-1)
            
            A[:, i, :3, 3] = matmul_result + A[:, parent_idx, :3, 3]
            J_transformed[:, i] = A[:, i, :3, 3]
    
    return J_transformed, A
def lbs(betas, global_orient, body_pose, v_template, shapedirs, posedirs, J_regressor, parents, lbs_weights):
    """
    完整的线性混合蒙皮实现
    Args:
        betas: (batch_size, num_betas) - 形状参数
        global_orient: (batch_size, 3) - 全局方向
        body_pose: (batch_size, 69) - 身体姿态
        v_template: (num_vertices, 3) - 模板顶点
        shapedirs: (num_vertices, 3, num_betas) - 形状形变基
        posedirs: (num_vertices, 3, 207) - 姿态形变基
        J_regressor: (num_joints, num_vertices) - 关节回归器
        parents: (num_joints,) - 父关节索引
        lbs_weights: (num_vertices, num_joints) - 蒙皮权重
    Returns:
        v_template: 模板顶点
        v_shaped: 形状校正后顶点
        J: 回归出的关节位置
        v_posed: 姿态校正后顶点
        pose_offsets: 姿态偏移
        verts: 最终蒙皮顶点
        J_transformed: 变换后的关节位置
    """
    batch_size = betas.shape[0]
    num_vertices = v_template.shape[0]
    num_joints = J_regressor.shape[0]
    
    # 设备设置
    device = betas.device
    
    # (a) 模板顶点
    v_template = v_template.unsqueeze(0).expand(batch_size, -1, -1)
    
    # (b) 形状校正
    shape_offsets = blend_shapes(betas, shapedirs)
    v_shaped = v_template + shape_offsets
    
    # 回归关节位置
    J = vertices2joints(J_regressor, v_shaped)
    
    # (c) 姿态校正
    # 合并全局方向和身体姿态
    full_pose = torch.cat([global_orient, body_pose], dim=1)  # (batch_size, 72)
    full_pose = full_pose.view(batch_size, num_joints, 3)  # (batch_size, 24, 3)
    
    # 转换为旋转矩阵
    rot_mats = batch_rodrigues(full_pose)  # (batch_size, 24, 3, 3)
    
    # 计算 pose feature (排除根关节)
    pose_feature = (rot_mats[:, 1:, :, :] - torch.eye(3, device=device).unsqueeze(0).unsqueeze(0)).view(batch_size, -1)
    
    # 计算姿态偏移
    pose_offsets = torch.matmul(pose_feature, posedirs.view(-1, num_vertices * 3).T).view(batch_size, num_vertices, 3)
    
    # 姿态校正后的顶点
    v_posed = v_shaped + pose_offsets
    
    # (d) 线性混合蒙皮
    # 计算关节全局变换
    J_transformed, A = batch_rigid_transform(rot_mats, J, parents)
    
    # 应用蒙皮权重
    W = lbs_weights.unsqueeze(0).expand(batch_size, -1, -1)  # (batch_size, num_vertices, num_joints)
    T = torch.matmul(W, A.view(batch_size, num_joints, -1)).view(batch_size, num_vertices, 4, 4)
    
    # 齐次坐标变换
    v_posed_homo = torch.cat([v_posed, torch.ones(batch_size, num_vertices, 1, device=device)], dim=-1)
    v_homo = torch.matmul(T, v_posed_homo.unsqueeze(-1))
    verts = v_homo[:, :, :3, 0]
    
    return {
        'v_template': v_template,
        'v_shaped': v_shaped,
        'J': J,
        'v_posed': v_posed,
        'pose_offsets': pose_offsets,
        'verts': verts,
        'J_transformed': J_transformed
    }