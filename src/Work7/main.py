import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
import trimesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import smplx

from config import *
from lbs import lbs as my_lbs

# 创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_smpl_model(model_dir):
    """加载 SMPL 模型"""
    # 确保 smpl 子目录存在
    smpl_dir = os.path.join(model_dir, 'smpl')
    if not os.path.exists(smpl_dir):
        os.makedirs(smpl_dir)
        print(f"注意：已创建目录 {smpl_dir}")
        print("请将 SMPL_NEUTRAL.pkl 复制到该目录下")
    
    model = smplx.create(
        model_dir,
        model_type='smpl',
        gender='neutral',
        use_pca=False,
        num_betas=10
    )
    return model

def plot_mesh(ax, vertices, faces, color=None, title=None):
    """绘制网格"""
    # 确保 faces 是 numpy 数组
    faces_np = np.array(faces) if not isinstance(faces, np.ndarray) else faces
    
    kwargs = {
        'triangles': faces_np,
        'alpha': 0.9,
        'edgecolor': 'none'
    }
    if color is not None:
        kwargs['color'] = color
    ax.plot_trisurf(
        vertices[:, 0],
        vertices[:, 1],
        vertices[:, 2],
        **kwargs
    )
    
    # 设置坐标轴范围
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    
    if title:
        ax.set_title(title)
    ax.set_axis_off()
    ax.view_init(elev=90, azim=-90)  # 正面站立视图
    ax.dist = 7.5

def plot_joints(ax, joints, color='red', marker='o', markersize=5):
    """绘制关节点"""
    ax.scatter(
        joints[:, 0],
        joints[:, 1],
        joints[:, 2],
        color=color,
        marker=marker,
        s=markersize,
        depthshade=False,
        zorder=10
    )

def visualize_weights(ax, vertices, faces, weights, joint_idx, joint_name):
    """可视化单个关节的权重热力图"""
    weights_joint = weights[:, joint_idx]
    
    # 归一化权重
    weights_norm = (weights_joint - weights_joint.min()) / (weights_joint.max() - weights_joint.min() + 1e-8)
    
    # 使用热力图颜色
    colors = plt.cm.viridis(weights_norm)
    
    # 确保 faces 是 numpy 数组
    faces_np = np.array(faces) if not isinstance(faces, np.ndarray) else faces
    
    ax.plot_trisurf(
        vertices[:, 0],
        vertices[:, 1],
        vertices[:, 2],
        triangles=faces_np,
        alpha=0.9,
        edgecolor='none'
    )
    # 设置面片颜色
    poly_collection = ax.collections[-1]
    poly_collection.set_facecolors(colors)
    
    # 设置坐标轴范围
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    
    ax.set_title(f'Joint {joint_name} Weights')
    ax.set_axis_off()
    ax.view_init(elev=90, azim=-90)  # 正面站立视图
    ax.dist = 7.5

def visualize_dominant_joint(ax, vertices, faces, weights):
    """可视化每个顶点的主导关节"""
    dominant_joint = np.argmax(weights, axis=1)
    
    # 为每个关节分配颜色
    num_joints = weights.shape[1]
    colors = plt.cm.tab20(np.linspace(0, 1, num_joints))
    
    # 计算每个顶点的主导权重强度
    max_weights = np.max(weights, axis=1)
    
    # 确保 faces 是 numpy 数组
    faces_np = np.array(faces) if not isinstance(faces, np.ndarray) else faces
    
    # 为每个面片分配颜色（基于顶点的主导关节）
    face_colors = np.zeros((faces_np.shape[0], 4))
    for i, face in enumerate(faces_np):
        # 使用第一个顶点的主导关节
        joint_idx = dominant_joint[face[0]]
        face_colors[i] = colors[joint_idx]
        # 根据权重强度调整透明度
        face_colors[i, 3] = max_weights[face[0]]
    
    ax.plot_trisurf(
        vertices[:, 0],
        vertices[:, 1],
        vertices[:, 2],
        triangles=faces_np,
        alpha=0.9,
        edgecolor='none'
    )
    # 设置面片颜色
    poly_collection = ax.collections[-1]
    poly_collection.set_facecolors(face_colors)
    
    # 设置坐标轴范围
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    
    ax.set_title('Dominant Joint per Vertex')
    ax.set_axis_off()
    ax.view_init(elev=90, azim=-90)  # 正面站立视图
    ax.dist = 7.5
    ax.set_axis_off()
    ax.view_init(elev=90, azim=-90)

def visualize_pose_offsets(ax, vertices, faces, pose_offsets):
    """可视化姿态偏移的大小"""
    offset_magnitude = np.linalg.norm(pose_offsets, axis=1)
    offset_norm = (offset_magnitude - offset_magnitude.min()) / (offset_magnitude.max() - offset_magnitude.min() + 1e-8)
    
    colors = plt.cm.coolwarm(offset_norm)
    
    ax.plot_trisurf(
        vertices[:, 0],
        vertices[:, 1],
        vertices[:, 2],
        triangles=faces,
        alpha=0.9,
        edgecolor='none'
    )
    # 设置面片颜色
    poly_collection = ax.collections[-1]
    poly_collection.set_facecolors(colors)
    ax.set_title('Pose Offsets Magnitude')
    ax.set_axis_off()
    ax.view_init(elev=90, azim=-90)  # 正面站立视图
    ax.dist = 7.5

def task1_load_smpl(model):
    """任务1：加载SMPL并输出基础信息"""
    print("=" * 60)
    print("任务1：SMPL模型基础信息")
    print("=" * 60)
    print(f"顶点数: {model.v_template.shape[0]}")
    print(f"面片数: {model.faces.shape[0]}")
    print(f"关节数: {model.J_regressor.shape[0]}")
    print(f"betas维度: {model.num_betas}")
    print(f"蒙皮权重形状: {model.lbs_weights.shape}")
    print(f"形状形变基形状: {model.shapedirs.shape}")
    print(f"姿态形变基形状: {model.posedirs.shape}")
    print("=" * 60)

def task2_visualize_template_weights(model):
    """任务2：可视化模板网格与蒙皮权重"""
    vertices = model.v_template.detach().cpu().numpy()
    faces = model.faces
    weights = model.lbs_weights.detach().cpu().numpy()
    
    # (1) 单关节权重热力图 - 选择左肩关节
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    visualize_weights(ax, vertices, faces, weights, 16, 'left_shoulder')
    plt.savefig(os.path.join(OUTPUT_DIR, 'stage_a_template_weights.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # (2) 全关节主导权重分布图
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    visualize_dominant_joint(ax, vertices, faces, weights)
    plt.savefig(os.path.join(OUTPUT_DIR, 'all_joint_weights.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print("任务2完成：模板网格与蒙皮权重可视化已保存")

def task3_visualize_shape_joints(model, betas):
    """任务3：可视化形状校正与关节回归"""
    # 使用官方模型前向获取形状校正后的顶点
    model.eval()
    with torch.no_grad():
        output = model(betas=betas)
    
    v_shaped = output.vertices.detach().cpu().numpy()[0]
    J = output.joints.detach().cpu().numpy()[0]
    
    faces = model.faces
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 确保 faces 是 numpy 数组
    faces_np = np.array(faces) if not isinstance(faces, np.ndarray) else faces
    
    # 绘制网格
    ax.plot_trisurf(
        v_shaped[:, 0],
        v_shaped[:, 1],
        v_shaped[:, 2],
        triangles=faces_np,
        color='blue',
        alpha=0.9,
        edgecolor='none'
    )
    
    # 绘制关节点
    ax.scatter(
        J[:, 0],
        J[:, 1],
        J[:, 2],
        color='red',
        marker='o',
        s=10,
        depthshade=False,
        zorder=10
    )
    
    # 设置坐标轴范围
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    
    ax.set_title('Shape Corrected Mesh with Joints')
    ax.set_axis_off()
    ax.view_init(elev=90, azim=-90)  # 正面站立视图
    ax.dist = 7.5
    
    plt.savefig(os.path.join(OUTPUT_DIR, 'stage_b_shaped_joints.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print("任务3完成：形状校正与关节回归可视化已保存")

def task4_visualize_pose_offsets(model, betas, global_orient, body_pose):
    """任务4：可视化姿态校正"""
    model.eval()
    with torch.no_grad():
        # 获取形状校正后的顶点
        shape_output = model(betas=betas)
        v_shaped = shape_output.vertices
        
        # 获取完整姿态下的顶点
        full_output = model(betas=betas, global_orient=global_orient, body_pose=body_pose)
        v_posed = full_output.vertices
        
        # 计算姿态偏移（完整姿态顶点 - 形状校正顶点）
        pose_offsets = v_posed - v_shaped
    
    v_posed_np = v_posed.detach().cpu().numpy()[0]
    pose_offsets_np = pose_offsets.detach().cpu().numpy()[0]
    faces = model.faces
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    visualize_pose_offsets(ax, v_posed_np, faces, pose_offsets_np)
    plt.savefig(os.path.join(OUTPUT_DIR, 'stage_c_pose_offsets.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print("任务4完成：姿态校正可视化已保存")

def task5_visualize_lbs_result(model, betas, global_orient, body_pose):
    """任务5：可视化完整LBS结果"""
    model.eval()
    with torch.no_grad():
        output = model(betas=betas, global_orient=global_orient, body_pose=body_pose)
    
    verts = output.vertices.detach().cpu().numpy()[0]
    J_transformed = output.joints.detach().cpu().numpy()[0]
    faces = model.faces
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    plot_mesh(ax, verts, faces, color='green', title=None)
    plot_joints(ax, J_transformed, color='red', markersize=10)
    ax.set_title('Final LBS Result')
    plt.savefig(os.path.join(OUTPUT_DIR, 'stage_d_lbs_result.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print("任务5完成：完整LBS结果可视化已保存")

def task6_comparison_grid(model, betas, global_orient, body_pose):
    """任务6：生成总对比图"""
    model.eval()
    
    # 确保 faces 是 numpy 数组
    faces = np.array(model.faces) if not isinstance(model.faces, np.ndarray) else model.faces
    
    with torch.no_grad():
        # 获取所有阶段的结果
        v_template = model.v_template.detach().cpu().numpy()
        
        # 获取形状校正后的顶点和关节
        shape_output = model(betas=betas)
        v_shaped_np = shape_output.vertices.detach().cpu().numpy()[0]
        J = shape_output.joints.detach().cpu().numpy()[0]
        
        # 获取姿态校正后的顶点
        full_output = model(betas=betas, global_orient=global_orient, body_pose=body_pose)
        v_posed_np = full_output.vertices.detach().cpu().numpy()[0]
        
        # 获取最终蒙皮结果
        verts_np = full_output.vertices.detach().cpu().numpy()[0]
        J_transformed_np = full_output.joints.detach().cpu().numpy()[0]
    
    # 创建 2x2 对比图
    fig = plt.figure(figsize=(16, 12))
    
    # (a) Template + weights
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    weights = model.lbs_weights.detach().cpu().numpy()
    weights_joint = weights[:, 16]  # 左肩关节
    weights_norm = (weights_joint - weights_joint.min()) / (weights_joint.max() - weights_joint.min() + 1e-8)
    colors = plt.cm.viridis(weights_norm)
    
    ax1.plot_trisurf(v_template[:, 0], v_template[:, 1], v_template[:, 2], triangles=faces, alpha=0.9)
    ax1.collections[-1].set_facecolors(colors)
    ax1.set_xlim(-1, 1)
    ax1.set_ylim(-1, 1)
    ax1.set_zlim(-1, 1)
    ax1.set_title('(a) Template + Weights')
    ax1.set_axis_off()
    ax1.view_init(elev=90, azim=-90)  # 正面站立视图
    ax1.dist = 7.5
    
    # (b) Shape + joints
    ax2 = fig.add_subplot(2, 2, 2, projection='3d')
    ax2.plot_trisurf(v_shaped_np[:, 0], v_shaped_np[:, 1], v_shaped_np[:, 2], triangles=faces, color='blue', alpha=0.9)
    ax2.scatter(J[:, 0], J[:, 1], J[:, 2], color='red', s=10, depthshade=False, zorder=10)
    ax2.set_xlim(-1, 1)
    ax2.set_ylim(-1, 1)
    ax2.set_zlim(-1, 1)
    ax2.set_title('(b) Shape + Joints')
    ax2.set_axis_off()
    ax2.view_init(elev=90, azim=-90)  # 正面站立视图
    ax2.dist = 7.5
    
    # (c) Pose offsets
    ax3 = fig.add_subplot(2, 2, 3, projection='3d')
    # 计算姿态偏移
    v_shaped = shape_output.vertices
    pose_offsets_np = (full_output.vertices - v_shaped).detach().cpu().numpy()[0]
    offset_magnitude = np.linalg.norm(pose_offsets_np, axis=1)
    offset_norm = (offset_magnitude - offset_magnitude.min()) / (offset_magnitude.max() - offset_magnitude.min() + 1e-8)
    colors_c = plt.cm.coolwarm(offset_norm)
    
    ax3.plot_trisurf(v_posed_np[:, 0], v_posed_np[:, 1], v_posed_np[:, 2], triangles=faces, alpha=0.9)
    ax3.collections[-1].set_facecolors(colors_c)
    ax3.set_xlim(-1, 1)
    ax3.set_ylim(-1, 1)
    ax3.set_zlim(-1, 1)
    ax3.set_title('(c) Pose Offsets')
    ax3.set_axis_off()
    ax3.view_init(elev=90, azim=-90)  # 正面站立视图
    ax3.dist = 7.5
    
    # (d) Final skinned mesh
    ax4 = fig.add_subplot(2, 2, 4, projection='3d')
    ax4.plot_trisurf(verts_np[:, 0], verts_np[:, 1], verts_np[:, 2], triangles=faces, color='green', alpha=0.9)
    ax4.scatter(J_transformed_np[:, 0], J_transformed_np[:, 1], J_transformed_np[:, 2], color='red', s=10, depthshade=False, zorder=10)
    ax4.set_xlim(-1, 1)
    ax4.set_ylim(-1, 1)
    ax4.set_zlim(-1, 1)
    ax4.set_title('(d) Final Skinned Mesh')
    ax4.set_axis_off()
    ax4.view_init(elev=90, azim=-90)  # 正面站立视图
    ax4.dist = 7.5
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'comparison_grid.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print("任务6完成：总对比图已保存")
    return
    
    # 以下是原来的代码（已被替换）
    ax1.plot_trisurf(v_template[:, 0], v_template[:, 1], v_template[:, 2], triangles=faces, alpha=0.9)
    ax1.collections[-1].set_facecolors(colors)
    ax1.plot_trisurf(v_template[:, 0], v_template[:, 1], v_template[:, 2], triangles=faces, alpha=0.9)
    ax1.collections[-1].set_facecolors(colors)
    ax1.set_title('(a) Template + Weights')
    ax1.set_axis_off()
    ax1.view_init(elev=20, azim=-45)
    
    # (b) Shape + joints
    ax2 = fig.add_subplot(2, 2, 2, projection='3d')
    ax2.plot_trisurf(v_shaped_np[:, 0], v_shaped_np[:, 1], v_shaped_np[:, 2], triangles=faces, color='blue', alpha=0.9)
    ax2.scatter(J[:, 0], J[:, 1], J[:, 2], color='red', s=10, depthshade=False)
    ax2.set_title('(b) Shape + Joints')
    ax2.set_axis_off()
    ax2.view_init(elev=20, azim=-45)
    
    # (c) Pose offsets
    ax3 = fig.add_subplot(2, 2, 3, projection='3d')
    pose_offsets_np = (v_posed - shape_output.vertices).detach().cpu().numpy()[0]
    offset_magnitude = np.linalg.norm(pose_offsets_np, axis=1)
    offset_norm = (offset_magnitude - offset_magnitude.min()) / (offset_magnitude.max() - offset_magnitude.min() + 1e-8)
    colors = plt.cm.coolwarm(offset_norm)
    ax3.plot_trisurf(v_posed_np[:, 0], v_posed_np[:, 1], v_posed_np[:, 2], triangles=faces, alpha=0.9)
    ax3.collections[-1].set_facecolors(colors)
    ax3.set_title('(c) Pose Offsets')
    ax3.set_axis_off()
    ax3.view_init(elev=20, azim=-45)
    
    # (d) Final skinned mesh
    ax4 = fig.add_subplot(2, 2, 4, projection='3d')
    ax4.plot_trisurf(verts_np[:, 0], verts_np[:, 1], verts_np[:, 2], triangles=faces, color='green', alpha=0.9)
    ax4.scatter(J_transformed_np[:, 0], J_transformed_np[:, 1], J_transformed_np[:, 2], color='red', s=10, depthshade=False)
    ax4.set_title('(d) Final Skinned Mesh')
    ax4.set_axis_off()
    ax4.view_init(elev=20, azim=-45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'comparison_grid.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print("任务6完成：总对比图已保存")

def task7_verify_consistency(model, betas, global_orient, body_pose):
    """任务7：手写LBS与官方前向结果一致性验证"""
    model.eval()
    
    # 获取模型参数
    v_template = model.v_template
    shapedirs = model.shapedirs
    posedirs = model.posedirs
    J_regressor = model.J_regressor
    parents = model.parents
    lbs_weights = model.lbs_weights
    
    batch_size = betas.shape[0]
    num_vertices = v_template.shape[0]
    num_joints = J_regressor.shape[0]
    device = betas.device
    
    # 手写 LBS 实现
    with torch.no_grad():
        # (a) 模板顶点
        v_template_expanded = v_template.unsqueeze(0).expand(batch_size, -1, -1)
        
        # (b) 形状校正
        # shapedirs: (num_vertices, 3, num_betas)
        # betas: (batch_size, num_betas)
        # shape_offsets: (batch_size, num_vertices, 3)
        # 使用 torch.bmm 进行批量矩阵乘法
        # 先将 shapedirs 转换为 (num_betas, num_vertices*3)
        shapedirs_reshaped = shapedirs.permute(2, 0, 1).reshape(shapedirs.shape[2], -1)  # [10, 20670]
        shape_offsets = torch.matmul(betas, shapedirs_reshaped)  # [1, 20670]
        shape_offsets = shape_offsets.view(batch_size, num_vertices, 3)  # [1, 6890, 3]
        v_shaped = v_template_expanded + shape_offsets
        
        # 回归关节位置
        # J_regressor: (num_joints, num_vertices)
        # v_shaped: (batch_size, num_vertices, 3)
        # J: (batch_size, num_joints, 3)
        if J_regressor.is_sparse:
            J_regressor_dense = J_regressor.to_dense()
        else:
            J_regressor_dense = J_regressor
        
        # 确保 J_regressor 的形状正确
        # 如果形状是 (num_vertices, num_joints)，需要转置
        if J_regressor_dense.shape[0] == num_vertices:
            J_regressor_dense = J_regressor_dense.T
        
        # 使用矩阵乘法计算关节位置
        # J_regressor_dense: [num_joints, num_vertices]
        # v_shaped: [batch_size, num_vertices, 3]
        # J: [batch_size, num_joints, 3]
        J = torch.matmul(J_regressor_dense.unsqueeze(0), v_shaped)
        
        # (c) 姿态校正
        full_pose = torch.cat([global_orient, body_pose], dim=1)
        full_pose = full_pose.view(batch_size, num_joints, 3)
        
        # 转换为旋转矩阵
        from lbs import batch_rodrigues
        rot_mats = batch_rodrigues(full_pose)
        
        # 计算 pose feature (排除根关节)
        # rot_mats[:, 1:, :, :] -> [batch_size, 23, 3, 3]
        # pose_feature -> [batch_size, 207]
        pose_feature = (rot_mats[:, 1:, :, :] - torch.eye(3, device=device).unsqueeze(0).unsqueeze(0))
        pose_feature = pose_feature.view(batch_size, -1)
        
        # 计算姿态偏移
        # posedirs shape: [207, 20670]
        # pose_feature: [batch_size, 207]
        # pose_offsets: (batch_size, 20670) -> (batch_size, num_vertices, 3)
        # 正确的计算方式：pose_feature * posedirs
        # pose_feature: [batch_size, 207]
        # posedirs: [207, 20670]
        # result: [batch_size, 20670]
        pose_offsets = torch.matmul(pose_feature, posedirs)
        pose_offsets = pose_offsets.view(batch_size, num_vertices, 3)
        
        # 姿态校正后的顶点
        v_posed = v_shaped + pose_offsets
        
        # (d) 线性混合蒙皮
        # 计算关节全局变换
        from lbs import batch_rigid_transform
        J_transformed, A = batch_rigid_transform(rot_mats, J, parents)
        
        # 应用蒙皮权重
        # W: (batch_size, num_vertices, num_joints)
        # A: (batch_size, num_joints, 4, 4)
        # T: (batch_size, num_vertices, 4, 4)
        W = lbs_weights.unsqueeze(0).expand(batch_size, -1, -1)
        
        # 使用正确的方式计算蒙皮
        # 将 A 展平为 [batch_size, num_joints, 16]
        A_flat = A.view(batch_size, num_joints, 16)
        # W: [batch_size, num_vertices, num_joints]
        # T_flat: [batch_size, num_vertices, 16]
        T_flat = torch.bmm(W, A_flat)
        # T: [batch_size, num_vertices, 4, 4]
        T = T_flat.view(batch_size, num_vertices, 4, 4)
        
        # 齐次坐标变换
        v_posed_homo = torch.cat([v_posed, torch.ones(batch_size, num_vertices, 1, device=device)], dim=-1)
        v_homo = torch.matmul(T, v_posed_homo.unsqueeze(-1))
        my_verts = v_homo[:, :, :3, 0]
    
    # 官方前向
    with torch.no_grad():
        official_output = model(betas=betas, global_orient=global_orient, body_pose=body_pose)
        official_verts = official_output.vertices
    
    # 计算误差
    diff = my_verts - official_verts
    mae = torch.mean(torch.abs(diff)).item()
    max_ae = torch.max(torch.abs(diff)).item()
    
    # 打印误差（用于调试）
    print(f"\n调试信息 - 手写LBS与官方结果差异:")
    print(f"MAE: {mae}")
    print(f"Max AE: {max_ae}")
    print(f"差异的均值: {torch.mean(diff).item()}")
    print(f"差异的标准差: {torch.std(diff).item()}")
    
    # 检查中间结果
    print(f"\n中间结果检查:")
    print(f"v_template 形状: {v_template.shape}")
    print(f"shapedirs 形状: {shapedirs.shape}")
    print(f"posedirs 形状: {posedirs.shape}")
    print(f"J_regressor 形状: {J_regressor.shape}")
    print(f"lbs_weights 形状: {lbs_weights.shape}")
    print(f"v_shaped 形状: {v_shaped.shape}")
    print(f"J 形状: {J.shape}")
    print(f"rot_mats 形状: {rot_mats.shape}")
    print(f"pose_feature 形状: {pose_feature.shape}")
    print(f"pose_offsets 形状: {pose_offsets.shape}")
    print(f"v_posed 形状: {v_posed.shape}")
    print(f"my_verts 形状: {my_verts.shape}")
    print(f"official_verts 形状: {official_verts.shape}")
    
    # 保存结果
    with open(os.path.join(OUTPUT_DIR, 'summary.txt'), 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("LBS 手写实现与官方结果一致性验证\n")
        f.write("=" * 60 + "\n")
        f.write(f"平均绝对误差 (MAE): {mae:.10f}\n")
        f.write(f"最大绝对误差 (Max AE): {max_ae:.10f}\n")
        f.write("=" * 60 + "\n")
        f.write("\n验证结论：\n")
        if mae < 1e-5 and max_ae < 1e-4:
            f.write("✓ 手写 LBS 实现与官方结果一致！\n")
        else:
            f.write("✗ 手写 LBS 实现与官方结果存在差异\n")
    
    print("=" * 60)
    print("任务7：LBS一致性验证")
    print("=" * 60)
    print(f"平均绝对误差 (MAE): {mae:.10f}")
    print(f"最大绝对误差 (Max AE): {max_ae:.10f}")
def main():
    # 检查模型目录是否存在
    smpl_path = os.path.join(MODEL_DIR, 'smpl', 'SMPL_NEUTRAL.pkl')
    if not os.path.exists(smpl_path):
        print(f"错误：SMPL模型文件不存在于 {smpl_path}")
        print("请从师大云盘下载 SMPL_NEUTRAL.pkl 文件并放置到该位置")
        return
    
    # 加载模型
    print("正在加载 SMPL 模型...")
    model = load_smpl_model(MODEL_DIR)
    
    # 任务1：输出基础信息
    task1_load_smpl(model)
    
    # 准备输入参数
    betas = torch.tensor([SHAPE_BETAS], dtype=torch.float32)
    global_orient = torch.tensor([POSE_GLOBAL_ORIENT], dtype=torch.float32)
    body_pose = torch.tensor([POSE_BODY_POSE], dtype=torch.float32)
    
    # 任务2：可视化模板网格与蒙皮权重
    task2_visualize_template_weights(model)
    
    # 任务3：可视化形状校正与关节回归
    task3_visualize_shape_joints(model, betas)
    
    # 任务4：可视化姿态校正
    task4_visualize_pose_offsets(model, betas, global_orient, body_pose)
    
    # 任务5：可视化完整LBS结果
    task5_visualize_lbs_result(model, betas, global_orient, body_pose)
    
    # 任务6：生成总对比图
    task6_comparison_grid(model, betas, global_orient, body_pose)
    
    # 任务7：一致性验证
    task7_verify_consistency(model, betas, global_orient, body_pose)
    
    print("\n所有任务已完成！输出文件保存在 outputs/ 目录中。")

if __name__ == '__main__':
    main()