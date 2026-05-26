# physics.py
import torch
from pytorch3d.structures import Meshes
from pytorch3d.loss import mesh_laplacian_smoothing, mesh_normal_consistency

def laplacian_smoothing_loss(mesh: Meshes, method: str = "uniform"):
    return mesh_laplacian_smoothing(mesh, method=method)

def edge_length_loss(mesh: Meshes, target_length: float = None):
    verts = mesh.verts_packed()
    edges = mesh.edges_packed()
    if edges.numel() == 0:
        return torch.tensor(0.0, device=mesh.device)
    v0 = verts[edges[:, 0]]
    v1 = verts[edges[:, 1]]
    edge_lengths = torch.norm(v1 - v0, dim=1)
    if target_length is None:
        target_length = edge_lengths.mean().detach()
    loss = ((edge_lengths - target_length) ** 2).mean()
    return loss

def normal_consistency_loss(mesh: Meshes):
    return mesh_normal_consistency(mesh)

def regularization_loss(mesh, w_lap, w_edge, w_norm, target_edge_length):
    lap_loss = laplacian_smoothing_loss(mesh)
    edge_loss = edge_length_loss(mesh, target_edge_length)
    norm_loss = normal_consistency_loss(mesh)
    total = w_lap * lap_loss + w_edge * edge_loss + w_norm * norm_loss
    return total, lap_loss, edge_loss, norm_loss