import taichi as ti
from .config import *

@ti.kernel
def get_model_matrix(angle:float)->ti.types.matrix(4, 4, float):
    true_angle=angle*ti.math.pi/180
    c=ti.cos(true_angle)
    s=ti.sin(true_angle)
    model=ti.Matrix([
        [c,-s,0.0,0.0],
        [s,c,0.0,0.0],
        [0.0,0.0,1.0,0.0],
        [0.0,0.0,0.0,1.0]
    ])
    return model

@ti.kernel
def get_view_matrix(eye_pos:ti.types.vector(3,float))->ti.types.matrix(4, 4, float):
    fwd=(cam_center-eye_pos).normalized()
    right=fwd.cross(up_vector).normalized()
    up=right.cross(fwd).normalized()
    view=ti.Matrix([
        [right[0],right[1],right[2],-right.dot(eye_pos)],
        [up[0],up[1],up[2],-up.dot(eye_pos)],
        [-fwd[0],-fwd[1],-fwd[2],fwd.dot(eye_pos)],
        [0.0,0.0,0.0,1.0]
    ])
    return view

@ti.kernel
def get_projection_matrix(eye_fov:float,aspect_ratio:float,zNear:float,zFar:float)->ti.types.matrix(4, 4, float):
    true_eye=eye_fov*ti.math.pi/180
    f=1.0/ti.tan(true_eye/2.0)
    perspective = ti.Matrix([
        [f/aspect_ratio,0.0,0.0,0.0],
        [0.0,f,0.0,0.0],
        [0.0,0.0,-(zFar+zNear)/(zFar-zNear),-2.0*zFar*zNear/(zFar-zNear)],
        [0.0,0.0,-1.0,0.0]
    ])
    return perspective