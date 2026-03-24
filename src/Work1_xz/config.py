import taichi as ti
v0=ti.Vector([1.0,0.0,-1.0])
v1=ti.Vector([0.0,1.0,-1.0])
v2=ti.Vector([-1.0,0.0,-1.0])
v3=ti.Vector([0.0,-1.0,-1.0])
v4=ti.Vector([1.0,0.0,1.0])
v5=ti.Vector([0.0,1.0,1.0])
v6=ti.Vector([-1.0,0.0,1.0])
v7=ti.Vector([0.0,-1.0,1.0])
point=[v0,v1,v2,v3,v4,v5,v6,v7]
eye_position=ti.Vector([0.0,0.0,5])
cam_center=ti.Vector([0.0,1.0,0.0])
up_vector=ti.Vector([0.0,1.0,0.0])
res=(700,700)