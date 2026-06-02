# SMPL LBS 实验配置文件

# 模型路径配置
# smplx.create() 需要的是包含模型文件的目录路径
# SMPL_NEUTRAL.pkl 应放在 MODEL_DIR/smpl/SMPL_NEUTRAL.pkl
MODEL_DIR = './models'

# 输出目录
OUTPUT_DIR = './outputs'

# 可视化参数
CAMERA_POSITION = [0, 1.5, 3.0]
CAMERA_LOOKAT = [0, 0, 0]

# 形状参数（用于任务3-7）
SHAPE_BETAS = [1.0, 0.5, -0.3, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0]

# 姿态参数（用于任务4-7）
# global_orient: 全局方向（3维）- 控制根关节(pelvis)的全局旋转
# body_pose: 身体姿态（23*3=69维）- 控制其他23个关节
POSE_GLOBAL_ORIENT = [0.0, 0.2, 0.1]  # 轻微旋转
POSE_BODY_POSE = [
    # left_hip, right_hip, spine1
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0,
    0.5, 0.3, 0.1,  # spine1 - 躯干轻微弯曲
    # left_knee, right_knee, spine2
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0,
    # left_ankle, right_ankle, spine3
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0,
    # left_foot, right_foot, neck
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0,
    # left_collar, right_collar, head
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0,
    # left_shoulder, right_shoulder, left_elbow
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0,
    1.0, 0.0, 0.0,  # left_elbow - 左肘弯曲
    # right_elbow, left_wrist, right_wrist
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0,
    # left_hand, right_hand
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0,
]

# 关节名称（用于权重可视化）
JOINT_NAMES = [
    'pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee',
    'spine2', 'left_ankle', 'right_ankle', 'spine3', 'left_foot', 'right_foot',
    'neck', 'left_collar', 'right_collar', 'head', 'left_shoulder', 'right_shoulder',
    'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hand', 'right_hand'
]