"""COCO-WholeBody 133 关键点定义（DWPose / Easy-DWPose 的输出格式）。

索引顺序:
  0-16   身体 17 点
  17-22  脚 6 点
  23-90  人脸 68 点
  91-111 左手 21 点
  112-132 右手 21 点
"""

from __future__ import annotations

BODY_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

FOOT_NAMES = [
    "left_big_toe",
    "left_small_toe",
    "left_heel",
    "right_big_toe",
    "right_small_toe",
    "right_heel",
]

HAND_NAMES = [
    "wrist",
    "thumb_cmc",
    "thumb_mcp",
    "thumb_ip",
    "thumb_tip",
    "index_mcp",
    "index_pip",
    "index_dip",
    "index_tip",
    "middle_mcp",
    "middle_pip",
    "middle_dip",
    "middle_tip",
    "ring_mcp",
    "ring_pip",
    "ring_dip",
    "ring_tip",
    "pinky_mcp",
    "pinky_pip",
    "pinky_dip",
    "pinky_tip",
]

KEYPOINT_NAMES = (
    BODY_NAMES
    + FOOT_NAMES
    + ["face_%02d" % i for i in range(68)]
    + ["left_hand_%s" % n for n in HAND_NAMES]
    + ["right_hand_%s" % n for n in HAND_NAMES]
)

TOTAL = 133
assert len(KEYPOINT_NAMES) == TOTAL

IDX = {name: i for i, name in enumerate(KEYPOINT_NAMES)}

# 身体
NOSE = 0
LEFT_EYE, RIGHT_EYE = 1, 2
LEFT_EAR, RIGHT_EAR = 3, 4
LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6
LEFT_ELBOW, RIGHT_ELBOW = 7, 8
LEFT_WRIST, RIGHT_WRIST = 9, 10
LEFT_HIP, RIGHT_HIP = 11, 12
LEFT_KNEE, RIGHT_KNEE = 13, 14
LEFT_ANKLE, RIGHT_ANKLE = 15, 16

# 脚
LEFT_BIG_TOE, LEFT_SMALL_TOE, LEFT_HEEL = 17, 18, 19
RIGHT_BIG_TOE, RIGHT_SMALL_TOE, RIGHT_HEEL = 20, 21, 22

FACE_START = 23
FACE_COUNT = 68
HAND_LEFT_START = 91
HAND_RIGHT_START = 112
HAND_COUNT = 21

# 68 点人脸里常用的几个位置（iBUG 300W 布局）
FACE_CHIN = FACE_START + 8
FACE_EYE_L_END = FACE_START + 36  # 画面左侧眼睛的外眼角
FACE_EYE_R_END = FACE_START + 45  # 画面右侧眼睛的外眼角
FACE_NOSE_TIP = FACE_START + 30
FACE_BROWS = list(range(FACE_START + 17, FACE_START + 27))
FACE_ALL = list(range(FACE_START, FACE_START + FACE_COUNT))

# 五官分组（iBUG 300W / dlib 布局）：头 / 脖子的朝向由这些点投票出来，见 FACE_SIDE_PAIRS /
# FACE_UP_PAIRS。名字里的 l / r 都是**画面**左右，与 FACE_EYE_L_END 的注释同一套约定
# （画面左侧 = 人物的右侧）—— 所以成对使用时的方向是“画面左 -> 画面右”，它是不是人物的
# “右 -> 左”取决于人物朝向，reconstruction 里会按肩线把符号统一过来。
FACE_EYE_L = list(range(FACE_START + 36, FACE_START + 42))    # 画面左侧眼睛（人物右眼）
FACE_EYE_R = list(range(FACE_START + 42, FACE_START + 48))    # 画面右侧眼睛（人物左眼）
FACE_BROW_R = list(range(FACE_START + 17, FACE_START + 22))   # 画面右侧眉毛
FACE_BROW_L = list(range(FACE_START + 22, FACE_START + 27))   # 画面左侧眉毛
FACE_NOSE_BRIDGE = list(range(FACE_START + 27, FACE_START + 31))
FACE_NOSE_BOTTOM = list(range(FACE_START + 31, FACE_START + 36))
FACE_MOUTH_OUTER = list(range(FACE_START + 48, FACE_START + 60))
FACE_MOUTH_INNER = list(range(FACE_START + 60, FACE_START + 68))
FACE_MOUTH = FACE_MOUTH_OUTER + FACE_MOUTH_INNER

# 五官点组（键名 -> 68 点里的索引；长度 1 就是单个点）。eye_mid / brow_mid 不在表里 ——
# 它们是 eye_l/eye_r、brow_l/brow_r 的中点，由 reconstruction._face_frame 现算。
FACE_GROUPS = {
    "chin": (FACE_CHIN,),
    "nose_bridge": tuple(FACE_NOSE_BRIDGE),
    "nose_bottom": tuple(FACE_NOSE_BOTTOM),
    "nose_tip": (FACE_NOSE_TIP,),
    "eye_l": tuple(FACE_EYE_L),
    "eye_r": tuple(FACE_EYE_R),
    "eye_l_inner": (FACE_START + 39,),
    "eye_r_inner": (FACE_START + 42,),
    "eye_l_outer": (FACE_EYE_L_END,),
    "eye_r_outer": (FACE_EYE_R_END,),
    "brow_l": tuple(FACE_BROW_L),
    "brow_r": tuple(FACE_BROW_R),
    "mouth_l": (FACE_START + 48,),
    "mouth_r": (FACE_START + 54,),
    "mouth_mid": tuple(FACE_MOUTH),
    "jaw_l": (FACE_START + 0,),
    "jaw_r": (FACE_START + 16,),
}

# 头部“侧轴”（眼线 / 左右方向）用的成对点：画面左 -> 画面右。权重 = 解剖上的可信度：
# 眼角最准（侧头时也还在脸上），眉梢还行，下颌角最次（它其实落在耳前，会被发型 / 侧头带跑）。
# 成对平均而不是只用两个外眼角：单点抖动（睫毛、嘴角）不会把 roll / 转头整体带歪。
FACE_SIDE_PAIRS = (
    ("eye_l_outer", "eye_r_outer", 1.0),
    ("eye_l", "eye_r", 1.0),
    ("eye_l_inner", "eye_r_inner", 0.9),
    ("mouth_l", "mouth_r", 0.8),
    ("brow_l", "brow_r", 0.6),
    ("jaw_l", "jaw_r", 0.4),
)

# 头部“上轴”（下巴 -> 头顶）用的成对点（下 -> 上）：下巴 -> 眼睛最长也最稳（成人 ≈ 11.5 cm），
# 其余几对用来投票降噪，顺便在没有下巴时也能给出方向。
FACE_UP_PAIRS = (
    ("chin", "eye_mid", 1.0),
    ("chin", "brow_mid", 0.9),
    ("mouth_mid", "eye_mid", 0.8),
    ("nose_tip", "brow_mid", 0.6),
    ("nose_bottom", "eye_mid", 0.5),
)


# 手指分组（相对手部 21 点局部索引）
HAND_GROUPS = {
    "thumb": [1, 2, 3, 4],
    "index": [5, 6, 7, 8],
    "middle": [9, 10, 11, 12],
    "ring": [13, 14, 15, 16],
    "pinky": [17, 18, 19, 20],
}
HAND_WRIST = 0
HAND_MIDDLE_MCP = 9
HAND_INDEX_MCP = 5
HAND_PINKY_MCP = 17

# 预览骨架连线（只用身体 + 脚，参考 OpenPose 画法）
BODY_LIMBS = [
    (NOSE, LEFT_EYE), (NOSE, RIGHT_EYE), (LEFT_EYE, LEFT_EAR), (RIGHT_EYE, RIGHT_EAR),
    (LEFT_SHOULDER, RIGHT_SHOULDER),
    (LEFT_SHOULDER, LEFT_ELBOW), (LEFT_ELBOW, LEFT_WRIST),
    (RIGHT_SHOULDER, RIGHT_ELBOW), (RIGHT_ELBOW, RIGHT_WRIST),
    (LEFT_SHOULDER, LEFT_HIP), (RIGHT_SHOULDER, RIGHT_HIP),
    (LEFT_HIP, RIGHT_HIP),
    (LEFT_HIP, LEFT_KNEE), (LEFT_KNEE, LEFT_ANKLE),
    (RIGHT_HIP, RIGHT_KNEE), (RIGHT_KNEE, RIGHT_ANKLE),
    (LEFT_ANKLE, LEFT_BIG_TOE), (LEFT_BIG_TOE, LEFT_SMALL_TOE),
    (RIGHT_ANKLE, RIGHT_BIG_TOE), (RIGHT_BIG_TOE, RIGHT_SMALL_TOE),
]

LIMB_COLORS = {
    (LEFT_SHOULDER, LEFT_ELBOW): (255, 80, 80),
    (LEFT_ELBOW, LEFT_WRIST): (255, 80, 80),
    (RIGHT_SHOULDER, RIGHT_ELBOW): (80, 80, 255),
    (RIGHT_ELBOW, RIGHT_WRIST): (80, 80, 255),
    (LEFT_HIP, LEFT_KNEE): (255, 160, 60),
    (LEFT_KNEE, LEFT_ANKLE): (255, 160, 60),
    (RIGHT_HIP, RIGHT_KNEE): (60, 220, 200),
    (RIGHT_KNEE, RIGHT_ANKLE): (60, 220, 200),
}


def hand_keypoint(index: int, side: str) -> int:
    """把某只手的局部索引(0-20)转换成全局索引。side: 'L' 或 'R'。"""
    base = HAND_LEFT_START if side.upper().startswith("L") else HAND_RIGHT_START
    return base + index
