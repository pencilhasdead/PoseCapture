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
