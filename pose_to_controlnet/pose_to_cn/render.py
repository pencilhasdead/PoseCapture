"""骨骼图渲染。

两种画法：
  openpose    ControlNet 官方 annotator/openpose/util.py 的画法（默认）
              · 关节点：半径 4 的实心圆，颜色取自 18 色虹色带
              · 肢体：  长轴=长度/2、短轴=stickwidth(4) 的旋转椭圆，
                        与画布按 0.4/0.6 混合；只画前 17 条（官方 19 条里
                        循环只到 17，肩-耳那两条实际没画）
              · 手部：  20 条边按 HSV 色相取色、线宽 2，关节点红色半径 4
              · 人脸：  68 点白色实心圆，半径 3
  dwpose133   直接复用插件预览那套 133 点画法（黑色背景），便于逐点校对。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import core_loader

# --------------------------------------------------------------------------- 常量
# OpenPose COCO-18 关键点顺序（索引 0 = 鼻子，1 = 脖子）
OP18_NAMES = [
    "nose", "neck",
    "r_shoulder", "r_elbow", "r_wrist",
    "l_shoulder", "l_elbow", "l_wrist",
    "r_hip", "r_knee", "r_ankle",
    "l_hip", "l_knee", "l_ankle",
    "r_eye", "l_eye", "r_ear", "l_ear",
]

OP_LIMB_SEQ = [
    [2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10],
    [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17],
    [1, 16], [16, 18], [3, 17], [6, 18],
]

OP_COLORS = [
    (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), (170, 255, 0),
    (85, 255, 0), (0, 255, 0), (0, 255, 85), (0, 255, 170), (0, 255, 255),
    (0, 170, 255), (0, 85, 255), (0, 0, 255), (85, 0, 255), (170, 0, 255),
    (255, 0, 255), (255, 0, 170), (255, 0, 85),
]

HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (0, 9),
    (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]

HAND_POINT_COLOR = (255, 0, 0)      # 官方是 cv2.circle(..., (0,0,255)) = BGR 红
FACE_POINT_COLOR = (255, 255, 255)

# DWPose(COCO-133) -> OpenPose-18 的索引映射（脖子由双肩中点求得）
OP18_FROM_COCO = (
    0,                      # 0  nose        <- 0
    None,                   # 1  neck        <- 双肩中点
    6, 8, 10,               # 2-4 右肩/肘/腕
    5, 7, 9,                # 5-7 左肩/肘/腕
    12, 14, 16,             # 8-10 右髋/膝/踝
    11, 13, 15,             # 11-13 左髋/膝/踝
    2, 1, 4, 3,             # 14-17 右眼/左眼/右耳/左耳
)


@dataclass
class RenderOptions:
    style: str = "openpose"
    score_thr: float = 0.3
    hands: bool = True
    face: bool = True
    feet: bool = False
    body_stick: int = 4          # 肢体椭圆短轴（官方 stickwidth）
    body_point_radius: int = 4
    hand_point_radius: int = 4
    face_point_radius: int = 3
    wholebody_radius: int = 4    # dwpose133 画法的关节点半径


# --------------------------------------------------------------------------- 绘图基元
def _blend_region(canvas, x0, x1, y0, y1, mask, color, alpha):
    """在 canvas[y0:y1, x0:x1] 里按 mask 把 color 以 alpha 混入。"""
    region = canvas[y0:y1, x0:x1]
    if alpha >= 1.0:
        region[mask] = np.asarray(color, dtype=np.float32)
    else:
        region[mask] = region[mask] * (1.0 - alpha) + np.asarray(color, np.float32) * alpha


def fill_disk(canvas: np.ndarray, cx: float, cy: float, radius: float,
              color, alpha: float = 1.0) -> None:
    """实心圆（对应 cv2.circle，坐标用 int() 截断，与官方代码一致）。"""
    height, width = canvas.shape[:2]
    radius = float(radius)
    if radius <= 0:
        return
    cx, cy = float(int(cx)), float(int(cy))
    r = int(radius)
    x0, x1 = max(0, int(cx) - r), min(width, int(cx) + r + 1)
    y0, y1 = max(0, int(cy) - r), min(height, int(cy) + r + 1)
    if x0 >= x1 or y0 >= y1:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1]
    mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius * radius
    _blend_region(canvas, x0, x1, y0, y1, mask, color, alpha)


def fill_ellipse(canvas: np.ndarray, cx: float, cy: float, a: float, b: float,
                 angle_deg: float, color, alpha: float = 1.0) -> None:
    """旋转实心椭圆（近似 cv2.ellipse2Poly + fillConvexPoly）。

    a 沿 angle 方向（肢体长度的一半），b 垂直于它（肢体半宽）。
    """
    height, width = canvas.shape[:2]
    if a <= 0 or b <= 0:
        return
    cx, cy = float(int(cx)), float(int(cy))
    reach = int(math.ceil(max(a, b))) + 1
    x0, x1 = max(0, int(cx) - reach), min(width, int(cx) + reach + 1)
    y0, y1 = max(0, int(cy) - reach), min(height, int(cy) + reach + 1)
    if x0 >= x1 or y0 >= y1:
        return

    theta = math.radians(angle_deg)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    dx = xx - cx
    dy = yy - cy
    u = dx * cos_t + dy * sin_t          # 沿肢体方向
    v = -dx * sin_t + dy * cos_t         # 垂直方向
    mask = (u / a) ** 2 + (v / b) ** 2 <= 1.0
    _blend_region(canvas, x0, x1, y0, y1, mask, color, alpha)


def fill_segment(canvas: np.ndarray, p0, p1, thickness: float, color,
                 alpha: float = 1.0) -> None:
    """两点之间的粗线段（等价 cv2.line(thickness=thickness)）。"""
    dx, dy = float(p1[0] - p0[0]), float(p1[1] - p0[1])
    length = math.hypot(dx, dy)
    if length <= 1e-6:
        fill_disk(canvas, p0[0], p0[1], thickness / 2.0, color, alpha)
        return
    fill_ellipse(canvas, (p0[0] + p1[0]) / 2.0, (p0[1] + p1[1]) / 2.0,
                 length / 2.0, thickness / 2.0, math.degrees(math.atan2(dy, dx)),
                 color, alpha)


def _hsv_to_rgb(hue: float, saturation: float = 1.0, value: float = 1.0):
    """matplotlib.colors.hsv_to_rgb 的等价实现（hue 取模到 0-1）。"""
    hue = hue % 1.0
    i = int(hue * 6.0)
    f = hue * 6.0 - i
    p = value * (1.0 - saturation)
    q = value * (1.0 - saturation * f)
    t = value * (1.0 - saturation * (1.0 - f))
    return [(value, t, p), (q, value, p), (p, value, t),
            (p, q, value), (t, p, value), (value, p, q)][i % 6]


def hand_edge_color(index: int, total: int = len(HAND_EDGES)):
    """手部第 index 条边的颜色。

    官方写法是 cv2.line(..., matplotlib.colors.hsv_to_rgb([index/total,1,1])*255, ...)，
    把 RGB 顺序的三元组喂给了按 BGR 解释的 cv2，等于红蓝互换。这里保留该行为
    （ControlNet 训练用的图就是这样生成的）。
    """
    r, g, b = [channel * 255.0 for channel in _hsv_to_rgb(index / float(total))]
    return (b, g, r)


# --------------------------------------------------------------------------- 格式转换
def to_openpose18(keypoints: np.ndarray, scores: np.ndarray):
    """DWPose(COCO-133) -> OpenPose COCO-18 的 (坐标, 置信度)。

    脖子（索引 1）OpenPose 有而 COCO 没有，用双肩中点补；双肩不可信时退回双耳中点。
    """
    keypoints = np.asarray(keypoints, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)

    points = np.zeros((18, 2), dtype=np.float32)
    conf = np.zeros(18, dtype=np.float32)
    for op_index, coco_index in enumerate(OP18_FROM_COCO):
        if coco_index is None:
            continue
        points[op_index] = keypoints[coco_index]
        conf[op_index] = scores[coco_index]

    neck_conf = float(min(scores[5], scores[6]))
    if neck_conf > 0.0:
        points[1] = (keypoints[5] + keypoints[6]) / 2.0
        conf[1] = neck_conf
    elif float(min(scores[3], scores[4])) > 0.0:      # 双耳中点
        points[1] = (keypoints[3] + keypoints[4]) / 2.0
        conf[1] = float(min(scores[3], scores[4]))
    elif float(scores[0]) > 0.0:                      # 只有鼻子时的兜底
        points[1] = keypoints[0]
        conf[1] = float(scores[0]) * 0.9
    return points, conf


def pose_bbox(keypoints: np.ndarray, scores: np.ndarray, score_thr: float = 0.3):
    """用身体 17 点 + 脚 6 点算人物框 (x0, y0, x1, y1)；可信点太少时返回 None。"""
    scores = np.asarray(scores, dtype=np.float32)
    valid = np.where(scores[:23] >= score_thr)[0]
    if valid.size < 3:
        return None
    points = np.asarray(keypoints, dtype=np.float32)[valid]
    x0, y0 = points.min(axis=0)
    x1, y1 = points.max(axis=0)
    return float(x0), float(y0), float(x1), float(y1)


def _mute_scores(scores: np.ndarray, hands: bool, face: bool, feet: bool) -> np.ndarray:
    """按开关把不需要的部位置信度清零（给 133 点画法用）。"""
    core = core_loader.core()
    coco = core.coco
    muted = np.array(scores, dtype=np.float32, copy=True)
    if not hands:
        muted[coco.HAND_LEFT_START:coco.HAND_LEFT_START + coco.HAND_COUNT] = 0.0
        muted[coco.HAND_RIGHT_START:coco.HAND_RIGHT_START + coco.HAND_COUNT] = 0.0
    if not face:
        muted[coco.FACE_ALL] = 0.0
    if not feet:
        muted[17:23] = 0.0
    return muted


# --------------------------------------------------------------------------- 渲染
def render_openpose(keypoints: np.ndarray, scores: np.ndarray, width: int, height: int,
                    options: RenderOptions) -> np.ndarray:
    """ControlNet 官方画法的骨骼图（黑底 RGB，uint8）。"""
    core = core_loader.core()
    coco = core.coco
    canvas = np.zeros((int(height), int(width), 3), dtype=np.float32)
    points, conf = to_openpose18(keypoints, scores)
    threshold = float(options.score_thr)

    # 1) 关节点：半径 4 的实心圆，颜色 = 该点自身的虹色
    for index in range(18):
        if conf[index] >= threshold:
            fill_disk(canvas, points[index][0], points[index][1],
                      options.body_point_radius, OP_COLORS[index], 1.0)

    # 2) 肢体：官方只画 limbSeq 的前 17 条，按 0.4/0.6 与画布混合
    for index in range(17):
        first, second = OP_LIMB_SEQ[index][0] - 1, OP_LIMB_SEQ[index][1] - 1
        if conf[first] < threshold or conf[second] < threshold:
            continue
        p0, p1 = points[first], points[second]
        dx, dy = float(p1[0] - p0[0]), float(p1[1] - p0[1])
        length = math.hypot(dx, dy)
        if length <= 1e-6:
            continue
        fill_ellipse(canvas, (p0[0] + p1[0]) / 2.0, (p0[1] + p1[1]) / 2.0,
                     int(length / 2.0), options.body_stick,
                     math.degrees(math.atan2(dy, dx)), OP_COLORS[index], 0.6)

    # 3) 脚（OpenPose-18 没有这 6 点，可选补画；用脚踝的颜色）
    if options.feet:
        for first, second in ((15, 17), (17, 18), (15, 19),
                              (16, 20), (20, 21), (16, 22)):
            if scores[first] < threshold or scores[second] < threshold:
                continue
            color = OP_COLORS[13] if first in (15, 17, 18, 19) else OP_COLORS[10]
            fill_segment(canvas, keypoints[first], keypoints[second],
                         options.body_stick, color, 0.6)

    # 4) 手部：20 条边 HSV 取色、线宽 2，关节点红色半径 4
    if options.hands:
        for base in (coco.HAND_LEFT_START, coco.HAND_RIGHT_START):
            local_conf = scores[base:base + coco.HAND_COUNT]
            local_pts = keypoints[base:base + coco.HAND_COUNT]
            for edge_index, (a, b) in enumerate(HAND_EDGES):
                if local_conf[a] < threshold or local_conf[b] < threshold:
                    continue
                fill_segment(canvas, local_pts[a], local_pts[b], 2,
                             hand_edge_color(edge_index), 1.0)
            for index in range(coco.HAND_COUNT):
                if local_conf[index] >= threshold:
                    fill_disk(canvas, local_pts[index][0], local_pts[index][1],
                              options.hand_point_radius, HAND_POINT_COLOR, 1.0)

    # 5) 人脸：68 点白色实心圆（半径 3）
    if options.face:
        for index in coco.FACE_ALL:
            if scores[index] < threshold:
                continue
            fill_disk(canvas, keypoints[index][0], keypoints[index][1],
                      options.face_point_radius, FACE_POINT_COLOR, 1.0)

    return np.clip(canvas, 0.0, 255.0).astype(np.uint8)


def render_wholebody(keypoints: np.ndarray, scores: np.ndarray, width: int, height: int,
                     options: RenderOptions) -> np.ndarray:
    """133 点全画（复用插件预览的画法，改成黑底）。"""
    core = core_loader.core()
    canvas = np.zeros((int(height), int(width), 3), dtype=np.uint8)
    muted = _mute_scores(scores, options.hands, options.face, options.feet)
    return core.imops.draw_pose_preview(canvas, keypoints, muted,
                                        float(options.score_thr),
                                        int(options.wholebody_radius))


def render(keypoints: np.ndarray, scores: np.ndarray, width: int, height: int,
           options: RenderOptions) -> np.ndarray:
    """按 options.style 渲染骨骼图（统一输出黑底 RGB uint8）。"""
    if options.style == "dwpose133":
        return render_wholebody(keypoints, scores, width, height, options)
    return render_openpose(keypoints, scores, width, height, options)


def draw_overlay(rgb: np.ndarray, keypoints: np.ndarray, scores: np.ndarray,
                 score_thr: float = 0.3, radius: int = 4) -> np.ndarray:
    """在原图上叠一层骨架（预览用，复用插件的绘制函数）。"""
    core = core_loader.core()
    return core.imops.draw_pose_preview(rgb, keypoints, scores,
                                        float(score_thr), int(radius))
