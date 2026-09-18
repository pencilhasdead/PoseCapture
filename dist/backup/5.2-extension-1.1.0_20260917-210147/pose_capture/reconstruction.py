"""2D -> 3D：用深度模型给每个关节解出真实深度，再按透视相机映射到骨架尺度。

原理：
  1. 深度模型（depth.py，Depth Anything V2）对整张图预测相对深度，按关键点像素
     采样，得到每个关节的深度 d（相对深度，仿射不变：真实深度 = α·d + β）；
  2. 相机：X=(u-cx)·Y/f，Z=(cy-v)·Y/f，Y=β+α·n（n 是归一化相对深度，f 是焦距）；
  3. (α, β) 由“全图所有骨段长度 = 骨架骨长”做一次**全局**最小二乘拟合（高斯-牛顿），
     所以不再逐关节用骨长反解、也不需要“手肘朝前还是朝后”的启发式：
     透视下“谁离镜头更近”会改变像素尺度，因此深度方向是可辨识的；
  4. 骨长解算（旧方式）只作为兜底：深度模型缺失、拟合失败，或拟合误差明显更大时使用；
  5. 最后按人物朝向把整套坐标旋到骨架的静止坐标系（角色面朝 -Y）。

得到的每个关节投影回图像后与原关键点一致，深度方向来自真实像素而不是猜测。
"""
from __future__ import annotations

import math

import numpy as np
from mathutils import Vector

from . import coco, depth as depth_model, utils

# 人物朝向（相机空间：+Y 指向画面内部，即远离镜头）
FACING_DIRS = {
    "FRONT": np.array((0.0, -1.0, 0.0)),   # 面朝镜头
    "BACK": np.array((0.0, 1.0, 0.0)),     # 背对镜头
    "FACE_RIGHT": np.array((1.0, 0.0, 0.0)),  # 人物朝画面右侧（侧面）
    "FACE_LEFT": np.array((-1.0, 0.0, 0.0)),
}
FACING_LABELS = {
    "FRONT": "正面朝镜头",
    "BACK": "背面朝镜头",
    "FACE_RIGHT": "面朝画面右侧（侧身）",
    "FACE_LEFT": "面朝画面左侧（侧身）",
}

# 深度来源：
#   DEPTH = 用深度模型给每个关节解深度（默认；前后方向来自像素，不再猜）
#   BONE  = 旧方式：骨长约束 dz=±√(L²-d²)，符号用固定的解剖学假设（兜底）
#   FLAT  = 不做深度，整套姿态压在画面平面内
DEPTH_MODES = ("DEPTH", "BONE", "FLAT")
DEPTH_MODE_LABELS = {
    "DEPTH": "深度模型（推荐）",
    "BONE": "骨长解算（旧方式）",
    "FLAT": "压平到画面平面",
}

DEFAULT_OPTIONS = {
    "score_thr": 0.3,
    "facing": "FRONT",
    "depth_mode": "DEPTH",
    "depth_focal": 1.2,     # 焦距系数：f = 系数 × max(图宽, 图高)，单位像素
    "depth_flip": False,    # 自动拟合的深度方向反了才勾
    "swap_lr": False,
    "fingers": False,
    "scale_hint": 1.0,
}

# 骨长兜底模式里固定的“前后方向”假设（旧版是可调开关，现在只当兜底常量）
FALLBACK_ELBOW_BACK = True
FALLBACK_KNEE_FRONT = True
FALLBACK_SPINE_LEAN_FRONT = True

# 深度拟合的两种参数化：深度模型可能与“深度”线性，也可能是“视差”（1/深度）线性。
# 两种都拟合，取残差小的那个，避免赌模型的输出语义。
DEPTH_TRANSFORMS = ("linear", "inverse")

# 深度解算的骨长误差超过这个值（米）就认为拟合不可信，退回骨长解算
MAX_DEPTH_BONE_ERROR = 0.06


def default_options(**overrides) -> dict:
    opts = dict(DEFAULT_OPTIONS)
    opts.update(overrides)
    return opts

def _pix(keypoints, scores, index, thr: float):
    if index is None:
        return None
    if float(scores[index]) < thr:
        return None
    x, y = float(keypoints[index][0]), float(keypoints[index][1])
    if not (math.isfinite(x) and math.isfinite(y)) or x < 0.0 or y < 0.0:
        return None
    return np.array((x, y), dtype=np.float64)


def _mean(points):
    points = [p for p in points if p is not None]
    if not points:
        return None
    return np.mean(np.stack(points, axis=0), axis=0)


def _resample_polyline(points: list[np.ndarray], count: int) -> list[np.ndarray]:
    """按弧长把折线重采样成 count+1 个点。"""
    if len(points) < 2 or count < 1:
        return list(points)
    pts = [np.asarray(p, dtype=np.float64) for p in points]
    seg = [np.linalg.norm(pts[i + 1] - pts[i]) for i in range(len(pts) - 1)]
    total = float(sum(seg))
    if total <= 1e-9:
        return [pts[0].copy() for _ in range(count + 1)]

    cumulative = [0.0]
    for length in seg:
        cumulative.append(cumulative[-1] + length)

    out = []
    for k in range(count + 1):
        target = total * k / count
        index = 0
        while index < len(seg) - 1 and cumulative[index + 1] < target:
            index += 1
        span = max(cumulative[index + 1] - cumulative[index], 1e-12)
        t = (target - cumulative[index]) / span
        out.append(pts[index] * (1.0 - t) + pts[index + 1] * t)
    return out


def _safe_cross(a: np.ndarray, b: np.ndarray):
    axis = np.cross(a, b)
    norm = float(np.linalg.norm(axis))
    if norm < 1e-9:
        return None
    return axis / norm


def _collect_2d(keypoints, scores, opts) -> dict:
    """收集所需的 2D 像素点；缺失的用镜像/插值补上，并记录哪些是补出来的。"""
    thr = float(opts["score_thr"])
    swap = bool(opts["swap_lr"])

    def kp(index):
        return _pix(keypoints, scores, index, thr)

    pairs = {
        "eye_L": (coco.LEFT_EYE, coco.RIGHT_EYE), "eye_R": (coco.RIGHT_EYE, coco.LEFT_EYE),
        "ear_L": (coco.LEFT_EAR, coco.RIGHT_EAR), "ear_R": (coco.RIGHT_EAR, coco.LEFT_EAR),
        "sh_L": (coco.LEFT_SHOULDER, coco.RIGHT_SHOULDER),
        "sh_R": (coco.RIGHT_SHOULDER, coco.LEFT_SHOULDER),
        "el_L": (coco.LEFT_ELBOW, coco.RIGHT_ELBOW), "el_R": (coco.RIGHT_ELBOW, coco.LEFT_ELBOW),
        "wr_L": (coco.LEFT_WRIST, coco.RIGHT_WRIST), "wr_R": (coco.RIGHT_WRIST, coco.LEFT_WRIST),
        "hip_L": (coco.LEFT_HIP, coco.RIGHT_HIP), "hip_R": (coco.RIGHT_HIP, coco.LEFT_HIP),
        "kn_L": (coco.LEFT_KNEE, coco.RIGHT_KNEE), "kn_R": (coco.RIGHT_KNEE, coco.LEFT_KNEE),
        "an_L": (coco.LEFT_ANKLE, coco.RIGHT_ANKLE), "an_R": (coco.RIGHT_ANKLE, coco.LEFT_ANKLE),
        "big_L": (coco.LEFT_BIG_TOE, coco.RIGHT_BIG_TOE),
        "big_R": (coco.RIGHT_BIG_TOE, coco.LEFT_BIG_TOE),
    }
    pts = {name: kp(a if not swap else b) for name, (a, b) in pairs.items()}
    synth: set[str] = set()

    pts["nose"] = kp(coco.NOSE)
    pts["chin"] = kp(coco.FACE_CHIN)
    pts["eye_a"] = kp(coco.FACE_EYE_L_END)   # 画面左侧眼角
    pts["eye_b"] = kp(coco.FACE_EYE_R_END)   # 画面右侧眼角
    pts["brow"] = _mean([kp(i) for i in coco.FACE_BROWS])

    left_base = coco.HAND_LEFT_START if not swap else coco.HAND_RIGHT_START
    right_base = coco.HAND_RIGHT_START if not swap else coco.HAND_LEFT_START
    hands = {}
    for side, base in (("L", left_base), ("R", right_base)):
        hands[side] = {local: kp(base + local) for local in range(coco.HAND_COUNT)}

    if pts["sh_L"] is None and pts["sh_R"] is None:
        return {"error": "没有检测到肩部关键点，无法重建姿态"}

    if pts["sh_L"] is not None and pts["sh_R"] is not None:
        pts["chest"] = (pts["sh_L"] + pts["sh_R"]) * 0.5
    elif pts["sh_L"] is not None:
        pts["chest"] = pts["sh_L"].copy()
    else:
        pts["chest"] = pts["sh_R"].copy()

    if pts["sh_L"] is None:
        pts["sh_L"] = (2.0 * pts["chest"] - pts["sh_R"]) if pts["sh_R"] is not None \
            else pts["chest"].copy()
        synth.add("sh_L")
    if pts["sh_R"] is None:
        pts["sh_R"] = (2.0 * pts["chest"] - pts["sh_L"]) if pts["sh_L"] is not None \
            else pts["chest"].copy()
        synth.add("sh_R")

    # 胯部
    if pts["hip_L"] is None and pts["hip_R"] is None:
        anchors = [p for p in (pts["an_L"], pts["an_R"], pts["kn_L"], pts["kn_R"])
                   if p is not None]
        bottom = _mean(anchors) if anchors else None
        pelvis = pts["chest"] + (bottom - pts["chest"]) * 0.45 if bottom is not None \
            else pts["chest"].copy()
        half = max(abs(pts["sh_L"][0] - pts["sh_R"][0]) * 0.4, 8.0)
        pts["hip_L"] = np.array((pelvis[0] + half, pelvis[1]))
        pts["hip_R"] = np.array((pelvis[0] - half, pelvis[1]))
        synth.update(("hip_L", "hip_R"))
    elif pts["hip_L"] is None:
        pts["hip_L"] = np.array((2.0 * pts["chest"][0] - pts["hip_R"][0], pts["hip_R"][1]))
        synth.add("hip_L")
    elif pts["hip_R"] is None:
        pts["hip_R"] = np.array((2.0 * pts["chest"][0] - pts["hip_L"][0], pts["hip_L"][1]))
        synth.add("hip_R")
    pts["pelvis"] = (pts["hip_L"] + pts["hip_R"]) * 0.5

    # 四肢关节缺失时插值/外推
    for side in ("L", "R"):
        for name, above, below in (("kn_%s" % side, "hip_%s" % side, "an_%s" % side),
                                   ("an_%s" % side, "kn_%s" % side, None),
                                   ("el_%s" % side, "sh_%s" % side, "wr_%s" % side),
                                   ("wr_%s" % side, "el_%s" % side, None),
                                   ("big_%s" % side, "an_%s" % side, None)):
            if pts.get(name) is not None:
                continue
            top = pts.get(above)
            bottom = pts.get(below) if below else None
            if top is None and bottom is None:
                continue
            if top is None:
                top = bottom
            if bottom is not None:
                pts[name] = (top + bottom) * 0.5 if name not in ("an_%s" % side, "wr_%s" % side) \
                    else bottom
            else:
                if name.startswith("big_"):
                    span = 0.45 * float(np.linalg.norm(pts["an_%s" % side]
                                                       - pts["kn_%s" % side])) \
                        if pts.get("kn_%s" % side) is not None else 12.0
                    pts[name] = top + np.array((0.0, max(span, 6.0)))
                else:
                    span = float(np.linalg.norm(pts["pelvis"] - pts["chest"]))
                    pts[name] = top + np.array((0.0, max(span * 0.45, 8.0)))
            synth.add(name)

    # 头部基准点（耳朵中点，退化为眼睛中点 / 鼻子）
    ears = _mean([pts.get("ear_L"), pts.get("ear_R")])
    if ears is None:
        ears = _mean([pts.get("eye_L"), pts.get("eye_R")])
    if ears is None:
        ears = pts.get("nose")
    if ears is not None:
        pts["head_base"] = ears.copy()
    else:
        pts["head_base"] = pts["chest"] + (pts["pelvis"] - pts["chest"]) * -0.35
        synth.add("head_base")

    if pts["brow"] is not None and pts["chin"] is not None:
        pts["head_tip"] = pts["brow"] + (pts["brow"] - pts["chin"]) * 0.25
    elif pts["nose"] is not None:
        base = pts["head_base"]
        pts["head_tip"] = pts["nose"] + (pts["nose"] - base) * 0.3
    else:
        pts["head_tip"] = None
        synth.add("head_tip")

    return {"points": pts, "hands": hands, "synth": synth}


# --------------------------------------------------------------------------- 公共工具
def _orthographic_scale(pts, metrics, image_size, opts) -> float:
    """用“关键点像素距离 / 骨架骨长”拟合“像素/米”（正交模式的全局比例）。"""
    spine_span = float((metrics["chest"] - metrics["pelvis"]).length)
    samples = []
    for side in ("L", "R"):
        samples.extend([
            (pts.get("sh_%s" % side), pts.get("el_%s" % side),
             metrics["upper_arm_len"][side]),
            (pts.get("el_%s" % side), pts.get("wr_%s" % side),
             metrics["forearm_len"][side]),
            (pts.get("hip_%s" % side), pts.get("kn_%s" % side),
             metrics["thigh_len"][side]),
            (pts.get("kn_%s" % side), pts.get("an_%s" % side),
             metrics["shin_len"][side]),
        ])
    samples.append((pts.get("pelvis"), pts.get("chest"), spine_span))
    samples = [item for item in samples
               if item[0] is not None and item[1] is not None and item[2] > 1e-6
               and float(np.linalg.norm(item[1] - item[0])) > 1.0]

    if samples:
        # 关键点距离只会因“透视缩短”而变小，所以真值应取偏大的分位数，
        # 否则大部分骨骼都会算不出深度（等价于把姿态压平）。
        ratios = sorted(float(np.linalg.norm(b - a)) / L for a, b, L in samples)
        selected = ratios[-max(1, len(ratios) // 4):]
        scale = float(sum(selected) / len(selected))
    else:
        scale = float(image_size[1]) / 1.7
    scale *= float(opts.get("scale_hint", 1.0) or 1.0)
    return float(max(scale, 1e-3))


def _bone_constraints(metrics) -> list[dict]:
    """骨段约束：点名用于像素/深度采样，关节名用于 3D 骨长误差，长度单位米。"""
    items = []
    for side in ("L", "R"):
        items.extend([
            {"point_a": "sh_%s" % side, "point_b": "el_%s" % side,
             "joint_a": "shoulder.%s" % side, "joint_b": "elbow.%s" % side,
             "length": float(metrics["upper_arm_len"][side] or 0.0)},
            {"point_a": "el_%s" % side, "point_b": "wr_%s" % side,
             "joint_a": "elbow.%s" % side, "joint_b": "wrist.%s" % side,
             "length": float(metrics["forearm_len"][side] or 0.0)},
            {"point_a": "hip_%s" % side, "point_b": "kn_%s" % side,
             "joint_a": "hip.%s" % side, "joint_b": "knee.%s" % side,
             "length": float(metrics["thigh_len"][side] or 0.0)},
            {"point_a": "kn_%s" % side, "point_b": "an_%s" % side,
             "joint_a": "knee.%s" % side, "joint_b": "ankle.%s" % side,
             "length": float(metrics["shin_len"][side] or 0.0)},
            {"point_a": "an_%s" % side, "point_b": "big_%s" % side,
             "joint_a": "ankle.%s" % side, "joint_b": "ball.%s" % side,
             "length": float(metrics["foot_len"][side] or 0.0)},
        ])
    spine_span = float((metrics["chest"] - metrics["pelvis"]).length)
    items.append({"point_a": "pelvis", "point_b": "chest",
                  "joint_a": "pelvis", "joint_b": "chest",
                  "length": float(metrics.get("pelvis_to_shoulders") or spine_span)})
    return [item for item in items if item["length"] > 1e-6]


def _bone_error_rms(joints, constraints) -> float:
    """实际 3D 骨长与骨架骨长的均方根误差（米）；两种解算方式用同一把尺子比较。"""
    errors = []
    for item in constraints:
        head = joints.get(item["joint_a"])
        tail = joints.get(item["joint_b"])
        if head is None or tail is None:
            continue
        errors.append(float(np.linalg.norm(np.asarray(tail) - np.asarray(head)))
                      - item["length"])
    if not errors:
        return float("inf")
    return float(np.sqrt(np.mean(np.square(errors))))


def _perspective_point(point, depth_axis, cx, cy, focal):
    """像素 + 深度 -> 相机空间 3D（X 右、Y 向画面内、Z 上，单位米）。"""
    y = float(depth_axis)
    return np.array(((float(point[0]) - cx) * y / focal, y,
                     (cy - float(point[1])) * y / focal))


def _huber(error: float, delta: float) -> float:
    """Huber 残差：小误差保持线性，大误差（关键点抖动/骨长不匹配）压成 sqrt。"""
    magnitude = abs(error)
    if magnitude <= delta:
        return error
    return math.copysign(math.sqrt(2.0 * delta * magnitude - delta * delta), error)


def _gauss_newton(residual_fn, theta0, iterations: int = 40) -> tuple[np.ndarray, float]:
    """两参数高斯-牛顿 + 回溯；返回 (参数, 残差平方和)。"""
    theta = np.asarray(theta0, dtype=np.float64)
    residual = residual_fn(theta)
    cost = float(residual @ residual)
    for _ in range(iterations):
        jacobian = np.zeros((residual.size, theta.size), dtype=np.float64)
        for index in range(theta.size):
            step = 1e-4 * max(1.0, abs(float(theta[index])))
            shifted = theta.copy()
            shifted[index] += step
            jacobian[:, index] = (residual_fn(shifted) - residual) / step
        try:
            delta = np.linalg.lstsq(jacobian, -residual, rcond=None)[0]
        except np.linalg.LinAlgError:
            break
        accepted = False
        for attempt in range(8):
            trial = theta + delta * (0.5 ** attempt)
            trial_residual = residual_fn(trial)
            trial_cost = float(trial_residual @ trial_residual)
            if trial_cost < cost - 1e-12:
                theta, residual, cost = trial, trial_residual, trial_cost
                accepted = True
                break
        if not accepted:
            break
    return theta, cost


# --------------------------------------------------------------------------- 骨长解算（兜底）
def _solve_bone_lengths(pts, hands, metrics, opts, image_size, use_depth: bool) -> dict:
    """旧方式：正交投影 + 骨长约束 dz=±√(L²-d²)，符号用固定的解剖学假设。

    只在深度模型不可用、拟合失败或误差明显更大时使用（见 reconstruct）。
    """
    image_w, image_h = float(image_size[0]), float(image_size[1])
    scale = _orthographic_scale(pts, metrics, image_size, opts)
    cx, cy = image_w * 0.5, image_h * 0.5
    front = FACING_DIRS.get(opts["facing"], FACING_DIRS["FRONT"])
    front_y_sign = 1.0 if front[1] > 0.0 else -1.0

    def plane(point):
        return np.array(((point[0] - cx) / scale, (cy - point[1]) / scale))

    def solve(parent3, point2, length, want_front: bool):
        """已知骨长与平面内位置，求子关节位置；深度符号由 want_front 决定。"""
        xy = plane(point2)
        dx, dy = xy[0] - parent3[0], xy[1] - parent3[2]
        distance = math.hypot(dx, dy)
        if not use_depth or length <= 1e-6 or distance >= length:
            depth = 0.0
        else:
            depth = math.sqrt(max(length * length - distance * distance, 0.0))
        offset = 0.0 if depth == 0.0 else (front_y_sign * depth if want_front
                                           else -front_y_sign * depth)
        return np.array((xy[0], parent3[1] + offset, xy[1]))

    joints: dict = {}
    pelvis_xy = plane(pts["pelvis"])
    pelvis_offset = float(metrics.get("pelvis_offset_z", 0.0) or 0.0)
    chest_offset = float(metrics.get("chest_offset_z", 0.0) or 0.0)
    # 深度解算用“未偏移”的骨盆点，偏移只用于后续方向（否则会污染骨长约束）
    pelvis_plain = np.array((pelvis_xy[0], 0.0, pelvis_xy[1]))
    joints["pelvis"] = pelvis_plain + np.array((0.0, 0.0, pelvis_offset))

    spine_span = float((metrics["chest"] - metrics["pelvis"]).length)
    chest_len = float(metrics.get("pelvis_to_shoulders") or spine_span)
    joints["chest"] = solve(pelvis_plain, pts["chest"], chest_len,
                            FALLBACK_SPINE_LEAN_FRONT)
    joints["chest"][2] += chest_offset
    spine_points = _resample_polyline([joints["pelvis"], joints["chest"]], 4)

    # 躯干上的刚性偏移：跟着躯干一起转，用来定肩/胯/颈/头基座的三维关系
    rest_pelvis = np.asarray(metrics["pelvis"], dtype=float)
    rest_chest = np.asarray(metrics["chest"], dtype=float)
    q_torso = Vector(rest_chest - rest_pelvis).normalized().rotation_difference(
        Vector(joints["chest"] - joints["pelvis"]).normalized())

    def torso_offset(vec):
        return np.asarray(q_torso @ Vector(np.asarray(vec, dtype=float)))

    for side in ("L", "R"):
        hip_xy = plane(pts["hip_%s" % side])
        hip_y = joints["pelvis"][1] + torso_offset(metrics["hip_offsets"][side])[1]
        joints["hip.%s" % side] = np.array((hip_xy[0], hip_y, hip_xy[1]))

    for side in ("L", "R"):
        sh_xy = plane(pts["sh_%s" % side])
        shoulder_y = joints["chest"][1] + torso_offset(
            metrics["shoulder_offsets"][side])[1]
        joints["shoulder.%s" % side] = np.array((sh_xy[0], shoulder_y, sh_xy[1]))
        clavicle = (metrics.get("clavicle_offsets") or {}).get(side)
        joints["shoulder_origin.%s" % side] = joints["chest"] + (
            torso_offset(clavicle) if clavicle is not None else 0.0)

    neck_offset = metrics.get("neck_offset")
    joints["neck_start"] = joints["chest"] + (torso_offset(neck_offset)
                                              if neck_offset is not None else 0.0)

    for side in ("L", "R"):
        shoulder = joints["shoulder.%s" % side]
        # 单张图看不出肘部到底在身体前面还是后面（两者的画面投影完全相同），
        # 兜底模式固定按“手肘朝后、膝盖朝前”的解剖学假设猜。
        elbow = solve(shoulder, pts["el_%s" % side], metrics["upper_arm_len"][side],
                      not FALLBACK_ELBOW_BACK)
        joints["elbow.%s" % side] = elbow
        wrist = solve(elbow, pts["wr_%s" % side], metrics["forearm_len"][side],
                      FALLBACK_ELBOW_BACK)
        joints["wrist.%s" % side] = wrist
        # 手：关键点只到指根，用“平面方向 + 腕部深度”推，避免骨长不匹配
        hand_point = hands[side].get(coco.HAND_MIDDLE_MCP)
        if hand_point is None:
            hand_point = pts["wr_%s" % side]
        hand_xy = plane(hand_point)
        joints["hand_tip.%s" % side] = np.array((hand_xy[0], wrist[1], hand_xy[1]))

        hip = joints["hip.%s" % side]
        knee = solve(hip, pts["kn_%s" % side], metrics["thigh_len"][side],
                     FALLBACK_KNEE_FRONT)
        joints["knee.%s" % side] = knee
        ankle = solve(knee, pts["an_%s" % side], metrics["shin_len"][side],
                      not FALLBACK_KNEE_FRONT)
        joints["ankle.%s" % side] = ankle
        ball = solve(ankle, pts["big_%s" % side], metrics["foot_len"][side], True)
        joints["ball.%s" % side] = ball
        toe_dir = ball - ankle
        norm = float(np.linalg.norm(toe_dir))
        toe_len = metrics["toe_len"][side] or 0.05
        joints["toe_tip.%s" % side] = ball + (toe_dir / norm if norm > 1e-9
                                              else np.array((0.0, -1.0, 0.0))) * toe_len

    # 头部：脸部的“上方向”只提供平面方向，深度沿用头部基座（俯仰无法从单图判定）
    head_len = metrics["head_len"] or 0.2
    head_xy = plane(pts["head_base"])
    head_offset = metrics.get("head_offset")
    head_y = joints["chest"][1] + (torso_offset(head_offset)[1]
                                   if head_offset is not None else 0.0)
    joints["head_base"] = np.array((head_xy[0], head_y, head_xy[1]))
    if pts["head_tip"] is not None:
        tip_xy = plane(pts["head_tip"])
        joints["head_tip"] = np.array((tip_xy[0], joints["head_base"][1], tip_xy[1]))
    else:
        joints["head_tip"] = joints["head_base"] + np.array((0.0, 0.0, head_len))

    refs = _build_refs(joints, _eye_line_from_plane(pts, plane))
    fingers = _build_fingers_bone(pts, hands, joints, opts, plane)
    return {"joints": joints, "refs": refs, "spine_points": spine_points,
            "fingers": fingers, "scale": scale,
            "mode": "BONE" if use_depth else "FLAT"}


def _eye_line_from_plane(pts, plane):
    """正交模式的眼线：两个眼角只取画面平面内的方向（深度沿用头部）。"""
    if pts.get("eye_a") is None or pts.get("eye_b") is None:
        return None
    a, b = plane(pts["eye_a"]), plane(pts["eye_b"])
    candidate = np.array((b[0] - a[0], 0.0, b[1] - a[1]))
    return candidate if float(np.linalg.norm(candidate)) > 1e-6 else None


def _build_refs(joints, eye_line=None) -> dict:
    """参照轴（用于 roll / 扭曲对齐）；眼线由调用方按自己的相机模型算好。"""
    refs = {"hip_line": joints["hip.L"] - joints["hip.R"],
            "shoulder_line": joints["shoulder.L"] - joints["shoulder.R"]}
    if eye_line is None or float(np.linalg.norm(eye_line)) <= 1e-6:
        refs["eye_line"] = refs["shoulder_line"]
    else:
        refs["eye_line"] = np.asarray(eye_line, dtype=float)

    for side in ("L", "R"):
        upper_arm = joints["elbow.%s" % side] - joints["shoulder.%s" % side]
        forearm = joints["wrist.%s" % side] - joints["elbow.%s" % side]
        thigh = joints["knee.%s" % side] - joints["hip.%s" % side]
        shin = joints["ankle.%s" % side] - joints["knee.%s" % side]
        # 弯曲轴用固定顺序的叉乘（上段 x 下段），与 rigify_map 里静止侧的算法一致：
        # 这样 roll 对齐才能分辨“肢体向前弯还是向后反折”。
        axis = _safe_cross(upper_arm, forearm)
        if axis is not None:
            refs["armbend.%s" % side] = axis
        axis = _safe_cross(thigh, shin)
        if axis is not None:
            refs["kneebend.%s" % side] = axis
    return refs


def _build_fingers_bone(pts, hands, joints, opts, plane) -> dict:
    """骨长兜底模式的手指：平面方向 + 腕部深度。"""
    fingers: dict = {}
    if not opts.get("fingers"):
        return fingers
    for side in ("L", "R"):
        base_y = joints["wrist.%s" % side][1]
        per_hand = {}
        for finger, indexes in coco.HAND_GROUPS.items():
            chain = []
            for local in indexes:
                point = hands[side].get(local)
                if point is None:
                    chain = []
                    break
                xy = plane(point)
                chain.append(np.array((xy[0], base_y, xy[1])))
            if chain:
                per_hand[finger] = chain
        if per_hand:
            fingers[side] = per_hand
    return fingers


# --------------------------------------------------------------------------- 深度模型解算
def _depth_variants(depths: dict, reference: str = "pelvis", span_keys=None) -> dict:
    """把采样到的相对深度转成“归一化相对深度”。

    深度模型可能与“深度”线性，也可能是“视差”（1/深度）线性；两种都给出，
    由拟合残差决定用哪个（不赌模型的输出语义）。
    span_keys 指定用哪些点算“跨度”（默认约束点）：个别关节采样到背景时不会把
    整体尺度带偏；跨度取 80% 分位，进一步容忍单个离群采样。
    返回 {参数化名: {点名: 归一化值}}。
    """
    base = depths.get(reference)
    if base is None or not math.isfinite(base):
        return {}
    out = {}
    for kind in DEPTH_TRANSFORMS:
        values = {}
        for name, value in depths.items():
            if value is None or not math.isfinite(value):
                continue
            if kind == "inverse":
                if abs(value) < 1e-6 or abs(base) < 1e-6:
                    continue
                value = 1.0 / value - 1.0 / base
            else:
                value = value - base
            values[name] = float(value)
        span_values = sorted(abs(values[name]) for name in (span_keys or values)
                             if name in values)
        if not span_values:
            span_values = sorted(abs(item) for item in values.values())
        if not span_values:
            continue
        span = span_values[min(len(span_values) - 1, int(len(span_values) * 0.8))]
        if span < 1e-6:
            continue
        out[kind] = {name: item / span for name, item in values.items()}
    return out


def _solve_depth(pts, hands, metrics, opts, image_size, depth_map, depth_size):
    """深度模型驱动的解算：每个关节的深度直接来自深度图。

    未知量只有两个**全局**参数：(α, β)。α 是“归一化相对深度 -> 米”的缩放（带符号，
    所以深度方向由拟合自己决定），β 是参考关节（骨盆）到镜头的距离。两者由
    “全图所有骨段的 3D 长度 = 骨架骨长”这组约束做一次最小二乘拟合（高斯-牛顿），
    不是逐关节用骨长反解，所以不存在“肘部朝前还是朝后”的歧义。

    返回 None 表示拟合不可信（调用方退回骨长解算）。
    """
    image_w, image_h = float(image_size[0]), float(image_size[1])
    cx, cy = image_w * 0.5, image_h * 0.5
    focal = max(float(opts.get("depth_focal", 1.2) or 1.2) * max(image_w, image_h), 1.0)
    unit = max(float(metrics.get("pelvis_to_shoulders") or 0.5), 0.05)
    constraints = _bone_constraints(metrics)
    if not constraints:
        return None
    # 约束点决定“身体深度范围”：个别关节采样到背景（头顶落空等）不会把尺度带偏
    span_keys = sorted({item["point_a"] for item in constraints}
                       | {item["point_b"] for item in constraints})

    # ---------------- 1) 采样每个关节的深度 ----------------
    sample_list = []
    for name in ("pelvis", "chest", "hip_L", "hip_R", "sh_L", "sh_R", "el_L", "el_R",
                 "wr_L", "wr_R", "kn_L", "kn_R", "an_L", "an_R", "big_L", "big_R",
                 "head_base", "head_tip", "eye_a", "eye_b"):
        sample_list.append((name, pts.get(name)))
    for side in ("L", "R"):
        sample_list.append(("mcp_%s" % side, hands[side].get(coco.HAND_MIDDLE_MCP)))
        if opts.get("fingers"):
            for local in range(coco.HAND_COUNT):
                sample_list.append(("h_%s_%02d" % (side, local), hands[side].get(local)))
    sampled = depth_model.sample_many(depth_map, depth_size, image_size,
                                      [point for _name, point in sample_list])
    depths = {name: float(value) for (name, _point), value in zip(sample_list, sampled)}

    variants = _depth_variants(depths, "pelvis", span_keys)
    if not variants:
        return None

    # ---------------- 2) 拟合 (α, β) ----------------
    scale = _orthographic_scale(pts, metrics, image_size, opts)
    beta0 = focal / max(scale, 1e-6)

    def constraint_range(normalized):
        """约束点的归一化深度范围（决定“身体深度跨度”）。"""
        values = [normalized[name] for name in span_keys if name in normalized]
        if not values:
            return -1.0, 1.0
        return min(values), max(values)

    def residuals(theta, normalized, n_range):
        alpha, beta = float(theta[0]), float(theta[1])
        out = []
        for item in constraints:
            na = normalized.get(item["point_a"])
            nb = normalized.get(item["point_b"])
            point_a = pts.get(item["point_a"])
            point_b = pts.get(item["point_b"])
            if na is None or nb is None or point_a is None or point_b is None:
                continue
            ya, yb = beta + alpha * na, beta + alpha * nb
            if ya <= 1e-3 or yb <= 1e-3:      # 关节跑到镜头后面：判死
                out.append(1.0)
                continue
            head = _perspective_point(point_a, ya, cx, cy, focal)
            tail = _perspective_point(point_b, yb, cx, cy, focal)
            error = float(np.linalg.norm(tail - head)) - item["length"]
            out.append(_huber(error, max(0.02, 0.1 * item["length"])))
        # 弱正则：姿态几乎平铺（深度无信息）时避免深度跨度乱跑（与归一化尺度无关）
        out.append(3e-3 * math.sqrt(max(len(constraints), 1)) * alpha * n_range / unit)
        return np.asarray(out, dtype=np.float64)

    solutions = []
    for kind, normalized in variants.items():
        low, high = constraint_range(normalized)
        n_range = max(high - low, 1e-6)
        for target in (0.3 * unit, 0.6 * unit, 1.0 * unit):
            # 初值按“身体深度跨度”给，与归一化方式无关
            alpha0 = target / n_range
            for sign in (1.0, -1.0):
                theta, cost = _gauss_newton(
                    lambda value, data=normalized, span=n_range:
                        residuals(value, data, span),
                    np.array((sign * alpha0, beta0), dtype=np.float64))
                if math.isfinite(cost):
                    solutions.append((cost, kind, normalized, theta))
    if not solutions:
        return None
    solutions.sort(key=lambda item: item[0])
    best = solutions[0]
    if bool(opts.get("depth_flip")):
        # 手动反转：在所有候选里挑“深度方向相反”的最优解
        mirrored = [item for item in solutions if item[3][0] * best[3][0] < 0.0]
        if mirrored:
            best = mirrored[0]
    cost, kind, normalized, theta = best
    alpha, beta = float(theta[0]), float(theta[1])
    if not (math.isfinite(alpha) and math.isfinite(beta)) or beta <= 0.05 or alpha == 0.0:
        return None
    low, high = constraint_range(normalized)
    depth_span = abs(alpha) * max(high - low, 1e-6)
    if depth_span > 4.0 * unit:            # 深度跨度大得不像人：判为拟合失败
        return None
    # 深度图里“值大”是近还是远（给伪彩色预览用）：线性参数化时 α<0 表示值大=近
    near_is_large = (alpha < 0.0) if kind == "linear" else (alpha > 0.0)

    # ---------------- 3) 组装关节 ----------------
    # 关节深度夹到“身体深度范围 ±30%”内：个别关节采样到背景（头顶落空等）时
    # 不会被甩到几米之外
    margin = 0.3 * max(high - low, 1e-6)
    n_low, n_high = low - margin, high + margin

    def place(name, fallback_y=None):
        point = pts.get(name)
        if point is None:
            return None
        value = normalized.get(name)
        if value is None:
            y = beta if fallback_y is None else float(fallback_y)
        else:
            value = min(max(value, n_low), n_high)
            y = beta + alpha * value
        return _perspective_point(point, max(y, 1e-3), cx, cy, focal)

    joints: dict = {}
    pelvis_offset = float(metrics.get("pelvis_offset_z", 0.0) or 0.0)
    chest_offset = float(metrics.get("chest_offset_z", 0.0) or 0.0)
    joints["pelvis"] = place("pelvis")
    joints["pelvis"][2] += pelvis_offset
    joints["chest"] = place("chest")
    joints["chest"][2] += chest_offset
    spine_points = _resample_polyline([joints["pelvis"], joints["chest"]], 4)

    for side in ("L", "R"):
        joints["hip.%s" % side] = place("hip_%s" % side)
        joints["shoulder.%s" % side] = place("sh_%s" % side)
        joints["elbow.%s" % side] = place("el_%s" % side)
        joints["wrist.%s" % side] = place("wr_%s" % side)
        joints["knee.%s" % side] = place("kn_%s" % side)
        joints["ankle.%s" % side] = place("an_%s" % side)
        joints["ball.%s" % side] = place("big_%s" % side)

        ankle = np.asarray(joints["ankle.%s" % side], dtype=float)
        ball = np.asarray(joints["ball.%s" % side], dtype=float)
        toe_dir = ball - ankle
        span = float(np.linalg.norm(toe_dir))
        toe_len = metrics["toe_len"][side] or 0.05
        joints["toe_tip.%s" % side] = ball + (
            toe_dir / span if span > 1e-9 else np.array((0.0, -1.0, 0.0))) * toe_len

        # 手：关键点只到指根，但有真实深度，方向比“沿用腕部深度”可靠
        hand_tip = place("mcp_%s" % side)
        if hand_tip is None:
            wrist = np.asarray(joints["wrist.%s" % side], dtype=float)
            elbow = np.asarray(joints["elbow.%s" % side], dtype=float)
            direction = wrist - elbow
            span = float(np.linalg.norm(direction))
            hand_len = float(metrics["hand_len"][side] or 0.08)
            hand_tip = wrist + (direction / span if span > 1e-9
                                else np.array((0.0, -1.0, 0.0))) * hand_len
        joints["hand_tip.%s" % side] = np.asarray(hand_tip, dtype=float)

    # 锁骨、脖子没有独立关键点，只能跟着躯干转（与骨长模式一致）
    rest_pelvis = np.asarray(metrics["pelvis"], dtype=float)
    rest_chest = np.asarray(metrics["chest"], dtype=float)
    q_torso = Vector(rest_chest - rest_pelvis).normalized().rotation_difference(
        Vector(joints["chest"] - joints["pelvis"]).normalized())

    def torso_offset(vec):
        return np.asarray(q_torso @ Vector(np.asarray(vec, dtype=float)))

    for side in ("L", "R"):
        clavicle = (metrics.get("clavicle_offsets") or {}).get(side)
        joints["shoulder_origin.%s" % side] = joints["chest"] + (
            torso_offset(clavicle) if clavicle is not None else 0.0)
    neck_offset = metrics.get("neck_offset")
    joints["neck_start"] = joints["chest"] + (torso_offset(neck_offset)
                                              if neck_offset is not None else 0.0)

    head_len = metrics["head_len"] or 0.2
    head_base = place("head_base")
    if head_base is None:
        head_offset = metrics.get("head_offset")
        head_base = joints["chest"] + (torso_offset(head_offset)
                                       if head_offset is not None else 0.0)
    joints["head_base"] = np.asarray(head_base, dtype=float)
    head_tip = place("head_tip")
    joints["head_tip"] = (np.asarray(head_tip, dtype=float) if head_tip is not None
                          else joints["head_base"] + np.array((0.0, 0.0, head_len)))

    # 眼线用两个眼角的真实深度算，比只取平面方向更准
    eye_a, eye_b = place("eye_a"), place("eye_b")
    eye_line = None
    if eye_a is not None and eye_b is not None:
        candidate = np.asarray(eye_b, dtype=float) - np.asarray(eye_a, dtype=float)
        if float(np.linalg.norm(candidate)) > 1e-6:
            eye_line = candidate
    refs = _build_refs(joints, eye_line)
    fingers = _build_fingers_depth(hands, opts, place)
    return {"joints": joints, "refs": refs, "spine_points": spine_points,
            "fingers": fingers, "scale": focal / max(beta, 1e-6), "mode": "DEPTH",
            "depth": {"transform": kind, "alpha": alpha, "beta": beta, "focal": focal,
                      "near_is_large": near_is_large, "cost": float(cost),
                      "bone_error": None, "bone_error_fallback": None}}


def _build_fingers_depth(hands, opts, place) -> dict:
    """深度模式的手指：每个指节用自己像素的深度。"""
    fingers: dict = {}
    if not opts.get("fingers"):
        return fingers
    for side in ("L", "R"):
        per_hand = {}
        for finger, indexes in coco.HAND_GROUPS.items():
            chain = [place("h_%s_%02d" % (side, local)) for local in indexes]
            if all(point is not None for point in chain):
                per_hand[finger] = [np.asarray(point, dtype=float) for point in chain]
        if per_hand:
            fingers[side] = per_hand
    return fingers


# --------------------------------------------------------------------------- 主入口
def reconstruct(keypoints, scores, metrics, image_size, options=None,
                depth_map=None, depth_size=None) -> dict:
    """主入口：返回 {'ok', 'joints', 'refs', 'spine_points', ...}（单位=米，骨架尺度）。

    depth_mode="DEPTH" 且给了 depth_map 时走“深度模型 + 透视相机”解算（每个关节的
    深度来自深度图，(α, β) 由全图骨段全局拟合）；否则退回骨长解算（旧行为）。
    深度解算的骨长误差明显更大时也会自动退回，并写进 notes。
    """
    opts = default_options(**(options or {}))
    collected = _collect_2d(keypoints, scores, opts)
    if "error" in collected:
        return {"ok": False, "message": collected["error"]}
    pts = collected["points"]
    hands = collected["hands"]
    synth = collected["synth"]

    mode = str(opts.get("depth_mode", "DEPTH")).upper()
    if mode not in DEPTH_MODES:
        mode = "DEPTH"
    constraints = _bone_constraints(metrics)
    notes: list[str] = []
    solved = None

    if mode == "DEPTH":
        if depth_map is None or depth_size is None:
            notes.append("没有深度图，已退回骨长解算")
        else:
            solved = _solve_depth(pts, hands, metrics, opts, image_size,
                                  depth_map, depth_size)
            if solved is None:
                notes.append("深度解算失败，已退回骨长解算")
            else:
                # 用同一把尺子（3D 骨长误差）比较两种解算，深度解算明显更差才退回
                fallback = _solve_bone_lengths(pts, hands, metrics, opts, image_size,
                                               use_depth=True)
                depth_error = _bone_error_rms(solved["joints"], constraints)
                fallback_error = _bone_error_rms(fallback["joints"], constraints)
                solved["depth"]["bone_error"] = depth_error
                solved["depth"]["bone_error_fallback"] = fallback_error
                if depth_error > max(MAX_DEPTH_BONE_ERROR, fallback_error * 1.05):
                    notes.append("深度解算骨长误差偏大（%.3f m），已退回骨长解算"
                                 % depth_error)
                    solved = fallback
    if solved is None:
        solved = _solve_bone_lengths(pts, hands, metrics, opts, image_size,
                                     use_depth=(mode != "FLAT"))

    joints = solved["joints"]
    refs = solved["refs"]
    spine_points = solved["spine_points"]
    fingers = solved["fingers"]
    scale = float(solved["scale"])

    # ---------------- 按人物朝向旋转到骨架静止坐标系（角色面朝 -Y） ----------------
    front = FACING_DIRS.get(opts["facing"], FACING_DIRS["FRONT"])
    yaw = math.atan2(float(front[0]), -float(front[1]))
    cos_y, sin_y = math.cos(-yaw), math.sin(-yaw)

    def rotate_vector(vec):
        return np.array((vec[0] * cos_y - vec[1] * sin_y,
                         vec[0] * sin_y + vec[1] * cos_y, vec[2]))

    pivot = np.asarray(joints["pelvis"], dtype=float).copy()

    def rotate_point(point):
        return rotate_vector(np.asarray(point, dtype=float) - pivot) + pivot

    for name in list(joints.keys()):
        joints[name] = rotate_point(joints[name])
    spine_points = [rotate_point(point) for point in spine_points]
    for name in list(refs.keys()):
        refs[name] = rotate_vector(refs[name])
    for side, per_hand in fingers.items():
        for finger in list(per_hand.keys()):
            per_hand[finger] = [rotate_point(point) for point in per_hand[finger]]

    result = {
        "ok": True,
        "scale": scale,
        "yaw_deg": math.degrees(yaw),
        "mode": solved.get("mode", "BONE"),
        "joints": joints,
        "refs": refs,
        "spine_points": spine_points,
        "fingers": fingers,
        "points2d": pts,
        "synth": sorted(synth),
        "notes": notes,
    }
    if "depth" in solved:
        result["depth"] = solved["depth"]
    return result