"""2D -> 3D：用骨架静止姿态的骨长做约束，把单张图片的 2D 关键点解算成 3D 关节。

原理（单目无深度时最实用的做法）：
  1. 用“像素骨长 / 骨架骨长”最小二乘拟合出全局比例 s（像素/米）；
  2. 平面内坐标直接由像素决定：X=(u-cx)/s, Z=(cy-v)/s；
  3. 深度方向用骨长约束补齐：|Δ| = 骨长已知、平面内长度已知 => dz = ±sqrt(L²-d²)，
     符号用“关节朝角色前方/后方弯曲”的启发式决定（可开关/翻转）。
  4. 最后按人物朝向把整套坐标旋到骨架的静止坐标系（角色面朝 -Y）。

得到的每个关节位置投影回图像后与原关键点一致，深度方向则是合理猜测。
"""
from __future__ import annotations

import math

import numpy as np
from mathutils import Vector

from . import coco, utils

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

DEFAULT_OPTIONS = {
    "score_thr": 0.3,
    "facing": "FRONT",
    "use_depth": True,
    "elbow_back": True,
    "arm_flip_L": False,
    "arm_flip_R": False,
    "knee_front": True,
    "spine_lean_front": True,
    "swap_lr": False,
    "fingers": False,
    "scale_hint": 1.0,
}


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

def reconstruct(keypoints, scores, metrics, image_size, options=None) -> dict:
    """主入口：返回 {'ok', 'joints', 'refs', 'spine_points', ...}（单位=米，骨架尺度）。"""
    opts = default_options(**(options or {}))
    image_w, image_h = float(image_size[0]), float(image_size[1])
    collected = _collect_2d(keypoints, scores, opts)
    if "error" in collected:
        return {"ok": False, "message": collected["error"]}
    pts = collected["points"]
    hands = collected["hands"]
    synth = collected["synth"]

    # ---------------- 1) 全局比例（像素/米）最小二乘拟合 ----------------
    spine_span = float((metrics["chest"] - metrics["pelvis"]).length)
    samples = []
    for side in ("L", "R"):
        samples.extend([
            (pts.get("sh_%s" % side), pts.get("el_%s" % side), metrics["upper_arm_len"][side]),
            (pts.get("el_%s" % side), pts.get("wr_%s" % side), metrics["forearm_len"][side]),
            (pts.get("hip_%s" % side), pts.get("kn_%s" % side), metrics["thigh_len"][side]),
            (pts.get("kn_%s" % side), pts.get("an_%s" % side), metrics["shin_len"][side]),
        ])
    samples.append((pts["pelvis"], pts["chest"], spine_span))
    samples = [s for s in samples
               if s[0] is not None and s[1] is not None and s[2] > 1e-6
               and float(np.linalg.norm(s[1] - s[0])) > 1.0]

    if samples:
        # 关键点距离只会因“透视缩短”而变小，所以真值应取偏大的分位数，
        # 否则大部分骨骼都会算不出深度（等价于把姿态压平）。
        per_sample = sorted(float(np.linalg.norm(b - a)) / L for a, b, L in samples)
        selected = per_sample[-max(1, len(per_sample) // 4):]
        scale = float(sum(selected) / len(selected))
    else:
        scale = image_h / 1.7
    scale = float(max(scale * float(opts.get("scale_hint", 1.0) or 1.0), 1e-3))

    # ---------------- 2) 平面坐标 + 深度解算 ----------------
    cx, cy = image_w * 0.5, image_h * 0.5
    front = FACING_DIRS.get(opts["facing"], FACING_DIRS["FRONT"])
    front_y_sign = 1.0 if front[1] > 0.0 else -1.0
    use_depth = bool(opts["use_depth"])

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

    # 胸腔（脊椎顶端）：长度用“骨盆 -> 肩中点”的静止距离，与观测点对应一致
    chest_len = float(metrics.get("pelvis_to_shoulders") or spine_span)
    joints["chest"] = solve(pelvis_plain, pts["chest"], chest_len,
                            bool(opts["spine_lean_front"]))
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
        shoulder_y = joints["chest"][1] + torso_offset(metrics["shoulder_offsets"][side])[1]
        joints["shoulder.%s" % side] = np.array((sh_xy[0], shoulder_y, sh_xy[1]))
        clavicle = (metrics.get("clavicle_offsets") or {}).get(side)
        joints["shoulder_origin.%s" % side] = joints["chest"] + (
            torso_offset(clavicle) if clavicle is not None else 0.0)

    neck_offset = metrics.get("neck_offset")
    joints["neck_start"] = joints["chest"] + (torso_offset(neck_offset)
                                              if neck_offset is not None else 0.0)

    elbow_back = bool(opts["elbow_back"])
    knee_front = bool(opts["knee_front"])
    # 单张图看不出肘部到底在身体前面还是后面（两者的画面投影完全相同），
    # 默认按“手肘朝后、手腕朝前”的解剖学启发式猜；左右两侧可以各自翻转。
    elbow_back_side = {"L": elbow_back != bool(opts.get("arm_flip_L", False)),
                       "R": elbow_back != bool(opts.get("arm_flip_R", False))}

    for side in ("L", "R"):
        shoulder = joints["shoulder.%s" % side]
        side_back = elbow_back_side[side]

        elbow = solve(shoulder, pts["el_%s" % side], metrics["upper_arm_len"][side],
                      not side_back)
        joints["elbow.%s" % side] = elbow
        wrist = solve(elbow, pts["wr_%s" % side], metrics["forearm_len"][side], side_back)
        joints["wrist.%s" % side] = wrist
        # 手：关键点只到指根，用“平面方向 + 腕部深度”推，避免骨长不匹配
        hand_point = hands[side].get(coco.HAND_MIDDLE_MCP)
        if hand_point is None:
            hand_point = pts["wr_%s" % side]
        hand_xy = plane(hand_point)
        joints["hand_tip.%s" % side] = np.array((hand_xy[0], wrist[1], hand_xy[1]))

        hip = joints["hip.%s" % side]
        knee = solve(hip, pts["kn_%s" % side], metrics["thigh_len"][side], knee_front)
        joints["knee.%s" % side] = knee
        ankle = solve(knee, pts["an_%s" % side], metrics["shin_len"][side], not knee_front)
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

    # ---------------- 3) 参照轴（用于 roll / 扭曲对齐） ----------------
    refs: dict = {}
    refs["hip_line"] = joints["hip.L"] - joints["hip.R"]
    refs["shoulder_line"] = joints["shoulder.L"] - joints["shoulder.R"]

    if pts.get("eye_a") is not None and pts.get("eye_b") is not None:
        a, b = plane(pts["eye_a"]), plane(pts["eye_b"])
        eye = np.array((b[0] - a[0], 0.0, b[1] - a[1]))
        refs["eye_line"] = eye if float(np.linalg.norm(eye)) > 1e-6 else refs["shoulder_line"]
    else:
        refs["eye_line"] = refs["shoulder_line"]

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

    # ---------------- 4) 手指（可选） ----------------
    fingers: dict = {}
    if opts.get("fingers"):
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

    # ---------------- 5) 按人物朝向旋转到骨架静止坐标系（角色面朝 -Y） ----------------
    yaw = math.atan2(float(front[0]), -float(front[1]))
    cos_y, sin_y = math.cos(-yaw), math.sin(-yaw)

    def rotate_vector(vec):
        return np.array((vec[0] * cos_y - vec[1] * sin_y,
                         vec[0] * sin_y + vec[1] * cos_y, vec[2]))

    pivot = joints["pelvis"].copy()

    def rotate_point(point):
        moved = point - pivot
        return rotate_vector(moved) + pivot

    for name in list(joints.keys()):
        joints[name] = rotate_point(joints[name])
    spine_points = [rotate_point(p) for p in spine_points]
    for name in list(refs.keys()):
        refs[name] = rotate_vector(refs[name])
    for side, per_hand in fingers.items():
        for finger in list(per_hand.keys()):
            per_hand[finger] = [rotate_point(p) for p in per_hand[finger]]

    return {
        "ok": True,
        "scale": scale,
        "yaw_deg": math.degrees(yaw),
        "joints": joints,
        "refs": refs,
        "spine_points": spine_points,
        "fingers": fingers,
        "points2d": pts,
        "synth": sorted(synth),
    }


