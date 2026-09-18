"""深度解算自检（不需要 ONNX 模型 / 不需要图片）。

思路：
  1. 造一副合成骨架的静止尺寸（metrics）与一个**已知真值姿态**（相机空间，米）；
  2. 用透视相机把真值关节投影成 COCO-133 关键点（模拟 DWPose 的输出）；
  3. 按真值深度“画”出一张深度图（可切换：线性=深度、倒数=视差、取反）；
  4. 跑 reconstruction.reconstruct(depth_mode="DEPTH")，比较解出来的 3D 关节与真值；
  5. 再检查 FLAT / 无深度图兜底 / 采样 / 预处理等边界。

用法：
  blender.exe -b --factory-startup --python "dev_tests/test_depth.py"
  可选：设置环境变量 POSE_CAPTURE_DEPTH_MODEL=<onnx 路径> 时额外验证深度模型 I/O
        （插件模型目录里已有 depth_anything_v2_small.onnx 时会自动启用）
"""
import math
import os
import sys

import bpy  # noqa: F401  （保证在 Blender 里跑，reconstruct 需要 mathutils）
import numpy as np
from mathutils import Matrix, Quaternion, Vector

ADDON_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ADDON_PARENT not in sys.path:
    sys.path.insert(0, ADDON_PARENT)

from pose_capture import coco, depth, reconstruction  # noqa: E402

IMAGE_SIZE = (1000, 1500)
FOCAL = 1.2 * max(IMAGE_SIZE)          # 与 reconstruction 的默认焦距系数一致
CX = IMAGE_SIZE[0] * 0.5
CY = IMAGE_SIZE[1] * 0.5

# 合成骨架的静止姿态（米，角色面朝 -Y）。
# pelvis / chest 对应关键点里的“胯中点 / 肩中点”，与插件的用法一致。
REST = {
    "pelvis": (0.0, 0.0, 1.00),
    "chest": (0.0, 0.0, 1.45),
    "hip.L": (0.10, 0.0, 1.00), "hip.R": (-0.10, 0.0, 1.00),
    "shoulder.L": (0.18, 0.0, 1.45), "shoulder.R": (-0.18, 0.0, 1.45),
    "elbow.L": (0.45, 0.0, 1.45), "elbow.R": (-0.45, 0.0, 1.45),
    "wrist.L": (0.72, 0.0, 1.45), "wrist.R": (-0.72, 0.0, 1.45),
    "knee.L": (0.10, 0.0, 0.55), "knee.R": (-0.10, 0.0, 0.55),
    "ankle.L": (0.10, 0.0, 0.10), "ankle.R": (-0.10, 0.0, 0.10),
    "ball.L": (0.10, -0.15, 0.05), "ball.R": (-0.10, -0.15, 0.05),
}

# 真值姿态：每个骨段额外绕世界轴转多少度（会产生真实的前后深度差）
EXTRA = {
    "chest": ((0.0, 0.0, 1.0), 25.0),          # 躯干侧身
    "upper_arm.L": ((0.0, 1.0, 0.0), -35.0),   # 抬臂
    "forearm.L": ((1.0, 0.0, 0.0), -55.0),     # 肘往前弯
    "upper_arm.R": ((0.0, 1.0, 0.0), 30.0),
    "forearm.R": ((1.0, 0.0, 0.0), 65.0),
    "thigh.R": ((1.0, 0.0, 0.0), -30.0),
    "shin.R": ((1.0, 0.0, 0.0), 45.0),
    "thigh.L": ((1.0, 0.0, 0.0), 18.0),
    "shin.L": ((1.0, 0.0, 0.0), -12.0),
    "foot.L": ((1.0, 0.0, 0.0), 20.0),
    "foot.R": ((1.0, 0.0, 0.0), -10.0),
}

# “大臂朝镜头伸、小臂回到画面平面内”：画面里肩肘几乎重合、手落在身体前面。
# 这是“手跑到身体后面”最常见的几何（手往前伸 / 拿东西 / 打字）。
TOWARD_CAMERA = {
    "upper_arm.L": ((0.0, 0.0, 1.0), -90.0),
    "forearm.L": ((0.0, 0.0, 1.0), 90.0),
}

SEGMENTS = [
    ("pelvis", "chest"),
    ("chest", "shoulder.L"), ("chest", "shoulder.R"),
    ("pelvis", "hip.L"), ("pelvis", "hip.R"),
    ("shoulder.L", "elbow.L"), ("elbow.L", "wrist.L"),
    ("shoulder.R", "elbow.R"), ("elbow.R", "wrist.R"),
    ("hip.L", "knee.L"), ("knee.L", "ankle.L"), ("ankle.L", "ball.L"),
    ("hip.R", "knee.R"), ("knee.R", "ankle.R"), ("ankle.R", "ball.R"),
]

CHECK_ROLES = ("chest", "shoulder.L", "shoulder.R", "elbow.L", "elbow.R",
               "wrist.L", "wrist.R", "hip.L", "hip.R", "knee.L", "knee.R",
               "ankle.L", "ankle.R", "ball.L", "ball.R")

FAILURES = []


def check(name, condition, detail=""):
    print("  %s %s%s" % ("✓" if condition else "✗", name,
                         ("  (%s)" % detail) if detail else ""))
    if not condition:
        FAILURES.append(name)
    return condition


# --------------------------------------------------------------------------- 合成数据
def _rotation(bone: str, extra: dict = None) -> Matrix:
    table = EXTRA if extra is None else extra
    if bone not in table:
        return Matrix.Identity(3)
    axis, degrees = table[bone]
    return Matrix.Rotation(math.radians(degrees), 3, Vector(axis))


def make_metrics() -> dict:
    rest = {name: Vector(vec) for name, vec in REST.items()}

    def length(a, b):
        return float((rest[b] - rest[a]).length)

    return {
        "errors": [],
        "pelvis": rest["pelvis"],
        "chest": rest["chest"],
        "pelvis_to_shoulders": length("pelvis", "chest"),
        "upper_arm_len": {s: length("shoulder.%s" % s, "elbow.%s" % s) for s in "LR"},
        "forearm_len": {s: length("elbow.%s" % s, "wrist.%s" % s) for s in "LR"},
        "thigh_len": {s: length("hip.%s" % s, "knee.%s" % s) for s in "LR"},
        "shin_len": {s: length("knee.%s" % s, "ankle.%s" % s) for s in "LR"},
        "foot_len": {s: length("ankle.%s" % s, "ball.%s" % s) for s in "LR"},
        "toe_len": {"L": 0.05, "R": 0.05},
        "head_len": 0.20,
        "hand_len": {"L": 0.08, "R": 0.08},
        "hip_offsets": {s: rest["hip.%s" % s] - rest["pelvis"] for s in "LR"},
        "shoulder_offsets": {s: rest["shoulder.%s" % s] - rest["chest"] for s in "LR"},
        "clavicle_offsets": {"L": None, "R": None},
        "neck_offset": Vector((0.0, 0.0, 0.10)),
        "head_offset": Vector((0.0, 0.0, 0.20)),
        "pelvis_offset_z": 0.0,
        "chest_offset_z": 0.0,
    }


def build_true_pose(distance: float = 3.5, extra: dict = None) -> dict:
    """用“旋转静止方向 + 按骨长串接”造真值姿态（相机空间：X 右、Y 向画面内、Z 上）。"""
    rest = {name: Vector(vec) for name, vec in REST.items()}
    torso = _rotation("chest", extra)
    joints = {"pelvis": Vector((0.0, distance, 0.0))}
    joints["chest"] = joints["pelvis"] + (torso @ (rest["chest"] - rest["pelvis"]))

    for side in ("L", "R"):
        joints["hip.%s" % side] = joints["pelvis"] + (
            torso @ (rest["hip.%s" % side] - rest["pelvis"]))
        joints["shoulder.%s" % side] = joints["chest"] + (
            torso @ (rest["shoulder.%s" % side] - rest["chest"]))

    chains = []
    for side in ("L", "R"):
        chains.append([
            ("upper_arm.%s" % side, "shoulder.%s" % side, "elbow.%s" % side),
            ("forearm.%s" % side, "elbow.%s" % side, "wrist.%s" % side)])
        chains.append([
            ("thigh.%s" % side, "hip.%s" % side, "knee.%s" % side),
            ("shin.%s" % side, "knee.%s" % side, "ankle.%s" % side),
            ("foot.%s" % side, "ankle.%s" % side, "ball.%s" % side)])

    for bones in chains:
        rotation = torso
        for bone, parent, child in bones:
            rest_dir = rest[child] - rest[parent]
            length = float(rest_dir.length)
            rotation = _rotation(bone, extra) @ rotation
            joints[child] = joints[parent] + (rotation @ rest_dir.normalized()) * length
    return joints


def project(point) -> np.ndarray:
    """相机空间 3D -> 像素（与 reconstruction._perspective_point 互逆）。"""
    x, y, z = float(point[0]), float(point[1]), float(point[2])
    return np.array((CX + FOCAL * x / y, CY - FOCAL * z / y))


def make_keypoints(joints, hands: bool = False, hand_bend: float = 60.0):
    """真值关节 -> COCO 133 像素关键点。

    ``hands=True`` 时额外补上手部 21 点（沿“手腕 -> 手”的方向，与测地弯曲 ``hand_bend`` 度：
    真实照片里手很少刚好顺着小臂，所以用它来验证“手到底跟的是关键点还是小臂”）。
    """
    keypoints = np.zeros((coco.TOTAL, 2), dtype=np.float32)
    scores = np.zeros(coco.TOTAL, dtype=np.float32)
    mapping = {
        coco.LEFT_SHOULDER: "shoulder.L", coco.RIGHT_SHOULDER: "shoulder.R",
        coco.LEFT_ELBOW: "elbow.L", coco.RIGHT_ELBOW: "elbow.R",
        coco.LEFT_WRIST: "wrist.L", coco.RIGHT_WRIST: "wrist.R",
        coco.LEFT_HIP: "hip.L", coco.RIGHT_HIP: "hip.R",
        coco.LEFT_KNEE: "knee.L", coco.RIGHT_KNEE: "knee.R",
        coco.LEFT_ANKLE: "ankle.L", coco.RIGHT_ANKLE: "ankle.R",
        coco.LEFT_BIG_TOE: "ball.L", coco.RIGHT_BIG_TOE: "ball.R",
    }
    for index, name in mapping.items():
        keypoints[index] = project(joints[name])
        scores[index] = 0.95

    if hands:
        for side in ("L", "R"):
            base = coco.HAND_LEFT_START if side == "L" else coco.HAND_RIGHT_START
            wrist = Vector(joints["wrist.%s" % side])
            elbow = Vector(joints["elbow.%s" % side])
            forward = (wrist - elbow).normalized()
            palm = Matrix.Rotation(math.radians(hand_bend), 3, Vector((1.0, 0.0, 0.0)))
            hand_dir = palm @ forward
            for local in range(coco.HAND_COUNT):
                step = 0.012 * (local // 5) + 0.004 * (local % 5)      # 指根到指尖依次外推
                keypoints[base + local] = project(wrist + hand_dir * (0.02 + step))
                scores[base + local] = 0.95
    return keypoints, scores


def render_depth_map(joints, convention: str = "linear", sign: float = 1.0,
                     brush: int = 3) -> np.ndarray:
    """按真值姿态画一张深度图：沿骨段插值真实深度。

    convention: linear = 与深度线性；inverse = 与 1/深度 线性（视差型）
    sign: -1 表示模型把前后判反（考验拟合能不能自己纠回来）
    """
    height, width = IMAGE_SIZE[1], IMAGE_SIZE[0]
    canvas = np.full((height, width), 0.2 if convention == "inverse" else 6.0,
                     dtype=np.float32)
    yy, xx = np.mgrid[-brush:brush + 1, -brush:brush + 1]
    mask = (xx * xx + yy * yy) <= brush * brush
    offsets = np.stack([xx[mask], yy[mask]], axis=1)

    for name_a, name_b in SEGMENTS:
        a, b = joints[name_a], joints[name_b]
        for t in np.linspace(0.0, 1.0, 240):
            point = a * (1.0 - t) + b * t
            pixel = project(point)
            value = float(point[1])
            if convention == "inverse":
                value = 1.0 / max(value, 1e-3)
            value *= sign
            xs = np.round(pixel[0]).astype(int) + offsets[:, 0]
            ys = np.round(pixel[1]).astype(int) + offsets[:, 1]
            inside = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
            canvas[ys[inside], xs[inside]] = value
    return canvas


def smear_depth(depth_map, base: float, quantize: float = 0.08) -> np.ndarray:
    """模拟单目深度模型对细肢体的“糊”：把深度量化成台阶。

    真实模型（Depth Anything V2）在细肢体上会把深度抹到旁边躯干的深度上：小臂明明朝着镜头
    伸出 26 cm，图上量出来只差 1~2 cm（甚至同档）。量化成台阶就是这种“信息被抹掉”的样子。
    """
    steps = np.round((np.asarray(depth_map, dtype=np.float32) - base) / quantize)
    return (base + steps * quantize).astype(np.float32)


def pixel_points(joints) -> dict:
    """真值 -> ``_solve_bone_lengths`` 要的 pts 字典（点名 -> 像素），与 _collect_2d 的命名一致。"""
    keypoints, _scores = make_keypoints(joints)

    def pixel(index):
        return np.asarray(keypoints[index], dtype=float)

    pairs = {
        "sh_L": coco.LEFT_SHOULDER, "sh_R": coco.RIGHT_SHOULDER,
        "el_L": coco.LEFT_ELBOW, "el_R": coco.RIGHT_ELBOW,
        "wr_L": coco.LEFT_WRIST, "wr_R": coco.RIGHT_WRIST,
        "hip_L": coco.LEFT_HIP, "hip_R": coco.RIGHT_HIP,
        "kn_L": coco.LEFT_KNEE, "kn_R": coco.RIGHT_KNEE,
        "an_L": coco.LEFT_ANKLE, "an_R": coco.RIGHT_ANKLE,
        "big_L": coco.LEFT_BIG_TOE, "big_R": coco.RIGHT_BIG_TOE,
    }
    pts = {name: pixel(index) for name, index in pairs.items()}
    pts["chest"] = (pts["sh_L"] + pts["sh_R"]) * 0.5
    pts["pelvis"] = (pts["hip_L"] + pts["hip_R"]) * 0.5
    # 头部：与 _collect_2d 一样在没有五官关键点时外推一个（这里只为了凑齐字典）
    pts["head_base"] = pts["chest"] + (pts["pelvis"] - pts["chest"]) * -0.35
    pts["head_tip"] = pts["head_base"] + np.array((0.0, -25.0))
    return pts


def compare(recon: dict, truth: dict) -> dict:
    """比较相对骨盆的 3D 关节位置（米）。"""
    origin = np.asarray(recon["joints"]["pelvis"], dtype=float)
    base = np.asarray(truth["pelvis"], dtype=float)
    return {role: float(np.linalg.norm(
        np.asarray(recon["joints"][role], dtype=float) - origin
        - (np.asarray(truth[role], dtype=float) - base))) for role in CHECK_ROLES}


# --------------------------------------------------------------------------- 用例
def solve(metrics, keypoints, scores, depth_map, options=None) -> dict:
    size = (depth_map.shape[1], depth_map.shape[0]) if depth_map is not None else None
    return reconstruction.reconstruct(
        keypoints, scores, metrics, IMAGE_SIZE,
        reconstruction.default_options(**(options or {})),
        depth_map=depth_map, depth_size=size)


def run_depth_case(title, convention, sign):
    print("\n[%s]" % title)
    metrics = make_metrics()
    truth = build_true_pose()
    keypoints, scores = make_keypoints(truth)
    depth_map = render_depth_map(truth, convention, sign)
    recon = solve(metrics, keypoints, scores, depth_map)
    if not recon.get("ok"):
        return check("%s：重建成功" % title, False, str(recon.get("message")))

    info = recon.get("depth") or {}
    check("用了深度解算", recon.get("mode") == "DEPTH", str(recon.get("mode")))
    check("参数化自动选对", info.get("transform") == convention, str(info.get("transform")))
    error = float(info.get("bone_error") or 9.9)
    check("骨长误差 < 2cm", error < 0.02, "%.4f m" % error)

    errors = compare(recon, truth)
    worst_role = max(errors, key=errors.get)
    worst = errors[worst_role]
    mean = sum(errors.values()) / len(errors)
    check("关节位置误差（平均 < 2cm 且最大 < 5cm）", mean < 0.02 and worst < 0.05,
          "平均 %.3f m，最大 %.3f m（%s）" % (mean, worst, worst_role))

    alpha = float(info.get("alpha") or 0.0)
    check("深度方向由拟合自动决定", (alpha < 0.0) == (sign < 0.0), "α=%.4f" % alpha)
    return True


def run_flat_and_fallback():
    print("\n[FLAT / 兜底]")
    metrics = make_metrics()
    truth = build_true_pose()
    keypoints, scores = make_keypoints(truth)

    flat = solve(metrics, keypoints, scores, None, {"depth_mode": "FLAT"})
    check("FLAT 模式生效", flat.get("mode") == "FLAT", str(flat.get("mode")))
    base_y = float(flat["joints"]["pelvis"][1])
    offsets = [abs(float(flat["joints"][role][1]) - base_y) for role in CHECK_ROLES]
    check("FLAT 时所有关节同一深度", max(offsets) < 1e-6, "%.2e" % max(offsets))

    bone = solve(metrics, keypoints, scores, None, {"depth_mode": "BONE"})
    check("BONE 模式可跑", bone.get("mode") == "BONE", str(bone.get("mode")))

    fallback = solve(metrics, keypoints, scores, None)
    check("没深度图时自动兜底", fallback.get("mode") == "BONE", str(fallback.get("mode")))
    check("兜底有文字说明",
          any("深度" in note for note in (fallback.get("notes") or [])),
          str(fallback.get("notes")))

    constant = np.full((IMAGE_SIZE[1], IMAGE_SIZE[0]), 1.0, dtype=np.float32)
    degenerate = solve(metrics, keypoints, scores, constant)
    check("深度图无变化时兜底", degenerate.get("mode") == "BONE",
          str(degenerate.get("mode")))

    normal = solve(metrics, keypoints, scores, render_depth_map(truth, "linear", 1.0))
    forced = solve(metrics, keypoints, scores, render_depth_map(truth, "linear", 1.0),
                   {"depth_flip": True})
    info = forced.get("depth") or {}
    normal_info = normal.get("depth") or {}
    alpha = float(info.get("alpha") or 0.0)
    check("手动反转方向时误差变大并被自动拦下",
          forced.get("mode") == "BONE" or alpha < 0.0,
          "mode=%s α=%.4f" % (forced.get("mode"), alpha))
    # 旧版这里其实是坏的：高斯-牛顿从正负初值都收敛到同一个解，候选里只有一个方向，
    # “反转”只换了个参数化，姿态根本没翻过来。现在核对“深度语义真的反了”。
    check("手动反转方向：深度语义真的反过来（不是只换参数化）",
          bool(info.get("near_is_large")) is not bool(normal_info.get("near_is_large"))
          if info and normal_info else False,
          "正常 near_is_large=%s -> 反转 near_is_large=%s"
          % (normal_info.get("near_is_large"), info.get("near_is_large")))


def run_foreshortened_case():
    """大臂朝着镜头伸（画面里肩肘几乎重合）—— 真实照片上“手跑到身体后面”的典型几何。

    深度模型给的是相对深度，量级常常比骨长小一个数量级（实测小臂 26 cm，深度图只给出
    1~2 cm 的前后差），还会把细肢体“糊”到躯干深度上。这时如果直接拿深度图的量级/符号定
    关节位置，肘->腕的方向几乎完全由（重合的）像素噪声决定，手就会被甩到身体后面去。
    回修之后：深度差量级用骨架骨长、前后符号仍来自深度图（模型没意见的段落留在平面内）。
    """
    print("\n[大臂朝镜头伸 + 深度被模型糊掉]")
    metrics = make_metrics()
    truth = build_true_pose(extra=TOWARD_CAMERA)
    check("大臂朝镜头伸：真值里肘在肩前面（这就是要保住的东西）",
          float(truth["elbow.L"][1]) < float(truth["shoulder.L"][1]) - 0.2,
          "肘-肩 Y = %+.3f m" % float(truth["elbow.L"][1] - truth["shoulder.L"][1]))
    keypoints, scores = make_keypoints(truth)
    base = float(truth["pelvis"][1])
    depth_map = smear_depth(render_depth_map(truth, "linear", 1.0), base, 0.06)

    recon = solve(metrics, keypoints, scores, depth_map, {"depth_near_large": False})
    if not recon.get("ok"):
        return check("大臂朝镜头伸：重建成功", False, str(recon.get("message")))
    check("大臂朝镜头伸：用了深度解算", recon.get("mode") == "DEPTH",
          str(recon.get("mode")))
    info = recon.get("depth") or {}
    check("大臂朝镜头伸：报了回修段数", int(info.get("repaired") or 0) > 0,
          "repaired=%s" % info.get("repaired"))

    elbow = np.asarray(recon["joints"]["elbow.L"], dtype=float)
    wrist = np.asarray(recon["joints"]["wrist.L"], dtype=float)
    shoulder = np.asarray(recon["joints"]["shoulder.L"], dtype=float)
    chest = np.asarray(recon["joints"]["chest"], dtype=float)
    check("大臂朝镜头伸：手腕仍在身体（胸）前面，没被甩到后面",
          float(wrist[1]) < float(chest[1]),
          "腕 Y=%.3f / 胸 Y=%.3f（越小越近）" % (float(wrist[1]), float(chest[1])))
    check("大臂朝镜头伸：大臂长度回到骨架骨长",
          abs(float(np.linalg.norm(elbow - shoulder)) - metrics["upper_arm_len"]["L"]) < 0.02,
          "%.3f m / 骨架 %.3f m" % (float(np.linalg.norm(elbow - shoulder)),
                                    metrics["upper_arm_len"]["L"]))
    errors = compare(recon, truth)
    worst_role = max(errors, key=errors.get)
    mean = sum(errors.values()) / len(errors)
    check("大臂朝镜头伸：关节位置误差（平均 < 2cm 且最大 < 6cm）",
          mean < 0.02 and errors[worst_role] < 0.06,
          "平均 %.3f m，最大 %.3f m（%s）" % (mean, errors[worst_role], worst_role))


def run_fallback_sign_case():
    """退回骨长解算时也要用深度图给的前后符号（老假设“手肘朝后”会把整条手臂推到背后）。

    这条单独测：深度解算被拦下（拟合不可信）时走的就是它。没深度图时行为不变（还是老假设），
    有深度图时按深度图给的符号 —— 这正是“深度图里手在前、应用后手跑到身体后面”的另一半成因。
    """
    print("\n[兜底：骨长解算也用深度图的前后符号]")
    metrics = make_metrics()
    truth = build_true_pose(extra=TOWARD_CAMERA)
    pts = pixel_points(truth)
    hands = {"L": {}, "R": {}}
    opts = reconstruction.default_options()

    def run(signs=None):
        return reconstruction._solve_bone_lengths(pts, hands, metrics, opts, IMAGE_SIZE,
                                                  True, signs=signs)["joints"]

    old = run()
    old_elbow = float(np.asarray(old["elbow.L"], dtype=float)[1])
    old_shoulder = float(np.asarray(old["shoulder.L"], dtype=float)[1])
    old_wrist = float(np.asarray(old["wrist.L"], dtype=float)[1])
    old_chest = float(np.asarray(old["chest"], dtype=float)[1])
    check("兜底（没有深度图）：还是老假设“手肘朝后” —— 手确实会在身体后面",
          old_elbow > old_shoulder and old_wrist > old_chest,
          "肘-肩 %+.3f m，腕-胸 %+.3f m（正则 = 在后面）"
          % (old_elbow - old_shoulder, old_wrist - old_chest))

    signs = {("sh_L", "el_L"): True, ("el_L", "wr_L"): None}   # 肘在肩前面 / 小臂没意见
    fixed = run(signs)
    elbow = float(np.asarray(fixed["elbow.L"], dtype=float)[1])
    shoulder = float(np.asarray(fixed["shoulder.L"], dtype=float)[1])
    wrist = float(np.asarray(fixed["wrist.L"], dtype=float)[1])
    chest = float(np.asarray(fixed["chest"], dtype=float)[1])
    check("兜底（有深度图）：手肘按深度图摆到肩前面",
          elbow < shoulder, "肘-肩 %+.3f m" % (elbow - shoulder))
    check("兜底（有深度图）：手（腕）落在身体前面", wrist < chest,
          "腕-胸 %+.3f m（越小越近）" % (wrist - chest))
    right = float(np.asarray(fixed["elbow.R"], dtype=float)[1])
    right_shoulder = float(np.asarray(fixed["shoulder.R"], dtype=float)[1])
    check("兜底：没给符号的右侧仍按老假设（互不干扰）", right > right_shoulder,
          "右肘-肩 %+.3f m" % (right - right_shoulder))


def _cost_pair(metrics, keypoints, scores, depth_map, options=None):
    """返回 (正常方向代价, 反转方向代价, 正常 α, 反转 α)：用来量“方向有多难分辨”。"""
    normal = solve(metrics, keypoints, scores, depth_map, options)
    flipped = solve(metrics, keypoints, scores, depth_map,
                    dict(options or {}, depth_flip=True))
    info, other = normal.get("depth") or {}, flipped.get("depth") or {}
    return (float(info.get("cost") or 0.0), float(other.get("cost") or 0.0),
            float(info.get("alpha") or 0.0), float(other.get("alpha") or 0.0))


def run_direction_hint_case():
    """深度方向：骨长能分辨时听拟合的，分辨不出时听物理判据（depth_near_large）。

    “深度图里手在前面、应用后手却在身体后面”最常见的成因就是这一条：人正面站、身体在深度上
    本来就扁，α 的两种符号残差几乎一样（实测真实照片上只差 0.1%），方向本来等于抛硬币。
    """
    print("\n[深度方向：拟合 vs 物理判据]")
    metrics = make_metrics()

    # 1) 辨识得出来的姿态（躯干侧身 25° + 四肢抬/弯）：错误提示不该翻掉拟合的结果
    truth = build_true_pose()
    keypoints, scores = make_keypoints(truth)
    depth_map = render_depth_map(truth, "linear", 1.0)
    cost, cost_flip, alpha, _flip_alpha = _cost_pair(metrics, keypoints, scores, depth_map)
    check("可辨识的姿势：两种方向代价差得开",
          cost_flip > cost * reconstruction.DEPTH_SIGN_TIE_RATIO,
          "%.6f vs %.6f（比值 %.3f）" % (cost, cost_flip, cost_flip / max(cost, 1e-12)))
    hinted = solve(metrics, keypoints, scores, depth_map, {"depth_near_large": True})
    check("可辨识的姿势：物理判据给错方向也翻不动拟合结果",
          bool((hinted.get("depth") or {}).get("near_is_large")) is False,
          "near_is_large=%s（真值方向是“值大 = 远”）"
          % (hinted.get("depth") or {}).get("near_is_large"))

    # 2) 挑解的规则本身：代价差在阈值内才听物理判据（直接喂候选解，避免依赖“刚好很难分辨”
    #    的合成姿势 —— 真实照片上难分辨，合成数据里想造出一样难的反而不稳）
    def candidate(cost, kind, alpha):
        return (cost, kind, {}, np.array((alpha, 3.5), dtype=np.float64))

    def hint_of(options, best):
        picked = reconstruction._apply_direction_hint(solutions, best, options)
        return reconstruction._depth_near_is_large(picked[1], float(picked[3][0]))

    solutions = [candidate(0.00100, "linear", 0.20),      # 最优：值大 = 远（linear α>0）
                 candidate(0.00105, "linear", -0.20),     # 镜像：代价只差 5%
                 candidate(0.00200, "inverse", -0.30)]
    best = solutions[0]
    check("方向提示：不给提示时不动，听拟合",
          hint_of({"depth_near_large": None}, best) is False, "值大 = 远（α>0）")
    check("方向提示：代价差在阈值内（5%）时听物理判据",
          hint_of({"depth_near_large": True}, best) is True, "值大 = 近（镜像解）")

    solutions = [candidate(0.00100, "linear", 0.20),
                 candidate(0.00130, "linear", -0.20)]     # 镜像贵 30%，超过阈值
    best = solutions[0]
    check("方向提示：代价差超过阈值（30%）时不动，还是听拟合",
          hint_of({"depth_near_large": True}, best) is False, "值大 = 远（α>0）")


def run_hand_points_case():
    """手部关键点必须参与重建，并且最终“尽量顺着小臂伸展”（hand_align + 偏角上限）。

    旧版 `place()` 只查 pts 字典，手部 21 点整体被丢掉：`hand_tip` 永远等于“沿小臂外推
    手长”、手指一根都不生成 —— 画面里手的方向跟照片不一样。
    """
    print("\n[手部关键点：手的方向 / 手指 / 沿小臂伸展]")
    metrics = make_metrics()
    truth = build_true_pose()
    keypoints, scores = make_keypoints(truth, hands=True)     # 手部关键点比小臂偏 60 度
    depth_map = render_depth_map(truth, "linear", 1.0)

    def hand_vs_forearm(recon) -> float:
        joints = recon["joints"]
        hand = Vector(np.asarray(joints["hand_tip.L"], dtype=float)
                      - np.asarray(joints["wrist.L"], dtype=float))
        fore = Vector(np.asarray(joints["wrist.L"], dtype=float)
                      - np.asarray(joints["elbow.L"], dtype=float))
        return math.degrees(hand.normalized().angle(fore.normalized()))

    raw = solve(metrics, keypoints, scores, depth_map, {"fingers": True, "hand_align": 0.0})
    half = solve(metrics, keypoints, scores, depth_map, {"fingers": True})
    straight = solve(metrics, keypoints, scores, depth_map,
                     {"fingers": True, "hand_align": 1.0})
    for title, recon in (("不拉", raw), ("默认", half), ("完全顺小臂", straight)):
        if not recon.get("ok"):
            return check("手部关键点：重建成功（%s）" % title, False,
                         str(recon.get("message")))

    fingers = half.get("fingers") or {}
    check("手部关键点：手指被生成（旧版在深度模式下永远是空的）",
          bool(fingers.get("L")) and bool(fingers.get("R")),
          "手指：%s" % {side: sorted(chain) for side, chain in fingers.items()})

    angle_raw, angle_half, angle_straight = (hand_vs_forearm(item)
                                             for item in (raw, half, straight))
    check("手部关键点：hand_align=0 时手跟的是手部关键点（不是小臂方向）",
          angle_raw > 20.0, "手 - 小臂夹角 %.1f 度" % angle_raw)
    check("手部沿小臂：默认（0.5）把偏角拉掉一半",
          abs(angle_half - angle_raw * 0.5) < 5.0,
          "%.1f 度 -> %.1f 度" % (angle_raw, angle_half))
    check("手部沿小臂：1.0 时手完全顺着小臂",
          angle_straight < 1.0, "%.2f 度" % angle_straight)

    joints = half["joints"]
    hand_len = float((Vector(joints["hand_tip.L"]) - Vector(joints["wrist.L"])).length)
    check("手部沿小臂：手长仍是骨架手长",
          abs(hand_len - metrics["hand_len"]["L"]) < 1e-6,
          "%.4f m / 骨架 %.4f m" % (hand_len, metrics["hand_len"]["L"]))

    # 手指必须跟着手掌一起转（同一个旋转、以腕为轴 => 与腕的距离不变、方向一起被拉向小臂）
    def finger_angles(recon) -> dict:
        wrist = np.asarray(recon["joints"]["wrist.L"], dtype=float)
        fore = Vector(np.asarray(recon["joints"]["wrist.L"], dtype=float)
                      - np.asarray(recon["joints"]["elbow.L"], dtype=float)).normalized()
        out = {}
        for name, chain in (recon.get("fingers") or {}).get("L", {}).items():
            out[name] = [(float(np.linalg.norm(np.asarray(point, dtype=float) - wrist)),
                          math.degrees(Vector(np.asarray(point, dtype=float)
                                              - wrist).normalized().angle(fore)))
                         for point in chain]
        return out

    before, after = finger_angles(raw), finger_angles(straight)
    same_span = bool(before) and sorted(before) == sorted(after)
    turned = bool(before)
    for name, rows in before.items():
        for index, (span, angle) in enumerate(rows):
            new_span, new_angle = after[name][index]
            same_span = same_span and abs(new_span - span) < 1e-6
            turned = turned and (new_angle <= angle + 1e-6)
    check("手部沿小臂：手指跟着手掌一起转（与腕的距离不变）", same_span,
          "指节 %d 个" % sum(len(rows) for rows in before.values()))
    check("手部沿小臂：手指方向也被拉向小臂（不是手掌转了手指没动）", turned,
          "指根 %.1f 度 -> %.1f 度" % (before["middle"][0][1], after["middle"][0][1]))


def run_hand_align_case():
    """手部对齐的数学：`_bend_toward` 必须“朝参照方向转、按比例、夹上限”。

    这里既量“转掉了多少度”，也量“离参照方向还剩多少度” —— 后者才能证明是朝参照转，
    而不是转到另一边去（参照搞反过一次）。
    """
    print("\n[手部沿小臂：混合比例 / 偏角上限]")
    axis = np.array((1.0, 0.0, 0.0))
    reference = np.array((0.0, 1.0, 0.0))     # 与 axis 正好 90 度

    def turned(blend, max_deg=90.0):
        turn = reconstruction._bend_toward(axis, reference, blend, max_deg)
        result = turn @ Vector(axis)
        return (math.degrees(Vector(axis).angle(result)),
                math.degrees(result.angle(Vector(reference))))

    check("混合 1.0 -> 完全对齐参照（转 90 度，剩 0 度）",
          abs(turned(1.0)[0] - 90.0) < 0.01 and turned(1.0)[1] < 0.01,
          "转了 %.2f 度，还剩 %.3f 度" % turned(1.0))
    check("混合 0.5 -> 偏角减半（转 45 度，剩 45 度）",
          abs(turned(0.5)[0] - 45.0) < 0.1 and abs(turned(0.5)[1] - 45.0) < 0.1,
          "转了 %.2f 度，还剩 %.2f 度" % turned(0.5))
    check("混合 0 -> 不按比例缩，只夹上限（剩 30 度）",
          abs(turned(0.0, 30.0)[1] - 30.0) < 0.1,
          "转了 %.2f 度，还剩 %.2f 度" % turned(0.0, 30.0))
    check("偏角上限压得住混合后的角度（90×0.5=45 > 30 -> 剩 30 度）",
          abs(turned(0.5, 30.0)[1] - 30.0) < 0.1,
          "转了 %.2f 度，还剩 %.2f 度" % turned(0.5, 30.0))
    check("偏角没到上限时不夹（混合 0.9 -> 转 81 度，剩 9 度）",
          abs(turned(0.9, 30.0)[0] - 81.0) < 0.1 and abs(turned(0.9, 30.0)[1] - 9.0) < 0.1,
          "转了 %.2f 度，还剩 %.2f 度" % turned(0.9, 30.0))
    check("混合越大剩下的偏角越小（单调）",
          turned(0.3, 90.0)[1] > turned(0.6, 90.0)[1] > turned(0.9, 90.0)[1],
          "%.1f > %.1f > %.1f" % (turned(0.3, 90.0)[1], turned(0.6, 90.0)[1],
                                  turned(0.9, 90.0)[1]))
    check("方向退化（长度 0）-> 不转",
          reconstruction._bend_toward(np.zeros(3), reference, 1.0, 30.0)
          == Quaternion((1.0, 0.0, 0.0, 0.0)), "单位四元数")
    check("本来就算齐了 -> 不转",
          reconstruction._bend_toward(axis, axis * 2.0, 1.0, 30.0)
          == Quaternion((1.0, 0.0, 0.0, 0.0)), "单位四元数")
    opposite = reconstruction._bend_toward(axis, -axis, 1.0, 30.0)
    check("正好反向（180 度）也能转到位（退化轴不炸）",
          math.degrees(Vector(opposite @ Vector(axis)).angle(Vector(-axis))) < 0.01,
          "转了 %.1f 度，落在参照方向上" % math.degrees(opposite.angle))


def run_utility_checks():
    print("\n[采样 / 预处理]")
    width, height = depth.target_size(1920, 1080)
    check("长边缩到 518 以内且都是 14 的倍数",
          max(width, height) <= depth.INPUT_SIZE and width % 14 == 0 and height % 14 == 0,
          "%dx%d" % (width, height))

    ramp = np.tile(np.arange(60, dtype=np.float32)[:, None], (1, 80))
    values = depth.sample_many(ramp, (80, 60), (160, 120),
                               [(40.0, 30.0), None, (0.0, 0.0)])
    check("按像素采样（含 None）",
          abs(float(values[0]) - 15.0) < 1.0 and np.isnan(values[1])
          and float(values[2]) <= 1.0, str(np.round(values, 2)))

    colored = depth.colorize(ramp)
    check("伪彩色输出合法", colored.shape == (60, 80, 3) and colored.dtype == np.uint8)
    # ramp 行号越大 = 深度值越大；near_is_large=True 表示值大 = 近
    near_rgb = colored[-1].mean(axis=0)        # 值大 = 近
    far_rgb = colored[0].mean(axis=0)          # 值小 = 远
    print("    近处 RGB=%s 远处 RGB=%s" % (np.round(near_rgb, 1), np.round(far_rgb, 1)))
    check("伪彩约定：近 = 蓝、远 = 红（和灰度“越白=越远”配套）",
          near_rgb[2] > near_rgb[0] + 100 and far_rgb[0] > far_rgb[2] + 100,
          "近=%s 远=%s" % (np.round(near_rgb, 1), np.round(far_rgb, 1)))
    flipped = depth.colorize(ramp, near_is_large=False)
    flipped_near = flipped[0].mean(axis=0)
    check("near_is_large=False 时“值大”改判成远，色带整体翻过来",
          flipped_near[2] > flipped_near[0] + 100, str(np.round(flipped_near, 1)))
    close = depth.nearness(ramp)
    check("nearness 归一到 0~1 且值大 = 近",
          abs(float(close[-1, 0]) - 1.0) < 1e-6 and abs(float(close[0, 0])) < 1e-6,
          "%.6f~%.6f" % (float(close[0, 0]), float(close[-1, 0])))

    preview = depth.preview_image(np.zeros((120, 160, 3), dtype=np.uint8), ramp)
    check("预览尺寸回到原图", preview.shape == (120, 160, 3), str(preview.shape))


def run_onnx_check():
    print("\n[深度模型 I/O（可选）]")
    path = os.environ.get("POSE_CAPTURE_DEPTH_MODEL", "")
    if not path:
        try:
            from pose_capture import utils
            candidate = os.path.join(utils.models_dir(),
                                     depth.model_filename(depth.DEFAULT_VARIANT))
            path = candidate if os.path.isfile(candidate) else ""
        except Exception:  # noqa: BLE001
            path = ""
    if not path or not os.path.isfile(path):
        print("  - 跳过：没找到 ONNX 模型（设 POSE_CAPTURE_DEPTH_MODEL 可启用）")
        return

    from pose_capture import utils
    utils.ensure_libs_on_path()
    rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    rgb[:, :, 1] = np.linspace(20, 220, 640, dtype=np.uint8)[None, :]
    estimator = depth.DepthEstimator(path)
    values, size = estimator.predict(rgb)
    check("输出尺寸与模型输入一致", values.shape == (size[1], size[0]),
          "%s vs %s" % (values.shape, size))
    check("输出有限且非恒定",
          bool(np.isfinite(values).all())
          and float(values.max() - values.min()) > 1e-6,
          "min=%.3f max=%.3f" % (float(values.min()), float(values.max())))


def main():
    print("深度解算自检：合成真值姿态 + 合成深度图")
    run_depth_case("线性深度图", "linear", 1.0)
    run_depth_case("视差（1/深度）图", "inverse", 1.0)
    run_depth_case("前后取反的深度图", "linear", -1.0)
    run_flat_and_fallback()
    run_foreshortened_case()
    run_fallback_sign_case()
    run_direction_hint_case()
    run_hand_points_case()
    run_hand_align_case()
    run_utility_checks()
    run_onnx_check()
    print("")
    if FAILURES:
        print("结果：未通过 ✗  (%d 项失败：%s)" % (len(FAILURES), "；".join(FAILURES)))
        return 1
    print("结果：通过 ✓")
    return 0


if __name__ == "__main__":
    sys.exit(main())