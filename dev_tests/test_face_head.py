"""五官 -> 头 / 脖子 自检（合成关键点，不需要模型 / 不需要图片）。

思路：
  1. 造一副合成骨架的静止尺寸（metrics）与一张**已知头部姿态**的脸：五官（68 点的分组，
     见 coco.FACE_GROUPS）按头部局部坐标系摆好，再整体 yaw / pitch / roll；
  2. 投影成 COCO-133 关键点（模拟 DWPose 的输出），按真值“画”一张深度图；
  3. 跑 reconstruction（BONE / DEPTH 两种模式），检查：
       * head 骨骼方向 = 五官的上轴（下巴 -> 眼睛 / 眉毛 / 嘴角投票）；
       * eye_line 参照轴 = 头部的侧轴，符号与肩线一致（背对镜头时不拧 180°）；
       * 转头（yaw）真的落在 eye_line 上、头基座与脖子方向 = 胸上 -> 耳线；
       * 耳朵只有一只 / 明显检歪时，头基座改用五官估；
  4. 再用 Rigify 元骨架验证 `metrics["ref"]["eye_line"]` 存在且**真的驱动 head 的扭转**
     （旧版缺这个键，solve_orientation 会静默跳过扭转 —— 照片里的转头 / 歪头全丢）。

用法：
  blender.exe -b --factory-startup --python "dev_tests/test_face_head.py"
"""
import math
import os
import sys

import bpy  # noqa: F401  （保证在 Blender 里跑；rigify 部分要它）
import numpy as np
from mathutils import Matrix, Vector

ADDON_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ADDON_PARENT not in sys.path:
    sys.path.insert(0, ADDON_PARENT)

from pose_capture import coco, reconstruction, retarget, rigify_map  # noqa: E402

IMAGE = (1000, 1500)
FOCAL = 1.2 * max(IMAGE)
CX, CY = IMAGE[0] * 0.5, IMAGE[1] * 0.5
DISTANCE = 5.0              # 人在镜头前 5 m（透视比例 360 像素/米）

FAILURES = []


def check(name, condition, detail=""):
    print("  %s %s%s" % ("✓" if condition else "✗", name,
                         ("  (%s)" % detail) if detail else ""))
    if not condition:
        FAILURES.append(name)
    return condition


def project(point):
    point = np.asarray(point, dtype=float)
    return np.array((CX + FOCAL * point[0] / point[1], CY - FOCAL * point[2] / point[1]))


def rotation(yaw=0.0, pitch=0.0, roll=0.0):
    """头部旋转：yaw 绕世界 Z、pitch 绕世界 X、roll 绕面部前向(-Y)。"""
    matrix = np.eye(3)
    for axis, degrees in (((0.0, 0.0, 1.0), yaw), ((1.0, 0.0, 0.0), pitch),
                          ((0.0, 1.0, 0.0), roll)):
        angle = math.radians(degrees)
        axis = np.asarray(axis, dtype=float)
        cross = np.array(((0.0, -axis[2], axis[1]),
                          (axis[2], 0.0, -axis[0]),
                          (-axis[1], axis[0], 0.0)))
        matrix = (np.eye(3) + math.sin(angle) * cross
                  + (1.0 - math.cos(angle)) * (cross @ cross)) @ matrix
    return matrix


# 静止骨架（相机空间：X 右、Y 向画面内、Z 上；人物面朝镜头 = 面朝 -Y）。
# 手臂 / 脚在前后方向上错开：深度解算的“深度跨度”靠它们撑开，全平的话归一化深度会被夹成
# 同一个值，五官的前后分量就没了（这一点和真实照片上的判据一致）。
REST = {
    "pelvis": (0.0, 0.0, 1.00), "chest": (0.0, -0.02, 1.45), "head_base": (0.0, -0.01, 1.60),
    "hip.L": (0.10, 0.02, 1.00), "hip.R": (-0.10, 0.02, 1.00),
    "shoulder.L": (0.18, -0.04, 1.45), "shoulder.R": (-0.18, -0.04, 1.45),
    "elbow.L": (0.42, -0.16, 1.36), "elbow.R": (-0.42, -0.16, 1.36),
    "wrist.L": (0.55, -0.34, 1.30), "wrist.R": (-0.55, -0.34, 1.30),
    "knee.L": (0.10, 0.06, 0.55), "knee.R": (-0.10, 0.06, 0.55),
    "ankle.L": (0.10, 0.10, 0.10), "ankle.R": (-0.10, 0.10, 0.10),
    "ball.L": (0.10, -0.05, 0.05), "ball.R": (-0.10, -0.05, 0.05),
}
NECK_OFFSET_Z = 0.10        # chest -> neck 骨起点
HEAD_CENTER_Z = 0.0178      # 眼线（头部中心）比耳线高多少 ≈ 0.15 ×（眼睛 -> 下巴）

# 五官相对“眼睛线（头部中心）”的偏移，按 (侧轴 Fs=人物右->左, 前向 Ff=面部朝外, 上轴 Fu)
FACE_OFFSETS = {
    "chin": (0.0, 0.045, -0.115),
    "eye_l": (-0.032, 0.075, 0.0), "eye_r": (0.032, 0.075, 0.0),
    "eye_l_inner": (-0.012, 0.075, 0.005), "eye_r_inner": (0.012, 0.075, 0.005),
    "eye_l_outer": (-0.045, 0.070, 0.0), "eye_r_outer": (0.045, 0.070, 0.0),
    "brow_l": (-0.030, 0.075, 0.042), "brow_r": (0.030, 0.075, 0.042),
    "nose_tip": (0.0, 0.105, -0.020), "nose_bridge": (0.0, 0.085, 0.020),
    "nose_bottom": (0.0, 0.095, -0.035),
    "mouth_l": (-0.025, 0.075, -0.070), "mouth_r": (0.025, 0.075, -0.070),
    "mouth_mid": (0.0, 0.075, -0.070),
    "jaw_l": (-0.062, 0.015, -0.055), "jaw_r": (0.062, 0.015, -0.055),
}
EAR_OFFSETS = {"L": (0.085, 0.010, -HEAD_CENTER_Z), "R": (-0.085, 0.010, -HEAD_CENTER_Z)}


def make_metrics():
    rest = {name: Vector(vec) for name, vec in REST.items()}

    def length(a, b):
        return float((rest[b] - rest[a]).length)

    return {
        "errors": [],
        "pelvis": rest["pelvis"], "chest": rest["chest"],
        "pelvis_to_shoulders": length("pelvis", "chest"),
        "upper_arm_len": {s: length("shoulder.%s" % s, "elbow.%s" % s) for s in "LR"},
        "forearm_len": {s: length("elbow.%s" % s, "wrist.%s" % s) for s in "LR"},
        "thigh_len": {s: length("hip.%s" % s, "knee.%s" % s) for s in "LR"},
        "shin_len": {s: length("knee.%s" % s, "ankle.%s" % s) for s in "LR"},
        "foot_len": {s: length("ankle.%s" % s, "ball.%s" % s) for s in "LR"},
        "toe_len": {"L": 0.05, "R": 0.05},
        "head_len": 0.15,
        "hand_len": {"L": 0.08, "R": 0.08},
        "hip_offsets": {s: rest["hip.%s" % s] - rest["pelvis"] for s in "LR"},
        "shoulder_offsets": {s: rest["shoulder.%s" % s] - rest["chest"] for s in "LR"},
        "clavicle_offsets": {"L": None, "R": None},
        "neck_offset": Vector((0.0, 0.0, NECK_OFFSET_Z)),
        "head_offset": Vector((0.0, 0.0, HEAD_CENTER_Z)),
        "pelvis_offset_z": 0.0,
        "chest_offset_z": 0.0,
    }


def build(yaw=0.0, pitch=0.0, roll=0.0, pivot="base", ears="both", jitter=0.0, seed=7):
    """造一帧合成数据：头部按 yaw/pitch/roll 转，其余身体不动。"""
    rot = rotation(yaw, pitch, roll)
    base = np.array(REST["head_base"], dtype=float) + np.array((0.0, DISTANCE, 0.0))
    origin = base.copy()
    if pivot == "neck":
        origin = (np.array(REST["chest"], dtype=float) + np.array((0.0, DISTANCE, 0.0))
                  + np.array((0.0, 0.0, NECK_OFFSET_Z)))
    base_local = base - origin

    def head_point(offset):
        local = np.array((offset[0], -offset[1], offset[2]))     # Ff = 面部朝外 = -Y
        return origin + rot @ (base_local + np.array((0.0, 0.0, HEAD_CENTER_Z)) + local)

    rng = np.random.default_rng(seed)
    landmarks = {}
    for key, offset in FACE_OFFSETS.items():
        point = head_point(offset)
        if jitter:
            point = point + rng.normal(0.0, jitter, size=3)
        landmarks[key] = point
    for side in ("L", "R"):
        landmarks["ear_%s" % side] = head_point(EAR_OFFSETS[side])
    for name, key_a, key_b in (("eye_mid", "eye_l", "eye_r"),
                               ("brow_mid", "brow_l", "brow_r"),
                               ("jaw_mid", "jaw_l", "jaw_r")):
        landmarks[name] = (landmarks[key_a] + landmarks[key_b]) * 0.5
    if ears == "bad":       # 耳朵被检歪：整体横向挪 0.3 m（远超 FACE_EAR_TOLERANCE）
        for side in ("L", "R"):
            landmarks["ear_%s" % side] = landmarks["ear_%s" % side] + np.array((0.30, 0.0, 0.0))

    joints3d = {name: np.array(vec, dtype=float) + np.array((0.0, DISTANCE, 0.0))
                for name, vec in REST.items()}
    joints3d["head_base"] = origin + rot @ base_local

    keypoints = np.zeros((coco.TOTAL, 2), dtype=np.float32)
    scores = np.zeros(coco.TOTAL, dtype=np.float32)

    def put(index, point, score=0.9):
        keypoints[index] = project(point)
        scores[index] = score

    for key, indexes in coco.FACE_GROUPS.items():
        if key not in landmarks:
            continue
        for index in indexes:
            put(index, landmarks[key])

    body = {
        coco.NOSE: head_point((0.0, 0.095, 0.0)),
        coco.LEFT_EYE: head_point((0.035, 0.075, 0.020)),
        coco.RIGHT_EYE: head_point((-0.035, 0.075, 0.020)),
        coco.LEFT_SHOULDER: joints3d["shoulder.L"], coco.RIGHT_SHOULDER: joints3d["shoulder.R"],
        coco.LEFT_ELBOW: joints3d["elbow.L"], coco.RIGHT_ELBOW: joints3d["elbow.R"],
        coco.LEFT_WRIST: joints3d["wrist.L"], coco.RIGHT_WRIST: joints3d["wrist.R"],
        coco.LEFT_HIP: joints3d["hip.L"], coco.RIGHT_HIP: joints3d["hip.R"],
        coco.LEFT_KNEE: joints3d["knee.L"], coco.RIGHT_KNEE: joints3d["knee.R"],
        coco.LEFT_ANKLE: joints3d["ankle.L"], coco.RIGHT_ANKLE: joints3d["ankle.R"],
        coco.LEFT_BIG_TOE: joints3d["ball.L"], coco.RIGHT_BIG_TOE: joints3d["ball.R"],
        coco.LEFT_SMALL_TOE: joints3d["ball.L"], coco.RIGHT_SMALL_TOE: joints3d["ball.R"],
    }
    for side in ("L", "R"):
        index = coco.LEFT_EAR if side == "L" else coco.RIGHT_EAR
        if ears == "one" and side == "R":
            continue            # 右耳没检出（分数留在 0）
        body[index] = landmarks["ear_%s" % side]
    for index, point in body.items():
        put(index, point)

    return {"keypoints": keypoints, "scores": scores, "joints3d": joints3d,
            "landmarks": landmarks, "rot": rot,
            "neck_start": (np.array(REST["chest"], dtype=float)
                           + np.array((0.0, DISTANCE, NECK_OFFSET_Z)))}


def render_depth(joints3d, landmarks):
    """深度图：身体各段 + 头部整块 + 五官点（值 = -相机空间 Y，即“值大 = 近”）。"""
    canvas = np.full((IMAGE[1], IMAGE[0]), 6.0, dtype=np.float32)
    yy, xx = np.mgrid[-14:15, -14:15]
    big = (xx * xx + yy * yy) <= 14 * 14
    big_offsets = np.stack([xx[big], yy[big]], axis=1)
    small = (xx * xx + yy * yy) <= 2 * 2
    small_offsets = np.stack([xx[small], yy[small]], axis=1)

    def paint(point, offsets):
        pixel = project(point)
        xs = np.round(pixel[0]).astype(int) + offsets[:, 0]
        ys = np.round(pixel[1]).astype(int) + offsets[:, 1]
        inside = (xs >= 0) & (xs < IMAGE[0]) & (ys >= 0) & (ys < IMAGE[1])
        canvas[ys[inside], xs[inside]] = float(np.asarray(point, dtype=float)[1])

    segments = [("pelvis", "chest"), ("pelvis", "hip.L"), ("pelvis", "hip.R"),
                ("chest", "shoulder.L"), ("chest", "shoulder.R"),
                ("shoulder.L", "elbow.L"), ("elbow.L", "wrist.L"),
                ("shoulder.R", "elbow.R"), ("elbow.R", "wrist.R"),
                ("hip.L", "knee.L"), ("knee.L", "ankle.L"), ("ankle.L", "ball.L"),
                ("hip.R", "knee.R"), ("knee.R", "ankle.R"), ("ankle.R", "ball.R"),
                ("chest", "head_base")]
    for name_a, name_b in segments:
        a = np.asarray(joints3d[name_a], dtype=float)
        b = np.asarray(joints3d[name_b], dtype=float)
        for t in np.linspace(0.0, 1.0, 240):
            paint(a * (1.0 - t) + b * t, big_offsets)
    # 头部整块：真实深度模型在头上给出的是一整片（头基座像素落在头发上也拿得到头深）
    head_center = np.asarray(joints3d["head_base"], dtype=float)
    for dx in np.linspace(-1.0, 1.0, 9):
        for dz in np.linspace(-1.0, 1.0, 9):
            paint(head_center + np.array((dx * 0.085, 0.0, dz * 0.085)), big_offsets)
    for point in landmarks.values():
        paint(point, small_offsets)
    return (-canvas).astype(np.float32)


def expected_axis(landmarks, pairs, in_plane=False):
    """独立实现的“成对点投票”（只按几何算，不看插件内部）。in_plane = 只留画面内的分量。"""
    total = np.zeros(3)
    for key_a, key_b, weight in pairs:
        a = np.asarray(landmarks[key_a], dtype=float)
        b = np.asarray(landmarks[key_b], dtype=float)
        vector = b - a
        span = float(np.linalg.norm(vector))
        if span < 1e-9:
            continue
        total += vector / span * (float(weight) * span)
    span = float(np.linalg.norm(total))
    if span <= 1e-9:
        return None
    total = total / span
    if in_plane:
        total = np.array((total[0], 0.0, total[2]))
        total = total / max(float(np.linalg.norm(total)), 1e-12)
    return total


def expected_axis_plane(landmarks, pairs):
    """画面平面（正交模式）里的“成对点投票”：像素方向 (du, dv) -> 世界 (x, z) = (du, -dv)。

    正交模式本来就没有深度，但**透视会把侧向转头的眼线压斜**（近的眼角投影更低），
    这一点是像素里真实存在的信息，平面模式也确实吃到了，所以期望值要按像素算。
    """
    total = np.zeros(2)
    for key_a, key_b, weight in pairs:
        a, b = project(landmarks[key_a]), project(landmarks[key_b])
        vector = b - a
        span = float(np.linalg.norm(vector))
        if span < 1e-9:
            continue
        total += vector / span * (float(weight) * span)
    span = float(np.linalg.norm(total))
    if span <= 1e-9:
        return None
    total = total / span
    return np.array((total[0], 0.0, -total[1]))


def angle_deg(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    cosine = float(np.dot(a, b)) / max(float(np.linalg.norm(a)) * float(np.linalg.norm(b)), 1e-12)
    return math.degrees(math.acos(min(max(cosine, -1.0), 1.0)))


def solve(metrics, data, mode):
    depth = render_depth(data["joints3d"], data["landmarks"]) if mode == "DEPTH" else None
    # 深度图给的是 -Y（值大 = 近）；生产流程里这个物理方向由 operators 算好塞进选项，
    # 这里直接给真值，避免拟合在“前后翻过来”的两个解之间抛硬币
    options = reconstruction.default_options(depth_mode=mode, depth_near_large=True)
    return reconstruction.reconstruct(
        data["keypoints"], data["scores"], metrics, IMAGE, options,
        depth_map=depth, depth_size=(IMAGE[0], IMAGE[1]) if depth is not None else None)


# --------------------------------------------------------------------------- 用例
def test_face_frame():
    print("\n[五官坐标系：上轴 / 侧轴 / 头基座]")
    thr = reconstruction.default_options()["score_thr"]
    for yaw, pitch, roll in ((0.0, 0.0, 0.0), (35.0, 0.0, 0.0), (0.0, 25.0, 0.0),
                             (0.0, 0.0, 20.0), (40.0, -20.0, 15.0)):
        title = "yaw=%g pitch=%g roll=%g" % (yaw, pitch, roll)
        data = build(yaw=yaw, pitch=pitch, roll=roll)
        face = reconstruction._face_frame(data["keypoints"], data["scores"], thr)
        check("%s：认出五官" % title, face["reliable"], "组数 %d" % face["count"])
        if not face["reliable"]:
            continue
        # 上轴：像素平面的投票（v 向下，翻成“向上为正”再比）
        axis = expected_axis(data["landmarks"], coco.FACE_UP_PAIRS)
        pixel_up = np.array((axis[0], -axis[2]))
        pixel_up = pixel_up / max(float(np.linalg.norm(pixel_up)), 1e-12)
        error = angle_deg(face["up_dir"], pixel_up)
        check("%s：上轴角误差 < 2°" % title, error < 2.0, "%.2f°" % error)
        check("%s：侧轴指向画面右（正脸时 = 人物右 -> 左）" % title, face["side_dir"][0] > 0.5,
              str(np.round(face["side_dir"], 3)))
        # 头基座：独立判据 = 真实耳朵中点投影
        ear_mid = (data["landmarks"]["ear_L"] + data["landmarks"]["ear_R"]) * 0.5
        offset = float(np.linalg.norm(project(ear_mid) - face["base"]))
        check("%s：头基座落在真实耳线附近（< 0.5 脸长）" % title, offset < 0.5 * face["face_len"],
              "%.1f px（脸长 %.1f px）" % (offset, face["face_len"]))

    data = build(yaw=20.0, pitch=10.0, jitter=0.005)
    face = reconstruction._face_frame(data["keypoints"], data["scores"], thr)
    axis = expected_axis(data["landmarks"], coco.FACE_UP_PAIRS)
    pixel_up = np.array((axis[0], -axis[2]))
    pixel_up = pixel_up / max(float(np.linalg.norm(pixel_up)), 1e-12)
    error = angle_deg(face["up_dir"], pixel_up)
    check("五官带 5mm 抖动时上轴仍稳（< 3°）", error < 3.0, "%.2f°" % error)


def test_head_base_source():
    print("\n[头基座来源：耳朵 / 五官 / 外推]")
    opts = reconstruction.default_options()

    def collect(data):
        return reconstruction._collect_2d(data["keypoints"], data["scores"], opts)

    data = build()
    collected = collect(data)
    check("两只耳朵都在：用耳朵中点", collected["face"]["head_base_source"] == "ears",
          collected["face"]["head_base_source"])
    ear_mid = (project(data["landmarks"]["ear_L"]) + project(data["landmarks"]["ear_R"])) * 0.5
    check("头基座 = 耳朵中点",
          float(np.linalg.norm(collected["points"]["head_base"] - ear_mid)) < 1e-3)

    collected = collect(build(ears="one"))
    check("只检出一只耳朵：改用五官估", collected["face"]["head_base_source"] == "face",
          collected["face"]["head_base_source"])

    bad = build(ears="bad")
    collected = collect(bad)
    check("耳朵和五官差 0.3m（检歪）：改用五官估",
          collected["face"]["head_base_source"] == "face",
          collected["face"]["head_base_source"])
    ear_bad = (project(bad["landmarks"]["ear_L"]) + project(bad["landmarks"]["ear_R"])) * 0.5
    check("改用五官后头基座不再跟着歪耳朵跑",
          float(np.linalg.norm(np.asarray(collected["points"]["head_base"], dtype=float)
                               - ear_bad)) > 20.0)

    plain = build(ears="one")
    for index in coco.FACE_ALL:
        plain["scores"][index] = 0.0
    plain["scores"][coco.LEFT_EAR] = 0.0
    plain["scores"][coco.RIGHT_EAR] = 0.0
    collected = collect(plain)
    check("没有耳朵也没有五官：退回眼睛中点", collected["face"]["head_base_source"] == "eye",
          collected["face"]["head_base_source"])


def test_reconstruct():
    print("\n[重建：head 方向 / eye_line / 脖子方向]")
    metrics = make_metrics()
    for mode in ("BONE", "DEPTH"):
        for yaw, pitch, roll, pivot in ((0.0, 0.0, 0.0, "base"), (35.0, 0.0, 0.0, "base"),
                                        (0.0, 25.0, 0.0, "neck"), (0.0, 0.0, 20.0, "base"),
                                        (25.0, -15.0, 10.0, "neck")):
            title = "%s yaw=%g pitch=%g roll=%g pivot=%s" % (mode, yaw, pitch, roll, pivot)
            data = build(yaw=yaw, pitch=pitch, roll=roll, pivot=pivot)
            recon = solve(metrics, data, mode)
            if not recon.get("ok"):
                check("%s：重建成功" % title, False, str(recon.get("message")))
                continue
            check("%s：模式正确" % title, recon["mode"] == mode, recon["mode"])
            joints, refs = recon["joints"], recon["refs"]

            # 1) head 方向 = 五官的上轴（下巴 -> 眼睛 / 眉毛 / 嘴角投票）
            head_dir = np.asarray(joints["head_tip"], dtype=float) - np.asarray(
                joints["head_base"], dtype=float)
            if mode == "BONE":
                head_dir = np.array((head_dir[0], 0.0, head_dir[2]))
                expected = expected_axis(data["landmarks"], coco.FACE_UP_PAIRS, in_plane=True)
            else:
                expected = expected_axis(data["landmarks"], coco.FACE_UP_PAIRS)
            error = angle_deg(head_dir, expected)
            check("%s：head 方向 = 五官上轴（< 6°）" % title, error < 6.0, "%.2f°" % error)

            # 2) 点头 / 仰头 = head 方向的前后分量（旧做法只看“额头发际”那一个像素的深度，
            #    这一分量最容易被头发 / 背景带跑）；BONE 模式没有深度，跳过
            if pitch != 0.0 and mode == "DEPTH":
                want = expected / np.linalg.norm(expected)
                got = head_dir / np.linalg.norm(head_dir)
                check("%s：点头体现在前后分量上（真值 Y%+.2f）" % (title, want[1]),
                      float(want[1] * got[1]) > 0.0 and abs(got[1] - want[1]) < 0.35,
                      "实际 Y%+.3f" % got[1])

            # 3) eye_line：符号统一到肩线，且 = 头部侧轴（retarget 里 head 的 roll 参照轴）
            shoulder = np.asarray(refs["shoulder_line"], dtype=float)
            eye = np.asarray(refs["eye_line"], dtype=float)
            check("%s：eye_line 与肩线同向（背对镜头也不拧 180°）" % title,
                  float(np.dot(eye, shoulder)) > 0.0)
            if mode == "BONE":
                eye_cmp = np.array((eye[0], 0.0, eye[2]))
                expected_side = expected_axis_plane(data["landmarks"], coco.FACE_SIDE_PAIRS)
            else:
                eye_cmp = eye
                expected_side = expected_axis(data["landmarks"], coco.FACE_SIDE_PAIRS)
            error = angle_deg(eye_cmp, expected_side)
            check("%s：eye_line = 头部侧轴（< 6°）" % title, error < 6.0, "%.2f°" % error)
            if yaw and mode == "DEPTH":
                turn = angle_deg(eye, shoulder)
                check("%s：转头 %.0f° 落在 eye_line 上（±8°）" % (title, yaw),
                      abs(turn - abs(yaw)) < 8.0, "%.1f°" % turn)
            elif yaw:
                check("%s：没有深度时 eye_line 至少转对方向" % title,
                      angle_deg(eye, shoulder) > 0.5,
                      "%.1f°（真值 %.0f°）" % (angle_deg(eye, shoulder), abs(yaw)))
            elif pitch == 0.0 and roll == 0.0:
                check("%s：正脸时 eye_line 与肩线重合（< 6°）" % title,
                      angle_deg(eye, shoulder) < 6.0, "%.2f°" % angle_deg(eye, shoulder))

            # 4) 脖子方向 = 脖子根 -> 头基座（头颈链靠它定）；骨长兜底模式的前后是“猜”的，
            #    只比画面平面内的方向
            neck_dir = np.asarray(joints["head_base"], dtype=float) - np.asarray(
                joints["neck_start"], dtype=float)
            want_dir = np.asarray(data["joints3d"]["head_base"], dtype=float) - np.asarray(
                data["neck_start"], dtype=float)
            if mode == "BONE":
                neck_dir = np.array((neck_dir[0], 0.0, neck_dir[2]))
                want_dir = np.array((want_dir[0], 0.0, want_dir[2]))
            error = angle_deg(neck_dir, want_dir)
            check("%s：脖子方向 = 脖子根 -> 头基座（< 12°）" % title, error < 12.0,
                  "%.2f°" % error)

            # 5) 头基座：绝对距离不可知（β 由像素比例估出来），比的是相对胸口的位移
            want = (np.asarray(data["joints3d"]["head_base"], dtype=float)
                    - np.asarray(data["joints3d"]["chest"], dtype=float))
            got = (np.asarray(joints["head_base"], dtype=float)
                   - np.asarray(joints["chest"], dtype=float))
            if mode == "BONE":
                want = np.array((want[0], 0.0, want[2]))
                got = np.array((got[0], 0.0, got[2]))
            offset = float(np.linalg.norm(got - want))
            check("%s：头基座相对胸口的位置误差 < 3cm" % title, offset < 0.03,
                  "%.4f m" % offset)


def test_rigify_head_twist():
    """Rigify：静止侧的 eye_line 参照轴**必须存在**，否则 head 的扭转会被静默跳过。

    旧版 rigify_map 只写了 hip_line / shoulder_line，retarget 里 head 用的却是 "eye_line"，
    solve_orientation 拿到 rest_ref=None 就直接返回摆动结果 —— 照片里的转头 / 歪头整体丢失
    （头永远朝着静止方向）。这里把“键存在”和“键真的驱动扭转”都测一遍。
    """
    print("\n[Rigify：静止眼线 -> head 扭转（转头 / 歪头）]")
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.preferences.addon_enable(module="rigify")
    bpy.ops.object.armature_human_metarig_add()
    arm = bpy.context.object
    bmap = rigify_map.auto_detect(arm, False)
    metrics = rigify_map.read_metrics(arm, bmap)
    if metrics.get("errors"):
        check("元骨架读取成功", False, str(metrics["errors"]))
        return
    check("元骨架读取成功", True)

    eye_rest = (metrics.get("ref") or {}).get("eye_line")
    check("静止参照里有 eye_line（旧版缺这个键 -> 扭转被静默跳过）", eye_rest is not None,
          str(eye_rest))
    if eye_rest is None:
        return
    shoulder_rest = Vector(metrics["ref"]["shoulder_line"]).normalized()
    agreement = Vector(eye_rest).normalized().dot(shoulder_rest)
    check("静止眼线 = 人物右 -> 左（与肩线同向）", agreement > 0.9, "dot=%.3f" % agreement)

    name = bmap.get("head")
    bone = arm.data.bones.get(name or "")
    if bone is None:
        check("映射里有 head 骨骼", False, str(name))
        return
    rest_mat = bone.matrix_local.to_3x3()
    rest_dir = (Vector(bone.tail_local) - Vector(bone.head_local)).normalized()
    head_base = Vector(bone.head_local)
    target = (head_base, head_base + rest_dir * 0.15, "eye_line")
    to_arm = lambda vec: Vector(vec).normalized()          # noqa: E731

    def twist(rest_refs, refs):
        matrix = retarget.solve_role_matrix("head", target, refs, rest_refs,
                                            rest_mat, to_arm)
        delta = (matrix @ rest_mat.inverted()).to_quaternion()
        return math.degrees(delta.angle), delta.axis

    yaw = 40.0
    turned = {"shoulder_line": metrics["ref"]["shoulder_line"],
              "eye_line": Matrix.Rotation(math.radians(yaw), 3, Vector((0.0, 0.0, 1.0)))
              @ Vector(eye_rest)}
    angle, axis = twist(metrics["ref"], turned)
    check("眼线转 %.0f° -> head 姿态跟着转（±5°）" % yaw, abs(angle - yaw) < 5.0,
          "%.1f°" % angle)
    check("转轴 = head 骨的瞄准轴（上轴）", abs(axis.normalized().dot(rest_dir)) > 0.99,
          "dot=%.4f" % abs(axis.normalized().dot(rest_dir)))

    plain = {"shoulder_line": metrics["ref"]["shoulder_line"],
             "eye_line": metrics["ref"]["shoulder_line"]}
    angle, _axis = twist(metrics["ref"], plain)
    check("眼线与肩线一致时不产生扭转（< 1°）", angle < 1.0, "%.3f°" % angle)

    stale = {"hip_line": metrics["ref"]["hip_line"],
             "shoulder_line": metrics["ref"]["shoulder_line"]}
    angle, _axis = twist(stale, turned)
    check("静止参照缺 eye_line 时扭转被跳过（= 旧版行为）", angle < 1.0, "%.3f°" % angle)


def main():
    print("五官 -> 头 / 脖子 自检（合成关键点，两种解算模式 + Rigify 扭转）")
    test_face_frame()
    test_head_base_source()
    test_reconstruct()
    test_rigify_head_twist()
    if FAILURES:
        print("\n失败 %d 项：" % len(FAILURES))
        for name in FAILURES:
            print("  - %s" % name)
        return 1
    print("\n全部通过 ✓")
    return 0


if __name__ == "__main__":
    sys.exit(main())

