"""SAM 3D Body 适配层的纯逻辑自检 —— **普通 Python 就能跑，不需要 Blender / torch**。

覆盖：
  1. MHR70 常量：70 个名字、顺序、左右互换表；
  2. 相机投票：Y 向下 / Y 向上 / 是否加 cam_t，用 2D 重投影误差选；
  3. MHR70 -> 插件 recon：角色点、参照轴、朝向 yaw、脊椎点、手指链；
  4. 变换一致性：网格顶点（``mesh_arrays``）和关节用的是同一套变换；
  5. 和重定向的接口：``retarget.build_targets`` 能直接吃这份 recon（用假 bpy 载入真模块）。

用法：
  python dev_tests/test_sam3d.py
  （没装 numpy 时会跳过 —— 也可以用 Blender 自带 python 跑，它一定有 numpy）
"""
import importlib.util
import math
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
REAL = os.path.join(ROOT, "pose_capture")

try:
    import numpy as np
except ImportError:                                    # pragma: no cover
    np = None

failures = []


def check(ok, label, detail=""):
    print("  %s %s %s" % ("OK " if ok else "!! ", label, detail))
    if not ok:
        failures.append(label)
    return ok


# --------------------------------------------------------------------------- 假 bpy
class _Any:
    """万能假对象（只为了能让 retarget.py 里的 bpy / mathutils 引用过一遍）。"""

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return _Any()

    def __getattr__(self, name):
        return _Any

    def __setattr__(self, name, value):
        pass

    def __getitem__(self, key):
        return _Any()

    def __setitem__(self, key, value):
        pass

    def __iter__(self):
        return iter(())

    def __len__(self):
        return 0

    def __bool__(self):
        return False

    __add__ = __radd__ = __sub__ = __rsub__ = __mul__ = __rmul__ = __truediv__ = __init__


def _stub(name):
    module = types.ModuleType(name)
    module.__getattr__ = lambda attr: _Any          # PEP 562
    sys.modules[name] = module
    return module


def _resample_polyline(points, count):
    """reconstruction._resample_polyline 的等价实现（按弧长重采样成 count+1 个点）。"""
    if len(points) < 2 or count < 1:
        return list(points)
    pts = [np.asarray(point, dtype=np.float64) for point in points]
    seg = [float(np.linalg.norm(pts[i + 1] - pts[i])) for i in range(len(pts) - 1)]
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


def load_module(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(REAL, name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_retarget():
    """用假 bpy / mathutils + 真 numpy 载入真的 retarget.py（不启动 Blender）。"""
    for name in ("bpy", "mathutils"):
        _stub(name)
    sys.modules["mathutils"].Vector = _Any
    sys.modules["mathutils"].Matrix = _Any

    package = types.ModuleType("pctest")
    package.__path__ = []
    sys.modules["pctest"] = package

    stub_recon = _stub("pctest.reconstruction")
    stub_recon._resample_polyline = _resample_polyline
    _stub("pctest.utils")
    _stub("pctest.rigify_limbs")

    for name in ("rigify_limbs", "retarget"):
        spec = importlib.util.spec_from_file_location("pctest." + name,
                                                      os.path.join(REAL, name + ".py"))
        module = importlib.util.module_from_spec(spec)
        sys.modules["pctest." + name] = module
        spec.loader.exec_module(module)
    return sys.modules["pctest.retarget"]


# --------------------------------------------------------------------------- 合成数据
FINGERS = ("thumb", "index", "middle", "ring", "pinky")
STEPS = ("third-joint", "second-joint", "first-joint", "tip")


def synthetic_person(sam3d, angle_deg=0.0, focal=1500.0, size=(2000, 3000)):
    """造一个 T 姿势的人（OpenCV 相机系：X 右 / Y 下 / Z 前），返回 (k3, k2, extra)。

    人站在 z=3 m、面朝镜头（-Z），人物左侧 = +X。``angle_deg`` 是把人绕竖直轴转过去，
    用来验证朝向归一（转完之后 recon 里应该永远面朝 -Y）。
    """
    base = {
        "left-shoulder": (0.20, 0.55, 3.00), "right-shoulder": (-0.20, 0.55, 3.00),
        "left-acromion": (0.22, 0.48, 3.00), "right-acromion": (-0.22, 0.48, 3.00),
        "left-elbow": (0.50, 0.55, 2.88), "right-elbow": (-0.50, 0.55, 2.88),
        "left-wrist": (0.78, 0.55, 2.78), "right-wrist": (-0.78, 0.55, 2.78),
        "left-hip": (0.10, 1.20, 3.00), "right-hip": (-0.10, 1.20, 3.00),
        "left-knee": (0.10, 1.65, 3.00), "right-knee": (-0.10, 1.65, 3.00),
        "left-ankle": (0.10, 2.05, 3.00), "right-ankle": (-0.10, 2.05, 3.00),
        "left-big-toe-tip": (0.10, 2.12, 2.86), "left-small-toe-tip": (0.13, 2.12, 2.86),
        "left-heel": (0.10, 2.12, 3.12),
        "right-big-toe-tip": (-0.10, 2.12, 2.86), "right-small-toe-tip": (-0.13, 2.12, 2.86),
        "right-heel": (-0.10, 2.12, 3.12),
        "neck": (0.0, 0.45, 3.00),
        "left-ear": (0.075, 0.28, 3.00), "right-ear": (-0.075, 0.28, 3.00),
        "left-eye": (0.032, 0.30, 2.93), "right-eye": (-0.032, 0.30, 2.93),
        "nose": (0.0, 0.30, 2.89),
    }
    for side, sign in (("left", 1.0), ("right", -1.0)):
        wrist_x = 0.78 * sign
        base["%s-olecranon" % side] = (0.50 * sign, 0.55, 2.88 + 0.03)
        base["%s-cubital-fossa" % side] = (0.50 * sign, 0.55, 2.88 - 0.03)
        for index, finger in enumerate(FINGERS):
            spread = (index - 2) * 0.014
            for step, stage in enumerate(STEPS):
                length = 0.05 + 0.022 * step
                base["%s-%s-%s" % (side, finger, stage)] = (
                    wrist_x + sign * length,
                    0.55 + spread * (0.4 + 0.2 * step),
                    2.78 - 0.008 * step,
                )

    points = np.zeros((sam3d.JOINT_COUNT, 3), dtype=np.float64)
    points[:, 2] = 3.0
    for name, value in base.items():
        points[sam3d.JOINT[name]] = value

    # 绕竖直轴（相机 Y）转过去：绕 (x=0, z=3) 这根轴
    theta = math.radians(float(angle_deg))
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    dx = points[:, 0].copy()
    dz = points[:, 2] - 3.0
    points[:, 0] = dx * cos_t + dz * sin_t
    points[:, 2] = -dx * sin_t + dz * cos_t + 3.0

    height, width = size
    cx, cy = (width - 1) * 0.5, (height - 1) * 0.5
    keypoints_2d = np.stack([points[:, 0] * focal / points[:, 2] + cx,
                             points[:, 1] * focal / points[:, 2] + cy], axis=1)
    extra = {"size": (height, width), "focal": focal,
             "wrist.L": np.array(base["left-wrist"], dtype=np.float64),
             "hip_mid": 0.5 * (np.asarray(base["left-hip"]) + np.asarray(base["right-hip"]))}
    return points, keypoints_2d, extra


def as_data(k3, k2, extra, cam_t=None):
    return {"keypoints_3d": np.asarray(k3, dtype=np.float32),
            "keypoints_2d": np.asarray(k2, dtype=np.float32),
            "cam_t": np.asarray(cam_t if cam_t is not None else np.zeros(3), dtype=np.float32),
            "focal_length": np.asarray([extra["focal"]], dtype=np.float32),
            "image_size": np.asarray(list(extra["size"]), dtype=np.int32),
            "meta": {"people": 1}}



def _unit(vec):
    vec = np.asarray(vec, dtype=np.float64)
    return vec / max(float(np.linalg.norm(vec)), 1e-12)


def test_constants(sam3d):
    print("\n[1] MHR70 常量")
    check(len(sam3d.MHR_NAMES) == sam3d.JOINT_COUNT == 70,
          "70 个关键点", "%d" % len(sam3d.MHR_NAMES))
    check(len(set(sam3d.MHR_NAMES)) == 70, "名字不重复")
    check(sam3d.JOINT["left-shoulder"] == 5 and sam3d.JOINT["right-shoulder"] == 6,
          "肩 5/6")
    check(sam3d.JOINT["left-wrist"] == 62 and sam3d.JOINT["right-wrist"] == 41,
          "手腕 62/41", "left=%d right=%d" % (sam3d.JOINT["left-wrist"],
                                              sam3d.JOINT["right-wrist"]))
    check(sam3d.JOINT["neck"] == 69, "鼻 0 / 颈 69")
    order = sam3d._swap_permutation()
    check(order[sam3d.JOINT["left-shoulder"]] == sam3d.JOINT["right-shoulder"]
          and order[sam3d.JOINT["right-index-tip"]] == sam3d.JOINT["left-index-tip"],
          "左右互换表")
    check(order[sam3d.JOINT["nose"]] == sam3d.JOINT["nose"], "中线点不换")


def test_camera(sam3d):
    print("\n[2] 相机投票（Y 上/下 × cam_t）")
    k3, k2, extra = synthetic_person(sam3d)
    y_down, use_offset, error = sam3d._choose_camera(
        k3, k2, extra["focal"], extra["size"], np.zeros(3))
    check(y_down is True and use_offset is False, "OpenCV（Y 下）被选中",
          "y_down=%s offset=%s err=%.2f" % (y_down, use_offset, error))
    check(error < 1.0, "重投影误差 < 1 px", "%.3f px" % error)

    flipped = k3 * np.array((1.0, -1.0, 1.0))
    y_down, use_offset, error = sam3d._choose_camera(
        flipped, k2, extra["focal"], extra["size"], np.zeros(3))
    check(y_down is False, "Y 向上的输入也能认出来", "err=%.2f" % error)

    cam_t = np.array((0.0, 0.0, 1.0))
    shifted = k3 - cam_t
    shift_2d = np.stack([k3[:, 0] * extra["focal"] / k3[:, 2] + (extra["size"][1] - 1) * 0.5,
                         k3[:, 1] * extra["focal"] / k3[:, 2] + (extra["size"][0] - 1) * 0.5],
                        axis=1)
    y_down, use_offset, error = sam3d._choose_camera(
        shifted, shift_2d, extra["focal"], extra["size"], cam_t)
    check(use_offset is True and error < 1.0, "自动加 pred_cam_t",
          "offset=%s err=%.2f" % (use_offset, error))

    y_down, use_offset, error = sam3d._choose_camera(k3, None, 0.0, (0, 0), np.zeros(3))
    check(y_down is True and error == float("inf"), "没有 2D/焦距时退回 OpenCV 约定")


def test_recon(sam3d):
    print("\n[3] MHR70 -> recon")
    k3, k2, extra = synthetic_person(sam3d)
    recon = sam3d.build_recon(as_data(k3, k2, extra))
    joints, refs = recon["joints"], recon["refs"]

    check(recon["ok"] and recon["mode"] == "SAM3D", "ok / mode")
    needed = ("pelvis", "chest", "neck_start", "head_base", "head_tip",
              "shoulder_origin.L", "shoulder.L", "elbow.R", "wrist.L",
              "hand_tip.R", "hip.L", "knee.R", "ankle.L", "ball.R")
    missing = [name for name in needed if name not in joints]
    check(not missing, "角色点齐全", "缺 %s" % missing if missing else "%d 个" % len(joints))

    pelvis = joints["pelvis"]
    expect = np.array((extra["hip_mid"][0], extra["hip_mid"][2], -extra["hip_mid"][1]))
    check(np.linalg.norm(pelvis - expect) < 1e-6, "骨盆换算正确（x, z, -y）",
          "%s" % np.round(pelvis, 3))

    shoulder = refs["shoulder_line"]
    check(shoulder[0] > 0.35 and abs(shoulder[1]) < 1e-6 and abs(shoulder[2]) < 1e-6,
          "肩线 = 人物左 - 右（+X）", "%s" % np.round(shoulder, 3))
    check(float(np.dot(refs["eye_line"], shoulder)) > 0.0, "眼线符号统一到肩线")

    up_dir = _unit(joints["head_tip"] - joints["head_base"])
    check(up_dir[2] > 0.95, "head 方向朝上", "%s" % np.round(up_dir, 3))

    arm = _unit(joints["elbow.L"] - joints["shoulder.L"])
    check(arm[0] > 0.85, "左大臂朝 +X（T 姿势）", "%s" % np.round(arm, 3))
    leg = _unit(joints["knee.L"] - joints["hip.L"])
    check(leg[2] < -0.95, "大腿朝下", "%s" % np.round(leg, 3))

    for key in ("armbend.L", "armbend.R"):
        axis = refs.get(key)
        check(axis is not None and abs(float(np.linalg.norm(axis)) - 1.0) < 1e-6,
              "%s 是单位向量" % key, "%s" % (np.round(axis, 3) if axis is not None else None))

    spine = recon["spine_points"]
    check(len(spine) == 5, "脊椎点 5 个")
    check(np.linalg.norm(spine[0] - pelvis) < 1e-9
          and np.linalg.norm(spine[-1] - joints["chest"]) < 1e-9,
          "脊椎从骨盆到胸")

    fingers = recon["fingers"]
    ok = (set(fingers) == {"L", "R"}
          and all(len(fingers[side][finger]) == 4 for side in fingers for finger in FINGERS))
    check(ok, "两只手各 5 指 × 4 点")
    chain = fingers["L"]["index"]
    lengths = [float(np.linalg.norm(chain[i + 1] - chain[i])) for i in range(3)]
    check(all(length > 0.0 for length in lengths), "手指链长度正常",
          "%s" % np.round(lengths, 4))
    tip = joints["hand_tip.L"]
    check(tip[0] > joints["wrist.L"][0], "hand_tip 在手腕外侧")

    check(any("SAM 3D Body" in note for note in recon["notes"]), "notes 写了来源",
          recon["notes"][0][:40])
    check(recon["sam3d"]["reproj_error"] < 1.0, "recon 里带重投影误差")




def test_facing(sam3d):
    print("\n[4] 朝向归一（转过去也要面朝 -Y）")
    up = np.array((0.0, 0.0, 1.0))
    for angle in (0.0, 90.0, -90.0, 180.0):
        k3, k2, extra = synthetic_person(sam3d, angle_deg=angle)
        recon = sam3d.build_recon(as_data(k3, k2, extra))
        front = _unit(np.cross(_unit(recon["refs"]["shoulder_line"]), up))
        check(abs(float(front[1]) + 1.0) < 0.05,
              "%+.0f°：归一后面朝 -Y" % angle, "%s" % np.round(front, 3))
        shoulder = recon["refs"]["shoulder_line"]
        check(abs(float(np.dot(_unit(shoulder), front))) < 0.05
              and abs(float(shoulder[2])) < 1e-6, " 肩线垂直朝向且水平")

        diff = (recon["yaw_deg"] + angle + 180.0) % 360.0 - 180.0
        check(abs(diff) < 5.0, " yaw 与旋转角一致",
              "yaw=%.1f° 期望 %.1f°" % (recon["yaw_deg"], -angle))


def test_mesh(sam3d):
    print("\n[5] 网格顶点与关节同一套变换")
    k3, k2, extra = synthetic_person(sam3d, angle_deg=35.0)
    data = as_data(k3, k2, extra)
    data["vertices"] = np.asarray([extra["hip_mid"], extra["wrist.L"]], dtype=np.float32)
    data["faces"] = np.asarray([[0, 1, 0]], dtype=np.int32)
    recon = sam3d.build_recon(data)
    verts, faces = sam3d.mesh_arrays(data, recon)

    check(verts.shape == (2, 3) and faces.shape == (1, 3), "顶点/面片形状")
    error = float(np.linalg.norm(verts[0] - recon["joints"]["pelvis"]))
    check(error < 1e-4, "髋中点顶点落到骨盆", "%.5f m" % error)
    before = float(np.linalg.norm(np.asarray(data["vertices"][1], np.float64) - extra["hip_mid"]))
    after = float(np.linalg.norm(verts[1] - verts[0]))
    check(abs(before - after) < 1e-4, "刚体变换（距离不变）",
          "%.4f -> %.4f" % (before, after))

    try:
        sam3d.mesh_arrays({"keypoints_3d": data["keypoints_3d"]}, recon)
    except sam3d.Sam3DError:
        check(True, "没有顶点时给出明确错误")
    else:
        check(False, "没有顶点时给出明确错误")


def test_retarget(sam3d, retarget):
    print("\n[6] retarget.build_targets 能吃这份 recon")
    k3, k2, extra = synthetic_person(sam3d)
    recon = sam3d.build_recon(as_data(k3, k2, extra))

    class FakeMap:
        def spine_bones(self):
            return ["spine", "spine.001", "spine.002"]

        def get(self, role):
            return role

    targets = retarget.build_targets(recon, FakeMap(), {})
    wanted = ("hips", "spine_00", "spine_01", "spine_02", "neck", "head",
              "shoulder.L", "upper_arm.R", "forearm.L", "hand.R", "palm.L",
              "thigh.L", "shin.R", "foot.L", "thumb.01.L", "f_pinky.03.R")

    missing = [name for name in wanted if name not in targets]
    check(not missing, "目标齐全", "缺 %s" % missing if missing else "%d 个" % len(targets))

    refs = recon["refs"]
    broken = [role for role, (_a, _b, ref) in targets.items()
              if ref not in refs and not ref.startswith("kneebend.")]
    check(not broken, "参照轴都在 recon 里",
          ("缺 %s" % broken) if broken else "（直腿时 kneebend 允许退化）")

    arm = _unit(np.asarray(targets["upper_arm.L"][1]) - np.asarray(targets["upper_arm.L"][0]))
    check(arm[0] > 0.85, "upper_arm.L 指向 +X", "%s" % np.round(arm, 3))
    check(np.linalg.norm(np.asarray(targets["spine_00"][0]) - recon["joints"]["pelvis"]) < 1e-9,
          "spine_00 从骨盆开始")
    check(np.linalg.norm(np.asarray(targets["spine_02"][1]) - recon["joints"]["chest"]) < 1e-6,
          "spine_02 到胸结束")
    head_dir = _unit(np.asarray(targets["head"][1]) - np.asarray(targets["head"][0]))
    check(head_dir[2] > 0.9, "head 朝上", "%s" % np.round(head_dir, 3))


def main():
    print("SAM 3D Body 适配层自检（纯逻辑，不需要 Blender / torch）")
    if np is None:
        print("  (跳过：当前 python 没有 numpy —— 换成 Blender 自带 python 再跑)")
        return 0

    sam3d = load_module("sam3d")
    test_constants(sam3d)
    test_camera(sam3d)
    test_recon(sam3d)
    test_facing(sam3d)
    test_mesh(sam3d)

    try:
        retarget = load_retarget()
    except Exception as exc:                        # noqa: BLE001
        print("\n[6] retarget 段跳过：%s" % exc)
    else:
        test_retarget(sam3d, retarget)

    print()
    if failures:
        print("!! 失败 %d 项：%s" % (len(failures), ", ".join(failures)))
        return 1
    print("通过 (all OK)")
    return 0



if __name__ == "__main__":
    sys.exit(main())
