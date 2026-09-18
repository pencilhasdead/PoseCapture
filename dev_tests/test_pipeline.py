"""合成姿态自检（不需要模型 / 不需要图片）。

思路：
  1. 新建 Rigify 人形元骨架，按已知的旋转摆出一个姿态；
  2. 把这个姿态的关节投影成 COCO-WholeBody 133 点（模拟 DWPose 的输出）；
  3. 跑插件管线：自动匹配骨骼 -> 读取静止尺寸 -> 2D->3D 重建 -> 重定向写回；
  4. 比较写回后的骨骼方向与最初的已知姿态，输出误差表。

用法：
  blender.exe -b --factory-startup --python "dev_tests/test_pipeline.py"
"""
import math
import os
import sys

import bpy
from mathutils import Matrix, Vector

ADDON_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ADDON_PARENT not in sys.path:
    sys.path.insert(0, ADDON_PARENT)

import pose_capture  # noqa: E402
from pose_capture import operators, props, reconstruction, retarget, rigify_map, ui  # noqa: E402


def ensure_registered():
    """在后台模式下手动注册（跳过 AddonPreferences，它需要真正“启用插件”的环境）。"""
    if hasattr(bpy.types.Scene, "pose_capture"):
        return
    for cls in (props.PC_BoneMapping, props.PC_Settings) + operators.CLASSES + ui.CLASSES:
        try:
            bpy.utils.register_class(cls)
        except Exception as exc:  # noqa: BLE001
            print("注册失败：", cls, exc)
    bpy.types.Scene.pose_capture = bpy.props.PointerProperty(type=props.PC_Settings)


ensure_registered()

SCALE = 700.0  # 像素 / 米
IMAGE_SIZE = (1200, 1600)
CX = IMAGE_SIZE[0] * 0.5
CY = 1500.0  # 让人物正好落在画面内

# 测试姿态：全部绕世界 Y 轴（画面内旋转），便于精确校验
TEST_POSE = [
    ("upper_arm.L", 35.0),
    ("forearm.L", -70.0),
    ("upper_arm.R", -20.0),
    ("forearm.R", 40.0),
    ("thigh.R", 25.0),
    ("shin.R", -55.0),
    ("thigh.L", -8.0),
    ("shin.L", 12.0),
    ("spine.001", 8.0),
    ("spine.006", -12.0),
]

CHECK_ROLES = (
    "shoulder.L", "upper_arm.L", "forearm.L", "hand.L",
    "shoulder.R", "upper_arm.R", "forearm.R", "hand.R",
    "thigh.L", "shin.L", "foot.L", "thigh.R", "shin.R", "foot.R",
)

# 脊椎段只做参考：单张图无法判断每节的弯曲分布，重建时按“直线脊椎”近似
EXTRA_ROLES = ("spine_00", "spine_01", "spine_02", "spine_03")


# --------------------------------------------------------------------------- 场景
def setup_rig():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.preferences.addon_enable(module="rigify")
    bpy.ops.object.armature_human_metarig_add()
    obj = bpy.context.object
    for pose_bone in obj.pose.bones:
        pose_bone.rotation_mode = "QUATERNION"
    bpy.context.view_layer.update()
    return obj


def rotate_bone_world(obj, name, axis, degrees):
    """绕世界轴、以骨骼头部为支点旋转（写 pbone.matrix，与插件内部一致）。"""
    pose_bone = obj.pose.bones[name]
    head = pose_bone.head.copy()
    rotation = Matrix.Rotation(math.radians(degrees), 4, Vector(axis).normalized())
    matrix = rotation @ pose_bone.matrix
    matrix.translation = head
    pose_bone.matrix = matrix
    bpy.context.view_layer.update()


def apply_test_pose(obj):
    for name, degrees in TEST_POSE:
        rotate_bone_world(obj, name, (0.0, 1.0, 0.0), degrees)


# --------------------------------------------------------------------------- 合成关键点
def _axis_of(pose_bone, world_dir):
    rest3 = pose_bone.bone.matrix_local.to_3x3()
    best, best_dot = 0, 0.0
    for index in range(3):
        dot = rest3.col[index].normalized().dot(Vector(world_dir))
        if abs(dot) > abs(best_dot):
            best, best_dot = index, dot
    local = Vector((0.0, 0.0, 0.0))
    local[best] = 1.0 if best_dot > 0.0 else -1.0
    return local


def _posed_dir(obj, pose_bone, local_axis):
    matrix = obj.matrix_world.to_3x3() @ pose_bone.matrix.to_3x3()
    return (matrix @ local_axis).normalized()


def _world(obj, pose_bone, use_tail=False):
    return obj.matrix_world @ (pose_bone.tail if use_tail else pose_bone.head)


def _project(vec):
    return (CX + SCALE * vec.x, CY - SCALE * vec.z)


def synthesize_keypoints(obj):
    keypoints = [[0.0, 0.0] for _ in range(133)]
    scores = [0.0] * 133

    def put(index, vec):
        keypoints[index] = list(_project(vec))
        scores[index] = 1.0

    pose = obj.pose.bones

    # 头 / 脸
    head = pose["spine.006"]
    center = (_world(obj, head) + _world(obj, head, True)) * 0.5
    forward = _posed_dir(obj, head, _axis_of(head, (0.0, -1.0, 0.0)))
    lateral = _posed_dir(obj, head, _axis_of(head, (1.0, 0.0, 0.0)))
    up = _posed_dir(obj, head, _axis_of(head, (0.0, 0.0, 1.0)))

    put(0, center + forward * 0.095)
    put(1, center + forward * 0.07 + lateral * 0.035 + up * 0.02)   # 人物左眼（画面右）
    put(2, center + forward * 0.07 - lateral * 0.035 + up * 0.02)
    put(3, center + lateral * 0.085)
    put(4, center - lateral * 0.085)
    put(23 + 36, center + forward * 0.07 - lateral * 0.04 + up * 0.02)
    put(23 + 45, center + forward * 0.07 + lateral * 0.04 + up * 0.02)
    for index in range(17, 27):
        put(23 + index, center + forward * 0.075 + up * 0.045)
    put(23 + 8, center + forward * 0.055 - up * 0.085)

    # 身体 / 四肢
    for side in ("L", "R"):
        put(5 if side == "L" else 6, _world(obj, pose["upper_arm.%s" % side]))
        put(7 if side == "L" else 8, _world(obj, pose["forearm.%s" % side]))
        put(9 if side == "L" else 10, _world(obj, pose["hand.%s" % side]))
        put(11 if side == "L" else 12, _world(obj, pose["thigh.%s" % side]))
        put(13 if side == "L" else 14, _world(obj, pose["shin.%s" % side]))
        put(15 if side == "L" else 16, _world(obj, pose["foot.%s" % side]))
        put(17 if side == "L" else 20, _world(obj, pose["toe.%s" % side]))
        put(18 if side == "L" else 21, _world(obj, pose["toe.%s" % side], True))

        hand = pose["hand.%s" % side]
        wrist = _world(obj, hand)
        tip = _world(obj, hand, True)
        offset_vec = tip - wrist
        length = offset_vec.length
        direction = offset_vec.normalized()
        palm_lat = direction.cross(Vector((0.0, 0.0, 1.0)))
        if palm_lat.length < 1e-6:
            palm_lat = Vector((1.0, 0.0, 0.0))
        palm_lat = palm_lat.normalized()

        base = 91 if side == "L" else 112
        put(base + 0, wrist)
        knuckle_t = {1: 0.2, 2: 0.4, 3: 0.6, 4: 0.8}
        lateral_off = {1: -0.35, 2: -0.3, 3: -0.25, 4: -0.2,
                       5: -0.12, 6: -0.1, 7: -0.08, 8: -0.06,
                       9: 0.0, 10: 0.0, 11: 0.0, 12: 0.0,
                       13: 0.1, 14: 0.1, 15: 0.1, 16: 0.1,
                       17: 0.2, 18: 0.22, 19: 0.24, 20: 0.26}
        for index in range(1, 21):
            if index in knuckle_t:
                t = knuckle_t[index]
            else:
                group_start = index - ((index - 1) % 4)
                t = 0.55 + 0.12 * (index - group_start)
            point = wrist + direction * (length * t) + palm_lat * (0.02 * lateral_off[index])
            put(base + index, point)

    return keypoints, scores


def bone_directions(obj, names):
    result = {}
    for name in names:
        pose_bone = obj.pose.bones.get(name)
        if pose_bone is None:
            continue
        direction = (obj.matrix_world.to_3x3() @ (pose_bone.tail - pose_bone.head))
        if direction.length > 1e-9:
            result[name] = direction.normalized()
    return result


def main():
    obj = setup_rig()
    apply_test_pose(obj)

    names = []
    bmap_probe = rigify_map.auto_detect(obj, False)
    for role in CHECK_ROLES:
        name = bmap_probe.get(role)
        if name:
            names.append(name)
    for role in EXTRA_ROLES:
        name = bmap_probe.get(role)
        if name:
            names.append(name)
    print("自动匹配结果：", bmap_probe.roles)
    print("未匹配：", bmap_probe.missing)

    original = bone_directions(obj, names)

    keypoints, scores = synthesize_keypoints(obj)
    metrics = rigify_map.read_metrics(obj, bmap_probe)
    if metrics.get("errors"):
        print("!! 骨架读取失败：", metrics["errors"])
        return 1

    options = reconstruction.default_options()
    recon = reconstruction.reconstruct(keypoints, scores, metrics, IMAGE_SIZE, options)
    if not recon.get("ok"):
        print("!! 重建失败：", recon.get("message"))
        return 1

    print("脊椎链：", metrics["spine_names"])
    print("比例拟合：%.2f 像素/米（真值 %.2f，误差 %.2f%%）"
          % (recon["scale"], SCALE, abs(recon["scale"] - SCALE) / SCALE * 100.0))
    print("yaw = %.2f 度" % recon["yaw_deg"])
    if recon["synth"]:
        print("兜底补点：", recon["synth"])

    report = retarget.apply_pose(obj, bmap_probe, metrics, recon,
                                 {"twist": True, "twist_blend": 1.0,
                                  "fk_mode": False, "root_motion": False})
    print("写入报告：", report["message"], "跳过", len(report["skipped"]))
    for role, reason in report["skipped"][:8]:
        print("   跳过", role, reason)

    # 目标方向 vs 原姿态方向（区分“重建误差”和“写入误差”）
    targets = retarget.build_targets(recon, bmap_probe, metrics)
    print("\n目标方向对比（重建误差）")
    for role in ("shoulder.L", "upper_arm.L", "hand.L", "neck", "head",
                 "spine_00", "spine_03", "thigh.R"):
        if role not in targets:
            continue
        name = bmap_probe.get(role)
        if not name or name not in original:
            continue
        p_from, p_to, _ref = targets[role]
        direction = (Vector(p_to) - Vector(p_from)).normalized()
        angle = math.degrees(direction.angle(original[name]))
        print("  %-12s 目标 %s  原姿态 %s  差 %.1f 度"
              % (role, tuple(round(v, 3) for v in direction),
                 tuple(round(v, 3) for v in original[name]), angle))
    print("重建关节 Y 深度：pelvis %.3f chest %.3f shoulder.L %.3f hip.L %.3f"
          % (recon["joints"]["pelvis"][1], recon["joints"]["chest"][1],
             recon["joints"]["shoulder.L"][1], recon["joints"]["hip.L"][1]))

    new = bone_directions(obj, names)
    print("\n%-16s %10s" % ("骨骼", "方向误差(度)"))
    errors = []
    for role in CHECK_ROLES + EXTRA_ROLES:
        name = bmap_probe.get(role)
        if not name or name not in original or name not in new:
            continue
        angle = math.degrees(original[name].angle(new[name]))
        if role in CHECK_ROLES:
            errors.append((role, name, angle))
        print("%-16s %10.2f   (%s)%s" % (role, angle, name,
                                         "" if role in CHECK_ROLES else "  [参考]"))

    if not errors:
        print("!! 没有任何骨骼被检查")
        return 1
    worst = max(errors, key=lambda item: item[2])
    mean = sum(item[2] for item in errors) / len(errors)
    print("\n平均误差 %.2f 度，最大误差 %.2f 度（%s）" % (mean, worst[2], worst[1]))
    ok = worst[2] < 10.0 and mean < 5.0
    print("结果：", "通过 ✓" if ok else "未通过 ✗")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())


