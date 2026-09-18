"""Rigify 生成骨架（含 FK/IK 控制）自检。

验证：
  1. 自动骨骼映射能否找到 spine_fk / upper_arm_fk.L / thigh_fk.L / neck / head / torso；
  2. 用 DEF- 变形骨骼合成 2D 关键点，跑完整管线后变形骨架在画面内是否能复原；
  3. IK_FK 滑块是否被自动切到 FK。

用法：
  blender.exe -b --factory-startup --python "dev_tests/test_rigify_rig.py"
"""
import math
import os
import sys

import bpy
from mathutils import Matrix, Vector

HERE = os.path.dirname(os.path.abspath(__file__))
ADDON_PARENT = os.path.dirname(HERE)
for path in (ADDON_PARENT, HERE):
    if path not in sys.path:
        sys.path.insert(0, path)

from test_pipeline import (IMAGE_SIZE, SCALE, ensure_registered,  # noqa: E402
                           _axis_of, _posed_dir, _project, _world, rotate_bone_world)

from pose_capture import reconstruction, retarget, rigify_map  # noqa: E402

DEF_MAP = {
    "head": "DEF-spine.006",
    "upper_arm": "DEF-upper_arm.%s",
    "forearm": "DEF-forearm.%s",
    "hand": "DEF-hand.%s",
    "thigh": "DEF-thigh.%s",
    "shin": "DEF-shin.%s",
    "foot": "DEF-foot.%s",
    "toe": "DEF-toe.%s",
}

FK_POSE = [
    ("spine_fk.001", 6.0),
    ("upper_arm_fk.L", 30.0),
    ("forearm_fk.L", -50.0),
    ("upper_arm_fk.R", -15.0),
    ("forearm_fk.R", 35.0),
    ("thigh_fk.R", 22.0),
    ("shin_fk.R", -45.0),
    ("thigh_fk.L", -6.0),
]


def setup_generated_rig():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.preferences.addon_enable(module="rigify")
    bpy.ops.object.armature_human_metarig_add()
    bpy.ops.object.mode_set(mode="POSE")
    bpy.ops.pose.rigify_generate()
    rig = bpy.context.object
    bpy.ops.object.mode_set(mode="OBJECT")
    for pose_bone in rig.pose.bones:
        pose_bone.rotation_mode = "QUATERNION"
    bpy.context.view_layer.update()
    return rig


def set_fk(rig):
    """Rigify：IK_FK = 1.0 才是 FK 模式。"""
    for side in ("L", "R"):
        for base in ("upper_arm", "thigh"):
            parent = rig.pose.bones.get("%s_parent.%s" % (base, side))
            if parent is not None and "IK_FK" in parent.keys():
                parent["IK_FK"] = 1.0
    bpy.context.view_layer.update()


def synth_from_def(rig):
    keypoints = [[0.0, 0.0] for _ in range(133)]
    scores = [0.0] * 133
    missing = []

    def put(index, vec):
        keypoints[index] = list(_project(vec))
        scores[index] = 1.0

    def bone(template, side=None):
        name = template % side if side else template
        pose_bone = rig.pose.bones.get(name)
        if pose_bone is None:
            missing.append(name)
        return pose_bone

    head = bone(DEF_MAP["head"])
    center = (_world(rig, head) + _world(rig, head, True)) * 0.5
    forward = _posed_dir(rig, head, _axis_of(head, (0.0, -1.0, 0.0)))
    lateral = _posed_dir(rig, head, _axis_of(head, (1.0, 0.0, 0.0)))
    up = _posed_dir(rig, head, _axis_of(head, (0.0, 0.0, 1.0)))
    put(0, center + forward * 0.095)
    put(1, center + forward * 0.07 + lateral * 0.035 + up * 0.02)
    put(2, center + forward * 0.07 - lateral * 0.035 + up * 0.02)
    put(3, center + lateral * 0.085)
    put(4, center - lateral * 0.085)
    put(23 + 36, center + forward * 0.07 - lateral * 0.04 + up * 0.02)
    put(23 + 45, center + forward * 0.07 + lateral * 0.04 + up * 0.02)
    for index in range(17, 27):
        put(23 + index, center + forward * 0.075 + up * 0.045)
    put(23 + 8, center + forward * 0.055 - up * 0.085)

    for side in ("L", "R"):
        upper = bone(DEF_MAP["upper_arm"], side)
        fore = bone(DEF_MAP["forearm"], side)
        hand = bone(DEF_MAP["hand"], side)
        thigh = bone(DEF_MAP["thigh"], side)
        shin = bone(DEF_MAP["shin"], side)
        foot = bone(DEF_MAP["foot"], side)
        toe = bone(DEF_MAP["toe"], side)
        if None in (upper, fore, hand, thigh, shin, foot, toe):
            continue

        put(5 if side == "L" else 6, _world(rig, upper))
        put(7 if side == "L" else 8, _world(rig, fore))
        put(9 if side == "L" else 10, _world(rig, hand))
        put(11 if side == "L" else 12, _world(rig, thigh))
        put(13 if side == "L" else 14, _world(rig, shin))
        put(15 if side == "L" else 16, _world(rig, foot))
        put(17 if side == "L" else 20, _world(rig, toe))
        put(18 if side == "L" else 21, _world(rig, toe, True))

        wrist = _world(rig, hand)
        offset = _world(rig, hand, True) - wrist
        direction = offset.normalized()
        base = 91 if side == "L" else 112
        put(base + 0, wrist)
        for index in range(1, 21):
            put(base + index, wrist + direction * (offset.length * (0.3 + 0.035 * index)))

    if missing:
        print("!! 缺少变形骨骼：", sorted(set(missing)))
        return None, None
    return keypoints, scores


def landmark_error(keypoints, other):
    indices = (5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 20)
    total = 0.0
    count = 0
    for index in indices:
        if keypoints[index][0] == 0.0 and other[index][0] == 0.0:
            continue
        total += math.hypot(keypoints[index][0] - other[index][0],
                            keypoints[index][1] - other[index][1])
        count += 1
    return (total / count) if count else float("inf")


def main():
    ensure_registered()
    rig = setup_generated_rig()

    bmap = rigify_map.auto_detect(rig, False)
    print("生成的骨架：", rig.name, "骨骼数", len(rig.data.bones))
    print("自动匹配：", bmap.roles)
    print("未匹配：", bmap.missing)

    expect = {"upper_arm.L": "upper_arm_fk.L", "thigh.R": "thigh_fk.R",
              "hips": "torso", "root": "root", "neck": "neck", "head": "head",
              "spine_00": "spine_fk", "forearm.R": "forearm_fk.R"}
    failed = []
    for role, name in expect.items():
        got = bmap.get(role)
        if got != name:
            failed.append((role, name, got))
        print("  %s%-14s -> %s（期望 %s）" % ("OK " if got == name else "!! ",
                                              role, got, name))

    metrics = rigify_map.read_metrics(rig, bmap)
    if metrics.get("errors"):
        print("!! read_metrics 失败：", metrics["errors"])
        return 1
    print("度量：pelvis=%s chest=%s pelvis_offset_z=%.4f chest_offset_z=%.4f "
          "pelvis_to_shoulders=%.4f" % (tuple(round(v, 3) for v in metrics["pelvis"]),
                                        tuple(round(v, 3) for v in metrics["chest"]),
                                        metrics["pelvis_offset_z"], metrics["chest_offset_z"],
                                        metrics["pelvis_to_shoulders"]))
    print("     大腿=%s 小腿=%s 大臂=%s 小臂=%s 脚=%s 头=%s" % (
        metrics["thigh_len"], metrics["shin_len"], metrics["upper_arm_len"],
        metrics["forearm_len"], metrics["foot_len"], metrics["head_len"]))
    print("     肩关节=%s 胯关节=%s" % (
        tuple(round(v, 3) for v in metrics["shoulder_joints"]["L"]),
        tuple(round(v, 3) for v in metrics["hip_joints"]["L"])))

    set_fk(rig)
    for name, degrees in FK_POSE:
        rotate_bone_world(rig, name, (0.0, 1.0, 0.0), degrees)

    keypoints, scores = synth_from_def(rig)
    if keypoints is None:
        return 1

    recon = reconstruction.reconstruct(keypoints, scores, metrics, IMAGE_SIZE,
                                       reconstruction.default_options())
    if not recon.get("ok"):
        print("!! 重建失败：", recon.get("message"))
        return 1
    joints = recon["joints"]
    targets = retarget.build_targets(recon, bmap, metrics)
    print("重建关节（米）：pelvis %s chest %s hip.L %s knee.L %s ankle.L %s" % (
        tuple(round(v, 3) for v in joints["pelvis"]),
        tuple(round(v, 3) for v in joints["chest"]),
        tuple(round(v, 3) for v in joints["hip.L"]),
        tuple(round(v, 3) for v in joints["knee.L"]),
        tuple(round(v, 3) for v in joints["ankle.L"])))
    print("重建比例 %.1f 像素/米" % recon["scale"])
    for role in ("hips", "spine_00", "spine_03"):
        if role in targets:
            p_from, p_to, _ref = targets[role]
            direction = (Vector(p_to) - Vector(p_from)).normalized()
            print("  目标 %-9s %s  静止 %s" % (
                role, tuple(round(v, 3) for v in direction),
                tuple(round(v, 3) for v in metrics["rest_dir"].get(role, Vector()))))
    print("  静止方向与目标夹角过大的角色（>40 度，说明该骨骼的静止方向不是身体轴）：")
    for role, (p_from, p_to, _ref) in targets.items():
        rest_dir = metrics["rest_dir"].get(role)
        if rest_dir is None or role == "hips":
            continue
        direction = (Vector(p_to) - Vector(p_from)).normalized()
        angle = math.degrees(rest_dir.angle(direction))
        if angle > 40.0:
            print("    %-14s %5.1f 度  静止 %s" % (role, angle,
                                                  tuple(round(v, 3) for v in rest_dir)))
    torso_bone = rig.data.bones.get("torso")
    if torso_bone is not None:
        print("  torso head_local %s tail_local %s 静止方向 %s" % (
            tuple(round(v, 3) for v in torso_bone.head_local),
            tuple(round(v, 3) for v in torso_bone.tail_local),
            tuple(round(v, 3) for v in metrics["rest_dir"].get("hips", Vector()))))
        print("  torso 骨盆枢轴 %s" % (tuple(round(v, 3) for v in metrics["pelvis"]),))
    before_torso = tuple(round(v, 4) for v in rig.pose.bones["torso"].head) \
        if rig.pose.bones.get("torso") else None

    report = retarget.apply_pose(rig, bmap, metrics, recon,
                                 {"twist": True, "twist_blend": 1.0,
                                  "fk_mode": True, "root_motion": False})
    print("写入：", report["message"], "IK_FK 切换：", report.get("fk_switched"))
    if before_torso is not None:
        print("  torso 头位置 前 %s 后 %s" % (
            before_torso, tuple(round(v, 4) for v in rig.pose.bones["torso"].head)))
    for role, reason in report["skipped"][:6]:
        print("   跳过", role, reason)

    after, _scores = synth_from_def(rig)
    error = landmark_error(keypoints, after)
    print("变形骨架画面内平均偏差：%.2f 像素（%.0f 像素/米）" % (error, SCALE))
    print("逐点偏差：")
    for index in (5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 20):
        if keypoints[index][0] == 0.0:
            continue
        print("  kp%-3d %6.1f 像素  (前 %s / 后 %s)"
              % (index,
                 math.hypot(keypoints[index][0] - after[index][0],
                            keypoints[index][1] - after[index][1]),
                 tuple(round(v) for v in keypoints[index]),
                 tuple(round(v) for v in after[index])))
    for name in ("upper_arm_fk.L", "spine_fk", "thigh_fk.R"):
        pose_bone = rig.pose.bones.get(name)
        if pose_bone is not None:
            print("  %s 旋转四元数 %s" % (name, tuple(round(v, 3)
                                                       for v in pose_bone.rotation_quaternion)))
    print("重建比例：%.1f 像素/米" % recon["scale"])

    ok = not failed and error < 25.0
    print("结果：", "通过 ✓" if ok else "未通过 ✗")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

