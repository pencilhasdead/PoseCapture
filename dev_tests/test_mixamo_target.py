"""Mixamo 目标骨架自检：把抓到的姿态直接写到 ``mixamorig:*`` 命名的骨架上。

验证：
  1. ``rigify_map.auto_detect(..., rig_type="MIXAMO")`` 把角色映射到 Mixamo 骨名；
  2. 前缀三种写法（``mixamorig:`` / ``mixamorig1:`` / 无）都能认；
  3. ``read_metrics`` 能按 Mixamo 骨架量出骨长 / 朝向；
  4. ``retarget.apply_pose`` 写进去以后，骨骼方向与 recon 一致。

用法：
  blender.exe -b --factory-startup --python "dev_tests/test_mixamo_target.py"
"""
import math
import os
import sys

import bpy
from mathutils import Vector

HERE = os.path.dirname(os.path.abspath(__file__))
ADDON_PARENT = os.path.dirname(HERE)
for path in (ADDON_PARENT, HERE):
    if path not in sys.path:
        sys.path.insert(0, path)

from test_mmd_mixamo_rig import SKELETON                        # noqa: E402
from test_pipeline import ensure_registered                     # noqa: E402
from test_sam3d import as_data, synthetic_person                # noqa: E402


from pose_capture import mmd_mixamo, props, retarget, rigify_map, sam3d  # noqa: E402

ensure_registered()

failures = []


def check(ok, label, detail=""):
    print("  %s %s %s" % ("OK " if ok else "!! ", label, detail))
    if not ok:
        failures.append(label)
    return ok


def mixamo_skeleton(prefix="mixamorig:"):
    """把 MMD 骨架表换算成 Mixamo 骨名（用插件自己的对照表）。"""
    renamed = {}
    rows = []
    for name, parent, head, tail in SKELETON:
        target = mmd_mixamo.mixamo_name(name, prefix) or name
        parent_target = renamed.get(parent, parent) if parent else None
        renamed[name] = target
        rows.append((target, parent_target, head, tail))
    return rows


def build_armature(prefix="mixamorig:", name="MixamoRig"):
    armature = bpy.data.armatures.new(name)
    arm = bpy.data.objects.new(name, armature)
    bpy.context.scene.collection.objects.link(arm)
    bpy.context.view_layer.objects.active = arm
    arm.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    made = {}
    for bone_name, parent, head, tail in mixamo_skeleton(prefix):
        bone = armature.edit_bones.new(bone_name)
        bone.head = head
        bone.tail = tail
        if parent and parent in made:
            bone.parent = made[parent]
        made[bone_name] = bone
    bpy.ops.object.mode_set(mode="OBJECT")
    return arm


def bone_dir(rig, name):
    pose_bone = rig.pose.bones.get(name)
    if pose_bone is None:
        return None
    direction = pose_bone.tail - pose_bone.head
    if direction.length < 1e-9:
        return None
    return (rig.matrix_world.to_3x3() @ direction).normalized()


def test_mapping():
    print("\n[1] 角色 -> Mixamo 骨名")
    bpy.ops.wm.read_factory_settings(use_empty=True)
    arm = build_armature("mixamorig:")
    bmap = rigify_map.auto_detect(arm, True, {"Arm": "FK", "Leg": "FK"}, "MIXAMO")

    expect = {
        "hips": "mixamorig:Hips", "root": "mixamorig:Hips",
        "spine_00": "mixamorig:Spine", "spine_01": "mixamorig:Spine1",
        "neck": "mixamorig:Neck", "head": "mixamorig:Head",
        "shoulder.L": "mixamorig:LeftShoulder", "upper_arm.L": "mixamorig:LeftArm",
        "forearm.L": "mixamorig:LeftForeArm", "hand.R": "mixamorig:RightHand",
        "thigh.L": "mixamorig:LeftUpLeg", "shin.L": "mixamorig:LeftLeg",
        "foot.R": "mixamorig:RightFoot", "toe.L": "mixamorig:LeftToe_End",
        "thumb.01.L": "mixamorig:LeftHandThumb1", "f_pinky.03.R": "mixamorig:RightHandPinky3",
    }
    bad = {role: (bmap.get(role), target) for role, target in expect.items()
           if bmap.get(role) != target}
    check(not bad, "16 个角色映射正确", bad or "%d 项" % len(expect))
    check(rigify_map.detect_mixamo_prefix(
        [bone.name for bone in arm.data.bones]) == "mixamorig:", "认出 mixamorig: 前缀")


def test_prefixes():
    print("\n[2] 前缀 / 无前缀")
    for prefix in ("mixamorig1:", ""):
        bpy.ops.wm.read_factory_settings(use_empty=True)
        arm = build_armature(prefix, "Rig_%s" % (prefix.strip(":") or "none"))
        bmap = rigify_map.auto_detect(arm, False, None, "MIXAMO")
        check(bmap.get("hips") == prefix + "Hips", "前缀 %r 的 Hips" % prefix,
              bmap.get("hips"))
        check(bmap.get("thigh.R") == prefix + "RightUpLeg", "前缀 %r 的 UpLeg" % prefix)


def test_apply():
    print("\n[3] 量尺寸 + 写姿态")
    bpy.ops.wm.read_factory_settings(use_empty=True)
    settings = bpy.context.scene.pose_capture
    arm = build_armature("mixamorig:")
    settings.armature = arm
    settings.rig_type = "MIXAMO"
    settings.fingers = True
    settings.limb_mode = "FK"          # Mixamo 骨架没有 Rigify 的 IK 控制器
    settings.twist = True

    bmap = rigify_map.auto_detect(arm, True, {"Arm": "FK", "Leg": "FK"}, "MIXAMO")
    metrics = rigify_map.read_metrics(arm, bmap)
    check(not metrics.get("errors"), "read_metrics",
          "；".join(metrics.get("errors") or []) or "ok")
    check(metrics["spine_names"][0].endswith("Spine"), "脊椎链是 Mixamo 的 Spine…",
          str(metrics["spine_names"]))


    k3, k2, extra = synthetic_person(sam3d)
    recon = sam3d.build_recon(as_data(k3, k2, extra))
    report = retarget.apply_pose(arm, bmap, metrics, recon, props.options_from_settings(settings))
    check(report.get("ok"), "apply_pose", report.get("message", ""))
    applied = set(report.get("applied") or [])
    check({"upper_arm.L", "thigh.R", "head", "hips"} <= applied, "关键角色写入",
          "%d 个" % len(applied))

    pairs = (("upper_arm.L", "shoulder.L", "elbow.L", "mixamorig:LeftArm"),
             ("forearm.L", "elbow.L", "wrist.L", "mixamorig:LeftForeArm"),
             ("thigh.R", "hip.R", "knee.R", "mixamorig:RightUpLeg"),
             ("shin.R", "knee.R", "ankle.R", "mixamorig:RightLeg"),
             ("head", "head_base", "head_tip", "mixamorig:Head"))
    worst = 0.0
    for role, start, end, bone_name in pairs:
        target = Vector(recon["joints"][end]) - Vector(recon["joints"][start])
        target.normalize()
        posed = bone_dir(arm, bone_name)
        if posed is None:
            check(False, "找不到 %s" % bone_name)
            continue
        angle = math.degrees(math.acos(max(-1.0, min(1.0, posed.dot(target)))))
        worst = max(worst, angle)
        check(angle < 2.0, "%s -> %s" % (role, bone_name), "差 %.2f°" % angle)
    print("     最大方向误差 %.2f°" % worst)


def main():
    print("Mixamo 目标骨架自检")
    test_mapping()
    test_prefixes()
    test_apply()
    print()
    if failures:
        print("!! 失败 %d 项：%s" % (len(failures), ", ".join(failures)))
        return 1
    print("通过 (all OK)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

