"""SAM 3D Body 路径的 Blender 自检：MHR70 -> recon -> Rigify 骨骼 + 网格导入。

不跑 torch（没有外部环境也能测）：
  * 用 `test_sam3d.synthetic_person` 造一帧「MHR70 + 相机 + 2D」的合成结果；
  * 把左手 / 右腿掰弯，验证 `sam3d.build_recon` 给出的角色点真的写进了 Rigify 骨骼
    （FK 骨骼的朝向应该和 recon 里的方向一致）；
  * 再调 `pose_capture.import_sam3d_mesh`，确认网格能导入、骨盆对齐、脚落在骨架地面附近。

用法：
  blender.exe -b --factory-startup --python "dev_tests/test_sam3d_rig.py"
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

from test_pipeline import ensure_registered                       # noqa: E402
from test_sam3d import synthetic_person, as_data                   # noqa: E402

from pose_capture import operators, props, retarget, rigify_map, sam3d  # noqa: E402

ensure_registered()

failures = []


def check(ok, label, detail=""):
    print("  %s %s %s" % ("OK " if ok else "!! ", label, detail))
    if not ok:
        failures.append(label)
    return ok


def setup_rig():
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


def bend_person(points):
    """把左手放下来、右腿屈膝：造一个和静止姿势明显不同的姿态（原地改 MHR70 点）。"""
    points = points.copy()

    def put(name, value):
        points[sam3d.JOINT[name]] = value

    put("left-elbow", (0.38, 0.82, 2.94))
    put("left-wrist", (0.44, 1.02, 2.86))
    put("left-olecranon", (0.38, 0.84, 2.99))
    put("left-cubital-fossa", (0.38, 0.80, 2.90))
    for finger in ("thumb", "index", "middle", "ring", "pinky"):
        base = "left-%s-" % finger
        for step, stage in enumerate(("third-joint", "second-joint", "first-joint", "tip")):
            put(base + stage, (0.455 + 0.02 * step, 1.04 + 0.02 * step, 2.84 - 0.006 * step))

    put("right-knee", (-0.10, 1.64, 2.84))
    put("right-ankle", (-0.10, 1.99, 2.58))
    put("right-big-toe-tip", (-0.10, 1.96, 2.44))
    put("right-small-toe-tip", (-0.13, 1.97, 2.44))
    put("right-heel", (-0.10, 2.05, 2.62))
    return points


def bone_dir(rig, name):
    pose_bone = rig.pose.bones.get(name)
    if pose_bone is None:
        return None
    direction = pose_bone.tail - pose_bone.head
    if direction.length < 1e-9:
        return None
    return (rig.matrix_world.to_3x3() @ direction).normalized()


def main():
    print("SAM 3D Body -> Rigify 自检")
    rig = setup_rig()
    settings = bpy.context.scene.pose_capture
    settings.armature = rig
    settings.solver = "SAM3D"
    settings.limb_mode = "FK"
    settings.fingers = True
    settings.twist = True
    settings.root_motion = False

    print("\n[0] 3D 来源解析（这个环境没配外部 python）")
    settings.solver = "AUTO"
    solver, note = props.resolve_solver(settings)
    check(solver in ("DEPTH", "BONE") and bool(note),
          "AUTO 没配 SAM 3D Body 时退回老路子", "%s / %s" % (solver, note[:40]))
    settings.solver = "SAM3D"
    solver, note = props.resolve_solver(settings)
    check(solver == "SAM3D" and "不可用" in note, "强制 SAM3D 时给出原因", note[:60])

    print("\n[1] 合成 SAM 3D Body 结果")
    k3, k2, extra = synthetic_person(sam3d)
    k3 = bend_person(k3)
    data = as_data(k3, k2, extra)
    recon = sam3d.build_recon(data, props.options_from_settings(settings))
    check(recon.get("ok"), "build_recon 成功",
          "；".join(recon.get("notes") or [])[:70])

    print("\n[2] 映射 + 写姿态")
    bmap = rigify_map.auto_detect(rig, True, {"Arm": "FK", "Leg": "FK"})
    check(bool(bmap.roles), "自动匹配骨骼", "%d 个角色" % len(bmap.roles))
    metrics = rigify_map.read_metrics(rig, bmap)
    check(not metrics.get("errors"), "骨架测量",
          "；".join(metrics.get("errors") or []) or "ok")

    options = props.options_from_settings(settings)
    report = retarget.apply_pose(rig, bmap, metrics, recon, options)
    check(report.get("ok"), "apply_pose 成功", report.get("message", ""))
    applied = set(report.get("applied") or [])
    check({"upper_arm.L", "forearm.L", "thigh.R", "shin.R", "head"} <= applied,
          "关键角色都写入了", "写入 %d 个" % len(applied))

    print("\n[3] 骨架朝向和 recon 一致")
    pairs = (("upper_arm.L", "shoulder.L", "elbow.L", "upper_arm_fk.L"),
             ("forearm.L", "elbow.L", "wrist.L", "forearm_fk.L"),
             ("thigh.R", "hip.R", "knee.R", "thigh_fk.R"),
             ("shin.R", "knee.R", "ankle.R", "shin_fk.R"),
             ("head", "head_base", "head_tip", "head"))
    for role, start, end, bone_name in pairs:
        target = Vector(recon["joints"][end]) - Vector(recon["joints"][start])
        target.normalize()
        posed = bone_dir(rig, bone_name)
        if posed is None:
            check(False, "%s：找不到骨骼 %s" % (role, bone_name))
            continue
        dot = max(-1.0, min(1.0, posed.dot(target)))
        angle = math.degrees(math.acos(dot))
        check(angle < 5.0, "%s 方向一致（%s）" % (role, bone_name), "差 %.1f°" % angle)

    print("\n[4] 确实离开了静止姿势")
    rest = rig.data.bones["upper_arm_fk.L"]
    rest_dir = (rig.matrix_world.to_3x3() @ (rest.tail_local - rest.head_local)).normalized()
    posed = bone_dir(rig, "upper_arm_fk.L")
    if posed is None:
        check(False, "找不到 upper_arm_fk.L")
    else:
        angle = math.degrees(math.acos(max(-1.0, min(1.0, posed.dot(rest_dir)))))
        check(angle > 15.0, "左大臂相对静止转了 %.1f°" % angle)

    print("\n[5] 导入网格")
    data["vertices"] = k3.astype("float32")
    data["faces"] = [[index, index + 1, index + 2] for index in range(len(k3) - 2)]
    operators._RESULT["sam3d"] = dict(data)
    operators._RESULT["sam3d_recon"] = recon
    result = bpy.ops.pose_capture.import_sam3d_mesh()
    check(result == {"FINISHED"}, "操作器返回 FINISHED", str(result))
    obj = bpy.data.objects.get("SAM3DBody")
    check(obj is not None and obj.type == "MESH", "场景里有 SAM3DBody")
    if obj is not None:
        check(len(obj.data.vertices) == len(k3), "顶点数一致",
              "%d" % len(obj.data.vertices))
        lowest = min(float((obj.matrix_world @ vertex.co).z) for vertex in obj.data.vertices)
        floor = float(metrics.get("lowest_foot_z") or 0.0)
        check(abs(lowest - floor) < 0.2, "脚落在骨架地面附近",
              "网格最低 %.3f / 骨架地面 %.3f" % (lowest, floor))
        pelvis = rig.matrix_world @ Vector(metrics["pelvis"])
        verts = [obj.matrix_world @ vertex.co for vertex in obj.data.vertices]
        nearest = min(verts, key=lambda vec: (vec - pelvis).length)
        check((nearest - pelvis).length < 0.25, "骨盆对齐（最近顶点）",
              "%.3f m" % (nearest - pelvis).length)


    print()
    if failures:
        print("!! 失败 %d 项：%s" % (len(failures), ", ".join(failures)))
        return 1
    print("通过 (all OK)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

