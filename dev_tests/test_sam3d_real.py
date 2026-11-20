"""真实 SAM 3D Body 结果 -> Rigify 骨架 + 网格 + 渲染（需要一张 worker 产出的 npz）。

用法：
  blender.exe -b --factory-startup --python "dev_tests/test_sam3d_real.py" -- D:\\code\\sam3d_out\\sam3d_0.npz

做的事（全程用真数据，不跑 torch）：
  1. sam3d.load_npz + build_recon；
  2. 建 Rigify 骨架 -> 自动映射 -> apply_pose，逐根骨骼比对方向；
  3. 调插件自己的 import_sam3d_mesh 导网格（顺带验证对齐）；
  4. 关节球 + 网格渲染两张图到 output/sam3d_real/。
"""
import math
import os
import sys

import bmesh
import bpy
import numpy as np
from mathutils import Vector


HERE = os.path.dirname(os.path.abspath(__file__))
ADDON_PARENT = os.path.dirname(HERE)
for path in (ADDON_PARENT, HERE):
    if path not in sys.path:
        sys.path.insert(0, path)

from test_sam3d_rig import bone_dir, setup_rig                 # noqa: E402

from pose_capture import operators, props, retarget, rigify_map, sam3d  # noqa: E402

failures = []


def check(ok, label, detail=""):
    print("  %s %s %s" % ("OK " if ok else "!! ", label, detail))
    if not ok:
        failures.append(label)
    return ok


def add_joint_spheres(name, points, radius=0.022):
    mesh = bpy.data.meshes.new(name)
    bm = bmesh.new()
    for position in points:
        try:
            geom = bmesh.ops.create_icosphere(bm, subdivisions=1, radius=radius)
        except TypeError:                      # 老版本参数名是 diameter
            geom = bmesh.ops.create_icosphere(bm, subdivisions=1, diameter=radius)
        bmesh.ops.translate(bm, vec=Vector(position), verts=geom["verts"])
    bm.to_mesh(mesh)
    bm.free()
    obj = bpy.data.objects.new(name, mesh)
    obj.color = (1.0, 0.25, 0.1, 1.0)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def add_bone_sticks(rig, radius=0.009, prefix="DEF-"):
    """把骨架的 DEF 变形骨画成细骨条（直观核对“骨架姿态”和“模型姿态”是否重合）。"""
    mesh = bpy.data.meshes.new("SAM3DBones")
    bm = bmesh.new()
    matrix_world = rig.matrix_world
    count = 0
    for pose_bone in rig.pose.bones:
        if not pose_bone.name.startswith(prefix):
            continue
        head = matrix_world @ pose_bone.head
        tail = matrix_world @ pose_bone.tail
        vector = tail - head
        if vector.length < 1e-4:
            continue
        try:
            geom = bmesh.ops.create_cone(bm, cap_ends=True, segments=5,
                                         radius1=radius, radius2=radius, depth=vector.length)
        except TypeError:
            geom = bmesh.ops.create_cone(bm, cap_ends=True, segments=5,
                                         diameter1=radius, diameter2=radius, depth=vector.length)
        rotation = vector.to_track_quat("Z", "Y")
        middle = (head + tail) * 0.5
        for vert in geom["verts"]:
            vert.co = rotation @ vert.co + middle
        count += 1
    bm.to_mesh(mesh)
    bm.free()
    obj = bpy.data.objects.new("SAM3DBones", mesh)
    obj.color = (1.0, 0.55, 0.06, 1.0)
    bpy.context.scene.collection.objects.link(obj)
    return obj, count


def render(path, location, target, lens=52, resolution=(760, 1000)):

    scene = bpy.context.scene
    camera = bpy.data.objects.get("PC_TestCam")
    if camera is None:
        data = bpy.data.cameras.new("PC_TestCam")
        camera = bpy.data.objects.new("PC_TestCam", data)
        scene.collection.objects.link(camera)
    scene.camera = camera
    camera.data.lens = lens
    camera.location = Vector(location)
    camera.rotation_euler = (Vector(target) - Vector(location)).to_track_quat("-Z", "Y").to_euler()

    scene.render.engine = "BLENDER_WORKBENCH"
    shading = scene.display.shading
    shading.light = "STUDIO"
    shading.color_type = "OBJECT"
    shading.show_shadows = True
    scene.render.resolution_x, scene.render.resolution_y = resolution
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.filepath = path
    bpy.ops.render.render(write_still=True)
    return os.path.isfile(path)


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    if not argv or not os.path.isfile(argv[0]):
        print("跳过：没有给 npz（blender ... -- <path>）")
        return 0
    npz = argv[0]
    print("真实数据：%s" % npz)

    print("\n[1] 读取 + 换算")
    data = sam3d.load_npz(npz)
    recon = sam3d.build_recon(data, {"fingers": True})
    check(recon.get("ok"), "build_recon", "yaw %.1f°，重投影 %.2f px"
          % (recon["yaw_deg"], recon["sam3d"]["reproj_error"]))
    check(recon["sam3d"]["has_mesh"], "带网格",
          "%d 顶点" % recon["sam3d"]["mesh_vertices"])
    want = ("shoulder_origin.L", "shoulder.R", "elbow.L", "wrist.R", "hand_tip.L",
            "hip.R", "knee.L", "ankle.R", "ball.L", "head_base", "head_tip", "neck_start")
    missing = [key for key in want if key not in recon["joints"]]
    check(not missing, "角色点齐全", missing or "%d 项" % len(recon["joints"]))

    print("\n[2] 写进 Rigify")
    rig = setup_rig()
    settings = bpy.context.scene.pose_capture
    settings.armature = rig
    settings.limb_mode = "FK"
    settings.fingers = True
    bmap = rigify_map.auto_detect(rig, True, {"Arm": "FK", "Leg": "FK"})
    metrics = rigify_map.read_metrics(rig, bmap)
    check(not metrics.get("errors"), "骨架测量", "；".join(metrics.get("errors") or []) or "ok")
    report = retarget.apply_pose(rig, bmap, metrics, recon, props.options_from_settings(settings))
    check(report.get("ok"), "apply_pose", report.get("message", ""))
    print("     写入：%s" % ", ".join(sorted(report.get("applied") or [])[:12]))

    pairs = (("upper_arm.L", "shoulder.L", "elbow.L", "upper_arm_fk.L"),
             ("forearm.L", "elbow.L", "wrist.L", "forearm_fk.L"),
             ("upper_arm.R", "shoulder.R", "elbow.R", "upper_arm_fk.R"),
             ("thigh.L", "hip.L", "knee.L", "thigh_fk.L"),
             ("shin.R", "knee.R", "ankle.R", "shin_fk.R"),
             ("head", "head_base", "head_tip", "head"))
    worst = 0.0
    for role, start, end, name in pairs:
        target = Vector(recon["joints"][end]) - Vector(recon["joints"][start])
        target.normalize()
        posed = bone_dir(rig, name)
        if posed is None:
            check(False, "%s 找不到骨骼 %s" % (role, name))
            continue
        angle = math.degrees(math.acos(max(-1.0, min(1.0, posed.dot(target)))))
        worst = max(worst, angle)
        check(angle < 2.0, "%s（%s）" % (role, name), "差 %.2f°" % angle)
    print("     最大方向误差 %.2f°" % worst)

    print("\n[3] 导入网格")
    operators._RESULT["sam3d"] = dict(data)
    operators._RESULT["sam3d_recon"] = recon
    result = bpy.ops.pose_capture.import_sam3d_mesh()
    obj = bpy.data.objects.get("SAM3DBody")
    check(result == {"FINISHED"} and obj is not None, "import_sam3d_mesh",
          "%d 顶点" % (len(obj.data.vertices) if obj else 0))

    joints = recon["joints"]
    scale = float(operators._LAST_SAM3D_ALIGN.get("scale", 1.0))
    offset = Vector(operators._LAST_SAM3D_ALIGN.get("offset", (0.0, 0.0, 0.0)))
    print("     对齐：缩放 %.3f，平移 %s" % (scale, tuple(round(v, 3) for v in offset)))
    raw = np.asarray(data["vertices"], dtype=float)
    plugin_verts, _faces = sam3d.mesh_arrays(data, recon)
    span = float(plugin_verts[:, 2].max() - plugin_verts[:, 2].min())
    check(1.4 < span < 2.1, "模型网格高度像个人", "%.2f m（未缩放）" % span)

    if obj is not None:
        zs = [(obj.matrix_world @ vertex.co).z for vertex in obj.data.vertices]
        check(abs(min(zs) - float(metrics.get("lowest_foot_z") or 0.0)) < 0.05,
              "脚落在骨架地面",
              "网格最低 %.3f / 骨架地面 %.3f" % (min(zs), float(metrics.get("lowest_foot_z") or 0.0)))


    print("\n[4] 渲染")
    key_joints = ("pelvis", "chest", "neck_start", "head_base", "head_tip",
                  "shoulder.L", "shoulder.R", "elbow.L", "elbow.R",
                  "wrist.L", "wrist.R", "hand_tip.L", "hand_tip.R",
                  "hip.L", "hip.R", "knee.L", "knee.R", "ankle.L", "ankle.R",
                  "ball.L", "ball.R")
    positions = [Vector(joints[name]) * scale + offset for name in key_joints]
    add_joint_spheres("SAM3DJoints", positions)
    bone_obj, bone_count = add_bone_sticks(rig)
    check(bone_count >= 20, "DEF 骨条", "%d 根" % bone_count)
    if obj is not None:
        obj.color = (0.45, 0.55, 0.78, 1.0)


    out_dir = os.path.join(ADDON_PARENT, "output", "sam3d_real")
    os.makedirs(out_dir, exist_ok=True)
    mesh_obj = obj
    joints_obj = bpy.data.objects.get("SAM3DJoints")
    bones_obj = bpy.data.objects.get("SAM3DBones")
    every = (mesh_obj, joints_obj, bones_obj)

    def show(*visible):
        for item in every:
            if item is None:
                continue
            item.hide_render = item not in visible

    show(mesh_obj)
    check(render(os.path.join(out_dir, "front.png"), (0.0, -3.6, 1.35), (0.0, 0.0, 0.95)),
          "正视图：只有网格（模型重建结果）")
    check(render(os.path.join(out_dir, "side.png"), (3.4, -1.2, 1.6), (0.0, 0.0, 0.95)),
          "侧视图：只有网格")
    show(bones_obj, joints_obj)
    check(render(os.path.join(out_dir, "skeleton.png"), (0.0, -3.6, 1.35), (0.0, 0.0, 0.95)),
          "骨架图：Rigify DEF 骨条 + 模型关节球")
    show(mesh_obj, bones_obj)
    check(render(os.path.join(out_dir, "overlay.png"), (0.0, -3.6, 1.35), (0.0, 0.0, 0.95)),
          "叠加图：网格 + 骨架")


    print()
    if failures:
        print("!! 失败 %d 项：%s" % (len(failures), ", ".join(failures)))
        return 1
    print("通过 (all OK)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

