"""MMD → Mixamo 的 Blender 自检：合成一根 MMD 骨架（日文骨名）+ 网格顶点组，
跑插件的转换操作器，核对骨名 / 层级 / 顶点组 / 权重合并 / 控制骨删除 / 头发裙子保留。

用法：
  blender.exe -b --factory-startup --python "dev_tests/test_mmd_mixamo_rig.py"
"""
import os
import sys

import bpy

HERE = os.path.dirname(os.path.abspath(__file__))
ADDON_PARENT = os.path.dirname(HERE)
for path in (ADDON_PARENT, HERE):
    if path not in sys.path:
        sys.path.insert(0, path)

from test_pipeline import ensure_registered                       # noqa: E402

from pose_capture import props                                    # noqa: E402

ensure_registered()

failures = []


def check(ok, label, detail=""):
    print("  %s %s %s" % ("OK " if ok else "!! ", label, detail))
    if not ok:
        failures.append(label)
    return ok


# name, parent, head, tail
SKELETON = (
    ("全ての親", None, (0, 0, 0.00), (0, 0, 0.20)),
    ("センター", "全ての親", (0, 0, 0.90), (0, 0, 1.00)),
    ("グルーブ", "センター", (0, 0, 0.85), (0, 0, 0.95)),
    ("下半身", "センター", (0, 0, 0.95), (0, 0, 0.85)),
    ("上半身", "センター", (0, 0, 0.95), (0, 0, 1.15)),
    ("上半身2", "上半身", (0, 0, 1.15), (0, 0, 1.35)),
    ("首", "上半身2", (0, 0, 1.35), (0, 0, 1.45)),
    ("頭", "首", (0, 0, 1.45), (0, 0, 1.65)),
    ("左肩P", "上半身2", (0.02, 0, 1.32), (0.06, 0, 1.32)),
    ("左肩", "左肩P", (0.06, 0, 1.32), (0.12, 0, 1.32)),
    ("左腕", "左肩", (0.12, 0, 1.32), (0.32, 0, 1.32)),
    ("左腕捩", "左腕", (0.20, 0, 1.32), (0.30, 0, 1.32)),
    ("左ひじ", "左腕捩", (0.32, 0, 1.32), (0.52, 0, 1.32)),
    ("左手首", "左ひじ", (0.52, 0, 1.32), (0.60, 0, 1.32)),
    ("左中指1", "左手首", (0.60, 0, 1.32), (0.64, 0, 1.32)),
    ("左中指2", "左中指1", (0.64, 0, 1.32), (0.68, 0, 1.32)),
    ("左中指3", "左中指2", (0.68, 0, 1.32), (0.72, 0, 1.32)),
    ("左親指0", "左手首", (0.58, 0.02, 1.30), (0.61, 0.03, 1.31)),
    ("左親指1", "左親指0", (0.61, 0.03, 1.31), (0.64, 0.04, 1.32)),
    ("左親指2", "左親指1", (0.64, 0.04, 1.32), (0.66, 0.05, 1.32)),
    ("左小指1", "左手首", (0.58, -0.02, 1.30), (0.61, -0.02, 1.30)),
    ("左小指2", "左小指1", (0.61, -0.02, 1.30), (0.64, -0.03, 1.30)),
    ("左小指3", "左小指2", (0.64, -0.03, 1.30), (0.66, -0.03, 1.30)),
    ("右小指1", "右手首", (-0.50, -0.02, 1.30), (-0.53, -0.02, 1.30)),
    ("右小指2", "右小指1", (-0.53, -0.02, 1.30), (-0.56, -0.03, 1.30)),
    ("右小指3", "右小指2", (-0.56, -0.03, 1.30), (-0.58, -0.03, 1.30)),

    ("右肩", "上半身2", (-0.02, 0, 1.32), (-0.06, 0, 1.32)),
    ("右腕", "右肩", (-0.06, 0, 1.32), (-0.26, 0, 1.32)),
    ("右ひじ", "右腕", (-0.26, 0, 1.32), (-0.46, 0, 1.32)),
    ("右手首", "右ひじ", (-0.46, 0, 1.32), (-0.54, 0, 1.32)),
    ("左足", "下半身", (0.08, 0, 0.85), (0.08, 0, 0.45)),
    ("左ひざ", "左足", (0.08, 0, 0.45), (0.08, 0, 0.10)),
    ("左足首", "左ひざ", (0.08, 0, 0.10), (0.08, 0.06, 0.03)),
    ("左つま先", "左足首", (0.08, 0.06, 0.03), (0.08, 0.14, 0.03)),
    ("左足D", "左足", (0.08, 0, 0.80), (0.08, 0, 0.50)),
    ("左足ＩＫ", "全ての親", (0.08, 0, 0.03), (0.08, 0.10, 0.03)),
    ("左つま先ＩＫ", "左足ＩＫ", (0.08, 0.10, 0.03), (0.08, 0.16, 0.03)),
    ("右足", "下半身", (-0.08, 0, 0.85), (-0.08, 0, 0.45)),
    ("右ひざ", "右足", (-0.08, 0, 0.45), (-0.08, 0, 0.10)),
    ("右足首", "右ひざ", (-0.08, 0, 0.10), (-0.08, 0.06, 0.03)),
    ("右つま先", "右足首", (-0.08, 0.06, 0.03), (-0.08, 0.14, 0.03)),
    ("左髪1", "頭", (0.05, 0, 1.62), (0.05, 0, 1.47)),
    ("左髪2", "左髪1", (0.05, 0, 1.47), (0.05, 0, 1.32)),
    ("右髪1", "頭", (-0.05, 0, 1.62), (-0.05, 0, 1.47)),
    ("スカート_0_1", "下半身", (0, 0.02, 0.90), (0, 0.04, 0.75)),
    ("スカート_0_2", "スカート_0_1", (0, 0.04, 0.75), (0, 0.06, 0.60)),
    ("胸", "上半身2", (0.05, 0.08, 1.22), (0.05, 0.10, 1.14)),
)


def build_rig(name="MMD"):
    armature = bpy.data.armatures.new(name)
    arm = bpy.data.objects.new(name, armature)
    bpy.context.scene.collection.objects.link(arm)
    bpy.context.view_layer.objects.active = arm
    arm.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    made = {}
    pending = []
    for bone_name, parent, head, tail in SKELETON:
        bone = armature.edit_bones.new(bone_name)
        bone.head = head
        bone.tail = tail
        made[bone_name] = bone
        if parent:
            pending.append((bone_name, parent))
    for bone_name, parent in pending:          # 表里的顺序可以随便写，父级后建也行
        made[bone_name].parent = made[parent]
    bpy.ops.object.mode_set(mode="OBJECT")
    return arm



def build_mesh(arm, name="MMD_Mesh"):
    bpy.ops.mesh.primitive_cube_add(size=0.2, location=(0, 0, 1.0))
    obj = bpy.context.active_object
    obj.name = name
    obj.data.name = name
    modifier = obj.modifiers.new("Armature", "ARMATURE")
    modifier.object = arm

    def group(group_name, vertex, weight):
        handle = obj.vertex_groups.get(group_name) or obj.vertex_groups.new(name=group_name)
        handle.add([vertex], weight, "REPLACE")
        return handle

    group("下半身", 0, 1.0)
    group("左腕", 1, 1.0)
    group("左足", 2, 0.4)
    group("左足D", 2, 0.6)          # 会被并进 左足 -> LeftUpLeg
    group("左肩P", 3, 0.3)          # 会被并进 上半身2 -> Spine1
    group("左髪1", 4, 1.0)
    group("スカート_0_1", 5, 1.0)
    group("左足ＩＫ", 6, 1.0)       # 没有保留的祖先 -> 权重不合并
    return obj


def weight_of(obj, group_name, vertex_index):
    group = obj.vertex_groups.get(group_name)
    if group is None:
        return None
    for element in obj.data.vertices[vertex_index].groups:
        if element.group == group.index:
            return round(element.weight, 4)
    return 0.0


def run_conversion(settings, arm, duplicate=False, delete_source=False):
    settings.mmd_duplicate = duplicate
    settings.mmd_delete_source = delete_source
    settings.mmd_simplify = True
    settings.mmd_cleanup = True
    settings.mmd_translate_extras = True
    settings.mmd_rename_groups = True
    settings.mmd_merge_weights = True
    settings.mmd_strip_helpers = True
    settings.mmd_prefix = "mixamorig:"
    bpy.context.view_layer.objects.active = arm
    arm.select_set(True)
    return bpy.ops.pose_capture.mmd_to_mixamo()


def test_in_place():
    print("\n[1] 原地转换（不改原模型的选项关掉）")
    bpy.ops.wm.read_factory_settings(use_empty=True)
    settings = bpy.context.scene.pose_capture      # 重置文件后要重新取引用
    arm = build_rig("MMD")
    mesh = build_mesh(arm)


    # MMD Tools 的痕迹：约束 + 驱动器
    pose_bone = arm.pose.bones["左足"]
    pose_bone.constraints.new("COPY_ROTATION")
    driver = pose_bone.driver_add("location", 0)
    driver.driver.expression = "0.0"
    before_drivers = len(arm.animation_data.drivers)
    check(before_drivers >= 1 and len(pose_bone.constraints) == 1, "造好约束 + 驱动器",
          "drivers=%d" % before_drivers)

    result = run_conversion(settings, arm, duplicate=False)
    check(result == {"FINISHED"}, "操作器返回 FINISHED", str(result))

    bones = arm.data.bones
    print("\n[2] 骨名")
    expected = ("mixamorig:Hips", "mixamorig:Spine", "mixamorig:Spine1", "mixamorig:Neck",
                "mixamorig:Head", "mixamorig:LeftShoulder", "mixamorig:LeftArm",
                "mixamorig:LeftForeArm", "mixamorig:LeftHand", "mixamorig:LeftHandMiddle1",
                "mixamorig:LeftHandMiddle3", "mixamorig:LeftUpLeg", "mixamorig:LeftLeg",
                "mixamorig:LeftFoot", "mixamorig:LeftToe_End", "mixamorig:RightUpLeg",
                "mixamorig:RightFoot", "mixamorig:RightToe_End")
    missing = [name for name in expected if name not in bones]
    check(not missing, "18 个 Mixamo 骨名", missing or "齐全")
    gone = [name for name in ("下半身", "上半身", "首", "頭", "左腕", "左ひじ", "左足", "左ひざ",
                              "全ての親", "センター", "グルーブ", "左肩P", "左足ＩＫ",
                              "左つま先ＩＫ", "左足D")
            if name in bones]
    check(not gone, "旧名 / 控制骨都清掉了", gone or "干净")

    print("\n[3] 层级（Mixamo 要求 Spine 是 Hips 的子级）")
    def parent_of(name):
        bone = bones.get(name)
        return bone.parent.name if bone is not None and bone.parent else None

    check(parent_of("mixamorig:Hips") is None, "Hips 是根骨", str(parent_of("mixamorig:Hips")))
    check(parent_of("mixamorig:Spine") == "mixamorig:Hips", "Spine <- Hips",
          str(parent_of("mixamorig:Spine")))
    check(parent_of("mixamorig:Spine1") == "mixamorig:Spine", "Spine1 <- Spine")
    check(parent_of("mixamorig:Neck") == "mixamorig:Spine1", "Neck <- Spine1")
    check(parent_of("mixamorig:Head") == "mixamorig:Neck", "Head <- Neck")
    check(parent_of("mixamorig:LeftShoulder") == "mixamorig:Spine1", "Shoulder <- Spine1")
    check(parent_of("mixamorig:LeftArm") == "mixamorig:LeftShoulder", "Arm <- Shoulder")
    check(parent_of("mixamorig:LeftUpLeg") == "mixamorig:Hips", "UpLeg <- Hips",
          str(parent_of("mixamorig:LeftUpLeg")))


    print("\n[4] 头发 / 裙子 / 其他骨保留")
    for name in ("hair1.L", "hair2.L", "hair1.R", "skirt_0_1", "skirt_0_2", "breast"):
        check(name in bones, "保留 %s" % name)
    check(parent_of("hair1.L") == "mixamorig:Head", "头发挂在 Head 下",
          str(parent_of("hair1.L")))
    check(parent_of("hair2.L") == "hair1.L", "头发链完整")
    check(parent_of("skirt_0_1") == "mixamorig:Hips", "裙子挂在 Hips 下",
          str(parent_of("skirt_0_1")))
    check(parent_of("skirt_0_2") == "skirt_0_1", "裙子链完整")
    japanese = [bone.name for bone in bones if any(ord(char) > 0x2E80 for char in bone.name)]
    check(not japanese, "没有残留日文骨名", japanese or "无")

    print("\n[5] 顶点组 / 权重")
    groups = {group.name for group in mesh.vertex_groups}
    check("mixamorig:LeftArm" in groups and "左腕" not in groups, "左腕 顶点组改名")
    check("mixamorig:Hips" in groups and "下半身" not in groups, "下半身 顶点组改名")
    check("mixamorig:LeftUpLeg" in groups, "左足 -> LeftUpLeg 顶点组")
    merged = weight_of(mesh, "mixamorig:LeftUpLeg", 2)
    check(merged is not None and abs(merged - 1.0) < 1e-4, "足D 的权重并进 LeftUpLeg",
          "%s" % merged)
    check(weight_of(mesh, "左足D", 2) is None, "足D 顶点组已删除")
    shoulder = weight_of(mesh, "mixamorig:Spine1", 3)
    check(shoulder is not None and abs(shoulder - 0.3) < 1e-4, "肩P 的权重并进 Spine1",
          "%s" % shoulder)
    check(weight_of(mesh, "hair1.L", 4) == 1.0, "头发权重保留")
    check(weight_of(mesh, "skirt_0_1", 5) == 1.0, "裙子权重保留")

    print("\n[6] MMD Tools 痕迹")
    check(len(arm.pose.bones["mixamorig:LeftUpLeg"].constraints) == 0, "约束已摘掉")
    drivers = len(arm.animation_data.drivers) if arm.animation_data else 0
    check(drivers == 0, "驱动器已摘掉", "drivers=%d" % drivers)
    check(bool(settings.mmd_result) and "头发" in settings.mmd_result, "面板结果文案",
          (settings.mmd_result or "")[:50])
    check(settings.armature == arm, "目标骨架指到转换后的对象")


def test_duplicate():
    print("\n[7] 复制一份再改（原模型不动）")
    bpy.ops.wm.read_factory_settings(use_empty=True)
    settings = bpy.context.scene.pose_capture
    arm = build_rig("MMD_Origin")

    mesh = build_mesh(arm, "MMD_Origin_Mesh")
    result = run_conversion(settings, arm, duplicate=True)
    check(result == {"FINISHED"}, "操作器返回 FINISHED", str(result))

    new_arm = settings.armature
    check(new_arm is not None and new_arm != arm, "得到新骨架对象",
          getattr(new_arm, "name", "?"))
    check("下半身" in arm.data.bones and "mixamorig:Hips" not in arm.data.bones,
          "原骨架保持 MMD 骨名")
    check(new_arm is not None and "mixamorig:Hips" in new_arm.data.bones, "新骨架是 Mixamo 名")
    new_meshes = [obj for obj in bpy.data.objects
                  if obj.type == "MESH" and any(modifier.type == "ARMATURE"
                                                and modifier.object == new_arm
                                                for modifier in obj.modifiers)]
    check(bool(new_meshes), "网格也复制了一份并指向新骨架",
          ", ".join(obj.name for obj in new_meshes) or "无")
    check(any(modifier.type == "ARMATURE" and modifier.object == arm
              for modifier in mesh.modifiers), "原网格还指向原骨架")


def test_delete_source():
    print("\n[8] 转换后删掉原 PMX 模型（1.4.3 默认）")
    bpy.ops.wm.read_factory_settings(use_empty=True)
    settings = bpy.context.scene.pose_capture
    arm = build_rig("MMD_Origin")
    build_mesh(arm, "MMD_Origin_Mesh")
    result = run_conversion(settings, arm, duplicate=True, delete_source=True)
    check(result == {"FINISHED"}, "操作器返回 FINISHED", str(result))

    left = [obj.name for obj in bpy.data.objects if obj.name.startswith("MMD_Origin")]
    check(len(left) == 2 and not any(name[-3:].isdigit() for name in left),
          "原模型被删掉，只剩骨架 + 网格且没有 .001", str(left))
    new_arm = settings.armature
    check(new_arm is not None and new_arm.name == "MMD_Origin", "新骨架接回原名（没有 .001）",
          getattr(new_arm, "name", "?"))
    check(new_arm is not None and "mixamorig:Hips" in new_arm.data.bones, "留下的是 Mixamo 骨架")
    meshes = [obj for obj in bpy.data.objects if obj.type == "MESH"]
    check(len(meshes) == 1 and meshes[0].name == "MMD_Origin_Mesh",
          "网格也接回原名", ", ".join(obj.name for obj in meshes) or "无")
    check(settings.rig_type == "MIXAMO", "骨架标准自动切成 Mixamo", settings.rig_type)
    check(settings.limb_mode == "FK", "肢体写入方式自动切成 FK", settings.limb_mode)
    check(len(settings.bone_map) > 0 and bool(settings.bone_map[0].role),
          "角色映射已同步", "%d 行" % len(settings.bone_map))
    check("原 PMX 模型已删除" in (settings.mmd_result or ""), "面板写明了删除",
          (settings.mmd_result or "")[:60])
    check("骨骼映射" in (settings.mmd_result or ""), "面板写了映射角色数",
          (settings.mmd_result or "")[-40:])


def main():
    print("MMD → Mixamo Blender 自检")
    try:
        test_in_place()
        test_duplicate()
        test_delete_source()
    except Exception:  # noqa: BLE001  出错要看得见，不能静默退出
        import traceback

        traceback.print_exc()
        failures.append("异常中断")

    print()
    if failures:
        print("!! 失败 %d 项：%s" % (len(failures), ", ".join(failures)))
        return 1
    print("通过 (all OK)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

