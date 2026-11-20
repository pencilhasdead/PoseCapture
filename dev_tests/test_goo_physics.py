"""Goo Physics 一键加物理自检（需要装了 ``goo_physics`` 插件；没装就跳过）。

覆盖：
  1. 骨名分类（头发 / 裙子 / 飘带，日文·中文·英文）；
  2. AUTO 类型：头发/飘带=软体链、裙子=布料链；
  3. 一键加物理：骨头上真的挂上了物理标记、碰撞体对象生成；
  4. 一键移除：标记清空。

用法：
  blender.exe -b --factory-startup --python "dev_tests/test_goo_physics.py"
"""
import os
import sys

import bpy

HERE = os.path.dirname(os.path.abspath(__file__))
ADDON_PARENT = os.path.dirname(HERE)
for path in (ADDON_PARENT, HERE):
    if path not in sys.path:
        sys.path.insert(0, path)

from test_pipeline import ensure_registered                     # noqa: E402

from pose_capture import goo_physics                            # noqa: E402

ensure_registered()

failures = []


def check(ok, label, detail=""):
    print("  %s %s %s" % ("OK " if ok else "!! ", label, detail))
    if not ok:
        failures.append(label)
    return ok


SKELETON = (
    ("Hips", None, (0, 0, 0.90), (0, 0, 1.00)),
    ("Spine", "Hips", (0, 0, 1.00), (0, 0, 1.20)),
    ("Head", "Spine", (0, 0, 1.40), (0, 0, 1.60)),
    ("髪1", "Head", (0.05, 0, 1.55), (0.05, 0, 1.45)),
    ("髪2", "髪1", (0.05, 0, 1.45), (0.05, 0, 1.35)),
    ("髪3", "髪2", (0.05, 0, 1.35), (0.05, 0, 1.25)),
    ("裙_0_0", "Hips", (0, 0.02, 0.90), (0, 0.04, 0.78)),
    ("裙_0_1", "Hips", (0.06, 0.02, 0.90), (0.06, 0.04, 0.78)),
    ("裙_0_2", "Hips", (-0.06, 0.02, 0.90), (-0.06, 0.04, 0.78)),
    ("リボン1", "Spine", (0, 0.06, 1.15), (0, 0.07, 1.02)),
    ("リボン2", "リボン1", (0, 0.07, 1.02), (0, 0.08, 0.90)),
)


def build_rig():
    armature = bpy.data.armatures.new("GooRig")
    arm = bpy.data.objects.new("GooRig", armature)
    bpy.context.scene.collection.objects.link(arm)
    bpy.context.view_layer.objects.active = arm
    arm.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    made = {}
    for name, parent, head, tail in SKELETON:
        bone = armature.edit_bones.new(name)
        bone.head = head
        bone.tail = tail
        if parent:
            bone.parent = made[parent]
        made[name] = bone
    bpy.ops.object.mode_set(mode="OBJECT")
    return arm


def build_mesh(arm):
    bpy.ops.mesh.primitive_cube_add(size=0.3, location=(0, 0, 1.0))
    obj = bpy.context.active_object
    obj.name = "GooBody"
    modifier = obj.modifiers.new("Armature", "ARMATURE")
    modifier.object = arm
    return obj


def has_flag(arm, name: str, flag: str) -> bool:
    pose_bone = arm.pose.bones.get(name)
    return bool(pose_bone is not None and getattr(pose_bone, flag, False))


def test_classify():
    print("\n[1] 骨名分类 / 类型选择（纯逻辑）")
    groups = goo_physics.classify_bones([name for name, *_rest in SKELETON])
    check(groups["hair"] == ["髪1", "髪2", "髪3"], "头发", str(groups["hair"]))
    check(groups["skirt"] == ["裙_0_0", "裙_0_1", "裙_0_2"], "裙子（中文“裙”）",
          str(groups["skirt"]))
    check(groups["ribbon"] == ["リボン1", "リボン2"], "飘带（リボン）",
          str(groups["ribbon"]))
    more = goo_physics.classify_bones(
        ["hair_01", "Skirt_1", "ponytail1", "スカート_0_1", "Hips", "Spine"])
    check(more["hair"] == ["hair_01", "ponytail1"], "英文 hair / ponytail",
          str(more["hair"]))
    check(more["skirt"] == ["Skirt_1", "スカート_0_1"], "英文 Skirt / スカート",
          str(more["skirt"]))
    check(not any("Hips" in names or "Spine" in names for names in more.values()),
          "身体骨不会被误判")
    check(goo_physics.type_for_kind("hair", "AUTO") == "SOFT"
          and goo_physics.type_for_kind("skirt", "AUTO") == "CLOTH"
          and goo_physics.type_for_kind("hair", "GEO") == "GEO", "AUTO 类型选择")
    check(goo_physics.chain_count({"髪1": "Head", "髪2": "髪1", "髪3": "髪2"},
                                  ["髪1", "髪2", "髪3"]) == 1, "头发算 1 条链")


def set_selected(arm, names):
    """Blender 5 起用 ``PoseBone.select``，更早的版本用 ``Bone.select``。"""
    names = set(names)
    for pose_bone in arm.pose.bones:
        state = pose_bone.name in names
        if bpy.app.version[0] >= 5:
            pose_bone.select = state
        else:
            pose_bone.bone.select = state


def test_apply():
    print("\n[2] 一键加物理")
    bpy.ops.wm.read_factory_settings(use_empty=True)   # 注意：这会重置插件，之后要重新启用
    if "goo_physics" not in bpy.context.preferences.addons:
        try:
            bpy.ops.preferences.addon_enable(module="goo_physics")
        except Exception as exc:  # noqa: BLE001
            print("      （启用 goo_physics 失败：%r）" % exc)
    status = goo_physics.status()

    print("      %s" % status["message"])
    if not (status["enabled"] and status["operators"]):
        print("  (跳过：Goo Physics 没启用)")
        return

    settings = bpy.context.scene.pose_capture
    arm = build_rig()
    build_mesh(arm)

    settings.armature = arm
    settings.goo_hair = True
    settings.goo_skirt = True
    settings.goo_ribbon = True
    settings.goo_type = "AUTO"
    settings.goo_collider = True

    result = bpy.ops.pose_capture.goo_physics_add()
    check(result == {"FINISHED"}, "操作器返回 FINISHED", str(result))
    check(has_flag(arm, "髪1", "gp_has_sb_physics"), "头发挂了软体链")
    check(has_flag(arm, "髪3", "gp_has_sb_physics"), "头发链末端也有物理")
    check(has_flag(arm, "裙_0_0", "gp_has_cl_physics"), "裙子挂了布料链")
    check(has_flag(arm, "リボン1", "gp_has_sb_physics"), "飘带挂了软体链")
    check(not has_flag(arm, "Hips", "gp_has_sb_physics")
          and not has_flag(arm, "Hips", "gp_has_cl_physics"), "身体骨没被加物理")

    collider = bpy.data.objects.get("GP_Selected_Collision Objects")
    check(collider is not None and collider.type == "MESH", "碰撞体对象已生成",
          getattr(collider, "name", "无"))
    text = settings.goo_result or ""
    check("头发" in text and "裙子" in text, "面板结果文案", text[:70])
    check(goo_physics.goo_prefs().gp_physics_active is True, "Physics Active 已打开")

    print("\n[3] 每类上限 / 只对选中骨加物理")
    arm = build_rig()
    build_mesh(arm)
    settings.armature = arm
    settings.goo_hair = True
    settings.goo_skirt = True
    settings.goo_ribbon = True
    settings.goo_type = "AUTO"
    settings.goo_collider = False
    settings.goo_max_bones = 2
    settings.goo_selected_only = False
    result = bpy.ops.pose_capture.goo_physics_add()
    check(result == {"FINISHED"}, "上限模式返回 FINISHED", str(result))
    hair_on = [name for name in ("髪1", "髪2", "髪3") if has_flag(arm, name, "gp_has_sb_physics")]
    check(len(hair_on) == 2, "头发只加了前 2 根（上限）", str(hair_on))
    check("超过上限" in (settings.goo_result or ""), "结果里提示了上限",
          (settings.goo_result or "")[:70])
    bpy.ops.pose_capture.goo_physics_remove()

    print("\n[4] 只对选中骨（自定义骨）")
    settings.goo_max_bones = 64
    settings.goo_selected_only = True
    settings.goo_type = "SOFT"
    bpy.context.view_layer.objects.active = arm
    arm.select_set(True)
    if arm.mode != "POSE":
        bpy.ops.object.mode_set(mode="POSE")
    set_selected(arm, ("髪1", "髪2"))
    arm.data.bones.active = arm.data.bones["髪1"]
    bpy.context.view_layer.update()
    result = bpy.ops.pose_capture.goo_physics_add()
    check(result == {"FINISHED"}, "选中骨模式返回 FINISHED", str(result))
    check(has_flag(arm, "髪1", "gp_has_sb_physics") and has_flag(arm, "髪2", "gp_has_sb_physics"),
          "只给选中的两根头发加了物理")
    check(not has_flag(arm, "髪3", "gp_has_sb_physics"), "没选中的骨没被加")
    check(not has_flag(arm, "裙_0_0", "gp_has_cl_physics"), "裙子没被顺手加（只处理选中骨）")
    bpy.ops.pose_capture.goo_physics_remove()

    print("\n[5] 一键移除")
    settings.goo_selected_only = False
    settings.goo_max_bones = 64
    result = bpy.ops.pose_capture.goo_physics_add()
    check(result == {"FINISHED"}, "全量再加一次", str(result))
    result = bpy.ops.pose_capture.goo_physics_remove()
    check(result == {"FINISHED"}, "移除返回 FINISHED", str(result))
    left = [name for name, *_rest in SKELETON
            if any(getattr(arm.pose.bones[name], flag, False) for flag in
                   ("gp_has_sb_physics", "gp_has_cl_physics", "gp_has_gn_physics",
                    "gp_has_jg_physics"))]
    check(not left, "物理标记都清掉了", left or "无")


def main():
    print("Goo Physics 一键加物理自检")
    try:
        test_classify()
        test_apply()
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

