"""Goo Physics 桥接：一键给**头发 / 裙子 / 飘带**链加物理模拟。

Goo Physics（外部插件，模块名 ``goo_physics``）的操作器吃 **pose 模式下的选骨**：

* ``goophys.add_soft_physics_to_selected``      软体链（Soft Body）
* ``goophys.add_cloth_physics_to_selected``     布料链（Cloth）
* ``goophys.add_geonodes_physics_to_selected``  几何节点链（Geo Nodes）
* ``goophys.add_jiggle_physics_to_selected``    抖动（单骨，要求“断开父级 / 保持偏移”）
* ``goophys.remove_physics_from_selected``      移除
* ``goophys.add_collider``                      给选中的网格生成碰撞体

类型由它自己的偏好 ``gp_physics_type`` 决定，所以本模块的职责是：
**认骨头（头发 / 裙子 / 飘带）→ 选上 → 设类型 → 调操作器 → 回来核对**。

本模块不 import bpy 到模块级（面板/逻辑测试可以在无 Blender 时导入）。
"""
from __future__ import annotations

import unicodedata

GOO_ADDON = "goo_physics"

# key -> (Goo 偏好里的类型值, 操作器名（bpy.ops.goophys.<名字>）, 加完后骨骼上的“已加物理”标记)
GOO_TYPES = {
    "SOFT": ("SIMP_SOFTBODY", "add_soft_physics_to_selected", "gp_has_sb_physics"),
    "CLOTH": ("CLOTH", "add_cloth_physics_to_selected", "gp_has_cl_physics"),
    "GEO": ("GEO_NODES", "add_geonodes_physics_to_selected", "gp_has_gn_physics"),
    "JIGGLE": ("JIGGLE", "add_jiggle_physics_to_selected", "gp_has_jg_physics"),
}

TYPE_LABELS = {"SOFT": "软体链", "CLOTH": "布料链", "GEO": "几何节点链", "JIGGLE": "抖动"}
KINDS = ("hair", "skirt", "ribbon")
KIND_LABELS = {"hair": "头发", "skirt": "裙子", "ribbon": "飘带"}
_STATUS_CACHE: dict = {"time": 0.0, "value": None}


# 骨名关键字（日文 / 中文 / 英文都覆盖；先匹配“头发”，再“裙子”，最后“飘带”）
KEYWORDS = {
    "hair": ("髪", "髮", "hair", "bangs", "劉海", "刘海", "アホ毛", "ahoge",
             "ツインテ", "twintail", "ポニテ", "馬尾", "马尾", "ponytail", "braid",
             "お団子", "sideburn", "もみあげ"),
    "skirt": ("スカート", "skirt", "裙", "裾", "hem", "摆", "襬"),
    "ribbon": ("リボン", "ribbon", "ネクタイ", "necktie", "領結", "結", "紐", "ひも",
               "string", "sash", "帶", "带", "飘带", "飄帶", "缎带", "tail", "尻尾",
               "しっぽ", "cape", "マント", "scarf", "マフラー", "veil", "ベール"),
}


def _norm(text: str) -> str:
    return unicodedata.normalize("NFKC", str(text)).lower()


def classify_bones(bone_names) -> dict[str, list[str]]:
    """按名字把骨分到 hair / skirt / ribbon（每根只归一类，认不出就不管）。"""
    groups: dict[str, list[str]] = {kind: [] for kind in KINDS}
    for name in bone_names:
        lowered = _norm(name)
        for kind in KINDS:
            if any(word in name or _norm(word) in lowered for word in KEYWORDS[kind]):
                groups[kind].append(name)
                break
    return groups


def chain_count(parents: dict[str, str | None], names) -> int:
    """数一下这批骨能组成多少条链（父级不在其中 = 链根）。"""
    picked = set(names)
    roots = [name for name in names if parents.get(name) not in picked]
    return len(roots)


def type_for_kind(kind: str, requested: str) -> str:
    """``AUTO``：裙子用布料链，头发 / 飘带用软体链。"""
    key = str(requested or "AUTO").upper()
    if key in GOO_TYPES:
        return key
    return "CLOTH" if kind == "skirt" else "SOFT"


def status() -> dict:
    """Goo Physics 装没装 / 启用没启用（面板每次重绘都会调，做 2 秒缓存）。"""
    import time

    import bpy

    global _STATUS_CACHE
    now = time.time()
    if _STATUS_CACHE.get("value") is not None and now - _STATUS_CACHE.get("time", 0) < 2.0:
        return _STATUS_CACHE["value"]

    entry = bpy.context.preferences.addons.get(GOO_ADDON)
    enabled = entry is not None
    operators = hasattr(bpy.ops.goophys, "add_soft_physics_to_selected")
    installed = enabled
    if not installed:
        try:
            import addon_utils

            installed = any(getattr(module, "__name__", "") == GOO_ADDON
                            for module in addon_utils.modules())
        except Exception:  # noqa: BLE001
            installed = False
    if enabled and operators:
        message = "Goo Physics 已启用"
    elif installed and not enabled:
        message = "装了 Goo Physics 但没启用（偏好设置 → 附加组件里勾选）"
    else:
        message = "没有找到 Goo Physics 插件"
    value = {"installed": installed, "enabled": enabled, "operators": operators,
             "message": message}
    _STATUS_CACHE = {"time": now, "value": value}
    return value



def goo_prefs():
    """Goo Physics 的 AddonPreferences（没启用就返回 None）。"""
    import bpy

    entry = bpy.context.preferences.addons.get(GOO_ADDON)
    if entry is None:
        return None
    return getattr(entry, "preferences", None)


# --------------------------------------------------------------------------- 执行（bpy）
def _ensure_pose_mode(context, arm_obj) -> None:
    import bpy

    if context.object is not arm_obj:
        for obj in list(context.selected_objects):
            obj.select_set(False)
        arm_obj.select_set(True)
        context.view_layer.objects.active = arm_obj
    if arm_obj.mode != "POSE":
        bpy.ops.object.mode_set(mode="POSE")


def _select_pose_bones(arm_obj, names) -> list:
    """只选中给定骨头，并挑第一根当活动骨（Goo 的预设是套在活动骨上的）。"""
    wanted = set(names)
    picked = []
    for pose_bone in arm_obj.pose.bones:
        selected = pose_bone.name in wanted
        try:
            pose_bone.bone.select = selected
        except Exception:  # noqa: BLE001
            pass
        try:
            pose_bone.select = selected
        except Exception:  # noqa: BLE001
            pass
        if selected:
            picked.append(pose_bone)
    if picked:
        try:
            arm_obj.data.bones.active = picked[0].bone
        except Exception:  # noqa: BLE001
            pass
    return picked


def bones_by_kind(arm_obj) -> dict[str, list[str]]:
    """把骨架里“能加物理”的头发 / 裙子 / 飘带骨挑出来（必须有父级，Goo 要求）。"""
    usable = [bone.name for bone in arm_obj.data.bones if bone.parent is not None]
    return classify_bones(usable)


def mesh_objects(arm_obj) -> list:
    """带指向这根骨架的 Armature 修改器的网格（碰撞体就用它们）。"""
    import bpy

    found = []
    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        if any(modifier.type == "ARMATURE" and modifier.object == arm_obj
               for modifier in obj.modifiers):
            found.append(obj)
    return found



def apply(context, arm_obj, kinds, requested_type: str = "AUTO",
          add_collider: bool = True) -> dict:
    """给选中的几类骨一键加物理，返回报告。"""
    import bpy

    info = status()
    report = {"ok": False, "groups": {}, "collider": False,
              "warnings": [], "message": info["message"]}
    if not (info["enabled"] and info["operators"]):
        report["warnings"].append(info["message"])
        return report

    prefs = goo_prefs()
    if prefs is None:
        report["warnings"].append("读不到 Goo Physics 的偏好设置")
        return report

    groups = bones_by_kind(arm_obj)
    _ensure_pose_mode(context, arm_obj)
    done = 0
    for kind in kinds:
        names = groups.get(kind) or []
        if not names:
            continue
        key = type_for_kind(kind, requested_type)
        pref_value, operator_short, flag = GOO_TYPES[key]
        prefs.gp_physics_type = pref_value
        picked = _select_pose_bones(arm_obj, names)
        if not picked:
            continue
        context.view_layer.update()
        try:
            result = getattr(bpy.ops.goophys, operator_short)()
        except Exception as exc:  # noqa: BLE001
            report["warnings"].append("%s 加物理失败：%s" % (KIND_LABELS[kind], exc))
            continue
        if "FINISHED" not in result:
            report["warnings"].append("%s 加物理被取消（%s）" % (KIND_LABELS[kind], result))
            continue
        attached = [pb for pb in picked if getattr(pb, flag, False)]
        report["groups"][kind] = {"bones": len(names), "type": key,
                                  "attached": len(attached),
                                  "operator": "goophys." + operator_short}
        if attached:
            done += len(attached)
        else:
            report["warnings"].append("%s：调用了 goophys.%s，但骨头上没有物理标记"
                                      "（可能没选上）" % (KIND_LABELS[kind], operator_short))


    if done and hasattr(prefs, "gp_physics_active"):
        prefs.gp_physics_active = True

    if add_collider:
        meshes = mesh_objects(arm_obj)
        if meshes:
            if arm_obj.mode != "OBJECT":
                bpy.ops.object.mode_set(mode="OBJECT")
            for obj in list(context.selected_objects):
                obj.select_set(False)
            for obj in meshes:
                obj.select_set(True)
            context.view_layer.objects.active = meshes[0]
            try:
                result = bpy.ops.goophys.add_collider()
                report["collider"] = "FINISHED" in result
                if not report["collider"]:
                    report["warnings"].append("碰撞体被取消（%s）" % (result,))
            except Exception as exc:  # noqa: BLE001
                report["warnings"].append("加碰撞体失败：%s" % exc)
        else:
            report["warnings"].append("没找到带骨架修改器的网格，跳过碰撞体")

    report["ok"] = done > 0
    return report



def remove(context, arm_obj) -> dict:
    """把骨架里所有头发 / 裙子 / 飘带骨上的物理移除。"""
    import bpy

    info = status()
    report = {"ok": False, "removed": 0, "message": info["message"], "warnings": []}
    if not (info["enabled"] and info["operators"]):
        report["warnings"].append(info["message"])
        return report

    groups = bones_by_kind(arm_obj)
    names = [name for kind in KINDS for name in groups.get(kind) or []]
    if not names:
        report["warnings"].append("没认出头发 / 裙子 / 飘带骨")
        return report

    _ensure_pose_mode(context, arm_obj)
    picked = _select_pose_bones(arm_obj, names)
    context.view_layer.update()
    flags = tuple(GOO_TYPES[key][2] for key in GOO_TYPES)
    before = [pb for pb in picked if any(getattr(pb, flag, False) for flag in flags)]
    try:
        result = bpy.ops.goophys.remove_physics_from_selected()
    except Exception as exc:  # noqa: BLE001
        report["warnings"].append("移除失败：%s" % exc)
        return report
    if "FINISHED" not in result:
        report["warnings"].append("移除被取消（%s）" % (result,))
        return report
    left = [pb for pb in picked if any(getattr(pb, flag, False) for flag in flags)]
    report["removed"] = max(len(before) - len(left), 0)

    report["ok"] = not left
    if left:
        report["warnings"].append("还有 %d 根骨带着物理" % len(left))
    return report


def summarize(report: dict) -> str:
    parts = []
    for kind, data in (report.get("groups") or {}).items():
        parts.append("%s %d 根（%s，已挂 %d）"
                     % (KIND_LABELS.get(kind, kind), data["bones"],
                        TYPE_LABELS.get(data["type"], data["type"]), data["attached"]))
    if report.get("removed"):
        parts.append("移除 %d 根" % report["removed"])
    if report.get("collider"):
        parts.append("碰撞体已生成")
    if not parts:
        parts.append(report.get("message") or "没有可加物理的骨")
    warnings = report.get("warnings") or []
    if warnings:
        parts.append("⚠ " + "；".join(str(item) for item in warnings[:2]))
    return "；".join(parts)

