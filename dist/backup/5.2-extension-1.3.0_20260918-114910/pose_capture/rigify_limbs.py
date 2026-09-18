"""Rigify 肢体 IK 控制器桥接（Rigify 的胳膊/大腿默认走 IK）。

背景（Blender 5.2 + 内置 rigify 实测）：
  * 生成骨架时 ``upper_arm_parent.L`` / ``thigh_parent.R`` 上的 ``IK_FK`` 自定义属性
    默认是 0，也就是 **IK 模式**；
  * IK 模式下变形骨跟的是 ``hand_ik`` / ``foot_ik`` 这些 IK 控制器，直接写 ``*_fk``
    骨骼的旋转不会改变形 —— 用户会以为“控制器坏了”；
  * Rigify 自带的 “IK->FK (hand.L)” 按钮本质是
    ``pose.rigify_limb_ik2fk_<rig_id>``：读 FK 链的姿态矩阵，算出 IK 控制器的变换。
    实测把 FK 姿态搬过去、再把 IK_FK 切回 0 之后，所有 DEF 骨的位置误差 0.000000 m，
    手指 / 脚趾也全部跟随。

于是插件写姿态的顺序是：

* **IK 模式（默认）**：直接把姿态写进 IK 控制器——链根 ``upper_arm_ik`` / ``thigh_ik`` 写旋转
  （决定整条 IK 链的姿态与弯曲平面）、末端 ``hand_ik`` / ``foot_ik`` 写朝向、IK 目标
  ``*_ik_target`` 按“手腕 / 脚踝关键点 - 静止位置”的位移搬过去；FK 链也顺手写一份（IK 模式下
  它是隐藏的，用户拖滑块回 FK 时姿态不丢）。写完用 ``DEF-*`` 变形骨核对一遍，对不上的肢体自动
  退回下面的老路径，所以最坏情况也只是“没直写成功”，不会写出错姿态。**不切 FK、不做快照**；
* **退回路径（老行为）**：切到 FK -> 写 FK 旋转 -> 调 Rigify 自带的 “IK->FK” 快照把 IK 控制器
  对齐（pole / 拉伸 / 脚跟的数学都由 Rigify 算）-> 停在 IK 或 FK。

切到 FK 时默认会把 IK 控制器那一组骨骼集合藏起来（反之亦然），否则用户滑到 FK 后
发现控制器不动，会以为出了 bug。
"""
from __future__ import annotations

import ast
import json
import re

import bpy

from . import utils

MODE_IK = "IK"
MODE_FK = "FK"
MODES = (MODE_IK, MODE_FK)

_JSON_KEYS = ("fk_bones", "ik_bones", "ctrl_bones", "tail_bones", "extra_ctrls")
_STR_KEYS = ("prop_bone", "pole_prop")

# 标准四肢的链名（生成脚本被删掉时兜底；只在骨骼确实存在时才用）
_DEFAULT_CHAINS = {
    "upper_arm": {"fk": ("upper_arm_fk.{s}", "forearm_fk.{s}", "hand_fk.{s}"),
                  "ik": ("upper_arm_ik.{s}", "MCH-forearm_ik.{s}",
                         "MCH-upper_arm_ik_target.{s}"),
                  "ctrl": ("upper_arm_ik.{s}", "upper_arm_ik_target.{s}", "hand_ik.{s}"),
                  "tail": (), "extra": (), "parent": "upper_arm_parent.{s}",
                  "kind": "Arm"},
    "thigh": {"fk": ("thigh_fk.{s}", "shin_fk.{s}", "foot_fk.{s}", "toe_fk.{s}"),
              "ik": ("thigh_ik.{s}", "MCH-shin_ik.{s}", "MCH-thigh_ik_target.{s}"),
              "ctrl": ("thigh_ik.{s}", "thigh_ik_target.{s}", "foot_ik.{s}"),
              "tail": ("toe_ik.{s}",),
              "extra": ("foot_heel_ik.{s}", "foot_spin_ik.{s}"),
              "parent": "thigh_parent.{s}", "kind": "Leg"},
}

_SPEC_CACHE: dict[tuple[str, str], dict] = {}


def rig_id(arm_obj) -> str:
    """Rigify 生成骨架在 armature 数据上存的 rig_id（非生成骨架返回空串）。"""
    if arm_obj is None or getattr(arm_obj, "type", "") != "ARMATURE":
        return ""
    try:
        return str(arm_obj.data.get("rig_id") or "")
    except Exception:  # noqa: BLE001
        return ""


def _generated_script(rid: str) -> str:
    """在所有文本块里找 Rigify 生成的那份 UI 脚本（里面带 limb ik2fk 操作符属性）。"""
    needle = "rigify_limb_ik2fk_%s" % rid
    for text in bpy.data.texts:
        try:
            body = text.as_string()
        except Exception:  # noqa: BLE001
            continue
        if needle in body:
            return body
    return ""


def _parse_operator_props(body: str, op_name: str) -> list[dict]:
    """把生成脚本里的 ``props.X = ...`` 解析成每个操作符一份属性字典。

    注意：其它操作符（比如 “FK->IK” 的 generic_snap 快照）也有 ``props.ctrl_bones``
    这类同名属性，所以**每个 ``.operator(=`` 行都要断块**，只留名字完全匹配的那些，
    否则会把别的块的属性并进来（合并结果会指向对侧肢体）。
    """
    found: list[dict] = []
    current: dict | None = None
    quoted = "'%s'" % op_name
    for line in body.splitlines():
        line = line.strip()
        if ".operator(" in line:
            current = {} if quoted in line else None
            if current is not None:
                found.append(current)
            continue
        match = re.match(r"props\.(\w+)\s*=\s*(.+)$", line)
        if match and current is not None:
            raw = match.group(2).strip()
            try:
                current[match.group(1)] = ast.literal_eval(raw)
            except (ValueError, SyntaxError):
                current[match.group(1)] = raw.strip("'\"")
    return found



def _spec_from_raw(raw: dict, arm_obj) -> dict | None:
    """属性字典 -> 规格字典（骨骼不存在就返回 None）。"""
    spec: dict = {}
    for key, value in raw.items():
        if key in _JSON_KEYS:
            if isinstance(value, str):
                try:
                    spec[key] = list(json.loads(value))
                except ValueError:
                    spec[key] = []
            else:
                spec[key] = list(value or [])
        elif key in _STR_KEYS:
            spec[key] = str(value)
    for key in _JSON_KEYS:
        spec.setdefault(key, [])
    prop_bone = spec.get("prop_bone") or ""
    bones = spec.get("fk_bones") or []
    ik_bones = spec.get("ik_bones") or []
    ctrl_bones = spec.get("ctrl_bones") or []
    tail_bones = spec.get("tail_bones") or []
    extra_ctrls = spec.get("extra_ctrls") or []
    if not prop_bone or not bones or not ctrl_bones:
        return None
    # Rigify 自己的断言：fk = ik + tail；控制骨与 IK 结果骨一一对应。
    # 用这两条把“属性被别的块污染”的块直接扔掉。
    if len(bones) != len(ik_bones) + len(tail_bones) or len(ctrl_bones) != len(ik_bones):
        return None
    pb = arm_obj.pose.bones.get(prop_bone)
    if pb is None or "IK_FK" not in pb.keys():
        return None
    for name in bones + ik_bones + ctrl_bones + tail_bones:
        if arm_obj.pose.bones.get(name) is None:
            return None
    base, _dot, side = bones[0].rpartition(".")
    if base.startswith("upper_arm"):
        spec["kind"], spec["side"] = "Arm", side
    elif base.startswith("thigh"):
        spec["kind"], spec["side"] = "Leg", side
    else:
        return None
    suffix = ".%s" % side
    for name in ik_bones + ctrl_bones + tail_bones + extra_ctrls:
        if not name.endswith(suffix):
            # 左右混了（多半是解析/生成脚本变了），宁可不用也不要写错边
            return None
    spec["pole_prop"] = spec.get("pole_prop") or "pole_vector"
    return spec


def _default_specs(arm_obj) -> dict:
    """生成脚本不在（被删/被改名）时，按 Rigify 标准命名兜底。"""
    specs: dict = {}
    for item in _DEFAULT_CHAINS.values():
        for side in ("L", "R"):
            fk_bones = [name.format(s=side) for name in item["fk"]]
            if any(arm_obj.pose.bones.get(name) is None for name in fk_bones):
                continue
            spec = {
                "prop_bone": item["parent"].format(s=side),
                "pole_prop": "pole_vector",
                "fk_bones": fk_bones,
                "ik_bones": [n.format(s=side) for n in item["ik"]],
                "ctrl_bones": [n.format(s=side) for n in item["ctrl"]],
                "tail_bones": [n.format(s=side) for n in item["tail"]],
                "extra_ctrls": [n.format(s=side) for n in item["extra"]],
                "kind": item["kind"],
                "side": side,
            }
            pb = arm_obj.pose.bones.get(spec["prop_bone"])
            if pb is None or "IK_FK" not in pb.keys():
                continue
            specs[fk_bones[0]] = spec
    return specs


def limb_specs(arm_obj) -> dict:
    """``{FK 链第一根骨骼: 规格}``；不是 Rigify 生成骨架时返回空字典。

    规格含 prop_bone / pole_prop / fk_bones / ik_bones / ctrl_bones /
    tail_bones / extra_ctrls / kind(Arm|Leg) / side(L|R)。
    """
    rid = rig_id(arm_obj)
    if not rid:
        return {}
    key = (arm_obj.name, rid)
    cached = _SPEC_CACHE.get(key)
    if cached is not None:
        return cached

    specs: dict = {}
    body = _generated_script(rid)
    if body:
        for raw in _parse_operator_props(body, "pose.rigify_limb_ik2fk_%s" % rid):
            spec = _spec_from_raw(raw, arm_obj)
            if spec is not None:
                specs.setdefault(spec["fk_bones"][0], spec)
    if not specs:
        specs = _default_specs(arm_obj)
    _SPEC_CACHE[key] = specs
    return specs


def _limb_controls(spec: dict) -> set:
    """这条肢体上“用户能直接摆”的控制器骨名（Rigify 的 ctrl / tail / extra 三组）。"""
    return set(spec.get("ctrl_bones") or []) | set(spec.get("tail_bones") or []) \
        | set(spec.get("extra_ctrls") or [])


def ik_write_plan(spec: dict) -> dict | None:
    """IK 模式下可以直接写姿态的控制器；认不出来时返回 None（调用方退回 FK 路径）。

    返回 ``{"side", "target", "controls": {角色: 控制骨}, "roles", "fk_of_role",
    "tip_role", "tip_fk"}``：

      * ``controls`` 只放“本身就是控制器”的 IK 骨——链根 ``upper_arm_ik.L`` /
        ``thigh_ik.L``（写旋转就带整条 IK 链的姿态与弯曲平面）和末端 ``hand_ik.L`` /
        ``foot_ik.L``（手 / 脚的朝向）。中间段由 IK 解算决定，不写；
      * ``target`` 是 IK 目标控制器（``upper_arm_ik_target.L``），只写位置；
      * ``tip_role`` / ``tip_fk``：这条肢体“终点”的角色与 FK 骨（``forearm_fk`` 的尾巴是
        手腕、``shin_fk`` 的尾巴是脚踝），关键点里对应的那个点就是 IK 目标要放的位置。
    """
    side = str(spec.get("side") or "")
    suffix = ".%s" % side
    fk_bones = list(spec.get("fk_bones") or [])
    if not side or len(fk_bones) < 2:
        return None

    target = ""
    for name in spec.get("ctrl_bones") or []:
        if name.endswith("_ik_target%s" % suffix):
            target = name
            break
    if not target:
        return None

    roles: list[str] = []
    stems: list[str] = []
    for name in fk_bones:
        if not name.endswith(suffix):
            return None
        stem = name[:-len(suffix)]
        if stem.endswith("_fk"):
            stem = stem[:-len("_fk")]
        stems.append(stem)
        roles.append("%s%s" % (stem, suffix))

    controls = _limb_controls(spec)
    writes: dict = {}
    for stem, role in zip(stems, roles):
        candidate = "%s_ik%s" % (stem, suffix)
        if candidate in controls:
            writes[role] = candidate
    if not writes:
        return None
    return {"side": side,
            "target": target,
            "controls": writes,
            "roles": roles,
            "fk_of_role": dict(zip(roles, fk_bones)),
            "tip_role": roles[1],
            "tip_fk": fk_bones[1]}


def mapped_limb_mode(arm_obj, bmap, specs: dict = None) -> str:
    """映射里的四肢骨整体指向哪种模式：``"IK"`` / ``"FK"`` / ``""``（没四肢或分不出来）。

    改了“肢体写入方式”之后用它判断该不该重新自动匹配肢体骨名：映射指向 FK 骨骼却选了 IK，
    写出来的姿态在 IK 模式下是看不见的。
    """
    specs = specs if specs is not None else limb_specs(arm_obj)
    if not specs:
        return ""
    ik = fk = 0
    for spec in specs.values():
        plan = ik_write_plan(spec)
        roles = (plan or {}).get("roles") or []
        controls = _limb_controls(spec)
        fk_bones = set(spec.get("fk_bones") or [])
        for role in roles:
            name = bmap.get(role)
            if not name:
                continue
            if name in controls:
                ik += 1
            elif name in fk_bones:
                fk += 1
    if ik > fk:
        return MODE_IK
    if fk > ik:
        return MODE_FK
    return ""


def collection_name(spec: dict, mode: str) -> str:
    """Rigify 给每组控制器建的骨骼集合名字：``Arm.L (IK)`` / ``Leg.R (FK)``。"""
    return "%s.%s (%s)" % (spec.get("kind", "?"), spec.get("side", "?"), mode)


def limb_modes(arm_obj, specs: dict = None) -> dict:
    """``{prop_bone: IK_FK}``（IK_FK 是 0/1 的浮点自定义属性，0=IK，1=FK）。"""
    specs = specs if specs is not None else limb_specs(arm_obj)
    modes: dict = {}
    for spec in specs.values():
        pb = arm_obj.pose.bones.get(spec["prop_bone"])
        if pb is None or "IK_FK" not in pb.keys():
            continue
        try:
            modes[spec["prop_bone"]] = float(pb["IK_FK"])
        except (TypeError, ValueError):
            continue
    return modes


def current_mode(arm_obj, spec: dict) -> str | None:
    """单条肢体当前是 IK 还是 FK（读不到属性时返回 None）。"""
    pb = arm_obj.pose.bones.get(spec["prop_bone"])
    if pb is None or "IK_FK" not in pb.keys():
        return None
    try:
        return MODE_FK if float(pb["IK_FK"]) >= 0.5 else MODE_IK
    except (TypeError, ValueError):
        return None


def set_mode(arm_obj, mode: str, report: dict = None, specs: dict = None,
             sync: bool = True) -> list[str]:
    """把（指定的）肢体切到 IK / FK，并同步控制器可见性。返回被改过的 prop bone。"""
    specs = dict(specs) if specs is not None else limb_specs(arm_obj)
    if not specs:
        return []
    value = 1.0 if mode == MODE_FK else 0.0
    changed: list[str] = []
    for spec in specs.values():
        pb = arm_obj.pose.bones.get(spec["prop_bone"])
        if pb is None or "IK_FK" not in pb.keys():
            continue
        try:
            if abs(float(pb["IK_FK"]) - value) < 1e-6:
                continue
        except (TypeError, ValueError):
            pass
        pb["IK_FK"] = value
        changed.append(spec["prop_bone"])
    if sync:
        touched = sync_visibility(arm_obj, mode, specs)
        if report is not None and touched:
            report["visibility"] = touched
    return changed


def sync_visibility(arm_obj, mode: str = None, specs: dict = None) -> list[str]:
    """按 IK/FK 状态显示“能用”的那组控制器，藏掉另一组。

    ``mode=None`` 时逐条读 ``IK_FK`` 当前值（用户自己拖过 Rigify 的滑块之后，
    点我们的“同步控制器显示”用的就是这个）。
    """
    specs = specs if specs is not None else limb_specs(arm_obj)
    if not specs:
        return []
    collections = getattr(arm_obj.data, "collections", None)
    if collections is None:
        return []
    touched: list[str] = []
    for spec in specs.values():
        limb_mode = mode or current_mode(arm_obj, spec)
        if limb_mode not in MODES:
            continue
        for tag in MODES:
            coll = collections.get(collection_name(spec, tag))
            if coll is None:
                continue
            visible = (tag == limb_mode)
            if bool(coll.is_visible) != visible:
                coll.is_visible = visible
                touched.append(coll.name)
    return touched


# --------------------------------------------------------------------------- 快照 FK -> IK
def _ensure_pose_context(arm_obj):
    """``bpy.ops.pose.*`` 需要“激活对象 = 骨架 + Pose 模式”。返回 (原激活对象, 原模式)。"""
    view_layer = bpy.context.view_layer
    previous = None
    previous_mode = arm_obj.mode
    try:
        previous = view_layer.objects.active if view_layer is not None else None
        if previous is not arm_obj:
            for obj in list(bpy.context.selected_objects):
                obj.select_set(False)
            arm_obj.select_set(True)
            view_layer.objects.active = arm_obj
        if arm_obj.mode != "POSE":
            bpy.ops.object.mode_set(mode="POSE")
    except Exception as exc:  # noqa: BLE001
        utils.warn("切到 Pose 模式失败（IK 快照需要）：%s" % exc)
    return previous, previous_mode


def snap_fk_to_ik(arm_obj, spec: dict, report: dict = None) -> bool:
    """调用 Rigify 自带的 “IK->FK” 快照：把 FK 链的姿态搬到 IK 控制器上。

    它是 Rigify 面板上那个按钮的同一个操作符，pole / 拉伸 / 脚跟脚掌的数学都由
    Rigify 自己算，所以我们不用手解 IK。
    """
    rid = rig_id(arm_obj)
    op_name = "rigify_limb_ik2fk_%s" % rid
    op = getattr(bpy.ops.pose, op_name, None)
    if op is None:
        if report is not None:
            report.setdefault("warnings", []).append(
                "找不到操作符 %s（%s 不是 Rigify 生成的骨架？）" % (op_name, arm_obj.name))
        return False
    kwargs = {
        "prop_bone": spec["prop_bone"],
        "pole_prop": spec.get("pole_prop") or "pole_vector",
        "fk_bones": json.dumps(spec["fk_bones"]),
        "ik_bones": json.dumps(spec["ik_bones"]),
        "ctrl_bones": json.dumps(spec["ctrl_bones"]),
        "tail_bones": json.dumps(spec.get("tail_bones") or []),
        "extra_ctrls": json.dumps(spec.get("extra_ctrls") or []),
    }
    try:
        result = op(**kwargs)
    except Exception as exc:  # noqa: BLE001
        if report is not None:
            report.setdefault("warnings", []).append(
                "IK 快照失败（%s）：%s" % (spec["fk_bones"][0], exc))
        return False
    if "FINISHED" not in result and report is not None:
        report.setdefault("warnings", []).append(
            "IK 快照未完成（%s）：%s" % (spec["fk_bones"][0], result))
    return "FINISHED" in result


def snap_limbs(arm_obj, specs: dict, report: dict = None) -> list[str]:
    """对多条肢体做 FK->IK 快照，返回成功搬过去的 FK 链首骨名。"""
    if not specs:
        return []
    previous, previous_mode = _ensure_pose_context(arm_obj)
    done: list[str] = []
    try:
        for spec in specs.values():
            if snap_fk_to_ik(arm_obj, spec, report):
                done.append(spec["fk_bones"][0])
            view_layer = bpy.context.view_layer
            if view_layer is not None:
                view_layer.update()
    finally:
        if previous_mode != "POSE":
            try:
                bpy.ops.object.mode_set(mode=previous_mode)
            except Exception:  # noqa: BLE001
                pass
        if previous is not None and previous is not arm_obj:
            try:
                arm_obj.select_set(False)
                previous.select_set(True)
                bpy.context.view_layer.objects.active = previous
            except Exception:  # noqa: BLE001
                pass
    return done


# --------------------------------------------------------------------------- 自动同步（尽力）
_WATCH_MEMO: dict[str, dict] = {}
_WATCH_BUSY = [False]
_CANDIDATES: dict = {"key": -1, "list": []}


def _scan_tracked_armatures() -> list:
    out = []
    for obj in bpy.data.objects:
        if obj.type != "ARMATURE":
            continue
        try:
            if not obj.data.get("rig_id"):
                continue
            if not any("IK_FK" in bone.keys() for bone in obj.pose.bones
                       if bone.name.endswith(("_parent.L", "_parent.R"))):
                continue
        except Exception:  # noqa: BLE001
            continue
        out.append(obj)
    return out


def _tracked_armatures() -> list:
    """带缓存：handler 每帧都会被调，别每次都扫全场景对象表。"""
    count = len(bpy.data.objects)
    if _CANDIDATES["key"] != count:
        _CANDIDATES["key"] = count
        _CANDIDATES["list"] = _scan_tracked_armatures()
    alive = []
    for obj in _CANDIDATES["list"]:
        try:
            _ = obj.name          # 对象被删掉后访问会抛 ReferenceError
        except Exception:  # noqa: BLE001
            continue
        alive.append(obj)
    return alive


def _on_depsgraph_update(_scene, _depsgraph) -> None:
    """用户自己拖 Rigify 的 IK-FK 滑块后，尽量把控制器显示同步过来。

    说明：``IK_FK`` 是 PoseBone 上的自定义属性，改它**不会**触发 depsgraph 回调，
    所以这里只能在别的更新发生时（旋转视角 / 选择骨骼 / 改其它属性…）顺带比一次；
    N 面板里另外留了“同步控制器显示”按钮兜底。
    """
    if _WATCH_BUSY[0]:
        return
    _WATCH_BUSY[0] = True
    try:
        for arm_obj in _tracked_armatures():
            specs = limb_specs(arm_obj)
            if not specs:
                continue
            modes = limb_modes(arm_obj, specs)
            if _WATCH_MEMO.get(arm_obj.name) == modes:
                continue
            _WATCH_MEMO[arm_obj.name] = modes
            sync_visibility(arm_obj, None, specs)
    except Exception:  # noqa: BLE001  回调里绝不能让异常冒出去
        pass
    finally:
        _WATCH_BUSY[0] = False


def watcher_installed() -> bool:
    return _on_depsgraph_update in bpy.app.handlers.depsgraph_update_post


def ensure_watcher() -> None:
    """装上自动同步回调（插件注册时、以及载入新文件后（会清空 handler）再装一次）。"""
    if not watcher_installed():
        bpy.app.handlers.depsgraph_update_post.append(_on_depsgraph_update)


def remove_watcher() -> None:
    if watcher_installed():
        bpy.app.handlers.depsgraph_update_post.remove(_on_depsgraph_update)
    _WATCH_MEMO.clear()

