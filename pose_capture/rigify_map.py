"""Rigify 骨骼映射：把“语义角色”对应到具体骨架骨名，并读取静止姿态尺寸。

同时兼容两类骨架：
  * Rigify 生成的骨架：spine_fk / upper_arm_fk.L / thigh_fk.L / neck / head / torso ...
  * Rigify 元骨架(metarig) 或普通骨架：spine / spine.001 / upper_arm.L / thigh.L ...
"""
from __future__ import annotations

import math

from mathutils import Matrix, Vector

# 每个角色的候选骨名（{s} 会被替换成 L / R，按优先级从上到下匹配）
ROLE_DEFS = [
    ("root", "根骨骼 / 整体位移", ["root", "torso"]),
    ("hips", "骨盆 / 躯干朝向（需 Rigify 生成骨架）", ["torso"]),
    ("neck", "脖子", ["neck", "neck_fk", "spine.005", "ORG-spine.005", "DEF-spine.005"]),
    ("head", "头部", ["head", "head_fk", "spine.006", "ORG-spine.006", "DEF-spine.006"]),
    ("shoulder.{s}", "锁骨 {s}", ["shoulder.{s}", "shoulder_fk.{s}", "ORG-shoulder.{s}",
                                  "DEF-shoulder.{s}"]),
    ("upper_arm.{s}", "大臂 {s}", ["upper_arm_fk.{s}", "upper_arm.{s}", "upper_arm_ik.{s}",
                                   "ORG-upper_arm.{s}", "DEF-upper_arm.{s}"]),
    ("forearm.{s}", "小臂 {s}", ["forearm_fk.{s}", "forearm.{s}", "forearm_ik.{s}",
                                 "ORG-forearm.{s}", "DEF-forearm.{s}"]),
    ("hand.{s}", "手掌 {s}", ["hand_fk.{s}", "hand.{s}", "hand_ik.{s}",
                              "ORG-hand.{s}", "DEF-hand.{s}"]),
    ("palm.{s}", "掌心 {s}(可选)", ["palm.{s}", "palm.02.{s}", "palm.01.{s}", "DEF-palm.01.{s}"]),
    ("thigh.{s}", "大腿 {s}", ["thigh_fk.{s}", "thigh.{s}", "thigh_ik.{s}",
                               "ORG-thigh.{s}", "DEF-thigh.{s}"]),
    ("shin.{s}", "小腿 {s}", ["shin_fk.{s}", "shin.{s}", "shin_ik.{s}",
                              "ORG-shin.{s}", "DEF-shin.{s}"]),
    ("foot.{s}", "脚掌 {s}", ["foot_fk.{s}", "foot.{s}", "foot_ik.{s}",
                              "ORG-foot.{s}", "DEF-foot.{s}"]),
    ("toe.{s}", "脚趾 {s}", ["toe_fk.{s}", "toe.{s}", "toe_ik.{s}",
                             "ORG-toe.{s}", "DEF-toe.{s}"]),
    ("thumb.01.{s}", "拇指 1 {s}", ["thumb.01.{s}", "DEF-thumb.01.{s}"]),
    ("thumb.02.{s}", "拇指 2 {s}", ["thumb.02.{s}", "DEF-thumb.02.{s}"]),
    ("thumb.03.{s}", "拇指 3 {s}", ["thumb.03.{s}", "DEF-thumb.03.{s}"]),
    ("f_index.01.{s}", "食指 1 {s}", ["f_index.01.{s}", "DEF-f_index.01.{s}"]),
    ("f_index.02.{s}", "食指 2 {s}", ["f_index.02.{s}", "DEF-f_index.02.{s}"]),
    ("f_index.03.{s}", "食指 3 {s}", ["f_index.03.{s}", "DEF-f_index.03.{s}"]),
    ("f_middle.01.{s}", "中指 1 {s}", ["f_middle.01.{s}", "DEF-f_middle.01.{s}"]),
    ("f_middle.02.{s}", "中指 2 {s}", ["f_middle.02.{s}", "DEF-f_middle.02.{s}"]),
    ("f_middle.03.{s}", "中指 3 {s}", ["f_middle.03.{s}", "DEF-f_middle.03.{s}"]),
    ("f_ring.01.{s}", "无名指 1 {s}", ["f_ring.01.{s}", "DEF-f_ring.01.{s}"]),
    ("f_ring.02.{s}", "无名指 2 {s}", ["f_ring.02.{s}", "DEF-f_ring.02.{s}"]),
    ("f_ring.03.{s}", "无名指 3 {s}", ["f_ring.03.{s}", "DEF-f_ring.03.{s}"]),
    ("f_pinky.01.{s}", "小指 1 {s}", ["f_pinky.01.{s}", "DEF-f_pinky.01.{s}"]),
    ("f_pinky.02.{s}", "小指 2 {s}", ["f_pinky.02.{s}", "DEF-f_pinky.02.{s}"]),
    ("f_pinky.03.{s}", "小指 3 {s}", ["f_pinky.03.{s}", "DEF-f_pinky.03.{s}"]),
]

SPINE_SLOTS = 6
SPINE_MAX_BONES = 5  # 脊椎最多参与 5 段
SPINE_PATTERNS = ["spine_fk", "spine", "ORG-spine", "DEF-spine"]

FINGERS = ("thumb", "f_index", "f_middle", "f_ring", "f_pinky")

# “肢体写入方式 = IK” 时优先匹配的 Rigify IK 控制器（只在骨骼确实存在时才用得上，
# 拿不到就退回 ROLE_DEFS 里的常规候选，所以 metarig / 普通骨架不受影响）。
# 参考 `rigify_limbs._DEFAULT_CHAINS`：链根与末端控制器是 upper_arm_ik.L / hand_ik.L /
# thigh_ik.L / foot_ik.L / toe_ik.L，中间段（forearm_ik / shin_ik）由 IK 解算决定。
IK_ROLE_CANDIDATES = {
    "upper_arm.{s}": ["upper_arm_ik.{s}"],
    "forearm.{s}": ["forearm_ik.{s}"],
    "hand.{s}": ["hand_ik.{s}"],
    "thigh.{s}": ["thigh_ik.{s}"],
    "shin.{s}": ["shin_ik.{s}"],
    "foot.{s}": ["foot_ik.{s}"],
    "toe.{s}": ["toe_ik.{s}"],
}


def role_label(role: str) -> str:
    for rid, label, _cands in ROLE_DEFS:
        if rid == role:
            return label
    if role.startswith("spine_"):
        return "脊椎 第%d段" % (int(role.split("_")[1]) + 1)
    return role


def all_role_ids(include_fingers: bool = True) -> list[str]:
    roles = []
    for role, _label, _cands in ROLE_DEFS:
        if "{s}" in role:
            if not include_fingers and any(role.startswith(f) for f in FINGERS):
                continue
            roles.extend([role.format(s="L"), role.format(s="R")])
        else:
            roles.append(role)
    roles.extend(["spine_%02d" % i for i in range(SPINE_SLOTS)])
    return roles


class BoneMap:
    """角色 -> 实际骨名 的映射结果。"""

    def __init__(self):
        self.roles: dict[str, str] = {}
        self.missing: list[str] = []

    def get(self, role: str) -> str:
        return self.roles.get(role, "")

    def set(self, role: str, name: str) -> None:
        if name:
            self.roles[role] = name
        else:
            self.roles.pop(role, None)

    def spine_bones(self) -> list[str]:
        names = []
        for i in range(SPINE_SLOTS):
            name = self.roles.get("spine_%02d" % i, "")
            if name:
                names.append(name)
        return names

    def used_bones(self) -> list[str]:
        return list(self.roles.values())


# --------------------------------------------------------------------------- 自动识别
def _collect_indexed(names: list[str], pattern: str) -> list[str]:
    """收集 pattern 与 pattern.NN 形式的骨骼，按序号排序。"""
    import re

    rx = re.compile(r"^%s(?:\.(\d+))?$" % re.escape(pattern))
    found = []
    for name in names:
        match = rx.match(name)
        if match:
            found.append((int(match.group(1) or 0), name))
    return [name for _index, name in sorted(found)]


def _pick(name_set, candidates: list[str]) -> str:
    for cand in candidates:
        if cand in name_set:
            return cand
    return ""


def ordered_candidates(role_id: str, side: str = "", limb_mode: str = None) -> list[str]:
    """角色的候选骨名（``{s}`` 已替换成 side）。

    ``limb_mode="IK"`` 时把 IK 控制器（``upper_arm_ik.L`` / ``hand_ik.L`` …）提到最前：
    Rigify 生成骨架的胳膊 / 大腿默认挂在 IK 上，映射到 FK 骨骼的话写出来的姿态看不见。
    FK 模式或没指定 ``limb_mode`` 时保持原来的 FK 优先顺序（兼容 metarig / 普通骨架）。
    """
    for rid, _label, cands in ROLE_DEFS:
        if rid != role_id:
            continue
        ordered: list[str] = []
        if str(limb_mode or "").upper() == "IK":
            ordered.extend(name.format(s=side) for name in IK_ROLE_CANDIDATES.get(rid, ()))
        for name in cands:
            resolved = name.format(s=side)
            if resolved not in ordered:
                ordered.append(resolved)
        return ordered
    return []


def auto_detect(arm_obj, include_fingers: bool = True, limb_mode: str = None) -> BoneMap:
    """在骨架上自动匹配角色骨名。

    ``limb_mode`` 跟设置里的“肢体写入方式”一致：``"IK"`` 时肢体优先匹配
    ``*_ik``（IK 控制器），``"FK"`` 或不指定时优先 ``*_fk``。
    """
    bones = arm_obj.data.bones
    names = [bone.name for bone in bones]
    name_set = set(names)
    bmap = BoneMap()

    for role, _label, cands in ROLE_DEFS:
        if "{s}" in role:
            if not include_fingers and any(role.startswith(f) for f in FINGERS):
                continue
            for side in ("L", "R"):
                rid = role.format(s=side)
                resolved = _pick(name_set, ordered_candidates(role, side, limb_mode))
                if resolved:
                    bmap.set(rid, resolved)
                else:
                    bmap.missing.append(rid)
        else:
            resolved = _pick(name_set, list(cands))
            if resolved:
                bmap.set(role, resolved)
            else:
                bmap.missing.append(role)

    # 脊椎链：优先取匹配数量最多的命名风格，并按“肩膀高度”截断
    shoulder_z = None
    for side in ("L", "R"):
        name = bmap.get("upper_arm.%s" % side)
        if name:
            z = bones[name].head_local.z
            shoulder_z = z if shoulder_z is None else max(shoulder_z, z)

    best_chain: list[str] = []
    for pattern in SPINE_PATTERNS:  # 按优先级取第一个命中的命名风格
        chain = _collect_indexed(names, pattern)
        if chain:
            best_chain = chain
            break

    if shoulder_z is not None and best_chain:
        limited = [name for name in best_chain
                   if bones[name].head_local.z < shoulder_z + 1e-4]
        if limited:
            best_chain = limited
    best_chain = best_chain[:SPINE_MAX_BONES]

    for index, name in enumerate(best_chain):
        bmap.set("spine_%02d" % index, name)
    for index in range(len(best_chain), SPINE_SLOTS):
        bmap.missing.append("spine_%02d" % index)

    return bmap


def read_metrics(arm_obj, bmap: BoneMap) -> dict:
    """读取静止姿态下的长度 / 方向 / 参照轴，用于 2D 姿态解算与重定向。"""
    bones = arm_obj.data.bones
    metrics: dict = {"errors": [], "warnings": []}

    def bone(role: str):
        name = bmap.get(role)
        return bones.get(name) if name else None

    spine_names = bmap.spine_bones()
    if not spine_names:
        metrics["errors"].append("没有找到脊椎骨（spine / spine_fk 链）")
        return metrics

    spine_bones = [bones[name] for name in spine_names]
    metrics["spine_names"] = spine_names
    metrics["pelvis"] = Vector(spine_bones[0].head_local)
    metrics["chest"] = Vector(spine_bones[-1].tail_local)
    metrics["spine_len"] = float(sum(item.length for item in spine_bones))

    shoulder_heads = [bone("upper_arm.%s" % side) for side in ("L", "R")]
    hip_heads = [bone("thigh.%s" % side) for side in ("L", "R")]
    if any(item is None for item in shoulder_heads):
        metrics["errors"].append("没有找到大臂骨（upper_arm）")
    if any(item is None for item in hip_heads):
        metrics["errors"].append("没有找到大腿骨（thigh）")
    if metrics["errors"]:
        return metrics

    metrics["shoulder_joints"] = {side: Vector(item.head_local)
                                  for side, item in zip(("L", "R"), shoulder_heads)}
    metrics["hip_joints"] = {side: Vector(item.head_local)
                             for side, item in zip(("L", "R"), hip_heads)}
    metrics["shoulder_line"] = (metrics["shoulder_joints"]["L"]
                                - metrics["shoulder_joints"]["R"])
    metrics["hip_line"] = metrics["hip_joints"]["L"] - metrics["hip_joints"]["R"]

    # Rigify 的脊椎根节点(hips)略低于大腿根，胸腔顶端略高于肩关节：
    # 这里记录这两个偏移，重建时按同样比例挪一下，锁骨/脊椎方向才自然。
    hip_avg_z = (metrics["hip_joints"]["L"].z + metrics["hip_joints"]["R"].z) * 0.5
    shoulder_avg_z = (metrics["shoulder_joints"]["L"].z
                      + metrics["shoulder_joints"]["R"].z) * 0.5
    metrics["pelvis_offset_z"] = float(metrics["pelvis"].z - hip_avg_z)
    metrics["chest_offset_z"] = float(metrics["chest"].z - shoulder_avg_z)
    metrics["shoulder_mid"] = Vector(((metrics["shoulder_joints"]["L"]
                                       + metrics["shoulder_joints"]["R"]) * 0.5))
    metrics["hip_mid"] = Vector(((metrics["hip_joints"]["L"]
                                  + metrics["hip_joints"]["R"]) * 0.5))
    # 关键点里的“胯”对应骨架的大腿根，所以量取“大腿根中点 -> 肩关节中点”
    metrics["pelvis_to_shoulders"] = float((metrics["shoulder_mid"]
                                            - metrics["hip_mid"]).length)

    # 躯干上的刚性偏移（重建时会随躯干一起旋转）
    chest = metrics["chest"]
    metrics["hip_offsets"] = {side: Vector(metrics["hip_joints"][side] - metrics["pelvis"])
                              for side in ("L", "R")}
    metrics["shoulder_offsets"] = {side: Vector(metrics["shoulder_joints"][side] - chest)
                                   for side in ("L", "R")}
    clavicle_offsets = {}
    for side in ("L", "R"):
        item = bone("shoulder.%s" % side)
        clavicle_offsets[side] = Vector(item.head_local - chest) if item is not None else None
    metrics["clavicle_offsets"] = clavicle_offsets
    neck_bone = bone("neck")
    metrics["neck_offset"] = (Vector(neck_bone.head_local - chest) if neck_bone is not None
                              else Vector((0.0, 0.0, 0.0)))
    head_bone = bone("head")
    metrics["head_offset"] = (Vector(head_bone.head_local - chest) if head_bone is not None
                              else Vector((0.0, 0.0, 0.0)))

    def chain_len(*roles) -> float:
        return float(sum(bone(role).length for role in roles if bone(role) is not None))

    metrics["upper_arm_len"] = {s: chain_len("upper_arm.%s" % s) for s in ("L", "R")}
    metrics["forearm_len"] = {s: chain_len("forearm.%s" % s) for s in ("L", "R")}
    metrics["hand_len"] = {s: chain_len("hand.%s" % s, "palm.%s" % s) for s in ("L", "R")}
    metrics["thigh_len"] = {s: chain_len("thigh.%s" % s) for s in ("L", "R")}
    metrics["shin_len"] = {s: chain_len("shin.%s" % s) for s in ("L", "R")}
    metrics["foot_len"] = {s: chain_len("foot.%s" % s) for s in ("L", "R")}
    metrics["toe_len"] = {s: chain_len("toe.%s" % s) for s in ("L", "R")}
    metrics["neck_len"] = chain_len("neck")
    metrics["head_len"] = chain_len("head")

    metrics["rest_dir"] = {}
    metrics["rest_mat"] = {}
    for role, name in bmap.roles.items():
        item = bones.get(name)
        if item is None:
            continue
        direction = Vector(item.tail_local) - Vector(item.head_local)
        if direction.length > 1e-9:
            metrics["rest_dir"][role] = direction.normalized()
        metrics["rest_mat"][role] = item.matrix_local.to_3x3()

    metrics["ref"] = {"hip_line": metrics["hip_line"],
                      "shoulder_line": metrics["shoulder_line"]}
    # 静止侧的“眼线”（人物右 -> 左）。它的用处是让 head 的 roll / 转头真的生效：
    # retarget 里 head 的参照轴是 "eye_line"，解算结果那侧现在由五官（眼角 / 嘴角 / 眉梢 /
    # 下颌角成对投票）给出；静止侧以前**没有**这个键 —— solve_orientation 拿到 rest_ref=None
    # 会静默跳过扭转，于是照片里转头 / 歪头的头部信息整体被丢掉（头永远朝着静止方向）。
    # 这里用肩线（没有就用胯线）：head 骨自己的 roll 约定各家骨架都不同，而肩线 / 胯线天然
    # 定义“人物左右”（.L 在人物左侧），静止 A/T pose 下头部本来就与肩线对齐。
    head_lateral = metrics["shoulder_line"]
    if head_lateral.length < 1e-9:
        head_lateral = metrics["hip_line"]
    if head_lateral.length > 1e-9:
        metrics["ref"]["eye_line"] = head_lateral.normalized()
    # 弯曲轴（肘/膝）刻意不做“对齐到肩线/胯线”的符号翻转：
    # 静止与目标用同一个叉乘约定（上段 x 下段，顺序固定）才是有信息的，
    # 用参考线翻符号在“弯曲轴几乎垂直于参考线”时会变成噪声，导致 roll 随机反向。
    for side in ("L", "R"):
        upper, fore = bone("upper_arm.%s" % side), bone("forearm.%s" % side)
        thigh, shin = bone("thigh.%s" % side), bone("shin.%s" % side)
        if upper and fore:
            axis = (Vector(upper.tail_local) - Vector(upper.head_local)).cross(
                Vector(fore.tail_local) - Vector(fore.head_local))
            if axis.length > 1e-6:
                metrics["ref"]["armbend.%s" % side] = axis.normalized()
        if thigh and shin:
            axis = (Vector(thigh.tail_local) - Vector(thigh.head_local)).cross(
                Vector(shin.tail_local) - Vector(shin.head_local))
            if axis.length > 1e-6:
                metrics["ref"]["kneebend.%s" % side] = axis.normalized()

    low_z = None
    for role in ("foot.L", "foot.R", "toe.L", "toe.R"):
        item = bone(role)
        if item is None:
            continue
        for point in (item.head_local, item.tail_local):
            low_z = point.z if low_z is None else min(low_z, point.z)
    metrics["lowest_foot_z"] = 0.0 if low_z is None else float(low_z)
    metrics["pelvis_height"] = float(metrics["pelvis"].z - metrics["lowest_foot_z"])
    return metrics


