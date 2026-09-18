"""把 3D 关节解算结果重定向到 Rigify 骨架的 Pose 骨骼上。

默认只写“旋转”通道（保持骨架比例不变），可选写根骨骼位移 + 关键帧。
每个骨骼的姿态矩阵由两部分决定：
  * 摆动(aim)：静止方向的极小旋转 -> 目标方向；
  * 扭曲(twist)：用参照轴（肩线/胯线/手臂弯曲轴/眼线…）对齐 roll，避免手臂/头部拧麻花。
"""
from __future__ import annotations

import math

import bpy
from mathutils import Matrix, Vector

from . import reconstruction as recon_mod
from . import rigify_limbs, utils

FINGER_CHAIN = {
    "thumb": ("thumb.01", "thumb.02", "thumb.03"),
    "index": ("f_index.01", "f_index.02", "f_index.03"),
    "middle": ("f_middle.01", "f_middle.02", "f_middle.03"),
    "ring": ("f_ring.01", "f_ring.02", "f_ring.03"),
    "pinky": ("f_pinky.01", "f_pinky.02", "f_pinky.03"),
}


def build_targets(recon: dict, bmap, metrics: dict) -> dict:
    """role -> (起点 3D, 终点 3D, 参照轴名字)。坐标为“相机/角色”世界系。"""
    joints = recon["joints"]
    spine_points = recon["spine_points"]
    count = max(len(bmap.spine_bones()), 1)
    pts = recon_mod._resample_polyline(spine_points, count)

    targets = {"hips": (spine_points[0], pts[1], "hip_line")}
    for index in range(count):
        targets["spine_%02d" % index] = (pts[index], pts[index + 1], "shoulder_line")

    targets["neck"] = (joints.get("neck_start", joints["chest"]), joints["head_base"],
                       "shoulder_line")
    targets["head"] = (joints["head_base"], joints["head_tip"], "eye_line")

    for side in ("L", "R"):
        origin = joints.get("shoulder_origin.%s" % side, joints["chest"])
        targets["shoulder.%s" % side] = (origin, joints["shoulder.%s" % side],
                                         "shoulder_line")
        targets["upper_arm.%s" % side] = (joints["shoulder.%s" % side],
                                          joints["elbow.%s" % side], "armbend.%s" % side)
        targets["forearm.%s" % side] = (joints["elbow.%s" % side], joints["wrist.%s" % side],
                                        "armbend.%s" % side)
        # 手掌/手指的 roll 也跟随“手臂弯曲平面”：
        # 手腕前后屈的弯曲轴在手腕伸直时几乎退化（叉乘趋近 0），用它对齐 roll 会抖动几十度；
        # 用肘弯曲轴等价于“腕部不带额外扭转”，与静止姿态下掌心和手臂平面的关系一致。
        targets["hand.%s" % side] = (joints["wrist.%s" % side], joints["hand_tip.%s" % side],
                                     "armbend.%s" % side)
        targets["palm.%s" % side] = targets["hand.%s" % side]
        targets["thigh.%s" % side] = (joints["hip.%s" % side], joints["knee.%s" % side],
                                      "hip_line")
        targets["shin.%s" % side] = (joints["knee.%s" % side], joints["ankle.%s" % side],
                                     "kneebend.%s" % side)
        targets["foot.%s" % side] = (joints["ankle.%s" % side], joints["ball.%s" % side],
                                     "hip_line")
        # 脚趾不做单独写入：单张图无法判断脚趾弯曲，保持与脚掌的静止相对关系更稳妥

    for side, per_hand in recon.get("fingers", {}).items():
        for finger, chain in per_hand.items():
            names = FINGER_CHAIN.get(finger)
            if not names or len(chain) < 4:
                continue
            for index, base in enumerate(names):
                targets["%s.%s" % (base, side)] = (chain[index], chain[index + 1],
                                                   "armbend.%s" % side)
    return targets


def solve_orientation(rest_mat3: Matrix, target_dir: Vector, ref: Vector = None,
                      rest_ref: Vector = None, use_twist: bool = True,
                      twist_blend: float = 1.0) -> Matrix:
    """返回骨骼在 armature 空间下的目标姿态旋转矩阵（3x3）。

    摆动(aim)：静止方向 -> 目标方向的极小旋转。
    扭转(twist)：把静止参照轴按摆动搬运后，绕骨骼轴转到目标参照轴。
      * 必须搬运“参照轴本身”，不能拿离它最近的矩阵列代替：Rigify 的肘弯曲轴
        与任何一列都差几十度，用列近似会让 roll 整体偏掉。
      * 必须按“有符号”对齐：肢体前后反折时（例如手肘朝前而不是朝后），
        roll 需要转过接近 180 度，取最近的一侧会把弯曲平面镜像过来。
    """
    rest_dir = rest_mat3.col[1].normalized()
    aim = rest_dir.rotation_difference(target_dir.normalized())
    matrix = aim.to_matrix() @ rest_mat3

    if not use_twist or ref is None or rest_ref is None or twist_blend <= 1e-6:
        return matrix
    if ref.length < 1e-6 or rest_ref.length < 1e-6:
        return matrix

    axis = target_dir.normalized()
    local_ref = rest_mat3.inverted() @ rest_ref.normalized()
    carried = matrix @ local_ref
    carried = carried - axis * carried.dot(axis)
    if carried.length < 1e-6:
        return matrix
    carried.normalize()

    desired = ref - axis * ref.dot(axis)
    if desired.length < 1e-6:
        return matrix
    desired.normalize()

    angle = math.atan2(carried.cross(desired).dot(axis), carried.dot(desired))
    return Matrix.Rotation(angle * twist_blend, 3, axis) @ matrix


def solve_role_matrix(role: str, target, refs: dict, rest_refs: dict, rest_mat3: Matrix,
                      to_arm, use_twist: bool = True, twist_blend: float = 1.0):
    """一个角色的目标姿态矩阵（armature 空间 3x3）；方向退化时返回 None。

    ``target`` 是 ``(起点, 终点, 参照轴名字)``；``refs`` / ``rest_refs`` 分别是“目标系”和
    “静止系”的参照轴字典。IK 直写与 FK 写入共用同一套数学，两边结果才对得上。
    """
    p_from, p_to, ref_name = target[0], target[1], target[2]
    direction = Vector(p_to) - Vector(p_from)
    if direction.length < 1e-6:
        return None
    ref_arm = rest_ref = None
    if ref_name:
        ref_vec = refs.get(ref_name)
        # 注意：解算结果的参照轴是 numpy 数组，没有 .length 属性，
        # 必须先用 Vector() 包一层，否则这里会判定为“无效参照轴”而静默跳过扭转。
        try:
            ref_vec = Vector(ref_vec) if ref_vec is not None else None
        except (TypeError, ValueError):
            ref_vec = None
        if ref_vec is not None and ref_vec.length > 1e-9:
            ref_arm = to_arm(ref_vec)
            rest_ref = rest_refs.get(ref_name)
    return solve_orientation(rest_mat3, to_arm(direction), ref_arm, rest_ref,
                             use_twist, twist_blend)


# IK 直写结果的核对阈值：超过就退回“切 FK + Rigify 快照”的老路径。
IK_DIRECT_POS_TOL = 0.02     # 米
IK_DIRECT_ANGLE_TOL = 10.0   # 度


# --------------------------------------------------------------------------- 应用姿态
def limb_mode_option(options: dict) -> str:
    """这次写完后要停在 IK（Rigify 默认控制器）还是 FK。

    兼容旧键 ``fk_mode``：历史上 True 表示“自动切到 FK 模式”。
    """
    mode = str(options.get("limb_mode") or "").upper()
    if mode in rigify_limbs.MODES:
        return mode
    return rigify_limbs.MODE_FK if options.get("fk_mode") else rigify_limbs.MODE_IK


def _limb_bone_names(spec: dict) -> set:
    """一条肢体可能被我们写到的所有骨名（FK 链 / IK 链 / 控制器）。"""
    names = set(spec.get("fk_bones") or [])
    for key in ("ik_bones", "ctrl_bones", "tail_bones", "extra_ctrls"):
        names |= set(spec.get(key) or [])
    return names


def _limbs_written(written_names, specs: dict) -> dict:
    """挑出这次真正写到的 Rigify 肢体（该肢体的 FK 链 / IK 控制器里有骨名被写入）。

    只对“写过的肢体”做 IK 快照 / 直写：没写过的肢体不该被我们改姿态，
    否则会把它当前的 IK 摆位覆盖成 FK 静止姿态。
    """
    out = {}
    written = set(written_names)
    for key, spec in (specs or {}).items():
        if _limb_bone_names(spec) & written:
            out[key] = spec
    return out


def _limb_keyframe_bones(limbs: dict, mode: str) -> list[str]:
    """IK 模式下要一起打帧的 IK 控制器（可见的那几个，不含 MCH-）。"""
    if mode != rigify_limbs.MODE_IK:
        return []
    out: list[str] = []
    for spec in (limbs or {}).values():
        for name in spec["ctrl_bones"] + spec["tail_bones"] + spec["extra_ctrls"]:
            if name not in out:
                out.append(name)
    return out


def close_limbs(arm_obj, limbs: dict, mode: str, report: dict, options: dict,
                specs: dict = None, direct: dict = None) -> None:
    """收尾：IK 模式下把（没能直写的）IK 控制器对齐到 FK 姿态，再按模式同步控制器显示。

    * ``direct``：已经**直接**把姿态写进 IK 控制器的肢体 —— 它们不切 FK、不做快照；
    * 其余肢体在 IK 模式仍走老路径：切 FK -> Rigify 的 “IK->FK” 快照 -> 回到 IK；
    * FK 模式不用快照 —— 姿态本来就停在 FK 骨骼上。
    """
    if not limbs:
        # 只有“是 Rigify 生成骨架但连一条 IK 链都找不到”才值得提示
        if mode == rigify_limbs.MODE_IK and not (specs or {}):
            if rigify_limbs.rig_id(arm_obj):
                report["warnings"].append("骨架里找不到 Rigify 的 IK 控制器链，按普通 FK 写入")
        return

    report["limb_mode"] = mode
    if mode == rigify_limbs.MODE_IK:
        snap = {key: spec for key, spec in limbs.items() if key not in (direct or {})}
        if snap:
            # Rigify 的快照要求“FK 链上的姿态就是当前姿态”，所以先切 FK、快照完再切回来。
            rigify_limbs.set_mode(arm_obj, rigify_limbs.MODE_FK, report, snap, sync=False)
            view_layer = bpy.context.view_layer
            if view_layer is not None:
                view_layer.update()
            snapped = rigify_limbs.snap_limbs(arm_obj, snap, report)
            report["ik_snapped"] = snapped
            missing = [spec["fk_bones"][0] for key, spec in snap.items()
                       if key not in snapped]
            if missing:
                report["warnings"].append("这些肢体没做上 IK 快照：%s" % ", ".join(missing))
        rigify_limbs.set_mode(arm_obj, rigify_limbs.MODE_IK, report, limbs, sync=False)
    else:
        report["fk_switched"] = sorted(spec["prop_bone"] for spec in limbs.values())

    if bool(options.get("sync_visibility", True)):
        touched = rigify_limbs.sync_visibility(arm_obj)
        if touched:
            report["visibility"] = touched


# --------------------------------------------------------------------------- IK 直写
def direct_limb_plans(bmap, limbs: dict) -> dict:
    """挑出这次“可以直写 IK 控制器”的肢体：``key -> {"spec", "plan"}``。

    两个条件：
      1. 用户的映射确实指向这条肢体的 IK 控制器（IK 模式下的自动匹配就是这么映射的；
         手动把肢体映射成 FK 骨骼的话我们也不越权改别的骨骼）；
      2. Rigify 的链信息里认得出一根可写的控制器（链根 / 末端）和 IK 目标骨。
    """
    out: dict = {}
    for key, spec in (limbs or {}).items():
        plan = rigify_limbs.ik_write_plan(spec)
        if not plan:
            continue
        names = set(plan["controls"].values()) | {plan["target"]}
        if not any(bmap.get(role) in names for role in plan["roles"]):
            continue
        out[key] = {"spec": spec, "plan": plan}
    return out


def _update_view() -> None:
    view_layer = bpy.context.view_layer
    if view_layer is not None:
        view_layer.update()


def _write_limb_fk_chain(arm_obj, plan: dict, targets: dict, refs: dict, metrics: dict,
                         to_arm, use_twist: bool, twist_blend: float) -> None:
    """把一条肢体的 FK 链按 FK 模式那样写一份（静止矩阵用 FK 骨自己的）。

    IK 模式下这条链是隐藏的，但 (a) 用户拖 IK_FK 滑块回 FK 时姿态不丢，(b) 退回 Rigify
    快照时它就是快照读的“当前姿态”，(c) 核对直写结果时它就是期待值。必须按链的顺序写。
    """
    rest_refs = metrics.get("ref") or {}
    for role in plan["roles"]:
        name = plan["fk_of_role"].get(role) or ""
        bone = arm_obj.data.bones.get(name)
        pbone = arm_obj.pose.bones.get(name)
        target = targets.get(role)
        if bone is None or pbone is None or target is None:
            continue
        mat3 = solve_role_matrix(role, target, refs, rest_refs,
                                 bone.matrix_local.to_3x3(), to_arm, use_twist, twist_blend)
        if mat3 is None:
            continue
        pbone.matrix = Matrix.Translation(pbone.head.copy()) @ mat3.to_4x4()
        _update_view()


def _fk_pose_delta(arm_obj, fk_name: str):
    """FK 骨在 armature 空间的“世界增量”矩阵（静止 -> 当前）；缺骨返回 None。

    Rigify 自己的 “IK->FK” 快照就是把 FK 链的这套增量搬到 IK 控制器上的。用增量而不是绝对
    坐标，Rigify 内部怎么搭的（MCH 骨、COPY_LOCATION 的把手骨）就自动带过去了，根骨位移 /
    父级姿态也已经包含在 FK 链的姿态里。
    """
    bone = arm_obj.data.bones.get(fk_name)
    pbone = arm_obj.pose.bones.get(fk_name)
    if bone is None or pbone is None:
        return None
    return pbone.matrix @ bone.matrix_local.inverted()


def _limb_check_score(arm_obj, spec: dict, roles) -> float:
    """核对偏差折算成一个可比较的分数（越小越好）；核对不了按无穷大。

    关节位置权重高一些：姿态看得见的误差是关节，链条骨骼自己的 roll 差个几十度是正常的
    （Rigify 自己的快照也有，实测 ``DEF-upper_arm`` 差 64.7°），别让 roll 把搜索带偏。
    """
    check = _ik_direct_check(arm_obj, spec, roles)
    if check is None:
        return float("inf")
    return (4.0 * check[0] / IK_DIRECT_POS_TOL
            + math.radians(check[1]) / math.radians(IK_DIRECT_ANGLE_TOL))


def _search_root_roll(arm_obj, spec: dict, plan: dict, steps: int = 16, rounds: int = 5) -> bool:
    """搜链根控制器的“扭转”（绕自身骨轴转），让核对偏差最小 —— 抄 Rigify ``correct_rotation`` 的思路。

    IK 链的弯曲平面由链根的扭转决定（膝盖 / 手肘朝哪边），Rigify 的快照也会这么搜一遍。
    只有一次直写没通过核对时才调用，所以通过的那些肢体不用付这个代价。
    返回是否真的搜出了更小的偏差。
    """
    roles = plan.get("roles") or []
    root_name = plan["controls"].get(roles[0]) if roles else None
    pbone = arm_obj.pose.bones.get(root_name or "")
    if pbone is None:
        return False
    matrix = pbone.matrix.copy()
    base, translation = matrix.to_3x3(), matrix.translation.copy()

    def apply(angle: float) -> None:
        rolled = base @ Matrix.Rotation(angle, 3, "Y")
        pbone.matrix = Matrix.Translation(translation) @ rolled.to_4x4()
        _update_view()

    start = _limb_check_score(arm_obj, spec, roles)
    best_angle, best_score = 0.0, start
    step = 2.0 * math.pi / steps
    for index in range(1, steps + 1):
        angle = -math.pi + step * index
        apply(angle)
        score = _limb_check_score(arm_obj, spec, roles)
        if score < best_score:
            best_angle, best_score = angle, score
    for _ in range(rounds):                  # 粗搜到方向后再逐步缩小步长细化
        step /= 2.0
        for delta in (-step, step):
            apply(best_angle + delta)
            score = _limb_check_score(arm_obj, spec, roles)
            if score < best_score:
                best_angle, best_score = best_angle + delta, score
    apply(best_angle)
    return best_score < start


def _write_limb_ik_controls(arm_obj, spec: dict, plan: dict) -> int:
    """IK 控制器：按 Rigify “IK->FK” 快照（``RigifyLimbIk2FkBase.apply_frame_state``）的算法直写。

    跟 Rigify 一致的三件事：

      * 末端控制器 ``hand_ik.L`` / ``foot_ik.L`` **就是这条 IK 链的末端**（Rigify 用一个带
        ``COPY_LOCATION`` 的 MCH 骨把它接到 IK 链上），所以它要拿 FK 手 / 脚的位置 **和** 朝向 ——
        只写旋转的话手腕 / 脚踝根本不会动，链上的关节也就全错；
      * 链根 ``upper_arm_ik.L`` / ``thigh_ik.L``、尾控制器 ``toe_ik.L`` 同样套对应 FK 骨的增量；
      * ``*_ik_target`` 是**极向目标**（IK 弯曲平面靠它定，不是链末端的把手）：开了
        ``pole_vector`` 就按 Rigify 的 ``match_pole_target`` 把它摆进 FK 链的平面
        （见 :func:`rigify_limbs.place_pole`），没开就不动它 —— Rigify 自己也只在需要时摆。

    中间的 IK 解算骨（``MCH-*``）由求解器决定，不写。返回写到的控制器个数。
    """
    written = 0
    for role, control in plan["controls"].items():
        delta = _fk_pose_delta(arm_obj, plan["fk_of_role"].get(role) or "")
        bone = arm_obj.data.bones.get(control)
        pbone = arm_obj.pose.bones.get(control)
        if delta is None or bone is None or pbone is None:
            continue
        # 位置 + 旋转一起写：控制器的静止矩阵整体跟着 FK 骨的世界增量走
        pbone.matrix = delta @ bone.matrix_local
        _update_view()
        written += 1

    # 极向目标（``*_ik_target``）：IK 链的弯曲平面靠它定，Rigify 的快照也摆它（见 place_pole）。
    if written and rigify_limbs.pole_enabled(arm_obj, spec):
        roles = plan.get("roles") or []
        root = arm_obj.pose.bones.get(plan["fk_of_role"].get(roles[0]) or "") if roles else None
        if root is not None:
            rigify_limbs.place_pole(arm_obj, spec, root.matrix)
            _update_view()
    return written


def _ik_direct_check(arm_obj, spec: dict, roles) -> tuple | None:
    """核对直写结果：IK 链带出来的 ``DEF-*`` 变形骨 vs 刚写进 FK 链的期待姿态。

    比的是**关节位置**（骨头）与**骨骼方向**，不比尾巴：IK 链带出来的 DEF 骨会被自己的
    stretch 约束拉长，Rigify 自己那条快照路径也一样（实测关节差 0.000 m / 尾巴差 0.27 m），
    拿尾巴当判据会把“跟快照一样对”的结果误判成失败。

    返回 ``(最大位置差 m, 最大方向差 度)``；一根能核对的骨都找不到时返回 ``None``
    （调用方按“没通过”处理，退回实测 0.000000 m 的老路径）。
    """
    side = str(spec.get("side") or "")
    if not side:
        return None
    suffix = ".%s" % side
    matrix = arm_obj.matrix_world
    wanted = set(roles or ())
    worst_pos = 0.0
    worst_deg = 0.0
    checked = 0
    for fk_name in spec.get("fk_bones") or []:
        stem = fk_name[:-len(suffix)] if fk_name.endswith(suffix) else ""
        if stem.endswith("_fk"):
            stem = stem[:-len("_fk")]
        if not stem or ("%s.%s" % (stem, side)) not in wanted:
            continue
        fk_pb = arm_obj.pose.bones.get(fk_name)
        def_pb = arm_obj.pose.bones.get("DEF-%s.%s" % (stem, side))
        if fk_pb is None or def_pb is None:
            continue
        worst_pos = max(worst_pos, (matrix @ fk_pb.head - matrix @ def_pb.head).length)
        fk_dir, df_dir = fk_pb.tail - fk_pb.head, def_pb.tail - def_pb.head
        if fk_dir.length > 1e-6 and df_dir.length > 1e-6:
            worst_deg = max(worst_deg, math.degrees(fk_dir.angle(df_dir)))
        checked += 1
    if not checked:
        return None
    return worst_pos, worst_deg


def apply_ik_direct(arm_obj, plans: dict, targets: dict, refs: dict, metrics: dict,
                    options: dict, report: dict) -> dict:
    """IK 模式：把姿态直接写进 IK 控制器（不切 FK、不调 Rigify 的 IK->FK 快照）。

    写完立刻用 ``DEF-*`` 变形骨核对一遍；对不上的肢体不进返回值，调用方会让它们走老路径
    （切 FK + Rigify 快照），所以最坏情况也只是“直写没生效”，不会写出错姿态。
    返回 ``{肢体 key: spec}``（真正直写成功的），并把结果写进 ``report``。
    """
    if not plans:
        return {}
    use_twist = bool(options.get("twist", True))
    twist_blend = float(options.get("twist_blend", 1.0))
    obj_mat3_inv = arm_obj.matrix_world.to_3x3().inverted()

    def to_arm(vec) -> Vector:
        return (obj_mat3_inv @ Vector(vec)).normalized()

    # 直写的前提是这些肢体停在 IK（否则写了也看不见）
    rigify_limbs.set_mode(arm_obj, rigify_limbs.MODE_IK, report,
                          {key: item["spec"] for key, item in plans.items()}, sync=False)
    _update_view()

    failed: list = []
    ready: list = []
    # 1) 先把这些肢体的 FK 链写完整：万一后面直写没通过、退回 Rigify 的快照，
    #    快照读的就是这条链上的姿态（否则会拿静止姿态去对齐 IK 控制器）。
    for key, item in plans.items():
        try:
            _write_limb_fk_chain(arm_obj, item["plan"], targets, refs, metrics, to_arm,
                                 use_twist, twist_blend)
            ready.append(key)
        except Exception as exc:  # noqa: BLE001
            failed.append((item["spec"]["fk_bones"][0], "写 FK 链失败：%s" % exc))
    _update_view()

    # 2) 再写 IK 控制器，并用 DEF-* 变形骨核对
    done: dict = {}
    measured: dict = {}
    for key in ready:
        item = plans[key]
        spec, plan = item["spec"], item["plan"]
        name = spec["fk_bones"][0]
        try:
            written = _write_limb_ik_controls(arm_obj, spec, plan)
            _update_view()
            check = _ik_direct_check(arm_obj, spec, plan["roles"])
        except Exception as exc:  # noqa: BLE001 直写出问题不该拖累整次写入
            failed.append((name, "直写报错：%s" % exc))
            continue
        if written <= 0:
            failed.append((name, "没找到能直写的控制器"))
            continue
        if check is not None:
            measured[name] = [round(float(check[0]), 4), round(float(check[1]), 2)]
        if check is None or check[0] > IK_DIRECT_POS_TOL or check[1] > IK_DIRECT_ANGLE_TOL:
            # 没过：按 Rigify 的思路搜一下链根的扭转（弯曲平面），能救回来就救
            if check is not None and _search_root_roll(arm_obj, spec, plan):
                check = _ik_direct_check(arm_obj, spec, plan["roles"])
                measured[name] = ([round(float(check[0]), 4), round(float(check[1]), 2)]
                                  if check is not None else None)
        if check is None or check[0] > IK_DIRECT_POS_TOL or check[1] > IK_DIRECT_ANGLE_TOL:
            failed.append((name, "没法核对（没有 DEF-* 变形骨）" if check is None
                           else "偏差 %.3f m / %.1f 度" % (check[0], check[1])))
            continue
        done[key] = spec

    report["ik_direct"] = [spec["fk_bones"][0] for spec in done.values()]
    if measured:
        report["ik_direct_error"] = measured
    if failed:
        report["warnings"].append(
            "这些肢体改用 Rigify 的 IK 快照（不直写）：%s"
            % "；".join("%s %s" % (name, text) for name, text in failed))
    return done


def _apply_root_translation(arm_obj, bmap, metrics, recon, options, report) -> None:
    """把人物的水平位置/站立高度写到根骨骼的位移通道（可选）。"""
    root_name = bmap.get("root")
    if not root_name:
        report["warnings"].append("骨架里没有 root/torso 骨骼，跳过根位移")
        return
    pbone = arm_obj.pose.bones.get(root_name)
    if pbone is None:
        report["warnings"].append("找不到根骨骼 %s" % root_name)
        return

    joints = recon["joints"]
    rest_pelvis_world = arm_obj.matrix_world @ Vector(metrics["pelvis"])
    pelvis = Vector(joints["pelvis"])
    target = Vector((pelvis.x, rest_pelvis_world.y, rest_pelvis_world.z))

    if bool(options.get("ground_align", True)):
        lows = [Vector(joints["ankle.%s" % side]) for side in ("L", "R")]
        lows += [Vector(joints["ball.%s" % side]) for side in ("L", "R")]
        lowest = min(vec.z for vec in lows)
        # 保持“骨盆高于最低脚点”的高度，把脚放回骨架静止时的地面
        target.z = metrics["lowest_foot_z"] + max(pelvis.z - lowest, 0.0)

    delta_world = target - rest_pelvis_world
    if bool(options.get("keep_depth", True)):
        delta_world.y = 0.0

    delta_arm = arm_obj.matrix_world.to_3x3().inverted() @ delta_world
    matrix = pbone.matrix.copy()
    matrix.translation = Vector(arm_obj.data.bones[root_name].head_local) + delta_arm
    pbone.matrix = matrix
    if bpy.context.view_layer is not None:
        bpy.context.view_layer.update()
    report["root_offset"] = tuple(round(float(v), 4) for v in delta_arm)


def insert_keyframes(arm_obj, bmap, roles, frame: int, report: dict, extra_bones=None,
                     prop_bones=None) -> None:
    """为被写入的骨骼插入关键帧。

    * ``extra_bones``：IK 模式下要一起打帧的 IK 控制器（不然只录了 FK 骨骼，
      动画里看不到变化）；
    * ``prop_bones``：Rigify 的 ``IK_FK`` 滑块属性（切 IK/FK 也要能录进动画）。
    """
    extra = list(extra_bones or [])
    names: list[str] = []
    for role in set(list(roles) + ["root"]):
        name = bmap.get(role)
        if name and name not in names:
            names.append(name)
    for name in extra:
        if name not in names:
            names.append(name)

    count = 0
    for name in names:
        pbone = arm_obj.pose.bones.get(name)
        if pbone is None:
            continue
        if name == bmap.get("root") or name in extra:
            pbone.keyframe_insert(data_path="location", frame=frame)
        path = "rotation_quaternion" if pbone.rotation_mode == "QUATERNION" \
            else "rotation_euler"
        pbone.keyframe_insert(data_path=path, frame=frame)
        if name in extra:
            pbone.keyframe_insert(data_path="scale", frame=frame)
        count += 1

    for name in (prop_bones or []):
        pbone = arm_obj.pose.bones.get(name)
        if pbone is None or "IK_FK" not in pbone.keys():
            continue
        pbone.keyframe_insert(data_path='["IK_FK"]', frame=frame)
        count += 1
    report["keyframes"] = count


def apply_pose(arm_obj, bmap, metrics, recon, options=None) -> dict:
    """主入口：把解算结果写到 Pose 骨骼，返回执行报告。"""
    options = options or {}
    report = {"ok": False, "applied": [], "skipped": [],
              "warnings": list(metrics.get("warnings", [])), "message": ""}

    if metrics.get("errors"):
        report["message"] = "；".join(metrics["errors"])
        return report
    if not recon or not recon.get("ok"):
        report["message"] = (recon or {}).get("message", "姿态重建失败")
        return report

    use_twist = bool(options.get("twist", True))
    twist_blend = float(options.get("twist_blend", 1.0))
    obj_mat3_inv = arm_obj.matrix_world.to_3x3().inverted()

    def to_arm(vec) -> Vector:
        return (obj_mat3_inv @ Vector(vec)).normalized()

    targets = build_targets(recon, bmap, metrics)
    refs = recon["refs"]
    plan = []

    for role, (p_from, p_to, ref_name) in targets.items():
        name = bmap.get(role)
        if not name:
            report["skipped"].append((role, "未映射到骨骼"))
            continue
        pbone = arm_obj.pose.bones.get(name)
        bone = arm_obj.data.bones.get(name)
        if pbone is None or bone is None:
            report["skipped"].append((role, "骨架中没有 %s" % name))
            continue
        direction = Vector(p_to) - Vector(p_from)
        if direction.length < 1e-6:
            report["skipped"].append((role, "方向退化"))
            continue

        rest_mat = metrics["rest_mat"].get(role)
        if rest_mat is None:
            rest_mat = bone.matrix_local.to_3x3()
        mat3 = solve_role_matrix(role, (p_from, p_to, ref_name), refs, metrics["ref"],
                                 rest_mat, to_arm, use_twist, twist_blend)
        if mat3 is None:
            report["skipped"].append((role, "方向退化"))
            continue
        plan.append((len(bone.parent_recursive), role, name, mat3))

    plan.sort(key=lambda item: item[0])

    view_layer = bpy.context.view_layer

    # Rigify 生成骨架的手臂 / 大腿默认挂在 IK 控制器上。IK 模式下能直写 IK 控制器的肢体就
    # 直写（不切 FK、不做快照）；其余肢体（写 FK / 退回快照）仍然要先把它们切到 FK，
    # 否则 IK 链会盖住 FK 结果，而且后面的 IK 快照读到的也是错的姿态。
    limb_mode = limb_mode_option(options)
    limb_specs = rigify_limbs.limb_specs(arm_obj)
    written_limbs = _limbs_written({item[2] for item in plan}, limb_specs)
    direct_plans = (direct_limb_plans(bmap, written_limbs)
                    if limb_mode == rigify_limbs.MODE_IK else {})
    if written_limbs and not direct_plans:
        rigify_limbs.set_mode(arm_obj, rigify_limbs.MODE_FK, report, written_limbs,
                              sync=False)
        if view_layer is not None:
            view_layer.update()

    pivot = Vector(metrics.get("pelvis") or Vector((0.0, 0.0, 0.0)))
    hip_line_rest = metrics["ref"].get("hip_line")
    hip_line_target = recon["refs"].get("hip_line")
    for _depth, role, name, mat3 in plan:
        pbone = arm_obj.pose.bones[name]
        head = pbone.head.copy()
        bone = arm_obj.data.bones[name]
        rest_mat = metrics["rest_mat"].get(role) or bone.matrix_local.to_3x3()

        if role == "hips" and hip_line_rest is not None and hip_line_target is not None:
            # Rigify 的 torso 是“脊椎 IK 控制”，静止方向不是身体轴（实测指向 +Y），
            # 用方向对齐会把它转 90 度把整个人甩飞。这里改成：
            # 用胯线求出骨盆旋转，再叠到它的静止姿态上。
            rotation = hip_line_rest.normalized().rotation_difference(
                to_arm(hip_line_target))
            mat3 = rotation.to_matrix() @ rest_mat
        else:
            rotation = mat3 @ rest_mat.inverted()

        matrix = Matrix.Translation(head) @ mat3.to_4x4()
        if role == "hips":
            # 让骨盆保持原位（等效于绕骨盆旋转，而不是绕骨骼头部转）
            offset = pivot - Vector(bone.head_local)
            matrix.translation = head + (offset - rotation @ offset)
        try:
            pbone.matrix = matrix
        except Exception as exc:  # noqa: BLE001
            report["skipped"].append((role, "写入失败：%s" % exc))
            continue
        if view_layer is not None:
            view_layer.update()
        report["applied"].append(role)

    if options.get("root_motion", False):
        _apply_root_translation(arm_obj, bmap, metrics, recon, options, report)

    # IK 模式：能直写的肢体直接把姿态写进 IK 控制器（不切 FK、不做快照）
    direct_done: dict = {}
    if direct_plans:
        direct_done = apply_ik_direct(arm_obj, direct_plans, targets, recon["refs"],
                                      metrics, options, report)

    close_limbs(arm_obj, written_limbs, limb_mode, report, options, limb_specs,
                direct=direct_done)

    if options.get("keyframe", False):
        frame = int(options.get("frame", bpy.context.scene.frame_current))
        insert_keyframes(arm_obj, bmap, list(report["applied"]), frame, report,
                         extra_bones=_limb_keyframe_bones(written_limbs, limb_mode),
                         prop_bones=[spec["prop_bone"] for spec in written_limbs.values()])

    report["ok"] = bool(report["applied"])
    report["message"] = "已写入 %d 根骨骼" % len(report["applied"])
    return report


