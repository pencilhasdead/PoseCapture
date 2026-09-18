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


# --------------------------------------------------------------------------- 应用姿态
def limb_mode_option(options: dict) -> str:
    """这次写完后要停在 IK（Rigify 默认控制器）还是 FK。

    兼容旧键 ``fk_mode``：历史上 True 表示“自动切到 FK 模式”。
    """
    mode = str(options.get("limb_mode") or "").upper()
    if mode in rigify_limbs.MODES:
        return mode
    return rigify_limbs.MODE_FK if options.get("fk_mode") else rigify_limbs.MODE_IK


def _limbs_written(written_names, specs: dict) -> dict:
    """挑出这次真正写到的 Rigify 肢体（该肢体的 FK 链里有骨名被写入）。

    只对“写过的肢体”做 IK 快照：没写过的肢体不该被我们改姿态，
    否则会把它当前的 IK 摆位覆盖成 FK 静止姿态。
    """
    out = {}
    for key, spec in (specs or {}).items():
        if any(name in written_names for name in spec["fk_bones"]):
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
                specs: dict = None) -> None:
    """收尾：IK 模式先把 IK 控制器对齐到 FK 姿态，再按模式同步控制器显示。

    FK 模式不用快照 —— 姿态本来就停在 FK 骨骼上。
    """
    if not limbs:
        # 只有“是 Rigify 生成骨架但连一条 IK 链都找不到”才值得提示
        if mode == rigify_limbs.MODE_IK and not (specs or {}):
            if rigify_limbs.rig_id(arm_obj):
                report["warnings"].append("骨架里找不到 Rigify 的 IK 控制器链，按普通 FK 写入")
        return

    report["limb_mode"] = mode
    if mode == rigify_limbs.MODE_IK:
        # 用 Rigify 自带的 “IK->FK” 快照（pole/拉伸/脚跟都由 Rigify 算），
        # 实测所有 DEF 骨位置误差 0.000000 m，之后 hand_ik/foot_ik 依旧可编辑。
        snapped = rigify_limbs.snap_limbs(arm_obj, limbs, report)
        report["ik_snapped"] = snapped
        missing = [spec["fk_bones"][0] for key, spec in limbs.items()
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
                rest_ref = metrics["ref"].get(ref_name)

        mat3 = solve_orientation(rest_mat, to_arm(direction), ref_arm, rest_ref,
                                 use_twist, twist_blend)
        plan.append((len(bone.parent_recursive), role, name, mat3))

    plan.sort(key=lambda item: item[0])

    view_layer = bpy.context.view_layer

    # Rigify 生成骨架的手臂 / 大腿默认挂在 IK 控制器上：写 FK 旋转前先把这些肢体切到 FK，
    # 否则 IK 链会盖住 FK 结果，而且后面的 IK 快照读到的也是错的姿态。
    limb_mode = limb_mode_option(options)
    limb_specs = rigify_limbs.limb_specs(arm_obj)
    written_limbs = _limbs_written({item[2] for item in plan}, limb_specs)
    if written_limbs:
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

    close_limbs(arm_obj, written_limbs, limb_mode, report, options, limb_specs)

    if options.get("keyframe", False):
        frame = int(options.get("frame", bpy.context.scene.frame_current))
        insert_keyframes(arm_obj, bmap, list(report["applied"]), frame, report,
                         extra_bones=_limb_keyframe_bones(written_limbs, limb_mode),
                         prop_bones=[spec["prop_bone"] for spec in written_limbs.values()])

    report["ok"] = bool(report["applied"])
    report["message"] = "已写入 %d 根骨骼" % len(report["applied"])
    return report


