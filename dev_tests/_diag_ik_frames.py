"""诊断（开发用）：Rigify 的 IK 控制器姿态与 FK 链姿态之间到底差一个什么变换。

思路：先用**老路径**跑一遍（切 FK 写旋转 -> Rigify 自带的 `ik2fk` 快照 -> 回到 IK）。
这时 IK 控制器上带着的就是 Rigify 自己认可的“正确”值（实测 DEF 误差 0.000000 m），
把它和 FK 骨骼姿态、各自的静止矩阵放在一起算，就能**定出**直写该用的公式，而不是猜。

输出每个控制器：静止坐标系偏移 `ctrl_rest^-1 @ fk_rest` 的轴角；快照实际姿态 vs 当前直写
公式（`solve_role_matrix`）vs 两个候选公式（A = Δfk @ ctrl_rest；B = ctrl_rest @ fk_rest^-1
@ fk_pose）的方向差；IK 目标位置的各种候选算法与快照实际的差。

用法：
  blender.exe -b --factory-startup --python "dev_tests/_diag_ik_frames.py"
"""
import math
import os
import sys

import bpy
from mathutils import Matrix, Vector

HERE = os.path.dirname(os.path.abspath(__file__))
ADDON_PARENT = os.path.dirname(HERE)
for path in (ADDON_PARENT, HERE):
    if path not in sys.path:
        sys.path.insert(0, path)

from test_pipeline import IMAGE_SIZE, ensure_registered, rotate_bone_world  # noqa: E402
from test_rigify_rig import FK_POSE, set_fk, setup_generated_rig, synth_from_def  # noqa: E402

from pose_capture import reconstruction, retarget, rigify_limbs, rigify_map  # noqa: E402

OPTIONS = {"twist": True, "twist_blend": 1.0, "root_motion": False, "limb_mode": "IK"}


def rest_mat(rig, name):
    bone = rig.data.bones.get(name)
    return bone.matrix_local.to_3x3() if bone is not None else None


def pose_mat(rig, name):
    pose_bone = rig.pose.bones.get(name)
    return pose_bone.matrix.to_3x3() if pose_bone is not None else None


def pose_head(rig, name):
    pose_bone = rig.pose.bones.get(name)
    return pose_bone.matrix.translation.copy() if pose_bone is not None else None


def angle_between(a, b):
    return math.degrees(a.to_quaternion().rotation_difference(b.to_quaternion()).angle)


def axis_angle(mat3):
    quat = mat3.to_quaternion()
    return tuple(round(v, 2) for v in quat.axis), round(math.degrees(quat.angle), 1)


def xyz(vec):
    return tuple(round(float(v), 3) for v in vec)


def reset_pose(rig):
    for pose_bone in rig.pose.bones:
        pose_bone.matrix_basis = Matrix.Identity(4)
    bpy.context.view_layer.update()



def main():
    ensure_registered()
    rig = setup_generated_rig()
    bpy.context.scene.pose_capture.armature = rig

    # --- 和正式自检一样造一个真值姿态 -------------------------------------
    bmap_fk = rigify_map.auto_detect(rig, False)          # 故意用 FK 映射 -> 走老的快照路径
    metrics = rigify_map.read_metrics(rig, bmap_fk)
    set_fk(rig)
    for name, degrees in FK_POSE:
        rotate_bone_world(rig, name, (0.0, 1.0, 0.0), degrees)
    keypoints, scores = synth_from_def(rig)
    recon = reconstruction.reconstruct(keypoints, scores, metrics, IMAGE_SIZE,
                                       reconstruction.default_options())
    if not recon.get("ok"):
        print("!! 重建失败：", recon.get("message"))
        return 1

    # --- 老路径：快照出来的 IK 控制器姿态就是“标准答案” --------------------
    reset_pose(rig)
    report = retarget.apply_pose(rig, bmap_fk, metrics, recon, dict(OPTIONS))
    print("快照路径：%s | 快照肢体：%s | 警告：%s"
          % (report["message"], report.get("ik_snapped"), report.get("warnings")))

    targets = retarget.build_targets(recon, bmap_fk, metrics)
    obj_mat3_inv = rig.matrix_world.to_3x3().inverted()
    obj_mat_inv = rig.matrix_world.inverted()

    def to_arm(vec):
        return (obj_mat3_inv @ Vector(vec)).normalized()

    print("")
    print("每行：角色 控制器 | 静止偏移(轴,角度) | 当前公式 / A / B 与快照实际的方向差")
    for key, spec in sorted(rigify_limbs.limb_specs(rig).items()):
        plan = rigify_limbs.ik_write_plan(spec)
        print("")
        print("== %s" % key)
        if not plan:
            print("   !! 没有直写计划")
            continue
        for role, control in plan["controls"].items():
            if role not in targets:
                print("   %-12s -> %-18s（该角色没有目标，跳过）" % (role, control))
                continue
            fk_name = plan["fk_of_role"].get(role)
            ctrl_rest, fk_rest = rest_mat(rig, control), rest_mat(rig, fk_name)
            fk_pose, ctrl_pose = pose_mat(rig, fk_name), pose_mat(rig, control)
            if None in (ctrl_rest, fk_rest, fk_pose, ctrl_pose):
                print("   !! 缺骨：%s / %s" % (fk_name, control))
                continue
            offset_axis, offset_deg = axis_angle(ctrl_rest.inverted() @ fk_rest)
            current = retarget.solve_role_matrix(role, targets[role], recon["refs"],
                                                 metrics["ref"], ctrl_rest, to_arm,
                                                 True, 1.0)
            delta = fk_pose @ fk_rest.inverted()
            cand_a, cand_b = delta @ ctrl_rest, ctrl_rest @ fk_rest.inverted() @ fk_pose
            print("   %-12s %-18s" % (role, control))
            print("      静止偏移 %s %.1f° | 当前 %.2f° / A %.2f° / B %.2f°"
                  % (offset_axis, offset_deg, angle_between(current, ctrl_pose),
                     angle_between(cand_a, ctrl_pose), angle_between(cand_b, ctrl_pose)))
            print("      Y 轴：FK %s -> %s ；控制器 %s -> %s"
                  % (xyz(fk_rest @ Vector((0, 1, 0))), xyz(fk_pose @ Vector((0, 1, 0))),
                     xyz(ctrl_rest @ Vector((0, 1, 0))), xyz(ctrl_pose @ Vector((0, 1, 0)))))

        # --- IK 目标的位置 ------------------------------------------------
        target_name, tip_fk, tip_role = plan["target"], plan["tip_fk"], plan["tip_role"]
        if tip_role not in targets:
            print("   （%s 没有目标，跳过 IK 目标对比）" % tip_role)
            continue
        actual = pose_head(rig, target_name)
        rest_target_head = rig.data.bones[target_name].head_local
        rest_tip_tail = rig.data.bones[tip_fk].tail_local
        now_tip_tail = rig.pose.bones[tip_fk].tail
        keypoint = obj_mat_inv @ Vector(targets[tip_role][1])
        print("   IK 目标 %s；快照实际 head=%s；关键点=%s" % (target_name, xyz(actual), xyz(keypoint)))
        print("      静止：目标 head=%s / %s 静止尾巴=%s"
              % (xyz(rest_target_head), tip_fk, xyz(rest_tip_tail)))
        for label, value in (
                ("当前公式 静止目标头+FK尾部位移", rest_target_head + (now_tip_tail - rest_tip_tail)),
                ("候选 静止目标头+关键点-静止尾部", rest_target_head + (keypoint - rest_tip_tail)),
                ("候选 关键点绝对位置", keypoint),
        ):
            diff = Vector(value) - actual
            print("      %-26s 差 %.4f m %s" % (label, diff.length, xyz(diff)))

        # 候选：把 IK 目标当成末端 FK 骨的“挂件”——整体套用末端的 4x4 世界增量
        tip_rest4 = rig.data.bones[tip_fk].matrix_local
        delta_tip4 = rig.pose.bones[tip_fk].matrix @ tip_rest4.inverted()
        cand4 = delta_tip4 @ rig.data.bones[target_name].matrix_local
        actual_delta4 = rig.pose.bones[target_name].matrix @ rig.data.bones[target_name].matrix_local.inverted()
        print("      候选 末端增量@目标静止矩阵 差 %.4f m / %.2f°"
              % ((cand4.translation - actual).length,
                 angle_between(cand4.to_3x3(), pose_mat(rig, target_name))))
        print("      Δ 平移：末端 %s ；目标 %s" % (xyz(delta_tip4.translation), xyz(actual_delta4.translation)))
        print("      Δ 轴角：末端 %s %.1f° ；目标 %s %.1f°"
              % (axis_angle(delta_tip4.to_3x3()) + axis_angle(actual_delta4.to_3x3())))

    print("")
    print("结构：目标/控制器归属与约束")
    for key, spec in sorted(rigify_limbs.limb_specs(rig).items()):
        print("== %s  极向目标开关：%s" % (key, rigify_limbs.pole_enabled(rig, spec)))
        print("   ik=%s" % (spec.get("ik_bones"),))
        print("   ctrl=%s tail=%s extra=%s"
              % (spec.get("ctrl_bones"), spec.get("tail_bones"), spec.get("extra_ctrls")))
        for name in list(spec.get("ik_bones") or []) + list(spec.get("ctrl_bones") or []) \
                + list(spec.get("tail_bones") or []) + list(spec.get("extra_ctrls") or []):
            pb = rig.pose.bones.get(name)
            if pb is None:
                print("   !! 缺骨 %s" % name)
                continue
            basis = pb.matrix_basis
            loc, rot, _scale = basis.decompose()
            cons = ["%s(%s->%s/%s,chain=%s,infl=%.2f)"
                    % (c.type, c.name, getattr(c, "target", None) and c.target.name,
                       getattr(c, "subtarget", ""), getattr(c, "chain_count", ""),
                       c.influence)
                    for c in pb.constraints]
            print("   %-24s parent=%-22s basis loc=%.4f rot=%.1f° head=%s%s"
                  % (name, pb.parent.name if pb.parent else "-", loc.length,
                     math.degrees(rot.angle), xyz(pb.matrix.translation),
                     ("  约束: " + "; ".join(cons)) if cons else ""))
        fk_list = spec.get("fk_bones") or []
        ik_list = spec.get("ik_bones") or []
        for fk_name, ik_name in zip(fk_list, ik_list):
            fk_pb, ik_pb = rig.pose.bones.get(fk_name), rig.pose.bones.get(ik_name)
            if fk_pb is None or ik_pb is None:
                continue
            print("   FK/IK 对比 %-18s vs %-24s 头差 %.4f 尾差 %.4f 方向差 %.1f°"
                  % (fk_name, ik_name, (fk_pb.head - ik_pb.head).length,
                     (fk_pb.tail - ik_pb.tail).length,
                     angle_between(fk_pb.matrix.to_3x3(), ik_pb.matrix.to_3x3())))

    print("")
    print("直写逐骨核对（IK 映射 -> 直写；差 = FK - DEF，拿真代码跑）")
    reset_pose(rig)
    bmap_ik = rigify_map.auto_detect(rig, False, "IK")
    metrics_ik = rigify_map.read_metrics(rig, bmap_ik)
    report_direct = retarget.apply_pose(rig, bmap_ik, metrics_ik, recon, dict(OPTIONS))
    print("  直写：%s" % (report_direct.get("ik_direct"),))
    print("  偏差：%s" % (report_direct.get("ik_direct_error"),))
    print("  警告：%s" % (report_direct.get("warnings"),))
    for key, spec in sorted(rigify_limbs.limb_specs(rig).items()):
        side = spec["side"]
        for fk_name in spec["fk_bones"]:
            stem = fk_name[:-2]
            if stem.endswith("_fk"):
                stem = stem[:-3]
            def_name = "DEF-%s.%s" % (stem, side)
            fk_pb, def_pb = rig.pose.bones.get(fk_name), rig.pose.bones.get(def_name)
            if fk_pb is None or def_pb is None:
                continue
            print("    %-11s %-15s 关节差 %.4f m 尾巴差 %.4f m 方向差 %.2f°"
                  % (key, def_name, (fk_pb.head - def_pb.head).length,
                     (fk_pb.tail - def_pb.tail).length,
                     angle_between(fk_pb.matrix.to_3x3(), def_pb.matrix.to_3x3())))

    print("")
    print("DEF 与 FK 的对照（确认快照确实是标准答案；差 = FK - DEF）")
    for fk_name, def_name in (("upper_arm_fk.L", "DEF-upper_arm.L"),
                              ("forearm_fk.L", "DEF-forearm.L"),
                              ("hand_fk.L", "DEF-hand.L"),
                              ("thigh_fk.R", "DEF-thigh.R"),
                              ("shin_fk.R", "DEF-shin.R"),
                              ("foot_fk.R", "DEF-foot.R"),
                              ("toe_fk.R", "DEF-toe.R")):
        fk_pb, def_pb = rig.pose.bones.get(fk_name), rig.pose.bones.get(def_name)
        if fk_pb is None or def_pb is None:
            continue
        print("  %-14s 头差 %.4f m 尾差 %.4f m 方向差 %.2f°"
              % (def_name, (fk_pb.head - def_pb.head).length,
                 (fk_pb.tail - def_pb.tail).length,
                 angle_between(fk_pb.matrix.to_3x3(), def_pb.matrix.to_3x3())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
