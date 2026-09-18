"""Rigify 肢体 IK 控制器自检（新的默认写入路径）。

验证：
  1. 能从 Rigify 生成脚本里解析出四条肢体链（fk / ik / ctrl / tail / extra）；
  2. 默认（limb_mode="IK" + IK 模式自动匹配）写入时：**直接把姿态写进 IK 控制器**
     （链根 `upper_arm_ik` / `thigh_ik`、末端 `hand_ik` / `foot_ik`、尾控制器 `toe_ik`；
     位置与旋转一起写，链根还要按 Rigify 的 `correct_rotation` 搜一遍扭转 = 弯曲平面）。
     一次写不到容差里的肢体**自动退回** Rigify 的 IK 快照，所以结果一定是
     “N 条直写 + 其余快照”，两边加起来等于 4；两种路径的 DEF-* 都跟 FK 结果对得上；
  3. 映射仍然指向 FK 骨骼时（老配置）退回老路径：切 FK -> 写 FK 旋转 -> Rigify 自带的
     “IK->FK” 快照 -> 停在 IK（实测误差 0.000000 m）；
  4. IK 模式下 hand_ik / foot_ik 拖动仍然带动变形骨；
  5. 切到 FK 时藏起 IK 控制器、显示 FK 控制器（反向同理，含“自己拖滑块”的兜底同步）。

用法：
  blender.exe -b --factory-startup --python "dev_tests/test_rigify_limbs.py"
"""
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
from test_rigify_rig import (FK_POSE, landmark_error, set_fk,  # noqa: E402
                             setup_generated_rig, synth_from_def)

from pose_capture import reconstruction, retarget, rigify_limbs, rigify_map  # noqa: E402

DEF_NAMES = ("DEF-upper_arm.L", "DEF-forearm.L", "DEF-hand.L", "DEF-f_index.01.L",
             "DEF-thigh.L", "DEF-shin.L", "DEF-foot.L",
             "DEF-thigh.R", "DEF-shin.R", "DEF-foot.R", "DEF-toe.R")

LIMBS = ("Arm.L", "Arm.R", "Leg.L", "Leg.R")
OPTIONS = {"twist": True, "twist_blend": 1.0, "root_motion": False}


def world_point(rig, name):
    pose_bone = rig.pose.bones[name]
    return rig.matrix_world @ pose_bone.head


def snapshot(rig, names=DEF_NAMES):
    return {name: world_point(rig, name) for name in names if name in rig.pose.bones}


def worst_shift(before, after):
    worst = 0.0
    for name, point in before.items():
        other = after.get(name)
        if other is not None:
            worst = max(worst, (point - other).length)
    return worst


def collection_state(rig):
    return {name: bool(rig.data.collections[name].is_visible)
            for name in (rig.data.collections.keys()) if name.startswith(("Arm.", "Leg."))}


def expect_collections(rig, mode, failures):
    """IK 模式 -> IK 集合可见 / FK 隐藏；FK 模式反之。"""
    other = "FK" if mode == "IK" else "IK"
    state = collection_state(rig)
    for limb in LIMBS:
        want = state.get("%s (%s)" % (limb, mode))
        got_other = state.get("%s (%s)" % (limb, other))
        ok = want is True and got_other is False
        print("  %s%s：%s 可见 / %s 隐藏" % ("OK " if ok else "!! ", limb, mode, other))
        if not ok:
            failures.append("%s 集合可见性不对（%s=%s, %s=%s）"
                            % (limb, mode, want, other, got_other))


def check(result, label, failures, detail=""):
    print("  %s%s %s" % ("OK " if result else "!! ", label, detail))
    if not result:
        failures.append(label)
    return result


def reset_pose(rig):
    """回到静止姿态（实际用法也是“清空 Pose 再应用”，这样两次运行才可比）。"""
    for pose_bone in rig.pose.bones:
        pose_bone.matrix_basis = Matrix.Identity(4)
    bpy.context.view_layer.update()


def nudge(rig, bone_name, offset):
    """模拟用户拖 IK 控制器：位移后返回（前, 后）两次 DEF 骨快照。"""
    names = ("DEF-hand.L", "DEF-f_index.01.L", "DEF-foot.R", "DEF-toe.R")
    before = snapshot(rig, names)
    pose_bone = rig.pose.bones[bone_name]
    pose_bone.matrix = Matrix.Translation(Vector(offset)) @ pose_bone.matrix
    bpy.context.view_layer.update()
    after = snapshot(rig, names)
    pose_bone.matrix = Matrix.Translation(Vector([-v for v in offset])) @ pose_bone.matrix
    bpy.context.view_layer.update()
    return before, after


def action_curve_paths(action) -> list:
    """列出 action 里所有 fcurve 的 data_path（兼容 Blender 4.4+ 的 layered action）。"""
    paths: set = set()
    curves = getattr(action, "fcurves", None)
    if curves is not None:
        for curve in curves:
            paths.add(curve.data_path)
        return sorted(paths)
    for layer in getattr(action, "layers", []):
        for strip in getattr(layer, "strips", []):
            for bag in getattr(strip, "channelbags", []):
                for curve in getattr(bag, "fcurves", []):
                    paths.add(curve.data_path)
    return sorted(paths)


def main():
    ensure_registered()
    rig = setup_generated_rig()
    bpy.context.scene.pose_capture.armature = rig
    failures = []

    # ---------------------------------------------------------------- 1) 解析肢体链
    print("[1] 解析 Rigify 肢体链")
    specs = rigify_limbs.limb_specs(rig)
    print("  找到：", ", ".join(sorted(specs)))
    check(len(specs) == 4, "四条肢体链都解析到了", failures, "(%d)" % len(specs))
    for key, spec in sorted(specs.items()):
        print("    %-18s -> %-20s FK %s / IK ctrl %s"
              % (key, spec["prop_bone"], spec["fk_bones"], spec["ctrl_bones"]))
        check(spec["prop_bone"].endswith(("_parent.L", "_parent.R")),
              "%s 的 prop_bone 正常" % key, failures, spec["prop_bone"])
    check(rigify_limbs.rig_id(rig) != "", "rig_id 可读", failures)

    # ---------------------------------------------------------------- 2) 造真值姿态
    bmap = rigify_map.auto_detect(rig, False)
    metrics = rigify_map.read_metrics(rig, bmap)
    if metrics.get("errors"):
        print("!! read_metrics 失败：", metrics["errors"])
        return 1
    set_fk(rig)
    for name, degrees in FK_POSE:
        rotate_bone_world(rig, name, (0.0, 1.0, 0.0), degrees)
    keypoints, scores = synth_from_def(rig)
    if keypoints is None:
        return 1
    recon = reconstruction.reconstruct(keypoints, scores, metrics, IMAGE_SIZE,
                                      reconstruction.default_options())
    if not recon.get("ok"):
        print("!! 重建失败：", recon.get("message"))
        return 1

    # ---------------------------------------------------------------- 3) FK 模式基线
    print("[2] limb_mode=FK（旧行为）")
    reset_pose(rig)
    report_fk = retarget.apply_pose(rig, bmap, metrics, recon,
                                    dict(OPTIONS, limb_mode="FK"))
    check(report_fk.get("ok", False), "写入成功", failures, report_fk["message"])
    print("  报告：", report_fk.get("message"), "| fk_switched =", report_fk.get("fk_switched"))
    fk_points = snapshot(rig)
    check(all(abs(v - 1.0) < 1e-6 for v in rigify_limbs.limb_modes(rig).values()),
          "四条肢体的 IK_FK 都 = 1（FK）", failures)
    expect_collections(rig, "FK", failures)
    error_fk = landmark_error(keypoints, synth_from_def(rig)[0])
    print("  FK 模式画布内平均偏差 %.2f 像素" % error_fk)

    # ---------------------------------------------------------------- 4) IK 模式：直写控制器
    print("[3] limb_mode=IK + IK 控制器映射（直写，不切 FK、不做快照）")
    reset_pose(rig)
    bmap_ik = rigify_map.auto_detect(rig, False, "IK")
    metrics_ik = rigify_map.read_metrics(rig, bmap_ik)
    print("  IK 模式匹配：", {role: bmap_ik.get(role) for role in
                              ("upper_arm.L", "forearm.L", "hand.L",
                               "thigh.L", "shin.L", "foot.L")})
    check(bmap_ik.get("upper_arm.L") == "upper_arm_ik.L", "大臂匹配到 upper_arm_ik.L",
          failures, str(bmap_ik.get("upper_arm.L")))
    check(bmap_ik.get("hand.R") == "hand_ik.R" and bmap_ik.get("foot.R") == "foot_ik.R",
          "手 / 脚匹配到 hand_ik / foot_ik", failures,
          "hand=%s foot=%s" % (bmap_ik.get("hand.R"), bmap_ik.get("foot.R")))
    report_ik = retarget.apply_pose(rig, bmap_ik, metrics_ik, recon,
                                    dict(OPTIONS, limb_mode="IK"))
    check(report_ik.get("ok", False), "写入成功", failures, report_ik["message"])
    print("  报告：", report_ik.get("message"))
    print("  直写：", report_ik.get("ik_direct"), "| 核对偏差：",
          report_ik.get("ik_direct_error"))
    print("  退回快照：", report_ik.get("ik_snapped"), "| 显示同步：",
          report_ik.get("visibility"))
    for line in (report_ik.get("warnings") or []):
        print("   警告：", line)
    direct = report_ik.get("ik_direct") or []
    snapped = report_ik.get("ik_snapped") or []
    check(bool(direct), "至少有一条肢体直写进了 IK 控制器", failures,
          "；".join(report_ik.get("warnings") or []))
    check(len(direct) + len(snapped) == 4,
          "四条肢体都落在“直写”或“退回快照”其中一边", failures,
          "直写 %s / 快照 %s" % (direct, snapped))
    check(report_ik.get("fk_switched") is None, "没有切 FK（IK_FK 滑块没动）", failures)
    check(all(abs(v) < 1e-6 for v in rigify_limbs.limb_modes(rig).values()),
          "四条肢体的 IK_FK 都 = 0（IK）", failures)

    ik_points = snapshot(rig)
    shift = worst_shift(fk_points, ik_points)
    print("  IK 直写与 FK 模式的 DEF 骨最大位置差 %.6f m" % shift)
    for name in sorted(fk_points):
        delta = (fk_points[name] - ik_points.get(name, fk_points[name])).length
        print("    %-20s %.6f m" % (name, delta))
    check(shift < retarget.IK_DIRECT_POS_TOL, "IK 直写结果与 FK 结果一致", failures,
          "（%.4f m，阈值 %.2f m）" % (shift, retarget.IK_DIRECT_POS_TOL))
    error_ik = landmark_error(keypoints, synth_from_def(rig)[0])
    print("  IK 模式画布内平均偏差 %.2f 像素（FK %.2f）" % (error_ik, error_fk))
    check(error_ik < 25.0, "画面内偏差在阈值内", failures)
    expect_collections(rig, "IK", failures)

    # ---------------------------------------------------------------- 4b) 老配置：退回快照
    print("[3b] limb_mode=IK + FK 骨骼映射（退回 Rigify 快照）")
    reset_pose(rig)
    report_fb = retarget.apply_pose(rig, bmap, metrics, recon,
                                    dict(OPTIONS, limb_mode="IK"))
    check(report_fb.get("ok", False), "写入成功", failures, report_fb["message"])
    print("  报告：", report_fb.get("message"))
    check(not report_fb.get("ik_direct"), "没有直写（映射指向 FK 骨骼）", failures)
    check(len(report_fb.get("ik_snapped") or []) == 4, "四条肢体都做了 IK 快照", failures)
    check(not report_fb.get("warnings"), "没有警告", failures,
          "；".join(report_fb.get("warnings") or []))
    fb_points = snapshot(rig)
    fb_shift = worst_shift(fk_points, fb_points)
    print("  快照路径与 FK 模式的 DEF 骨最大位置差 %.6f m" % fb_shift)
    check(fb_shift < 1e-4, "IK 快照结果与 FK 结果一致（< 0.1 mm）", failures)
    check(all(abs(v) < 1e-6 for v in rigify_limbs.limb_modes(rig).values()),
          "四条肢体的 IK_FK 都 = 0（IK）", failures)

    # ---------------------------------------------------------------- 5) 控制器还活着
    print("[4] IK 模式下拖 IK 控制器")
    before, after = nudge(rig, "hand_ik.L", (0.0, 0.0, 0.12))
    moved_hand = (before["DEF-hand.L"] - after["DEF-hand.L"]).length
    moved_finger = (before["DEF-f_index.01.L"] - after["DEF-f_index.01.L"]).length
    print("  移动 hand_ik.L 0.12 m -> DEF-hand.L %.4f m / 手指 %.4f m"
          % (moved_hand, moved_finger))
    check(abs(moved_hand - 0.12) < 0.02, "hand_ik 能带动手掌", failures)
    check(abs(moved_finger - 0.12) < 0.02, "手指跟着手掌走", failures)

    before, after = nudge(rig, "foot_ik.R", (0.0, 0.0, 0.10))
    moved_foot = (before["DEF-foot.R"] - after["DEF-foot.R"]).length
    moved_toe = (before["DEF-toe.R"] - after["DEF-toe.R"]).length
    print("  移动 foot_ik.R 0.10 m -> DEF-foot.R %.4f m / 脚趾 %.4f m"
          % (moved_foot, moved_toe))
    check(abs(moved_foot - 0.10) < 0.02, "foot_ik 能带动脚掌", failures)
    check(moved_toe > 0.05, "脚趾也跟着脚掌", failures)

    # ---------------------------------------------------------------- 6) 手动拖滑块的兜底
    print("[5] 用户自己拖 IK_FK 滑块后的显示同步")
    rig.pose.bones["upper_arm_parent.L"]["IK_FK"] = 1.0
    rigify_limbs._on_depsgraph_update(None, None)
    state = collection_state(rig)
    check(state.get("Arm.L (FK)") is True and state.get("Arm.L (IK)") is False,
          "watcher 把 Arm.L 的显示切到 FK", failures,
          "FK=%s IK=%s" % (state.get("Arm.L (FK)"), state.get("Arm.L (IK)")))

    rig.pose.bones["upper_arm_parent.L"]["IK_FK"] = 0.0
    result = bpy.ops.pose_capture.sync_limb_display()
    state = collection_state(rig)
    check(result == {"FINISHED"} and state.get("Arm.L (IK)") is True
          and state.get("Arm.L (FK)") is False,
          "同步按钮把 Arm.L 的显示切回 IK", failures,
          "FK=%s IK=%s" % (state.get("Arm.L (FK)"), state.get("Arm.L (IK)")))

    # ---------------------------------------------------------------- 7) 关键帧
    print("[6] IK 模式 + 插入关键帧")
    frame = bpy.context.scene.frame_current
    reset_pose(rig)
    report_kf = retarget.apply_pose(rig, bmap, metrics, recon,
                                    dict(OPTIONS, limb_mode="IK", keyframe=True, frame=frame))
    action = rig.animation_data.action if rig.animation_data else None
    paths = action_curve_paths(action) if action else []
    print("    关键帧 %s 个；曲线 %d 条" % (report_kf.get("keyframes"), len(paths)))
    check(report_kf.get("keyframes", 0) > 0, "插入了关键帧", failures)
    check(any("hand_ik" in path for path in paths), "IK 控制器有关键帧", failures)
    check(any('["IK_FK"]' in path for path in paths), "IK_FK 滑块有关键帧", failures)
    if action is not None:
        rig.animation_data_clear()
        bpy.context.view_layer.update()

    # ---------------------------------------------------------------- 8) 兜底与坏数据
    print("[7] 兜底：标准链名 + 坏属性块剔除")
    defaults = rigify_limbs._default_specs(rig)
    print("    兜底解析：", sorted(defaults))
    check(len(defaults) == 4, "生成脚本不在时也能按标准命名认出四条肢体", failures)
    check(defaults.get("upper_arm_fk.L", {}).get("prop_bone") == "upper_arm_parent.L",
          "兜底 prop_bone 正确", str(defaults.get("upper_arm_fk.L", {}).get("prop_bone")))

    mixed = {"prop_bone": "upper_arm_parent.L",
             "fk_bones": '["upper_arm_fk.L", "forearm_fk.L", "hand_fk.L"]',
             "ik_bones": '["upper_arm_ik.R", "MCH-forearm_ik.R", "MCH-upper_arm_ik_target.R"]',
             "ctrl_bones": '["upper_arm_ik.R", "upper_arm_ik_target.R", "hand_ik.R"]',
             "tail_bones": "[]", "extra_ctrls": "[]"}
    check(rigify_limbs._spec_from_raw(mixed, rig) is None,
          "左右混掉的属性块会被丢掉（之前真出过这个 bug）", failures)

    wrong_len = dict(mixed)
    wrong_len["ik_bones"] = '["upper_arm_ik.L"]'
    wrong_len["ctrl_bones"] = '["upper_arm_ik.L"]'
    check(rigify_limbs._spec_from_raw(wrong_len, rig) is None,
          "链长对不上的属性块会被丢掉", failures)

    # ---------------------------------------------------------------- 9) 直写计划 + 模式感知匹配
    print("[8] 直写计划 + 模式感知匹配")
    direct = {key: rigify_limbs.ik_write_plan(spec) for key, spec in sorted(specs.items())}
    for key, plan in sorted(direct.items()):
        print("    %-18s -> 控制器 %s  目标 %s" % (key, (plan or {}).get("controls"),
                                                 (plan or {}).get("target")))
    check(all(plan for plan in direct.values()), "四条肢体都认得出来可直写的控制器", failures)
    check((direct.get("upper_arm_fk.L") or {}).get("controls", {}).get("upper_arm.L")
          == "upper_arm_ik.L", "大臂直写 upper_arm_ik.L", failures,
          str((direct.get("upper_arm_fk.L") or {}).get("controls")))
    check((direct.get("upper_arm_fk.L") or {}).get("tip_role") == "forearm.L",
          "手腕关键点取自 forearm.L", failures)
    check((direct.get("thigh_fk.R") or {}).get("target") == "thigh_ik_target.R",
          "大腿的 IK 目标是 thigh_ik_target.R", failures)

    fk_map = rigify_map.auto_detect(rig, False)
    ik_map = rigify_map.auto_detect(rig, False, "IK")
    check(rigify_limbs.mapped_limb_mode(rig, fk_map) == "FK", "FK 映射能被认出来", failures,
          str(rigify_limbs.mapped_limb_mode(rig, fk_map)))
    check(rigify_limbs.mapped_limb_mode(rig, ik_map) == "IK", "IK 映射能被认出来", failures,
          str(rigify_limbs.mapped_limb_mode(rig, ik_map)))
    check(rigify_map.ordered_candidates("upper_arm.{s}", "L", "IK")[0] == "upper_arm_ik.L"
          and rigify_map.ordered_candidates("upper_arm.{s}", "L")[0] == "upper_arm_fk.L",
          "候选骨名顺序跟写入方式一致", failures)

    print("结果：", "通过 ✓" if not failures else "未通过 ✗")
    for item in failures:
        print("   -", item)
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
