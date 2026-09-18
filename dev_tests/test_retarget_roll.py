"""写入姿态后的 roll（扭转）自检。

背景（这次修掉的两个坑）：
  1. apply_pose 里用 `ref_vec.length` 判断参照轴是否有效，而解算结果里的参照轴
     是 numpy 数组（没有 .length 属性），于是“扭转(roll)”环节从来没执行过：
     手臂的 roll 只是极小旋转的副产物，看上去就像肘/腕被拧了几十度；
  2. 扭转对齐原来用“离参照轴最近的矩阵列”代替参照轴本身，而 Rigify 的肘弯曲
     轴与任何一列都差几十度，即使执行了 roll 也整体偏掉。
本测试把这两点盯住：写入后的 DEF 变形骨姿态矩阵必须与真值一致（既含方向也含 roll）。

做法：
  1. 用骨架自己的肘弯曲轴摆出解剖学合法的屈曲姿态（真值），这样 roll 就是
     “腕部不带额外扭转”的静止约定；
  2. 把该姿态合成成 COCO-WholeBody 133 点，走完整管线（重建 + 重定向）；
  3. 比较写入后 DEF 骨的姿态矩阵与真值。

用法：
  blender.exe -b --factory-startup --python "dev_tests/test_retarget_roll.py"
"""
import math
import os
import sys

import bpy
from mathutils import Vector

HERE = os.path.dirname(os.path.abspath(__file__))
ADDON_PARENT = os.path.dirname(HERE)
for path in (ADDON_PARENT, HERE):
    if path not in sys.path:
        sys.path.insert(0, path)

from test_rigify_rig import set_fk, setup_generated_rig, synth_from_def  # noqa: E402
from test_pipeline import IMAGE_SIZE, rotate_bone_world  # noqa: E402

from pose_capture import reconstruction, retarget, rigify_map  # noqa: E402

SIDE = "L"
WATCH = ("DEF-upper_arm.%s" % SIDE, "DEF-forearm.%s" % SIDE, "DEF-hand.%s" % SIDE)
UPPER = "upper_arm_fk.%s" % SIDE
FOREARM = "forearm_fk.%s" % SIDE


def rest_hinge(metrics, side=SIDE) -> Vector:
    """静止姿态下的肘弯曲轴（上臂 x 小臂）。"""
    return Vector(metrics["rest_dir"]["upper_arm.%s" % side]).cross(
        Vector(metrics["rest_dir"]["forearm.%s" % side])).normalized()


def posed_axis(rig, bone_name: str, world_axis: Vector) -> Vector:
    """把静止世界轴换算到该骨骼当前姿态下的世界方向。"""
    pose_bone = rig.pose.bones[bone_name]
    local = pose_bone.bone.matrix_local.to_3x3().inverted() @ Vector(world_axis)
    return (pose_bone.matrix.to_3x3() @ local).normalized()


def matrix_error(truth, now) -> float:
    """两个 3x3 姿态矩阵之间的旋转角（度）。"""
    return math.degrees(abs((truth.inverted() @ now).to_quaternion().angle))


def direction_error(a, b) -> float:
    a, b = Vector(a), Vector(b)
    if a.length < 1e-9 or b.length < 1e-9:
        return float("nan")
    return math.degrees(a.normalized().angle(b.normalized()))


# (名字, 大臂摆动, 肘屈曲角) —— 都是“人真的做得出来”的姿势
POSES = (
    ("静止位屈曲 45 度", (), 45.0),
    ("手臂垂下屈曲 60 度", ((UPPER, (0.0, 1.0, 0.0), 75.0),), 60.0),
    ("手抬到胸前屈曲 90 度", ((UPPER, (0.0, 1.0, 0.0), 105.0),), 90.0),
)


def run_pose(name, swings, flexion):
    rig = setup_generated_rig()
    set_fk(rig)
    bmap = rigify_map.auto_detect(rig, False)
    metrics = rigify_map.read_metrics(rig, bmap)
    if metrics.get("errors"):
        print("!! read_metrics 失败：", metrics["errors"])
        return None

    for bone, axis, degrees in swings:
        rotate_bone_world(rig, bone, axis, degrees)
    if abs(flexion) > 1e-6:
        rotate_bone_world(rig, FOREARM, posed_axis(rig, UPPER, rest_hinge(metrics)),
                          flexion)

    keypoints, scores = synth_from_def(rig)
    if keypoints is None:
        print("!! 缺少 DEF 变形骨，无法合成关键点")
        return None
    truth = {name: rig.pose.bones[name].matrix.to_3x3().copy() for name in WATCH}
    truth_dir = {name: (Vector(rig.pose.bones[name].tail)
                        - Vector(rig.pose.bones[name].head)).normalized()
                 for name in WATCH}

    recon = reconstruction.reconstruct(keypoints, scores, metrics, IMAGE_SIZE,
                                       reconstruction.default_options())
    if not recon.get("ok"):
        print("!! 重建失败：", recon.get("message"))
        return None
    report = retarget.apply_pose(rig, bmap, metrics, recon,
                                 {"twist": True, "twist_blend": 1.0,
                                  "fk_mode": True, "root_motion": False})

    print("%s  （写入 %s，IK_FK=%s）" % (name, report["message"],
                                        report.get("fk_switched")))
    worst = 0.0
    for label, bone_name in (("upper_arm", WATCH[0]), ("forearm", WATCH[1]),
                             ("hand", WATCH[2])):
        pose_bone = rig.pose.bones[bone_name]
        direction = (Vector(pose_bone.tail) - Vector(pose_bone.head)).normalized()
        err_dir = direction_error(direction, truth_dir[bone_name])
        err = matrix_error(truth[bone_name], pose_bone.matrix.to_3x3())
        # 手部只作参考：reconstruction 里手的方向是“画面内的指根方向 + 腕部深度”，
        # 手掌朝向镜头时本来就有偏差，这条不算 roll 的问题。
        if label != "hand":
            worst = max(worst, err)
        print("   %-12s 姿态矩阵误差 %6.2f 度（其中方向误差 %5.2f 度）%s"
              % (label, err, err_dir, "（仅参考）" if label == "hand" else ""))
    return worst


def main():
    worst_all = 0.0
    for name, swings, flexion in POSES:
        worst = run_pose(name, swings, flexion)
        if worst is None:
            print("结果：未通过 ✗")
            return 1
        worst_all = max(worst_all, worst)
    print("最大姿态矩阵误差：%.2f 度（阈值 15 度）" % worst_all)
    ok = worst_all < 15.0
    print("结果：", "通过 ✓" if ok else "未通过 ✗（roll/扭转没有对齐，检查 retarget.solve_orientation）")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
