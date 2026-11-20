"""同一张照片：SAM 3D Body 与「DWPose + 深度模型」两条路的结果差多少（角度）。

用来回答“我配了 SAM 3D Body，为什么看起来没变化”：
如果两条路的骨骼方向差很大、而你的结果和以前一模一样，说明实际跑的还是深度那条路。

用法（需要 Blender，因为深度/关键点模型都装在插件数据目录里）：
  blender.exe -b --factory-startup --python "dev_tests/test_sam3d_vs_depth.py" -- <照片> <npz>
"""
import math
import os
import sys

import bpy
import numpy as np
from mathutils import Vector

HERE = os.path.dirname(os.path.abspath(__file__))
ADDON_PARENT = os.path.dirname(HERE)
for path in (ADDON_PARENT, HERE):
    if path not in sys.path:
        sys.path.insert(0, path)


from test_pipeline import ensure_registered                      # noqa: E402
from test_sam3d_rig import setup_rig                             # noqa: E402

from pose_capture import (depth as depth_mod, dwpose, props,  # noqa: E402
                          reconstruction, rigify_map, sam3d, utils)

ensure_registered()



def angle(pair_a, pair_b):
    a = Vector(pair_a[1]) - Vector(pair_a[0])
    b = Vector(pair_b[1]) - Vector(pair_b[0])
    if a.length < 1e-9 or b.length < 1e-9:
        return None
    return math.degrees(math.acos(max(-1.0, min(1.0, a.normalized().dot(b.normalized())))))


def bone(joints, start, end):
    return (np.asarray(joints[start], float), np.asarray(joints[end], float))


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    if len(argv) < 2:
        print("用法：blender -b --python test_sam3d_vs_depth.py -- <照片> <npz>")
        return 2
    image_path, npz = argv[0], argv[1]

    print("[1] 跑 DWPose + 深度模型（老路子）")
    rig = setup_rig()                       # 老路子需要骨架骨长当约束
    settings = bpy.context.scene.pose_capture
    settings.armature = rig
    bmap = rigify_map.auto_detect(rig, False, {"Arm": "FK", "Leg": "FK"})
    metrics = rigify_map.read_metrics(rig, bmap)
    from pose_capture import deps
    detector, pose_path = deps.model_paths()          # 不用偏好设置（--factory-startup 里没有）
    depth_path = deps.depth_model_path(depth_mod.DEFAULT_VARIANT)
    if not (os.path.isfile(detector) and os.path.isfile(pose_path)):
        print("  没有 DWPose 模型，跳过对比")
        return 2
    depth_ready = os.path.isfile(depth_path)
    rgb = utils.load_image_rgb8(image_path)

    estimator = dwpose.DwPoseEstimator(detector, pose_path, "auto", None)
    result = estimator.run(rgb, det_thr=0.3, person_index=0)
    depth_map = depth_size = None
    if depth_ready:
        depth_map, depth_size = depth_mod.DepthEstimator(depth_path, "auto", None).predict(rgb)
    options = dict(props.options_from_settings(settings))
    if depth_map is not None:
        options["depth_near_large"] = True
    old = reconstruction.reconstruct(result["keypoints"], result["scores"], metrics,
                                     (rgb.shape[1], rgb.shape[0]), options,
                                     depth_map=depth_map, depth_size=depth_size)

    print("  老路子：mode=%s ok=%s" % (old.get("mode"), old.get("ok")))
    if not old.get("ok"):
        print("  重建失败：%s" % old.get("message"))
        return 1

    print("[2] 读 SAM 3D Body 的结果")
    data = sam3d.load_npz(npz)
    new = sam3d.build_recon(data, options)
    print("  SAM：mode=%s yaw=%.1f° 重投影 %.2f px"
          % (new["mode"], new["yaw_deg"], new["sam3d"]["reproj_error"]))

    print("\n[3] 骨骼方向差（SAM 与 老路子）")
    rows = (("上臂 L", "shoulder.L", "elbow.L"),
            ("小臂 L", "elbow.L", "wrist.L"),
            ("上臂 R", "shoulder.R", "elbow.R"),
            ("小臂 R", "elbow.R", "wrist.R"),
            ("大腿 L", "hip.L", "knee.L"),
            ("小腿 L", "knee.L", "ankle.L"),
            ("大腿 R", "hip.R", "knee.R"),
            ("小腿 R", "knee.R", "ankle.R"),
            ("头", "head_base", "head_tip"),
            ("躯干", "pelvis", "chest"))
    worst = 0.0
    for label, start, end in rows:
        value = angle(bone(old["joints"], start, end), bone(new["joints"], start, end))
        if value is None:
            print("  %-6s 方向退化" % label)
            continue
        worst = max(worst, value)
        print("  %-6s 差 %5.1f°" % (label, value))
    print("  最大方向差：%.1f°" % worst)

    print("\n[4] 关节位置差（换算到同一骨盆原点、同一尺度：用骨长比例归一）")
    def normalise(joints):
        pelvis = np.asarray(joints["pelvis"], float)
        scale = np.linalg.norm(np.asarray(joints["knee.L"], float) - pelvis)
        scale = max(scale, 1e-6)
        return {key: (np.asarray(value, float) - pelvis) / scale for key, value in joints.items()}

    left, right = normalise(old["joints"]), normalise(new["joints"])
    for label, key in (("左手腕", "wrist.L"), ("右手腕", "wrist.R"),
                       ("左膝", "knee.L"), ("右膝", "knee.R"), ("头基座", "head_base")):
        if key in left and key in right:
            print("  %-6s %.3f（骨盆到膝盖长度的 %.0f%%）"
                  % (label, float(np.linalg.norm(left[key] - right[key])),
                     float(np.linalg.norm(left[key] - right[key])) * 100))
    print("\n结论：差值越大，说明两条路本来就该长得不一样；"
          "如果你的结果和以前一模一样，那实际跑的是老路子（看面板「最近一次解算」）。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
