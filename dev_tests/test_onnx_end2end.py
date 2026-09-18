"""真实模型端到端自检：图片 -> DWPose(ONNX) -> 2D->3D -> 写入 Rigify 生成骨架。

前置：先跑 dev_tests/_setup_deps.py 装好 onnxruntime 并下载模型。
用法：
  blender.exe -b --factory-startup --python "dev_tests/test_onnx_end2end.py"
"""
import math
import os
import sys
import urllib.request

import numpy as np

import bpy
from mathutils import Matrix, Vector

HERE = os.path.dirname(os.path.abspath(__file__))
ADDON_PARENT = os.path.dirname(HERE)
for path in (ADDON_PARENT, HERE):
    if path not in sys.path:
        sys.path.insert(0, path)

from test_pipeline import ensure_registered  # noqa: E402

from pose_capture import (coco, deps, dwpose, reconstruction, retarget,  # noqa: E402
                          rigify_limbs, rigify_map, utils)

IMAGE_URL = ("https://raw.githubusercontent.com/open-mmlab/mmpose/main/"
             "tests/data/coco/000000000785.jpg")

PROJECTION_PAIRS = [
    (5, "DEF-upper_arm.L"), (7, "DEF-forearm.L"), (9, "DEF-hand.L"),
    (6, "DEF-upper_arm.R"), (8, "DEF-forearm.R"), (10, "DEF-hand.R"),
    (11, "DEF-thigh.L"), (13, "DEF-shin.L"), (15, "DEF-foot.L"), (17, "DEF-toe.L"),
    (12, "DEF-thigh.R"), (14, "DEF-shin.R"), (16, "DEF-foot.R"), (20, "DEF-toe.R"),
]


def fetch_image(path):
    if os.path.isfile(path) and os.path.getsize(path) > 1024:
        return path
    print("下载测试图片 %s" % IMAGE_URL)
    request = urllib.request.Request(IMAGE_URL, headers={"User-Agent": "pose-capture-test"})
    with urllib.request.urlopen(request, timeout=120) as response, open(path, "wb") as handle:
        handle.write(response.read())
    return path


def setup_generated_rig():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.preferences.addon_enable(module="rigify")
    bpy.ops.object.armature_human_metarig_add()
    bpy.ops.object.mode_set(mode="POSE")
    bpy.ops.pose.rigify_generate()
    rig = bpy.context.object
    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.context.view_layer.update()
    return rig


def world_point(rig, name):
    return rig.matrix_world @ rig.pose.bones[name].head


def check_depth_solve(rgb, keypoints, scores, rig, bmap, metrics, image_size) -> bool:
    """真实照片：深度模型 + 透视全局拟合，并走一遍 apply_pose 操作器。"""
    from pose_capture import depth, operators, props

    path, ready = props.resolve_depth_model_path()
    if not ready:
        print("\n[深度模型] 未就绪（跳过）：%s" % path)
        return True
    print("\n[深度模型] %s" % os.path.basename(path))
    estimator = depth.DepthEstimator(path)
    depth_map, depth_size = estimator.predict(rgb)
    print("  深度图 %dx%d（模型输入 %s）" % (depth_map.shape[1], depth_map.shape[0],
                                            depth_size))

    recon = reconstruction.reconstruct(keypoints, scores, metrics, image_size,
                                       reconstruction.default_options(),
                                       depth_map=depth_map, depth_size=depth_size)
    info = recon.get("depth") or {}
    print("  解算模式 %s；参数化 %s；α=%.3f；β=%.2f m；骨长误差：深度 %.4f / 骨长 %.4f m"
          % (recon.get("mode"), info.get("transform"), float(info.get("alpha") or 0.0),
             float(info.get("beta") or 0.0), float(info.get("bone_error") or -1.0),
             float(info.get("bone_error_fallback") or -1.0)))
    for note in recon.get("notes") or []:
        print("  提示：%s" % note)
    if recon.get("mode") != "DEPTH":
        print("!! 深度解算没有被采用")
        return False

    names = ("pelvis", "chest", "shoulder.L", "shoulder.R", "elbow.L", "elbow.R",
             "wrist.L", "wrist.R", "hip.L", "hip.R", "knee.L", "knee.R",
             "ankle.L", "ankle.R")
    depths = {name: float(recon["joints"][name][1]) for name in names}
    print("  关节深度（米）：%s" % {k: round(v, 3) for k, v in depths.items()})
    span = max(depths.values()) - min(depths.values())
    print("  深度跨度 %.3f m" % span)

    # 真实照片上核对深度方向：物理判据（人比背景近）必须和拟合出来的方向一致，
    # 而且“离镜头最近的鼻子”在伪彩里要落在蓝的那一端 —— 反过来就是前后整反了
    body = [point for point, score in zip(keypoints, scores) if float(score) > 0.3]
    box = (min(float(p[0]) for p in body), min(float(p[1]) for p in body),
           max(float(p[0]) for p in body), max(float(p[1]) for p in body))
    direction = operators.depth_direction(depth_map, box, image_size, recon)
    print("  %s" % direction["text"])
    scale_x = depth_map.shape[1] / float(image_size[0])
    scale_y = depth_map.shape[0] / float(image_size[1])
    color = depth.colorize(depth_map, direction["near_is_large"])
    nose = color[int(float(keypoints[0][1]) * scale_y), int(float(keypoints[0][0]) * scale_x)]
    print("  鼻子像素（伪彩，近=蓝 远=红）RGB = %s" % nose.tolist())
    ok_direction = True
    if not direction["agree"]:
        print("!! %s" % direction["warning"])
        ok_direction = False
    if int(nose[2]) <= int(nose[0]):
        print("!! 鼻子（离镜头最近的部位）不是偏蓝的：近远方向可能反了")
        ok_direction = False

    # 操作器路径：让 apply_pose 也用深度图
    operators._RESULT["depth_map"] = depth_map
    operators._RESULT["depth_size"] = depth_size
    result_op = bpy.ops.pose_capture.apply_pose()
    print("  apply_pose：%s | %s" % (result_op, bpy.context.scene.pose_capture.depth_info))
    if result_op != {"FINISHED"}:
        return False

    worst_angle = 0.0
    for role, (p_from, p_to, _ref) in retarget.build_targets(recon, bmap, metrics).items():
        name = bmap.get(role)
        if not name or role in ("hips", "palm.L", "palm.R"):
            continue
        pose_bone = rig.pose.bones.get(name)
        if pose_bone is None:
            continue
        actual = (rig.matrix_world.to_3x3()
                  @ (pose_bone.tail - pose_bone.head)).normalized()
        expected = (Vector(p_to) - Vector(p_from)).normalized()
        worst_angle = max(worst_angle, math.degrees(actual.angle(expected)))
    print("  写入一致性（深度解算）：最大 %.2f 度" % worst_angle)
    bpy.ops.pose_capture.reset_pose()
    return worst_angle < 8.0 and span > 0.05 and ok_direction


def main():
    ensure_registered()
    status = deps.backend_status()
    print("后端:", status)
    if not (status["onnxruntime"] or status["opencv"]):
        print("!! 没有推理后端，请先运行 dev_tests/_setup_deps.py")
        return 1

    detector, pose_model = deps.model_paths()
    if not (os.path.isfile(detector) and os.path.isfile(pose_model)):
        print("!! 缺少模型，请先运行 dev_tests/_setup_deps.py")
        return 1

    image_path = fetch_image(os.path.join(utils.user_data_dir(), "test_coco.jpg"))
    rgb = utils.load_image_rgb8(image_path)
    height, width = rgb.shape[:2]
    print("图片 %s  %dx%d" % (os.path.basename(image_path), width, height))

    estimator = dwpose.DwPoseEstimator(detector, pose_model, "auto", None)
    result = estimator.run(rgb, det_thr=0.3, person_index=0)
    if result.get("keypoints") is None:
        print("!! 没有检测到人体")
        return 1
    keypoints = result["keypoints"]
    scores = result["scores"]
    print("检测到 %d 个人体框：%s" % (len(result["boxes"]),
                                     np.round(result["boxes"], 1).tolist()))
    print("身体 17 点置信度：%s" % np.round(scores[:17], 2))
    print("脚部 6 点置信度：%s" % np.round(scores[17:23], 2))
    print("人脸均值 %.2f，左手 %.2f，右手 %.2f" % (
        scores[coco.FACE_ALL].mean(),
        scores[coco.HAND_LEFT_START:coco.HAND_LEFT_START + coco.HAND_COUNT].mean(),
        scores[coco.HAND_RIGHT_START:coco.HAND_RIGHT_START + coco.HAND_COUNT].mean()))

    def kp(index):
        return np.array(keypoints[index], dtype=float)

    names = coco.KEYPOINT_NAMES
    for index in (0, 5, 6, 11, 12, 15, 16):
        print("  %-14s %s  置信度 %.2f" % (names[index], np.round(kp(index), 1),
                                           scores[index]))

    sanity = [
        ("鼻子在双肩之上", kp(0)[1] < kp(5)[1] - 5 and kp(0)[1] < kp(6)[1] - 5),
        ("肩膀在胯之上", kp(5)[1] < kp(11)[1] and kp(6)[1] < kp(12)[1]),
        ("胯在脚踝之上", kp(11)[1] < kp(15)[1] and kp(12)[1] < kp(16)[1]),
        ("左右肩横向分开", abs(kp(5)[0] - kp(6)[0]) > 15),
        ("左肩置信度 > 0.3", scores[5] > 0.3),
    ]
    ok_sanity = True
    for label, passed in sanity:
        print("  %s %s" % ("OK " if passed else "!! ", label))
        ok_sanity = ok_sanity and bool(passed)

    rig = setup_generated_rig()
    bmap = rigify_map.auto_detect(rig, False)
    metrics = rigify_map.read_metrics(rig, bmap)
    if metrics.get("errors"):
        print("!! 骨架读取失败：", metrics["errors"])
        return 1
    # 写入用“按肢体写入方式匹配”的那份映射：默认 IK ⇒ 肢体匹配 IK 控制器 ⇒ 直写路径。
    # metrics 还是用 FK 链量的（重建 / 严格校验都按它算，跟以前一致）。
    bmap_ik = rigify_map.auto_detect(rig, False, "IK")
    print("IK 模式映射：", {role: bmap_ik.get(role) for role in
                            ("upper_arm.L", "forearm.L", "hand.L", "thigh.R", "foot.R")})

    options = reconstruction.default_options()
    recon = reconstruction.reconstruct(keypoints, scores, metrics, (width, height),
                                       options)
    if not recon.get("ok"):
        print("!! 重建失败：", recon.get("message"))
        return 1
    joints = recon["joints"]
    print("比例 %.1f 像素/米；yaw %.1f 度；肩宽 %.2f 米；头顶到脚约 %.2f 米" % (
        recon["scale"], recon["yaw_deg"],
        float(np.linalg.norm(joints["shoulder.L"] - joints["shoulder.R"])),
        float(joints["head_tip"][2] - min(joints["ankle.L"][2], joints["ankle.R"][2])) + 0.2))
    print("重建关节（相机系，米）：")
    for key in ("head_base", "shoulder.L", "elbow.L", "wrist.L", "hip.L", "knee.L",
                "ankle.L", "chest", "pelvis"):
        print("   %-11s %s" % (key, tuple(np.round(joints[key], 3))))

    report = retarget.apply_pose(rig, bmap_ik, metrics, recon,
                                 {"twist": True, "twist_blend": 1.0, "limb_mode": "IK",
                                  "root_motion": False})
    print("写入：%s" % report["message"])
    print("  肢体模式 %s；直写 IK 控制器 %s；IK 快照 %s；显示同步 %d 个集合；跳过 %d 项"
          % (report.get("limb_mode"), report.get("ik_direct"), report.get("ik_snapped"),
             len(report.get("visibility") or []), len(report["skipped"])))
    modes = rigify_limbs.limb_modes(rig)
    print("  IK_FK 滑块：%s" % {key: round(value, 3) for key, value in sorted(modes.items())})
    if report.get("limb_mode") != "IK" \
            or not (report.get("ik_direct") or report.get("ik_snapped")):
        print("!! 没有走 IK 控制器写入")
        return 1
    if any(abs(value) > 1e-6 for value in modes.values()):
        print("!! 四条肢体没有全部停在 IK 模式")
        return 1

    # IK 模式的核心诉求：控制器还能继续拖
    hand = rig.pose.bones["hand_ik.L"]
    before = world_point(rig, "DEF-hand.L")
    hand.matrix = Matrix.Translation(Vector((0.0, 0.0, 0.12))) @ hand.matrix
    bpy.context.view_layer.update()
    moved = (world_point(rig, "DEF-hand.L") - before).length
    hand.matrix = Matrix.Translation(Vector((0.0, 0.0, -0.12))) @ hand.matrix
    bpy.context.view_layer.update()
    print("  移动 hand_ik.L 0.12 m -> DEF-hand.L 移动 %.4f m" % moved)
    if abs(moved - 0.12) > 0.02:
        print("!! IK 控制器拖不动变形骨")
        return 1
    print("  （FK 骨骼仍保留目标姿态，方向校验见下）")

    # 严格校验：写入后骨架的每个骨骼方向是否等于重建给出的目标方向
    from pose_capture import operators

    targets = retarget.build_targets(recon, bmap, metrics)
    mismatches = []
    for role, (p_from, p_to, _ref) in targets.items():
        name = bmap.get(role)
        if not name or role in ("hips", "palm.L", "palm.R"):
            continue
        pose_bone = rig.pose.bones.get(name)
        if pose_bone is None:
            continue
        actual = (rig.matrix_world.to_3x3()
                  @ (pose_bone.tail - pose_bone.head)).normalized()
        expected = (Vector(p_to) - Vector(p_from)).normalized()
        angle = math.degrees(actual.angle(expected))
        mismatches.append((role, angle))
    worst = max(mismatches, key=lambda item: item[1])
    mean_angle = sum(item[1] for item in mismatches) / max(len(mismatches), 1)
    print("写入一致性：%d 根骨骼，平均 %.2f 度，最大 %.2f 度（%s）"
          % (len(mismatches), mean_angle, worst[1], worst[0]))

    # 信息性：骨架 vs 关键点的方向/位置差异（受骨架比例影响，仅参考）
    hips_rig = (world_point(rig, "DEF-thigh.L") + world_point(rig, "DEF-thigh.R")) * 0.5
    hips_kp = (kp(11) + kp(12)) * 0.5
    to_camera = Matrix.Rotation(math.radians(recon["yaw_deg"]), 3, "Z")

    def rig_2d(name):
        offset = to_camera @ (world_point(rig, name) - hips_rig)
        return np.array((offset.x, -offset.z))

    def kp_2d(index):
        point = kp(index) - hips_kp
        return np.array((point[0], point[1]))

    segments = [
        ("左大腿", "DEF-thigh.L", "DEF-shin.L", 11, 13),
        ("左小腿", "DEF-shin.L", "DEF-foot.L", 13, 15),
        ("右大腿", "DEF-thigh.R", "DEF-shin.R", 12, 14),
        ("右小腿", "DEF-shin.R", "DEF-foot.R", 14, 16),
        ("左大臂", "DEF-upper_arm.L", "DEF-forearm.L", 5, 7),
        ("左小臂", "DEF-forearm.L", "DEF-hand.L", 7, 9),
        ("右大臂", "DEF-upper_arm.R", "DEF-forearm.R", 6, 8),
        ("右小臂", "DEF-forearm.R", "DEF-hand.R", 8, 10),
    ]
    print("参考：各段在画面内的方向差异（段越短越受骨架比例影响）")
    for label, bone_a, bone_b, kp_a, kp_b in segments:
        rig_dir = rig_2d(bone_b) - rig_2d(bone_a)
        kp_dir = kp_2d(kp_b) - kp_2d(kp_a)
        norm = float(np.linalg.norm(rig_dir) * np.linalg.norm(kp_dir))
        if norm < 1e-9:
            continue
        cosine = float(np.clip(np.dot(rig_dir, kp_dir) / norm, -1.0, 1.0))
        print("  %-8s %5.1f 度  （检测段长 %.0f 像素）"
              % (label, math.degrees(math.acos(cosine)), float(np.linalg.norm(kp_dir))))

    # 操作器层自检（等同于面板点击）
    settings = bpy.context.scene.pose_capture
    settings.armature = rig
    settings.image_source = "FILE"
    settings.image_filepath = image_path
    operators._RESULT.update({"keypoints": keypoints, "scores": scores,
                              "image_size": (width, height),
                              "rgb": rgb, "boxes": result["boxes"]})
    settings.detected = True
    bpy.ops.pose_capture.auto_map()
    print("映射表行数 %d，map_ready=%s" % (len(settings.bone_map), settings.map_ready))
    result_op = bpy.ops.pose_capture.apply_pose()
    print("apply_pose 操作器:", result_op, settings.detect_info[:80])
    preview = operators.update_preview_image("PoseCapture_Preview", rgb, keypoints,
                                             scores, 0.3)
    image = bpy.data.images.get(preview)
    print("预览图：%s %dx%d" % (preview, image.size[0], image.size[1]))
    bpy.ops.pose_capture.reset_pose()

    ok_depth = check_depth_solve(rgb, keypoints, scores, rig, bmap, metrics,
                                 (width, height))
    ok = (ok_sanity and worst[1] < 8.0 and mean_angle < 3.0
          and len(mismatches) > 15 and ok_depth)
    print("结果：", "通过 ✓" if ok else "未通过 ✗")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

