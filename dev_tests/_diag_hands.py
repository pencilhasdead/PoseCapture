"""排错用：逐关节核对“深度图说的远近”和“重建出来的前后”，重点是肩 / 手。

为什么要它：深度解算里每个关节的深度是 `y = β + α · n`（n = 该关节像素处的深度采样），
所以“深度图说手在前、重建却把手放到身后”只可能来自三处：

  1. 深度方向 α 的符号选错（前后整体反了）—— 骨长约束能不能分辨，看两次拟合的代价差；
  2. 采样到的深度本身不对（像素落错、3×3 中值混进背景、关键点被插值成了别的点）；
  3. 关键点根本没被用上：`place()` 只认 pts 字典，手部关键点在 hands 字典里（见输出末段）。

用法：
  blender.exe -b --factory-startup --python "dev_tests/_diag_hands.py"
  blender.exe -b --factory-startup --python "dev_tests/_diag_hands.py" -- <图片路径> [人物序号]
"""
import math
import os
import sys

import numpy as np
from mathutils import Vector

HERE = os.path.dirname(os.path.abspath(__file__))
ADDON_PARENT = os.path.dirname(HERE)
for path in (ADDON_PARENT, HERE):
    if path not in sys.path:
        sys.path.insert(0, path)

from pose_capture import (coco, deps, depth, dwpose, operators,  # noqa: E402
                          reconstruction, rigify_map, utils)

# 关节角色 -> 2D 关键点在 pts 里的名字
BODY_ROLES = [
    ("chest", "chest"),
    ("shoulder.L", "sh_L"), ("shoulder.R", "sh_R"),
    ("elbow.L", "el_L"), ("elbow.R", "el_R"),
    ("wrist.L", "wr_L"), ("wrist.R", "wr_R"),
    ("hip.L", "hip_L"), ("hip.R", "hip_R"),
    ("knee.L", "kn_L"), ("knee.R", "kn_R"),
    ("ankle.L", "an_L"), ("ankle.R", "an_R"),
]
DEFAULT_IMAGE = "test_coco.jpg"


def parse_args():
    argv = sys.argv
    rest = argv[argv.index("--") + 1:] if "--" in argv else []
    image = rest[0] if rest else os.path.join(utils.user_data_dir(), DEFAULT_IMAGE)
    person = int(rest[1]) if len(rest) > 1 else 0
    return image, person


def hand_pixel(keypoints, scores, side: str, local: int, thr=0.3):
    """某只手某个局部关键点的像素（置信度不够返回 None）。"""
    base = coco.HAND_LEFT_START if side == "L" else coco.HAND_RIGHT_START
    index = base + local
    if float(scores[index]) < thr:
        return None
    return np.array(keypoints[index], dtype=float)


def make_metrics():
    """Rigify 生成骨架的静止尺寸（与 test_onnx_end2end 用同一套）。"""
    import bpy
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.preferences.addon_enable(module="rigify")
    bpy.ops.object.armature_human_metarig_add()
    bpy.ops.object.mode_set(mode="POSE")
    bpy.ops.pose.rigify_generate()
    rig = bpy.context.object
    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.context.view_layer.update()
    return rigify_map.read_metrics(rig, rigify_map.auto_detect(rig, False))


def hand_tip_is_fallback(recon, metrics, side: str) -> bool:
    """重建出的 hand_tip 是不是“沿小臂外推手长”的兜底值（= 手部关键点没被用上）。"""
    joints = recon["joints"]
    wrist = np.asarray(joints["wrist.%s" % side], dtype=float)
    elbow = np.asarray(joints["elbow.%s" % side], dtype=float)
    tip = np.asarray(joints["hand_tip.%s" % side], dtype=float)
    vec = tip - wrist
    length, want = float(np.linalg.norm(vec)), float(metrics["hand_len"][side] or 0.08)
    if abs(length - want) > 1e-6 or length < 1e-9:
        return False
    fore = wrist - elbow
    norm = float(np.linalg.norm(fore))
    if norm < 1e-9:
        return False
    return float(np.dot(vec / length, fore / norm)) > 0.999999


def make_keypoint_box(keypoints, scores, thr=0.3):
    body = [point for point, score in zip(keypoints, scores) if float(score) > thr]
    return (min(float(p[0]) for p in body), min(float(p[1]) for p in body),
            max(float(p[0]) for p in body), max(float(p[1]) for p in body))


def main() -> int:
    image_path, person = parse_args()
    if not os.path.isfile(image_path):
        print("!! 找不到图片：%s" % image_path)
        return 1
    print("后端：", deps.backend_status())
    detector, pose_model = deps.model_paths()
    from pose_capture import props
    depth_path, depth_ready = props.resolve_depth_model_path()

    rgb = utils.load_image_rgb8(image_path)
    height, width = rgb.shape[:2]
    print("\n图片 %s  %dx%d，人物序号 %d"
          % (os.path.basename(image_path), width, height, person))

    result = dwpose.DwPoseEstimator(detector, pose_model, "auto", None).run(
        rgb, det_thr=0.3, person_index=person)
    if result.get("keypoints") is None:
        print("!! 没有检测到人体")
        return 1
    keypoints, scores = result["keypoints"], result["scores"]

    depth_map = depth_size = None
    if depth_ready:
        depth_map, depth_size = depth.DepthEstimator(depth_path).predict(rgb)
        print("深度图 %dx%d（模型输入 %s）"
              % (depth_map.shape[1], depth_map.shape[0], depth_size))

    metrics = make_metrics()
    recon = reconstruction.reconstruct(
        keypoints, scores, metrics, (width, height),
        reconstruction.default_options(),
        depth_map=depth_map, depth_size=depth_size)
    if not recon.get("ok"):
        print("!! 重建失败：%s" % recon.get("message"))
        return 1
    info = recon.get("depth") or {}
    print("\n[解算] 模式 %s；参数化 %s；α=%.4f；β=%.2f m；骨长误差 %.4f m"
          % (recon.get("mode"), info.get("transform"), float(info.get("alpha") or 0.0),
             float(info.get("beta") or 0.0), float(info.get("bone_error") or -1.0)))
    for note in recon.get("notes") or []:
        print("  提示：%s" % note)

    box = make_keypoint_box(keypoints, scores)
    direction = operators.depth_direction(depth_map, box, (width, height), recon)
    print("\n[深度方向] %s" % direction["text"])
    if direction.get("agree") is False:
        print("  ★ %s" % direction.get("warning"))
    if recon.get("mode") != "DEPTH":
        print("（不是深度解算，逐关节核对用不上）")
        _report_hands(recon, metrics, keypoints, scores, depth_map, depth_size,
                      (width, height))
        return 0

    # ---------------- 逐关节：深度采样 vs 重建 Y ----------------
    alpha = float(info.get("alpha") or 0.0)
    near_large = bool(direction["near_is_large"])      # 预览取色用的物理方向
    pts, joints = recon["points2d"], recon["joints"]
    y_pelvis = float(joints["pelvis"][1])
    pixels = [pts.get(key) for _role, key in BODY_ROLES]
    sampled = depth.sample_many(depth_map, depth_size, (width, height),
                                [pts.get("pelvis")] + pixels)
    raw_pelvis = float(sampled[0])
    print("\n[逐关节] Y 越小 = 离镜头越近；深度差 = 该点深度采样 - 骨盆深度采样")
    print("  %-11s %-15s %10s %10s %9s %9s  %s"
          % ("关节", "像素", "深度采样", "深度差", "Y", "Y-骨盆Y", "模型说/重建说"))
    disagree = []
    for (role, _key), pixel, value in zip(BODY_ROLES, pixels, sampled[1:]):
        if pixel is None:
            print("  %-11s （没有关键点）" % role)
            continue
        diff = float(value) - raw_pelvis
        y = float(joints[role][1])
        by_model = "近" if (diff > 0.0) == near_large else "远"
        by_recon = "近" if (y - y_pelvis) < 0.0 else "远"
        if by_model != by_recon:
            disagree.append(role)
        print("  %-11s (%5.0f,%5.0f) %10.4f %10.4f %9.4f %9.4f  %s / %s%s"
              % (role, pixel[0], pixel[1], float(value), diff, y, y - y_pelvis,
                 by_model, by_recon, "   ★ 不一致" if by_model != by_recon else ""))
    print("  模型说 = 深度采样差 + 物理方向（深度预览“近=蓝”的判据）；"
          "重建说 = 拟合出的 α 方向（α=%.4f）" % alpha)
    if disagree:
        print("  ★ 有 %d 个关节两边对不上：%s" % (len(disagree), ", ".join(disagree)))
    else:
        print("  ✓ 所有关节两边一致（重建确实按深度图的远近摆的）")
    print("  注：骨长回修只保留“谁在前”的符号、按骨架骨长改深度量级，"
          "所以量级对不上是正常的（下面按段核对符号）")

    _report_signs(recon, info)
    _report_hands(recon, metrics, keypoints, scores, depth_map, depth_size, (width, height))

    # ---------------- 深度方向的歧义程度 ----------------
    flipped = reconstruction.reconstruct(
        keypoints, scores, metrics, (width, height),
        reconstruction.default_options(depth_flip=True),
        depth_map=depth_map, depth_size=depth_size)
    flipped_info = flipped.get("depth") or {}
    cost, cost_flip = float(info.get("cost") or 0.0), float(flipped_info.get("cost") or 0.0)
    ratio = (cost_flip / cost) if cost > 1e-12 else float("nan")
    print("\n[方向歧义] 正常 α=%.4f 代价=%.6f；反转 α=%.4f 代价=%.6f；比值 %.3f×"
          % (alpha, cost, float(flipped_info.get("alpha") or 0.0), cost_flip, ratio))
    if ratio < 1.05:
        print("  ★ 两种方向代价几乎一样：骨长分不出前后，方向等于靠运气"
              "（物理判据才是可靠的那一个）")
    else:
        print("  两种方向代价差得开，方向是可辨识的")
    print("  反转后手腕 Y：L %.4f  R %.4f（骨盆 %.4f，Y 越小越近）"
          % (float(flipped["joints"]["wrist.L"][1]),
             float(flipped["joints"]["wrist.R"][1]),
             float(flipped["joints"]["pelvis"][1])))
    return 0


def _report_signs(recon, info):
    """按段核对“谁在前”：深度图给的符号 vs 最终姿态里的符号（回修后应该仍然一致）。"""
    signs = info.get("signs") or {}
    if not signs:
        return
    joints = recon["joints"]
    names = {"pelvis": "pelvis", "chest": "chest", "sh_L": "shoulder.L", "sh_R": "shoulder.R",
             "el_L": "elbow.L", "el_R": "elbow.R", "wr_L": "wrist.L", "wr_R": "wrist.R",
             "hip_L": "hip.L", "hip_R": "hip.R", "kn_L": "knee.L", "kn_R": "knee.R",
             "an_L": "ankle.L", "an_R": "ankle.R", "big_L": "ball.L", "big_R": "ball.R",
             "mcp_L": "hand_tip.L", "mcp_R": "hand_tip.R"}
    print("\n[段符号] 子关节是否比父关节更靠近镜头（回修后应保持深度图给的符号）")
    bad = []
    for (point_a, point_b), wanted in sorted(signs.items()):
        head = joints.get(names.get(point_a, point_a))
        tail = joints.get(names.get(point_b, point_b))
        if head is None or tail is None:
            continue
        got = float(np.asarray(tail, dtype=float)[1]) < float(np.asarray(head, dtype=float)[1])
        delta = float(np.asarray(tail, dtype=float)[1] - np.asarray(head, dtype=float)[1])
        if abs(delta) <= 1e-3:
            word, flag = "同深度", "—（深度图没意见 / 画面内就超过骨长，修不了）"
        else:
            word = "在前" if got else "在后"
            flag = ("一致" if wanted is None or got == bool(wanted)
                    else "★ 不一致")
        if wanted is not None and abs(delta) > 1e-3 and got != bool(wanted):
            bad.append("%s→%s" % (point_a, point_b))
        print("  %-18s 深度图说 %-4s / 最终 %-4s  %s"
              % ("%s→%s" % (point_a, point_b),
                 "没意见" if wanted is None else ("在前" if wanted else "在后"),
                 word, flag))
    print("  手在身体前后：%s"
          % "；".join("%s 腕 %s 胸（%+.3f m）"
                      % (side,
                         "在前" if float(joints["wrist.%s" % side][1])
                         < float(joints["chest"][1]) else "在后",
                         float(joints["wrist.%s" % side][1]) - float(joints["chest"][1]))
                      for side in ("L", "R")))
    if bad:
        print("  ★ 有符号对不上的段：%s" % ", ".join(bad))


def _report_hands(recon, metrics, keypoints, scores, depth_map, depth_size, image_size):
    """手部：关键点有没有被用上 / 深度模型怎么看手指 / fingers 有没有生成。"""
    print("\n[手部]")
    for side in ("L", "R"):
        fallback = hand_tip_is_fallback(recon, metrics, side)
        print("  %s hand_tip：%s" % (side, "沿小臂外推的兜底值（手部关键点没参与）"
                                     if fallback else "用了关键点"))
    pixels, labels = [], []
    for side in ("L", "R"):
        for local, name in ((coco.HAND_WRIST, "腕"), (coco.HAND_MIDDLE_MCP, "指根")):
            pixels.append(hand_pixel(keypoints, scores, side, local))
            labels.append("%s%s" % (side, name))
    if depth_map is not None:
        values = depth.sample_many(depth_map, depth_size, image_size, pixels)
        parts = []
        for label, pixel, value in zip(labels, pixels, values):
            if pixel is None:
                parts.append("%s 缺" % label)
                continue
            parts.append("%s 采样 %.3f%s" % (label, float(value),
                                            "" if label.endswith("腕") else "（没用到）"))
        print("  手部关键点深度：%s" % "；".join(parts))
    print("  fingers：%s（%s）"
          % (sorted((recon.get("fingers") or {}).keys()),
             "没生成" if not recon.get("fingers") else "已生成"))
    joints = recon["joints"]
    for side in ("L", "R"):
        hand = np.asarray(joints["hand_tip.%s" % side], dtype=float) \
            - np.asarray(joints["wrist.%s" % side], dtype=float)
        fore = np.asarray(joints["wrist.%s" % side], dtype=float) \
            - np.asarray(joints["elbow.%s" % side], dtype=float)
        span = float(np.linalg.norm(hand))
        if span < 1e-9 or float(np.linalg.norm(fore)) < 1e-9:
            continue
        print("  %s 手 - 小臂夹角 %.1f 度（手长 %.3f m / 骨架 %.3f m；"
              "`手部沿小臂` 越大这个角越小，上限 %.0f 度）"
              % (side, math.degrees(Vector(hand).angle(Vector(fore))), span,
                 float(metrics["hand_len"][side] or 0.0),
                 reconstruction.HAND_ALIGN_MAX_DEG))


if __name__ == "__main__":
    sys.exit(main())
