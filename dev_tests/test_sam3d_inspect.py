"""真实 SAM 3D Body 结果 -> 插件的数值检查（纯 numpy，不需要 Blender / torch）。

用法：
  python dev_tests/test_sam3d_inspect.py D:\\code\\sam3d_out\\sam3d_0.npz
"""
import importlib.util
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
REAL = os.path.join(ROOT, "pose_capture")


def load_module(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(REAL, name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def bone(a, b):
    return float(np.linalg.norm(np.asarray(b, float) - np.asarray(a, float)))


def main():
    if len(sys.argv) < 2 or not os.path.isfile(sys.argv[1]):
        print("用法：python dev_tests/test_sam3d_inspect.py <npz>")
        return 2
    path = sys.argv[1]
    sam3d = load_module("sam3d")
    data = sam3d.load_npz(path)
    print("npz：%s" % path)
    print("  字段：%s" % ", ".join(sorted(k for k in data if k != "meta")))
    k3 = np.asarray(data["keypoints_3d"])[:70, :3]
    print("  keypoints_3d %s，每轴范围 %s .. %s"
          % (k3.shape, np.round(k3.min(0), 2).tolist(), np.round(k3.max(0), 2).tolist()))

    print("  focal=%s image_size=%s cam_t=%s"
          % (np.asarray(data.get("focal_length")).reshape(-1)[:1],
             np.asarray(data.get("image_size")).tolist(),
             np.round(np.asarray(data.get("cam_t", [0, 0, 0])).reshape(-1)[:3], 3).tolist()))

    recon = sam3d.build_recon(data, {"fingers": True})
    print("\nrecon：mode=%s yaw=%.1f° 重投影=%.2f px"
          % (recon["mode"], recon["yaw_deg"], recon["sam3d"]["reproj_error"]))
    for note in recon["notes"]:
        print("  - %s" % note)

    j = recon["joints"]
    print("\n骨长（米，模型预测，可与真实人体对照）：")
    rows = (("肩宽", "shoulder.R", "shoulder.L", 0.35),
            ("大臂 L", "shoulder.L", "elbow.L", 0.30),
            ("小臂 L", "elbow.L", "wrist.L", 0.25),
            ("大腿 L", "hip.L", "knee.L", 0.42),
            ("小腿 L", "knee.L", "ankle.L", 0.42),
            ("躯干", "pelvis", "chest", 0.50),
            ("头", "head_base", "head_tip", 0.12))
    for label, a, b, expect in rows:
        value = bone(j[a], j[b])
        flag = "" if 0.4 * expect <= value <= 2.0 * expect else "   <-- 可疑"
        print("  %-6s %-22s %.3f m（量级参考 %.2f）%s" % (label, "%s -> %s" % (a, b), value,
                                                          expect, flag))

    print("\n左右对称性（模型自洽性检查，两侧应该接近）：")
    for label, a, b in (("大臂", "shoulder.%s", "elbow.%s"), ("小臂", "elbow.%s", "wrist.%s"),
                        ("大腿", "hip.%s", "knee.%s"), ("小腿", "knee.%s", "ankle.%s")):
        left = bone(j[a % "L"], j[b % "L"])
        right = bone(j[a % "R"], j[b % "R"])
        print("  %-4s L=%.3f R=%.3f  差 %.1f%%"
              % (label, left, right, abs(left - right) / max(left, right, 1e-6) * 100))

    print("\n朝向 / 高度：")
    print("  朝向 yaw = %.1f°（角色已转到面朝 -Y）" % recon["yaw_deg"])
    print("  骨盆 z = %.3f，最低脚点 z = %.3f"
          % (j["pelvis"][2], min(j["ankle.L"][2], j["ankle.R"][2],
                                 j["ball.L"][2], j["ball.R"][2])))
    print("  手：L 腕 %s  指尖 %s" % (np.round(j["wrist.L"], 2), np.round(j["hand_tip.L"], 2)))

    fingers = recon.get("fingers") or {}
    count = sum(len(per) for per in fingers.values())
    lengths = [float(np.linalg.norm(np.asarray(chain[3], float) - np.asarray(chain[0], float)))
               for per in fingers.values() for chain in per.values()]
    print("\n手指：%d 条链，%d 个点；指长 %.3f ~ %.3f m"
          % (count, sum(len(c) for per in fingers.values() for c in per.values()),
             min(lengths) if lengths else 0.0, max(lengths) if lengths else 0.0))

    missing = [key for key in ("pelvis", "chest", "head_base", "head_tip", "neck_start",
                               "shoulder_origin.L", "shoulder.L", "elbow.L", "wrist.L",
                               "hand_tip.L", "hip.L", "knee.L", "ankle.L", "ball.L")
               if key not in j]
    refs = recon["refs"]
    print("\n结构：joints %d 项，refs %s，缺 %s"
          % (len(j), sorted(refs), missing or "无"))

    bad = [key for key, value in j.items() if not np.all(np.isfinite(value))]
    print("非有限值：%s" % (bad or "无"))
    return 0 if not missing and not bad else 1


if __name__ == "__main__":
    sys.exit(main())
