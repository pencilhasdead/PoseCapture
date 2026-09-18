"""深度预览自检：图像写入（黑图 bug 回归）+ 独立 Tk 预览页面用的数据。

背景：Blender 5.2 里给 GENERATED 图像**写完像素之后**再赋 colorspace 会把缓冲清空，
       之前“深度图 / 骨架预览一直是纯黑”就是这个原因。这里把写入结果回读校验。

验证：
  1. operators._write_image 写进去的画布，回读后与原画布一致（不是全黑）；
  2. 存出来的 PNG 用 Blender 重新加载后同样不是全黑，尺寸正确；
  3. build_depth_view 生成 3 张 PNG + stats.json（有深度模型就用真实模型，没有就用合成深度图）；
  4. depth_view 这个独立页面的读取 / 文本格式化能正常工作；
  5. 报告系统里有没有带 tkinter 的 Python（找不到不算失败）。

用法：
  blender.exe -b --factory-startup --python "dev_tests/test_depth_view.py"
"""
import json
import os
import sys
import tempfile

import numpy as np

import bpy

HERE = os.path.dirname(os.path.abspath(__file__))
ADDON_PARENT = os.path.dirname(HERE)
for path in (ADDON_PARENT, HERE):
    if path not in sys.path:
        sys.path.insert(0, path)

from test_pipeline import ensure_registered  # noqa: E402

from pose_capture import depth, depth_view, operators, props  # noqa: E402

OUT_DIR = os.path.join(tempfile.gettempdir(), "pose_capture_depth_view_test")
FAILURES: list[str] = []


def check(result, label, detail=""):
    print("  %s%s %s" % ("OK " if result else "!! ", label, detail))
    if not result:
        FAILURES.append(label)
    return result


def gradient_canvas(height=180, width=260) -> np.ndarray:
    x = np.linspace(0, 255, width, dtype=np.float32)[None, :]
    y = np.linspace(0, 255, height, dtype=np.float32)[:, None]
    canvas = np.zeros((height, width, 3), dtype=np.float32)
    canvas[:, :, 0] = x
    canvas[:, :, 1] = y
    canvas[:, :, 2] = 128.0
    return canvas.astype(np.uint8)


def synthetic_depth(height=320, width=480) -> np.ndarray:
    """合成深度图：左边近（值大）右边远，用来在没有模型时也能验证预览链路。"""
    x = np.linspace(3.0, 0.5, width, dtype=np.float32)[None, :]
    return np.repeat(x, height, axis=0)


def read_image_rgb(name: str) -> np.ndarray:
    image = bpy.data.images[name]
    buf = np.empty(len(image.pixels), dtype=np.float32)
    image.pixels.foreach_get(buf)
    return buf.reshape(-1, 4)[:, :3]


def test_write_image() -> None:
    print("[1] Image 像素写入（黑图回归）")
    canvas = gradient_canvas()
    name = operators._write_image("PC_TEST_VIEW", canvas)
    back = read_image_rgb(name)
    expected = canvas.reshape(-1, 3).astype(np.float32) / 255.0
    shape = (canvas.shape[0], canvas.shape[1], 3)
    back_flipped = back.reshape(shape)[::-1].reshape(-1, 3)
    worst = float(np.abs(back_flipped - expected).max()) if back.shape == expected.shape \
        else 1.0
    print("    回读 mean=%.4f max=%.4f（期望 mean=%.4f）"
          % (back.mean(), back.max(), expected.mean()))
    check(back.mean() > 0.01, "写进去的图不是全黑", "mean=%.4f" % back.mean())
    check(worst < 0.02, "像素值与画布一致（按 Blender 的上下翻转比对）", "max diff=%.4f" % worst)


def test_save_png() -> None:
    print("[2] PNG 落盘 + 重新加载")
    os.makedirs(OUT_DIR, exist_ok=True)
    canvas = gradient_canvas()
    path = os.path.join(OUT_DIR, "gradient.png")
    operators._save_canvas_png(path, canvas)

    image = None
    check(os.path.isfile(path), "PNG 已写出", "%d 字节" % os.path.getsize(path))
    try:
        image = bpy.data.images.load(path, check_existing=False)
        buf = np.empty(len(image.pixels), dtype=np.float32)
        image.pixels.foreach_get(buf)
        rgb = buf.reshape(-1, 4)[:, :3]
        print("    重新加载 mean=%.4f max=%.4f size=%s"
              % (rgb.mean(), rgb.max(), tuple(image.size)))
        check(rgb.mean() > 0.01, "磁盘上的 PNG 不是全黑", "mean=%.4f" % rgb.mean())
        check(tuple(image.size) == (canvas.shape[1], canvas.shape[0]),
              "PNG 尺寸正确", str(tuple(image.size)))
    finally:
        if image is not None:
            bpy.data.images.remove(image)


def test_depth_view_files() -> None:
    print("[3] 生成预览文件（PNG + stats.json）")
    rgb = gradient_canvas(320, 480)
    depth_map = synthetic_depth(320, 480)
    depth_size = (480, 320)
    path, ready = props.resolve_depth_model_path()
    if ready:
        estimator = depth.DepthEstimator(path)
        depth_map, depth_size = estimator.predict(rgb)
        print("    真实深度模型：%dx%d（模型输入 %s）"
              % (depth_map.shape[1], depth_map.shape[0], depth_size))
    else:
        print("    没下载深度模型（%s），用合成深度图验证链路" % path)

    operators._RESULT["rgb"] = rgb
    operators._RESULT["depth_map"] = depth_map
    operators._RESULT["depth_size"] = depth_size
    operators._RESULT["image_size"] = (rgb.shape[1], rgb.shape[0])
    operators._RESULT["path"] = "synthetic.png"
    operators._RESULT["recon"] = {"depth": {"near_is_large": True, "alpha": -0.07,
                                            "beta": 4.4}}

    settings = props.get_settings(bpy.context)
    manifest = operators.build_depth_view(settings, OUT_DIR)
    names = [os.path.basename(panel["path"]) for panel in manifest["panels"]]
    print("    生成：", names)
    check(len(manifest["panels"]) == 3, "三张图都生成了", str(names))
    check(all(os.path.isfile(panel["path"]) for panel in manifest["panels"]),
          "PNG 文件都在磁盘上")
    check(os.path.isfile(os.path.join(OUT_DIR, "stats.json")), "stats.json 已写出")

    gray = next(panel for panel in manifest["panels"] if "灰度" in panel["label"])
    stats = gray["stats"]
    print("    深度统计 min=%.4f max=%.4f mean=%.4f std=%.4f 灰度均值=%.2f"
          % (stats["min"], stats["max"], stats["mean"], stats["std"], stats["mean255"]))
    check(stats["max"] > stats["min"], "深度统计范围有效")
    check(stats["mean255"] > 2.0, "灰度图不是全黑", "mean255=%.2f" % stats["mean255"])
    check(not manifest["notes"], "没有异常提示", "；".join(manifest["notes"]))

    direction = manifest.get("direction") or {}
    print("    方向核对：%s（物理=%s 拟合=%s）"
          % (direction.get("text"), direction.get("physical"), direction.get("fit")))
    check(direction.get("text", "").startswith("深度方向"), "stats.json 里带方向核对")
    check(direction.get("agree") is True, "物理判据与拟合判据一致（合成数据）")
    check("越白 = 越远" in gray["label"], "灰度面板标题写明方向", gray["label"])
    color_panel = next(panel for panel in manifest["panels"] if "伪彩" in panel["label"])
    check("近=蓝" in color_panel["label"] and "远=红" in color_panel["label"],
          "伪彩面板标题写明配色", color_panel["label"])


def test_color_convention() -> None:
    """配色约定（近=蓝 远=红、灰度越白=越远）+ “物理 vs 拟合”方向对不上时的告警。"""
    print("[3b] 配色约定 + 深度方向核对")
    height, width = 200, 300
    rgb = gradient_canvas(height, width)
    # 左半边近（值大）、右半边远（值小）；人框放在左半边 → 物理判据“值大 = 近”
    row = np.linspace(3.0, 0.5, width, dtype=np.float32)[None, :]
    depth_map = np.repeat(row, height, axis=0)
    box = (0.0, 0.0, width / 2.0, float(height))
    check(operators.depth_orientation(depth_map, box, (width, height)) is True,
          "物理判据认出“值大 = 近”")

    color = depth.colorize(depth_map)
    gray_map = operators._normalize_gray(depth_map, near_is_large=True)
    near_rgb = color[:, 5].mean(axis=0)
    far_rgb = color[:, -6].mean(axis=0)
    print("    近处 RGB=%s（灰度 %.1f）；远处 RGB=%s（灰度 %.1f）"
          % (np.round(near_rgb, 1), gray_map[:, 5].mean(),
             np.round(far_rgb, 1), gray_map[:, -6].mean()))
    check(near_rgb[2] > near_rgb[0] + 100, "伪彩：近处是蓝的", str(np.round(near_rgb, 1)))
    check(far_rgb[0] > far_rgb[2] + 100, "伪彩：远处是红的", str(np.round(far_rgb, 1)))
    check(gray_map[:, 5].mean() < gray_map[:, -6].mean(),
          "灰度：越白 = 越远（近处更黑）")

    saved = dict(operators._RESULT)
    settings = props.get_settings(bpy.context)
    try:
        operators._RESULT.update({"rgb": rgb, "depth_map": depth_map,
                                  "depth_size": (width, height), "box": box,
                                  "image_size": (width, height), "path": "synthetic.png"})
        operators._RESULT["recon"] = {"depth": {"near_is_large": True}}
        agreed = operators.build_depth_view(settings, OUT_DIR)
        check(agreed["direction"]["agree"] is True and not agreed["notes"],
              "方向一致时不报警")

        # 拟合方向反过来（就是“近和远看着反了”的现场）→ 必须明确报出来
        operators._RESULT["recon"] = {"depth": {"near_is_large": False}}
        bad = operators.build_depth_view(settings, OUT_DIR)
        print("    告警：%s" % "；".join(bad["notes"]))
        check(bad["direction"]["agree"] is False and bool(bad["notes"]),
              "拟合方向和物理判据相反时写进 notes")
        text = depth_view.format_stats(bad)
        check("前后可能反了" in text, "预览页面的文本里也会提示方向对不上")

        # 拟合被判为不可信（正文退回骨长解算）时，不该拿它来质疑方向
        operators._RESULT["recon"] = {"depth": {"near_is_large": False, "rejected": True}}
        dropped = operators.build_depth_view(settings, OUT_DIR)
        check(dropped["direction"]["fit"] is None and not dropped["notes"],
              "拟合不可信时不做方向告警", dropped["direction"]["text"])
    finally:
        operators._RESULT.clear()
        operators._RESULT.update(saved)


def test_viewer_page() -> None:
    print("[4] 独立预览页面的读取 / 文本")
    manifest = depth_view.load_manifest(OUT_DIR)
    text = depth_view.format_stats(manifest)
    lines = text.splitlines()
    print("    格式化文本 %d 行" % len(lines))
    for line in lines[:6]:
        print("      %s" % line)
    check("输入照片" in text, "文本里有面板标题")
    check(any("百分位" in line for line in lines), "文本里有百分位统计")
    check(any("方向核对" in line for line in lines), "文本里有深度方向核对")
    with open(os.path.join(OUT_DIR, "stats.json"), "r", encoding="utf-8") as handle:
        check(json.load(handle).get("panels") is not None, "stats.json 能被 json 读取")


def test_python_lookup() -> None:
    print("[5] 找带 tkinter 的 Python（找不到不算失败）")
    candidates = depth_view.candidate_pythons()
    found = depth_view.find_python()
    print("    候选 %d 个；可用：%s" % (len(candidates), found or "无"))
    for item in candidates[:6]:
        print("      %s -> tkinter %s" % (item, depth_view._can_import_tkinter(item)))
    if found is None:
        print("    提示：没有带 tkinter 的 Python，插件会自动降级成“存 PNG + 打开目录”")


def test_viewer_window() -> None:
    print("[6] 真的开一次预览窗口（--check，建完界面立刻关闭）")
    python = depth_view.find_python()
    script = os.path.join(ADDON_PARENT, "pose_capture", "depth_view.py")
    if python is None:
        print("    跳过（没有带 tkinter 的 Python）")
        return
    import subprocess

    result = subprocess.run([python, script, "--dir", OUT_DIR, "--check"],
                            capture_output=True, text=True, timeout=90)
    print("    退出码 %d；输出：%s" % (result.returncode, (result.stdout or "").strip()))
    if result.returncode != 0:
        print("    stderr：%s" % (result.stderr or "").strip()[-300:])
    check(result.returncode == 0, "预览窗口能建起来", (result.stdout or "").strip())
    check("面板 3 个" in (result.stdout or ""), "界面里有 3 个面板")


def main():
    ensure_registered()
    test_write_image()
    test_save_png()
    test_depth_view_files()
    test_color_convention()
    test_viewer_page()
    test_python_lookup()
    test_viewer_window()
    print("结果：", "通过 ✓" if not FAILURES else "未通过 ✗")
    for item in FAILURES:
        print("   -", item)
    return 0 if not FAILURES else 1


if __name__ == "__main__":
    sys.exit(main())
