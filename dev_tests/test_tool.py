"""pose_to_controlnet 独立工具自检：导入 / 核心 / 渲染 / JSON / 流程 / 界面 / 真实模型。

用法（推荐用系统的 python，界面依赖 tkinter 就在那里）：
  python dev_tests/test_tool.py

也可以借 Blender 自带的 python 跑（能直接复用插件里已装好的 onnxruntime 和模型）：
  D:\\...\\Blender\\5.2\\python\\bin\\python.exe dev_tests/test_tool.py

各段自己判断依赖，缺什么就跳过并说明，不算失败。
"""
from __future__ import annotations

import math
import os
import struct
import sys
import tempfile
import time
import traceback
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(HERE)
TOOL_DIR = os.path.join(REPO_DIR, "pose_to_controlnet")
if TOOL_DIR not in sys.path:
    sys.path.insert(0, TOOL_DIR)

IMAGE_URL = ("https://raw.githubusercontent.com/open-mmlab/mmpose/main/"
             "tests/data/coco/000000000785.jpg")


def out(text: str = "") -> None:
    """打印（GBK 控制台下遇到编码不了的字也不炸）。"""
    try:
        print(text, flush=True)
    except UnicodeEncodeError:
        encoding = sys.stdout.encoding or "utf-8"
        print(text.encode(encoding, "replace").decode(encoding, "replace"), flush=True)


class Checker:
    def __init__(self) -> None:
        self.count = 0
        self.failures: list[str] = []
        self.skipped: list[str] = []

    def section(self, title: str) -> None:
        out("\n== %s ==" % title)

    def check(self, name: str, ok: bool, detail: str = "") -> bool:
        self.count += 1
        out("  %s %s%s" % ("[ok]" if ok else "[!!]", name,
                           ("  [%s]" % detail) if detail else ""))
        if not ok:
            self.failures.append(name)
        return bool(ok)

    def skip(self, name: str, reason: str) -> None:
        self.skipped.append(name)
        out("  [--] 跳过 %s（%s）" % (name, reason))


# --------------------------------------------------------------------------- 素材
def synthetic_person(width: int = 640, height: int = 480,
                     cx: float | None = None, cy: float | None = None):
    """造一个 133 点的假人，用于不依赖 ONNX 的渲染 / JSON 自检。

    坐标比例参照人体结构（单位是身高的一部分），保证肢体长度不为零、
    人物框有合理尺寸，这样渲染与裁剪逻辑都能被真正走到。
    """
    import numpy as np

    keypoints = np.zeros((133, 2), dtype=np.float32)
    scores = np.zeros(133, dtype=np.float32)
    center_x = width * 0.5 if cx is None else cx
    center_y = height * 0.45 if cy is None else cy
    unit = height * 0.30

    body = {
        0: (0.00, -1.00), 1: (-0.12, -1.06), 2: (0.12, -1.06),
        3: (-0.24, -1.04), 4: (0.24, -1.04),
        5: (-0.30, -0.55), 6: (0.30, -0.55),
        7: (-0.50, -0.20), 8: (0.50, -0.20),
        9: (-0.62, 0.16), 10: (0.62, 0.16),
        11: (-0.18, 0.02), 12: (0.18, 0.02),
        13: (-0.20, 0.62), 14: (0.20, 0.62),
        15: (-0.21, 1.10), 16: (0.21, 1.10),
    }
    for index, (dx, dy) in body.items():
        keypoints[index] = (center_x + dx * unit, center_y + dy * unit)
        scores[index] = 0.9

    feet = {17: (-0.26, 1.12), 18: (-0.16, 1.14), 19: (-0.24, 1.20),
            20: (0.26, 1.12), 21: (0.16, 1.14), 22: (0.24, 1.20)}
    for index, (dx, dy) in feet.items():
        keypoints[index] = (center_x + dx * unit, center_y + dy * unit)
        scores[index] = 0.8

    face_x, face_y = keypoints[0]
    radius = unit * 0.22
    for i in range(68):
        angle = 2.0 * math.pi * i / 68.0
        keypoints[23 + i] = (face_x + radius * math.cos(angle),
                             face_y + radius * math.sin(angle))
        scores[23 + i] = 0.7

    for base, sign in ((91, -1.0), (112, 1.0)):
        wrist = keypoints[9] if sign < 0 else keypoints[10]
        keypoints[base] = wrist
        scores[base] = 0.75
        for i in range(1, 21):
            finger = (i - 1) // 4
            step = (i - 1) % 4 + 1
            keypoints[base + i] = (wrist[0] + sign * step * unit * 0.045,
                                   wrist[1] + (finger - 2) * unit * 0.05)
            scores[base + i] = 0.6
    return keypoints, scores


class StubEngine:
    """假引擎：不加载 ONNX，直接返回给定的 (keypoints, scores)。"""

    def __init__(self, people) -> None:
        self._people = people
        self.calls: list[tuple] = []

    def people(self, rgb, score_thr=0.3, det_thr=0.35, person=0):
        self.calls.append((tuple(rgb.shape), score_thr, det_thr, person))
        if person < 0:
            return [(kps.copy(), scores.copy()) for kps, scores in self._people]
        index = min(person, len(self._people) - 1)
        kps, scores = self._people[index]
        return [(kps.copy(), scores.copy())]

    def close(self) -> None:
        pass


def png_header(data: bytes) -> tuple[int, int, int, int]:
    """只解析 PNG 的 IHDR，不依赖任何图像库。"""
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("不是 PNG 数据")
    width, height, depth, color = struct.unpack(">IIBB", data[16:26])
    return width, height, depth, color


def rgb_array(width: int, height: int):
    import numpy as np

    rows = np.linspace(0, 255, height, dtype=np.uint8)[:, None]
    cols = np.linspace(0, 255, width, dtype=np.uint8)[None, :]
    return np.stack([np.broadcast_to(rows, (height, width)),
                     np.broadcast_to(cols, (height, width)),
                     np.full((height, width), 128, np.uint8)], axis=-1)


def has_module(name: str) -> bool:
    try:
        __import__(name)
    except Exception:  # noqa: BLE001
        return False
    return True


# --------------------------------------------------------------------------- 段落
def test_imports(check: Checker) -> None:
    """不需要 numpy 的部分：路径 / 依赖管理 / 命令行 / 界面模块能否导入。"""
    check.section("导入与模块划分")
    import importlib

    for name in ("pose_to_cn.paths", "pose_to_cn.depinstall", "pose_to_cn.core_loader",
                 "pose_to_cn.cli", "pose_to_cn.launcher", "pose_to_cn.__main__"):
        try:
            importlib.import_module(name)
        except Exception as exc:  # noqa: BLE001
            check.check("import %s" % name, False, repr(exc))
        else:
            check.check("import %s" % name, True)

    import pose_to_cn

    check.check("版本号 / 标题", bool(pose_to_cn.__version__ and pose_to_cn.APP_TITLE),
                "%s %s" % (pose_to_cn.__version__, pose_to_cn.APP_TITLE))

    if has_module("tkinter"):
        try:
            importlib.import_module("pose_to_cn.gui")
        except Exception as exc:  # noqa: BLE001
            check.check("import pose_to_cn.gui", False, repr(exc))
        else:
            check.check("import pose_to_cn.gui", True)
    else:
        check.skip("import pose_to_cn.gui", "当前 Python 没有 tkinter")

    from pose_to_cn import cli, paths

    parser = cli.build_parser()
    options = parser.parse_args(["--input", "a.jpg", "--no-hands", "--size-mode", "person",
                                 "--person", "-1", "--long-side", "768"])
    ok = (options.input == "a.jpg" and options.hands is False
          and options.size_mode == "person" and options.person == -1
          and options.long_side == 768)
    check.check("命令行参数解析", ok, "hands=%s person=%s" % (options.hands, options.person))

    data = paths.data_dir()
    check.check("工具数据目录可用", os.path.isdir(data), data)
    check.check("设置读写", _settings_roundtrip())


def _settings_roundtrip() -> bool:
    from pose_to_cn import paths

    before = paths.load_settings()
    paths.save_settings({"last_output_dir": os.path.join(tempfile.gettempdir(), "pose_to_cn")})
    after = paths.load_settings()
    paths.save_settings({"last_output_dir": before.get("last_output_dir", "")})
    return after.get("last_output_dir", "").endswith("pose_to_cn")


def test_core(check: Checker) -> None:
    """插件核心模块（coco / imops / dwpose / utils / deps）能否被复用。"""
    check.section("插件核心复用")
    from pose_to_cn import core_loader, paths

    try:
        core = core_loader.core()
    except Exception as exc:  # noqa: BLE001
        check.check("加载 posecore", False, repr(exc))
        return
    check.check("加载 posecore", True, core_loader.addon_dir())

    for name in ("coco", "imops", "dwpose", "utils", "deps"):
        check.check("core.%s 可用" % name, getattr(core, name, None) is not None)

    coco = core.coco
    check.check("133 点定义", len(coco.KEYPOINT_NAMES) == 133,
                "手 %d-%d，脸 %d-%d" % (coco.HAND_LEFT_START,
                                        coco.HAND_LEFT_START + coco.HAND_COUNT - 1,
                                        coco.FACE_ALL[0], coco.FACE_ALL[-1]))
    check.check("deps 里带模型下载地址",
                bool(core.deps.dwpose.DETECTOR_URLS and core.deps.dwpose.POSE_URLS))

    from pose_to_cn import render

    check.check("render.pose_bbox 存在", callable(getattr(render, "pose_bbox", None)))
    check.check("render.to_openpose18 存在", callable(getattr(render, "to_openpose18", None)))

    found = paths.find_models()
    check.check("能找到 DWPose 模型", bool(found),
                os.path.dirname(found[0]) if found else "只搜过：%s" % paths.candidate_model_dirs())


def test_render_and_json(check: Checker) -> None:
    """渲染与 JSON：坐标映射、图片尺寸、字段长度。"""
    check.section("渲染与 JSON（不需要 ONNX）")
    import numpy as np

    from pose_to_cn import imgio, json_io, render

    keypoints, scores = synthetic_person(640, 480)
    points, conf = render.to_openpose18(keypoints, scores)
    check.check("COCO133 -> OpenPose18", points.shape == (18, 2) and conf.shape == (18,),
                "脖子置信度 %.2f" % conf[1])
    check.check("脖子 = 双肩中点",
                abs(float(points[1][0]) - float((keypoints[5][0] + keypoints[6][0]) / 2)) < 1e-3)

    box = render.pose_bbox(keypoints, scores, 0.3)
    check.check("pose_bbox 返回 (x0,y0,x1,y1)",
                box is not None and box[2] > box[0] and box[3] > box[1],
                ("%.0f,%.0f,%.0f,%.0f" % box) if box else "None")

    painted = {}
    for style in ("openpose", "dwpose133"):
        image = render.render(keypoints, scores, 640, 480,
                              render.RenderOptions(style=style))
        painted[style] = int((image.reshape(-1, 3).max(axis=1) > 0).sum())
        check.check("渲染 %s" % style,
                    image.shape == (480, 640, 3) and image.dtype == np.uint8
                    and painted[style] > 500,
                    "上色像素 %d" % painted[style])

    control = render.render(keypoints, scores, 640, 480,
                            render.RenderOptions(style="openpose", feet=True))
    with_feet = int((control.reshape(-1, 3).max(axis=1) > 0).sum())
    check.check("feet 开关增加像素", with_feet > painted["openpose"],
                "%d -> %d" % (painted["openpose"], with_feet))

    data = json_io.openpose_dict([(keypoints, scores)], 640, 480, 0.3)
    person = data["people"][0]
    ok = (data["canvas_width"] == 640 and data["canvas_height"] == 480
          and len(person["pose_keypoints_2d"]) == 18 * 3
          and len(person["face_keypoints_2d"]) == 68 * 3
          and len(person["hand_left_keypoints_2d"]) == 21 * 3
          and len(person["hand_right_keypoints_2d"]) == 21 * 3
          and person["pose_keypoints_3d"] == [])
    check.check("OpenPose JSON 字段", ok,
                "pose 数组 %d 个数字" % len(person["pose_keypoints_2d"]))
    check.check("JSON 里鼻子坐标有效",
                person["pose_keypoints_2d"][2] > 0.5 and person["pose_keypoints_2d"][0] > 0)

    raw = imgio.png_bytes(control)
    header = png_header(raw)
    check.check("PNG 编码（纯 numpy，无需 PIL）",
                header == (640, 480, 8, 2),
                "IHDR %dx%d depth=%d color=%d" % header)
    if has_module("PIL"):
        import io

        from PIL import Image

        decoded = np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"))
        check.check("PNG 往返像素一致", bool((decoded == control).all()))
    else:
        check.skip("PNG 往返像素一致", "没有 PIL 可解码比对")

    folder = tempfile.mkdtemp(prefix="pose_cn_render_")
    path = imgio.write_image(os.path.join(folder, "shot.png"), control)
    check.check("写盘 PNG", os.path.getsize(path) > 1000, path)
    if has_module("PIL"):
        check.check("imgio.read_image 读回尺寸",
                    imgio.read_image(path).shape == (480, 640, 3))
    else:
        check.skip("imgio.read_image 读回尺寸", "没有 PIL / tkinter 解码器")


def test_pipeline(check: Checker) -> None:
    """三种尺寸模式 + 落盘 / 批量（用假引擎，不需要 ONNX）。"""
    check.section("转换流程（假引擎）")
    import numpy as np

    from pose_to_cn import imgio, pipeline

    source = rgb_array(640, 480)
    keypoints, scores = synthetic_person(640, 480)
    engine = StubEngine([(keypoints, scores)])

    modes = (("source", (640, 480)), ("long_side", (400, 300)))
    for mode, size in modes:
        options = pipeline.ConvertOptions(size_mode=mode, long_side=400)
        result = pipeline.convert_rgb(source, engine, options)
        painted = int((result.control.reshape(-1, 3).max(axis=1) > 0).sum())
        check.check("size_mode=%s 输出 %dx%d" % (mode, size[0], size[1]),
                    (result.width, result.height) == size and painted > 500
                    and result.source.shape == (size[1], size[0], 3),
                    "上色像素 %d" % painted)

    options = pipeline.ConvertOptions(size_mode="person", margin=0.10)
    result = pipeline.convert_rgb(source, engine, options)
    painted = int((result.control.reshape(-1, 3).max(axis=1) > 0).sum())
    check.check("size_mode=person 裁成方形",
                result.width == result.height and result.width >= 300 and painted > 500,
                "%dx%d，上色像素 %d" % (result.width, result.height, painted))
    check.check("person 模式下 JSON 尺寸与图一致",
                result.json_data["canvas_width"] == result.width
                and result.json_data["canvas_height"] == result.height)

    two = StubEngine([(keypoints, scores),
                      synthetic_person(640, 480, cx=140, cy=200)[0:2]])
    options = pipeline.ConvertOptions(person=-1, write_json=True)
    result = pipeline.convert_rgb(source, two, options)
    check.check("person=-1 画所有人", len(result.people) == 2
                and len(result.json_data["people"]) == 2)

    if not (has_module("PIL") or has_module("tkinter")):
        check.skip("落盘 / 批量", "没有图片解码器（PIL / tkinter）")
        return

    folder = tempfile.mkdtemp(prefix="pose_cn_in_")
    out_dir = tempfile.mkdtemp(prefix="pose_cn_out_")
    imgio.write_image(os.path.join(folder, "a.png"), rgb_array(320, 240))
    imgio.write_image(os.path.join(folder, "b.png"), rgb_array(320, 240))
    imgio.write_image(os.path.join(folder, "a_pose.png"), rgb_array(320, 240))

    options = pipeline.ConvertOptions(write_json=True, size_mode="source")
    single = pipeline.convert_file(os.path.join(folder, "a.png"), engine, options,
                                   os.path.join(out_dir, "one.png"))
    check.check("convert_file 落盘",
                os.path.isfile(os.path.join(out_dir, "one.png"))
                and os.path.isfile(os.path.join(out_dir, "one.json")))
    check.check("JSON 可被 json.load 读回", _json_ok(os.path.join(out_dir, "one.json")))

    results = pipeline.convert_folder(folder, engine, options, out_dir)
    ok = all(not str(item[1]).startswith("失败") for item in results)
    check.check("convert_folder 批量处理", len(results) == 2 and ok,
                "处理 %d 张" % len(results))
    check.check("跳过上一次生成的 *_pose.png",
                not os.path.isfile(os.path.join(out_dir, "a_pose_pose.png")))

    broken = os.path.join(folder, "broken.png")
    with open(broken, "wb") as handle:
        handle.write(b"not an image")
    results = pipeline.convert_folder(folder, engine, options, out_dir)
    failed = [item for item in results if str(item[1]).startswith("失败")]
    check.check("坏图不打断批量", len(results) == 3 and len(failed) == 1,
                "失败 %d 张" % len(failed))

    # 输出目录还不存在时也要能用（命令行常见的 -o outdir 写法）
    fresh_in = tempfile.mkdtemp(prefix="pose_cn_in2_")
    imgio.write_image(os.path.join(fresh_in, "c.png"), rgb_array(320, 240))
    new_dir = os.path.join(tempfile.mkdtemp(prefix="pose_cn_new_"), "out")
    results = pipeline.convert_folder(fresh_in, engine, options, new_dir)
    check.check("输出目录不存在时自动创建",
                os.path.isfile(os.path.join(new_dir, "c_pose.png"))
                and os.path.isfile(os.path.join(new_dir, "c_pose.json")), new_dir)

    paths = (
        pipeline.output_paths(os.path.join(folder, "a.png"), None, options)[0],
        pipeline.output_paths(os.path.join(folder, "a.png"),
                              os.path.join(new_dir, "sub"), options)[0],
        pipeline.output_paths(os.path.join(folder, "a.png"),
                              os.path.join(new_dir, "custom.png"), options)[0],
    )
    check.check("输出路径三种写法（空 / 目录 / 文件名）",
                paths == (os.path.join(folder, "a_pose.png"),
                          os.path.join(new_dir, "sub", "a_pose.png"),
                          os.path.join(new_dir, "custom.png")),
                str(paths))


def _json_ok(path: str) -> bool:
    import json

    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    return (data["canvas_width"] > 0 and data["canvas_height"] > 0
            and len(data["people"]) == 1
            and len(data["people"][0]["pose_keypoints_2d"]) == 54)


def fetch_photo(path: str | None = None) -> str | None:
    """取一张 COCO 测试照片（有缓存就直接用）；失败返回 None。"""
    path = path or os.path.join(tempfile.gettempdir(), "pose_to_cn_test.jpg")
    if os.path.isfile(path) and os.path.getsize(path) > 1024:
        return path
    try:
        request = urllib.request.Request(IMAGE_URL,
                                         headers={"User-Agent": "pose-cn-test"})
        with urllib.request.urlopen(request, timeout=120) as response, \
                open(path, "wb") as handle:
            handle.write(response.read())
    except Exception as exc:  # noqa: BLE001 - 没网就退回合成图
        out("  （下载测试照片失败：%s）" % exc)
        return None
    return path


def _json_ok_size(path: str, width: int, height: int) -> bool:
    import json

    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    values = data["people"][0]["pose_keypoints_2d"]
    return (data["canvas_width"] == width and data["canvas_height"] == height
            and sum(1 for value in values if value) > 0)


def test_real_engine(check: Checker) -> None:
    """真模型：加载 session、检测、出图出 JSON（依赖 ortruntime + 模型 + 照片）。"""
    check.section("真实模型（DWPose ONNX）")
    from pose_to_cn import depinstall, engine, paths

    status = depinstall.package_status()
    if not status.get("onnxruntime"):
        check.skip("onnxruntime 推理", "当前 Python 没有可用的 onnxruntime")
        return
    found = paths.find_models()
    if not found:
        check.skip("真实模型", "没有模型，先跑 --download-models")
        return
    check.check("找到模型", True, os.path.dirname(found[0]))

    pose = engine.PoseEngine()
    try:
        estimator = pose.estimator()
    except Exception as exc:  # noqa: BLE001
        check.check("加载模型 session", False, repr(exc))
        return
    check.check("加载模型 session", True,
                "%s / %s，设备 %s" % (estimator.detector.name, estimator.pose.name,
                                      getattr(estimator.pose, "active_provider", "?")))

    photo = fetch_photo() if has_module("PIL") else None
    if photo is None:
        try:
            people = pose.people(rgb_array(512, 512), 0.3, 0.35, 0)
        except Exception as exc:  # noqa: BLE001
            check.check("推理跑通", False, repr(exc))
        else:
            check.check("推理跑通（合成图，检不到人属正常）", isinstance(people, list),
                        "%d 人" % len(people))
        finally:
            pose.close()
        return

    from pose_to_cn import imgio, pipeline

    rgb = imgio.read_image(photo)
    check.check("读入测试照片", rgb.shape[2] == 3,
                "%dx%d" % (rgb.shape[1], rgb.shape[0]))
    people = pose.people(rgb, 0.3, 0.35, 0)
    if not check.check("检测到人", len(people) >= 1, "%d 人" % len(people)):
        pose.close()
        return

    keypoints, scores = people[0]
    check.check("关键点形状 (133,2) / (133,)",
                keypoints.shape == (133, 2) and scores.shape == (133,))
    check.check("身体置信度合理", float(scores[:23].max()) > 0.5,
                "身体最高 %.2f" % float(scores[:23].max()))
    bbox = pipeline.render.pose_bbox(keypoints, scores, 0.3)
    check.check("人物框落在图内",
                bbox is not None and bbox[0] > -5 and bbox[2] < rgb.shape[1] + 5,
                ("%.0f,%.0f,%.0f,%.0f" % bbox) if bbox else "None")
    check.check("person=-1 返回全部人物", len(pose.people(rgb, 0.3, 0.35, -1)) >= 1)

    out_dir = tempfile.mkdtemp(prefix="pose_cn_model_")
    options = pipeline.ConvertOptions(size_mode="long_side", long_side=768,
                                      write_json=True)
    result = pipeline.convert_file(photo, pose, options,
                                   os.path.join(out_dir, "pose.png"))
    with open(os.path.join(out_dir, "pose.png"), "rb") as handle:
        header = png_header(handle.read())
    painted = int((result.control.reshape(-1, 3).max(axis=1) > 0).sum())
    check.check("端到端：长边 768 出图",
                max(result.width, result.height) == 768 and painted > 2000,
                "%dx%d，上色像素 %d" % (result.width, result.height, painted))
    check.check("端到端：PNG 文件尺寸一致",
                (header[0], header[1]) == (result.width, result.height))
    check.check("端到端：JSON 与图尺寸一致",
                _json_ok_size(os.path.join(out_dir, "pose.json"),
                              result.width, result.height))
    check.check("输出目录内容", sorted(os.listdir(out_dir)) == ["pose.json", "pose.png"],
                str(sorted(os.listdir(out_dir))))
    pose.close()


def test_gui(check: Checker) -> None:
    """界面：能建窗口、能跑环境自检、参数收集正确、没有依赖也不崩。"""
    check.section("tkinter 界面")
    if not has_module("tkinter"):
        check.skip("界面", "当前 Python 没有 tkinter（Blender 自带的 python 就是这种）")
        return
    import tkinter as tk

    from pose_to_cn import gui

    try:
        root = tk.Tk()
    except Exception as exc:  # noqa: BLE001 - 无显示环境（比如远程会话）
        check.skip("界面", "创建窗口失败：%s" % exc)
        return
    root.withdraw()

    try:
        app = gui.ToolApp(root)
        check.check("窗口构建", True, root.title())

        def wait_log(needle: str, tries: int = 60) -> str:
            """跑事件循环直到日志里出现需要的文字（队列是 after 定时刷的）。"""
            text = ""
            for _ in range(tries):
                root.update()
                text = app.text_log.get("1.0", "end")
                if needle in text:
                    return text
                time.sleep(0.05)
            return text

        log_text = wait_log("解释器")             # 等环境自检线程把结果送回来
        check.check("环境自检有输出", "解释器" in log_text, app.var_status.get())
        check.check("日志框只读", str(app.text_log.cget("state")) == "disabled")

        options = app.build_options()
        check.check("参数收集：默认值",
                    options.style in ("openpose", "dwpose133")
                    and options.size_mode in ("source", "long_side", "person"),
                    "%s / %s / %d 人" % (options.style, options.size_mode, options.person))

        app.var_style.set("dwpose133")
        app.var_hands.set(False)
        app.var_size_mode.set("long_side")
        app.var_long_side.set(768)
        app.var_person.set(-1)
        options = app.build_options()
        check.check("参数收集：跟随界面",
                    options.style == "dwpose133" and options.hands is False
                    and options.long_side == 768 and options.person == -1)

        from pose_to_cn import paths

        check.check("参数已存进设置",
                    paths.load_settings().get("long_side") == 768)

        if has_module("numpy"):
            photo = gui._photo_from_rgb(rgb_array(120, 90))
            check.check("预览转换（numpy -> PhotoImage）",
                        photo.width() == 120 and photo.height() == 90,
                        "%dx%d" % (photo.width(), photo.height()))
        else:
            check.skip("预览转换", "没有 numpy")

        app.save_as()          # 还没有结果时应直接返回，不弹窗
        check.check("空状态不弹窗（save_as）", True)
        app._log("自检写日志")
        check.check("日志写入", "自检写日志" in wait_log("自检写日志", 20))
        check.check("未出结果时「另存」不可点",
                    str(app.btn_save.cget("state")) == "disabled")
    finally:
        root.destroy()


def main() -> int:
    check = Checker()
    out("工具目录：%s" % TOOL_DIR)
    out("解释器  ：%s" % sys.executable)

    try:                      # 和工具运行时一致：工具依赖目录优先，插件目录兜底
        from pose_to_cn import paths

        paths.ensure_libs_on_path()
    except Exception as exc:  # noqa: BLE001
        out("（准备依赖路径失败：%s）" % exc)

    steps = (
        ("导入与环境", test_imports, ()),
        ("插件核心", test_core, ("numpy",)),
        ("渲染与 JSON", test_render_and_json, ("numpy",)),
        ("转换流程", test_pipeline, ("numpy",)),
        ("真实模型", test_real_engine, ("numpy",)),
        ("界面", test_gui, ()),
    )
    for title, func, requires in steps:
        missing = [name for name in requires if not has_module(name)]
        if missing:
            check.skip(title, "缺少 %s" % ", ".join(missing))
            continue
        try:
            func(check)
        except Exception as exc:  # noqa: BLE001 - 段落里没兜住的异常算失败
            check.check("%s 未抛异常" % title, False, repr(exc))
            out(traceback.format_exc())

    out("\n共 %d 项检查：失败 %d，跳过 %d"
        % (check.count, len(check.failures), len(check.skipped)))
    for name in check.failures:
        out("  !! %s" % name)
    ok = not check.failures
    out("结果：" + ("通过" if ok else "未通过"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())




