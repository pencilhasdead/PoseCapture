"""图片 -> ControlNet 骨骼图 的完整流程（含批量）。"""
from __future__ import annotations

import os

import numpy as np

from . import core_loader, imgio, json_io, render
# 参数与结果放在 options 里（那边不依赖 numpy），这里再导出一次，方便调用方只 import pipeline
from .options import (SIZE_MODES, SKIP_SUFFIXES, OUTPUT_SUFFIXES,  # noqa: F401
                      ConvertOptions, ConvertResult)

__all__ = ["SIZE_MODES", "SKIP_SUFFIXES", "OUTPUT_SUFFIXES", "ConvertOptions",
           "ConvertResult", "convert_rgb", "convert_file", "convert_folder",
           "output_paths"]



# --------------------------------------------------------------------------- 尺寸
def _resize_for_long_side(rgb: np.ndarray, long_side: int) -> tuple[np.ndarray, str]:
    height, width = rgb.shape[:2]
    current = max(height, width)
    if long_side <= 0 or current == long_side:
        return rgb, ""
    scale = float(long_side) / float(current)
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))
    core = core_loader.core()
    resized = core.imops.resize(rgb, new_w, new_h)
    return resized, "缩放到 %dx%d" % (new_w, new_h)


def _crop_square(rgb: np.ndarray, box, margin: float):
    """按人物框裁成正方形（越界部分补黑）。返回 (图, 左上角偏移)。"""
    x0, y0, x1, y1 = box
    center_x = (x0 + x1) / 2.0
    center_y = (y0 + y1) / 2.0
    half = max(x1 - x0, y1 - y0) / 2.0 * (1.0 + 2.0 * margin)
    half = max(half, 8.0)
    size = int(round(half * 2.0))
    left = int(round(center_x - half))
    top = int(round(center_y - half))

    height, width = rgb.shape[:2]
    canvas = np.zeros((size, size, 3), dtype=rgb.dtype)
    src_x0, src_y0 = max(0, left), max(0, top)
    src_x1, src_y1 = min(width, left + size), min(height, top + size)
    if src_x1 > src_x0 and src_y1 > src_y0:
        canvas[src_y0 - top:src_y1 - top, src_x0 - left:src_x1 - left] = \
            rgb[src_y0:src_y1, src_x0:src_x1]
    return canvas, (float(left), float(top))


# --------------------------------------------------------------------------- 主流程
def convert_rgb(rgb: np.ndarray, engine, options: ConvertOptions) -> ConvertResult:
    """对一张已读入的图片做检测 + 渲染（不落盘）。"""
    source = np.ascontiguousarray(np.asarray(rgb, dtype=np.uint8))
    notes: list[str] = []

    if options.size_mode == "long_side":
        source, note = _resize_for_long_side(source, options.long_side)
        if note:
            notes.append(note)

    people = engine.people(source, options.score_thr, options.det_thr, options.person)
    if not people:
        raise RuntimeError("没有检测到人体（可调低检测阈值，或换一张照片）")

    offset = (0.0, 0.0)
    if options.size_mode == "person":
        boxes = [render.pose_bbox(keypoints, scores, options.score_thr)
                 for keypoints, scores in people]
        boxes = [box for box in boxes if box]
        if boxes:
            union = (min(box[0] for box in boxes), min(box[1] for box in boxes),
                     max(box[2] for box in boxes), max(box[3] for box in boxes))
            source, offset = _crop_square(source, union, options.margin)
            notes.append("按人物框裁剪为 %dx%d" % (source.shape[1], source.shape[0]))
        else:
            notes.append("身体关键点太少，跳过裁剪")

    if offset != (0.0, 0.0):
        shift = np.asarray(offset, dtype=np.float32)
        people = [(keypoints - shift, scores) for keypoints, scores in people]

    height, width = source.shape[:2]
    render_options = options.render_options()
    control = np.zeros((height, width, 3), dtype=np.uint8)
    for keypoints, scores in people:          # 多人时逐个叠加（取较亮者）
        layer = render.render(keypoints, scores, width, height, render_options)
        control = np.maximum(control, layer)

    notes.append("%d 人，输出 %dx%d，风格 %s" % (len(people), width, height, options.style))
    json_data = None
    if options.write_json:
        json_data = json_io.openpose_dict(people, width, height, options.score_thr)

    return ConvertResult(control=control, source=source, people=people,
                         width=width, height=height, json_data=json_data,
                         notes="；".join(notes))


def output_paths(input_path: str, output_path: str | None,
                 options: ConvertOptions) -> tuple[str, str]:
    """给出 (骨骼图路径, JSON 路径)；JSON 路径为空串表示不写。

    output_path 的三种含义：
      空               -> 写在图片旁边，<图片名>_pose.png
      目录 / 没有扩展名 -> 当成目录，<目录>/<图片名>_pose.png（目录不存在会创建）
      带图片扩展名      -> 当成完整文件名
    """
    stem = os.path.splitext(os.path.basename(input_path))[0]
    suffix = os.path.splitext(output_path or "")[1].lower()
    as_file = bool(output_path) and suffix in OUTPUT_SUFFIXES

    if not output_path:
        png = os.path.join(os.path.dirname(os.path.abspath(input_path)),
                           stem + "_pose.png")
    elif as_file:
        png = output_path
    else:
        png = os.path.join(output_path, stem + "_pose.png")

    if not os.path.splitext(png)[1]:
        png += ".png"

    if not options.write_json:
        return png, ""
    if options.json_name:
        json_path = os.path.join(os.path.dirname(png), options.json_name)
    else:
        json_path = os.path.splitext(png)[0] + ".json"
    return png, json_path


def convert_file(input_path: str, engine, options: ConvertOptions,
                 output_path: str | None = None) -> ConvertResult:
    """读图 -> 检测 -> 渲染 -> 落盘。"""
    rgb = imgio.read_image(input_path)
    result = convert_rgb(rgb, engine, options)

    png_path, json_path = output_paths(input_path, output_path, options)
    imgio.write_image(png_path, result.control)
    if json_path and result.json_data is not None:
        json_io.write_json(json_path, result.json_data)
    result.notes = "%s → 已保存 %s" % (result.notes, os.path.basename(png_path))
    return result


def convert_folder(folder: str, engine, options: ConvertOptions,
                   output_dir: str | None = None, recursive: bool = False,
                   on_progress=None) -> list[tuple[str, str]]:
    """批量处理目录。返回 [(输入, 输出路径或错误信息), ...]。"""
    files = [path for path in imgio.list_images(folder, recursive)
             if not os.path.basename(path).lower().endswith(SKIP_SUFFIXES)]
    results: list[tuple[str, str]] = []
    for index, path in enumerate(files):
        if on_progress:
            on_progress(index, len(files), path)
        try:
            target = output_dir or folder
            out_png, _json_path = output_paths(path, target, options)
            convert_file(path, engine, options, out_png)
            results.append((path, out_png))
        except Exception as exc:  # noqa: BLE001 - 单张失败不影响整批
            results.append((path, "失败：%s" % exc))
    if on_progress:
        on_progress(len(files), len(files), "")
    return results
