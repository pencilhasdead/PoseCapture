"""操作器：安装依赖、下载模型、姿态检测、骨骼映射、把姿态写入骨架。"""
from __future__ import annotations

import json
import os
import queue
import subprocess
import threading
import traceback

import numpy as np

import bpy

from . import (coco, deps, depth, depth_view, dwpose, goo_physics, imops, mmd_mixamo,
               props, reconstruction, retarget, rigify_limbs, rigify_map, sam3d, utils)

# 模型 / 检测结果的会话缓存（不写入 .blend，重开文件后需要重新检测）
_ESTIMATOR_CACHE = {"key": None, "obj": None}
_DEPTH_CACHE = {"key": None, "obj": None}
_RESULT = {"keypoints": None, "scores": None, "boxes": None, "box": None,
           "image_size": None, "path": None, "rgb": None, "person_index": 0,
           "depth_map": None, "depth_size": None, "sam3d": None, "sam3d_recon": None}
# 上一次网格导入用的对齐（缩放 + 平移），演示/排错脚本可以直接读
_LAST_SAM3D_ALIGN = {"scale": 1.0, "offset": [0.0, 0.0, 0.0]}
# SAM 3D Body 结果缓存：键 = (图片, 尺寸, 人物序号)，值 = worker 的 npz 内容。
# 每跑一次子进程都要重新加载 2.7 GB 权重（实测 90 s 上下），同一张图换参数重检测时直接复用。
_SAM3D_CACHE = {}


def _sam3d_cache_key(settings, rgb, person: int):
    if getattr(settings, "image_source", "FILE") != "FILE":
        return None                      # 图像数据块没有稳定的身份，不缓存
    path = bpy.path.abspath(settings.image_filepath or "")
    if not path or not os.path.isfile(path):
        return None
    try:
        stat = os.stat(path)
    except OSError:
        return None
    return (os.path.abspath(path).lower(), int(stat.st_size), int(stat.st_mtime),
            int(rgb.shape[1]), int(rgb.shape[0]), int(person))





# --------------------------------------------------------------------------- 公共工具
def load_source_image(settings) -> np.ndarray:
    if settings.image_source == "DATA":
        image = bpy.data.images.get(settings.image_name)
        if image is None:
            raise ValueError("没有找到已加载的图像：%s" % settings.image_name)
        return utils.image_datablock_to_rgb8(image)
    return utils.load_image_rgb8(bpy.path.abspath(settings.image_filepath))


def get_estimator(detector: str, pose: str, backend: str, providers):
    key = (os.path.abspath(detector), os.path.abspath(pose), backend, tuple(providers or ()))
    if _ESTIMATOR_CACHE["key"] != key or _ESTIMATOR_CACHE["obj"] is None:
        _ESTIMATOR_CACHE["obj"] = dwpose.DwPoseEstimator(detector, pose, backend, providers)
        _ESTIMATOR_CACHE["key"] = key
    return _ESTIMATOR_CACHE["obj"]


def get_depth_estimator(path: str, backend: str, providers):
    """深度模型也缓存（加载一次约 1 秒，之后每次推理约 0.3 秒）。"""
    key = (os.path.abspath(path), backend, tuple(providers or ()))
    if _DEPTH_CACHE["key"] != key or _DEPTH_CACHE["obj"] is None:
        _DEPTH_CACHE["obj"] = depth.DepthEstimator(path, backend, providers)
        _DEPTH_CACHE["key"] = key
    return _DEPTH_CACHE["obj"]


def _write_image(name: str, canvas: np.ndarray) -> str:
    """把 (H,W,3) uint8 画布写进 Blender 图像数据块（尺寸变化时重建）。

    注意（Blender 5.2 实测）：GENERATED 图像在**写完像素之后**再赋
    colorspace_settings.name（哪怕赋成同样的 'sRGB'）会让 Blender 重建像素缓冲，
    内容直接变成全黑 —— 之前“深度图/骨架预览一直是纯黑”就是这个原因。
    所以色彩空间必须在写像素之前设置，写完不要再碰。
    """
    height, width = canvas.shape[:2]

    image = bpy.data.images.get(name)
    if image is not None and (image.size[0] != width or image.size[1] != height):
        bpy.data.images.remove(image)
        image = None
    if image is None:
        image = bpy.data.images.new(name, width=width, height=height, alpha=False)

    try:
        if image.colorspace_settings.name != "sRGB":
            image.colorspace_settings.name = "sRGB"
    except Exception:  # noqa: BLE001  某些生成的图像没有色彩空间设置
        pass

    rgba = np.empty((height, width, 4), dtype=np.float32)
    rgba[:, :, :3] = canvas.astype(np.float32) / 255.0
    rgba[:, :, 3] = 1.0
    rgba = rgba[::-1]  # Blender 从下往上存
    image.pixels.foreach_set(np.ascontiguousarray(rgba).reshape(-1))
    image.update()
    image.use_fake_user = True
    return image.name


def update_preview_image(name: str, rgb: np.ndarray, keypoints, scores, thr: float) -> str:
    return _write_image(name, imops.draw_pose_preview(rgb, keypoints, scores, thr))


def update_depth_preview_image(name: str, rgb: np.ndarray, depth_map,
                               near_is_large: bool, keypoints=None, scores=None,
                               thr: float = 0.3) -> str:
    """深度伪彩色（近=蓝/远=红）+ 骨架叠加，用来肉眼确认深度方向与质量。"""
    canvas = depth.preview_image(rgb, depth_map, near_is_large)
    if keypoints is not None and scores is not None:
        canvas = imops.draw_pose_preview(canvas, keypoints, scores, thr)
    return _write_image(name, canvas)


def depth_orientation(depth_map: np.ndarray, box, image_size) -> bool:
    """判断深度图里“值大”是近还是远：利用“人比背景离镜头近”这个事实。

    返回 True 表示“值大 = 近”（用于伪彩色预览的取色方向）。
    """
    if box is None or depth_map is None:
        return True
    height, width = depth_map.shape[:2]
    scale_x = width / max(float(image_size[0]), 1e-6)
    scale_y = height / max(float(image_size[1]), 1e-6)
    x0 = int(max(0, float(box[0]) * scale_x))
    x1 = int(min(width, float(box[2]) * scale_x))
    y0 = int(max(0, float(box[1]) * scale_y))
    y1 = int(min(height, float(box[3]) * scale_y))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return True
    inside = float(np.mean(depth_map[y0:y1, x0:x1]))
    mask = np.ones(depth_map.shape, dtype=bool)
    mask[y0:y1, x0:x1] = False
    if not mask.any():
        return True
    outside = float(np.mean(depth_map[mask]))
    return inside > outside


def depth_direction(depth_map: np.ndarray, box, image_size, recon=None) -> dict:
    """深度方向核对：**物理判据**（人比背景近）对照**拟合判据**（重建时 α 的符号）。

    预览的取色固定用物理判据：它只取决于像素本身，不会跟着“反转深度方向”一起翻，
    所以能反过来当拟合方向的独立校验 —— 两者不一致就说明 3D 姿态的前后整体反了。

    返回 ``{"physical", "fit", "agree", "near_is_large", "text", "warning"}``，
    其中 ``near_is_large``（True = 值大 = 近）就是该用来画预览的那个方向。
    """
    physical = depth_orientation(depth_map, box, image_size)
    info = (recon or {}).get("depth") or {}
    rejected = bool(info.get("rejected"))     # 拟合被判不可信、正文用的是骨长解算
    fit = bool(info["near_is_large"]) if ("near_is_large" in info and not rejected) else None

    def word(flag):
        return "近" if flag else "远"

    if box is None:
        text = "深度方向：无人体框，按默认“值大 = 近”上色（近=蓝 远=红）"
    else:
        text = "深度方向：值大 = %s（物理：人比背景近；伪彩 近=蓝 远=红）" % word(physical)
    if fit is not None:
        text += "；拟合“值大 = %s”%s" % (word(fit), "✓" if fit == physical else "✗")
    elif rejected:
        text += "；深度拟合被判为不可信（已退回骨长解算），方向不作核对"

    warning = ""
    if fit is not None and fit != physical:
        warning = ("深度方向和拟合结果对不上：物理判据说“值大 = %s”，但重建用到的却是"
                   "“值大 = %s”，整套姿态的前后可能反了 —— 勾上“反转方向”再点一次"
                   "“应用到骨架”" % (word(physical), word(fit)))
    return {"physical": bool(physical), "fit": fit, "agree": fit is None or fit == physical,
            "near_is_large": bool(physical), "text": text, "warning": warning}


def recon_summary(recon: dict) -> str:
    """把“用了哪种解算、拟合参数、兜底提示”写成一行，方便在面板/日志里核对。"""
    labels = {"DEPTH": "深度模型", "BONE": "骨长解算", "FLAT": "压平", "SAM3D": "SAM 3D Body"}

    mode = recon.get("mode", "?")
    parts = ["解算：%s" % labels.get(mode, mode)]
    info = recon.get("depth") or {}
    if info:
        parts.append("%s，α=%.3f，β=%.2fm，骨长残差 %.3f/%.3f m"
                     % (info.get("transform", "?"), float(info.get("alpha") or 0.0),
                        float(info.get("beta") or 0.0),
                        float(info.get("bone_error") or 0.0),
                        float(info.get("bone_error_fallback") or 0.0)))
        if info.get("repaired"):
            parts.append("骨长回修 %d 段" % int(info["repaired"]))
        if info.get("rejected"):
            parts.append("该拟合被判为不可信")
    parts.extend(recon.get("notes") or [])
    return "；".join(parts)


def report_summary(report: dict) -> str:
    parts = [report.get("message", "")]
    if report.get("ik_direct"):
        parts.append("已直写 IK 控制器（%s）" % ", ".join(report["ik_direct"]))
    errors = report.get("ik_direct_error") or {}
    worst = [value for value in errors.values() if value]
    if worst:
        parts.append("直写偏差 ≤ %.1f mm" % (max(value[0] for value in worst) * 1000.0))
    if report.get("ik_snapped"):
        parts.append("IK 控制器已对齐（%s）" % ", ".join(report["ik_snapped"]))
    if report.get("fk_switched"):
        parts.append("已把 %s 切到 FK" % ", ".join(report["fk_switched"]))
    if report.get("visibility"):
        parts.append("已同步控制器显示 %d 个集合" % len(report["visibility"]))
    if report.get("root_offset"):
        parts.append("根位移 %s" % (report["root_offset"],))
    if report.get("keyframes"):
        parts.append("插入 %d 个关键帧" % report["keyframes"])
    warnings = report.get("warnings") or []
    if warnings:
        parts.append("提示：" + "；".join(warnings[:3]))
    skipped = report.get("skipped") or []
    if skipped:
        parts.append("跳过 %d 项（例如 %s）" % (len(skipped), skipped[0][0]))
    return "；".join(part for part in parts if part)


# --------------------------------------------------------------------------- 模态后台任务基类
class PC_ModalWorker(bpy.types.Operator):
    bl_options = {"REGISTER", "INTERNAL"}

    _task_queue = None
    _thread = None
    _timer = None
    _cancel_flag = False
    _progress = None

    # -------------------------------------------------- 子类实现
    def worker(self, emit, cancel):
        raise NotImplementedError

    def on_done(self, context, result):
        return {"FINISHED"}

    def on_error(self, context, message):
        self.report({"ERROR"}, message)
        return {"CANCELLED"}

    # -------------------------------------------------- 内部
    def _push(self, kind, payload=None):
        if self._task_queue is not None:
            self._task_queue.put((kind, payload))

    def _thread_main(self):
        try:
            result = self.worker(self._push, lambda: self._cancel_flag)
            self._push("done", result)
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            self._push("error", "%s: %s" % (type(exc).__name__, exc))

    def start_worker(self, context):
        self._task_queue = queue.Queue()
        self._cancel_flag = False
        self._thread = threading.Thread(target=self._thread_main, daemon=True)
        self._thread.start()
        self._timer = context.window_manager.event_timer_add(0.1, window=context.window)
        context.window_manager.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def log_line(self, context, text: str) -> None:
        text = str(text)
        print(utils.LOG_PREFIX, text)
        settings = getattr(context.scene, "pose_capture", None)
        if settings is not None:
            settings.log = ((settings.log or "") + "\n" + text).strip()[-4000:]

    def status_text(self, context, text: str) -> None:
        settings = getattr(context.scene, "pose_capture", None)
        if settings is not None:
            settings.status = text
        if context.workspace is not None:
            context.workspace.status_text_set(text)

    def update_progress(self, context, done: int, total: int) -> None:
        window_manager = context.window_manager
        total = max(int(total), 1)
        if self._progress is None or self._progress[1] != total:
            window_manager.progress_begin(0, total)
            self._progress = [0, total]
        window_manager.progress_update(int(min(done, total)))

    def _cleanup(self, context) -> None:
        if self._timer is not None:
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None
        if self._progress is not None:
            context.window_manager.progress_end()
            self._progress = None
        if context.workspace is not None:
            context.workspace.status_text_set(None)
        for window in context.window_manager.windows:
            for area in window.screen.areas:
                area.tag_redraw()
        settings = getattr(context.scene, "pose_capture", None)
        if settings is not None:
            settings.status = ""

    def modal(self, context, event):
        if event.type == "ESC":
            self._cancel_flag = True
            self.status_text(context, "正在取消…")
            return {"RUNNING_MODAL"}
        if event.type != "TIMER":
            return {"RUNNING_MODAL"}

        finished = result = error = None
        while True:
            try:
                kind, payload = self._task_queue.get_nowait()
            except queue.Empty:
                break
            if kind == "log":
                self.log_line(context, payload)
            elif kind == "status":
                self.status_text(context, payload)
            elif kind == "progress":
                self.update_progress(context, payload[0], payload[1])
            elif kind == "done":
                finished, result = True, payload
            elif kind == "error":
                finished, error = True, payload

        if finished:
            self._cleanup(context)
            if error is not None:
                return self.on_error(context, error)
            return self.on_done(context, result)
        return {"RUNNING_MODAL"}


# --------------------------------------------------------------------------- 安装依赖
class PC_OT_install_deps(PC_ModalWorker):
    bl_idname = "pose_capture.install_deps"
    bl_label = "安装 / 更新推理依赖"

    def execute(self, context):
        prefs = props.get_prefs(context)
        self._package = prefs.backend_package
        self.report({"INFO"}, "开始安装 %s（可能需要几分钟）" % self._package)
        return self.start_worker(context)

    def worker(self, emit, cancel):
        emit(("status", "正在安装 %s…" % self._package))
        ok = deps.pip_install(self._package, log_cb=lambda line: emit(("log", line)))
        return {"ok": ok, "package": self._package}

    def on_done(self, context, result):
        status = deps.backend_status()
        if result.get("ok"):
            self.report({"INFO"}, "依赖安装完成：onnxruntime=%s, opencv=%s"
                        % (status["onnxruntime"], status["opencv"]))
        else:
            self.report({"ERROR"}, "安装失败，请查看日志（可能是网络或轮子不匹配）")
        return {"FINISHED"}


# --------------------------------------------------------------------------- 模型下载 / 导入
class PC_OT_download_models(PC_ModalWorker):
    bl_idname = "pose_capture.download_models"
    bl_label = "下载 DWPose 模型"

    def execute(self, context):
        prefs = props.get_prefs(context)
        det, pose = deps.model_paths()
        variant = getattr(prefs, "depth_variant", depth.DEFAULT_VARIANT)
        self._variant = variant
        self._targets = [(dwpose.DETECTOR_URLS, det, dwpose.DETECTOR_FILE),
                         (dwpose.POSE_URLS, pose, dwpose.POSE_FILE)]
        self._depth_target = (depth.model_urls(variant), deps.depth_model_path(variant),
                              depth.model_filename(variant))
        self.report({"INFO"}, "开始下载模型（DWPose 约 340MB + 深度模型）")
        return self.start_worker(context)

    def worker(self, emit, cancel):
        total_bytes = (deps.DETECTOR_SIZE_MB + deps.POSE_SIZE_MB
                       + int(depth.variant_info(self._variant)["size_mb"])) * 1024 * 1024
        offset = 0
        for urls, dest, name in self._targets:
            if cancel():
                return {"ok": False, "message": "已取消"}
            if os.path.isfile(dest):
                emit(("log", "已存在，跳过：%s" % name))
                offset += os.path.getsize(dest)
                continue
            emit(("status", "下载 %s…" % name))

            def progress(done, _total, _offset=offset):
                emit(("progress", (min(_offset + done, total_bytes), total_bytes)))

            ok = deps.download_model(urls, dest,
                                     progress_cb=progress,
                                     log_cb=lambda line: emit(("log", line)),
                                     cancel_cb=cancel)
            if not ok:
                return {"ok": False, "message": "下载失败：%s" % name}
            offset += os.path.getsize(dest)

        # 深度模型：下载失败不算致命（还可以用骨长解算兜底）
        urls, dest, name = self._depth_target
        depth_ok = False
        if cancel():
            emit(("log", "已取消深度模型下载"))
        elif os.path.isfile(dest):
            emit(("log", "已存在，跳过：%s" % name))
            depth_ok = True
        else:
            emit(("status", "下载 %s…" % name))

            def depth_progress(done, _total, _offset=offset):
                emit(("progress", (min(_offset + done, total_bytes), total_bytes)))

            depth_ok = deps.download_model(urls, dest, progress_cb=depth_progress,
                                           log_cb=lambda line: emit(("log", line)),
                                           cancel_cb=cancel)
            if not depth_ok:
                emit(("log", "深度模型下载失败：可改用“导入深度模型”，"
                             "或把档位换成 int8 再试"))
        return {"ok": True, "depth_ok": depth_ok}

    def on_done(self, context, result):
        if not result.get("ok"):
            self.report({"ERROR"}, result.get("message", "下载失败"))
            return {"FINISHED"}
        if result.get("depth_ok"):
            self.report({"INFO"}, "模型已就绪：%s" % utils.models_dir())
        else:
            self.report({"WARNING"},
                        "DWPose 模型已就绪；深度模型没下成功（可先改用骨长解算）")
        return {"FINISHED"}


class PC_OT_import_local_models(bpy.types.Operator):
    bl_idname = "pose_capture.import_local_models"
    bl_label = "从本地目录导入模型"

    def execute(self, context):
        prefs = props.get_prefs(context)
        root = bpy.path.abspath(prefs.local_dwpose_root)
        found = deps.find_local_models(root)
        if not found:
            self.report({"ERROR"}, "在 %s 里没找到 %s / %s"
                        % (root or "(空)", dwpose.DETECTOR_FILE, dwpose.POSE_FILE))
            return {"CANCELLED"}
        det, _pose = deps.copy_into_models_dir(found, log_cb=utils.log)
        # 同一个目录里如果有 depth*.onnx，顺手一起导入（ComfyUI 用户多半都有）
        depth_source = deps.find_local_depth_model(root)
        if depth_source:
            deps.copy_depth_into_models_dir(
                depth_source, getattr(prefs, "depth_variant", depth.DEFAULT_VARIANT),
                log_cb=utils.log)
        self.report({"INFO"}, "已导入模型：%s%s"
                    % (os.path.dirname(det), "（含深度模型）" if depth_source else ""))
        return {"FINISHED"}


class PC_OT_download_depth_model(PC_ModalWorker):
    bl_idname = "pose_capture.download_depth_model"
    bl_label = "下载深度模型"

    def execute(self, context):
        prefs = props.get_prefs(context)
        self._variant = getattr(prefs, "depth_variant", depth.DEFAULT_VARIANT)
        self._dest = deps.depth_model_path(self._variant)
        self._urls = depth.model_urls(self._variant)
        size = int(depth.variant_info(self._variant)["size_mb"])
        self.report({"INFO"}, "开始下载深度模型 %s（约 %d MB）"
                    % (depth.model_filename(self._variant), size))
        return self.start_worker(context)

    def worker(self, emit, cancel):
        if os.path.isfile(self._dest):
            emit(("log", "已存在，跳过：%s" % self._dest))
            return {"ok": True}
        total = int(depth.variant_info(self._variant)["size_mb"]) * 1024 * 1024
        emit(("status", "下载深度模型…"))
        ok = deps.download_model(
            self._urls, self._dest,
            progress_cb=lambda done, _total: emit(("progress", (done, total))),
            log_cb=lambda line: emit(("log", line)),
            cancel_cb=cancel)
        return {"ok": ok}

    def on_done(self, context, result):
        if result.get("ok"):
            self.report({"INFO"}, "深度模型已就绪：%s" % self._dest)
        else:
            self.report({"ERROR"}, "深度模型下载失败（可改用“导入深度模型”，"
                                   "或换成 int8 档位重试）")
        return {"FINISHED"}


class PC_OT_import_depth_model(bpy.types.Operator):
    """选择一个已有的深度模型 onnx（Depth Anything / MiDaS 等）并装到插件目录。"""

    bl_idname = "pose_capture.import_depth_model"
    bl_label = "导入深度模型"

    filepath: bpy.props.StringProperty(subtype="FILE_PATH")
    filter_glob: bpy.props.StringProperty(default="*.onnx", options={"HIDDEN"})

    def execute(self, context):
        prefs = props.get_prefs(context)
        source = bpy.path.abspath(self.filepath)
        if not os.path.isfile(source):
            self.report({"ERROR"}, "找不到文件：%s" % source)
            return {"CANCELLED"}
        variant = getattr(prefs, "depth_variant", depth.DEFAULT_VARIANT)
        target = deps.copy_depth_into_models_dir(source, variant, log_cb=utils.log)
        prefs.local_depth_model = target if os.path.isfile(target) else source
        self.report({"INFO"}, "深度模型已就绪：%s" % target)
        return {"FINISHED"}

    def invoke(self, context, event):
        if not self.filepath:
            prefs = props.get_prefs(context)
            self.filepath = getattr(prefs, "local_depth_model", "") or ""
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}


class PC_OT_show_depth(bpy.types.Operator):
    """把深度伪彩色图推到图像编辑器里显示，用来确认深度方向对不对。"""

    bl_idname = "pose_capture.show_depth"
    bl_label = "显示深度图"
    bl_options = {"REGISTER"}

    def execute(self, context):
        settings = props.get_settings(context)
        image = (bpy.data.images.get(settings.depth_preview_name)
                 if settings.depth_preview_name else None)
        if image is None:
            self.report({"ERROR"}, "还没有深度图：请先把“深度来源”设为深度模型并检测姿态")
            return {"CANCELLED"}
        for window in context.window_manager.windows:
            for area in window.screen.areas:
                if area.type == "IMAGE_EDITOR":
                    area.spaces.active.image = image
                    area.tag_redraw()
                    self.report({"INFO"}, "已在图像编辑器里显示深度图")
                    return {"FINISHED"}
        self.report({"INFO"}, "深度图已存在：%s（在图像编辑器里选择它即可）" % image.name)
        return {"FINISHED"}


# --------------------------------------------------------------------------- 深度图预览窗口
def _panel_stats(values, canvas) -> dict:
    """一张图的数值统计（用来判断“是不是真的全黑 / 恒定”）。"""
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = array[np.isfinite(array)]
    stats = {"min": -1.0, "max": -1.0, "mean": -1.0, "std": -1.0,
             "percentiles": [], "mean255": None}
    if finite.size:
        stats["min"] = float(finite.min())
        stats["max"] = float(finite.max())
        stats["mean"] = float(finite.mean())
        stats["std"] = float(finite.std())
        stats["percentiles"] = [float(v) for v in np.percentile(finite, [1, 2, 50, 98, 99])]
    if canvas is not None:
        stats["mean255"] = float(np.asarray(canvas, dtype=np.float64).mean())
    return stats


def _normalize_gray(values, near_is_large: bool = True, percentiles: bool = True) -> np.ndarray:
    """深度 -> 灰度图：**越白 = 越远**（和伪彩的“远=红”对得上，两张图能交叉核对）。

    默认按 2%~98% 百分位拉伸，避免个别异常值把整张压黑 / 压白。
    """
    array = np.asarray(values, dtype=np.float32)
    finite = array[np.isfinite(array)]
    if not finite.size:
        return np.zeros(array.shape, dtype=np.uint8)
    clean = np.where(np.isfinite(array), array, float(np.median(finite)))
    if percentiles:
        low, high = np.percentile(clean, [2.0, 98.0])
    else:
        low, high = float(clean.min()), float(clean.max())
    span = max(float(high) - float(low), 1e-6)
    norm = np.clip((clean - low) / span, 0.0, 1.0)      # 1 = 值大
    if near_is_large:
        norm = 1.0 - norm                               # 白 = 值小 = 远
    return (norm * 255.0 + 0.5).astype(np.uint8)


def _save_canvas_png(path: str, canvas: np.ndarray) -> None:
    """把 (H,W,3) uint8 画布存成 PNG（写完就删掉临时图像数据块）。"""
    height, width = canvas.shape[:2]
    image = bpy.data.images.new("PC_TMP_VIEW", width=width, height=height, alpha=False)
    try:
        if image.colorspace_settings.name != "sRGB":
            image.colorspace_settings.name = "sRGB"
        rgba = np.empty((height, width, 4), dtype=np.float32)
        rgba[:, :, :3] = canvas.astype(np.float32) / 255.0
        rgba[:, :, 3] = 1.0
        image.pixels.foreach_set(np.ascontiguousarray(rgba[::-1]).reshape(-1))
        image.update()
        image.filepath_raw = path
        image.file_format = "PNG"
        image.save()
    finally:
        try:
            bpy.data.images.remove(image)
        except Exception:  # noqa: BLE001
            pass


def _open_folder(directory: str) -> None:
    try:
        if hasattr(os, "startfile"):
            os.startfile(directory)  # noqa: S606
        elif os.name == "posix":
            subprocess.Popen(["xdg-open", directory])
    except Exception as exc:  # noqa: BLE001
        utils.warn("打不开目录 %s：%s" % (directory, exc))


def build_depth_view(settings, out_dir: str, keypoints=None, scores=None) -> dict:
    """把原图 / 深度灰度 / 深度伪彩写成预览目录里的 PNG + stats.json，返回清单。"""
    depth_map = _RESULT.get("depth_map")
    rgb = _RESULT.get("rgb")
    if depth_map is None or rgb is None:
        raise ValueError("还没有深度图：请先把“深度来源”设为深度模型并点“检测姿态”")

    depth_info = (_RESULT.get("recon") or {}).get("depth") or {}
    direction = depth_direction(depth_map, _RESULT.get("box"), _RESULT.get("image_size"),
                                _RESULT.get("recon"))
    near_is_large = direction["near_is_large"]
    thr = float(settings.score_thr)

    photo = rgb
    if keypoints is not None and scores is not None:
        photo = imops.draw_pose_preview(rgb, keypoints, scores, thr)
    gray = _normalize_gray(depth_map, near_is_large=near_is_large)
    gray_3 = np.repeat(gray[:, :, None], 3, axis=2)   # imops.resize 只吃 2D/3D 图
    gray_rgb = imops.resize(gray_3, rgb.shape[1], rgb.shape[0])
    color = depth.preview_image(rgb, depth_map, near_is_large)

    os.makedirs(out_dir, exist_ok=True)
    items = (
        ("photo", "输入照片（含检测骨架）", photo, photo,
         "先确认检测到的人是对的"),
        ("depth_gray", "深度灰度（越白 = 越远）", gray_rgb, depth_map,
         "2%~98% 拉伸；白 = 远、黑 = 近，和伪彩的“远=红”对得上"),
        ("depth_color", "深度伪彩（近=蓝 远=红）", color, depth_map,
         "和照片对照：脸/手这些离镜头近的地方应该是蓝的，远背景应该是红的"),
    )
    panels = []
    for name, label, canvas, values, caption in items:
        path = os.path.join(out_dir, "%s.png" % name)
        _save_canvas_png(path, canvas)
        panels.append({"label": label, "caption": caption, "path": path,
                       "stats": _panel_stats(values, canvas)})

    notes = []
    finite = np.asarray(depth_map, dtype=np.float32)
    finite = finite[np.isfinite(finite)]
    if finite.size:
        span = float(finite.max()) - float(finite.min())
        if span < 1e-3:
            notes.append("深度图几乎是常数（span=%.6f）：模型输出退化，检查模型文件是否正确"
                         % span)
    if float(gray.mean()) < 2.0:
        notes.append("灰度图几乎全黑：映射范围有问题")
    if float(np.asarray(color, dtype=np.float64).mean()) < 2.0:
        notes.append("伪彩图几乎全黑：深度值分布异常")
    if direction["warning"]:
        notes.append(direction["warning"])

    manifest = {
        "title": "Pose Capture 深度图预览（%s）"
                 % os.path.basename(_RESULT.get("path") or "未命名"),
        "source": _RESULT.get("path") or "",
        "image_size": list(_RESULT.get("image_size") or []),
        "depth_size": list(_RESULT.get("depth_size") or []),
        "near_is_large": near_is_large,
        "direction": direction,
        "depth_info": {key: value for key, value in depth_info.items()
                       if isinstance(value, (int, float, str, bool))},
        "panels": panels,
        "notes": notes,
    }
    with open(os.path.join(out_dir, "stats.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=1)
    return manifest


class PC_OT_open_depth_view(bpy.types.Operator):
    """单独开一个 Tk 预览窗口：原图 / 深度灰度 / 深度伪彩 + 数值统计。

    和 Blender 图像编辑器无关，所以“预览全黑到底是数据还是显示的问题”一眼就能分清。
    """

    bl_idname = "pose_capture.open_depth_view"
    bl_label = "深度图预览窗口"
    bl_options = {"REGISTER"}

    def execute(self, context):
        settings = props.get_settings(context)
        if _RESULT.get("depth_map") is None:
            self.report({"ERROR"}, "还没有深度图：请先把“深度来源”设为深度模型并点“检测姿态”")
            return {"CANCELLED"}

        out_dir = os.path.join(utils.user_data_dir("depth_view"), "latest")
        try:
            manifest = build_depth_view(settings, out_dir, _RESULT.get("keypoints"),
                                        _RESULT.get("scores"))
        except Exception as exc:  # noqa: BLE001
            self.report({"ERROR"}, "生成预览图失败：%s" % exc)
            return {"CANCELLED"}
        utils.log("深度预览清单：%s（%s）" % (manifest.get("title"), out_dir))

        script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "depth_view.py")
        prefs = props.get_prefs(context)
        python = depth_view.find_python(getattr(prefs, "depth_view_python", "") or "")
        if python is None:
            _open_folder(out_dir)
            self.report({"WARNING"},
                        "没找到带 tkinter 的 Python（Blender 自带的没有 tkinter）。"
                        "PNG 已存到 %s，可在插件偏好设置里手动指定解释器" % out_dir)
            return {"CANCELLED"}

        creation = 0
        for flag in ("DETACHED_PROCESS", "CREATE_NO_WINDOW"):
            creation |= getattr(subprocess, flag, 0)
        try:
            subprocess.Popen([python, script, "--dir", out_dir],
                             creationflags=creation, close_fds=True)
        except Exception as exc:  # noqa: BLE001
            self.report({"ERROR"}, "启动预览窗口失败：%s" % exc)
            return {"CANCELLED"}
        self.report({"INFO"}, "已打开深度图预览窗口：%s" % python)
        return {"FINISHED"}


# --------------------------------------------------------------------------- 姿态检测
class PC_OT_detect_pose(PC_ModalWorker):
    bl_idname = "pose_capture.detect"
    bl_label = "从图片检测姿态"

    def execute(self, context):
        settings = props.get_settings(context)
        prefs = props.get_prefs(context)

        detector, pose_path, ready = props.resolve_model_paths()
        if not ready:
            self.report({"ERROR"}, "缺少模型文件（%s / %s），请先在插件偏好设置里下载或导入"
                        % (dwpose.DETECTOR_FILE, dwpose.POSE_FILE))
            return {"CANCELLED"}

        try:
            rgb = load_source_image(settings)
        except Exception as exc:  # noqa: BLE001
            self.report({"ERROR"}, "读取图片失败：%s" % exc)
            return {"CANCELLED"}

        self._rgb = rgb
        self._paths = (detector, pose_path)
        self._backend = prefs.backend
        self._providers = props.resolve_providers()
        self._det_thr = float(settings.det_thr)
        self._score_thr = float(settings.score_thr)
        self._person = int(settings.person_index)

        # 3D 来源：SAM 3D Body（外部子进程）还是老路子（DWPose + 深度）
        self._solver, self._solver_note = props.resolve_solver(settings)
        self._sam3d_config = sam3d.config_from_prefs(prefs)
        self._sam3d_key = _sam3d_cache_key(settings, rgb, self._person)
        if self._solver_note:
            utils.warn(self._solver_note)


        # 深度模型：只在“3D 来源 = 深度模型”且模型就绪时才算（SAM 3D Body 不需要它）
        self._depth_path = None
        if self._solver == "DEPTH" and settings.depth_mode == "DEPTH":
            path, ready = props.resolve_depth_model_path()
            if ready:
                self._depth_path = path
            else:
                utils.warn("没有深度模型，本次用骨长解算兜底"
                           "（在插件偏好设置里可下载或导入深度模型）")
        return self.start_worker(context)


    def worker(self, emit, cancel):
        detector, pose_path = self._paths
        emit(("status", "加载模型…"))
        estimator = get_estimator(detector, pose_path, self._backend, self._providers)
        if cancel():
            return {"ok": False, "message": "已取消"}
        emit(("status", "检测中…（CPU 上大约 1~5 秒）"))
        result = estimator.run(self._rgb, det_thr=self._det_thr,
                               person_index=self._person)
        result["ok"] = result.get("keypoints") is not None
        result["depth_map"] = None
        result["depth_size"] = None
        result["sam3d"] = None
        result["sam3d_error"] = ""

        if result["ok"] and self._depth_path:
            if cancel():
                return {"ok": False, "message": "已取消"}
            emit(("status", "估算深度…（CPU 上约 1 秒）"))
            try:
                depth_estimator = get_depth_estimator(self._depth_path, self._backend,
                                                      self._providers)
                depth_map, depth_size = depth_estimator.predict(self._rgb)
                result["depth_map"] = depth_map
                result["depth_size"] = depth_size
            except Exception as exc:  # noqa: BLE001
                traceback.print_exc()
                emit(("log", "深度估计失败（本次改用骨长解算）：%s" % exc))

        # SAM 3D Body：3D 由模型给，不需要深度图
        if result["ok"] and self._solver == "SAM3D":
            if cancel():
                return {"ok": False, "message": "已取消"}
            cached = _SAM3D_CACHE.get(self._sam3d_key) if self._sam3d_key else None
            if cached is not None:
                result["sam3d"] = cached
                emit(("log", "SAM 3D Body：这张图 / 这个人物序号算过了，直接用缓存"))
                return result
            emit(("status", "SAM 3D Body 重建中…（要加载 2.7 GB 权重，首次/换图约 1~2 分钟）"))
            try:
                result["sam3d"] = sam3d.estimate(
                    self._sam3d_config, self._rgb, bbox=result.get("box"),
                    person_index=int(result.get("person_index", self._person)))
                if self._sam3d_key:
                    _SAM3D_CACHE[self._sam3d_key] = result["sam3d"]
                vertices = result["sam3d"].get("vertices")
                count = int(np.asarray(vertices).shape[0]) if vertices is not None else 0
                emit(("log", "SAM 3D Body 完成（%d 个网格顶点）" % count))

            except Exception as exc:  # noqa: BLE001
                traceback.print_exc()
                result["sam3d_error"] = str(exc)
                emit(("log", "SAM 3D Body 失败（本次退回深度 / 骨长解算）：%s" % exc))
        return result


    def on_done(self, context, result):
        settings = props.get_settings(context)
        if not result.get("ok"):
            settings.detected = False
            self.report({"WARNING"}, "没有检测到人体，请尝试换图或调整阈值")
            return {"FINISHED"}

        _RESULT.update({
            "keypoints": result["keypoints"],
            "scores": result["scores"],
            "boxes": result["boxes"],
            "box": result.get("box"),
            "image_size": (self._rgb.shape[1], self._rgb.shape[0]),
            "rgb": self._rgb,
            "path": settings.image_filepath,
            "person_index": result.get("person_index", 0),
            "depth_map": result.get("depth_map"),
            "depth_size": result.get("depth_size"),
            "sam3d": result.get("sam3d"),
            "sam3d_recon": None,
        })
        settings.detected = True
        settings.person_index = int(result.get("person_index", 0))
        info = "检测到 %d 人，已选第 %d 人" % (len(result["boxes"]),
                                              result.get("person_index", 0) + 1)
        sam = result.get("sam3d")
        if sam is not None:
            meta = sam.get("meta") or {}
            info += "；SAM 3D Body：%s" % ("ok" if meta.get("ok") else "有结果")
            settings.sam3d_info = "SAM 3D Body 完成（%d 人，%s）" % (
                int(meta.get("people") or 1), meta.get("device") or "?")
        elif result.get("sam3d_error"):
            info += "；SAM 3D Body 失败"
            settings.sam3d_info = "SAM 3D Body 失败：%s" % result["sam3d_error"][:200]
        settings.detect_info = info
        utils.log(info, "box =", result.get("box"))


        try:
            settings.preview_name = update_preview_image(
                "PoseCapture_Preview", self._rgb, result["keypoints"],
                result["scores"], self._score_thr)
        except Exception as exc:  # noqa: BLE001
            utils.warn("生成预览图失败：%s" % exc)

        depth_map = result.get("depth_map")
        if depth_map is not None:
            direction = depth_direction(depth_map, result.get("box"),
                                        _RESULT["image_size"])
            near_is_large = direction["near_is_large"]
            settings.depth_info = "深度图 %dx%d（%s；伪彩：近=蓝 远=红）" % (
                depth_map.shape[1], depth_map.shape[0],
                "值大=近" if near_is_large else "值大=远")
            settings.depth_direction = direction["text"]
            try:
                settings.depth_preview_name = update_depth_preview_image(
                    "PoseCapture_Depth", self._rgb, depth_map, near_is_large,
                    result["keypoints"], result["scores"], self._score_thr)
            except Exception as exc:  # noqa: BLE001
                utils.warn("生成深度预览失败：%s" % exc)
        else:
            settings.depth_info = ""
            settings.depth_direction = ""
            settings.depth_preview_name = ""

        self.report({"INFO"}, info)
        return {"FINISHED"}


class PC_OT_show_preview(bpy.types.Operator):
    """把预览图推到图像编辑器里显示。"""
    bl_idname = "pose_capture.show_preview"
    bl_label = "显示检测预览"
    bl_options = {"REGISTER"}

    def execute(self, context):
        settings = props.get_settings(context)
        image = bpy.data.images.get(settings.preview_name) if settings.preview_name else None
        if image is None:
            self.report({"ERROR"}, "还没有预览图，请先检测姿态")
            return {"CANCELLED"}
        for window in context.window_manager.windows:
            for area in window.screen.areas:
                if area.type == "IMAGE_EDITOR":
                    area.spaces.active.image = image
                    area.tag_redraw()
                    self.report({"INFO"}, "已在图像编辑器里显示预览")
                    return {"FINISHED"}
        self.report({"INFO"}, "预览图已存在：%s（在图像编辑器里选择它即可）" % image.name)
        return {"FINISHED"}


# --------------------------------------------------------------------------- 骨架相关
def _resolve_armature(settings, context):
    """优先使用面板里指定的骨架，否则用当前活动对象（如果是骨架）。"""
    arm_obj = settings.armature
    if arm_obj is None and context.object is not None and context.object.type == "ARMATURE":
        arm_obj = context.object
    return arm_obj


class PC_OT_use_active_armature(bpy.types.Operator):
    bl_idname = "pose_capture.use_active_armature"
    bl_label = "使用当前选中骨架"

    def execute(self, context):
        settings = props.get_settings(context)
        obj = context.object
        if obj is None or obj.type != "ARMATURE":
            self.report({"ERROR"}, "请先选中一个骨架对象")
            return {"CANCELLED"}
        settings.armature = obj
        bmap = rigify_map.auto_detect(obj, settings.fingers, props.limb_mode_map(settings), settings.rig_type)
        props.sync_mapping_to_settings(settings, bmap)
        self.report({"INFO"}, "已绑定 %s，映射角色 %d 个（缺失 %d 个）"
                    % (obj.name, len(bmap.roles), len(bmap.missing)))
        return {"FINISHED"}


class PC_OT_auto_map(bpy.types.Operator):
    bl_idname = "pose_capture.auto_map"
    bl_label = "自动匹配骨骼"

    def execute(self, context):
        settings = props.get_settings(context)
        arm_obj = _resolve_armature(settings, context)
        if arm_obj is None:
            self.report({"ERROR"}, "请先指定目标骨架")
            return {"CANCELLED"}
        bmap = rigify_map.auto_detect(arm_obj, settings.fingers, props.limb_mode_map(settings), settings.rig_type)
        props.sync_mapping_to_settings(settings, bmap)
        utils.log("骨骼映射：", bmap.roles)
        utils.log("未找到：", bmap.missing)
        self.report({"INFO"}, "匹配到 %d 个角色，缺少 %d 个（可手动在下拉框里指定）"
                    % (len(bmap.roles), len(bmap.missing)))
        return {"FINISHED"}


class PC_OT_add_metarig(bpy.types.Operator):
    bl_idname = "pose_capture.add_metarig"
    bl_label = "添加 Rigify 人形元骨架"

    def execute(self, context):
        settings = props.get_settings(context)
        try:
            bpy.ops.preferences.addon_enable(module="rigify")
        except Exception:  # noqa: BLE001
            pass
        try:
            bpy.ops.object.armature_human_metarig_add()
        except Exception as exc:  # noqa: BLE001
            self.report({"ERROR"}, "无法添加元骨架（Rigify 是否启用？）：%s" % exc)
            return {"CANCELLED"}

        obj = context.object
        settings.armature = obj
        settings.rig_type = "RIGIFY"          # 刚加的是 Rigify 元骨架，别按 Mixamo 去匹配
        bmap = rigify_map.auto_detect(obj, settings.fingers, props.limb_mode_map(settings),
                                      "RIGIFY")

        props.sync_mapping_to_settings(settings, bmap)
        self.report({"INFO"}, "已添加 %s" % obj.name)
        return {"FINISHED"}


class PC_OT_generate_rig(bpy.types.Operator):
    bl_idname = "pose_capture.generate_rig"
    bl_label = "生成 Rigify 控制骨架"

    def execute(self, context):
        settings = props.get_settings(context)
        arm_obj = _resolve_armature(settings, context)
        if arm_obj is None:
            self.report({"ERROR"}, "请先指定元骨架")
            return {"CANCELLED"}

        previous = context.view_layer.objects.active
        arm_obj.select_set(True)
        context.view_layer.objects.active = arm_obj
        try:
            if arm_obj.mode != "POSE":
                bpy.ops.object.mode_set(mode="POSE")
            bpy.ops.pose.rigify_generate()
        except Exception as exc:  # noqa: BLE001
            self.report({"ERROR"}, "生成失败：%s" % exc)
            if previous is not None:
                context.view_layer.objects.active = previous
            return {"CANCELLED"}

        generated = context.object
        if generated is not None and generated.type == "ARMATURE":
            settings.armature = generated
            settings.rig_type = "RIGIFY"
            bmap = rigify_map.auto_detect(generated, settings.fingers,
                                          props.limb_mode_map(settings), "RIGIFY")

            props.sync_mapping_to_settings(settings, bmap)
            self.report({"INFO"}, "已生成 %s 并更新映射" % generated.name)
        return {"FINISHED"}


class PC_OT_reset_pose(bpy.types.Operator):
    bl_idname = "pose_capture.reset_pose"
    bl_label = "清空被写入的 Pose"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        from mathutils import Matrix

        settings = props.get_settings(context)
        arm_obj = _resolve_armature(settings, context)
        if arm_obj is None:
            self.report({"ERROR"}, "请先指定目标骨架")
            return {"CANCELLED"}
        bmap = props.mapping_from_settings(settings)
        count = 0
        for name in set(bmap.used_bones()):
            pbone = arm_obj.pose.bones.get(name)
            if pbone is None:
                continue
            pbone.matrix_basis = Matrix.Identity(4)
            count += 1
        context.view_layer.update()
        self.report({"INFO"}, "已复位 %d 根骨骼" % count)
        return {"FINISHED"}


class PC_OT_sync_limb_display(bpy.types.Operator):
    """按当前 IK/FK 滑块状态藏起“不生效”的那组控制器。

    自己拖过 Rigify 面板上的 IK-FK 滑块之后点一下：滑到 FK 就藏 IK 控制器、显示 FK，
    反之亦然 —— 免得看见控制器却拖不动，以为插件/骨架坏了。
    """

    bl_idname = "pose_capture.sync_limb_display"
    bl_label = "同步控制器显示"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        settings = props.get_settings(context)
        arm_obj = _resolve_armature(settings, context)
        if arm_obj is None:
            self.report({"ERROR"}, "请先指定目标骨架")
            return {"CANCELLED"}
        specs = rigify_limbs.limb_specs(arm_obj)
        if not specs:
            self.report({"ERROR"}, "这不是 Rigify 生成的骨架：找不到 IK/FK 控制器")
            return {"CANCELLED"}
        touched = rigify_limbs.sync_visibility(arm_obj)
        modes = rigify_limbs.limb_modes(arm_obj, specs)
        detail = ", ".join("%s=%s" % (name, "FK" if value >= 0.5 else "IK")
                           for name, value in sorted(modes.items()))
        self.report({"INFO"}, "已同步 %d 个骨骼集合（%s）" % (len(touched), detail or "无"))
        return {"FINISHED"}


# --------------------------------------------------------------------------- 应用姿态
class PC_OT_apply_pose(bpy.types.Operator):
    bl_idname = "pose_capture.apply_pose"
    bl_label = "把姿态写入骨架"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        settings = props.get_settings(context)
        arm_obj = _resolve_armature(settings, context)
        if arm_obj is None:
            self.report({"ERROR"}, "请先指定目标骨架（右侧面板里选择）")
            return {"CANCELLED"}
        if _RESULT.get("keypoints") is None:
            self.report({"ERROR"}, "还没有检测结果：请先点“检测姿态”")
            return {"CANCELLED"}

        props.ensure_mapping_rows(settings)
        bmap = props.mapping_from_settings(settings)
        stale = [name for name in bmap.used_bones() if name not in arm_obj.data.bones]
        # 肢体骨跟“肢体写入方式”不一致（选了 IK 却映射到 FK 骨骼）时也要重匹配：
        # 否则 IK 模式下写到 FK 骨骼上的姿态是看不见的。按种类分别比（胳膊和腿可以不一样）。
        want = props.limb_mode_map(settings)
        got = rigify_limbs.mapped_limb_modes(arm_obj, bmap)
        switched = any(mode and mode != want.get(kind) for kind, mode in got.items())
        remap = (not bmap.roles or (stale and settings.auto_map_on_apply)
                 or (switched and settings.auto_map_on_apply))
        if remap:
            bmap = rigify_map.auto_detect(arm_obj, settings.fingers, props.limb_mode_map(settings), settings.rig_type)
            props.sync_mapping_to_settings(settings, bmap)
            utils.log("已重新自动匹配骨骼名")

        metrics = rigify_map.read_metrics(arm_obj, bmap)
        if metrics.get("errors"):
            self.report({"ERROR"}, "骨架不完整：%s" % "；".join(metrics["errors"]))
            return {"CANCELLED"}

        options = props.options_from_settings(settings)
        solver, solver_note = props.resolve_solver(settings)
        if solver_note:
            utils.warn(solver_note)
            settings.sam3d_info = solver_note
        if solver == "BONE":
            options = dict(options, depth_mode="BONE")

        recon = None
        if solver == "SAM3D":
            data = _RESULT.get("sam3d")
            if data is None:
                # 绝不静默回退：这是“我明明配了 SAM 3D Body，效果却没变”的最常见原因
                message = ("这次检测没有 SAM 3D Body 结果 —— 请重新点一次「检测姿态」"
                           "（3D 来源 = SAM 3D Body 时检测阶段就会跑模型）；"
                           "本次临时用深度 / 骨长解算")
                utils.warn(message)
                self.report({"WARNING"}, message)
                settings.sam3d_info = message
            else:
                try:
                    recon = sam3d.build_recon(data, options)
                except Exception as exc:  # noqa: BLE001
                    traceback.print_exc()
                    message = "SAM 3D Body 结果换算失败（%s），本次临时用深度 / 骨长解算" % exc
                    utils.warn(message)
                    self.report({"WARNING"}, message)
                    recon = None


        if recon is None:
            # 深度方向：骨长拟合在“正面站、身体深度本来就扁”的照片上分不出前后（实测两种方向的
            # 代价只差 0.1%），这时用物理判据（人比背景近，只看像素）钉住符号 —— 否则整套姿态会
            # 前后反过来，表现就是“深度图里手在前面，应用后手却在身体后面”。
            # 勾了“反转深度方向”就按用户的显式选择走（手动反转优先级最高，不用物理判据）。
            if _RESULT.get("depth_map") is not None and not options.get("depth_flip"):
                options = dict(options, depth_near_large=depth_orientation(
                    _RESULT["depth_map"], _RESULT.get("box"), _RESULT.get("image_size")))
            recon = reconstruction.reconstruct(_RESULT["keypoints"], _RESULT["scores"],
                                               metrics, _RESULT["image_size"], options,
                                               depth_map=_RESULT.get("depth_map"),
                                               depth_size=_RESULT.get("depth_size"))
            if not recon.get("ok"):
                self.report({"ERROR"}, "姿态重建失败：%s" % recon.get("message", ""))
                return {"CANCELLED"}
        _RESULT["recon"] = recon
        _RESULT["sam3d_recon"] = recon if recon.get("mode") == "SAM3D" else None

        solved_info = recon_summary(recon)
        utils.log(solved_info)
        mode_labels = {"SAM3D": "SAM 3D Body", "DEPTH": "深度模型", "BONE": "骨长解算",
                       "FLAT": "压平（无深度）"}
        settings.recon_source = mode_labels.get(recon.get("mode"), str(recon.get("mode") or "?"))
        if recon.get("mode") == "SAM3D":
            settings.recon_source += "（%d 顶点 / 重投影 %.1f px）" % (
                int(recon["sam3d"].get("mesh_vertices") or 0),
                float(recon["sam3d"].get("reproj_error") or 0.0))
        for note in recon.get("notes") or []:
            utils.warn(note)


        # 深度方向核对：预览取色固定按物理判据（人比背景近），并用它反过来查拟合方向
        # （SAM 3D Body 没有深度图，跳过这一段）
        if _RESULT.get("depth_map") is not None:
            direction = depth_direction(_RESULT.get("depth_map"), _RESULT.get("box"),
                                        _RESULT.get("image_size"), recon)
            settings.depth_direction = direction["text"]
            if direction["warning"]:
                utils.warn(direction["warning"])
                settings.depth_direction += "；★ " + direction["warning"]
        else:
            direction = {"near_is_large": True, "text": "", "warning": ""}
            settings.depth_direction = ""
        settings.depth_info = solved_info


        # 用物理判据重画一次深度预览（不再跟随拟合方向，前后反了才看得出来）
        if (_RESULT.get("recon") or {}).get("depth") and _RESULT.get("depth_map") is not None:
            try:
                settings.depth_preview_name = update_depth_preview_image(
                    "PoseCapture_Depth", _RESULT["rgb"], _RESULT["depth_map"],
                    direction["near_is_large"],
                    _RESULT.get("keypoints"), _RESULT.get("scores"),
                    float(settings.score_thr))
            except Exception as exc:  # noqa: BLE001
                utils.warn("更新深度预览失败：%s" % exc)

        target_options = dict(options)
        target_options["frame"] = context.scene.frame_current
        report = retarget.apply_pose(arm_obj, bmap, metrics, recon, target_options)
        summary = report_summary(report)
        settings.detect_info = summary
        utils.log(summary)

        if not report.get("ok"):
            self.report({"ERROR"}, summary or "没有任何骨骼被写入")
            return {"CANCELLED"}

        try:
            for obj in context.selected_objects:
                obj.select_set(False)
            arm_obj.select_set(True)
            context.view_layer.objects.active = arm_obj
            if arm_obj.mode != "POSE":
                bpy.ops.object.mode_set(mode="POSE")
        except Exception:  # noqa: BLE001
            pass

        self.report({"INFO"}, summary)
        return {"FINISHED"}


class PC_OT_export_json(bpy.types.Operator):
    bl_idname = "pose_capture.export_json"
    bl_label = "导出关键点 JSON"

    filepath: bpy.props.StringProperty(subtype="FILE_PATH")
    filter_glob: bpy.props.StringProperty(default="*.json", options={"HIDDEN"})

    def execute(self, context):
        if _RESULT.get("keypoints") is None:
            self.report({"ERROR"}, "还没有检测结果")
            return {"CANCELLED"}
        keypoints = np.asarray(_RESULT["keypoints"], dtype=float)
        scores = np.asarray(_RESULT["scores"], dtype=float)
        payload = {
            "source": _RESULT.get("path") or "",
            "image_size": list(_RESULT.get("image_size") or []),
            "person_index": _RESULT.get("person_index", 0),
            "keypoint_names": coco.KEYPOINT_NAMES,
            "keypoints": [[round(float(v), 3) for v in point] for point in keypoints],
            "scores": [round(float(v), 4) for v in scores],
        }
        with open(self.filepath, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=1)
        self.report({"INFO"}, "已导出：%s" % self.filepath)
        return {"FINISHED"}

    def invoke(self, context, event):
        if not self.filepath:
            self.filepath = os.path.join(utils.user_data_dir(), "keypoints.json")
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}


# --------------------------------------------------------------------------- SAM 3D Body
class PC_OT_sam3d_check(PC_ModalWorker):
    """在外部 python 里跑一次 --check：确认 torch / sam_3d_body 真的能导入。"""
    bl_idname = "pose_capture.sam3d_check"
    bl_label = "检查 SAM 3D Body 环境"

    def execute(self, context):
        prefs = props.get_prefs(context)
        self._config = sam3d.config_from_prefs(prefs)
        return self.start_worker(context)

    def worker(self, emit, cancel):
        emit(("status", "检查 SAM 3D Body 环境…（第一次导入 torch 要几秒）"))
        return sam3d.probe(self._config)

    def on_done(self, context, result):
        settings = props.get_settings(context)
        text = "SAM 3D Body 可用：%s / torch %s / %s / sam_3d_body %s" % (
            result.get("python") or result.get("executable") or "?",
            result.get("torch") or "?", "CUDA" if result.get("cuda") else "CPU",
            result.get("version") or "?")
        settings.sam3d_info = text
        utils.log(text)
        self.report({"INFO"}, text)
        return {"FINISHED"}


def _sam3d_material():
    name = "SAM3DBody"
    material = bpy.data.materials.get(name)
    if material is None:
        material = bpy.data.materials.new(name)
    material.use_nodes = True
    bsdf = material.node_tree.nodes.get("Principled BSDF")
    if bsdf is not None:
        bsdf.inputs["Base Color"].default_value = (0.65, 0.74, 0.86, 1.0)
        if "Roughness" in bsdf.inputs:
            bsdf.inputs["Roughness"].default_value = 0.6
    return material


class PC_OT_import_sam3d_mesh(bpy.types.Operator):
    """把 SAM 3D Body 的网格导进场景（和骨骼用同一套变换，骨盆对齐到目标骨架）。"""
    bl_idname = "pose_capture.import_sam3d_mesh"
    bl_label = "导入 SAM 3D Body 网格"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        settings = props.get_settings(context)
        data = _RESULT.get("sam3d")
        if data is None:
            self.report({"ERROR"},
                        "还没有 SAM 3D Body 结果：先把「3D 来源」设成 SAM 3D Body 再点检测")
            return {"CANCELLED"}

        recon = _RESULT.get("sam3d_recon")
        if not recon:
            try:
                recon = sam3d.build_recon(data, props.options_from_settings(settings))
            except Exception as exc:  # noqa: BLE001
                self.report({"ERROR"}, "SAM 3D Body 结果换算失败：%s" % exc)
                return {"CANCELLED"}

        try:
            verts, faces = sam3d.mesh_arrays(data, recon)
        except sam3d.Sam3DError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}

        scale, offset = self._align(settings, recon, verts)
        verts = verts * scale + offset


        mesh = bpy.data.meshes.new("SAM3DBody")
        mesh.from_pydata([tuple(float(value) for value in vertex) for vertex in verts], [],
                         [tuple(int(index) for index in face) for face in faces])
        mesh.validate()
        mesh.update()
        for polygon in mesh.polygons:
            polygon.use_smooth = True
        obj = bpy.data.objects.new("SAM3DBody", mesh)
        obj.data.materials.append(_sam3d_material())
        collection = getattr(context, "collection", None) or context.scene.collection
        collection.objects.link(obj)
        try:
            for other in context.selected_objects:
                other.select_set(False)
            obj.select_set(True)
            context.view_layer.objects.active = obj
        except Exception:  # noqa: BLE001
            pass
        self.report({"INFO"}, "已导入 SAM 3D Body 网格（%d 顶点，缩放 %.3f）"
                    % (len(verts), scale))
        _LAST_SAM3D_ALIGN.update(scale=float(scale),
                                 offset=[float(value) for value in offset])
        return {"FINISHED"}


    def _align(self, settings, recon, verts=None):
        """把网格对齐到目标骨架：按「大腿 + 小腿」骨长缩放，骨盆水平对齐，脚底贴地面。

        用骨长链而不是「骨盆 -> 脚底」的距离：蹲姿 / 劈叉时后者明显偏短，会把网格放大。
        """
        from mathutils import Vector

        arm = getattr(settings, "armature", None)
        if arm is None:
            return 1.0, np.zeros(3, dtype=np.float64)
        try:
            bmap = props.mapping_from_settings(settings)
            if not getattr(bmap, "roles", None):
                # 还没点过「自动匹配骨骼」也能对齐：就地认一下骨架结构
                bmap = rigify_map.auto_detect(arm, False, rig_type=getattr(settings, "rig_type", "RIGIFY"))

            metrics = rigify_map.read_metrics(arm, bmap)
            pelvis = metrics.get("pelvis")
        except Exception:  # noqa: BLE001
            return 1.0, np.zeros(3, dtype=np.float64)

        if pelvis is None:
            return 1.0, np.zeros(3, dtype=np.float64)

        rest = arm.matrix_world @ Vector(pelvis)
        joints = recon["joints"]
        floor = float(metrics.get("lowest_foot_z") or 0.0)

        # 骨架总高（所有骨骼头/尾的最高点 - 地面）
        height_rest = max((arm.matrix_world @ Vector(point)).z
                          for bone in arm.data.bones
                          for point in (bone.head_local, bone.tail_local)) - floor

        # 缩放：优先「网格高 -> 骨架高」（和姿态无关，蹲着也比得准）；
        # 没有网格时退回「大腿+小腿」骨长（比骨盆->脚底稳）
        scale = 1.0
        if verts is not None and len(verts) > 8:
            span = float(np.ptp(np.asarray(verts, dtype=np.float64)[:, 2]))
            if span > 1e-3 and height_rest > 1e-3:
                scale = height_rest / span
        if scale == 1.0:
            total = 0.0
            count = 0
            for role in ("thigh.L", "shin.L", "thigh.R", "shin.R"):
                name = bmap.get(role)
                bone = arm.data.bones.get(name) if name else None
                if bone is not None:
                    total += float(bone.length)
                    count += 1
            leg_rest = total / max(count / 2.0, 1.0) or max(rest.z - floor, 1e-3)

            def length(a, b):
                return float(np.linalg.norm(np.asarray(joints[b], dtype=np.float64)
                                            - np.asarray(joints[a], dtype=np.float64)))

            leg_sam = float(np.mean([length("hip.%s" % side, "knee.%s" % side)
                                     + length("knee.%s" % side, "ankle.%s" % side)
                                     for side in ("L", "R")]))
            scale = leg_rest / max(leg_sam, 1e-3)
        scale = float(np.clip(scale, 0.5, 2.0))

        sam_pelvis = np.asarray(joints["pelvis"], dtype=np.float64)
        offset = np.array((float(rest.x) - sam_pelvis[0] * scale,
                           float(rest.y) - sam_pelvis[1] * scale,
                           0.0))
        if verts is not None and len(verts):
            # 脚底贴骨架地面（网格最低点就是脚底）
            lowest = float((np.asarray(verts, dtype=np.float64) * scale)[:, 2].min())
            offset[2] = floor - lowest
        else:
            # 没有网格：让骨盆落在骨架骨盆高度（地面信息在 metrics 里）
            offset[2] = float(rest.z) - float(sam_pelvis[2]) * scale
        return scale, offset



        offset = np.asarray(rest, dtype=np.float64) - sam_pelvis * scale
        return scale, offset


# --------------------------------------------------------------------------- MMD -> Mixamo
def _selected_armature(context, settings):
    active = context.active_object
    if active is not None and active.type == "ARMATURE":
        return active
    return settings.armature


def _duplicate_model(context, arm):
    """连同网格复制一份（网格的 Armature 修改器会自动指向新骨架）。"""
    meshes = mmd_mixamo.find_meshes(arm)
    for obj in list(context.selected_objects):
        obj.select_set(False)
    arm.select_set(True)
    for obj in meshes:
        obj.select_set(True)
    context.view_layer.objects.active = arm
    bpy.ops.object.duplicate(linked=False)
    duplicated = context.view_layer.objects.active
    if duplicated is not None and duplicated.type == "ARMATURE":
        return duplicated
    return arm


class PC_OT_mmd_to_mixamo(bpy.types.Operator):
    """把 MMD 骨骼改成 Mixamo 命名（保留头发 / 裙子等非标准骨骼）。"""
    bl_idname = "pose_capture.mmd_to_mixamo"
    bl_label = "MMD 骨骼 → Mixamo"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        settings = props.get_settings(context)
        arm = _selected_armature(context, settings)
        if arm is None or arm.type != "ARMATURE":
            self.report({"ERROR"}, "先选中一根 MMD 骨架（或在上方面板指定目标骨架）")
            return {"CANCELLED"}

        bones = [bone.name for bone in arm.data.bones]
        parents = {bone.name: (bone.parent.name if bone.parent else None)
                   for bone in arm.data.bones}
        plan = mmd_mixamo.plan_conversion(
            bones, parents, prefix=settings.mmd_prefix,
            cleanup=bool(settings.mmd_cleanup),
            translate_extras=bool(settings.mmd_translate_extras))
        if not plan["rename"] and not plan["delete"]:
            message = "这个骨架看起来不是 MMD 骨架（一个标准 MMD 骨名都没认出来）"
            self.report({"WARNING"}, message)
            settings.mmd_result = message
            return {"CANCELLED"}

        source = arm
        if settings.mmd_duplicate:
            arm = _duplicate_model(context, arm)

        if settings.mmd_strip_helpers:
            removed = mmd_mixamo.strip_mmd_helpers(arm)
        else:
            removed = {"constraints": 0, "drivers": 0}

        report = mmd_mixamo.apply_plan(
            arm, plan, rename_groups=bool(settings.mmd_rename_groups),
            merge_weights=bool(settings.mmd_merge_weights))

        parts = [mmd_mixamo.summarize(plan),
                 "改名 %d / 删 %d / 改层级 %d" % (report["renamed"], report["deleted"],
                                                report["reparented"]),
                 "顶点组 %d 改名 / %d 合并" % (report["groups_renamed"],
                                             report["groups_merged"])]
        if removed["constraints"] or removed["drivers"]:
            parts.append("摘掉约束 %d / 驱动器 %d" % (removed["constraints"], removed["drivers"]))
        if settings.mmd_duplicate:
            parts.append("基于 %s（原模型保留）" % source.name)
        summary = "；".join(parts)
        settings.mmd_result = summary
        settings.armature = arm
        utils.log(summary)
        for note in plan.get("notes") or []:
            utils.warn(note)
        for warning in report.get("warnings") or []:
            utils.warn(warning)

        scale = arm.matrix_world.to_scale()
        if any(abs(value - 1.0) > 1e-4 for value in scale):
            utils.warn("骨架缩放不是 1（%s）：Mixamo 上传 / 重定向前先 Ctrl+A 应用缩放"
                       % tuple(round(value, 3) for value in scale))
        if not plan.get("hair"):
            utils.warn("没有认到头发骨（名字里没有 髪 / hair）：确认一下模型是否带头发骨")
        if report["deleted"] and not settings.mmd_merge_weights:
            utils.warn("删了控制骨但没有合并权重：这些骨头原来的蒙皮权重会丢失")

        self.report({"INFO"}, summary)
        return {"FINISHED"}


# --------------------------------------------------------------------------- Goo Physics
def _goo_kinds(settings) -> list[str]:
    kinds = []
    if settings.goo_hair:
        kinds.append("hair")
    if settings.goo_skirt:
        kinds.append("skirt")
    if settings.goo_ribbon:
        kinds.append("ribbon")
    return kinds


class PC_OT_goo_physics_add(bpy.types.Operator):
    """一键给头发 / 裙子 / 飘带链加 Goo Physics 物理模拟。"""
    bl_idname = "pose_capture.goo_physics_add"
    bl_label = "一键加物理（头发 / 裙子 / 飘带）"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        settings = props.get_settings(context)
        arm = _selected_armature(context, settings)
        if arm is None or arm.type != "ARMATURE":
            self.report({"ERROR"}, "先选中一根骨架（或在上方面板指定目标骨架）")
            return {"CANCELLED"}
        kinds = _goo_kinds(settings)
        if not kinds:
            self.report({"WARNING"}, "至少勾选一类（头发 / 裙子 / 飘带）")
            return {"CANCELLED"}

        report = goo_physics.apply(context, arm, kinds, settings.goo_type,
                                   bool(settings.goo_collider))
        summary = goo_physics.summarize(report)
        settings.goo_result = summary
        utils.log(summary)
        for warning in report.get("warnings") or []:
            utils.warn(warning)
        self.report({"INFO"} if report.get("ok") else {"WARNING"}, summary)
        return {"FINISHED"}


class PC_OT_goo_physics_remove(bpy.types.Operator):
    """移除头发 / 裙子 / 飘带骨上的 Goo Physics。"""
    bl_idname = "pose_capture.goo_physics_remove"
    bl_label = "移除头发 / 裙子上的物理"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        settings = props.get_settings(context)
        arm = _selected_armature(context, settings)
        if arm is None or arm.type != "ARMATURE":
            self.report({"ERROR"}, "先选中一根骨架")
            return {"CANCELLED"}
        report = goo_physics.remove(context, arm)
        summary = goo_physics.summarize(report)
        settings.goo_result = summary
        utils.log(summary)
        for warning in report.get("warnings") or []:
            utils.warn(warning)
        self.report({"INFO"} if report.get("ok") else {"WARNING"}, summary)
        return {"FINISHED"}


CLASSES = (
    PC_OT_install_deps,
    PC_OT_download_models,
    PC_OT_download_depth_model,
    PC_OT_import_local_models,
    PC_OT_import_depth_model,
    PC_OT_detect_pose,
    PC_OT_show_preview,
    PC_OT_show_depth,
    PC_OT_open_depth_view,
    PC_OT_use_active_armature,
    PC_OT_auto_map,
    PC_OT_add_metarig,
    PC_OT_generate_rig,
    PC_OT_reset_pose,
    PC_OT_sync_limb_display,
    PC_OT_apply_pose,
    PC_OT_export_json,
    PC_OT_sam3d_check,
    PC_OT_import_sam3d_mesh,
    PC_OT_mmd_to_mixamo,
    PC_OT_goo_physics_add,
    PC_OT_goo_physics_remove,
)








