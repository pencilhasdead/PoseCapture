"""操作器：安装依赖、下载模型、姿态检测、骨骼映射、把姿态写入骨架。"""
from __future__ import annotations

import json
import os
import queue
import threading
import traceback

import numpy as np

import bpy

from . import (coco, deps, depth, dwpose, imops, props, reconstruction, retarget,
               rigify_map, utils)

# 模型 / 检测结果的会话缓存（不写入 .blend，重开文件后需要重新检测）
_ESTIMATOR_CACHE = {"key": None, "obj": None}
_DEPTH_CACHE = {"key": None, "obj": None}
_RESULT = {"keypoints": None, "scores": None, "boxes": None, "box": None,
           "image_size": None, "path": None, "rgb": None, "person_index": 0,
           "depth_map": None, "depth_size": None}


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
    """把 (H,W,3) uint8 画布写进 Blender 图像数据块（尺寸变化时重建）。"""
    height, width = canvas.shape[:2]

    image = bpy.data.images.get(name)
    if image is not None and (image.size[0] != width or image.size[1] != height):
        bpy.data.images.remove(image)
        image = None
    if image is None:
        image = bpy.data.images.new(name, width=width, height=height, alpha=False)

    rgba = np.empty((height, width, 4), dtype=np.float32)
    rgba[:, :, :3] = canvas.astype(np.float32) / 255.0
    rgba[:, :, 3] = 1.0
    rgba = rgba[::-1]  # Blender 从下往上存
    image.pixels.foreach_set(rgba.reshape(-1))
    try:
        image.colorspace_settings.name = "sRGB"
    except Exception:  # noqa: BLE001
        pass
    image.update()
    image.use_fake_user = True
    return image.name


def update_preview_image(name: str, rgb: np.ndarray, keypoints, scores, thr: float) -> str:
    return _write_image(name, imops.draw_pose_preview(rgb, keypoints, scores, thr))


def update_depth_preview_image(name: str, rgb: np.ndarray, depth_map,
                               near_is_large: bool, keypoints=None, scores=None,
                               thr: float = 0.3) -> str:
    """深度伪彩色（近=红/远=蓝）+ 骨架叠加，用来肉眼确认深度方向与质量。"""
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


def recon_summary(recon: dict) -> str:
    """把“用了哪种解算、拟合参数、兜底提示”写成一行，方便在面板/日志里核对。"""
    labels = {"DEPTH": "深度模型", "BONE": "骨长解算", "FLAT": "压平"}
    mode = recon.get("mode", "?")
    parts = ["解算：%s" % labels.get(mode, mode)]
    info = recon.get("depth") or {}
    if info:
        parts.append("%s，α=%.3f，β=%.2fm，骨长残差 %.3f/%.3f m"
                     % (info.get("transform", "?"), float(info.get("alpha") or 0.0),
                        float(info.get("beta") or 0.0),
                        float(info.get("bone_error") or 0.0),
                        float(info.get("bone_error_fallback") or 0.0)))
        if info.get("rejected"):
            parts.append("该拟合被判为不可信")
    parts.extend(recon.get("notes") or [])
    return "；".join(parts)


def report_summary(report: dict) -> str:
    parts = [report.get("message", "")]
    if report.get("fk_switched"):
        parts.append("已把 %s 切到 FK" % ", ".join(report["fk_switched"]))
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
        # 深度模型：只在“深度来源 = 深度模型”且模型就绪时才算
        self._depth_path = None
        if settings.depth_mode == "DEPTH":
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
        })
        settings.detected = True
        settings.person_index = int(result.get("person_index", 0))
        info = "检测到 %d 人，已选第 %d 人" % (len(result["boxes"]),
                                              result.get("person_index", 0) + 1)
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
            near_is_large = depth_orientation(depth_map, result.get("box"),
                                              _RESULT["image_size"])
            settings.depth_info = "深度图 %dx%d（%s）" % (
                depth_map.shape[1], depth_map.shape[0],
                "值大=近" if near_is_large else "值大=远")
            try:
                settings.depth_preview_name = update_depth_preview_image(
                    "PoseCapture_Depth", self._rgb, depth_map, near_is_large,
                    result["keypoints"], result["scores"], self._score_thr)
            except Exception as exc:  # noqa: BLE001
                utils.warn("生成深度预览失败：%s" % exc)
        else:
            settings.depth_info = ""
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
        bmap = rigify_map.auto_detect(obj, settings.fingers)
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
        bmap = rigify_map.auto_detect(arm_obj, settings.fingers)
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
        bmap = rigify_map.auto_detect(obj, settings.fingers)
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
            bmap = rigify_map.auto_detect(generated, settings.fingers)
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
        if not bmap.roles or (stale and settings.auto_map_on_apply):
            bmap = rigify_map.auto_detect(arm_obj, settings.fingers)
            props.sync_mapping_to_settings(settings, bmap)
            utils.log("已重新自动匹配骨骼名")

        metrics = rigify_map.read_metrics(arm_obj, bmap)
        if metrics.get("errors"):
            self.report({"ERROR"}, "骨架不完整：%s" % "；".join(metrics["errors"]))
            return {"CANCELLED"}

        options = props.options_from_settings(settings)
        recon = reconstruction.reconstruct(_RESULT["keypoints"], _RESULT["scores"],
                                          metrics, _RESULT["image_size"], options,
                                          depth_map=_RESULT.get("depth_map"),
                                          depth_size=_RESULT.get("depth_size"))
        if not recon.get("ok"):
            self.report({"ERROR"}, "姿态重建失败：%s" % recon.get("message", ""))
            return {"CANCELLED"}
        _RESULT["recon"] = recon

        solved_info = recon_summary(recon)
        settings.depth_info = solved_info
        utils.log(solved_info)
        for note in recon.get("notes") or []:
            utils.warn(note)

        # 用拟合出来的深度方向重画一次深度预览（比检测时的“人比背景近”更准）
        depth_info = recon.get("depth") or {}
        if depth_info and _RESULT.get("depth_map") is not None:
            try:
                settings.depth_preview_name = update_depth_preview_image(
                    "PoseCapture_Depth", _RESULT["rgb"], _RESULT["depth_map"],
                    bool(depth_info.get("near_is_large", True)),
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


CLASSES = (
    PC_OT_install_deps,
    PC_OT_download_models,
    PC_OT_download_depth_model,
    PC_OT_import_local_models,
    PC_OT_import_depth_model,
    PC_OT_detect_pose,
    PC_OT_show_preview,
    PC_OT_show_depth,
    PC_OT_use_active_armature,
    PC_OT_auto_map,
    PC_OT_add_metarig,
    PC_OT_generate_rig,
    PC_OT_reset_pose,
    PC_OT_apply_pose,
    PC_OT_export_json,
)





