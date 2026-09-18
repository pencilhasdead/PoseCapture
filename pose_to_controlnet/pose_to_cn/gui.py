"""tkinter 界面。

只依赖标准库 + 本工具模块；numpy / onnxruntime 相关的模块都在函数内部按需导入，
这样没装依赖时界面也能打开，并提示点按钮安装。
"""
from __future__ import annotations

import base64
import os
import queue
import threading
import traceback

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from . import APP_TITLE, __version__
from . import depinstall, paths

PREVIEW_MAX = 460           # 预览图最长边（像素）
IMAGE_FILETYPES = [("图片", "*.png *.jpg *.jpeg *.bmp *.webp *.tif *.tiff"),
                   ("所有文件", "*.*")]


def _photo_from_rgb(rgb, max_side: int = PREVIEW_MAX) -> "tk.PhotoImage":
    """把 numpy 图缩放后用 PNG 字节喂给 Tk（不依赖 PIL）。"""
    from . import core_loader, imgio

    height, width = rgb.shape[:2]
    scale = min(1.0, float(max_side) / float(max(height, width)))
    if scale < 1.0:
        rgb = core_loader.core().imops.resize(
            rgb, max(1, int(width * scale)), max(1, int(height * scale)))
    return tk.PhotoImage(data=base64.b64encode(imgio.png_bytes(rgb)))


class ToolApp:
    """主窗口。"""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.settings = paths.load_settings()
        self.queue: "queue.Queue[tuple]" = queue.Queue()
        self.busy = False
        self.result = None
        self.engine = None
        self._engine_key = None
        self._photos: list = []
        self._cancel = threading.Event()

        root.title("%s  v%s" % (APP_TITLE, __version__))
        root.minsize(920, 700)

        self._build_vars()
        self._build_ui()
        self.root.after(120, self._poll_queue)
        self.root.after(300, self._startup_check)

    # ---------------------------------------------------------------- 变量
    def _build_vars(self) -> None:
        settings = self.settings
        self.var_image = tk.StringVar(value="")
        self.var_output = tk.StringVar(value=settings.get("last_output_dir") or "")
        self.var_model_dir = tk.StringVar(value=settings.get("model_dir") or "")
        self.var_status = tk.StringVar(value="正在检查环境…")
        self.var_progress = tk.DoubleVar(value=0.0)

        self.var_style = tk.StringVar(value=settings.get("style", "openpose"))
        self.var_score = tk.DoubleVar(value=float(settings.get("score_thr", 0.3)))
        self.var_det = tk.DoubleVar(value=float(settings.get("det_thr", 0.35)))
        self.var_person = tk.IntVar(value=int(settings.get("person", 0)))
        self.var_hands = tk.BooleanVar(value=bool(settings.get("hands", True)))
        self.var_face = tk.BooleanVar(value=bool(settings.get("face", True)))
        self.var_feet = tk.BooleanVar(value=bool(settings.get("feet", False)))
        self.var_json = tk.BooleanVar(value=bool(settings.get("write_json", True)))
        self.var_size_mode = tk.StringVar(value=settings.get("size_mode", "source"))
        self.var_long_side = tk.IntVar(value=int(settings.get("long_side", 1024)))

    # ---------------------------------------------------------------- 界面
    def _build_ui(self) -> None:
        outer = ttk.Frame(self.root, padding=10)
        outer.pack(fill="both", expand=True)

        paths_frame = ttk.LabelFrame(outer, text="文件", padding=8)
        paths_frame.pack(fill="x")

        ttk.Label(paths_frame, text="输入图片").grid(row=0, column=0, sticky="w", pady=2)
        ttk.Entry(paths_frame, textvariable=self.var_image).grid(
            row=0, column=1, sticky="ew", padx=6, pady=2)
        ttk.Button(paths_frame, text="选择图片…", command=self.pick_image).grid(
            row=0, column=2, pady=2)
        ttk.Button(paths_frame, text="选择文件夹…", command=self.pick_folder).grid(
            row=0, column=3, padx=(4, 0), pady=2)

        ttk.Label(paths_frame, text="输出到").grid(row=1, column=0, sticky="w", pady=2)
        ttk.Entry(paths_frame, textvariable=self.var_output).grid(
            row=1, column=1, sticky="ew", padx=6, pady=2)
        ttk.Button(paths_frame, text="选择输出…", command=self.pick_output).grid(
            row=1, column=2, pady=2)
        ttk.Button(paths_frame, text="打开目录", command=self.open_output_dir).grid(
            row=1, column=3, padx=(4, 0), pady=2)

        ttk.Label(paths_frame, text="模型目录").grid(row=2, column=0, sticky="w", pady=2)
        ttk.Entry(paths_frame, textvariable=self.var_model_dir).grid(
            row=2, column=1, sticky="ew", padx=6, pady=2)
        ttk.Button(paths_frame, text="选择目录…", command=self.pick_model_dir).grid(
            row=2, column=2, pady=2)
        ttk.Button(paths_frame, text="下载模型", command=self.download_models).grid(
            row=2, column=3, padx=(4, 0), pady=2)
        paths_frame.columnconfigure(1, weight=1)

        self._build_options(outer)
        self._build_actions(outer)
        self._build_preview(outer)
        self._build_log(outer)

        status = ttk.Frame(outer)
        status.pack(fill="x", pady=(6, 0))
        ttk.Label(status, textvariable=self.var_status).pack(side="left")
        ttk.Progressbar(status, variable=self.var_progress, maximum=100.0,
                        length=220).pack(side="right")

    def _build_options(self, outer: ttk.Frame) -> None:
        frame = ttk.LabelFrame(outer, text="参数", padding=8)
        frame.pack(fill="x", pady=(8, 0))

        ttk.Label(frame, text="画法").grid(row=0, column=0, sticky="w")
        style = ttk.Combobox(frame, textvariable=self.var_style, width=34,
                             state="readonly",
                             values=("openpose", "dwpose133"))
        style.grid(row=0, column=1, sticky="w", padx=6)
        style.bind("<<ComboboxSelected>>", lambda _e: self._sync)
        ttk.Label(frame, text="openpose = ControlNet 标准画法，"
                              "dwpose133 = 133 点全画（校对用）",
                  foreground="#666666").grid(row=0, column=2, columnspan=4, sticky="w")

        ttk.Label(frame, text="关键点阈值").grid(row=1, column=0, sticky="w", pady=(6, 0))
        ttk.Spinbox(frame, from_=0.05, to=0.95, increment=0.05, width=7,
                    textvariable=self.var_score).grid(row=1, column=1, sticky="w",
                                                      padx=6, pady=(6, 0))
        ttk.Label(frame, text="人框阈值").grid(row=1, column=2, sticky="w", pady=(6, 0))
        ttk.Spinbox(frame, from_=0.05, to=0.95, increment=0.05, width=7,
                    textvariable=self.var_det).grid(row=1, column=3, sticky="w",
                                                    padx=6, pady=(6, 0))
        ttk.Label(frame, text="人物序号").grid(row=1, column=4, sticky="w", pady=(6, 0))
        ttk.Spinbox(frame, from_=-1, to=20, width=7,
                    textvariable=self.var_person).grid(row=1, column=5, sticky="w",
                                                       padx=6, pady=(6, 0))
        ttk.Label(frame, text="0 = 最大的人，-1 = 全部", foreground="#666666").grid(
            row=1, column=6, sticky="w", pady=(6, 0))

        ttk.Label(frame, text="输出尺寸").grid(row=2, column=0, sticky="w", pady=(6, 0))
        size_box = ttk.Combobox(frame, textvariable=self.var_size_mode, width=14,
                                state="readonly",
                                values=("source", "long_side", "person"))
        size_box.grid(row=2, column=1, sticky="w", padx=6, pady=(6, 0))
        ttk.Label(frame, text="长边").grid(row=2, column=2, sticky="w", pady=(6, 0))
        ttk.Spinbox(frame, from_=64, to=4096, increment=64, width=7,
                    textvariable=self.var_long_side).grid(row=2, column=3, sticky="w",
                                                          padx=6, pady=(6, 0))
        ttk.Label(frame, text="source = 原图尺寸，long_side = 等比缩到长边，"
                              "person = 按人物框裁成方形", foreground="#666666").grid(
            row=2, column=4, columnspan=3, sticky="w", pady=(6, 0))

        checks = ttk.Frame(frame)
        checks.grid(row=3, column=0, columnspan=7, sticky="w", pady=(8, 0))
        ttk.Checkbutton(checks, text="画手", variable=self.var_hands).pack(side="left")
        ttk.Checkbutton(checks, text="画脸", variable=self.var_face).pack(side="left",
                                                                        padx=8)
        ttk.Checkbutton(checks, text="画脚（openpose 风格额外画 6 个脚点）",
                        variable=self.var_feet).pack(side="left", padx=8)
        ttk.Checkbutton(checks, text="同时导出 OpenPose JSON",
                        variable=self.var_json).pack(side="left", padx=8)

    def _build_actions(self, outer: ttk.Frame) -> None:
        frame = ttk.Frame(outer)
        frame.pack(fill="x", pady=(8, 0))
        self.btn_detect = ttk.Button(frame, text="检测并生成", command=self.run_detect)
        self.btn_detect.pack(side="left")
        self.btn_batch = ttk.Button(frame, text="批量处理文件夹", command=self.run_batch)
        self.btn_batch.pack(side="left", padx=6)
        self.btn_save = ttk.Button(frame, text="另存骨骼图…", command=self.save_as,
                                   state="disabled")
        self.btn_save.pack(side="left")
        ttk.Button(frame, text="安装依赖", command=self.install_deps).pack(side="left",
                                                                          padx=6)
        ttk.Button(frame, text="检查环境", command=self._startup_check).pack(side="left")

    def _build_preview(self, outer: ttk.Frame) -> None:
        frame = ttk.LabelFrame(outer, text="预览", padding=8)
        frame.pack(fill="both", expand=True, pady=(8, 0))

        left = ttk.Frame(frame)
        left.pack(side="left", expand=True)
        ttk.Label(left, text="原图").pack()
        self.label_source = tk.Label(left, background="#202020", width=64, height=24)
        self.label_source.pack()

        right = ttk.Frame(frame)
        right.pack(side="left", expand=True)
        ttk.Label(right, text="骨骼图（ControlNet 输入）").pack()
        self.label_control = tk.Label(right, background="#000000", width=64, height=24)
        self.label_control.pack()

    def _build_log(self, outer: ttk.Frame) -> None:
        frame = ttk.LabelFrame(outer, text="日志", padding=6)
        frame.pack(fill="both", expand=False, pady=(8, 0))
        self.text_log = tk.Text(frame, height=8, wrap="word", state="disabled")
        self.text_log.pack(side="left", fill="both", expand=True)
        bar = ttk.Scrollbar(frame, command=self.text_log.yview)
        bar.pack(side="right", fill="y")
        self.text_log.configure(yscrollcommand=bar.set)

    # ---------------------------------------------------------------- 线程 / 日志
    def _log(self, message: str) -> None:
        """线程安全地写日志（工作线程里调用）。"""
        self.queue.put(("log", message))

    def _poll_queue(self) -> None:
        try:
            while True:
                kind, payload = self.queue.get_nowait()
                if kind == "log":
                    self._append_log(payload)
                elif kind == "status":
                    self.var_status.set(payload)
                elif kind == "progress":
                    self.var_progress.set(float(payload))
                elif kind == "error":
                    self._append_log("错误：" + str(payload))
                    self.var_status.set("出错了，看日志")
                    messagebox.showerror(APP_TITLE, str(payload))
                elif kind == "busy":
                    self._set_busy(bool(payload))
                elif kind == "call":
                    payload()
        except queue.Empty:
            pass
        self.root.after(120, self._poll_queue)

    def _append_log(self, message: str) -> None:
        self.text_log.configure(state="normal")
        self.text_log.insert("end", str(message).rstrip() + "\n")
        self.text_log.see("end")
        self.text_log.configure(state="disabled")

    def _set_busy(self, busy: bool) -> None:
        self.busy = busy
        state = "disabled" if busy else "normal"
        for button in (self.btn_detect, self.btn_batch):
            button.configure(state=state)
        if not busy:
            self.btn_save.configure(state="normal" if self.result is not None else "disabled")
            self.var_progress.set(0.0)

    def _run(self, work, on_done=None) -> None:
        """在后台线程里跑 work；on_done(result) 在主线程执行。"""
        if self.busy:
            messagebox.showinfo(APP_TITLE, "正在处理，请等当前任务结束。")
            return
        self._cancel.clear()
        self._set_busy(True)

        def runner():
            try:
                result = work()
            except Exception as exc:  # noqa: BLE001 - 全部转成界面提示
                self.queue.put(("error", str(exc)))
                self.queue.put(("log", traceback.format_exc()))
            else:
                if on_done:
                    self.queue.put(("call", lambda: on_done(result)))
            finally:
                self.queue.put(("busy", False))

        threading.Thread(target=runner, daemon=True).start()

    # ---------------------------------------------------------------- 选文件
    def _sync(self) -> None:
        """把界面上的参数即时写回设置（下拉框选择后触发；参数不合法就忽略）。"""
        try:
            self.build_options()
        except Exception:  # noqa: BLE001 - 输入框还没填完时不要弹错
            pass

    def _remember(self, key: str, folder: str) -> None:
        settings = dict(self.settings)
        settings[key] = folder
        self.settings = settings
        paths.save_settings(settings)

    def pick_image(self) -> None:
        settings = paths.load_settings()
        chosen = filedialog.askopenfilename(
            title="选择图片", filetypes=IMAGE_FILETYPES,
            initialdir=settings.get("last_input_dir") or None)
        if chosen:
            self.var_image.set(chosen)
            self._remember("last_input_dir", os.path.dirname(chosen))

    def pick_folder(self) -> None:
        settings = paths.load_settings()
        chosen = filedialog.askdirectory(
            title="选择要批量处理的文件夹",
            initialdir=settings.get("last_input_dir") or None)
        if chosen:
            self.var_image.set(chosen)
            self._remember("last_input_dir", chosen)

    def pick_output(self) -> None:
        chosen = filedialog.askdirectory(title="选择输出目录",
                                         initialdir=self.var_output.get() or None)
        if chosen:
            self.var_output.set(chosen)
            self._remember("last_output_dir", chosen)

    def pick_model_dir(self) -> None:
        chosen = filedialog.askdirectory(title="选择 DWPose 模型目录")
        if chosen:
            self.var_model_dir.set(chosen)
            self.engine = None
            self._remember("model_dir", chosen)
            self._startup_check()

    def open_output_dir(self) -> None:
        folder = self.var_output.get().strip()
        if not folder:
            source = self.var_image.get().strip()
            folder = os.path.dirname(source) if os.path.isfile(source) else source
        if not folder or not os.path.isdir(folder):
            messagebox.showinfo(APP_TITLE, "还没有可打开的输出目录。")
            return
        try:
            os.startfile(folder)  # type: ignore[attr-defined]
        except Exception as exc:  # noqa: BLE001
            messagebox.showwarning(APP_TITLE, "打不开目录：%s" % exc)

    # ---------------------------------------------------------------- 环境
    def _startup_check(self) -> None:
        self._log("检查运行环境…")

        def work():
            from . import core_loader
            info: dict = {"packages": depinstall.package_status(),
                          "models": paths.models_status(),
                          "libs": paths.libs_dir(),
                          "data": paths.data_dir(),
                          "python": depinstall.python_executable()}
            try:
                info["addon"] = core_loader.addon_dir()
            except Exception as exc:  # noqa: BLE001
                info["addon"] = None
                info["addon_error"] = str(exc)
            return info

        self._run(work, on_done=self._report_environment)

    def _report_environment(self, info: dict) -> None:
        packages = info["packages"]
        errors = packages.get("errors") or {}
        self._log("解释器：%s" % info["python"])
        self._log("工具数据目录：%s" % info["data"])
        for name in ("numpy", "onnxruntime", "pillow"):
            version = packages.get(name)
            if version:
                self._log("  %-12s %s" % (name, version))
            else:
                reason = errors.get(name, "")
                self._log("  %-12s 缺失%s" % (name, ("（%s）" % reason) if reason else ""))
        if packages.get("providers"):
            self._log("  onnxruntime 可用后端：%s" % ", ".join(packages["providers"]))
        self._log("插件核心：%s" % (info.get("addon") or info.get("addon_error")))

        models = info["models"]
        if models["found"]:
            self._log("模型：%s" % models["detector"])
            self._log("      %s" % models["pose"])
            if not self.var_model_dir.get().strip():
                self.var_model_dir.set(os.path.dirname(models["detector"]))
        else:
            self._log("没找到模型，搜索过：")
            for folder in models["searched"]:
                self._log("  " + folder)

        missing = [name for name in depinstall.REQUIRED if not packages.get(name)]
        if missing:
            self.var_status.set("缺少依赖：%s —— 点「安装依赖」" % ", ".join(missing))
        elif not models["found"]:
            self.var_status.set("缺少 DWPose 模型 —— 点「下载模型」")
        elif not info.get("addon"):
            self.var_status.set("找不到插件核心模块，看日志")
        else:
            self.var_status.set("环境就绪")

    def install_deps(self) -> None:
        if self.busy:
            messagebox.showinfo(APP_TITLE, "正在处理，请等当前任务结束。")
            return
        message = ("将使用当前 Python 的 pip 把 numpy / onnxruntime / pillow 安装到\n"
                   "%s\n\n（不会动系统环境，约 60-200MB，需要联网）\n"
                   "继续吗？" % paths.libs_dir())
        if not messagebox.askokcancel(APP_TITLE, message):
            return
        self._log("开始安装依赖…")

        def work():
            return depinstall.install_everything(log_cb=self._log,
                                                 cancel_cb=self._cancel.is_set)

        self._run(work, on_done=self._after_install)

    def _after_install(self, ok) -> None:
        self._log("依赖安装成功。" if ok else "依赖安装失败或被取消。")
        self._startup_check()

    def download_models(self) -> None:
        if self.busy:
            messagebox.showinfo(APP_TITLE, "正在处理，请等当前任务结束。")
            return
        self._log("检查 / 下载 DWPose 模型…")

        def progress(done: int, total: int) -> None:
            if total:
                self.queue.put(("progress", 100.0 * done / total))

        def work():
            return depinstall.download_models(log_cb=self._log, progress_cb=progress,
                                              cancel_cb=self._cancel.is_set)

        self._run(work, on_done=self._after_download)

    def _after_download(self, found) -> None:
        if found:
            self._log("模型就绪：%s" % found[0])
            self.var_model_dir.set(os.path.dirname(found[0]))
            self.engine = None
            self._engine_key = None
        else:
            self._log("模型还没准备好（看上面的下载日志）。")
        self._startup_check()

    # ---------------------------------------------------------------- 处理
    def build_options(self):
        """把界面上的值收集成 ConvertOptions，并顺手存设置。"""
        from .options import ConvertOptions

        try:
            options = ConvertOptions(
                style=self.var_style.get(),
                score_thr=float(self.var_score.get()),
                det_thr=float(self.var_det.get()),
                person=int(self.var_person.get()),
                hands=bool(self.var_hands.get()),
                face=bool(self.var_face.get()),
                feet=bool(self.var_feet.get()),
                write_json=bool(self.var_json.get()),
                size_mode=self.var_size_mode.get(),
                long_side=int(self.var_long_side.get()),
            )
        except tk.TclError as exc:
            raise RuntimeError("参数填写有误：%s" % exc) from exc

        self.settings = paths.load_settings()
        self.settings.update({
            "style": options.style, "score_thr": options.score_thr,
            "det_thr": options.det_thr, "person": options.person,
            "hands": options.hands, "face": options.face, "feet": options.feet,
            "write_json": options.write_json, "size_mode": options.size_mode,
            "long_side": options.long_side,
            "model_dir": self.var_model_dir.get().strip(),
            "last_output_dir": self.var_output.get().strip(),
        })
        paths.save_settings(self.settings)
        return options

    def _deps_ready(self) -> bool:
        """推理前先确认依赖齐了，缺就给出明确提示（而不是抛 ImportError）。"""
        missing = depinstall.missing_packages()
        if missing:
            self._log("缺少依赖：%s" % ", ".join(missing))
            messagebox.showinfo(APP_TITLE, "缺少依赖：%s\n\n点上面的「安装依赖」装好再来。"
                                % ", ".join(missing))
            return False
        return True

    def _get_engine(self):
        """按当前模型目录取（必要时创建）推理引擎；只应在工作线程里调用。"""
        from . import engine as engine_mod

        model_dir = self.var_model_dir.get().strip() or None
        if self.engine is None or self._engine_key != model_dir:
            self.engine = engine_mod.PoseEngine(model_dir)
            self._engine_key = model_dir
        estimator = self.engine.estimator()
        self._log("推理后端：%s / %s，设备 %s，姿态输入 %s" % (
            estimator.detector.name, estimator.pose.name,
            getattr(estimator.pose, "active_provider", "?"),
            estimator.pose_input_size))
        return self.engine

    def run_detect(self) -> None:
        if not self._deps_ready():
            return
        from . import imgio, json_io, pipeline

        path = self.var_image.get().strip()
        if not path or not os.path.isfile(path):
            messagebox.showinfo(APP_TITLE, "先选一张图片（文件夹请用「批量处理」。）")
            return
        options = self.build_options()
        output = self.var_output.get().strip() or None
        self.var_status.set("检测中…")
        self._log("处理：%s" % path)

        def work():
            engine = self._get_engine()
            rgb = imgio.read_image(path)
            result = pipeline.convert_rgb(rgb, engine, options)
            png_path, json_path = pipeline.output_paths(path, output, options)
            imgio.write_image(png_path, result.control)
            saved = png_path
            if json_path and result.json_data is not None:
                json_io.write_json(json_path, result.json_data)
                saved += "  +  " + os.path.basename(json_path)
            result.notes = "%s → %s" % (result.notes, saved)
            return result

        self._run(work, on_done=self._on_single_done)

    def _on_single_done(self, result) -> None:
        self._log(result.notes)
        self._show_result(result)

    def run_batch(self) -> None:
        if not self._deps_ready():
            return
        from . import pipeline

        target = self.var_image.get().strip()
        folder = os.path.dirname(target) if os.path.isfile(target) else target
        if not folder or not os.path.isdir(folder):
            messagebox.showinfo(APP_TITLE, "先选一张图片或一个文件夹。")
            return
        options = self.build_options()
        output = self.var_output.get().strip() or None
        self.var_status.set("批量处理中…")
        self._log("批量处理目录：%s" % folder)

        def progress(index: int, total: int, path: str) -> None:
            self.queue.put(("progress", (100.0 * index / total) if total else 0.0))
            if path:
                self.queue.put(("status", "批量 %d/%d：%s"
                                % (index + 1, total, os.path.basename(path))))

        def work():
            engine = self._get_engine()
            return pipeline.convert_folder(folder, engine, options, output,
                                           on_progress=progress)

        self._run(work, on_done=self._on_batch_done)

    def _on_batch_done(self, results) -> None:
        ok = sum(1 for _name, out in results if not str(out).startswith("失败"))
        for path, out in results:
            self._log("%s -> %s" % (os.path.basename(path), out))
        summary = "批量结束：成功 %d / 共 %d" % (ok, len(results))
        self._log(summary)
        self.var_status.set(summary)

    def save_as(self) -> None:
        if self.result is None:
            return
        from . import imgio, json_io

        chosen = filedialog.asksaveasfilename(
            title="保存骨骼图", defaultextension=".png",
            filetypes=[("PNG 图片", "*.png"), ("JPEG 图片", "*.jpg")],
            initialdir=self.var_output.get() or None)
        if not chosen:
            return
        try:
            imgio.write_image(chosen, self.result.control)
            if self.var_json.get() and self.result.json_data is not None:
                json_io.write_json(os.path.splitext(chosen)[0] + ".json",
                                   self.result.json_data)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror(APP_TITLE, "保存失败：%s" % exc)
            return
        self._log("已另存：%s" % chosen)
        self.var_status.set("已另存 %s" % os.path.basename(chosen))

    def _show_result(self, result) -> None:
        self.result = result
        try:
            source_photo = _photo_from_rgb(result.source)
            control_photo = _photo_from_rgb(result.control)
        except Exception as exc:  # noqa: BLE001 - 预览失败不影响已保存的文件
            self._log("预览失败：%s" % exc)
            return
        self.label_source.configure(image=source_photo)
        self.label_control.configure(image=control_photo)
        self._photos = [source_photo, control_photo]
        self.btn_save.configure(state="normal")
        self.var_progress.set(100.0)
        self.var_status.set("完成：%dx%d，%d 人" % (result.width, result.height,
                                                 len(result.people)))

    # ---------------------------------------------------------------- 收尾
    def on_close(self) -> None:
        self._cancel.set()
        try:
            self.build_options()
        except Exception:  # noqa: BLE001 - 关窗时不再打扰用户
            pass
        self.root.destroy()


def main(argv=None) -> int:
    """启动界面。"""
    root = tk.Tk()
    try:
        ttk.Style().theme_use("vista")
    except Exception:  # noqa: BLE001 - 非 Windows / 没有该主题都无所谓
        pass

    app = ToolApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    app._log("工具数据目录：%s" % paths.data_dir())
    if not depinstall.backend_ready():
        app._log("缺少运行依赖，点上面的「安装依赖」按钮即可自动装好。")
    root.mainloop()
    return 0
