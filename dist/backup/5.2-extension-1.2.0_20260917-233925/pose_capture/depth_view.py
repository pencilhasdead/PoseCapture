"""深度图预览窗口（独立 Tk 页面）。

为什么要单独开一个窗口：Blender 图像编辑器里的预览受“像素写入 / 色彩空间”影响，
有时会整张显示成全黑，很难判断是模型输出不对还是显示不对。这个页面直接读磁盘上的
PNG + ``stats.json``（由插件在 Blender 里算好），把

    [输入照片]  [深度灰度]  [深度伪彩]

并排显示，并在下面列出数值统计（min/max/mean/std/百分位、是否恒定、是否接近全黑）。

用法（不需要 Blender，只需要一个带 tkinter 的 Python）：

    python depth_view.py --dir <预览目录>
    python depth_view.py --dir <预览目录> --columns 3 --max-size 520

预览目录由插件的“深度图预览窗口”按钮生成（Blender 侧负责跑模型并写 PNG）。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys

MANIFEST = "stats.json"
_PYTHON_CACHE: dict = {}


# --------------------------------------------------------------------------- 找解释器
def _can_import_tkinter(executable: str) -> bool:
    """跑一下候选解释器，看它能不能 import tkinter。"""
    try:
        result = subprocess.run([executable, "-c", "import tkinter"],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                timeout=20,
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        return result.returncode == 0
    except Exception:  # noqa: BLE001
        return False


def candidate_pythons() -> list[str]:
    """可能的解释器：当前进程 > PATH > py 启动器 > 常见安装位置。"""
    import shutil

    out: list[str] = []
    if sys.executable and sys.executable.lower().endswith(("python.exe", "pythonw.exe")):
        out.append(sys.executable)
    for name in ("pythonw.exe", "python.exe", "python3", "python"):
        found = shutil.which(name)
        if found:
            out.append(found)
    for name in ("py.exe", "py"):
        found = shutil.which(name)
        if found:
            out.append(found)
    roots = [r"C:\Python3", os.path.join(os.environ.get("LOCALAPPDATA") or "", "Programs",
                                         "Python")]
    for root in roots:
        if not root or not os.path.isdir(root):
            continue
        for entry in sorted(os.listdir(root), reverse=True):
            folder = os.path.join(root, entry)
            if os.path.isdir(folder):
                for name in ("pythonw.exe", "python.exe"):
                    path = os.path.join(folder, name)
                    if os.path.isfile(path):
                        out.append(path)
    seen = set()
    unique = []
    for path in out:
        key = os.path.abspath(path).lower()
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def find_python(preferred: str = "") -> str | None:
    """找一个能跑 tkinter 的解释器；找不到返回 None（结果按候选路径缓存）。"""
    if preferred:
        cached = _PYTHON_CACHE.get(("pref", preferred))
        if cached is None:
            cached = bool(os.path.isfile(preferred) and _can_import_tkinter(preferred))
            _PYTHON_CACHE[("pref", preferred)] = cached
        if cached:
            return preferred
        return None
    if "auto" in _PYTHON_CACHE:
        return _PYTHON_CACHE["auto"]
    found = None
    for path in candidate_pythons():
        cached = _PYTHON_CACHE.get(path)
        if cached is None:
            cached = bool(os.path.isfile(path) and _can_import_tkinter(path))
            _PYTHON_CACHE[path] = cached
        if cached:
            found = path
            break
    _PYTHON_CACHE["auto"] = found
    return found


# --------------------------------------------------------------------------- 读数据
def load_manifest(directory: str) -> dict:
    path = os.path.join(directory, MANIFEST)
    if not os.path.isfile(path):
        raise FileNotFoundError("预览目录里没有 %s：%s" % (MANIFEST, path))
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def format_stats(manifest: dict) -> str:
    """把 stats.json 里每张图的统计写成多行文本。"""
    lines = [manifest.get("title") or "", ""]
    for panel in manifest.get("panels", []):
        stats = panel.get("stats") or {}
        lines.append("[%s] %s" % (panel.get("label", "?"), panel.get("caption", "")))
        if stats:
            lines.append("    min=%.4f max=%.4f mean=%.4f std=%.4f"
                         % (stats.get("min", 0.0), stats.get("max", 0.0),
                            stats.get("mean", 0.0), stats.get("std", 0.0)))
            if stats.get("percentiles"):
                lines.append("    百分位 1/2/50/98/99 = %s"
                             % ", ".join("%.4f" % v for v in stats["percentiles"]))
            if stats.get("mean255") is not None:
                lines.append("    显示通道均值（0-255）= %.2f" % stats["mean255"])
        for note in panel.get("notes") or []:
            lines.append("    * %s" % note)
        lines.append("")

    for note in manifest.get("notes") or []:
        lines.append("!! %s" % note)
    return "\n".join(lines)


# --------------------------------------------------------------------------- 界面
def _make_photo(tk_module, path: str, max_size: tuple[int, int]):
    """返回 (PhotoImage, 显示尺寸)。有 PIL 就平滑缩放，没有就用整数缩放。"""
    try:
        from PIL import Image, ImageTk

        image = Image.open(path)
        image.thumbnail(max_size)
        return ImageTk.PhotoImage(image), image.size
    except Exception:  # noqa: BLE001
        photo = tk_module.PhotoImage(file=path)
        factor = max(1, int(math.ceil(photo.width() / float(max_size[0]))),
                     int(math.ceil(photo.height() / float(max_size[1]))))
        if factor > 1:
            photo = photo.subsample(factor, factor)
        return photo, (photo.width(), photo.height())


def _open_folder(directory: str) -> None:
    try:
        if hasattr(os, "startfile"):
            os.startfile(directory)  # noqa: S606  （Windows）
        else:
            subprocess.Popen(["xdg-open", directory])
    except Exception as exc:  # noqa: BLE001
        print("打不开目录：%s" % exc)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Pose Capture 深度图预览窗口")
    parser.add_argument("--dir", default="", help="预览目录（含 stats.json 与 PNG）")
    parser.add_argument("--max-size", type=int, default=520, help="单张图最大显示边长")
    parser.add_argument("--title", default="Pose Capture 深度图预览")
    parser.add_argument("--check", action="store_true",
                        help="只建一次界面验证数据能不能显示，不进消息循环（自检用）")
    args, _unknown = parser.parse_known_args(argv)
    directory = args.dir or os.path.dirname(os.path.abspath(__file__))

    try:
        import tkinter as tk
        from tkinter import ttk
    except Exception as exc:  # noqa: BLE001
        print("这个 Python 没有 tkinter，无法打开预览窗口：%s" % exc)
        return 1

    root = tk.Tk()
    root.title("%s - %s" % (args.title, directory))
    root.minsize(720, 520)

    header = tk.Label(root, text=directory, anchor="w", font=("Segoe UI", 9))
    header.pack(fill="x", padx=8, pady=(6, 0))

    body = tk.Frame(root)
    body.pack(fill="both", expand=True, padx=8, pady=6)
    info = tk.Text(root, height=15, wrap="none", font=("Consolas", 9))
    info.pack(fill="both", expand=False, padx=8, pady=(0, 6))

    state: dict = {"photos": [], "canvases": []}

    def render() -> None:
        manifest = load_manifest(directory)
        root.title("%s - %s" % (manifest.get("title") or args.title,
                                os.path.basename(directory)))
        for widget in body.winfo_children():
            widget.destroy()
        state["photos"].clear()
        max_size = (args.max_size, args.max_size)
        for column, panel in enumerate(manifest.get("panels", [])):
            cell = tk.Frame(body)
            cell.grid(row=0, column=column, padx=6, sticky="n")
            tk.Label(cell, text=panel.get("label", "?"),
                     font=("Segoe UI", 10, "bold")).pack()
            caption = panel.get("caption") or ""
            canvas = tk.Canvas(cell, width=max_size[0], height=max_size[1],
                               background="#101010", highlightthickness=1,
                               highlightbackground="#404040")
            canvas.pack()
            path = panel.get("path") or ""
            if path and os.path.isfile(path):
                try:
                    photo, size = _make_photo(tk, path, max_size)
                except Exception as exc:  # noqa: BLE001
                    canvas.create_text(10, 10, anchor="nw", fill="#ff8888",
                                       text="读图失败：%s" % exc)
                    photo, size = None, (0, 0)
                if photo is not None:
                    state["photos"].append(photo)
                    canvas.create_image(size[0] // 2, size[1] // 2, image=photo)
            else:
                canvas.create_text(10, 10, anchor="nw", fill="#888888",
                                   text="缺少文件：%s" % path)
            tk.Label(cell, text=caption, wraplength=max_size[0],
                     justify="left", font=("Segoe UI", 8),
                     foreground="#bbbbbb").pack()
        info.delete("1.0", "end")
        info.insert("1.0", format_stats(manifest))

    def reload_all() -> None:
        try:
            render()
        except Exception as exc:  # noqa: BLE001
            info.delete("1.0", "end")
            info.insert("1.0", "读取失败：%s" % exc)

    buttons = tk.Frame(root)
    buttons.pack(fill="x", padx=8, pady=(0, 8))
    ttk.Button(buttons, text="重新加载", command=reload_all).pack(side="left")
    ttk.Button(buttons, text="打开目录", command=lambda: _open_folder(directory)).pack(
        side="left", padx=6)
    ttk.Button(buttons, text="关闭", command=root.destroy).pack(side="right")
    root.bind("<Escape>", lambda _event: root.destroy())

    reload_all()
    if args.check:
        root.update_idletasks()
        print("界面自检：面板 %d 个，成功加载图片 %d 张"
              % (len(body.winfo_children()), len(state["photos"])))
        root.destroy()
        return 0
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
