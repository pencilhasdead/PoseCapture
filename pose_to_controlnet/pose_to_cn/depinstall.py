"""依赖与模型安装：装进工具自己的目录，不污染系统 Python。

提供：
  package_status()   探测 numpy / onnxruntime / pillow 是否可用
  missing_packages() 还缺什么
  pip_install()      用当前解释器 pip 安装到 <data>/libs
  download_models()  下载 DWPose 的 yolox_l.onnx + dw-ll_ucoco_384.onnx
"""
from __future__ import annotations

import os
import subprocess
import sys

from . import paths

REQUIRED = ("numpy", "onnxruntime")
OPTIONAL = ("pillow",)
MODULE_NAMES = {"numpy": "numpy", "onnxruntime": "onnxruntime", "pillow": "PIL"}

DETECTOR_MB = 207
POSE_MB = 128


# --------------------------------------------------------------------------- 探测
def package_status() -> dict:
    """返回 {包名: 版本或 None}，以及推理后端信息。

    errors 里放导入失败的简短原因（常见是装了别的 Python 版本的 wheel，
    比如 Blender 的 3.13 依赖被 3.14 的解释器 import）。
    """
    paths.ensure_libs_on_path()
    status = {"onnxruntime": None, "numpy": None, "pillow": None, "providers": [],
              "errors": {}}
    for name, module_name in MODULE_NAMES.items():
        try:
            module = __import__(module_name)
            status[name] = getattr(module, "__version__", "?")
            if name == "onnxruntime":
                status["providers"] = list(module.get_available_providers())
        except Exception as exc:  # noqa: BLE001 - 缺依赖 / ABI 不匹配等都算不可用
            status[name] = None
            status["errors"][name] = _short_error(exc)
    return status


def _short_error(exc: Exception) -> str:
    """把异常压成一行短说明（numpy 的 ABI 报错信息太长）。"""
    text = str(exc).strip().splitlines()
    for line in text:
        line = line.strip()
        if line and not line.startswith(("*", "IMPORTANT", "===")):
            return line[:160]
    return exc.__class__.__name__


def missing_packages() -> list[str]:
    status = package_status()
    return [name for name in REQUIRED if not status.get(name)]


def backend_ready() -> bool:
    return not missing_packages()


# --------------------------------------------------------------------------- pip
def python_executable() -> str:
    return sys.executable


def _safe_log(log_cb, text: str) -> None:
    """打日志绝不能影响安装本身（外部输出可能带控制台编码不了的字）。"""
    if not log_cb:
        return
    try:
        log_cb(text)
    except Exception:  # noqa: BLE001
        pass


def pip_install(packages, log_cb=None, cancel_cb=None) -> bool:
    """把包安装到工具自己的 libs 目录（--target），安装后立即可 import。"""
    target = paths.libs_dir()
    cmd = [python_executable(), "-m", "pip", "install", "--upgrade", "--no-cache-dir",
           "--disable-pip-version-check", "--no-warn-script-location",
           "--target", target]
    cmd.extend(packages)

    _safe_log(log_cb, "$ " + " ".join('"%s"' % part if " " in part else part
                                      for part in cmd))
    try:
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            universal_newlines=True, encoding="utf-8", errors="replace",
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except Exception as exc:  # noqa: BLE001
        _safe_log(log_cb, "启动 pip 失败：%s" % exc)
        return False

    assert process.stdout is not None
    for line in process.stdout:
        line = line.rstrip()
        if line:
            _safe_log(log_cb, line)
        if cancel_cb and cancel_cb():
            process.terminate()
            return False
    process.wait()

    if process.returncode != 0:
        _safe_log(log_cb, "pip 返回码 %d（可能是网络问题或该 Python 版本没有对应 wheel）"
                  % process.returncode)
        return False

    import importlib
    importlib.invalidate_caches()
    if target not in sys.path:
        sys.path.insert(0, target)
    return True


def install_everything(log_cb=None, cancel_cb=None) -> bool:
    """装齐运行需要的包（缺什么装什么；pillow 也一起装上，方便读 jpg/webp）。"""
    status = package_status()
    wanted = [name for name in REQUIRED + OPTIONAL if not status.get(name)]
    if not wanted:
        if log_cb:
            log_cb("依赖已齐全，无需安装。")
        return True
    if log_cb:
        log_cb("准备安装：%s" % ", ".join(wanted))
    return pip_install(wanted, log_cb=log_cb, cancel_cb=cancel_cb)


# --------------------------------------------------------------------------- 模型
def model_urls() -> tuple[list[str], list[str]]:
    from . import core_loader
    core = core_loader.core()
    return list(core.dwpose.DETECTOR_URLS), list(core.dwpose.POSE_URLS)


def download_models(log_cb=None, progress_cb=None, cancel_cb=None) -> tuple[str, str] | None:
    """下载两个 ONNX 模型到工具目录；已存在则跳过。返回 (检测器, 姿态模型)。"""
    from . import core_loader
    core = core_loader.core()
    deps = core.deps

    found = paths.find_models()
    if found:
        if log_cb:
            log_cb("已找到模型：%s" % found[0])
        return found

    target_dir = paths.models_dir()
    detector_urls, pose_urls = model_urls()
    targets = ((detector_urls, os.path.join(target_dir, paths.DETECTOR_FILE),
                "yolox_l.onnx", DETECTOR_MB),
               (pose_urls, os.path.join(target_dir, paths.POSE_FILE),
                "dw-ll_ucoco_384.onnx", POSE_MB))

    for urls, dest, name, size_mb in targets:
        if os.path.isfile(dest) and os.path.getsize(dest) > 1024:
            if log_cb:
                log_cb("已存在：%s" % dest)
            continue
        if log_cb:
            log_cb("开始下载 %s（约 %d MB）…" % (name, size_mb))
        if not deps.download_model(urls, dest, progress_cb=progress_cb,
                                   log_cb=log_cb, cancel_cb=cancel_cb):
            if log_cb:
                log_cb("下载失败：%s" % name)
            return None

    return paths.find_models()
