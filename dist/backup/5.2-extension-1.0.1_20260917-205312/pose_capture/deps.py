"""依赖与模型管理：检查/安装 onnxruntime，下载 DWPose 模型。

pip 安装与下载都在工作线程里跑，通过回调汇报进度（见 operators 里的 modal worker）。
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import urllib.request

from . import dwpose, utils

# 可选的推理后端包
BACKEND_PACKAGE_IDS = [
    "onnxruntime",
    "onnxruntime-directml",
    "onnxruntime-gpu",
    "opencv-python-headless",
]

DETECTOR_SIZE_MB = 207
POSE_SIZE_MB = 128


# --------------------------------------------------------------------------- 状态
def backend_status() -> dict:
    utils.ensure_libs_on_path()
    status = {"onnxruntime": None, "opencv": None, "providers": []}
    for key, module in (("onnxruntime", "onnxruntime"), ("opencv", "cv2")):
        try:
            mod = __import__(module)
            status[key] = getattr(mod, "__version__", "?")
            if key == "onnxruntime":
                status["providers"] = list(mod.get_available_providers())
        except Exception as exc:  # noqa: BLE001
            status[key] = None
            if key == "onnxruntime":
                status["providers_error"] = str(exc)
    return status


def backend_ready() -> bool:
    status = backend_status()
    return bool(status["onnxruntime"] or status["opencv"])


# --------------------------------------------------------------------------- pip
def pip_install(package: str, log_cb=None) -> bool:
    """用 Blender 自带 python 把依赖装到插件数据目录（不污染 Blender 本体）。"""
    utils.ensure_libs_on_path()
    python = utils.python_executable()
    target = utils.libs_dir()

    interpreter = python or sys.executable
    cmd = [interpreter, "-m", "pip", "install", "--upgrade", "--no-cache-dir",
           "--disable-pip-version-check", "--no-warn-script-location",
           "--target", target, package]

    if log_cb:
        log_cb("$ " + " ".join('"%s"' % c if " " in c else c for c in cmd))
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                universal_newlines=True, encoding="utf-8", errors="replace",
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except Exception as exc:  # noqa: BLE001
        if log_cb:
            log_cb("启动 pip 失败：%s" % exc)
        return False

    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip()
        if line and log_cb:
            log_cb(line)
    proc.wait()

    if proc.returncode != 0 and log_cb:
        log_cb("pip 返回码 %d（可能是网络问题，或该 Python 版本没有对应 wheel）"
               % proc.returncode)

    import importlib
    importlib.invalidate_caches()
    return proc.returncode == 0


# --------------------------------------------------------------------------- 下载
def _download_one(url: str, dest: str, progress_cb=None, cancel_cb=None) -> bool:
    tmp = dest + ".part"
    request = urllib.request.Request(url, headers={"User-Agent": "pose-capture-blender"})
    with urllib.request.urlopen(request, timeout=60) as response, open(tmp, "wb") as handle:
        total = int(response.headers.get("Content-Length") or 0)
        done = 0
        while True:
            if cancel_cb and cancel_cb():
                handle.close()
                if os.path.exists(tmp):
                    os.remove(tmp)
                return False
            chunk = response.read(1024 * 512)
            if not chunk:
                break
            handle.write(chunk)
            done += len(chunk)
            if progress_cb:
                progress_cb(done, total)
    if os.path.exists(dest):
        os.remove(dest)
    os.replace(tmp, dest)
    return True


def download_model(urls, dest: str, progress_cb=None, log_cb=None, cancel_cb=None) -> bool:
    for index, url in enumerate(urls):
        if log_cb:
            log_cb("下载 %s" % url)
        try:
            if _download_one(url, dest, progress_cb, cancel_cb):
                return True
        except Exception as exc:  # noqa: BLE001
            if log_cb:
                log_cb("失败：%s" % exc)
        if cancel_cb and cancel_cb():
            return False
        if index < len(urls) - 1 and log_cb:
            log_cb("尝试镜像地址…")
    return False


def model_paths() -> tuple[str, str]:
    folder = utils.models_dir()
    return (os.path.join(folder, dwpose.DETECTOR_FILE),
            os.path.join(folder, dwpose.POSE_FILE))


def models_present() -> bool:
    det, pose = model_paths()
    return os.path.isfile(det) and os.path.isfile(pose)


def find_local_models(root: str):
    """在给定目录里递归查找 DWPose 模型文件（复用已有的 ComfyUI / DWPose 安装）。"""
    if not root or not os.path.isdir(root):
        return None
    det = pose = None
    for dirpath, _dirs, files in os.walk(root):
        for name in files:
            lower = name.lower()
            if lower == dwpose.DETECTOR_FILE:
                det = det or os.path.join(dirpath, name)
            elif lower == dwpose.POSE_FILE:
                pose = pose or os.path.join(dirpath, name)
        if det and pose:
            break
    if det and pose:
        return det, pose
    return None


def copy_into_models_dir(sources, log_cb=None) -> tuple[str, str]:
    """把外部模型复制到插件的模型目录；复制失败时直接使用原路径。"""
    det_src, pose_src = sources
    det_dst, pose_dst = model_paths()
    os.makedirs(os.path.dirname(det_dst), exist_ok=True)

    result = []
    for src, dst, kind in ((det_src, det_dst, "detector"), (pose_src, pose_dst, "pose")):
        if os.path.abspath(src) == os.path.abspath(dst) or not os.path.isfile(src):
            result.append(dst if os.path.isfile(dst) else src)
            continue
        try:
            if os.path.exists(dst):
                os.remove(dst)
            shutil.copy2(src, dst)
            if log_cb:
                log_cb("已复制 %s 模型：%s" % (kind, dst))
            result.append(dst)
        except Exception as exc:  # noqa: BLE001
            if log_cb:
                log_cb("复制 %s 模型失败（改为直接引用原文件）：%s" % (kind, exc))
            result.append(src)
    return result[0], result[1]

