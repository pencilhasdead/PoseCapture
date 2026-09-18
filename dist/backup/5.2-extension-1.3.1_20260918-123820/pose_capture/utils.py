"""通用工具：路径、日志、Blender 图像读取、依赖查找。

本模块不依赖 onnxruntime / cv2，任何情况下都可以安全导入。
同时也刻意不「硬依赖」bpy：独立工具（pose_to_controlnet）会直接复用本模块，
那时没有 Blender，bpy 为 None。
"""
from __future__ import annotations

import os
import sys

import numpy as np

try:
    import bpy
except ImportError:  # 独立工具环境（没有 Blender）
    bpy = None  # type: ignore[assignment]

ADDON_ID = "pose_capture"
LOG_PREFIX = "[PoseCapture]"


def has_blender() -> bool:
    """当前是否运行在 Blender 内（独立工具里为 False）。"""
    return bpy is not None


def require_blender(feature: str) -> None:
    if bpy is None:
        raise RuntimeError("%s 需要在 Blender 中运行（当前是独立环境）" % feature)


# --------------------------------------------------------------------------- 日志
def log(*args) -> None:
    print(LOG_PREFIX, *args)


def warn(*args) -> None:
    print(LOG_PREFIX, "WARNING:", *args)


def error(*args) -> None:
    print(LOG_PREFIX, "ERROR:", *args)


# --------------------------------------------------------------------------- 路径
def user_data_dir(*subdirs: str) -> str:
    """返回插件的数据目录（放在 Blender 的用户数据目录里，避免污染扩展目录）。

    没有 Blender 时退回：环境变量 POSE_CAPTURE_DATA，否则 %LOCALAPPDATA%/PoseCapture。
    """
    if bpy is None:
        root = os.environ.get("POSE_CAPTURE_DATA") or os.path.join(
            os.environ.get("LOCALAPPDATA") or os.path.expanduser("~"), "PoseCapture")
        path = os.path.join(root, *subdirs) if subdirs else root
        os.makedirs(path, exist_ok=True)
        return path

    rel = os.path.join(ADDON_ID, *subdirs) if subdirs else ADDON_ID
    path = bpy.utils.user_resource("DATAFILES", path=rel, create=True)
    return str(path)


def models_dir() -> str:
    path = user_data_dir("models")
    os.makedirs(path, exist_ok=True)
    return path


def libs_dir() -> str:
    path = user_data_dir("libs")
    os.makedirs(path, exist_ok=True)
    return path


def ensure_libs_on_path() -> str | None:
    """把 pip 安装目录追加到 sys.path 末尾（末尾=让 Blender 自带的 numpy 优先）。

    独立工具（没有 Blender）里返回 None 而不是报错：依赖目录由工具自己
    （pose_to_controlnet/pose_to_cn/paths.py）管理，这里的调用方（dwpose 里
    建后端时）不应该因为缺 Blender 而失败。
    """
    if bpy is None:
        return None
    path = libs_dir()
    if path not in sys.path:
        sys.path.append(path)
    return path


def python_executable() -> str | None:
    """定位 Blender 自带的 python 解释器，用于执行 pip。

    独立环境里直接返回当前解释器（sys.executable）。
    """
    if bpy is None:
        return sys.executable

    candidates: list[str] = []

    binary = getattr(bpy.app, "binary_path", "") or ""
    if binary:
        blender_dir = os.path.dirname(os.path.abspath(binary))
        version = "%d.%d" % (bpy.app.version[0], bpy.app.version[1])
        for name in ("python.exe", "python"):
            candidates.append(os.path.join(blender_dir, version, "python", "bin", name))
        # Linux / macOS 布局
        candidates.append(os.path.join(blender_dir, version, "python", "bin", "python3"))
        for name in ("python3.13", "python3.12", "python3.11"):
            candidates.append(os.path.join(blender_dir, version, "python", "bin", name))

    legacy = getattr(bpy.app, "binary_path_python", None)
    if legacy:
        candidates.append(legacy)

    for path in candidates:
        if path and os.path.isfile(path):
            return path
    return None


# --------------------------------------------------------------------------- 图像
def image_datablock_to_rgb8(image: "bpy.types.Image") -> np.ndarray:
    """把 Blender 图像读取为 (H, W, 3) 的 uint8 数组（第 0 行 = 图像顶部）。

    读取前把色彩空间设为 Non-Color，这样拿到的是文件里的原始 sRGB 数值，
    与 OpenCV 的 imread 结果一致（模型就是按这个训练/导出的）。
    """
    require_blender("读取 Blender 图像（image_datablock_to_rgb8）")

    if image.size[0] == 0 or image.size[1] == 0:
        raise ValueError("图像尺寸为 0，无法读取：%s" % image.name)

    changed_space = False
    try:
        if image.colorspace_settings.name != "Non-Color":
            image.colorspace_settings.name = "Non-Color"
            changed_space = True
    except Exception:  # 某些生成图像没有色彩空间设置
        pass

    width, height = int(image.size[0]), int(image.size[1])
    buf = np.empty(width * height * 4, dtype=np.float32)
    image.pixels.foreach_get(buf)

    if changed_space:
        try:
            image.colorspace_settings.name = "sRGB"
        except Exception:
            pass

    pixels = buf.reshape(height, width, 4)[::-1]  # Blender 从下往上存，翻转
    rgb = np.clip(pixels[:, :, :3] * 255.0 + 0.5, 0.0, 255.0).astype(np.uint8)
    return np.ascontiguousarray(rgb)


def load_image_rgb8(filepath: str) -> np.ndarray:
    """从磁盘读取图片，返回 (H, W, 3) uint8（内部用 Blender 的 loader，无需 PIL）。"""
    require_blender("读取图片文件（load_image_rgb8）")

    if not os.path.isfile(filepath):
        raise FileNotFoundError("找不到图片文件：%s" % filepath)

    image = bpy.data.images.load(filepath, check_existing=False)
    try:
        return image_datablock_to_rgb8(image)
    finally:
        try:
            bpy.data.images.remove(image)
        except Exception:
            pass


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
