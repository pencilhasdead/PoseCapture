"""图片读写：读用 PIL（可选）或 tkinter 兜底，写用纯 numpy 的 PNG 编码器。

写 PNG 不依赖任何第三方库，这样预览和保存走同一条代码路径，预览正常=文件正常。
"""
from __future__ import annotations

import os
import struct
import zlib

import numpy as np

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff", ".gif")


# --------------------------------------------------------------------------- 读
def _read_with_pil(path: str):
    try:
        from PIL import Image  # type: ignore
    except ImportError:
        return None

    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _read_with_tk(path: str):
    """退路：Tk 8.6 自带 PNG/GIF/BMP/PPM 解码（需要能创建 Tk 对象）。"""
    import tkinter as tk

    root = tk._default_root  # type: ignore[attr-defined]
    created = False
    if root is None:
        root = tk.Tk()
        root.withdraw()
        created = True
    try:
        photo = tk.PhotoImage(master=root, file=path)
        width, height = photo.width(), photo.height()
        data = np.array(photo.get(0, 0, width - 1, height - 1), dtype=np.uint8)
        return np.ascontiguousarray(data.reshape(height, width, 3))
    finally:
        if created:
            root.destroy()


def read_image(path: str) -> np.ndarray:
    """读入图片，返回 (H, W, 3) uint8（第 0 行 = 图像顶部）。"""
    if not os.path.isfile(path):
        raise FileNotFoundError("找不到图片：%s" % path)

    rgb = _read_with_pil(path)
    if rgb is not None:
        return rgb
    try:
        return _read_with_tk(path)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "无法解码图片 %s：%s\n（建议安装 pillow：本工具「安装依赖」或 "
            "pip install pillow）" % (os.path.basename(path), exc)) from exc


# --------------------------------------------------------------------------- 写
def _png_chunk(tag: bytes, data: bytes) -> bytes:
    return (struct.pack(">I", len(data)) + tag + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))


def png_bytes(image: np.ndarray, compression: int = 6) -> bytes:
    """把 uint8 数组编码成 PNG 字节流（支持灰/RGB/RGBA）。"""
    array = np.ascontiguousarray(image, dtype=np.uint8)
    if array.ndim == 2:
        array = array[:, :, None]
    if array.ndim != 3 or array.shape[2] not in (1, 3, 4):
        raise ValueError("只支持 (H,W) 或 (H,W,1/3/4) 的图像，收到 %s" % (array.shape,))

    height, width, channels = array.shape
    color_type = {1: 0, 3: 2, 4: 6}[channels]
    raw = bytearray()
    for row in range(height):
        raw.append(0)                                  # 每行的 filter 类型 = None
        raw += array[row].tobytes()

    header = struct.pack(">IIBBBBB", width, height, 8, color_type, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n"
            + _png_chunk(b"IHDR", header)
            + _png_chunk(b"IDAT", zlib.compress(bytes(raw), compression))
            + _png_chunk(b"IEND", b""))


def write_png(path: str, image: np.ndarray) -> str:
    folder = os.path.dirname(os.path.abspath(path))
    os.makedirs(folder, exist_ok=True)
    with open(path, "wb") as handle:
        handle.write(png_bytes(image))
    return path


def write_image(path: str, image: np.ndarray) -> str:
    """按扩展名保存；非 png 时用 PIL（没装就报错提示）。"""
    suffix = os.path.splitext(path)[1].lower()
    if suffix in (".png", ""):
        return write_png(path, image)
    try:
        from PIL import Image  # type: ignore
    except ImportError as exc:
        raise RuntimeError("保存 %s 需要 pillow；也可以改存 .png" % suffix) from exc
    folder = os.path.dirname(os.path.abspath(path))
    os.makedirs(folder, exist_ok=True)
    Image.fromarray(np.asarray(image, dtype=np.uint8)).save(path)
    return path


def list_images(folder: str, recursive: bool = False) -> list[str]:
    """列出目录里的图片文件（按名称排序）。"""
    found: list[str] = []
    if recursive:
        for root, _dirs, files in os.walk(folder):
            for name in files:
                if name.lower().endswith(IMAGE_SUFFIXES):
                    found.append(os.path.join(root, name))
    else:
        for name in os.listdir(folder):
            if name.lower().endswith(IMAGE_SUFFIXES):
                found.append(os.path.join(folder, name))
    return sorted(found)
