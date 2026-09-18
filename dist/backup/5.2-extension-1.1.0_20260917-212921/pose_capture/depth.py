"""深度模型（Depth Anything V2 Small，ONNX）：给 DWPose 的 2D 关键点补真实深度。

为什么要它：单张图里“手肘在身体前面还是后面”的画面投影完全一样，靠骨长反解
只能猜（旧版用 手肘向后弯 / 膝盖向前弯 / 左臂前后翻转 这些启发式开关）。有了
深度图，每个关节的深度直接从像素读出来，前后方向不再需要猜。

预处理与 HuggingFace 的 DPTImageProcessor 完全一致：
  * 等比缩放到 518x518 以内，并保证长宽都是 14 的倍数（不补边）
  * ImageNet 均值/方差归一化，NCHW float32
输出与输入同分辨率，数值是**相对深度**（仿射不变：真实深度 = α·d + β）。
α/β 由 reconstruction.py 用“全图所有骨段”做一次全局最小二乘拟合得到，不是逐关节
反解骨长，所以不存在“肘部朝前还是朝后”的歧义。

本模块只负责“模型 -> 深度图 -> 按像素采样”，不认识骨架，可以单独测试。
"""
from __future__ import annotations

import math
import os

import numpy as np

from . import dwpose, imops, utils

# --------------------------------------------------------------------------- 模型
INPUT_SIZE = 518          # 长边缩放到这个尺寸以内
SIZE_MULTIPLE = 14        # ViT patch 尺寸：长宽必须是它的倍数
IMAGE_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGE_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_REMOTE_REPO = "onnx-community/depth-anything-v2-small"
_REMOTE_BASE = "https://huggingface.co/%s/resolve/main/" % _REMOTE_REPO
# huggingface.co 在部分网络（国内）下连不上，镜像放后面兜底
_MIRROR_BASE = "https://hf-mirror.com/%s/resolve/main/" % _REMOTE_REPO

MODEL_VARIANTS = {
    "fp32": {"file": "depth_anything_v2_small.onnx",
             "remote": "onnx/model.onnx", "size_mb": 95,
             "label": "标准精度（约 95 MB）"},
    "fp16": {"file": "depth_anything_v2_small_fp16.onnx",
             "remote": "onnx/model_fp16.onnx", "size_mb": 48,
             "label": "半精度（约 48 MB）"},
    "int8": {"file": "depth_anything_v2_small_int8.onnx",
             "remote": "onnx/model_quantized.onnx", "size_mb": 27,
             "label": "量化（约 27 MB，最快）"},
}
DEFAULT_VARIANT = "fp32"


def variant_ids() -> list[str]:
    return list(MODEL_VARIANTS.keys())


def variant_items() -> list[tuple[str, str, str]]:
    """给 EnumProperty 用：(id, 名称, 说明)。"""
    return [(key, key, item["label"]) for key, item in MODEL_VARIANTS.items()]


def variant_info(variant: str) -> dict:
    return MODEL_VARIANTS.get(variant) or MODEL_VARIANTS[DEFAULT_VARIANT]


def model_filename(variant: str) -> str:
    return variant_info(variant)["file"]


def model_urls(variant: str) -> list[str]:
    """官方地址在前、镜像在后（deps.download_model 会依次尝试）。"""
    remote = variant_info(variant)["remote"]
    return [_REMOTE_BASE + remote, _MIRROR_BASE + remote]


def looks_like_model(filename: str) -> bool:
    """本地导入时的粗筛：名字里带 depth 的 onnx（Depth Anything / MiDaS 都能用）。"""
    lower = os.path.basename(str(filename)).lower()
    return lower.endswith(".onnx") and "depth" in lower


# --------------------------------------------------------------------------- 预处理
def target_size(width: int, height: int) -> tuple[int, int]:
    """对齐 HF DPTImageProcessor：等比缩放 + 向上取整到 14 的倍数，不补边。"""
    width, height = max(int(width), 1), max(int(height), 1)
    ratio = min(INPUT_SIZE / float(height), INPUT_SIZE / float(width))
    new_w = max(SIZE_MULTIPLE, int(round(width * ratio)))
    new_h = max(SIZE_MULTIPLE, int(round(height * ratio)))
    new_w = int(math.ceil(new_w / float(SIZE_MULTIPLE))) * SIZE_MULTIPLE
    new_h = int(math.ceil(new_h / float(SIZE_MULTIPLE))) * SIZE_MULTIPLE
    return new_w, new_h


def preprocess(rgb8: np.ndarray):
    """(H,W,3) uint8 -> (blob(1,3,h,w) float32, 模型输入尺寸 (w,h))。"""
    height, width = rgb8.shape[:2]
    new_w, new_h = target_size(width, height)
    resized = imops.resize(rgb8, new_w, new_h).astype(np.float32) / 255.0
    normalized = (resized - IMAGE_MEAN) / IMAGE_STD
    blob = np.ascontiguousarray(np.transpose(normalized, (2, 0, 1))[None],
                                dtype=np.float32)
    return blob, (new_w, new_h)


class DepthEstimator:
    """一次加载、多次调用；推理后端复用 dwpose.create_backend（onnxruntime 优先）。"""

    def __init__(self, path: str, backend: str = "auto", providers=None):
        if not os.path.isfile(path):
            raise dwpose.ModelNotFound("找不到深度模型：%s" % path)
        self.path = path
        self.session = dwpose.create_backend(path, backend, providers)
        name, _shape, dtype = self.session.input_spec()
        self.input_name = name
        self.input_dtype = dtype
        utils.log("深度模型：%s（后端 %s，设备 %s）"
                  % (os.path.basename(path), self.session.name,
                     getattr(self.session, "active_provider", "?")))

    def predict(self, rgb8: np.ndarray):
        """返回 (深度图 float32 (h,w), 模型输入尺寸 (w,h))。"""
        blob, size = preprocess(rgb8)
        if self.input_dtype != np.float32:
            blob = blob.astype(self.input_dtype)
        outputs = self.session.run({self.input_name: blob})
        values = np.asarray(outputs[0], dtype=np.float32)
        while values.ndim > 2:
            values = values[0]
        return np.ascontiguousarray(values), size


# --------------------------------------------------------------------------- 采样
def _sample_one(values: np.ndarray, x: float, y: float) -> float:
    sampled = imops._bilinear(values, np.array([x], dtype=np.float32),
                              np.array([y], dtype=np.float32))
    return float(np.asarray(sampled).reshape(-1)[0])


def sample_many(values: np.ndarray, model_size, image_size, points,
                window: int = 1) -> np.ndarray:
    """按原图像素坐标采样深度；points 里允许 None（对应位置返回 nan）。

    关节周围取 3x3 中值：肢体边缘处深度图会混进背景深度，中值比单点稳。
    """
    scale_x = float(model_size[0]) / max(float(image_size[0]), 1e-6)
    scale_y = float(model_size[1]) / max(float(image_size[1]), 1e-6)
    out = np.full(len(points), np.nan, dtype=np.float64)
    for index, point in enumerate(points):
        if point is None:
            continue
        u = float(point[0]) * scale_x
        v = float(point[1]) * scale_y
        if window and window > 0:
            samples = [_sample_one(values, u + dx, v + dy)
                       for dx in (-window, 0, window) for dy in (-window, 0, window)]
            out[index] = float(np.median(samples))
        else:
            out[index] = _sample_one(values, u, v)
    return out


def colorize(values: np.ndarray, near_is_large: bool = True) -> np.ndarray:
    """深度图 -> 伪彩色（近=红、远=蓝），用来肉眼检查方向与质量。"""
    array = np.asarray(values, dtype=np.float32)
    low, high = float(np.min(array)), float(np.max(array))
    span = max(high - low, 1e-6)
    norm = np.clip((array - low) / span, 0.0, 1.0)
    if not near_is_large:
        norm = 1.0 - norm
    red = np.clip(2.0 * norm, 0.0, 1.0)
    green = np.clip(1.0 - np.abs(2.0 * norm - 1.0), 0.0, 1.0)
    blue = np.clip(2.0 - 2.0 * norm, 0.0, 1.0)
    return ((np.stack([red, green, blue], axis=-1)) * 255.0 + 0.5).astype(np.uint8)


def preview_image(rgb8: np.ndarray, values: np.ndarray,
                  near_is_large: bool = True) -> np.ndarray:
    """深度伪彩色图放大回原图尺寸，方便和照片对照着看。"""
    height, width = rgb8.shape[:2]
    return imops.resize(colorize(values, near_is_large), width, height)