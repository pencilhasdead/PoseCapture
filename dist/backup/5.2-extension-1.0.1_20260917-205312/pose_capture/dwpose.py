"""DWPose / Easy-DWPose ONNX 推理（检测器 YOLOX-L + 姿态 dw-ll_ucoco_384）。

预处理/后处理完全对齐官方实现（IDEA-Research/DWPose 的 onnx 分支）：
  * 检测器：640x640 letterbox(pad=114)、BGR、0-255 float、YOLOX 解码头
  * 姿态：bbox 中心/尺度(padding=1.25) -> 固定长宽比 -> 仿射裁剪 -> ImageNet 归一化
          -> SimCC 解码(split_ratio=2) -> 映射回原图
"""
from __future__ import annotations

import os

import numpy as np

from . import imops, utils

# 模型文件名与下载地址（HuggingFace，官方发布者 yzd-v / hr16）
DETECTOR_FILE = "yolox_l.onnx"
POSE_FILE = "dw-ll_ucoco_384.onnx"

DETECTOR_URLS = [
    "https://huggingface.co/hr16/yolox-onnx/resolve/main/yolox_l.onnx",
    "https://huggingface.co/yzd-v/DWPose/resolve/main/yolox_l.onnx",
]
POSE_URLS = [
    "https://huggingface.co/yzd-v/DWPose/resolve/main/dw-ll_ucoco_384.onnx",
]

DETECTOR_INPUT = (640, 640)
DETECTOR_STRIDES = (8, 16, 32)
PIXEL_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
PIXEL_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)
SIMCC_SPLIT_RATIO = 2.0

_NUMPY_DTYPES = {
    "tensor(float)": np.float32,
    "tensor(float16)": np.float16,
    "tensor(double)": np.float64,
    "tensor(uint8)": np.uint8,
    "tensor(int32)": np.int32,
    "tensor(int64)": np.int64,
}


class ModelNotFound(RuntimeError):
    pass


# --------------------------------------------------------------------------- 后端
class _OrtBackend:
    name = "onnxruntime"

    def __init__(self, path: str, providers=None):
        import onnxruntime as ort

        available = ort.get_available_providers()
        wanted = providers or [p for p in ("CUDAExecutionProvider",
                                           "DmlExecutionProvider",
                                           "TensorrtExecutionProvider",
                                           "CoreMLExecutionProvider",
                                           "CPUExecutionProvider") if p in available]
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(path, sess_options=options, providers=wanted)
        active = self.session.get_providers()
        self.active_provider = active[0] if active else "?"

    def input_spec(self, index=0):
        item = self.session.get_inputs()[index]
        shape = [d if isinstance(d, int) else None for d in item.shape]
        dtype = _NUMPY_DTYPES.get(item.type, np.float32)
        return item.name, shape, dtype

    def output_shapes(self):
        return [list(o.shape) for o in self.session.get_outputs()]

    def run(self, inputs):
        return self.session.run(None, inputs)


class _Cv2Backend:
    name = "opencv-dnn"

    def __init__(self, path: str, providers=None):
        import cv2

        if not hasattr(cv2, "dnn"):
            raise RuntimeError("当前 opencv 版本没有 dnn 模块")
        self.net = cv2.dnn.readNetFromONNX(path)
        self.active_provider = "cpu"

    def input_spec(self, index=0):
        return "images", [None, 3, None, None], np.float32

    def output_shapes(self):
        return []

    def run(self, inputs):
        blob = list(inputs.values())[0]
        self.net.setInput(blob)
        outs = self.net.forward(self.net.getUnconnectedOutLayersNames())
        return [np.asarray(o) for o in outs]


def create_backend(path: str, preferred: str = "auto", providers=None):
    """按偏好创建推理后端；'auto' 时优先 onnxruntime，其次 opencv.dnn。"""
    utils.ensure_libs_on_path()
    errors = []
    order = ["onnxruntime", "opencv"] if preferred == "auto" else [preferred]
    for name in order:
        try:
            if name == "onnxruntime":
                return _OrtBackend(path, providers)
            return _Cv2Backend(path, providers)
        except Exception as exc:  # noqa: PERF203
            errors.append("%s: %s" % (name, exc))
    raise RuntimeError("没有可用的推理后端（%s）。请在插件偏好设置里安装依赖。" % "; ".join(errors))


# --------------------------------------------------------------------------- 检测器
def detector_preprocess(bgr: np.ndarray):
    padded, ratio = imops.letterbox(bgr, DETECTOR_INPUT[1], DETECTOR_INPUT[0], 114)
    blob = np.ascontiguousarray(padded.transpose(2, 0, 1)[None], dtype=np.float32)
    return blob, ratio


def _decode_yolox_heads(prediction: np.ndarray, input_size, p6=False) -> np.ndarray:
    """YOLOX 头部解码，prediction 形状 (1, N, 85)。

    N 可能是所有 stride 拼在一起（8400），也可能是单个 stride（6400/1600/400）。
    """
    prediction = prediction.copy()
    all_strides = [8, 16, 32, 64] if p6 else list(DETECTOR_STRIDES)

    grids, strides = [], []
    for stride in all_strides:
        hsize, wsize = input_size[0] // stride, input_size[1] // stride
        xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
        grids.append(np.stack((xv, yv), 2).reshape(1, -1, 2))
        strides.append(np.full((1, hsize * wsize, 1), stride))

    if prediction.shape[1] == sum(g.shape[1] for g in grids):
        grid = np.concatenate(grids, 1)
        stride_arr = np.concatenate(strides, 1)
    else:
        matched = None
        for grid_one, stride_one in zip(grids, strides):
            if prediction.shape[1] == grid_one.shape[1]:
                matched = (grid_one, stride_one)
                break
        if matched is None:
            raise RuntimeError("YOLOX 输出形状无法解析：%s" % (prediction.shape,))
        grid, stride_arr = matched

    prediction[..., :2] = (prediction[..., :2] + grid) * stride_arr
    prediction[..., 2:4] = np.exp(prediction[..., 2:4]) * stride_arr
    return prediction


def decode_detector(outputs, ratio: float, score_thr: float = 0.3,
                    nms_thr: float = 0.45) -> np.ndarray:
    """把 YOLOX 输出解码成原图坐标下的 xyxy 框（只保留 person 类）。"""
    if len(outputs) == 1:
        predictions = _decode_yolox_heads(np.asarray(outputs[0]), DETECTOR_INPUT)[0]
    else:
        predictions = np.concatenate(
            [_decode_yolox_heads(np.asarray(out), DETECTOR_INPUT)[0] for out in outputs], 0)

    boxes = predictions[:, :4]
    scores_all = predictions[:, 4:5] * predictions[:, 5:]

    xyxy = np.ones_like(boxes)
    xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
    xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
    xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
    xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0
    xyxy /= max(ratio, 1e-6)

    cls_scores = scores_all[:, 0]  # 只要 person 类
    valid = cls_scores > score_thr
    if not valid.any():
        return np.zeros((0, 4), dtype=np.float32)
    picked = imops.nms(xyxy[valid], cls_scores[valid], nms_thr)
    boxes_out = xyxy[valid][picked]
    scores_out = cls_scores[valid][picked]
    order = scores_out.argsort()[::-1]
    return boxes_out[order].astype(np.float32)


# --------------------------------------------------------------------------- 姿态模型
def bbox_xyxy2cs(bbox, padding: float = 1.25):
    x0, y0, x1, y1 = (float(v) for v in bbox[:4])
    center = np.array(((x0 + x1) * 0.5, (y0 + y1) * 0.5), dtype=np.float32)
    scale = np.array(((x1 - x0) * padding, (y1 - y0) * padding), dtype=np.float32)
    return center, scale


def _fix_aspect_ratio(scale: np.ndarray, aspect_ratio: float) -> np.ndarray:
    width, height = float(scale[0]), float(scale[1])
    if width / max(height, 1e-6) < aspect_ratio:
        width = height * aspect_ratio
    else:
        height = width / aspect_ratio
    return np.array((width, height), dtype=np.float32)


def _get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    return np.array((src_point[0] * cs - src_point[1] * sn,
                     src_point[0] * sn + src_point[1] * cs), dtype=np.float32)


def _get_3rd_point(a, b):
    direct = a - b
    return b + np.array((-direct[1], direct[0]), dtype=np.float32)


def get_warp_matrix(center, scale, rot: float, output_size, shift=(0.0, 0.0),
                    inv: bool = False) -> np.ndarray:
    """mmpose 仿射矩阵（等价官方 onnxpose.get_warp_matrix / cv2.getAffineTransform）。"""
    src_w, dst_w, dst_h = float(scale[0]), float(output_size[0]), float(output_size[1])
    rot_rad = np.deg2rad(rot)
    src_dir = _get_dir(np.array((0.0, src_w * -0.5), dtype=np.float32), rot_rad)
    dst_dir = np.array((0.0, dst_w * -0.5), dtype=np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = np.asarray(center, dtype=np.float32) + np.array(shift, dtype=np.float32)
    src[1, :] = src[0, :] + src_dir
    dst[0, :] = (dst_w * 0.5, dst_h * 0.5)
    dst[1, :] = dst[0, :] + dst_dir
    src[2, :] = _get_3rd_point(src[0, :], src[1, :])
    dst[2, :] = _get_3rd_point(dst[0, :], dst[1, :])

    src_pts, dst_pts = (dst, src) if inv else (src, dst)

    # 解 [x y 1] 线性系统，等价于 cv2.getAffineTransform
    mat_src = np.hstack([src_pts, np.ones((3, 1), dtype=np.float32)])
    solution = np.linalg.solve(mat_src, dst_pts).T
    return np.asarray(solution, dtype=np.float32)


def pose_preprocess(bgr: np.ndarray, bbox, input_size):
    """裁剪出单人区域并归一化。input_size = (w, h)。返回 (blob, center, scale)。"""
    out_w, out_h = int(input_size[0]), int(input_size[1])
    center, scale = bbox_xyxy2cs(bbox, padding=1.25)
    scale = _fix_aspect_ratio(scale, aspect_ratio=out_w / float(out_h))
    warp_mat = get_warp_matrix(center, scale, 0.0, output_size=(out_w, out_h))

    cropped = imops.warp_affine(bgr.astype(np.float32), warp_mat, out_w, out_h, 0.0)
    normalized = (cropped - PIXEL_MEAN) / PIXEL_STD
    blob = np.ascontiguousarray(normalized.transpose(2, 0, 1)[None], dtype=np.float32)
    return blob, center, scale


def _simcc_maximum(simcc_x: np.ndarray, simcc_y: np.ndarray):
    n, k, _ = simcc_x.shape
    simcc_x = simcc_x.reshape(n * k, -1)
    simcc_y = simcc_y.reshape(n * k, -1)

    x_locs = np.argmax(simcc_x, axis=1)
    y_locs = np.argmax(simcc_y, axis=1)
    locs = np.stack((x_locs, y_locs), axis=-1).astype(np.float32)
    max_val_x = np.amax(simcc_x, axis=1)
    max_val_y = np.amax(simcc_y, axis=1)

    mask = max_val_x > max_val_y
    max_val_x[mask] = max_val_y[mask]
    vals = max_val_x
    locs[vals <= 0.0] = -1.0
    return locs.reshape(n, k, 2), vals.reshape(n, k)


def decode_simcc(outputs, input_size, center, scale, split_ratio: float = SIMCC_SPLIT_RATIO):
    """SimCC 解码并映射回原图坐标。input_size = (w, h)。"""
    width, height = float(input_size[0]), float(input_size[1])

    tensors = [np.asarray(o) for o in outputs if np.asarray(o).ndim == 3]
    if len(tensors) < 2:
        raise RuntimeError("姿态模型输出无法解析：%s" % ([np.shape(o) for o in outputs],))

    simcc_x = simcc_y = None
    for arr in tensors:
        if abs(arr.shape[-1] - width * split_ratio) <= 1 and simcc_x is None:
            simcc_x = arr
        elif abs(arr.shape[-1] - height * split_ratio) <= 1 and simcc_y is None:
            simcc_y = arr
    if simcc_x is None or simcc_y is None:
        ordered = sorted(tensors, key=lambda a: a.shape[-1])
        simcc_y, simcc_x = ordered[0], ordered[-1]

    keypoints, scores = _simcc_maximum(simcc_x, simcc_y)
    keypoints = keypoints / split_ratio

    scale_arr = np.asarray(scale, dtype=np.float32)
    center_arr = np.asarray(center, dtype=np.float32)
    keypoints = keypoints / np.array((width, height), dtype=np.float32) * scale_arr \
        + center_arr - scale_arr / 2.0
    return keypoints, scores


# --------------------------------------------------------------------------- 高层接口
class DwPoseEstimator:
    """封装检测 + 姿态估计，一次加载、多次调用。"""

    def __init__(self, detector_path: str, pose_path: str, backend: str = "auto",
                 providers=None):
        if not os.path.isfile(detector_path):
            raise ModelNotFound("找不到检测模型：%s" % detector_path)
        if not os.path.isfile(pose_path):
            raise ModelNotFound("找不到姿态模型：%s" % pose_path)

        self.detector = create_backend(detector_path, backend, providers)
        self.pose = create_backend(pose_path, backend, providers)

        name, shape, dtype = self.pose.input_spec()
        self.pose_input_name = name
        self.pose_input_dtype = dtype
        if len(shape) == 4 and shape[2] and shape[3]:
            self.pose_input_size = (int(shape[3]), int(shape[2]))  # (w, h)
        else:
            self.pose_input_size = (288, 384)
        self.detector_input_name = self.detector.input_spec()[0]

        utils.log("推理后端：%s / %s，姿态输入 %s，设备 %s" % (
            self.detector.name, self.pose.name, self.pose_input_size,
            getattr(self.pose, "active_provider", "?")))

    # ---------------------------------------------------------------- 检测
    def detect(self, bgr: np.ndarray, score_thr: float = 0.3, nms_thr: float = 0.45):
        blob, ratio = detector_preprocess(bgr)
        outputs = self.detector.run({self.detector_input_name: blob})
        return decode_detector(outputs, ratio, score_thr, nms_thr)

    # ---------------------------------------------------------------- 姿态
    def estimate(self, bgr: np.ndarray, boxes):
        """对每个框做单人姿态估计，返回 (keypoints (N,133,2), scores (N,133))。"""
        all_kps, all_scores = [], []
        for box in boxes:
            blob, center, scale = pose_preprocess(bgr, box, self.pose_input_size)
            if self.pose_input_dtype != np.float32:
                blob = blob.astype(self.pose_input_dtype)
            outputs = self.pose.run({self.pose_input_name: blob})
            kps, scores = decode_simcc(outputs, self.pose_input_size, center, scale)
            all_kps.append(kps[0])
            all_scores.append(scores[0])
        if not all_kps:
            return (np.zeros((0, 133, 2), dtype=np.float32),
                    np.zeros((0, 133), dtype=np.float32))
        return np.asarray(all_kps, dtype=np.float32), np.asarray(all_scores, dtype=np.float32)

    # ---------------------------------------------------------------- 组合
    def run(self, rgb8: np.ndarray, score_thr: float = 0.3, det_thr: float = 0.3,
            person_index: int = 0):
        """完整流程。rgb8: (H, W, 3) uint8。返回 dict。"""
        bgr = np.ascontiguousarray(rgb8[:, :, ::-1])
        boxes = self.detect(bgr, det_thr)
        if len(boxes) == 0:
            return {"boxes": boxes, "keypoints": None, "scores": None}

        # 按框面积从大到小排序，person_index 指第几个（0 = 最大的人）
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        order = areas.argsort()[::-1]
        boxes = boxes[order]
        if person_index >= len(boxes):
            person_index = 0
        picked = boxes[person_index:person_index + 1]

        kps, scores = self.estimate(bgr, picked)
        return {
            "boxes": boxes,
            "box": picked[0],
            "keypoints": kps[0],
            "scores": scores[0],
            "person_index": person_index,
        }

