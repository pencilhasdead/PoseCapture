"""DWPose（YOLOX-L 检测 + dw-ll_ucoco_384 姿态）ONNX 推理封装。

模型只加载一次，可重复调用；支持单人 / 全部人物。
"""
from __future__ import annotations

import os

import numpy as np

from . import core_loader, paths


class EngineError(RuntimeError):
    pass


class PoseEngine:
    """懒加载的推理引擎（第一次用到时才创建 ONNX session，约 350MB 模型）。"""

    def __init__(self, model_dir: str | None = None, backend: str = "auto"):
        self.model_dir = model_dir
        self.backend = backend
        self._estimator = None
        self.info = {}

    # ---------------------------------------------------------------- 模型
    def resolve_models(self) -> tuple[str, str]:
        if self.model_dir:
            detector = os.path.join(self.model_dir, paths.DETECTOR_FILE)
            pose = os.path.join(self.model_dir, paths.POSE_FILE)
            if os.path.isfile(detector) and os.path.isfile(pose):
                return detector, pose
            raise EngineError("模型目录里缺少 %s / %s：%s"
                              % (paths.DETECTOR_FILE, paths.POSE_FILE, self.model_dir))
        found = paths.find_models()
        if not found:
            raise EngineError(
                "找不到 DWPose 模型（%s + %s）。\n搜索过的目录：\n  %s\n"
                "可以在界面里点「下载模型」，或手动选择模型目录。"
                % (paths.DETECTOR_FILE, paths.POSE_FILE,
                   "\n  ".join(paths.candidate_model_dirs())))
        return found

    def estimator(self):
        if self._estimator is None:
            core = core_loader.core()
            detector, pose = self.resolve_models()
            try:
                self._estimator = core.dwpose.DwPoseEstimator(detector, pose, self.backend)
            except core.dwpose.ModelNotFound as exc:
                raise EngineError(str(exc)) from exc
            self.info = {
                "detector": detector,
                "pose": pose,
                "detector_backend": self._estimator.detector.name,
                "pose_backend": self._estimator.pose.name,
                "provider": getattr(self._estimator.pose, "active_provider", "?"),
                "pose_input": self._estimator.pose_input_size,
            }
        return self._estimator

    # ---------------------------------------------------------------- 推理
    def people(self, rgb: np.ndarray, score_thr: float = 0.3, det_thr: float = 0.35,
               person: int = 0) -> list[tuple[np.ndarray, np.ndarray]]:
        """返回 [(keypoints (133,2), scores (133)), ...]。

        person = -1 表示图里所有检测到的人（按框面积从大到小），否则取第 person 个。
        """
        estimator = self.estimator()
        bgr = np.ascontiguousarray(np.asarray(rgb)[:, :, ::-1])
        boxes = estimator.detect(bgr, det_thr)
        if len(boxes) == 0:
            return []

        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        boxes = boxes[areas.argsort()[::-1]]

        picked = boxes if person < 0 else boxes[person:person + 1]
        if len(picked) == 0:
            picked = boxes[0:1]

        keypoints, scores = estimator.estimate(bgr, picked)
        result = []
        for index in range(len(scores)):
            if float(scores[index][:23].max()) < 0.1:      # 明显是误检的空框
                continue
            result.append((keypoints[index], scores[index]))
        return result

    def detect_count(self, rgb: np.ndarray, det_thr: float = 0.35) -> int:
        estimator = self.estimator()
        bgr = np.ascontiguousarray(np.asarray(rgb)[:, :, ::-1])
        return len(estimator.detect(bgr, det_thr))

    def close(self) -> None:
        self._estimator = None
        self.info = {}
