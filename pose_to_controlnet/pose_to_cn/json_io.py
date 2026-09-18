"""OpenPose JSON 导出（兼容 ControlNet / controlnet_aux 的 openpose JSON 入口）。"""
from __future__ import annotations

import json
import os

import numpy as np

from . import core_loader
from .render import to_openpose18

# 未被采信的点统一写成 0,0,0（与 ControlNet 的 python annotator 一致）
MISSING = [0.0, 0.0, 0.0]


def _flat(points: np.ndarray, conf: np.ndarray, score_thr: float) -> list[float]:
    out: list[float] = []
    for index in range(len(conf)):
        if conf[index] >= score_thr:
            out.extend([float(points[index][0]), float(points[index][1]),
                        float(conf[index])])
        else:
            out.extend(MISSING)
    return out


def person_entry(keypoints: np.ndarray, scores: np.ndarray, score_thr: float) -> dict:
    """单个人的 JSON 条目。"""
    core = core_loader.core()
    coco = core.coco
    points, conf = to_openpose18(keypoints, scores)

    left = slice(coco.HAND_LEFT_START, coco.HAND_LEFT_START + coco.HAND_COUNT)
    right = slice(coco.HAND_RIGHT_START, coco.HAND_RIGHT_START + coco.HAND_COUNT)
    return {
        "pose_keypoints_2d": _flat(points, conf, score_thr),
        "face_keypoints_2d": _flat(keypoints[coco.FACE_ALL], scores[coco.FACE_ALL],
                                   score_thr),
        "hand_left_keypoints_2d": _flat(keypoints[left], scores[left], score_thr),
        "hand_right_keypoints_2d": _flat(keypoints[right], scores[right], score_thr),
        "pose_keypoints_3d": [],
        "face_keypoints_3d": [],
        "hand_left_keypoints_3d": [],
        "hand_right_keypoints_3d": [],
    }


def openpose_dict(people, width: int, height: int, score_thr: float = 0.3) -> dict:
    """people: [(keypoints, scores), ...]，返回 ControlNet 能读的字典。"""
    return {
        "canvas_width": int(width),
        "canvas_height": int(height),
        "people": [person_entry(kps, scores, score_thr) for kps, scores in people],
    }


def write_json(path: str, data: dict) -> str:
    folder = os.path.dirname(os.path.abspath(path))
    os.makedirs(folder, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False)
    return path
