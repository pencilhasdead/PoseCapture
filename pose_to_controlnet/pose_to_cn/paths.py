"""路径与设置：工具数据目录、已装依赖、模型目录发现、插件目录发现。

搜索顺序（模型）会优先复用已有的 DWPose 模型，避免重复下载 350MB。
"""
from __future__ import annotations

import glob
import json
import os
import sys

APP_ID = "PoseToControlNet"
DETECTOR_FILE = "yolox_l.onnx"
POSE_FILE = "dw-ll_ucoco_384.onnx"

# pose_to_controlnet/pose_to_cn/paths.py -> pose_to_controlnet/
TOOL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 仓库根目录（与插件 pose_capture/ 同级）
REPO_DIR = os.path.dirname(TOOL_DIR)

_SETTINGS: dict | None = None


# --------------------------------------------------------------------------- 目录
def data_dir() -> str:
    """工具自己的数据目录（依赖、模型、设置）。可用 POSE_TO_CN_HOME 覆盖。"""
    root = os.environ.get("POSE_TO_CN_HOME") or os.path.join(
        os.environ.get("LOCALAPPDATA") or os.path.expanduser("~"), APP_ID)
    os.makedirs(root, exist_ok=True)
    return root


def libs_dir() -> str:
    path = os.path.join(data_dir(), "libs")
    os.makedirs(path, exist_ok=True)
    return path


def models_dir() -> str:
    path = os.path.join(data_dir(), "models")
    os.makedirs(path, exist_ok=True)
    return path


def settings_path() -> str:
    return os.path.join(data_dir(), "settings.json")


# --------------------------------------------------------------------------- 设置
DEFAULT_SETTINGS = {
    "model_dir": "",
    "extra_model_dirs": [],
    "last_input_dir": "",
    "last_output_dir": "",
    "style": "openpose",
    "score_thr": 0.3,
    "det_thr": 0.35,
    "person": 0,
    "hands": True,
    "face": True,
    "feet": False,
    "write_json": True,
    "size_mode": "source",
    "long_side": 1024,
    "margin": 0.10,
}


def load_settings() -> dict:
    global _SETTINGS
    if _SETTINGS is None:
        data = dict(DEFAULT_SETTINGS)
        try:
            with open(settings_path(), "r", encoding="utf-8") as handle:
                stored = json.load(handle)
            if isinstance(stored, dict):
                data.update({k: v for k, v in stored.items() if k in DEFAULT_SETTINGS})
        except (OSError, ValueError):
            pass
        _SETTINGS = data
    return dict(_SETTINGS)


def save_settings(values: dict) -> None:
    global _SETTINGS
    data = load_settings()
    data.update({k: v for k, v in values.items() if k in DEFAULT_SETTINGS})
    _SETTINGS = data
    try:
        with open(settings_path(), "w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2)
    except OSError:
        pass


# --------------------------------------------------------------------------- Blender 插件定位
def _looks_like_addon(path: str) -> bool:
    return all(os.path.isfile(os.path.join(path, name))
               for name in ("coco.py", "imops.py", "dwpose.py", "utils.py"))


def find_addon_dir() -> str | None:
    """找到 pose_capture 插件的目录（用于复用它的纯 numpy 核心模块）。"""
    candidates: list[str] = []

    override = os.environ.get("POSE_CAPTURE_ADDON")
    if override:
        candidates.append(override)

    candidates.append(os.path.join(REPO_DIR, "pose_capture"))
    candidates.append(os.path.join(TOOL_DIR, "_vendor", "pose_capture"))

    for base_key in ("APPDATA", "LOCALAPPDATA"):
        base = os.environ.get(base_key)
        if not base:
            continue
        root = os.path.join(base, "Blender Foundation", "Blender")
        if not os.path.isdir(root):
            continue
        candidates.append(os.path.join(root, "*", "scripts", "addons", "pose_capture"))
        candidates.append(os.path.join(root, "*", "scripts", "addons_core", "pose_capture"))
        candidates.append(os.path.join(root, "*", "extensions", "*", "*", "pose_capture"))
        candidates.append(os.path.join(root, "*", "datafiles", "pose_capture", "source"))

    for pattern in candidates:
        if "*" in pattern:
            for hit in sorted(glob.glob(pattern), reverse=True):
                if _looks_like_addon(hit):
                    return hit
        elif _looks_like_addon(pattern):
            return pattern
    return None


def addon_data_dirs() -> list[str]:
    """插件的数据目录（里面可能有已下载的模型 / 已安装的 onnxruntime）。"""
    found: list[str] = []
    for addon in (find_addon_dir(), os.environ.get("POSE_CAPTURE_DATA")):
        if addon:
            found.append(addon)
    for base_key in ("APPDATA", "LOCALAPPDATA"):
        base = os.environ.get(base_key)
        if not base:
            continue
        pattern = os.path.join(base, "Blender Foundation", "Blender", "*",
                              "datafiles", "pose_capture")
        found.extend(sorted(glob.glob(pattern), reverse=True))

    unique: list[str] = []
    for path in found:
        if path not in unique and os.path.isdir(path):
            unique.append(path)
    return unique


# --------------------------------------------------------------------------- 依赖搜索
def ensure_libs_on_path() -> list[str]:
    """工具依赖目录放到 sys.path 最前，插件已装的 onnxruntime 作为兜底追加。"""
    added: list[str] = []
    first = libs_dir()
    if os.path.isdir(first):
        if first in sys.path:
            sys.path.remove(first)
        sys.path.insert(0, first)
        added.append(first)

    for base in addon_data_dirs():
        libs = os.path.join(base, "libs")
        if os.path.isdir(libs) and libs not in sys.path:
            sys.path.append(libs)
            added.append(libs)
    return added


# --------------------------------------------------------------------------- 模型搜索
def candidate_model_dirs() -> list[str]:
    settings = load_settings()
    dirs: list[str] = []

    def push(path: str) -> None:
        path = os.path.expandvars(os.path.expanduser(path or ""))
        if path and path not in dirs:
            dirs.append(path)

    override = os.environ.get("POSE_TO_CN_MODELS")
    if override:
        push(override)
    push(settings.get("model_dir") or "")
    for extra in settings.get("extra_model_dirs") or []:
        push(extra)
    push(models_dir())
    for base in addon_data_dirs():
        push(os.path.join(base, "models"))
        push(base)  # 有些安装把模型直接放在数据目录根部
    return [path for path in dirs if os.path.isdir(path)]


def find_models() -> tuple[str, str] | None:
    """返回 (检测器, 姿态模型) 路径；没找到返回 None。"""
    for folder in candidate_model_dirs():
        det = os.path.join(folder, DETECTOR_FILE)
        pose = os.path.join(folder, POSE_FILE)
        if os.path.isfile(det) and os.path.isfile(pose):
            return det, pose
    return None


def models_status() -> dict:
    found = find_models()
    return {
        "found": bool(found),
        "detector": found[0] if found else None,
        "pose": found[1] if found else None,
        "searched": candidate_model_dirs(),
    }
