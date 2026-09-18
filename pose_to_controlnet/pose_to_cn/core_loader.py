"""把插件的纯 numpy 核心（coco / imops / dwpose / utils）加载成独立包复用。

插件的 __init__.py 依赖 bpy，无法直接 import，所以这里按文件路径加载成
合成包 `posecore`（四个子模块一次性注册，保证相对 import 能互相找到）。
这样骨骼定义、图像预处理、ONNX 推理只有一份实现。
"""
from __future__ import annotations

import importlib.util
import os
import sys
import types

from . import paths

PKG_NAME = "posecore"
MODULE_NAMES = ("utils", "coco", "imops", "dwpose", "deps")
_core: types.ModuleType | None = None


class CoreNotFound(RuntimeError):
    pass


def addon_dir() -> str:
    folder = paths.find_addon_dir()
    if not folder:
        raise CoreNotFound(
            "找不到 pose_capture 插件目录（需要其中的 coco.py / imops.py / dwpose.py）。\n"
            "请把本工具放在插件仓库里（与 pose_capture/ 同级），"
            "或用环境变量 POSE_CAPTURE_ADDON 指定目录。")
    return folder


def load(force: bool = False) -> types.ModuleType:
    """加载并缓存 `posecore` 包（含 coco / imops / dwpose / utils 子模块）。"""
    global _core
    if _core is not None and not force:
        return _core

    folder = addon_dir()
    package = types.ModuleType(PKG_NAME)
    package.__path__ = [folder]          # 让它看起来是个包（相对 import 需要）
    package.__doc__ = "pose_capture 的纯 numpy 核心（由 pose_to_controlnet 复用）"
    sys.modules[PKG_NAME] = package

    pending: list[tuple[object, types.ModuleType]] = []
    for name in MODULE_NAMES:
        filepath = os.path.join(folder, name + ".py")
        if not os.path.isfile(filepath):
            raise CoreNotFound("插件目录缺少 %s：%s" % (name + ".py", folder))
        spec = importlib.util.spec_from_file_location("%s.%s" % (PKG_NAME, name), filepath)
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        setattr(package, name, module)   # 先注册，再执行，支持互相 import
        pending.append((spec, module))

    for spec, module in pending:
        try:
            spec.loader.exec_module(module)
        except Exception as exc:  # noqa: BLE001 - 补一层上下文，便于排查
            raise CoreNotFound("加载 %s 失败：%s" % (spec.name, exc)) from exc

    _core = package
    return package


def core() -> types.ModuleType:
    return load()


def clear() -> None:
    """卸载（测试用）。"""
    global _core
    for name in list(sys.modules):
        if name == PKG_NAME or name.startswith(PKG_NAME + "."):
            sys.modules.pop(name, None)
    _core = None
