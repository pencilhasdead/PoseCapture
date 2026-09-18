"""模块导入 / 语法自检：把所有模块 import 一遍，确认没有语法与导入错误。

用法：
  blender.exe -b --factory-startup --python "dev_tests/test_import.py"
"""
import os
import sys

import bpy

HERE = os.path.dirname(os.path.abspath(__file__))
ADDON_PARENT = os.path.dirname(HERE)
if ADDON_PARENT not in sys.path:
    sys.path.insert(0, ADDON_PARENT)

MODULES = ("utils", "coco", "imops", "dwpose", "deps", "rigify_map",
           "reconstruction", "retarget", "props", "operators", "ui")


def main():
    import importlib

    failures = []
    for name in MODULES:
        try:
            importlib.import_module("pose_capture.%s" % name)
        except Exception as exc:  # noqa: BLE001
            failures.append((name, repr(exc)))
    import pose_capture

    # 注册 / 注销各一次，确认类定义没问题
    try:
        pose_capture.register()
        ops = [n for n in dir(bpy.ops.pose_capture) if not n.startswith("_")]
        panels = [n for n in dir(bpy.types) if n.startswith("PC_PT_")]
        pose_capture.unregister()
    except Exception as exc:  # noqa: BLE001
        failures.append(("register", repr(exc)))
        ops, panels = [], []

    print("模块数：%d/%d" % (len(MODULES) - len(failures), len(MODULES)))
    print("操作器 %d 个：%s" % (len(ops), ops))
    print("面板 %d 个：%s" % (len(panels), panels))
    for name, error in failures:
        print("!! %s: %s" % (name, error))
    ok = not failures and len(ops) >= 12 and len(panels) == 3
    print("结果：", "通过 ✓" if ok else "未通过 ✗")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
