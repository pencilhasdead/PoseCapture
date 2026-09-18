"""安装自检：用 Blender 自己的“从磁盘安装”把打包好的 zip 装进来并启用。

用法（先跑 build_zip.py）：
  blender.exe -b --factory-startup --python "dev_tests/test_install.py"
"""
import glob
import os
import sys

import addon_utils
import bpy

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def find_zip(extension: bool) -> str | None:
    pattern = "pose_capture-*-%s.zip" % ("extension" if extension else "legacy")
    found = sorted(glob.glob(os.path.join(ROOT, "dist", pattern)))
    return found[-1] if found else None


def addon_modules():
    return [module.__name__ for module in addon_utils.modules()
            if "pose_capture" in module.__name__]


def cleanup_previous():
    """清掉之前可能装过的副本，保证这次是干净安装。"""
    import shutil

    removed = []
    legacy = os.path.join(bpy.utils.user_resource("SCRIPTS"), "addons", "pose_capture")
    if os.path.isdir(legacy):
        shutil.rmtree(legacy, ignore_errors=True)
        removed.append(legacy)
    for repo in ("user_default", "blender_org"):
        base = bpy.utils.user_resource("EXTENSIONS", path=repo)
        if base and os.path.isdir(base):
            for name in os.listdir(base):
                if "pose_capture" in name:
                    path = os.path.join(base, name)
                    if os.path.isdir(path):
                        shutil.rmtree(path, ignore_errors=True)
                        removed.append(path)
    for module in addon_modules():
        try:
            bpy.ops.preferences.addon_disable(module=module)
        except Exception:  # noqa: BLE001
            pass
    return removed


def main():
    import zipfile

    extension_zip = find_zip(True)
    legacy_zip = find_zip(False)
    if not (extension_zip or legacy_zip):
        print("!! 没有找到 dist/*.zip，先跑 python build_zip.py")
        return 1

    print("清理旧副本：", cleanup_previous())
    installed = None

    # 方式 A：扩展目录（4.2+ 推荐），直接把 zip 解到 extensions/user_default
    ext_root = bpy.utils.user_resource("EXTENSIONS", path="user_default")
    if extension_zip and ext_root:
        try:
            os.makedirs(ext_root, exist_ok=True)
            with zipfile.ZipFile(extension_zip) as archive:
                archive.extractall(ext_root)
            installed = "extension:" + extension_zip
            print("已解压扩展到：", ext_root)
        except Exception as exc:  # noqa: BLE001
            print("扩展方式失败（改试从磁盘安装）：%s" % exc)
            installed = None

    if installed is None and legacy_zip:
        try:
            result = bpy.ops.preferences.addon_install(filepath=legacy_zip, overwrite=True)
            print("从磁盘安装 %s -> %s" % (os.path.basename(legacy_zip), result))
            installed = "legacy:" + legacy_zip
        except Exception as exc:  # noqa: BLE001
            print("!! 安装失败：%s" % exc)
            return 1

    if installed is None:
        print("!! 两种方式都失败")
        return 1
    print("安装方式：", installed)

    modules = addon_modules()
    print("addon_utils 里看到的模块：", modules)
    if not modules:
        print("!! 安装后找不到模块")
        return 1

    module = modules[-1]
    try:
        bpy.ops.preferences.addon_enable(module=module)
    except Exception as exc:  # noqa: BLE001
        print("!! 启用失败：", exc)
        return 1
    print("已启用：", module)

    ops = [name for name in dir(bpy.ops.pose_capture) if not name.startswith("_")]
    panels = [name for name in dir(bpy.types) if name.startswith("PC_PT_")]
    print("操作器：", ops)
    print("面板：", panels)
    scene_ok = hasattr(bpy.types.Scene, "pose_capture")
    print("Scene.pose_capture 已注册：", scene_ok)

    settings = bpy.context.scene.pose_capture
    prefs = bpy.context.preferences.addons.get(module)
    print("偏好设置可用：", prefs is not None and prefs.preferences is not None)

    import importlib

    addon = importlib.import_module(module)
    addon.props.ensure_mapping_rows(settings)
    models_ready = addon.props.resolve_model_paths()[2]
    print("映射表行数：%d；模型就绪：%s；依赖目录：%s"
          % (len(settings.bone_map), models_ready, addon.utils.libs_dir()))
    print("后端：", addon.deps.backend_status()["onnxruntime"] or
          addon.deps.backend_status()["opencv"] or "未安装")

    ok = (len(ops) >= 10 and len(panels) == 3 and scene_ok and prefs is not None
          and len(settings.bone_map) > 40)
    print("结果：", "通过 ✓" if ok else "未通过 ✗")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
