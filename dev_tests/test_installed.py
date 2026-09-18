"""安装状态自检：确认 Blender 里“已装的 Pose Capture”就是当前仓库源码（逐文件 hash）。

用法：
  blender.exe -b --factory-startup --python "dev_tests/test_installed.py"

* 只读检查：不改偏好设置、不启用/停用任何附加组件
* 覆盖两种安装方式：扩展目录（extensions/user_default/pose_capture）
  与传统 scripts/addons/pose_capture
"""
import hashlib
import os
import re
import sys

import bpy

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PACKAGE = "pose_capture"


def sha256(path):
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def repo_hashes():
    """仓库源码的 {包内相对路径: sha256}（含扩展清单）。"""
    base = os.path.join(ROOT, PACKAGE)
    result = {}
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        for name in filenames:
            if name.endswith((".pyc", ".pyo")):
                continue
            full = os.path.join(dirpath, name)
            result[os.path.relpath(full, base).replace(os.sep, "/")] = sha256(full)
    manifest = os.path.join(ROOT, "blender_manifest.toml")
    if os.path.isfile(manifest):
        result["blender_manifest.toml"] = sha256(manifest)
    return result


def installed_hashes(path):
    result = {}
    for dirpath, dirnames, filenames in os.walk(path):
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        for name in filenames:
            if name.endswith((".pyc", ".pyo")):
                continue
            full = os.path.join(dirpath, name)
            result[os.path.relpath(full, path).replace(os.sep, "/")] = sha256(full)
    return result


def find_installs():
    """返回 [(说明, 路径)]，优先扩展目录。"""
    found = []
    ext_root = bpy.utils.user_resource("EXTENSIONS", path="user_default")
    candidates = []
    if ext_root:
        candidates.append(("extension:user_default", os.path.join(ext_root, PACKAGE)))
    scripts = bpy.utils.user_resource("SCRIPTS")
    if scripts:
        candidates.append(("legacy:scripts/addons", os.path.join(scripts, "addons", PACKAGE)))
    for label, path in candidates:
        if os.path.isdir(path):
            found.append((label, path))
    return found


def describe(path):
    """从已装副本里读出 __init__.py 的版本号。"""
    text = open(os.path.join(path, "__init__.py"), encoding="utf-8").read()
    start = text.find('"version"')
    if start < 0:
        return "?"
    digits = []
    for char in text[start:start + 60]:
        if char.isdigit():
            digits.append(char)
        elif len(digits) >= 3:
            break
    return ".".join(digits[:3]) if len(digits) >= 3 else "?"


def main():
    installs = find_installs()
    print("Blender：", bpy.app.version_string)
    if not installs:
        print("!! 没找到已安装的 pose_capture（扩展目录 / scripts/addons 都没有）")
        return 1

    expect = repo_hashes()
    ok = True
    for label, path in installs:
        got = installed_hashes(path)
        missing = sorted(set(expect) - set(got))
        extra = sorted(set(got) - set(expect)) if label.startswith("extension") else []
        different = sorted(n for n in set(expect) & set(got) if expect[n] != got[n])
        print("")
        print("[%s] %s" % (label, path))
        print("  bl_info 版本：", describe(path))
        print("  文件数：仓库 %d / 已装 %d" % (len(expect), len(got)))
        stale = False
        if missing:
            print("  !! 缺少：%s" % missing)
        if different:
            print("  !! 内容不一致（已装的是旧版）：%s" % different)
            stale = True
        if extra:
            print("  !! 多出：%s" % extra)
        if missing or different or extra:
            ok = False
        else:
            print("  与仓库源码完全一致 ✓")
        if label.startswith("extension") and not stale and not missing:
            # 真正 import 一次已装副本，确认能注册（用 bl_ext 之外的普通包名）
            if ext_root_path(path) not in sys.path:
                sys.path.insert(0, ext_root_path(path))
            import importlib

            for name in list(sys.modules):
                if name == PACKAGE or name.startswith(PACKAGE + "."):
                    del sys.modules[name]
            module = importlib.import_module(PACKAGE)
            module.register()
            ops = [n for n in dir(bpy.ops.pose_capture) if not n.startswith("_")]
            settings = bpy.context.scene.pose_capture
            keys = set(settings.bl_rna.properties.keys())
            pref_keys = set(module.props.PC_Preferences.bl_rna.properties.keys())
            flags = [name for name in ("depth_mode", "depth_focal", "depth_flip")
                     if name in keys]
            pref_flags = [name for name in ("depth_variant", "local_depth_model")
                          if name in pref_keys]
            stale = [name for name in ("use_depth", "elbow_back", "arm_flip_L",
                                       "arm_flip_R", "knee_front", "spine_lean_front")
                     if name in keys]
            # 面板里 icon= 的取值、以及 draw() 引用的属性必须真实存在
            # （这两类错误只在 Blender 绘制面板时才抛异常，所以在这里静态拦住）
            allowed_icons = icon_identifiers()
            bad_icons = sorted(name for name in source_icons(path)
                               if allowed_icons and name not in allowed_icons)
            bad_attrs = []
            for line, attr in source_attr_uses(path, "ui.py", r"settings\.([A-Za-z_]\w*)"):
                if attr not in keys:
                    bad_attrs.append("ui.py:%d settings.%s" % (line, attr))
            for line, attr in source_attr_uses(path, "props.py", r"self\.([A-Za-z_]\w*)"):
                if attr != "layout" and attr not in pref_keys:
                    bad_attrs.append("props.py:%d self.%s" % (line, attr))
            # 枚举字面量（例如 settings.depth_mode == "DEPTH"）必须是合法取值
            ui_source = open(os.path.join(path, "ui.py"), encoding="utf-8").read()
            for attr, value in re.findall(r'settings\.(\w+)\s*==\s*"([A-Z_]+)"', ui_source):
                prop = settings.bl_rna.properties.get(attr)
                allowed = ({item.identifier for item in prop.enum_items}
                           if prop is not None and prop.type == "ENUM" else None)
                if allowed is not None and value not in allowed:
                    bad_attrs.append('ui.py %s == "%s" 非法（合法：%s）'
                                     % (attr, value, sorted(allowed)))
            module.unregister()
            print("  可注册：操作器 %d 个；场景深度参数：%s；偏好深度参数：%s"
                  % (len(ops), "有 ✓" if len(flags) == 3 else "缺少 ✗ %s" % flags,
                     "有 ✓" if len(pref_flags) == 2 else "缺少 ✗ %s" % pref_flags))
            print("  旧的前后方向开关已移除：%s" % ("是 ✓" if not stale else "否 ✗ %s" % stale))
            print("  面板图标（%d 个）合法：%s"
                  % (len(allowed_icons) and len(source_icons(path)) or 0,
                     "是 ✓" if not bad_icons else "否 ✗ %s" % bad_icons))
            print("  draw() 属性引用：%s"
                  % ("全部存在 ✓" if not bad_attrs else "非法 ✗ %s" % bad_attrs))
            if len(ops) < 15 or len(flags) != 3 or len(pref_flags) != 2 or stale \
                    or bad_icons or bad_attrs:
                ok = False

    print("")
    print("结果：", "通过 ✓" if ok else "未通过 ✗（见到“内容不一致”就先跑 python build_zip.py 再重装）")
    return 0 if ok else 1


def ext_root_path(package_path):
    return os.path.dirname(package_path)


def icon_identifiers():
    """Blender 当前版本允许的图标标识符集合（面板里 icon= 只能用这些）。"""
    try:
        params = bpy.types.UILayout.bl_rna.functions["operator"].parameters
        return {item.identifier for item in params["icon"].enum_items}
    except Exception:  # noqa: BLE001
        return set()


def source_icons(path):
    """扫描副本里所有 icon= 表达式引用的图标名 -> {图标: [文件:行]}。"""
    used = {}
    for dirpath, dirnames, filenames in os.walk(path):
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        for name in sorted(filenames):
            if not name.endswith(".py"):
                continue
            full = os.path.join(dirpath, name)
            with open(full, encoding="utf-8") as handle:
                for index, line in enumerate(handle, 1):
                    if "icon=" not in line:
                        continue
                    tail = line[line.index("icon="):]
                    for token in re.findall(r'"([A-Z0-9_]+)"', tail):
                        used.setdefault(token, []).append(
                            "%s:%d" % (os.path.relpath(full, path), index))
    return used


def source_attr_uses(path, filename, pattern):
    """找出某个源码文件里 <对象>.<属性> 的引用 -> [(行号, 属性)]。"""
    full = os.path.join(path, filename)
    if not os.path.isfile(full):
        return []
    with open(full, encoding="utf-8") as handle:
        return [(index, attr) for index, line in enumerate(handle, 1)
                for attr in re.findall(pattern, line)]


if __name__ == "__main__":
    sys.exit(main())
