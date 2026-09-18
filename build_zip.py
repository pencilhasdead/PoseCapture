"""打包脚本：生成 dist/ 下的传统附加组件 zip 与 Blender 扩展 zip。

用法：
    python build_zip.py
"""
from __future__ import annotations

import os
import re
import zipfile

ROOT = os.path.dirname(os.path.abspath(__file__))
PACKAGE = "pose_capture"
DIST = os.path.join(ROOT, "dist")
MANIFEST = os.path.join(ROOT, "blender_manifest.toml")


def read_version() -> str:
    text = open(os.path.join(ROOT, PACKAGE, "__init__.py"), encoding="utf-8").read()
    match = re.search(r'"version":\s*\((\d+),\s*(\d+),\s*(\d+)\)', text)
    return ".".join(match.groups()) if match else "0.0.0"


def package_files() -> list[tuple[str, str]]:
    """返回 [(绝对路径, 包内相对路径)]，跳过 __pycache__。"""
    files = []
    for dirpath, dirnames, filenames in os.walk(os.path.join(ROOT, PACKAGE)):
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        for name in filenames:
            if name.endswith((".pyc", ".pyo")):
                continue
            absolute = os.path.join(dirpath, name)
            relative = os.path.relpath(absolute, os.path.join(ROOT, PACKAGE))
            files.append((absolute, relative.replace(os.sep, "/")))
    return sorted(files, key=lambda item: item[1])


def build_legacy(version: str) -> str:
    """传统附加组件：zip 里是 pose_capture/ 目录，解压到 scripts/addons。"""
    target = os.path.join(DIST, "pose_capture-%s-legacy.zip" % version)
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        for absolute, relative in package_files():
            archive.write(absolute, "%s/%s" % (PACKAGE, relative))
    return target


def build_extension(version: str) -> str:
    """扩展：zip 里是 pose_capture/ 目录，内含 blender_manifest.toml + 模块文件。"""
    target = os.path.join(DIST, "pose_capture-%s-extension.zip" % version)
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write(MANIFEST, "%s/blender_manifest.toml" % PACKAGE)
        for absolute, relative in package_files():
            archive.write(absolute, "%s/%s" % (PACKAGE, relative))
    return target


def main() -> None:
    os.makedirs(DIST, exist_ok=True)
    version = read_version()
    for path in (build_legacy(version), build_extension(version)):
        print("%s  (%.1f KB)" % (path, os.path.getsize(path) / 1024.0))
    print("")
    print("安装：Blender > Edit > Preferences > Add-ons > (右上角下拉) > Install from Disk...")


if __name__ == "__main__":
    main()
