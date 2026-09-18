"""把仓库里的最新源码装进本机 Blender：替换已安装的 Pose Capture。

用法：
    python dev_tests/install_local.py                # 打包 + 校验 + 备份 + 替换（自动挑目录）
    python dev_tests/install_local.py --dry-run      # 只看会改哪些目录，不动文件
    python dev_tests/install_local.py --version 5.2  # 只处理指定版本的 Blender
    python dev_tests/install_local.py --keep-dist    # 不把 dist/ 里的旧版 zip 挪走

流程：
    1) 调用 build_zip.py 重新打包 dist/*.zip
    2) 校验 zip 内容与 pose_capture/ 源码逐文件 hash 一致（不一致立即中止）
    3) 现有安装备份到 dist/backup/<版本>-<方式>-<旧版本>_<时间戳>/
    4) 覆盖安装：扩展目录优先（extensions/user_default），其次 scripts/addons
    5) 打印安装结果与下一步（重启 Blender 才会加载新代码）

注意：不会碰数据目录 datafiles/pose_capture（模型与 pip 依赖都在那里）。
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import time
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PACKAGE = "pose_capture"
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import build_zip  # noqa: E402


def config_base() -> str:
    """Blender 用户配置目录（含各版本子目录）。"""
    if os.name == "nt":
        appdata = os.environ.get("APPDATA") or os.path.expanduser("~")
        return os.path.join(appdata, "Blender Foundation", "Blender")
    if sys.platform == "darwin":
        return os.path.expanduser("~/Library/Application Support/Blender")
    return os.path.expanduser("~/.config/blender")


def sha256(path: str) -> str:
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def walk_hashes(base: str) -> dict[str, str]:
    """{相对路径: sha256}，跳过 __pycache__ 与 .pyc。"""
    result = {}
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        for name in filenames:
            if name.endswith((".pyc", ".pyo")):
                continue
            full = os.path.join(dirpath, name)
            result[os.path.relpath(full, base).replace(os.sep, "/")] = sha256(full)
    return result


def repo_hashes(with_manifest: bool) -> dict[str, str]:
    result = walk_hashes(os.path.join(ROOT, PACKAGE))
    if with_manifest:
        result["blender_manifest.toml"] = sha256(os.path.join(ROOT, "blender_manifest.toml"))
    return result


def zip_hashes(zip_path: str) -> dict[str, str]:
    with zipfile.ZipFile(zip_path) as archive:
        return {name: hashlib.sha256(archive.read(name)).hexdigest()
                for name in archive.namelist()
                if not name.endswith("/") and name.startswith(PACKAGE + "/")}


def prefixed(hashes: dict[str, str]) -> dict[str, str]:
    """给仓库侧的键加上 zip 内的包名前缀，方便和 zip 内容比对。"""
    return {("%s/%s" % (PACKAGE, name)): digest for name, digest in hashes.items()}


def compare(expect: dict[str, str], got: dict[str, str], label: str) -> bool:
    missing = sorted(set(expect) - set(got))
    extra = sorted(set(got) - set(expect))
    different = sorted(n for n in set(expect) & set(got) if expect[n] != got[n])
    if missing or extra or different:
        print("  %s：不一致 ✗" % label)
        for head, items in (("缺少", missing), ("多出", extra), ("内容不同", different)):
            if items:
                print("    %s：%s" % (head, items))
        return False
    print("  %s：%d 个文件全部一致 ✓" % (label, len(expect)))
    return True


def read_version(path: str) -> str:
    """读已装副本 __init__.py 里的版本号，例如 1.0.0。"""
    init = os.path.join(path, "__init__.py")
    if not os.path.isfile(init):
        return "?"
    text = open(init, encoding="utf-8").read()
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


def find_targets(version: str | None):
    """返回 [(blender 版本, 方式, 安装路径)]；没装过就装到最新版本目录的扩展目录。"""
    base = config_base()
    if not os.path.isdir(base):
        return []
    versions = sorted(name for name in os.listdir(base)
                      if os.path.isdir(os.path.join(base, name))
                      and (version is None or name == version))
    hits = []
    for ver in versions:
        root = os.path.join(base, ver)
        for kind, rel in (("extension", os.path.join("extensions", "user_default")),
                          ("legacy", os.path.join("scripts", "addons"))):
            path = os.path.join(root, rel, PACKAGE)
            if os.path.isdir(path):
                hits.append((ver, kind, path))
    if hits:
        return hits
    if not versions:
        return []
    newest = versions[-1]
    return [(newest, "extension",
             os.path.join(base, newest, "extensions", "user_default", PACKAGE))]


def backup_tree(path: str, dest: str) -> str:
    """把现有安装整体拷到 dest/<包名>。"""
    os.makedirs(dest, exist_ok=True)
    shutil.copytree(path, os.path.join(dest, PACKAGE), dirs_exist_ok=True)
    return dest


def replace_tree(path: str, zip_path: str) -> str:
    """删掉旧目录再解压新包；删不掉（被占用）就退回覆盖写入。"""
    parent = os.path.dirname(path)
    os.makedirs(parent, exist_ok=True)
    mode = "全新安装"
    if os.path.isdir(path):
        try:
            shutil.rmtree(path)
            mode = "先删除再解压"
        except OSError as exc:
            print("  ！旧目录删不掉（%s），改为覆盖写入" % exc)
            mode = "覆盖写入"
            shutil.rmtree(os.path.join(path, "__pycache__"), ignore_errors=True)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(parent)
    shutil.rmtree(os.path.join(path, "__pycache__"), ignore_errors=True)
    return mode


def retire_old_dist(dist: str, keep_version: str, stamp: str) -> list[str]:
    """把 dist/ 里非当前版本的 zip 挪进备份目录，避免和最新版混淆。"""
    moved = []
    if not os.path.isdir(dist):
        return moved
    dest = os.path.join(dist, "backup", "old-dist-%s" % stamp)
    for name in sorted(os.listdir(dist)):
        if not name.endswith(".zip") or ("-%s-" % keep_version) in name:
            continue
        os.makedirs(dest, exist_ok=True)
        shutil.move(os.path.join(dist, name), os.path.join(dest, name))
        moved.append(name)
    return moved


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="把仓库里的最新源码装进本机 Blender（替换已安装的 Pose Capture）")
    parser.add_argument("--dry-run", action="store_true", help="只报告会改什么，不动文件")
    parser.add_argument("--version", help="只处理指定版本的 Blender 目录，例如 5.2")
    parser.add_argument("--keep-dist", action="store_true", help="不挪走 dist/ 里的旧版 zip")
    args = parser.parse_args(argv)

    stamp = time.strftime("%Y%m%d-%H%M%S")
    version = build_zip.read_version()
    print("源码版本：%s；Blender 配置目录：%s" % (version, config_base()))

    print("\n[1/4] 打包")
    build_zip.main()

    dist = os.path.join(ROOT, "dist")
    extension_zip = os.path.join(dist, "pose_capture-%s-extension.zip" % version)
    legacy_zip = os.path.join(dist, "pose_capture-%s-legacy.zip" % version)
    if not (os.path.isfile(extension_zip) and os.path.isfile(legacy_zip)):
        print("!! 打包结果不完整：%s / %s" % (extension_zip, legacy_zip))
        return 1

    print("\n[2/4] 校验 zip 与源码一致")
    ok = compare(prefixed(repo_hashes(True)), zip_hashes(extension_zip), "extension.zip")
    ok = compare(prefixed(repo_hashes(False)), zip_hashes(legacy_zip), "legacy.zip") and ok
    if not ok:
        print("!! zip 与源码不一致，已中止（先检查 build_zip.py 是否漏文件）")
        return 1

    targets = find_targets(args.version)
    if not targets:
        print("!! 没找到可用的 Blender 配置目录（%s）" % config_base())
        return 1

    print("\n[3/4] 备份 + 替换")
    backup_root = os.path.join(dist, "backup")
    installed = []
    for ver, kind, path in targets:
        exists = os.path.isdir(path)
        old = read_version(path) if exists else "未安装"
        print("- Blender %s [%s] %s（当前 %s）" % (ver, kind, path, old))
        if args.dry_run:
            print("  （dry-run：不动文件）")
            continue
        if exists:
            dest = os.path.join(backup_root, "%s-%s-%s_%s" % (ver, kind, old, stamp))
            backup_tree(path, dest)
            print("  已备份到 %s" % dest)
        zip_path = extension_zip if kind == "extension" else legacy_zip
        print("  %s：%s" % (replace_tree(path, zip_path), zip_path))
        expect = repo_hashes(kind == "extension")
        if not compare(expect, walk_hashes(path), "安装后校验"):
            print("!! 安装后校验失败：%s" % path)
            return 1
        installed.append((ver, kind, path, old))

    if args.dry_run:
        print("\n(dry-run 结束，未做任何修改)")
        return 0

    print("\n[4/4] 收尾")
    if args.keep_dist:
        print("- 保留 dist/ 里的旧版 zip（--keep-dist）")
    else:
        moved = retire_old_dist(dist, version, stamp)
        print("- 旧版 zip 挪走：%s" % (moved or "无"))

    print("\n结果：已装 %d 处 ✓" % len(installed))
    for ver, kind, path, old in installed:
        print("  Blender %s [%s] %s -> %s" % (ver, kind, old, version))
    print("下一步：重启 Blender（正在运行的话改完文件不会热加载）。")
    print("验证：blender.exe -b --factory-startup --python \"dev_tests/test_installed.py\"")
    return 0


if __name__ == "__main__":
    sys.exit(main())
