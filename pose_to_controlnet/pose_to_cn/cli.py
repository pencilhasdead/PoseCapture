"""命令行入口。

  python -m pose_to_cn --input photo.jpg --output out
  python -m pose_to_cn --input photos/ --size-mode long_side --long-side 768
  python -m pose_to_cn --check          # 只做环境自检

不带参数（或加 --gui）时由 launcher 打开界面。
"""
from __future__ import annotations

import argparse
import os
import sys

from . import __version__, depinstall, paths

EXIT_OK = 0
EXIT_FAIL = 1
EXIT_NOT_READY = 2

STYLE_CHOICES = ("openpose", "dwpose133")
SIZE_CHOICES = ("source", "long_side", "person")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pose_to_controlnet",
        description="把照片转成 ControlNet 可用的 OpenPose 骨骼图 / JSON（独立工具，不需要 Blender）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("--input", "-i", help="输入图片或文件夹")
    parser.add_argument("--output", "-o",
                        help="输出文件（单图）或目录（批量）；默认图片所在目录")
    parser.add_argument("--style", default="openpose", choices=STYLE_CHOICES,
                        help="openpose = ControlNet 标准画法；dwpose133 = 133 点全画")
    parser.add_argument("--model-dir", default=None,
                        help="DWPose 模型所在目录（默认自动搜索）")
    parser.add_argument("--person", type=int, default=0,
                        help="人物序号，0 = 最大的人，-1 = 全部")
    parser.add_argument("--score-thr", type=float, default=0.3, help="关键点阈值")
    parser.add_argument("--det-thr", type=float, default=0.35, help="人体框检测阈值")
    parser.add_argument("--size-mode", default="source", choices=SIZE_CHOICES,
                        help="source = 原图尺寸；long_side = 等比缩到长边；"
                             "person = 按人物框裁成方形")
    parser.add_argument("--long-side", type=int, default=1024,
                        help="--size-mode long_side 时的长边像素")
    parser.add_argument("--margin", type=float, default=0.10,
                        help="--size-mode person 时人物框外留白比例")
    parser.add_argument("--recursive", action="store_true",
                        help="批量时递归子目录")
    parser.add_argument("--hands", dest="hands", action="store_true", default=True,
                        help="画手（默认画）")
    parser.add_argument("--no-hands", dest="hands", action="store_false", help="不画手")
    parser.add_argument("--face", dest="face", action="store_true", default=True,
                        help="画脸（默认画）")
    parser.add_argument("--no-face", dest="face", action="store_false", help="不画脸")
    parser.add_argument("--feet", dest="feet", action="store_true", default=False,
                        help="画脚（仅是 openpose 风格的额外 6 个脚点）")
    parser.add_argument("--no-feet", dest="feet", action="store_false",
                        help="不画脚（默认）")
    parser.add_argument("--json", dest="write_json", action="store_true", default=True,
                        help="同时导出 OpenPose JSON（默认导出）")
    parser.add_argument("--no-json", dest="write_json", action="store_false",
                        help="不导出 JSON")
    parser.add_argument("--check", action="store_true",
                        help="只检查依赖 / 模型 / 插件，不处理图片")
    parser.add_argument("--install-deps", action="store_true",
                        help="把 numpy / onnxruntime / pillow 装到工具目录后退出")
    parser.add_argument("--download-models", action="store_true",
                        help="下载 DWPose 模型到工具目录后退出")
    parser.add_argument("--quiet", "-q", action="store_true", help="少打印")
    return parser


def report_environment(log=print) -> bool:
    """打印依赖 / 模型 / 插件核心状态；返回是否可以开始推理。"""
    from . import core_loader

    packages = depinstall.package_status()
    errors = packages.get("errors") or {}
    log("解释器：%s" % depinstall.python_executable())
    log("数据目录：%s" % paths.data_dir())
    for name in ("numpy", "onnxruntime", "pillow"):
        version = packages.get(name)
        if version:
            log("  %-12s %s" % (name, version))
        else:
            reason = errors.get(name, "")
            log("  %-12s 缺失%s" % (name, ("（%s）" % reason) if reason else ""))
    if packages.get("providers"):
        log("  可用推理后端：%s" % ", ".join(packages["providers"]))
    try:
        log("插件核心：%s" % core_loader.addon_dir())
    except Exception as exc:  # noqa: BLE001
        log("插件核心：不可用（%s）" % exc)

    models = paths.models_status()
    if models["found"]:
        log("模型：%s" % models["detector"])
        log("      %s" % models["pose"])
    else:
        log("模型：没找到，搜索过：")
        for folder in models["searched"]:
            log("  " + folder)
    missing = [name for name in depinstall.REQUIRED if not packages.get(name)]
    if missing:
        log("提示：缺少 %s，可以运行 —— python -m pose_to_cn --install-deps"
            % ", ".join(missing))
    if not models["found"]:
        log("提示：缺少模型，可以运行 —— python -m pose_to_cn --download-models")
    return not missing and models["found"]


def _log_factory(quiet: bool):
    if quiet:
        return lambda *parts: None

    def log(*parts) -> None:
        text = " ".join(str(part) for part in parts)
        try:
            print(text, flush=True)
        except UnicodeEncodeError:   # GBK 控制台遇到外来的花哨字符
            encoding = sys.stdout.encoding or "utf-8"
            print(text.encode(encoding, "replace").decode(encoding, "replace"),
                  flush=True)

    return log


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    log = _log_factory(args.quiet)

    if args.install_deps:
        ok = depinstall.install_everything(log_cb=log)
        print("依赖安装：%s" % ("完成" if ok else "失败"))
        return EXIT_OK if ok else EXIT_FAIL

    if args.download_models:
        found = depinstall.download_models(log_cb=log)
        if not found:
            print("模型未就绪")
            return EXIT_FAIL
        print("模型：%s" % found[0])
        return EXIT_OK

    if args.check:
        return EXIT_OK if report_environment() else EXIT_NOT_READY

    if not args.input:
        build_parser().print_help()
        return EXIT_FAIL
    if not os.path.exists(args.input):
        print("找不到输入：%s" % args.input)
        return EXIT_FAIL

    if not report_environment(log) and not args.quiet:
        print("（上面标了缺失的项，先补齐再跑；下面继续尝试）")

    from . import engine, pipeline
    from .options import ConvertOptions

    options = ConvertOptions(
        style=args.style, score_thr=args.score_thr, det_thr=args.det_thr,
        person=args.person, hands=args.hands, face=args.face, feet=args.feet,
        write_json=args.write_json, size_mode=args.size_mode,
        long_side=args.long_side, margin=args.margin)
    paths.save_settings({"model_dir": args.model_dir} if args.model_dir else {})

    pose_engine = engine.PoseEngine(args.model_dir)
    try:
        if os.path.isdir(args.input):
            def progress(index: int, total: int, path: str) -> None:
                if path:
                    log("[%d/%d] %s" % (index + 1, total, os.path.basename(path)))

            results = pipeline.convert_folder(args.input, pose_engine, options,
                                              args.output, args.recursive,
                                              on_progress=progress)
            failed = [item for item in results if str(item[1]).startswith("失败")]
            for path, out in results:
                if str(out).startswith("失败"):
                    print("%s -> %s" % (os.path.basename(path), out))
            print("完成：成功 %d / 共 %d，输出目录 %s"
                  % (len(results) - len(failed), len(results),
                     args.output or args.input))
            return EXIT_OK if len(failed) < len(results) else EXIT_FAIL

        png_path, json_path = pipeline.output_paths(args.input, args.output, options)
        result = pipeline.convert_file(args.input, pose_engine, options, png_path)
        log(result.notes)
        if json_path:
            log("JSON：%s" % json_path)
        return EXIT_OK
    except Exception as exc:  # noqa: BLE001 - 命令行下直接给结论
        print("失败：%s" % exc)
        return EXIT_FAIL
    finally:
        pose_engine.close()

