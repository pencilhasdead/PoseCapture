"""准备依赖与模型（跑真实端到端自检之前执行一次）。

  blender.exe -b --factory-startup --python "dev_tests/_setup_deps.py"
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ADDON_PARENT = os.path.dirname(HERE)
if ADDON_PARENT not in sys.path:
    sys.path.insert(0, ADDON_PARENT)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import bpy  # noqa: E402

from pose_capture import deps, dwpose, utils  # noqa: E402


def progress(done, total):
    if total:
        print("   %5.1f%%  (%.0f/%.0f MB)" % (done * 100.0 / total, done / 1048576.0,
                                              total / 1048576.0))


def main():
    print("依赖目录:", utils.libs_dir())
    print("模型目录:", utils.models_dir())
    print("自带 Python:", utils.python_executable())
    print("当前后端:", deps.backend_status())

    if not deps.backend_status()["onnxruntime"]:
        print("== 安装 onnxruntime ==")
        ok = deps.pip_install("onnxruntime", log_cb=lambda line: print("   ", line))
        print("安装结果:", ok)
        print("安装后后端:", deps.backend_status())

    det, pose = deps.model_paths()
    for urls, dest, name in ((dwpose.DETECTOR_URLS, det, dwpose.DETECTOR_FILE),
                             (dwpose.POSE_URLS, pose, dwpose.POSE_FILE)):
        if os.path.isfile(dest):
            print("已存在:", dest)
            continue
        print("== 下载 %s ==" % name)
        ok = deps.download_model(urls, dest, progress_cb=progress,
                                 log_cb=lambda line: print("   ", line))
        print("结果:", ok, os.path.getsize(dest) if os.path.isfile(dest) else "-")
    return 0


if __name__ == "__main__":
    sys.exit(main())
