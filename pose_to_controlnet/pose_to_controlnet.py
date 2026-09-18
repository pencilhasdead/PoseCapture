"""Pose → ControlNet 骨骼图工具（独立版，不需要 Blender）。

用法：
  python pose_to_controlnet.py                      # 打开界面
  python pose_to_controlnet.py --input a.jpg        # 命令行处理单张
  python pose_to_controlnet.py --help               # 命令行参数说明

依赖（numpy / onnxruntime / pillow）与 DWPose 模型都可以在界面里
点按钮一键装好，或命令行：
  python pose_to_controlnet.py --install-deps
  python pose_to_controlnet.py --download-models
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from pose_to_cn.launcher import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
