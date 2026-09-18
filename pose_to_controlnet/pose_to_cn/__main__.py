"""python -m pose_to_cn —— 无参数开界面，带参数走命令行（详见 launcher）。"""
from __future__ import annotations

import sys

from .launcher import main

if __name__ == "__main__":
    sys.exit(main())
