"""统一入口：没有参数（或给 --gui）打开界面，带其它参数走命令行。

  python -m pose_to_cn                      # 界面
  python -m pose_to_cn --input a.jpg        # 命令行
"""
from __future__ import annotations

import sys


def main(argv=None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    wants_gui = "--gui" in args
    rest = [item for item in args if item not in ("--cli", "--gui")]

    if wants_gui or not rest:
        from . import gui

        return gui.main()

    from . import cli

    return cli.main(rest)
