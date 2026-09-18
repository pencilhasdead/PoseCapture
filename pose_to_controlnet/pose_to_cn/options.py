"""转换参数与结果（不依赖 numpy：界面在没有装依赖时也要能收集参数）。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

SIZE_MODES = ("source", "long_side", "person")

# 上一次生成的骨骼图不要再当输入（否则第二次跑会产出 xxx_pose_pose.png）
SKIP_SUFFIXES = ("_pose.png", "_pose.jpg", "_pose.jpeg")

# 输出路径带这些扩展名时，按"完整文件名"处理，否则当成目录
OUTPUT_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff")


@dataclass
class ConvertOptions:
    style: str = "openpose"          # openpose | dwpose133
    score_thr: float = 0.3
    det_thr: float = 0.35
    person: int = 0                  # -1 = 全部人物
    hands: bool = True
    face: bool = True
    feet: bool = False
    write_json: bool = True
    size_mode: str = "source"        # source 原图 | long_side 长边固定 | person 人物框
    long_side: int = 1024
    margin: float = 0.10             # person 模式的留边比例
    json_name: str = ""              # 留空则用 <图片名>_pose.json

    def render_options(self):
        from . import render

        return render.RenderOptions(style=self.style, score_thr=self.score_thr,
                                    hands=self.hands, face=self.face, feet=self.feet)


@dataclass
class ConvertResult:
    control: Any                     # 骨骼图（黑底 RGB，numpy uint8）
    source: Any                      # 与骨骼图同尺寸的原图（预览用）
    people: list                     # [(keypoints, scores), ...]，坐标已对齐输出尺寸
    width: int = 0
    height: int = 0
    json_data: dict | None = None
    notes: str = ""
