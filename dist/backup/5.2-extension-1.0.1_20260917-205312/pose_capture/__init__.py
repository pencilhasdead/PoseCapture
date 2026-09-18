"""Pose Capture —— 从图片检测人体骨架，并转换成 Rigify 骨架可用的 Pose。

流程：图片 -> Easy-DWPose/DWPose(ONNX) 关键点 -> 2D->3D 骨长约束解算 -> Rigify Pose 骨骼旋转
"""
bl_info = {
    "name": "Pose Capture (DWPose to Rigify)",
    "author": "Pose Capture",
    "version": (1, 0, 1),
    "blender": (4, 2, 0),
    "location": "View3D > Sidebar > Pose Capture",
    "description": "读取图片，用 DWPose / Easy-DWPose 识别人体骨架，并写入 Rigify 标准骨骼的 Pose",
    "category": "Animation",
}

import bpy

from . import operators, props, ui, utils

_CLASSES = (
    props.PC_BoneMapping,
    props.PC_Settings,
    props.PC_Preferences,
) + operators.CLASSES + ui.CLASSES


def register():
    utils.ensure_libs_on_path()
    for cls in _CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.pose_capture = bpy.props.PointerProperty(type=props.PC_Settings)
    utils.log("Pose Capture 已启用。依赖目录：%s" % utils.libs_dir())


def unregister():
    if hasattr(bpy.types.Scene, "pose_capture"):
        del bpy.types.Scene.pose_capture
    for cls in reversed(_CLASSES):
        try:
            bpy.utils.unregister_class(cls)
        except Exception:  # noqa: BLE001
            pass


if __name__ == "__main__":
    register()
