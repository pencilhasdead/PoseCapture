"""Pose Capture —— 从图片检测人体骨架，并转换成 Rigify 骨架可用的 Pose。

流程：图片
  -> SAM 3D Body（外部 python，推荐）或 DWPose(ONNX) 关键点 + 深度模型
  -> 2D->3D -> Rigify Pose 骨骼旋转（可写根位移 / 导入网格）
"""
# 注意：**扩展机制**（bl_ext.*）下 Blender 会在调用 register() 之前把 bl_info 从模块里删掉，
# 所以 register() 里绝对不要引用 bl_info，统一用这个 __version__。
__version__ = "1.4.0"

bl_info = {
    "name": "Pose Capture (SAM 3D Body / DWPose to Rigify)",
    "author": "Pose Capture",
    "version": (1, 4, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Sidebar > Pose Capture",
    "description": "读取图片，用 SAM 3D Body 或 DWPose + 深度模型识别人体骨架，并写入 Rigify 标准骨骼的 Pose",
    "category": "Animation",
}

import bpy

from . import operators, props, rigify_limbs, ui, utils

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
    rigify_limbs.ensure_watcher()
    version = __version__
    solver = "未解析"
    try:
        scene = getattr(bpy.context, "scene", None)
        settings = getattr(scene, "pose_capture", None) if scene is not None else None
        if settings is not None:
            solver = props.resolve_solver(settings)[0]
    except Exception:  # noqa: BLE001  注册阶段没有场景 / 属性还没就绪都正常
        pass

    utils.log("Pose Capture %s 已启用（3D 来源：%s）。依赖目录：%s"
              % (version, solver, utils.libs_dir()))



def unregister():
    rigify_limbs.remove_watcher()
    if hasattr(bpy.types.Scene, "pose_capture"):
        del bpy.types.Scene.pose_capture
    for cls in reversed(_CLASSES):
        try:
            bpy.utils.unregister_class(cls)
        except Exception:  # noqa: BLE001
            pass


if __name__ == "__main__":
    register()
