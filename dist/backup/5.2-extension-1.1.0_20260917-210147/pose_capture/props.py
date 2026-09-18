"""插件的场景属性与偏好设置。"""
from __future__ import annotations

import os

import bpy
from bpy.props import (BoolProperty, CollectionProperty, EnumProperty, FloatProperty,
                       IntProperty, PointerProperty, StringProperty)
from bpy.types import AddonPreferences, PropertyGroup

from . import deps, depth, reconstruction, rigify_map, utils


class PC_BoneMapping(PropertyGroup):
    """一行“角色 -> 骨骼”的映射。"""

    role: StringProperty(name="角色")
    label: StringProperty(name="说明")
    bone: StringProperty(name="骨骼")


class PC_Settings(PropertyGroup):
    # ---------------- 输入 ----------------
    image_source: EnumProperty(
        name="图片来源",
        items=[("FILE", "图片文件", "从磁盘选择一张图片"),
               ("DATA", "已加载图像", "使用 Blender 里已经加载的图像数据块")],
        default="FILE")
    image_filepath: StringProperty(name="图片", subtype="FILE_PATH", default="")
    image_name: StringProperty(name="图像", default="")
    person_index: IntProperty(name="人物序号", default=0, min=0, max=32,
                              description="画面中有多人时选择第几个（0 = 面积最大的人）")

    # ---------------- 目标 ----------------
    armature: PointerProperty(
        name="目标骨架", type=bpy.types.Object,
        poll=lambda self, obj: obj.type == "ARMATURE")
    bone_map: CollectionProperty(type=PC_BoneMapping)
    bone_map_index: IntProperty(default=0)
    map_ready: BoolProperty(default=False)
    auto_map_on_apply: BoolProperty(name="自动匹配骨骼名", default=True)

    # ---------------- 检测参数 ----------------
    score_thr: FloatProperty(name="关键点置信度阈值", default=0.3, min=0.05, max=1.0)
    det_thr: FloatProperty(name="人体框置信度阈值", default=0.3, min=0.05, max=1.0)

    # ---------------- 姿态解算参数 ----------------
    facing: EnumProperty(
        name="人物朝向",
        items=[(key, reconstruction.FACING_LABELS[key], "")
               for key in reconstruction.FACING_DIRS],
        default="FRONT")
    depth_mode: EnumProperty(
        name="深度来源",
        items=[(key, reconstruction.DEPTH_MODE_LABELS[key],
                "DEPTH：用深度模型（Depth Anything V2）给每个关节解真实深度，"
                "前后方向来自像素；BONE：旧的骨长约束解算（前后方向靠固定假设）；"
                "FLAT：不做深度，整套姿态压在画面平面内")
               for key in reconstruction.DEPTH_MODES],
        default="DEPTH")
    depth_variant: EnumProperty(
        name="深度模型档位", items=depth.variant_items(), default=depth.DEFAULT_VARIANT,
        description="fp32 最准（约 95 MB）、fp16 居中（约 48 MB）、int8 最小最快（约 27 MB）")
    depth_focal: FloatProperty(
        name="透视强度", default=1.2, min=0.3, max=4.0,
        description="焦距系数：f = 系数 × 图片长边。调大相当于换长焦（透视更弱），"
                    "深度解算偏平就调大，偏夸张就调小")
    depth_flip: BoolProperty(
        name="反转深度方向", default=False,
        description="深度模型把前后判反了才勾（正常情况下方向由拟合自动决定）")
    swap_lr: BoolProperty(name="交换左右", default=False,
                          description="检测把人物左右判反时才需要勾选")
    twist: BoolProperty(name="估算骨骼扭转(roll)", default=True,
                        description="用肩线/胯线/弯曲轴对齐骨骼轴向，避免手臂头部拧转")
    twist_blend: FloatProperty(name="扭转强度", default=1.0, min=0.0, max=1.0)
    scale_hint: FloatProperty(name="比例微调", default=1.0, min=0.5, max=2.0,
                              description="轻微调整“关键点像素 / 骨架米”的比例（一般不用改）")
    fingers: BoolProperty(name="写入手指", default=False,
                          description="需要手指关键点质量较好时才开启")
    root_motion: BoolProperty(name="写入根骨骼位移", default=False,
                              description="把人物的水平位置/站立高度写到 root 骨骼")
    ground_align: BoolProperty(name="落地对齐", default=True)
    fk_mode: BoolProperty(name="自动切到 FK 模式", default=True,
                          description="Rigify 生成骨架默认 IK，写 FK 旋转前把 IK_FK 滑块设为 0")
    keyframe: BoolProperty(name="插入关键帧", default=False)

    # ---------------- 状态 ----------------
    status: StringProperty(name="状态", default="")
    log: StringProperty(name="日志", default="")
    detected: BoolProperty(default=False)
    preview_name: StringProperty(default="")
    detect_info: StringProperty(default="")
    depth_info: StringProperty(default="")
    depth_preview_name: StringProperty(default="")


class PC_Preferences(AddonPreferences):
    bl_idname = __package__

    backend: EnumProperty(
        name="推理后端",
        items=[("auto", "自动（onnxruntime 优先）", ""),
               ("onnxruntime", "onnxruntime", ""),
               ("opencv", "OpenCV DNN", "")],
        default="auto")
    backend_package: EnumProperty(
        name="要安装的依赖包",
        items=[(name, name, "") for name in deps.BACKEND_PACKAGE_IDS],
        default="onnxruntime")
    device: EnumProperty(
        name="推理设备",
        items=[("auto", "自动", "按可用 provider 自动选择"),
               ("cpu", "CPU", ""),
               ("dml", "DirectML", ""),
               ("cuda", "CUDA", "")],
        default="auto")
    local_dwpose_root: StringProperty(
        name="本地模型目录（可选）", subtype="DIR_PATH", default="",
        description="已下载过 DWPose/Easy-DWPose 模型的目录，插件会递归查找 onnx 文件")
    local_depth_model: StringProperty(
        name="深度模型文件（可选）", subtype="FILE_PATH", default="",
        description="已有的深度模型 onnx（Depth Anything / MiDaS 等）。留空则用插件模型目录里的")
    auto_install: BoolProperty(name="缺少依赖时自动安装", default=True)
    verbose: BoolProperty(name="输出详细日志", default=True)

    def draw(self, context):
        layout = self.layout
        status = deps.backend_status()

        box = layout.box()
        box.label(text="依赖状态", icon="INFO")
        row = box.row()
        row.label(text="onnxruntime: %s" % (status["onnxruntime"] or "未安装"))
        row.label(text="opencv: %s" % (status["opencv"] or "未安装"))
        if status.get("providers"):
            box.label(text="可用 provider: %s" % ", ".join(status["providers"]))
        box.label(text="依赖目录: %s" % utils.libs_dir())
        box.label(text="模型目录: %s" % utils.models_dir())
        box.label(text="自带 Python: %s" % (utils.python_executable() or "未找到"))

        row = layout.row(align=True)
        row.prop(self, "backend")
        row.prop(self, "device")
        row = layout.row(align=True)
        row.prop(self, "backend_package", text="")
        row.operator("pose_capture.install_deps", text="安装/更新依赖", icon="IMPORT")
        layout.prop(self, "auto_install")
        layout.prop(self, "verbose")

        box = layout.box()
        box.label(text="模型来源", icon="FILE_FOLDER")
        box.prop(self, "local_dwpose_root")
        row = box.row(align=True)
        row.operator("pose_capture.import_local_models", text="从本地目录导入")
        row.operator("pose_capture.download_models", text="在线下载模型")

        box = layout.box()
        box.label(text="深度模型（给关节解真实深度）", icon="IMAGE_DEPTH")
        _path, ready = resolve_depth_model_path()
        row = box.row(align=True)
        row.prop(self, "depth_variant", text="")
        row.label(text="已就绪" if ready else "未下载",
                  icon="CHECKMARK" if ready else "ERROR")
        box.prop(self, "local_depth_model")
        row = box.row(align=True)
        row.operator("pose_capture.download_depth_model", text="下载深度模型", icon="URL")
        row.operator("pose_capture.import_depth_model", text="导入深度模型", icon="FILE_FOLDER")


# --------------------------------------------------------------------------- 工具函数
def get_settings(context=None) -> PC_Settings:
    context = context or bpy.context
    return context.scene.pose_capture


def get_prefs(context=None) -> PC_Preferences:
    context = context or bpy.context
    addon = context.preferences.addons.get(__package__)
    if addon is not None and addon.preferences is not None:
        return addon.preferences
    return context.preferences.addons[__package__].preferences


def ensure_mapping_rows(settings: PC_Settings) -> None:
    """保证映射表里有全部角色行（不覆盖用户已有设置）。"""
    existing = {row.role for row in settings.bone_map}
    for role in rigify_map.all_role_ids(include_fingers=True):
        if role in existing:
            continue
        row = settings.bone_map.add()
        row.role = role
        row.label = rigify_map.role_label(role)


def mapping_from_settings(settings: PC_Settings) -> rigify_map.BoneMap:
    bmap = rigify_map.BoneMap()
    for row in settings.bone_map:
        if row.bone:
            bmap.set(row.role, row.bone)
    return bmap


def sync_mapping_to_settings(settings: PC_Settings, bmap: rigify_map.BoneMap) -> None:
    ensure_mapping_rows(settings)
    for row in settings.bone_map:
        row.bone = bmap.get(row.role) or ""
    settings.map_ready = True


def options_from_settings(settings: PC_Settings) -> dict:
    return {
        "score_thr": settings.score_thr,
        "facing": settings.facing,
        "depth_mode": settings.depth_mode,
        "depth_focal": settings.depth_focal,
        "depth_flip": settings.depth_flip,
        "swap_lr": settings.swap_lr,
        "fingers": settings.fingers,
        "twist": settings.twist,
        "twist_blend": settings.twist_blend,
        "scale_hint": settings.scale_hint,
        "root_motion": settings.root_motion,
        "ground_align": settings.ground_align,
        "keep_depth": True,
        "fk_mode": settings.fk_mode,
        "keyframe": settings.keyframe,
    }


def resolve_model_paths() -> tuple[str, str, bool]:
    """返回 (检测模型, 姿态模型, 是否都已就绪)。"""
    prefs = get_prefs()
    det, pose = deps.model_paths()

    override = getattr(prefs, "local_dwpose_root", "")
    if override and (not os.path.isfile(det) or not os.path.isfile(pose)):
        found = deps.find_local_models(override)
        if found:
            det, pose = found

    ready = os.path.isfile(det) and os.path.isfile(pose)
    return det, pose, ready


def resolve_depth_model_path() -> tuple[str, bool]:
    """返回 (深度模型路径, 是否可用)。

    优先级：偏好里手动指定的文件 > 插件模型目录里的对应档位 > 本地模型目录里的
    depth*.onnx（复用 ComfyUI 之类已经下载好的）。
    """
    prefs = get_prefs()
    override = getattr(prefs, "local_depth_model", "")
    if override:
        path = bpy.path.abspath(override)
        if os.path.isfile(path):
            return path, True

    variant = getattr(prefs, "depth_variant", depth.DEFAULT_VARIANT)
    path = deps.depth_model_path(variant)
    if os.path.isfile(path):
        return path, True

    root = getattr(prefs, "local_dwpose_root", "")
    found = deps.find_local_depth_model(root) if root else None
    if found:
        return found, True
    return path, False


def resolve_providers():
    prefs = get_prefs()
    mapping = {"cpu": ["CPUExecutionProvider"],
               "dml": ["DmlExecutionProvider"],
               "cuda": ["CUDAExecutionProvider", "CPUExecutionProvider"]}
    device = getattr(prefs, "device", "auto")
    providers = mapping.get(device)
    if device == "auto" or providers is None:
        return None
    try:
        import onnxruntime as ort

        available = ort.get_available_providers()
        filtered = [name for name in providers if name in available]
        return filtered or None
    except Exception:  # noqa: BLE001
        return None

