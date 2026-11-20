"""插件的场景属性与偏好设置。"""
from __future__ import annotations

import os

import bpy
from bpy.props import (BoolProperty, CollectionProperty, EnumProperty, FloatProperty,
                       IntProperty, PointerProperty, StringProperty)
from bpy.types import AddonPreferences, PropertyGroup

from . import deps, depth, reconstruction, rigify_limbs, rigify_map, sam3d, utils


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
    rig_type: EnumProperty(
        name="骨架标准",
        items=[("RIGIFY", "Rigify 标准", "Rigify 生成骨架 / 元骨架（spine_fk / upper_arm_fk.L …）"),
               ("MIXAMO", "Mixamo 标准", "Mixamo 命名（mixamorig:Hips / LeftArm / LeftUpLeg …）")],
        default="RIGIFY")

    auto_map_on_apply: BoolProperty(
        name="自动匹配骨骼名", default=True,
        description="Apply 时如果骨骼名失效、或者肢体骨跟“肢体写入方式”不一致"
                    "（选了 IK 却映射到 FK 骨骼），就重新自动匹配（手动改过的行会被覆盖）")

    # ---------------- 检测参数 ----------------
    score_thr: FloatProperty(name="关键点置信度阈值", default=0.3, min=0.05, max=1.0)
    det_thr: FloatProperty(name="人体框置信度阈值", default=0.3, min=0.05, max=1.0)

    # ---------------- 姿态解算参数 ----------------
    solver: EnumProperty(
        name="3D 来源",
        items=[("AUTO", "自动（装了 SAM 3D Body 就用）",
                "配置好 SAM 3D Body 就用它（单图直接出 MHR 人体，含手指与脚），"
                "否则退回下面的「深度来源」"),
               ("SAM3D", "SAM 3D Body",
                "强制用 SAM 3D Body：3D 关节由模型预测（带人体先验，前后关系 / 遮挡 / "
                "侧身都比「关键点 + 单目深度」稳）。需要在偏好设置里配好外部 python"),
               ("DEPTH", "DWPose + 深度模型",
                "老路子：DWPose 133 点 + Depth Anything V2 逐点深度 + 透视全局拟合"),
               ("BONE", "骨长解算（旧）",
                "不想用深度时：正交投影 + 骨长约束，前后方向靠固定假设")],
        default="AUTO")
    sam3d_info: StringProperty(name="SAM 3D Body 状态", default="")
    recon_source: StringProperty(
        name="最近一次解算来源", default="",
        description="上一次「应用到骨架」实际用的 3D 来源（SAM 3D Body / 深度模型 / 骨长解算）")

    # ---------------- MMD -> Mixamo ----------------
    mmd_prefix: EnumProperty(
        name="Mixamo 前缀",
        items=[("mixamorig:", "mixamorig:（Mixamo 标准）", "Mixamo 动画 / Fbx 用的标准前缀"),
               ("mixamorig1:", "mixamorig1:（同场景多角色）", "同一场景里有多个角色时用编号前缀"),
               ("", "无前缀", "只留 Hips / Spine / LeftArm …")],
        default="mixamorig:")
    mmd_cleanup: BoolProperty(
        name="删除 MMD 控制骨", default=True,
        description="全ての親 / センター / グルーブ / 肩P / 肩C / 足ＩＫ / つま先ＩＫ / 足D …"
                    "（Mixamo 没有这些槽位；删之前把它们的权重并进父骨、子骨改挂祖先）")
    mmd_translate_extras: BoolProperty(
        name="头发 / 裙子等骨骼名翻成英文", default=True,
        description="例如 左髪1 → hair1.L、スカート_0_1 → skirt01。骨骼本身一根都不删")
    mmd_rename_groups: BoolProperty(
        name="同步重命名顶点组", default=True,
        description="只改骨名不同步顶点组的话蒙皮会全部失效")
    mmd_merge_weights: BoolProperty(
        name="被删骨骼的权重并入父骨", default=True)
    mmd_strip_helpers: BoolProperty(
        name="摘掉 MMD Tools 追加变换", default=True,
        description="去掉骨骼约束与驱动器，否则改名后它们还会把骨骼拉回原位")
    mmd_duplicate: BoolProperty(
        name="先复制一份再改", default=True,
        description="保留原始 MMD 模型；关掉就直接改当前骨架（可 Ctrl+Z）")
    mmd_result: StringProperty(name="MMD → Mixamo 结果", default="")

    # ---------------- Goo Physics（头发 / 裙子 / 飘带） ----------------
    goo_hair: BoolProperty(name="头发", default=True)
    goo_skirt: BoolProperty(name="裙子", default=True)
    goo_ribbon: BoolProperty(name="飘带 / 领结 / 尾巴", default=True)
    goo_type: EnumProperty(
        name="物理类型",
        items=[("AUTO", "自动（头发=软体，裙子=布料）", ""),
               ("SOFT", "软体链（Soft Body）", ""),
               ("CLOTH", "布料链（Cloth）", ""),
               ("GEO", "几何节点链（Geo Nodes）", ""),
               ("JIGGLE", "抖动（Jiggle，单骨）", "")],
        default="AUTO")
    goo_collider: BoolProperty(
        name="给身体网格加碰撞体", default=True,
        description="把带骨架修改器的网格交给 Goo Physics 生成碰撞体，裙子和头发才不会穿身体")
    goo_result: StringProperty(name="Goo Physics 结果", default="")



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
    depth_focal: FloatProperty(
        name="透视强度", default=1.2, min=0.3, max=4.0,
        description="焦距系数：f = 系数 × 图片长边。调大相当于换长焦（透视更弱），"
                    "深度解算偏平就调大，偏夸张就调小")
    depth_flip: BoolProperty(
        name="反转深度方向", default=False,
        description="正常情况下方向由拟合自动决定，拟合分不出前后时用物理判据（人比背景近）"
                    "定符号，都不用勾。确实反了（比如人在反光/玻璃后面，物理判据不适用）才勾："
                    "勾上会把整套姿态前后翻过来，并停用物理判据。深度预览里会打出“深度方向”一行"
                    "供核对")
    swap_lr: BoolProperty(name="交换左右", default=False,
                          description="检测把人物左右判反时才需要勾选")
    twist: BoolProperty(name="估算骨骼扭转(roll)", default=True,
                        description="用肩线/胯线/弯曲轴对齐骨骼轴向，避免手臂头部拧转")
    twist_blend: FloatProperty(name="扭转强度", default=1.0, min=0.0, max=1.0)
    scale_hint: FloatProperty(name="比例微调", default=1.0, min=0.5, max=2.0,
                              description="轻微调整“关键点像素 / 骨架米”的比例（一般不用改）")
    fingers: BoolProperty(name="写入手指", default=False,
                          description="需要手指关键点质量较好时才开启")
    hand_align: FloatProperty(
        name="手部沿小臂", default=0.5, min=0.0, max=1.0,
        description="把手（手掌 + 手指）朝小臂方向拉的比例，让手尽量顺着小臂伸展："
                    "0 = 不拉（只在偏角超过 30° 时夹住），0.5 = 偏角减半（默认），"
                    "1 = 完全顺着小臂。手部关键点只到指根，握拳 / 侧掌 / 指尖错位时"
                    "偏角会很大（手掌横着长在腕上），所以默认往小臂方向拉一半；"
                    "手的形状与手长不变，只改它相对小臂的朝向")
    root_motion: BoolProperty(name="写入根骨骼位移", default=False,
                              description="把人物的水平位置/站立高度写到 root 骨骼")
    ground_align: BoolProperty(name="落地对齐", default=True)
    limb_mode: EnumProperty(
        name="肢体写入方式",
        items=[("IK", "IK 控制器（Rigify 默认）",
                "四肢都直接写 IK 控制器（不切 FK、不做 IK->FK 快照），顺带把 FK 链也写一份。"
                "手 / 脚钉在解出来的位置上：之后调躯干 / 骨盆时手脚不会跟着跑"
                "（实测转躯干 10°，脚只动 0.2 mm；FK 模式会动 17 cm）"),
               ("FK", "FK 骨骼",
                "四肢都停在 FK 骨骼上（IK_FK 滑块 = 1）并藏起 IK 控制器。"
                "肘 / 膝朝向更好控、骨长不会被 IK 拉伸；需要烘焙 / 重定向到别的骨架时用"),
               ("IK_LEG", "腿 IK + 臂 FK",
                "腿写 IK（脚踩住的落点最值钱）、手臂写 FK（肘的 roll 直接由弯曲轴算出，"
                "比 IK 的极点搜索更准）"),
               ("FK_LEG", "腿 FK + 臂 IK",
                "反过来：腿写 FK、手臂写 IK（手要扶 / 抓东西时用）")],
        default="IK")
    ik_fallback: EnumProperty(
        name="直写没成功时",
        items=[("SNAP", "退回 Rigify 快照（精确）",
                "切 FK -> 调 Rigify 自己的 IK->FK 快照 -> 切回 IK：方向与 FK 一致、关节位置差几毫米"
                "（IK 链的 hinge / stretch），代价是要重新算一遍 IK（需要生成脚本还在）"),
               ("FK", "停在 FK 骨骼",
                "不做快照，这几条肢体停在 FK：姿态一样精确（就是 FK 链那份），"
                "只是控制器换成 FK 那组")],
        default="SNAP")
    sync_visibility: BoolProperty(
        name="同步控制器显示", default=True,
        description="IK 模式只显示 IK 控制器、FK 模式只显示 FK 控制器，"
                    "避免“看得见却不动”被当成 bug（可在骨骼集合面板里手动改回来）")
    keyframe: BoolProperty(name="插入关键帧", default=False)

    # ---------------- 状态 ----------------
    status: StringProperty(name="状态", default="")
    log: StringProperty(name="日志", default="")
    detected: BoolProperty(default=False)
    preview_name: StringProperty(default="")
    detect_info: StringProperty(default="")
    depth_info: StringProperty(default="")
    depth_direction: StringProperty(default="")
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
    depth_view_python: StringProperty(
        name="预览窗口解释器（可选）", subtype="FILE_PATH", default="",
        description="能跑 tkinter 的 python.exe（Blender 自带的 python 没有 tkinter）。"
                    "留空会自动在 PATH 与常见安装位置里找")
    depth_variant: EnumProperty(
        name="深度模型档位", items=depth.variant_items(), default=depth.DEFAULT_VARIANT,
        description="fp32 最准（约 95 MB）、fp16 居中（约 48 MB）、int8 最小最快（约 27 MB）")
    auto_install: BoolProperty(name="缺少依赖时自动安装", default=True)
    verbose: BoolProperty(name="输出详细日志", default=True)

    # ---------------- SAM 3D Body（外部 python） ----------------
    sam3d_python: StringProperty(
        name="外部 python", subtype="FILE_PATH", default="",
        description="装好 PyTorch + sam-3d-body 的 python.exe（按官方 INSTALL.md 建的 conda 环境）。"
                    "Blender 自带 python 装不下 torch / detectron2，所以推理走子进程")
    sam3d_repo: StringProperty(
        name="代码目录", subtype="DIR_PATH", default="",
        description="git clone 下来的 sam-3d-body 仓库根目录（里面有 sam_3d_body 这个包）")
    sam3d_ckpt: StringProperty(
        name="模型权重", subtype="DIR_PATH", default="",
        description="官方下载的 checkpoints/sam-3d-body-* 目录（里面有 model.ckpt 与 "
                    "assets/mhr_model.pt）；留空则从 HuggingFace 仓库在线加载")
    sam3d_mhr: StringProperty(
        name="MHR 资产（可选）", subtype="FILE_PATH", default="",
        description="mhr_model.pt 的路径；留空会自动在权重目录的 assets/ 里找")
    sam3d_hf: StringProperty(
        name="HuggingFace 仓库", default="facebook/sam-3d-body-vith",
        description="没有本地权重时用（需要先在 HuggingFace 上申请访问权限并 hf 登录）")
    sam3d_device: EnumProperty(
        name="设备", items=[("auto", "自动", ""), ("cuda", "CUDA", ""), ("cpu", "CPU", "")],
        default="auto")
    sam3d_body_only: BoolProperty(
        name="只跑身体解码", default=False,
        description="跳出手部解码器：更省显存 / 更快，但手指没有 3D 关键点（Rigify 手指骨不写）")
    sam3d_timeout: IntProperty(name="超时（秒）", default=900, min=60, max=7200,
                               description="子进程超时（首次加载模型 + CPU 推理会比较慢）")




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
        box.label(text="深度模型（给关节解真实深度）", icon="IMAGE_ZDEPTH")
        _path, ready = resolve_depth_model_path()
        row = box.row(align=True)
        row.prop(self, "depth_variant", text="")
        row.label(text="已就绪" if ready else "未下载",
                  icon="CHECKMARK" if ready else "ERROR")
        box.prop(self, "local_depth_model")
        row = box.row(align=True)
        row.operator("pose_capture.download_depth_model", text="下载深度模型", icon="URL")
        row.operator("pose_capture.import_depth_model", text="导入深度模型", icon="FILE_FOLDER")

        box = layout.box()
        box.label(text="深度图预览窗口（独立 Tk 页面）", icon="IMAGE_ZDEPTH")
        row = box.row(align=True)
        row.operator("pose_capture.open_depth_view", text="打开预览窗口", icon="WINDOW")
        box.prop(self, "depth_view_python")
        box.label(text="Blender 自带 python 没有 tkinter，这里需要一个系统 Python", icon="INFO")

        # ---------------- SAM 3D Body ----------------
        box = layout.box()
        box.label(text="SAM 3D Body（外部 python，推荐）", icon="OUTLINER_OB_ARMATURE")
        config = sam3d.config_from_prefs(self)
        info = sam3d.status(config)
        box.label(text="状态：%s" % info["message"],
                  icon="CHECKMARK" if info["available"] else "ERROR")
        box.prop(self, "sam3d_python")
        box.prop(self, "sam3d_repo")
        box.prop(self, "sam3d_ckpt")
        box.prop(self, "sam3d_mhr")
        row = box.row(align=True)
        row.prop(self, "sam3d_hf")
        row.prop(self, "sam3d_device", text="")
        row = box.row(align=True)
        row.prop(self, "sam3d_body_only")
        row.prop(self, "sam3d_timeout", text="超时")
        box.operator("pose_capture.sam3d_check", text="检查环境（跑一次 --check）", icon="PLAY")
        box.label(text="按官方 INSTALL.md 建好环境（torch + sam-3d-body），"
                       "权重需在 HuggingFace 上申请访问", icon="INFO")



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
        "hand_align": settings.hand_align,
        "twist": settings.twist,
        "twist_blend": settings.twist_blend,
        "scale_hint": settings.scale_hint,
        "root_motion": settings.root_motion,
        "ground_align": settings.ground_align,
        "keep_depth": True,
        "fk_mode": settings.limb_mode == "FK",
        "limb_mode": settings.limb_mode,
        "ik_fallback": settings.ik_fallback,
        "sync_visibility": settings.sync_visibility,
        "keyframe": settings.keyframe,
    }


def limb_mode_map(settings: PC_Settings) -> dict:
    """``{"Arm": "IK"/"FK", "Leg": ...}``：把“肢体写入方式”拆成按肢体种类。

    自动匹配骨骼名时用它：胳膊 / 腿可以分别要求 IK 控制器或 FK 骨骼。
    """
    return rigify_limbs.limb_mode_map(getattr(settings, "limb_mode", "IK"))


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
    prefs = None
    try:
        prefs = get_prefs()
    except Exception:  # noqa: BLE001  （开发脚本里手动注册时没有偏好项）
        prefs = None
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



# --------------------------------------------------------------------------- 3D 来源
def resolve_solver(settings: PC_Settings) -> tuple[str, str]:
    """把「3D 来源」解析成实际要跑的（``'SAM3D'`` / ``'DEPTH'`` / ``'BONE'``）+ 说明。

    * ``AUTO``：SAM 3D Body 配置齐全就用它，否则按「深度来源」退回；
    * ``SAM3D``：强制用（配置不全时返回 SAM3D + 说明，执行时会给出失败原因）；
    * ``DEPTH`` / ``BONE``：老路子。

    ``settings`` 允许是 None（注册阶段场景属性可能还没就绪）。
    """
    want = str(getattr(settings, "solver", "AUTO") or "AUTO").upper() if settings is not None else "AUTO"
    depth_mode = getattr(settings, "depth_mode", "DEPTH") if settings is not None else "DEPTH"
    if want in ("SAM3D", "AUTO"):
        try:
            config = sam3d.config_from_prefs(get_prefs())
        except Exception:                        # noqa: BLE001  注册阶段 / 开发脚本里可能没有偏好项
            config = {}
        info = sam3d.status(config) if config else {"available": False,
                                                    "message": "读不到偏好设置"}
        if info["available"]:
            return "SAM3D", ""
        if want == "SAM3D":
            return "SAM3D", "SAM 3D Body 不可用：%s" % info["message"]
        return ("DEPTH" if depth_mode == "DEPTH" else "BONE"), \
               "SAM 3D Body 不可用，已改用「深度来源」：%s" % info["message"]
    if want == "DEPTH":
        return "DEPTH", ""
    return "BONE", ""


