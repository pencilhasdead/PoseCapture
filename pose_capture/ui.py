"""3D 视图侧栏（N 面板）界面。"""
from __future__ import annotations

import bpy

from . import deps, props, rigify_limbs, rigify_map


class PC_UL_bone_map(bpy.types.UIList):
    """角色 -> 骨骼 的映射列表，行内可直接搜索骨名。"""

    def draw_item(self, _context, layout, _data, item, _icon, _active_data, _propname, _index):
        settings = bpy.context.scene.pose_capture
        armature = settings.armature
        row = layout.row(align=True)
        row.label(text=rigify_map.role_label(item.role))
        if armature is not None:
            row.prop_search(item, "bone", armature.data, "bones", text="")
            found = bool(item.bone) and item.bone in armature.data.bones
            row.label(text="", icon="CHECKMARK" if found else ("ERROR" if item.bone else "DOT"))
        else:
            row.prop(item, "bone", text="")


class PC_PT_main(bpy.types.Panel):
    bl_label = "Pose Capture"
    bl_idname = "PC_PT_main"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Pose Capture"

    def draw(self, context):
        layout = self.layout
        settings = props.get_settings(context)
        # 载入新文件会清空 python 侧的 handler，面板绘制时顺手补上（很便宜）
        rigify_limbs.ensure_watcher()

        # ---------------- 环境状态 ----------------
        status = deps.backend_status()
        backend_name = None
        if status["onnxruntime"]:
            backend_name = "onnxruntime %s" % status["onnxruntime"]
        elif status["opencv"]:
            backend_name = "opencv %s" % status["opencv"]
        _det, _pose, models_ready = props.resolve_model_paths()
        _depth_path, depth_ready = props.resolve_depth_model_path()

        box = layout.box()
        col = box.column(align=True)
        col.label(text="依赖：%s" % (backend_name or "未安装"),
                  icon="CHECKMARK" if backend_name else "ERROR")
        col.label(text="模型：%s" % ("已就绪" if models_ready else "未下载"),
                  icon="CHECKMARK" if models_ready else "ERROR")
        col.label(text="深度模型：%s" % ("已就绪" if depth_ready else "未下载"),
                  icon="CHECKMARK" if depth_ready else "INFO")
        if not (backend_name and models_ready):
            row = col.row(align=True)
            row.operator("pose_capture.install_deps", text="安装依赖", icon="IMPORT")
            row.operator("pose_capture.download_models", text="下载模型", icon="URL")

        # ---------------- 1 输入 ----------------
        box = layout.box()
        box.label(text="1. 输入图片", icon="IMAGE_DATA")
        col = box.column(align=True)
        col.prop(settings, "image_source", expand=True)
        if settings.image_source == "FILE":
            col.prop(settings, "image_filepath", text="")
        else:
            col.prop_search(settings, "image_name", bpy.data, "images", text="")
        col.prop(settings, "person_index")
        row = col.row(align=True)
        row.prop(settings, "det_thr")
        row.prop(settings, "score_thr")
        row = box.row(align=True)
        row.operator("pose_capture.detect", icon="PLAY",
                     text="检测姿态" if not settings.detected else "重新检测")
        row.operator("pose_capture.show_preview", text="", icon="IMAGE_REFERENCE")
        row.operator("pose_capture.show_depth", text="", icon="IMAGE_ZDEPTH")
        row.operator("pose_capture.export_json", text="", icon="EXPORT")
        # ---------------- 2 目标骨架 ----------------
        box = layout.box()
        box.label(text="2. 目标骨架", icon="ARMATURE_DATA")
        row = box.row(align=True)
        row.prop(settings, "armature", text="")
        row.operator("pose_capture.use_active_armature", text="", icon="EYEDROPPER")
        row = box.row(align=True)
        row.operator("pose_capture.add_metarig", text="新建元骨架")
        row.operator("pose_capture.generate_rig", text="生成控制骨架")
        row = box.row(align=True)
        row.operator("pose_capture.auto_map", text="自动匹配骨骼")
        row.operator("pose_capture.reset_pose", text="清空 Pose")
        box.prop(settings, "auto_map_on_apply")
        if settings.armature is not None:
            box.label(text="骨骼数：%d" % len(settings.armature.data.bones))

        # ---------------- 3 解算参数 ----------------
        box = layout.box()
        box.label(text="3. 姿态解算", icon="SETTINGS")
        col = box.column(align=True)
        col.prop(settings, "facing")
        col.prop(settings, "depth_mode")
        if settings.depth_mode == "DEPTH":
            col.label(text="深度模型：%s" % ("已就绪" if depth_ready else "未下载"),
                      icon="CHECKMARK" if depth_ready else "ERROR")
            if not depth_ready:
                col.operator("pose_capture.download_depth_model",
                             text="下载深度模型", icon="URL")
            row = col.row(align=True)
            row.prop(settings, "depth_focal")
            row.prop(settings, "depth_flip", text="反转方向")
            if settings.depth_preview_name:
                row = col.row(align=True)
                row.operator("pose_capture.show_depth", text="查看深度图",
                             icon="IMAGE_REFERENCE")
                row.operator("pose_capture.open_depth_view", text="深度图预览窗口",
                             icon="WINDOW")
        elif settings.depth_mode == "BONE":
            col.label(text="旧方式：前后方向靠固定假设（手肘朝后/膝盖朝前）", icon="INFO")
        row = col.row(align=True)
        row.prop(settings, "twist")
        row.prop(settings, "twist_blend", text="强度")
        col.prop(settings, "swap_lr")
        col.prop(settings, "fingers")
        col.prop(settings, "hand_align")
        col.prop(settings, "scale_hint")
        if settings.depth_info:
            box.label(text=settings.depth_info[:80], icon="INFO")
        if settings.depth_direction:
            mismatch = "★" in settings.depth_direction
            box.label(text=settings.depth_direction[:110],
                      icon="ERROR" if mismatch else "CHECKMARK")

        # ---------------- 4 写入 ----------------
        box = layout.box()
        box.label(text="4. 写入 Pose", icon="POSE_HLT")
        col = box.column(align=True)
        col.prop(settings, "limb_mode")
        if settings.armature is not None and rigify_limbs.limb_specs(settings.armature):
            modes = rigify_limbs.limb_modes(settings.armature)
            text = ", ".join("%s=%s" % (name.split(".")[0], "FK" if value >= 0.5 else "IK")
                             for name, value in sorted(modes.items()))
            col.label(text="当前滑块：%s" % (text or "无"), icon="INFO")
            col.operator("pose_capture.sync_limb_display", icon="FILE_REFRESH",
                         text="同步控制器显示（拖过 Rigify 滑块后点我）")
        col.prop(settings, "sync_visibility")
        col.prop(settings, "root_motion")
        if settings.root_motion:
            col.prop(settings, "ground_align")
        row = col.row(align=True)
        row.prop(settings, "keyframe")
        row.label(text="当前帧 %d" % context.scene.frame_current)
        box.operator("pose_capture.apply_pose", icon="CHECKMARK", text="应用到骨架")


class PC_PT_mapping(bpy.types.Panel):
    bl_label = "骨骼映射（可手动修改）"
    bl_idname = "PC_PT_mapping"
    bl_parent_id = "PC_PT_main"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = self.layout
        settings = props.get_settings(context)
        if settings.armature is None:
            layout.label(text="请先指定目标骨架", icon="INFO")
            return
        if not len(settings.bone_map):
            layout.label(text="点上方“自动匹配骨骼”生成映射表", icon="INFO")
            return
        layout.template_list("PC_UL_bone_map", "", settings, "bone_map",
                             settings, "bone_map_index", rows=8)


class PC_PT_log(bpy.types.Panel):
    bl_label = "日志"
    bl_idname = "PC_PT_log"
    bl_parent_id = "PC_PT_main"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = self.layout
        settings = props.get_settings(context)
        if settings.status:
            layout.label(text=settings.status, icon="TIME")
        lines = [line for line in (settings.log or "").splitlines() if line.strip()]
        for line in lines[-12:]:
            layout.label(text=line[:90])


CLASSES = (PC_UL_bone_map, PC_PT_main, PC_PT_mapping, PC_PT_log)

