"""按**真实路径**启用插件（addon_utils.enable），确认 register() 不抛异常。

为什么需要这个测试：`--factory-startup` + 手动 register_class 那条路**测不出**扩展机制特有的坑
（Blender 会在 register() 之前把 `bl_info` 从模块里删掉，早期版本因此在 register 里 NameError，
插件整个装不上，界面直接没有）。

用法（**不要**加 --factory-startup，要用你真实的偏好设置）：
  blender.exe -b --python "dev_tests/test_enable.py"
"""
import sys

import bpy

MODULE = "bl_ext.user_default.pose_capture"
failures = []


def check(ok, label, detail=""):
    print("  %s %s %s" % ("OK " if ok else "!! ", label, detail))
    if not ok:
        failures.append(label)
    return ok


def main():
    print("blender:", bpy.app.version_string)
    # 已经注册过就不要再 enable：重复 enable 会重载插件，把 AddonPreferences（SAM 路径）清空
    already = hasattr(bpy.types, "PC_PT_main") and MODULE in bpy.context.preferences.addons
    if already:
        error = ""
        print("  （启动时已自动注册，不重复 enable）")
    else:
        try:
            bpy.ops.preferences.addon_enable(module=MODULE)
            error = ""
        except Exception as exc:                        # noqa: BLE001
            error = repr(exc)
        check(not error, "addon_enable 没抛异常", error)

    check(MODULE in bpy.context.preferences.addons, "出现在偏好设置 addons 列表")

    package = sys.modules.get(MODULE)
    check(package is not None, "模块已导入",
          getattr(package, "__file__", "?"))
    version = getattr(package, "__version__", None)
    check(bool(version) and version.count(".") == 2, "版本号", version)


    check(hasattr(bpy.types, "PC_PT_main"), "面板类注册")
    check(hasattr(bpy.types, "PC_PT_mapping") and hasattr(bpy.types, "PC_PT_log"), "3 个面板")
    scene = bpy.context.scene
    check(hasattr(scene, "pose_capture"), "场景属性 scene.pose_capture")
    for name in ("detect", "apply_pose", "sam3d_check", "import_sam3d_mesh"):
        check(hasattr(bpy.ops.pose_capture, name), "操作器 %s" % name)

    if hasattr(scene, "pose_capture"):
        from importlib import import_module
        props = import_module(MODULE + ".props")
        sam3d = import_module(MODULE + ".sam3d")
        settings = scene.pose_capture
        solver, note = props.resolve_solver(settings)
        print("  solver = %s %s" % (solver, note))
        prefs = bpy.context.preferences.addons[MODULE].preferences
        print("  sam3d_python = %s" % prefs.sam3d_python)
        print("  status = %s" % sam3d.status(sam3d.config_from_prefs(prefs))["message"])
        check(solver == "SAM3D", "自动解析到 SAM 3D Body", solver)
        check(hasattr(bpy.ops.pose_capture, "mmd_to_mixamo"), "MMD → Mixamo 操作器")
        check(hasattr(bpy.types, "PC_PT_mmd"), "MMD → Mixamo 面板")

        ui = import_module(MODULE + ".ui")
        check(ui._version() == version, "面板版本号与 __version__ 一致", ui._version())


    print()
    if failures:
        print("!! 失败 %d 项：%s" % (len(failures), ", ".join(failures)))
        return 1
    print("通过 (all OK)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
