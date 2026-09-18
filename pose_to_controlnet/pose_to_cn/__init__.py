"""pose_to_cn —— 把照片转成 ControlNet 可用骨骼图的独立工具（不依赖 Blender）。

模块划分：
  paths          数据目录、设置、模型/插件目录发现
  core_loader    复用 pose_capture 插件里那套纯 numpy 核心（coco/imops/dwpose/utils）
  imgio          读图（PIL / tkinter 回退）与纯 numpy PNG 写出
  render         OpenPose(ControlNet 原生画法) / DWPose-133 骨骼图渲染
  json_io        OpenPose JSON 导出
  options        转换参数与结果（不依赖 numpy）
  engine         DWPose ONNX 推理封装（懒加载）
  pipeline       图片 -> 骨骼图 的完整流程 + 批量
  depinstall     pip 安装 numpy / onnxruntime / pillow 到工具自己的目录
  cli            命令行入口
  gui            tkinter 界面（仅 GUI 模式导入）
  launcher       统一入口：没参数开界面，带参数走命令行
"""

__version__ = "1.0.0"
APP_TITLE = "Pose → ControlNet 骨骼图"
