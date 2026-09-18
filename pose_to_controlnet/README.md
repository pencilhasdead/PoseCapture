# Pose → ControlNet 骨骼图（独立工具）

把照片转成 **ControlNet 可直接用的 OpenPose 骨骼图（PNG）+ OpenPose JSON**，不依赖 Blender：
推理用的是本仓库插件里同一套纯 numpy 核心（`coco / imops / dwpose`）和同一批 DWPose 模型，
所以插件已经装好的依赖和模型会被自动复用，不用再下一遍 340MB。

```
照片 ──► YOLOX-L 人体检测 ──► DWPose(dw-ll_ucoco_384) 133 关键点
     ──► 渲染成 ControlNet 官方画法（18 点 + 肢体 + 手 + 脸）──► PNG / JSON
```

---

## 1. 运行

```bat
python pose_to_controlnet\pose_to_controlnet.py                  :: 打开图形界面
python -m pose_to_cn                                            :: 同上（先 cd 到工具目录）
```

命令行：

```bat
:: 单张：输出到 out\ 目录（目录不存在会自动建）
python pose_to_controlnet\pose_to_controlnet.py --input photo.jpg --output out

:: 批量：源目录里所有图片，长边缩到 768，不画手
python pose_to_controlnet\pose_to_controlnet.py --input photos --output out ^
       --size-mode long_side --long-side 768 --no-hands

:: 只按人物框裁成方形（单人特写，适合 ControlNet 训练/出图）
python pose_to_controlnet\pose_to_controlnet.py --input photo.jpg --size-mode person

:: 环境自检 / 装依赖 / 下模型
python pose_to_controlnet\pose_to_controlnet.py --check
python pose_to_controlnet\pose_to_controlnet.py --install-deps
python pose_to_controlnet\pose_to_controlnet.py --download-models
```

用哪个 Python：**界面需要 tkinter**，所以用系统的 `python`（Blender 自带的 python 没有 tkinter）。
没有参数就开界面，带参数走命令行；`--gui` 强制开界面，`--cli` 强制命令行。

## 2. 命令行参数

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--input` / `-i` | 必填 | 图片或文件夹 |
| `--output` / `-o` | 图片所在目录 | 带图片扩展名 = 完整文件名；否则当目录（不存在会创建） |
| `--style` | `openpose` | `openpose` = ControlNet 标准画法；`dwpose133` = 133 点全画（校对用） |
| `--model-dir` | 自动搜索 | DWPose 两个 onnx 所在目录 |
| `--person` | `0` | 人物序号，`0` = 面积最大的人，`-1` = 全部 |
| `--score-thr` | `0.3` | 关键点阈值 |
| `--det-thr` | `0.35` | 人体框检测阈值 |
| `--size-mode` | `source` | `source` 原图尺寸；`long_side` 等比缩到长边；`person` 按人物框裁方形 |
| `--long-side` | `1024` | `long_side` 模式的目标长边 |
| `--margin` | `0.10` | `person` 模式的留白比例 |
| `--hands` / `--no-hands`、`--face` / `--no-face`、`--feet` / `--no-feet` | 手/脸画，脚不画 | 控制画哪些部位 |
| `--json` / `--no-json` | 导出 | 是否同时写 OpenPose JSON |
| `--recursive` | 关 | 批量时递归子目录 |
| `--check` / `--install-deps` / `--download-models` | | 自检 / 装依赖 / 下模型 |
| `--quiet` / `-q` | 关 | 少打印 |

输出命名：`<图片名>_pose.png`，JSON 同名 `.json`（ControlNet 的 `openpose` 预处理器可以直接读）。
批量时会跳过上一次生成的 `*_pose.png`，重复跑不会套娃。

## 3. 界面

* 上方选输入（单张 / 文件夹）、输出目录、模型目录；「检查环境」看依赖与模型状态。
* 中间是参数（画法、阈值、人物序号、输出尺寸、画手/脸/脚、是否导出 JSON）。
* 下面是「检测并生成」「批量处理文件夹」「另存骨骼图…」「安装依赖」。
* 预览分原图 / 骨骼图两栏；日志区显示环境自检、每一步结果和错误。
* 检测与批量都在后台线程跑，界面不卡。参数会记进设置，下次打开自动填上。

## 4. 依赖与模型

* 依赖：`numpy` + `onnxruntime`（必需）、`pillow`（可选，用来读 jpg/webp）。
  界面上的「安装依赖」= `pip install --target <工具数据目录>\libs numpy onnxruntime pillow`，
  **不污染系统 Python**；命令行是 `--install-deps`。
* 数据目录（可用环境变量 `POSE_TO_CN_HOME` 覆盖）：
  `%LOCALAPPDATA%\PoseToControlNet\`（`libs\` 依赖、`models\` 模型、`settings.json` 设置）。
* 模型搜索顺序：`POSE_TO_CN_MODELS` → 设置里的 `model_dir` → 工具自己的 `models\`
  → Blender 插件数据目录（`%APPDATA%\Blender Foundation\Blender\<版本>\datafiles\pose_capture\models`）。
  找到 `yolox_l.onnx` + `dw-ll_ucoco_384.onnx` 就能直接用，不会重复下载。
* 插件核心模块通过 `POSE_CAPTURE_ADDON` 或仓库同级目录 `pose_capture/` 自动定位。

## 5. 自检

```bat
python dev_tests\test_tool.py                       :: 导入 / 渲染 / JSON / 流程 / 界面 / 真实模型
```

各段自己判断依赖：缺 numpy 就跳过渲染与流程段，缺 onnxruntime 或模型就跳过真实模型段，
用 Blender 自带 python 跑也可以（那种情况界面段会跳过，因为没有 tkinter）。

## 6. 常见问题

**提示缺少 numpy / onnxruntime，但 Blender 插件里明明是好的**
两个 Python 版本不同（Blender 5.2 内置 3.13，系统 python 可能是 3.14），插件装的是 3.13 的
wheel，给系统 python 用不了。点「安装依赖」把对应版本的包装进工具自己的 `libs\` 即可；
日志里会显示导入失败的简短原因。

**「没有检测到人体」** 调低人体框阈值（`0.35 → 0.15`），或换主体更大的图。

**骨骼图跟原图对不上**：`long_side` / `person` 模式会改变画布，PNG 和 JSON 用的是同一套坐标，
所以两者始终一致；如果你在 ControlNet 里又缩了一次，要保证缩放比例一致。

**想要更接近官方 annotator**：默认 `openpose` 画法就是照 ControlNet 官方
`annotator/openpose/util.py` 复刻的（关节点半径 4、肢体椭圆 stickwidth 4、只画前 17 条肢体、
手部 20 条边按 HSV 取色、人脸 68 点白色圆）。

## 7. 文件说明

```
pose_to_controlnet/
├─ pose_to_controlnet.py   双击/直接运行入口（无参数开界面）
└─ pose_to_cn/
   ├─ launcher.py          统一入口：无参数 → 界面，有参数 → 命令行
   ├─ cli.py               命令行参数与流程
   ├─ gui.py               tkinter 界面（只依赖标准库）
   ├─ options.py           转换参数与结果（不依赖 numpy）
   ├─ pipeline.py          图片 → 骨骼图（含批量）
   ├─ engine.py            DWPose ONNX 推理封装（懒加载）
   ├─ render.py            OpenPose / DWPose-133 渲染
   ├─ json_io.py           OpenPose JSON 导出
   ├─ imgio.py             读图 + 纯 numpy PNG 编码
   ├─ depinstall.py        依赖安装 / 模型下载
   ├─ paths.py             数据目录、设置、模型与插件目录发现
   └─ core_loader.py       复用插件的纯 numpy 核心模块
```

## 8. 许可

与主仓库一致：工具代码 MIT；模型来自 [IDEA-Research/DWPose](https://github.com/IDEA-Research/DWPose)
与 [Megvii YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)（Apache-2.0），工具只提供下载，不分发模型文件。
