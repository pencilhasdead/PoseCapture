# Pose Capture — 图片 → Easy-DWPose / DWPose → Rigify 骨骼姿态

用一张人物图片，自动识别人体骨架，并把姿态转换成 **Blender Pose 模式下 Rigify 标准骨骼**可以直接使用的姿态数据。

```
图片 ──► YOLOX-L 人体检测 ──► DWPose(dw-ll_ucoco_384) 133 关键点
     ──► Depth Anything V2 深度图 ──► 透视相机 + 全局拟合（骨长解算兜底）
     ──► Rigify 控制骨骼旋转(+可选根位移)
```

* 已实测环境：**Blender 5.2.1 LTS（Windows，内置 Python 3.13 / numpy 2.3.4）**
* 代码按 Blender **4.2 ~ 5.x** 的 API 编写，使用相对导入，元骨架与生成骨架都支持
* 推理只用 `onnxruntime` + `numpy`（numpy 用 Blender 自带的），**不需要安装 PyTorch / OpenCV / mmpose**

---

## 1. 安装插件

```bash
python build_zip.py          # 生成 dist/ 下的两个 zip
```

### 方式 A：常规安装（最简单，实测通过）

* Blender → `编辑` → `偏好设置` → `附加组件` → 右上角 `▾` → `从磁盘安装…`
* 选择 `dist/pose_capture-1.3.3-legacy.zip`
* 在列表里勾选启用 **Pose Capture (DWPose to Rigify)**

### 方式 B：4.2+ 扩展机制（推荐，实测通过）

把 `dist/pose_capture-1.3.3-extension.zip` 解压到（目录名必须是 `pose_capture`）：

* Windows：`%APPDATA%\Blender Foundation\Blender\<版本>\extensions\user_default\`
* macOS：`~/Library/Application Support/Blender/<版本>/extensions/user_default/`
* Linux：`~/.config/blender/<版本>/extensions/user_default/`

然后在偏好设置里启用（模块名会显示为 `bl_ext.user_default.pose_capture`）。

### 方式 C：手动拷贝（开发调试）

把仓库里的 `pose_capture/` 目录整个复制到 `<版本>/scripts/addons/`，再启用。

### 方式 D：一条命令把已装版本换成当前源码（开发用）

```bash
python dev_tests/install_local.py            # 打包 → 逐文件 hash 校验 → 备份 → 替换
python dev_tests/install_local.py --dry-run  # 只看会改哪些目录，不动文件
```

* 自动找 Blender 的 `extensions/user_default/pose_capture`（其次 `scripts/addons/pose_capture`），
  旧版本先备份到 `dist/backup/<版本>-<方式>-<旧版本>_<时间戳>/`
* 不碰数据目录 `datafiles/pose_capture`（模型与 pip 依赖都在那里）
* 装完**必须重启 Blender**；验证：`blender.exe -b --factory-startup --python "dev_tests/test_installed.py"`

> 版本说明：在 **Blender 5.2.1 LTS（Windows）** 上完整实测（安装、检测、写入都跑通）。
> 代码只用了 4.2 ~ 5.x 通用 API（没有用任何 5.x 专有接口），因此在 4.2 LTS / 5.0 / 5.1 上应同样可用。
> **1.0.1**：修复手臂/手腕扭转（roll）对齐——旧版里 roll 分支因 numpy 数组没有 `.length` 而从未执行，
> 且扭转对齐用错了参考轴，会让肘关节旋转与照片不符；另新增 `左臂前后翻转` / `右臂前后翻转` 处理单张图的深度歧义。
> **1.1.0**：接入**深度模型（Depth Anything V2）**：每个关节的深度直接来自像素，深度方向由全局拟合自动决定，
> 于是删掉了 `手肘向后弯 / 膝盖向前弯 / 躯干前倾朝前 / 左臂前后翻转 / 右臂前后翻转` 这 5 个“猜方向”的开关；
> 旧的骨长解算保留为兜底（深度模型缺失、拟合失败或骨长误差偏大时自动使用）。
> **1.2.0**：默认改为**写 Rigify 的 IK 控制器**（`hand_ik` / `foot_ik`）：先切 FK 写旋转，再用 Rigify 自带的
> `pose.rigify_limb_ik2fk_<rig_id>` 快照把 IK 控制器对齐，最后停在 IK 模式 —— 控制器能继续拖、能 K 帧，
> 实测与 FK 结果误差 0.000000 m（`dev_tests/test_rigify_limbs.py`）；`肢体写入方式` 选 `FK 骨骼` 则停在 FK 并
> **自动藏起 IK 控制器、显示 FK 控制器**（反向同理），另加 **同步控制器显示** 按钮兜底。
> 修复**深度图/骨架预览一直是纯黑**：Blender 5.2 里给 GENERATED 图像写完像素后再赋 colorspace 会清空缓冲，
> 现在色彩空间在写像素之前设置；并新增**独立 Tk 深度预览窗口**（照片 / 深度灰度 / 深度伪彩 + 数值统计 + 深度方向核对）。
> **1.3.0**：`IK 控制器` 模式改成**直接写进 IK 控制器**（不切 FK、不调 Rigify 的快照操作符）：
> 跟 Rigify 的 “IK->FK” 按钮同一套算法（`RigifyLimbIk2FkBase.apply_frame_state`）——把 FK 链每根骨骼
> 的**世界增量**（位置 + 旋转）整体套到对应控制器上，链根 `upper_arm_ik` / `thigh_ik` 再按 Rigify 的
> `correct_rotation` 搜一遍绕自身轴的扭转（IK 链的弯曲平面 = 膝盖 / 手肘朝哪边），开了极向目标
> （`pole_vector`）就按 `match_pole_target` 把 `*_ik_target` 摆进 FK 链的平面。写完用 `DEF-*` 变形骨
> 核对（**关节位置** 2 cm / 骨骼方向 10°；不比尾巴，IK 带出来的 DEF 骨会被 stretch 拉长，Rigify
> 自己那条路径也一样），**对不上的肢体自动退回**原来的快照路径 —— 最坏情况只是“没直写成功”，
> 不会写出错姿态（实测手 / 脚朝向与快照一致 0.00°，手臂关节 4~7 mm、左腿 11 mm）。
> IK 模式下仍会把（隐藏的）FK 链写一份，拖 `IK_FK` 滑块回 FK 时姿态不丢。
> `肢体写入方式` 现在也影响**自动匹配骨骼名**：IK 模式优先匹配 `*_ik` 控制器、FK 模式优先 `*_fk`；
> 切换此项后在应用姿态时会按新方式自动重匹配四肢骨名（受 `应用时自动匹配` 控制）。新增纯 Python
> 自检 `dev_tests/test_limb_logic.py`（模式感知匹配 / IK 直写计划 / 直写触发，不需要 Blender）
> 与 Rigify 结构诊断 `dev_tests/_diag_ik_frames.py`（把 IK 控制器与 FK 链的变换关系打出来）。
> **1.3.1**：修“**深度图里手在前面，应用后手却跑到身体后面**”（肩 / 手最明显）。四个成因一起修：
> ① **深度方向定符号**：正面站、身体在深度上本来就扁的照片里，α 的两种符号残差几乎一样
> （实测真实照片只差 0.1%），方向本来等于抛硬币 —— 现在两种情况都试出来放进候选里（旧版
> 高斯-牛顿从正负初值都收敛到同一个解，候选里只有一半），并请**物理判据**（人比背景近，只看
> 像素、与骨长无关）来定夺：代价差不到 10% 时按它定符号，差得开才听拟合的；
> ② **“反转深度方向”真的能翻**（旧版只换了个参数化，姿态其实没翻）；
> ③ **段级骨长回修**：单目深度模型在细肢体上量级偏软（实测小臂 26 cm、深度图只给出 1~2 cm）
> 且会把肢体“糊”到躯干深度上，直接拿它的量级定关节位置时，肘→腕方向几乎由（几乎重合的）
> 像素决定，手就甩到后面 —— 现在只保留深度图的**符号**（谁在前），深度差量级用骨架骨长，
> 沿关节自己的像素射线回修（画面内姿态不变），面板会打印“骨长回修 N 段”；深度图对某段
> 没意见（前后差 < 1 cm）就当没有前后信息、留在画面平面里，**绝不硬塞一个“在后面”**；
> ④ **兜底也用深度图的符号**：退回骨长解算时不再一律按老假设“手肘朝后”（那正是把整条手臂
> 推到背后的原因），有深度图就听深度图的；
> 另外修**手部关键点根本没参与重建**（`place()` 只查 `pts` 字典，手部 21 点在 `hands` 里）——
> 于是深度模式下 `hand_tip` 永远等于“沿小臂外推手长”、**手指一根都不生成**，现在手的方向跟
> 关键点走、手指也能正常生成。自检见 `dev_tests/test_depth.py` 的
> `[大臂朝镜头伸 + 深度被模型糊掉]` / `[兜底：骨长解算也用深度图的前后符号]` /
> `[深度方向：拟合 vs 物理判据]` / `[手部关键点：手的方向 / 手指]` 四组，
> 排错工具 `dev_tests/_diag_hands.py`（逐关节 + 逐段核对前后方向）。
> **1.3.2**：新增 `手部沿小臂`（默认 0.5）—— **让手尽量顺着小臂的方向伸展**。
> 手部关键点只到指根（腕 + 21 点），握拳 / 侧掌 / 指尖错到掌根另一侧时，`腕 -> 指根`
> 与小臂的夹角能到 40~90°，照单全收会让手掌**横着长在腕上**；而真实照片里手基本都
> 顺着小臂。现在以**腕为轴**把整只手（手掌 + 手指）朝小臂方向转：手长仍用骨架手长、
> 手的形状（握拳 / 张开）完全保留、手指跟着手掌一起转（与腕的距离不变）；
> `手部沿小臂` = 0 不拉（只在偏角 > 30° 时夹住）、0.5 偏角减半（默认）、1 完全顺着小臂。
> 自检见 `dev_tests/test_depth.py` 的 `[手部关键点：手的方向 / 手指 / 沿小臂伸展]`
> 与 `[手部沿小臂：混合比例 / 偏角上限]`（后者连“朝小臂转而不是转反”一起量）。
> **1.3.3**：**用 68 点人脸优化头 / 脖子的骨骼映射**（以前只用“两只耳朵 + 眉毛/下巴”）。
> ① **头部朝向改用五官投票**：眼线（眼角 / 嘴角 / 眉梢 / 下颌角成对平均）给出头部的**侧轴**，
> 下巴 -> 眼睛（+ 嘴角 / 鼻梁投票）给出**上轴**；`head` 骨骼瞄准上轴，`eye_line` 参照轴拿侧轴
> —— 单点抖动（睫毛、嘴角、下巴检测误差）不再把整个头带歪；
> ② **修“转头 / 歪头根本没写进去”**：`rigify_map` 以前只给了 `hip_line` / `shoulder_line`，
> 而 `head` 用的参照轴是 `eye_line`，`solve_orientation` 拿到 `rest_ref=None` 会**静默跳过扭转**
> —— 照片里转头、歪头的头部信息整体丢失（头永远朝着静止方向）。现在静止眼线取肩线（人物右 -> 左），
> 并且目标眼线的符号也统一到肩线（背对镜头时不会拧 180°）；
> ③ **头基座（head 骨起点 / 脖子顶端）不再迷信耳朵**：耳朵是身体分支，只检出一只、或者和五官
> 推出的头基座（眼线下 0.15 ×（眼睛 -> 下巴）≈ 耳道口高度）差得离谱时改用五官估；
> ④ **head 方向不再采“额头发际”那一个像素的深度**（那里常常是头发 / 背景）：深度模式下方向由
> 十几对五官点各自带真实深度投票，点头 / 仰头终于有可信的前后分量；
> ⑤ 五官成对点互相矛盾（深度图在脸上给错）时**丢掉前后分量**、只保留画面内的方向 ——
> 宁可不转头，也不让深度噪声把脑袋拧过来。
> 自检见 `dev_tests/test_face_head.py`（五官坐标系 / 头基座来源 / 两种解算模式下的 head 方向、
> eye_line、脖子方向 + Rigify 扭转是否真的生效）。


---

## 2. 首次准备（只需一次）

打开 `编辑 > 偏好设置 > 附加组件 > Pose Capture`，展开后：

1. **安装/更新依赖**：自动调用 Blender 自带的 Python 执行
   `pip install --target <插件数据目录>/libs onnxruntime`
   * 想用 GPU 可把“要安装的依赖包”改成 `onnxruntime-directml`（Windows）或 `onnxruntime-gpu`（NVIDIA）
   * 装完后“可用 provider”里会显示 `CPUExecutionProvider` 等
2. **在线下载模型**（DWPose 约 340MB + 深度模型，一次性）
   * `yolox_l.onnx`（人体检测，216MB）
   * `dw-ll_ucoco_384.onnx`（姿态，134MB）
   * `depth_anything_v2_small.onnx`（深度，94MB；偏好设置里可换 fp16 48MB / int8 27MB 档位）
   * 也可以填“本地模型目录”（例如已经给 ComfyUI/SD 下载过 DWPose / Depth Anything 的目录），
     点 **从本地目录导入**，插件会递归查找这些文件（`depth*.onnx` 会一起导入）

数据目录（依赖 + 模型都放这里，不会污染 Blender 安装目录）：

```
Windows: %APPDATA%\Blender Foundation\Blender\<版本>\datafiles\pose_capture\
macOS:   ~/Library/Application Support/Blender/<版本>/datafiles/pose_capture/
Linux:   ~/.config/blender/<版本>/datafiles/pose_capture/
```

若网络不通（`huggingface.co` 连不上时插件会自动改试镜像 `hf-mirror.com`），
也可以手动下载后放进 `.../pose_capture/models/`：

* `https://huggingface.co/hr16/yolox-onnx/resolve/main/yolox_l.onnx`
* `https://huggingface.co/yzd-v/DWPose/resolve/main/dw-ll_ucoco_384.onnx`
* `https://huggingface.co/onnx-community/depth-anything-v2-small/resolve/main/onnx/model.onnx`
  → 存成 `depth_anything_v2_small.onnx`
* 或者用偏好设置里的 **导入深度模型** 直接指定任意已有的 `depth*.onnx`

---

## 3. 使用步骤

在 3D 视图按 `N` 打开侧栏，切到 **Pose Capture** 标签页：

| 步骤 | 操作 |
| --- | --- |
| 1 输入图片 | 选“图片文件”并指定路径；“已加载图像”则从 Blender 里挑一个图像数据块。多人时用**人物序号**（0 = 面积最大的人）挑一个 |
| 2 检测 | 点 **检测姿态**（后台线程跑，不卡界面；按 ESC 可取消）。完成后会生成 `PoseCapture_Preview` 预览图，点旁边的小图标可推到图像编辑器查看骨架；深度模型就绪时会同时算出深度图（`PoseCapture_Depth`，**近=蓝、远=红**，灰度图则**越白=越远**）。旁边的 **深度图预览窗口** 会另开一个独立 Tk 页面，把照片 / 深度灰度 / 深度伪彩并排显示、打出数值统计、并给出**深度方向核对**（物理判据“人比背景近” vs 重建拟合方向，不一致会直接标出来）—— 判断“深度是不是真的算对了”用它最稳（不受 Blender 图像显示链路影响） |
| 3 目标骨架 | 点 **新建元骨架** + **生成控制骨架** 得到标准 Rigify 人形骨架；也可以直接用你自己已经绑好的 Rigify 骨架（用吸管按钮绑定当前选中对象） |
| 4 映射 | 点 **自动匹配骨骼**（生成骨架会自动认出 `spine_fk / upper_arm_fk.L / thigh_fk.L / neck / head / torso`，元骨架认出 `spine / spine.001 / upper_arm.L …`）。个别骨骼可以在“骨骼映射”子面板里手动指定 |
| 5 参数 | 按需调姿态解算参数（见下表）。`深度来源` 默认用**深度模型**（前后方向来自像素）；想用旧方式就选 `骨长解算` |
| 6 应用 | 点 **应用到骨架**：只写旋转（可选根位移），自动切到 Pose 模式方便查看；不满意可点 **清空 Pose** 重来 |

> 生成骨架的手臂/腿默认挂在 **IK 控制器**（`hand_ik.L` / `foot_ik.R` …）上。
> 插件默认按 `IK 控制器` 写入：**直接把姿态写进 IK 控制器** —— 跟 Rigify 那个 “IK->FK” 按钮
> 同一套算法（`RigifyLimbIk2FkBase.apply_frame_state`）：把 FK 链每根骨骼的“世界增量”整体套到
> 对应控制器上（**位置和旋转一起**，`hand_ik` / `foot_ik` 本身就是 IK 链末端，Rigify 用一根带
> `COPY_LOCATION` 的 MCH 骨把它接到链上），链根 `upper_arm_ik` / `thigh_ik` 再按 Rigify 的
> `correct_rotation` 搜一遍绕自身轴的扭转（= 弯曲平面，膝盖 / 手肘朝哪边）；开了极向目标
> （`pole_vector`）就按 `match_pole_target` 把它摆进 FK 链的平面。
> 不切 FK、也不调 Rigify 的快照操作符，停在 IK 模式，控制器继续能拖、能 K 帧。
> 自动匹配骨骼名时肢体也会优先匹配 IK 控制器（`upper_arm_ik.L` / `hand_ik.L` …）。
> FK 链会顺手写一份（IK 模式下它是隐藏的），所以你拖 `IK_FK` 滑块回 FK 时姿态不丢。
> 写完插件会用 `DEF-*` 变形骨核对一遍（关节位置 2 cm / 方向 10° 以内），对不上的肢体自动退回
> “切 FK + Rigify 快照”的老路径（那条路径实测 0.000000 m），
> 面板摘要里会写清楚哪些是直写、哪些走了快照。
> 想停在 FK 就把 `肢体写入方式` 改成 `FK 骨骼`；这时插件会自动**藏起 IK 控制器、显示 FK 控制器**
> （反向同理），免得看见一组拖不动的控制器以为插件坏了。自己拖过 Rigify 的 IK-FK 滑块之后，
> 点面板里的 **同步控制器显示** 就能让显示跟上滑块。


---

## 4. 参数说明

| 参数 | 说明 |
| --- | --- |
| 人体框 / 关键点置信度阈值 | 检测的过滤阈值。检测不到人就降低人体框阈值；关键点乱飘就提高关键点阈值 |
| **人物朝向** | 正面朝镜头 / 背面朝镜头 / 面朝画面左 / 右。决定“画面深处”与左右手的关系，选错会出现整体镜像或反向 |
| **深度来源** | `深度模型`（默认）：用 Depth Anything V2 给每个关节解真实深度，前后方向来自像素；`骨长解算`：旧的 `dz=±√(L²-d²)` + 固定方向假设（兜底）；`压平到画面平面`：不做深度 |
| 深度模型档位 / 透视强度 / 反转深度方向 | 档位在插件偏好设置里选（fp32 95MB 最准 / fp16 48MB / int8 27MB 最快）；透视强度 = 焦距系数 `f = 系数 × 图片长边`（偏平调大、偏夸张调小）；方向反了才勾“反转” |
| 估算骨骼扭转(roll) | 用肩线、胯线、手臂弯曲轴对齐骨骼 roll，避免手臂/头部拧麻花。出问题可关掉或降低“扭转强度” |
| 交换左右 | 检测把人物左右判反时用（极少见） |
| 写入手指 | 用 21 点手部关键点写 3 节手指（`thumb.01-03` / `f_index.01-03` …）。手部检测质量差时建议关闭 |
| 手部沿小臂 | **让手尽量顺着小臂的方向伸展**（默认 0.5）。手部关键点只到指根，握拳 / 侧掌时 `腕 -> 指根` 能偏出 40~90°，照单全收会让手掌横在腕上；这里以腕为轴把整只手（含手指）朝小臂转：`0` 不拉（偏角 > 30° 才夹住）、`0.5` 偏角减半、`1` 完全顺着小臂。手长与手的形状都不变 |
| 写入根骨骼位移 | 把人物的水平位置 + 站立高度写进 `root` 骨骼；配合“落地对齐”把脚放回地面高度 |
| **肢体写入方式** | `IK 控制器`（默认）：把姿态**直写**进 `upper_arm_ik` / `thigh_ik`（链根旋转）、`hand_ik` / `foot_ik`（末端朝向）和 IK 目标（手腕 / 脚踝位置），不切 FK、不做快照；写完用 `DEF-*` 变形骨核对，对不上的肢体自动退回 Rigify 的 “IK->FK” 快照（老路径，实测误差 0.000000 m）；`FK 骨骼`：姿态停在 FK 骨骼上并把 `IK_FK` 滑块设为 1 |
| 同步控制器显示 | 按 IK/FK 状态只显示“能用”的那组控制器（骨骼集合可见性）。滑到 FK 时藏 IK、显示 FK，反之亦然；自己拖过 Rigify 滑块后点面板里的 **同步控制器显示** 即可 |
| 插入关键帧 | 在当前帧给被写入的骨骼插关键帧 |
| 比例微调 | 全局比例的乘算修正（默认 1.0）。姿态整体偏“扁平”就调大，偏“夸张三维”就调小 |
| 骨骼映射 | 角色 → 骨名。留空表示跳过该角色 |

---

## 5. 实现原理

1. **检测**：`yolox_l.onnx`（640×640 letterbox，BGR、0-255）→ NMS → 取面积最大的人（或指定序号）
2. **姿态**：`dw-ll_ucoco_384.onnx`。按 mmpose 的做法把人框扩 1.25 倍、固定 0.75 宽高比、仿射裁剪到 288×384、ImageNet 均值方差归一化 → SimCC 解码（split ratio 2）→ 映射回原图，得到 COCO-WholeBody **133** 点（17 身体 + 6 脚 + 68 脸 + 21×2 手）
3. **深度**：`depth_anything_v2_small.onnx`（等比缩放到 518×518 内、长宽取 14 的倍数、ImageNet 均值方差归一化）预测相对深度，按关键点像素采样（3×3 中值，抗边缘渗色），得到每个关节的深度值
4. **相机 + 全局拟合**：`X=(u-cx)·Y/f`、`Z=(cy-v)·Y/f`、`Y=β+α·n`（`n` = 归一化相对深度，`f` = 焦距系数 × 图片长边）。未知量只有 `(α, β)` 两个**全局**参数，用“全图所有骨段的 3D 长度 = 骨架骨长”这组约束做一次**高斯-牛顿最小二乘**（多起点 + Huber 鲁棒核 + 弱正则），并同时试“深度线性 / 视差(1/深度)线性”两种参数化取残差小的那个。**两个方向都会显式评估**（α 取反 = 整套姿态前后翻过来），选代价小的；拟合分不出前后时（两种方向代价差不到 10% —— 正面站、身体在深度上本来就扁的照片实测只差 0.1%）改用**物理判据**（人所在的框里深度值整体比框外大还是小，只看像素、与骨长无关）来定符号。于是**深度方向、模型输出的语义都由拟合自己决定**——透视下“谁离镜头更近”会改变像素尺度，方向是可辨识的，不再需要 `手肘朝前还是朝后` 这类猜测
5. **段级骨长回修**：深度图的**符号**（谁在前）决定方向，深度差**量级**用骨架骨长 —— 沿每个关节自己的像素射线解出“让这段骨长成立”的深度（画面内位置不变）。单目深度模型的量级在细肢体上偏软（实测小臂 26 cm 只给出 1~2 cm），量级不对时关节方向会被像素带跑（肘腕像素几乎重合时尤其严重，手就甩到身体后面）。只在 |实际骨长 - 骨架骨长| 超过 `max(2 cm, 12%)` 时回修，面板的“解算：…”一行会打印“骨长回修 N 段”；深度图对某段没意见（前后差 < 1 cm）时当它没有前后信息、留在画面平面里
6. **兜底**：深度模型缺失 / 拟合失败 / 骨长误差明显大于旧方法时自动退回骨长解算（正交投影 + `dz=±√(L²-d²)`），深度符号优先用深度图给的（**同一段的符号**），没有深度图才用固定方向假设；实际用了哪种会显示在面板的“解算：…”一行里
7. **手部沿小臂**（`手部沿小臂`，默认 0.5）：手部关键点只到指根，握拳 / 侧掌 / 指尖错位时 `腕 -> 指根` 与小臂夹角可以到 40~90°。最后一步以**腕为轴**把整只手（手掌 + 手指）朝小臂方向转一个角度：`偏角 -> min(偏角 × (1 - 手部沿小臂), 30°)`，手长仍取骨架手长（`hand.L` 的长度）、手指按同一个旋转刚性跟随（与腕的距离不变，握拳/张开的形状保留）。两种模式（深度 / 骨长兜底）都做这一步，所以 `hand_tip` 与手指永远一致
8. **躯干刚性偏移**：肩、胯、颈、头基座相对躯干的静止偏移会随重建出的躯干方向一起旋转（否则锁骨、脖子方向会明显拐）
9. **五官定头 / 脖子**（见 `reconstruction._face_frame`）：耳朵只检出一只、或者与五官推出的头基座差得远时，
   头基座（`head` 骨起点 / 脖子顶端）= 眼线往下 `0.15 ×（眼睛 -> 下巴）`（≈ 耳道口高度）；
   `head` 骨瞄准的“上轴”与 `eye_line` 参照轴的“侧轴”都由五官**成对点投票**（眼角 / 嘴角 / 眉梢 / 下颌角、
   下巴 -> 眼睛 / 眉毛 / 嘴角 / 鼻梁），深度模式下每一对点带自己的真实深度 —— 转头、歪头、点头都从像素里来，
   而不是从两三个点的检测噪声里来；几对点互相矛盾（深度图在脸上给错）时丢掉前后分量、只保留画面内的方向
10. **朝向归一**：按“人物朝向”把整套坐标旋到骨架静止坐标系（角色面朝 -Y），于是左/右关键点天然对应 `.L` / `.R`。
    五官给出的眼线是“画面左 -> 画面右”，符号会按肩线统一成“人物右 -> 人物左”（背对镜头时不会拧 180°）
11. **重定向**：
   * 摆动(aim)：`rest_dir.rotation_difference(target_dir)`
   * 扭曲(twist)：把静止参照轴（肩线/胯线/**肘弯曲轴**/**眼线**）按摆动搬运到姿态下，再绕骨骼轴**有符号**对齐到目标参照轴。手掌与手指也跟随“手臂弯曲平面”，等价于腕部不带额外扭转
   * 静止眼线取**肩线**（人物右 -> 左；head 骨自己的 roll 约定各家骨架不同，而肩线天然定义人物左右）：
     没有这个键时 `solve_orientation` 会静默跳过扭转 —— 照片里的**转头 / 歪头整体丢失**
   * 用 `pose_bone.matrix` 逐级写入（父级先写、写完更新），只改旋转不动位置，骨架比例保持不变
   * `torso` 这类 Rigify 特制的“脊椎 IK 控制”静止方向不是身体轴，插件对它改用“胯线旋转 + 位移补偿”，等效于绕骨盆旋转
12. Rigify 生成骨架的肢体：
   * `IK 控制器`（默认）：直接把姿态写进 IK 控制器 —— 跟 Rigify 的 “IK->FK” 按钮同一套算法（`RigifyLimbIk2FkBase.apply_frame_state`）：把 FK 链每根骨骼的世界增量整体套到对应控制器上（**位置 + 旋转**，`hand_ik` / `foot_ik` 就是 IK 链末端），链根 `upper_arm_ik` / `thigh_ik` 再按 Rigify 的 `correct_rotation` 搜一遍扭转（弯曲平面），开了 `pole_vector` 就按 `match_pole_target` 摆极向目标 `*_ik_target`。IK 模式下 FK 链是隐藏的，但也会写一份（拖 `IK_FK` 滑块回 FK 时姿态不丢，退回快照时也靠它）。写完用 `DEF-*` 变形骨核对（**关节位置** 2 cm / 骨骼方向 10° 以内；不比尾巴 —— IK 链带出来的 DEF 骨会被 stretch 拉长，Rigify 自己那条路径也一样），对不上的肢体**自动退回**老路径 —— 所以最坏情况只是“没直写成功”，不会写出错姿态。实测：手 / 脚朝向与 Rigify 快照完全一致（0.00°），手臂关节 4~7 mm、左腿 11 mm，姿势变化特别大的那条腿会退回快照
   * 老路径（`FK 骨骼` 模式、映射指向 FK 骨骼、或直写没通过）：先把写到的肢体切到 FK，写旋转，再调用 Rigify 自带的 `pose.rigify_limb_ik2fk_<rig_id>` 快照把 IK 控制器对齐，最后按 `肢体写入方式` 停在 IK（默认）或 FK
   * 两种路径都按 `肢体写入方式` 同步控制器可见性（切 FK 藏 IK，反之亦然）

---

## 6. 已知限制（单张图固有问题）

* **深度是模型估的，不是测的**：Depth Anything V2 属单目估计，严重遮挡（手挡在身前、肢体交叉）或大面积纯色背景会有误差，表现为**个别肢体前后略偏**。先点 **深度图预览窗口**（独立 Tk 页面：照片 / 深度灰度 / 深度伪彩 + 数值统计 + 方向核对）看质量：伪彩里**脸/手这些近的地方应该是蓝的、远背景是红的**（灰度图则是越白越远），再考虑换 fp32 档位、调 `透视强度`，或把 `深度来源` 改成 `骨长解算`
* **整张图前后反了**：正常情况下方向由拟合自动决定，拟合分不出前后时由**物理判据**（人比背景近）兜底定符号；确实反了才勾 `反转深度方向`（1.3.1 起这个开关真的会把整套姿态前后翻过来；勾了以后就不用物理判据了，物理判据不适用时用它），勾错会自动退回骨长解算
* **脊椎按直线近似**：单张图看不出每节脊椎的弯曲分布，插件把“骨盆→肩中点”的直线均匀分配到 4 段脊椎骨骼，所以最下面一节脊椎的方向误差可能到 10~15°（整体躯干朝向仍是对的）
* **骨架比例与人物不同**：插件保证“方向”正确，骨骼长度用你骨架的，所以绝对位置会有偏差
* **脚趾不单独写入**：单张图无法判断脚趾弯曲，`toe.L/R` 保持与脚掌的静止关系（跟着脚掌一起转）
* **侧身照**：深度来自深度模型，侧身/半侧身比旧版（骨长反解）好很多；但深度图质量差的照片仍会偏
* **脸部**：头部朝向来自 68 点人脸（眼线 / 上轴投票 + 头基座），不做表情 / 面骨；深度图在脸上的质量决定**转头**的可信度（转头的前后分量全部来自五官点的深度差，模型糊掉时会自动只保留画面内的歪头）
* **比例微调**：只影响深度解算强度与根位移，不影响画面内的投影（投影始终等于检测到的关键点）

---

## 7. 常见问题

**装依赖失败**：手动在命令行执行（Windows 示例，路径按自己安装位置改）：

```bat
"D:\SteamLibrary\steamapps\common\Blender\5.2\python\bin\python.exe" -m pip install ^
  --target "%APPDATA%\Blender Foundation\Blender\5.2\datafiles\pose_capture\libs" onnxruntime
```

装完重启 Blender。

**没有检测到人体**：把“人体框置信度阈值”降到 0.1~0.2，或换一张主体更大的图。

**姿态很扁平 / 很夸张**：调 `透视强度`（偏平→调大；偏夸张→调小），或调“比例微调”。

**手肘/膝盖的朝向和照片不符**：先点 **深度图预览窗口** 看深度图是否合理（伪彩近=蓝、远=红，人物应该比背景蓝；灰度图越白越远），再核对窗口里那行**深度方向**：如果写着物理与拟合不一致（✗），说明整套姿态前后反了，勾 **反转方向** 重算一次。
深度图正常但个别肢体仍反了，说明那处遮挡严重：可以把 `深度来源` 改成 `骨长解算` 用旧方式，或勾 `反转深度方向` 试另一个方向。

**提示“深度解算骨长误差偏大，已退回骨长解算”**：深度图与骨架比例对不上（人物太小、背景杂乱、或档位太低），
插件已自动改用骨长解算保证不出错；想用深度就把人物拍大一点或换 fp32 档位。

**手臂、头拧成麻花**：先确认 `估算骨骼扭转` 是开着的（关掉会退回“极小旋转”，手臂 roll 完全随机）；仍然拧就降低“扭转强度”。若整条小臂像被翻折 180°，多半是上面那条深度歧义，按前一节翻转对应手臂。

**手掌横着长在腕上 / 手的方向和照片差很多**：调 `手部沿小臂` —— 它是“把手朝小臂方向拉多少”，默认 0.5（偏角减半），`1` 完全顺着小臂伸展，`0` 完全不拉（只把超过 30° 的偏角夹住）。手部关键点只到指根，握拳、侧掌、指尖错到掌根另一侧时偏角能到 40~90°，照单全收就会横在腕上；调大更“顺着手臂”，调小更“照手部关键点”。手长与手的形状（握拳 / 张开）不受影响。

**点了应用没反应**：看日志里的“已写入 N 根骨骼”。如果骨架是 Rigify 生成骨架，再点一下 **同步控制器显示**（确认当前 IK/FK 那组控制器是可见的）。

**日志**：面板底部“日志”子面板显示最近操作；`窗口 > 切换系统控制台` 里有完整输出（含 pip 输出）。

---

## 8. 附加：把骨骼图给 ControlNet 用（独立工具）

同一个仓库里带了一个独立小工具 `pose_to_controlnet/`，把照片直接转成 **ControlNet 可用的
OpenPose 骨骼图 PNG + OpenPose JSON**（不需要 Blender，推理复用插件那套纯 numpy 核心和同一批模型）：

```bat
python pose_to_controlnet\pose_to_controlnet.py                       :: 图形界面
python pose_to_controlnet\pose_to_controlnet.py --input photo.jpg -o out
python pose_to_controlnet\pose_to_controlnet.py --input photos -o out --size-mode long_side --long-side 768
python pose_to_controlnet\pose_to_controlnet.py --check               :: 环境自检
```

* 默认 `openpose` 画法 = ControlNet 官方 annotator 的画法；输出 `<名字>_pose.png` + 同名 `.json`
* 尺寸三选一：`source` 原图 / `long_side` 长边固定 / `person` 按人物框裁方形
* 界面里可以「安装依赖」（装进工具自己的目录，不动系统环境）和「下载模型」；
  也会自动复用插件已装好的依赖与模型
* 详细用法见 `pose_to_controlnet/README.md`，自检：`python dev_tests/test_tool.py`

---

## 9. 开发者自检

```bat
:: 0) 模块导入 / 注册自检
blender.exe -b --factory-startup --python "dev_tests/test_import.py"
:: 1) 合成姿态精度自检（元骨架）
blender.exe -b --factory-startup --python "dev_tests/test_pipeline.py"
:: 1b) 深度解算自检（合成真值姿态 + 合成深度图；含方向判定 / 骨长回修 / 手部关键点 / 兜底）
blender.exe -b --factory-startup --python "dev_tests/test_depth.py"
:: 1b1) 头 / 脖子自检（五官坐标系 / 头基座来源 / 两种模式下的 head 方向与 eye_line / Rigify 扭转）
blender.exe -b --factory-startup --python "dev_tests/test_face_head.py"
:: 1b2) 肩 / 手前后方向的排错诊断（逐关节 + 逐段核对，可传自己的图片）
blender.exe -b --factory-startup --python "dev_tests/_diag_hands.py" -- <图片路径> [人物序号]
:: 1c) 肢体写入的纯逻辑自检（模式感知匹配 / IK 直写计划 / 直写触发）—— 普通 Python 就能跑
python dev_tests/test_limb_logic.py
:: 2) Rigify 生成骨架自检：自动映射 + FK/IK + 变形位置还原
blender.exe -b --factory-startup --python "dev_tests/test_rigify_rig.py"
:: 2a) Rigify 肢体 IK 控制器自检：解析链 / IK 直写 + 快照退回 / 控制器可用 / 显示同步
blender.exe -b --factory-startup --python "dev_tests/test_rigify_limbs.py"
:: 2a2) Rigify 结构诊断：把 IK 控制器 / 极向目标 / 约束关系与 FK 链的变换关系打出来（排错用）
blender.exe -b --factory-startup --python "dev_tests/_diag_ik_frames.py"
:: 2b) 深度预览自检：图像写入（黑图回归）+ PNG + 独立预览页面的数据
blender.exe -b --factory-startup --python "dev_tests/test_depth_view.py"
:: 2b) 写入后的 roll（扭转）自检：DEF 骨姿态矩阵必须与真值一致
blender.exe -b --factory-startup --python "dev_tests/test_retarget_roll.py"
:: 3) 真实模型端到端（先准备依赖与模型）
blender.exe -b --factory-startup --python "dev_tests/_setup_deps.py"
blender.exe -b --factory-startup --python "dev_tests/test_onnx_end2end.py"
:: 4) 打包 + 安装自检（会真的装进 Blender）
python build_zip.py
blender.exe -b --factory-startup --python "dev_tests/test_install.py"
:: 4b) 或者：把本机已装版本直接换成当前源码（打包 + hash 校验 + 备份 + 替换）
python dev_tests/install_local.py
:: 5) 独立工具（骨骼图 → ControlNet）自检，用系统的 python
python dev_tests/test_tool.py
:: 6) 已安装副本自检：确认 Blender 里那份就是当前仓库源码（只读，不动偏好设置）
blender.exe -b --factory-startup --python "dev_tests/test_installed.py"
```

预期：`test_pipeline` 比例误差 <1%、四肢方向平均 ~3°/最大 <10°；`test_rigify_rig` 映射全 OK、画面内偏差 <25 像素；`test_rigify_limbs` 四条肢体链解析 OK、**直写的肢体**与 FK 的 DEF 关节最大位置差 ≤2 cm（实测手臂 4~7 mm、腿 11~12 mm）、**退回快照的肢体** <0.1 mm、控制器拖动有效、显示同步 OK；`test_limb_logic` 全部通过（不需要 Blender）；`test_depth` 三组合成深度图（线性 / 视差 / 前后取反）关节误差平均 ~4 mm、最大 ~10 mm，**大臂朝镜头伸**时手仍在身体前面且大臂长度回到骨长，**兜底**时也用深度图的符号（没深度图才退回老假设），**手部关键点**被用上、手指能生成；`test_depth_view` 写入的图不是全黑、PNG 回读正常、预览窗口能建起来；`test_retarget_roll` 手臂姿态矩阵误差 <15°（roll 对齐）；`test_onnx_end2end` 人体结构检查全 OK、写入一致性 0.00°、停在 IK 模式且控制器可拖；`test_installed` 显示“与仓库源码完全一致 ✓”且版本号与 `bl_info` 一致；“通过 ✓”。

---

## 10. 目录结构

```
Pose Capture/
├─ pose_capture/            插件本体（可直接拷到 scripts/addons）
│  ├─ __init__.py           注册、bl_info
│  ├─ props.py              场景属性 + 偏好设置
│  ├─ operators.py          操作器（含后台线程模态任务）
│  ├─ ui.py                 3D 视图 N 面板
│  ├─ dwpose.py             ONNX 推理（onnxruntime / opencv.dnn 两种后端）
│  ├─ imops.py              纯 numpy 图像处理（缩放/仿射/NMS/预览绘制）
│  ├─ coco.py               COCO-WholeBody 133 点定义
│  ├─ depth.py              深度模型（Depth Anything V2）推理 + 按关键点采样
│  ├─ reconstruction.py     2D→3D（深度模型 + 透视全局拟合，骨长解算兜底）
│  ├─ rigify_map.py         角色↔骨骼映射、静止尺寸测量
│  ├─ rigify_limbs.py       Rigify 肢体 IK 控制器桥接（IK 直写计划 + IK->FK 快照兜底 + 显示同步）
│  ├─ retarget.py           方向+扭转 → Pose 骨骼矩阵（默认直写 IK 控制器，可退回 FK + 快照）
│  ├─ depth_view.py         独立 Tk 深度图预览页面（只依赖 tkinter，可单独运行）
│  ├─ deps.py               依赖安装、模型下载/导入
│  └─ utils.py              路径、日志、图像读取
├─ pose_to_controlnet/      独立工具：照片 → ControlNet 用的骨骼图 + JSON
│  ├─ pose_to_controlnet.py 直接运行入口（无参数开界面）
│  ├─ README.md             工具用法
│  └─ pose_to_cn/           实现（paths/core_loader/imgio/render/json_io/options/
│                           engine/pipeline/depinstall/cli/gui/launcher）
├─ dev_tests/               自检脚本（不随插件发布）
│  ├─ test_depth.py         深度解算自检（合成真值姿态 + 合成深度图）
│  ├─ test_face_head.py     头 / 脖子自检（五官坐标系 / 头基座来源 / head 方向与 eye_line / Rigify 扭转）
│  ├─ test_limb_logic.py    肢体写入纯逻辑自检（模式感知匹配 / IK 直写计划；普通 Python 可跑）
│  ├─ test_rigify_limbs.py  Rigify 肢体 IK 控制器自检（直写 + 快照退回 / 显示同步）
│  ├─ _diag_ik_frames.py    Rigify 结构诊断（IK 控制器 / 极向目标 / 约束 vs FK 链；排错用）
│  ├─ test_depth_view.py    深度预览自检（黑图回归 / PNG / 预览页面数据）
│  ├─ test_installed.py     已装副本 vs 仓库源码：逐文件 hash 比对 + 能否注册
│  └─ install_local.py      一条命令替换本机已装版本（打包+校验+备份+替换）
├─ dist/                    打包产物 zip；dist/backup/ 放旧版本备份
├─ blender_manifest.toml    扩展清单模板
├─ build_zip.py             打包脚本
└─ README.md
```

---

## 11. 许可

* 插件代码：MIT
* 模型：`dw-ll_ucoco_384.onnx` 来自 [IDEA-Research/DWPose](https://github.com/IDEA-Research/DWPose)（Apache-2.0），`yolox_l.onnx` 来自 [Megvii YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)（Apache-2.0），`depth_anything_v2_small.onnx` 来自 [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2)（Apache-2.0，ONNX 导出：[onnx-community](https://huggingface.co/onnx-community/depth-anything-v2-small)）。插件**只提供下载/导入功能，不随插件分发模型文件**。
