# Pose Capture — 图片 → Easy-DWPose / DWPose → Rigify 骨骼姿态

用一张人物图片，自动识别人体骨架，并把姿态转换成 **Blender Pose 模式下 Rigify 标准骨骼**可以直接使用的姿态数据。

```
（推荐）图片 ──► YOLOX-L 人框 ──► SAM 3D Body（外部 python）MHR70 三维人体（含手指/脚）
      ──► 朝向/相机自动判定 ──► Rigify 控制骨骼旋转(+可选根位移 / 网格导入)

（兜底）图片 ──► YOLOX-L 人体检测 ──► DWPose(dw-ll_ucoco_384) 133 关键点
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
> 实测与 FK 结果一致（当时量到 0.000000 m，后来发现是依赖图没推完读到的旧值 —— 真实关节位置差几毫米，
> 见 1.3.4；`dev_tests/test_rigify_limbs.py`）；`肢体写入方式` 选 `FK 骨骼` 则停在 FK 并
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
> 另外 README 补了 **2.1 离线包**：依赖 / 模型下载不下来时，把 `pose_capture_datafiles.zip`
> 解压到 `<版本>\datafiles\` 即可（含“别解压进插件目录 / 别多套一层 / 大版本要换 libs”等坑）。
> **1.3.4**：按“FK / IK 实测结论”优化照片转姿态：
> ① **按肢体种类选写入方式**：`肢体写入方式` 新增 `腿 IK + 臂 FK` / `腿 FK + 臂 IK`。实测两种模式
> 的变形结果只差 0.2 px 量级（**FK/IK 不是精度来源**），差别在“好不好改”：`foot_ik` 在根空间，
> 改躯干 / 骨盆时脚还钉在原位（转 torso 10° 实测脚动 0.2 mm，FK 会动 174 mm），而手臂的 roll
> 由弯曲轴解析算出、比 IK 的极点搜索更直接 —— 于是可以“腿钉住接触、臂要精确 roll”；
> ② **直写后再复核一遍**：全部肢体写完，用 `DEF-*` 按**实际最终结果**再量（8 mm / 4°），超标的
> 肢体自动改走 Rigify 的 `IK->FK` 快照（以前只按“写那一下”的 2 cm 核对，膝盖能差 2 cm）；
> ③ **直写没成功时的收尾可选**：`直写没成功时` = `退回 Rigify 快照`（默认）或 `停在 FK`；
> 快照失败（生成脚本被删）时也会**停在 FK**，不再切回 IK 把姿态藏在控制器后面（看起来像没生效）；
> ④ **修“IK 模式量错骨架尺寸”**：`read_metrics` 现在把 `hand_ik` / `foot_ik` 这些控制器换回
> FK 几何再量（实测 `foot_ik` 静止方向是 `(0,1,0)`、`foot_fk` 是 `(0,-0.85,-0.53)`），
> 同一个人同一张照片，切成 IK 或 FK 不再解出不同比例 / 扭转（IK 画布内偏差 23.42 px vs FK 23.54 px）；
> ⑤ **修“切回 FK 后膝盖还差 37 mm”**：改完 `IK_FK` 要等依赖图把驱动（约束影响）推两遍，
> 现在改完模式会 `evaluated_depsgraph_get().update()` 一次再读（以前读到的是上一帧的 `DEF-*`，
> 所以“快照 0.000000 m”那个数其实是旧值 —— 真实关节位置差几毫米，膝盖最明显）。
> 面板摘要会打印 `直写偏差 ≤ N mm`；`test_limb_logic` 新增按种类拆分的用例，
> `test_rigify_limbs` 新增 `[3c] 腿 IK + 臂 FK`、`[3d] 停在 FK` 两组。
> **1.4.0**：新增 **SAM 3D Body** 作为单图 3D 来源（`3D 来源 = 自动 / SAM 3D Body /
> DWPose+深度 / 骨长解算`，见 **5.1**）。推理走**外部 python 子进程**
> （`sam3d_worker.py`，torch / detectron2 不进 Blender），返回 MHR70 的 70 个 3D 关键点
> + 相机 + 网格，再由 `sam3d.build_recon()` 换算成和 `reconstruction.reconstruct()`
> **同结构**的 recon —— 重定向 / IK / 快照那一整套完全复用。相机 Y 轴方向与是否要加
> `pred_cam_t` 由**2D 重投影投票**决定（不靠猜）；朝向由 3D 肩线 / 胯线自动判定（这项选项
> 在 SAM 模式下不参与）；头向上轴用「鼻子-耳中点 × 眼线」并按解剖常数（鼻尖比耳道低
> 10°）校正，平视时不会常年低头；手指/脚趾直接来自模型（`fingers` 不必再靠猜）。
> 另加 **导入 SAM 3D Body 网格**（`pose_capture.import_sam3d_mesh`：与关节同一套变换，
> 骨盆对齐到目标骨架、按腿长比例缩放，脚正好落在骨架地面）。自检：纯逻辑
> `dev_tests/test_sam3d.py`（相机投票 / 角色映射 / 朝向归一 / 网格变换 / 与
> `retarget.build_targets` 的接口），Blender 端 `dev_tests/test_sam3d_rig.py`
> （合成 MHR70 → 写 Rigify：5 根骨骼方向差 0.0°、左大臂相对静止转 35.7°、网格脚底
> 0.017 m = 骨架地面 0.017 m）。
> **1.4.1**：新增 **MMD 骨骼 → Mixamo**（`MMD → Mixamo` 子面板 / `pose_capture.mmd_to_mixamo`，
> 见 **5.2**）。骨骼对照表整理自 MMD Bone Tools（`QIangIQ/MMD-bone-SDK`，MIT）：下半身→Hips、
> 上半身(2/3)→Spine(1/2)、足/ひざ/足首→UpLeg/Leg/Foot、腕/ひじ/手首→Arm/ForeArm/Hand、
> 指→HandXxx1..4；并把「上半身」挂到 Hips 下（MMD 里是兄弟，Mixamo 要求子级）。
> **头发 / 裙子 / 胸 / 捩り等非标准骨一根不删**（`左髪1 → hair1.L`、`スカート_0_1 → skirt01`），
> MMD 控制骨（全ての親 / センター / グルーブ / 肩P / 足ＩＫ / 足D…）默认删除且**权重并进父骨**、
> 子骨改挂最近的保留祖先；同步重命名顶点组（否则蒙皮全丢）；顺带摘掉 MMD Tools 的约束与驱动器。
> 默认「先复制一份再改」。自检：`test_mmd_mixamo.py`（纯逻辑：映射 / 层级 / 删除 / 保留 /
> 幂等）、`test_mmd_mixamo_rig.py`（真实骨架：18 个 Mixamo 骨名、Spine<-Hips、头发挂 Head、
> 裙子挂 Hips、足D 权重并进 LeftUpLeg、约束与驱动器清零、复制模式原模型不动）。
> **1.4.2**：① `骨架标准` 下拉（`Rigify 标准` / `Mixamo 标准`）——**抓到的姿态可以直接写到
> `mixamorig:*` 骨架上**（Hips/Spine/Spine1/Neck/Head/{Side}Arm/{Side}ForeArm/{Side}Hand/
> {Side}UpLeg/{Side}Leg/{Side}Foot/{Side}ToeBase + 15 根手指；`mixamorig:` / `mixamorig1:` /
> 无前缀自动探测，根位移写 Hips）；实测骨骼方向误差 0.00~0.02°（见 **5.3**）。
> ② 新增 **物理（Goo Physics）** 子面板：一键给**头发 / 裙子 / 飘带**加 Goo Physics
> （按骨名分类 → 自动选骨 → 设类型 → 调 `goophys.add_*_physics_to_selected` → 回来逐个骨头核对
> `gp_has_*_physics` 标记；自动打开 Physics Active；可一键给身体网格加碰撞体，也可一键移除）。
> 实测：头发 3 根→软体链、裙子 3 根→布料链、飘带 2 根→软体链，碰撞体生成，移除后标记全清（见 **5.4**）。
> 自检：`test_mixamo_target.py`、`test_goo_physics.py`。





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

### 2.1 依赖 / 模型都下不下来？用离线包

`dist/pose_capture_datafiles.zip`（约 430 MB）已经把 **onnxruntime 依赖 + 三个模型**打包好了，
离线 / 公司网络 / 连不上 huggingface 时**解压到“数据目录的上一级”，也就是 `<版本>\datafiles\`**，
重启 Blender 就能直接检测：

> 这个 zip 是**单独提供**的（`python build_zip.py` 只打包插件本体，不含依赖与模型）；
> 手里没有的话，看本节最后一条“自己做一份”。

```bat
:: Windows（把 5.2 换成你自己的 Blender 版本；仓库根目录下执行）
powershell -NoProfile -Command "Expand-Archive -Force -Path 'dist\pose_capture_datafiles.zip' -DestinationPath \"%APPDATA%\Blender Foundation\Blender\5.2\datafiles\""
```

macOS / Linux 用系统自带的解压工具（或 `unzip`），解压到同样的一层：

```
macOS: 解压到 ~/Library/Application Support/Blender/<版本>/datafiles/
Linux: 解压到 ~/.config/blender/<版本>/datafiles/
```

解压完应该长这样（**`libs/` 与 `models/` 直接挂在 `pose_capture/` 下面**）：

```
<版本>\datafiles\pose_capture\
├─ libs\                                onnxruntime（+ 可选的 cv2 / numpy）
└─ models\
   ├─ yolox_l.onnx                      人体检测（207 MB）
   ├─ dw-ll_ucoco_384.onnx              姿态（128 MB）
   └─ depth_anything_v2_small.onnx      深度（94 MB，fp32 档）
```

然后重启 Blender，在偏好设置 / 面板顶部确认这三行都正常：

```
依赖：onnxruntime x.y.z      模型：已就绪      深度模型：已就绪
```

要点：

* **一定要解压到 `datafiles\`，不要解压进插件目录**（`…\extensions\user_default\pose_capture\`
  或 `…\scripts\addons\pose_capture\`）：那是代码目录，更新插件时会整个被替换掉
* **别多套一层**。如果出现 `datafiles\pose_capture\pose_capture\libs`，把里层 `pose_capture\`
  的内容挪到 `datafiles\pose_capture\`（或者删掉重新解压到 `datafiles\` 这一级）
* 包里的 `libs` 是**给 Blender 自带的 Python 编译的二进制**，这份随发行版附带的包对应
  **Blender 5.x（Python 3.13）**。换成别的大版本（例如 4.2 LTS 是 Python 3.11）时
  `libs/` 要重新装一次（点 **安装依赖**，或按下面“装依赖失败”里的 pip 命令手动装）；
  **`models/` 里的 `*.onnx` 与 Python 版本无关，可以照用**
* 只想补模型：把 `models\*.onnx` 单独放进去就行；也可以完全不解压，用偏好设置里的
  **本地模型目录**（填解压出来的目录）+ **从本地目录导入**，插件会递归查找、不复制文件
* 深度模型只有 fp32 档。想要 fp16 48MB / int8 27MB：在偏好设置里换 `深度模型档位` 后再点
  **下载深度模型**，或用 **导入深度模型** 指定任意已有的 `depth*.onnx`
* 想自己做一份同样的包：在**已经装好依赖和模型**的机器上，把 `…\datafiles\pose_capture\`
  整个压成 zip（保持里面一层 `pose_capture/`），拷到别的机器解压即可：

  ```bat
  :: Windows（在已经装好依赖 + 模型的机器上执行；把 5.2 换成你的版本）
  powershell -NoProfile -Command "Compress-Archive -Force -Path \"$env:APPDATA\Blender Foundation\Blender\5.2\datafiles\pose_capture\" -DestinationPath \"$env:USERPROFILE\Desktop\pose_capture_datafiles.zip\""
  ```

  （Windows 自带 `Compress-Archive` 生成的条目分隔符是 `\`，Windows 上解压没问题；
  要给 macOS / Linux 用建议改用 `7z a -tzip` 或 `zip -r pose_capture_datafiles pose_capture`。）

---

## 3. 使用步骤

在 3D 视图按 `N` 打开侧栏，切到 **Pose Capture** 标签页：

| 步骤 | 操作 |
| --- | --- |
| 1 输入图片 | 选“图片文件”并指定路径；“已加载图像”则从 Blender 里挑一个图像数据块。多人时用**人物序号**（0 = 面积最大的人）挑一个 |
| 2 检测 | 点 **检测姿态**（后台线程跑，不卡界面；按 ESC 可取消）。完成后会生成 `PoseCapture_Preview` 预览图，点旁边的小图标可推到图像编辑器查看骨架；深度模型就绪时会同时算出深度图（`PoseCapture_Depth`，**近=蓝、远=红**，灰度图则**越白=越远**）。旁边的 **深度图预览窗口** 会另开一个独立 Tk 页面，把照片 / 深度灰度 / 深度伪彩并排显示、打出数值统计、并给出**深度方向核对**（物理判据“人比背景近” vs 重建拟合方向，不一致会直接标出来）—— 判断“深度是不是真的算对了”用它最稳（不受 Blender 图像显示链路影响） |
| 3 目标骨架 | 先选 **骨架标准**：`Rigify 标准`（点 **新建元骨架** + **生成控制骨架**，或用自己的 Rigify 骨架）或 `Mixamo 标准`（`mixamorig:Hips` 这类骨架，见 5.3，用吸管按钮绑定当前选中对象） |

| 4 映射 | 点 **自动匹配骨骼**（生成骨架会自动认出 `spine_fk / upper_arm_fk.L / thigh_fk.L / neck / head / torso`，元骨架认出 `spine / spine.001 / upper_arm.L …`）。个别骨骼可以在“骨骼映射”子面板里手动指定 |
| 5 参数 | 按需调姿态解算参数（见下表）。`3D 来源` 默认 **自动**：配置好 SAM 3D Body 就用它（见 5.1），否则用 `深度来源` 的**深度模型**（前后方向来自像素）；想用旧方式就选 `骨长解算` |
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
> 写完插件会用 `DEF-*` 变形骨核对一遍（关节位置 2 cm / 方向 10° 以内），**全部肢体写完后再按
> 最终结果复核**一遍（8 mm / 4°）—— 直写不跑 Rigify 内部的 `correct_rotation`，最后的关节还会被
> IK 的 stretch / 极点再解一次，膝盖上能差到 2 cm；超标的肢体自动改成“切 FK + Rigify 快照”
> （快照的骨骼方向与 FK 一致、关节位置差几毫米），或者按 `直写没成功时` 停在 FK。
> 面板摘要里会写清楚哪些是直写、哪些走了快照 / 停在 FK。
> 想停在 FK 就把 `肢体写入方式` 改成 `FK 骨骼`；这时插件会自动**藏起 IK 控制器、显示 FK 控制器**
> （反向同理），免得看见一组拖不动的控制器以为插件坏了。自己拖过 Rigify 的 IK-FK 滑块之后，
> 点面板里的 **同步控制器显示** 就能让显示跟上滑块。


---

## 4. 参数说明

| 参数 | 说明 |
| --- | --- |
| **骨架标准** | `Rigify 标准`（默认）：按 `spine_fk` / `upper_arm_fk.L` / `torso` 这套名字匹配；`Mixamo 标准`：按 `mixamorig:Hips` / `LeftArm` / `LeftUpLeg` 这套匹配（前缀 `mixamorig` / `mixamorig1:` / 无前缀都能自动认，见 **5.3**） |
| **3D 来源** | `自动`（默认）：SAM 3D Body 配置齐全就用它，否则退回下面的「深度来源」；`SAM 3D Body`：单图直接出 MHR 人体（含手指 / 脚，带人体先验，见 **5.1**，需要外部 python 环境）；`DWPose + 深度模型`；`骨长解算` |

| 人体框 / 关键点置信度阈值 | 检测的过滤阈值。检测不到人就降低人体框阈值；关键点乱飘就提高关键点阈值 |
| **人物朝向** | 正面朝镜头 / 背面朝镜头 / 面朝画面左 / 右。决定“画面深处”与左右手的关系，选错会出现整体镜像或反向（`3D 来源 = SAM 3D Body` 时由 3D 肩线 / 胯线自动判定，这项不参与） |

| **深度来源** | `深度模型`（默认）：用 Depth Anything V2 给每个关节解真实深度，前后方向来自像素；`骨长解算`：旧的 `dz=±√(L²-d²)` + 固定方向假设（兜底）；`压平到画面平面`：不做深度 |
| 深度模型档位 / 透视强度 / 反转深度方向 | 档位在插件偏好设置里选（fp32 95MB 最准 / fp16 48MB / int8 27MB 最快）；透视强度 = 焦距系数 `f = 系数 × 图片长边`（偏平调大、偏夸张调小）；方向反了才勾“反转” |
| 估算骨骼扭转(roll) | 用肩线、胯线、手臂弯曲轴对齐骨骼 roll，避免手臂/头部拧麻花。出问题可关掉或降低“扭转强度” |
| 交换左右 | 检测把人物左右判反时用（极少见） |
| 写入手指 | 用 21 点手部关键点写 3 节手指（`thumb.01-03` / `f_index.01-03` …）。手部检测质量差时建议关闭 |
| 手部沿小臂 | **让手尽量顺着小臂的方向伸展**（默认 0.5）。手部关键点只到指根，握拳 / 侧掌时 `腕 -> 指根` 能偏出 40~90°，照单全收会让手掌横在腕上；这里以腕为轴把整只手（含手指）朝小臂转：`0` 不拉（偏角 > 30° 才夹住）、`0.5` 偏角减半、`1` 完全顺着小臂。手长与手的形状都不变 |
| 写入根骨骼位移 | 把人物的水平位置 + 站立高度写进 `root` 骨骼；配合“落地对齐”把脚放回地面高度 |
| **肢体写入方式** | `IK 控制器`（默认）：把姿态**直写**进 `upper_arm_ik` / `thigh_ik`（链根旋转）、`hand_ik` / `foot_ik`（末端朝向）和 IK 目标（手腕 / 脚踝位置），不切 FK、不做快照；写完用 `DEF-*` 变形骨核对（2 cm / 10°），**全部写完后再按最终结果复核**（8 mm / 4°），对不上的肢体改走 Rigify 的 “IK->FK” 快照（方向一致、关节位置差几毫米）或按下一项停在 FK。`FK 骨骼`：姿态停在 FK 骨骼上并把 `IK_FK` 滑块设为 1。`腿 IK + 臂 FK` / `腿 FK + 臂 IK`：按肢体种类分开 —— 腿 IK 让脚踏点钉住（改躯干时脚不跟着跑），臂 FK 让肘的 roll 直接由弯曲轴算出 |
| **直写没成功时** | `退回 Rigify 快照`（默认）：切 FK -> 调 Rigify 自带的 “IK->FK” 快照 -> 切回 IK，极点 / 拉伸 / 脚跟都由 Rigify 算（依赖生成脚本还在）；`停在 FK 骨骼`：不做快照，这几条肢体停在 FK —— 姿态就是 FK 链那份，控制器换成 FK 那组。两条路在“快照失败 / 直写复核不过”时都保证姿态是**看得见**的（不会再切回 IK 把姿态藏起来） |
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
   * **按肢体种类决定写哪条肢体用什么方式**：`肢体写入方式` 可以是 `IK 控制器`（默认）/ `FK 骨骼`，
     也可以是 `腿 IK + 臂 FK` / `腿 FK + 臂 IK` —— 自动匹配骨骼名时也按种类走（腿匹配 `thigh_ik` /
     `foot_ik`，臂匹配 `*_fk`）。实测两种模式写出来的变形结果只差 **0.2 px** 量级，真正的区别是
     “好不好改”：`foot_ik` 在根空间，转躯干 10° 时脚只动 **0.2 mm**（FK 模式动 **174 mm**）。
   * `IK 控制器`（默认）：直接把姿态写进 IK 控制器 —— 跟 Rigify 的 “IK->FK” 按钮同一套算法（`RigifyLimbIk2FkBase.apply_frame_state`）：把 FK 链每根骨骼的世界增量整体套到对应控制器上（**位置 + 旋转**，`hand_ik` / `foot_ik` 就是 IK 链末端），链根 `upper_arm_ik` / `thigh_ik` 再按 Rigify 的 `correct_rotation` 搜一遍扭转（弯曲平面），开了 `pole_vector` 就按 `match_pole_target` 摆极向目标 `*_ik_target`。IK 模式下 FK 链是隐藏的，但也会写一份（拖 `IK_FK` 滑块回 FK 时姿态不丢，退回快照时也靠它）。写完用 `DEF-*` 变形骨核对（**关节位置** 2 cm / 骨骼方向 10° 以内；不比尾巴 —— IK 链带出来的 DEF 骨会被 stretch 拉长，Rigify 自己那条路径也一样）；**全部肢体写完后再按最终结果复核**（8 mm / 4°）—— 直写不跑 Rigify 内部的 `correct_rotation`，关节最后还会被 IK 的 stretch / 极点再解一次（实测膝盖能差 2 cm）。复核不过的肢体改走下面的快照或停在 FK，所以最坏情况只是“没直写成功”，不会写出错姿态。实测：手臂关节 4~7 mm（直写保留），腿 11 mm 以上（改走精确快照）
   * 老路径（`FK 骨骼` 模式、映射指向 FK 骨骼、或直写 / 复核没通过）：先把写到的肢体切到 FK，写旋转，再调用 Rigify 自带的 `pose.rigify_limb_ik2fk_<rig_id>` 快照把 IK 控制器对齐（骨骼方向与 FK 一致，关节位置差几毫米 —— IK 链自己的 hinge / stretch），最后按 `直写没成功时` 停在 IK（默认）或 FK。**快照失败**（生成脚本被删）时停在 FK
   * 两种路径都按最终的 IK / FK 状态同步控制器可见性（切 FK 藏 IK，反之亦然）

### 5.1 SAM 3D Body 模式（推荐的单图 3D 来源）

上面的 1~12 是「关键点 + 单目深度」那条路。1.4.0 起多了一条**把人当成生成式 3D 人体问题**
的路子：`3D 来源 = SAM 3D Body` 时，3D 关节不由像素深度推，而是由 Meta 的
[SAM 3D Body](https://github.com/facebookresearch/sam-3d-body)（MHR70，单图 → 参数化人体）
直接预测 —— 前后关系、遮挡、侧身、透视缩短都带人体先验，代码里那些「深度方向猜错 / 手臂甩到
背后 / 手部沿小臂硬拉」的补丁就不再需要了。

```
图片 ──► YOLOX + DWPose(133)          ← 仍然跑：2D 预览、人框、跟老路子共用一份缓存
     ──► 人框交给外部 python ──► sam3d_worker.py ──► SAM 3D Body ──► npz
         （70 点 3D / 70 点 2D / focal / cam_t / 网格顶点+面片）
     ──► sam3d.build_recon()  ← MHR70 → 插件统一 recon（joints / refs / spine_points / fingers）
     ──► retarget.apply_pose()  ← 完全复用（FK / IK / 快照 / 关键帧都一样）
```

**为什么走子进程**：SAM 3D Body 要 PyTorch / torchvision（想用它的自带检测器还要
detectron2 / SAM3 / MoGe），这些既装不进也不该装进 Blender 自带的 python。插件只负责
「拼参数 → 起子进程 → 读 npz」，模型加载和显存都在那个独立环境里，Blender 本体不受影响。

**一次性准备**（按官方 `INSTALL.md`）：

```bat
:: 1) 建环境（Python 3.11），装 PyTorch（按 pytorch.org 选 CUDA 版）
conda create -n sam_3d_body python=3.11 -y
conda activate sam_3d_body
pip install pytorch-lightning opencv-python yacs scikit-image einops timm dill pandas rich ^
            hydra-core pyrootutils webdataset networkx==3.2.1 roma joblib seaborn appdirs ^
            loguru optree fvcore pycocotools tensorboard huggingface_hub

:: 2) 拿代码（插件要的是这个仓库的根目录）
git clone https://github.com/facebookresearch/sam-3d-body

:: 3) 拿权重（官方仓库是 gated：需要在 HF 上申请访问权限并登录）
hf download facebook/sam-3d-body-dinov3 --local-dir checkpoints\sam-3d-body-dinov3
::   没有权限时可以用公开镜像（文件字节数与官方完全一致，实测 2109129346 / 696110248 字节）
::   python -c "from huggingface_hub import hf_hub_download as d; [d('Blendcap/sam-3d-body', f, local_dir=r'checkpoints\sam-3d-body-dinov3') for f in ('model_config.yaml','model.ckpt','assets/mhr_model.pt')]"

```

然后在 `编辑 > 偏好设置 > 附加组件 > Pose Capture > SAM 3D Body` 里填：

| 项 | 填什么 |
| --- | --- |
| 外部 python | 上面那个环境的 `python.exe`（conda 的 `envs\sam_3d_body\python.exe`） |
| 代码目录 | `sam-3d-body` 仓库根目录（里面有 `sam_3d_body/`） |
| 模型权重 | `checkpoints\sam-3d-body-dinov3`（里面有 `model.ckpt` 与 `assets/mhr_model.pt`） |
| HuggingFace 仓库 | 没有本地权重时用（默认 `facebook/sam-3d-body-vith`） |
| 设备 | 有 N 卡选 CUDA；CPU 也能跑，就是慢 |

点 **检查环境**（= 跑一次 `sam3d_worker.py --check`），成功会显示 `torch 版本 / CUDA / sam_3d_body 版本`。
面板里 `3D 来源 = 自动` 时，配置齐全就用 SAM 3D Body，否则自动退回深度 / 骨长解算并把原因写进日志。

**它是怎么换算成插件坐标的**（`pose_capture/sam3d.py`，纯 numpy，可脱离 Blender 单测）：

| 步骤 | 做法 |
| --- | --- |
| 相机约定 | 用 `pred_keypoints_2d` 对「Y 上/下 × 是否加 `pred_cam_t`」做**重投影投票**，取误差最小的那一组（`_choose_camera`），结论写进日志（例如「Y 向下(OpenCV)，重投影误差 3.1 px」） |
| 坐标 | 相机系 `(x, y, z)` → 插件系 `(x, z, -y)`（X 右 / Y 朝画面里 / Z 上） |
| 朝向 | 由 3D 肩线 × 躯干轴算出人物正面，转到「角色面朝 -Y」，和 `reconstruction` 用同一个公式（所以「人物朝向」这项在 SAM 模式下不参与） |
| 胸 / 颈 / 头 | 胸 = 肩峰中点（和颈根顺序冲突时取中点）、颈根 = MHR 的 `neck`；头骨起点取耳中点往颈轴 75% 处，头向上轴 =（鼻子 - 耳中点）× 眼线，**先按解剖常数把鼻子方向抬平 10°**（否则平视时 head 骨会常年低头 10°） |
| 参照轴 | `hip_line` / `shoulder_line` / `eye_line`（符号统一到肩线）/ `armbend` = 大臂 × 小臂 / `kneebend` = 大腿 × 小腿 —— 与 `reconstruction._build_refs` 完全一致 |
| 手指 / 脚趾 | MHR70 的「第三关节 → 第二 → 第一 → 指尖」正好是 Rigify 的 `*.01/.02/.03` 链，`fingers` 直接是 3D 的；脚掌用大脚趾尖当 `ball` |

**网格**：点 **导入网格**（`pose_capture.import_sam3d_mesh`）会把 `pred_vertices` 导进场景 ——
和骨骼用同一套变换；缩放按「网格高 → 骨架高」（和姿态无关，蹲着也准；没有网格时退回
大腿+小腿骨长），骨盆水平对齐到目标骨架，**脚底贴骨架地面**。适合对着网格修姿态、或者拿它
做深度 / 法线 ControlNet 的条件图。

**耗时与缓存**（实测：RTX 4060 Laptop 8 GB / cu128 / 640×425 的 COCO 照片）：
子进程每次都要重新加载 2.7 GB 权重，一次 `estimate()` 约 **90 s**，推理本身只占几秒。
所以插件按 `(图片路径, 文件大小, 修改时间, 尺寸, 人物序号)` **缓存结果**：同一张图重检测 /
重应用不会重新跑；换图或换人才会重新加载。想再快就得让 worker 常驻（目前没做）。


**回退**：外部环境没配好、子进程报错、或者这次结果里没有 3D 点，都会**只影响这一次**——
检测照常完成（2D 关键点 / 预览都有），应用姿态时退回深度 / 骨长解算，并在面板与日志里
写清楚原因；不会出现「点了没反应」。

**它的限制**（和深度那条路一样要心里有数）：

* **单张图没有时序**：来回摆动的肢体仍可能被解成某一瞬间的合理姿态（但不会像深度那样把
  手臂整条甩到身体后面）；
* **遮挡 / 相切**：手贴脸、双手交叠时手指方向仍可能抖，`只跑身体解码` 可以跳过手部解码器；
* **人体模型不同**：MHR ≠ SMPL-X，绝对尺寸看模型，插件只取「方向」；骨架比例始终用你自己的；
* **穿模不会自动修**：网格是模型给的，手插进身体这类情况要手动调（或直接当造型用）。

**真机实测**（2026-11-20，RTX 4060 Laptop 8 GB，torch 2.11.0+cu128，COCO `000000000785`）：

| 指标 | 结果 |
| --- | --- |
| 推理 | `process_one_image` 1 人，70 点 3D/2D + 18439 顶点 / 36874 面，GPU |
| 相机投票 | **Y 向下(OpenCV) + 加 `pred_cam_t`**，重投影误差 **0.71 px** |
| 骨长（模型预测） | 肩宽 0.334 / 大臂 0.275 / 小臂 0.251 / 大腿 0.423 / 小腿 0.386 / 躯干 0.560 m |
| 左右对称 | 大臂 0.9% / 小臂 1.1% / 大腿 0.0% / 小腿 0.7% |
| 朝向 | 自动判出 yaw **-59°**（照片里人是半侧身） |
| 写进 Rigify | **53 根骨骼**（含 10 条手指链），骨骼方向与 recon 差 **0.00~0.02°** |
| 网格导入 | 18439 顶点，脚底 0.017 m = 骨架地面 0.017 m（缩放 1.25） |
| 耗时 | `estimate()` **92.5 s**（其中 ~85 s 是加载 2.7 GB 权重） |


---


### 5.2 MMD 骨骼 → Mixamo（保留头发 / 裙子骨）

功能对应 **MMD Bone Tools**（[`QIangIQ/MMD-bone-SDK`](https://github.com/QIangIQ/MMD-bone-SDK)，MIT）
的 `convert --to mixamo`，但搬进了本插件、直接在 Blender 里改骨架：

| 做的事 | 说明 |
| --- | --- |
| 标准骨改名 | 下半身→`mixamorig:Hips`、上半身→`Spine`、上半身2→`Spine1`、首→`Neck`、頭→`Head`、肩→`Shoulder`、腕→`Arm`、ひじ→`ForeArm`、手首→`Hand`、指 1/2/3/先→`HandXxx1..4`、足→`UpLeg`、ひざ→`Leg`、足首→`Foot`、つま先→`Toe_End`、足先EX→`ToeBase` |
| 层级修正 | MMD 里「上半身」和「下半身」是**兄弟**；Mixamo 要求 Spine 是 Hips 的子级 —— 不改这一步，Mixamo 动画转 Hips 时上半身不跟着动 |
| 控制骨 | 全ての親 / センター / グルーブ / 腰キャンセル / 肩P / 肩C / 足ＩＫ / つま先ＩＫ / 足D / ひざD / 足首D / ダミー 默认删除；删之前**子骨改挂最近的保留祖先**、**权重并进父骨** |
| **头发 / 裙子** | **一根都不删**：`左髪1 → hair1.L`、`スカート_0_1 → skirt01`、`胸 → breast`… 名字可选翻成英文；翻不出来的（含生僻汉字）保持原名。父级、层级、顶点组权重原样保留 |
| 蒙皮 | 同步重命名顶点组（不勾选的话蒙皮会全丢）；被删骨的权重并进祖先，没有去处的孤儿组连同权重一起清掉并报警 |
| MMD Tools 残留 | 摘掉骨骼约束 + 驱动器（否则改名后它们还会把骨骼拉回原位），避免 Mixamo 动画被“追加变换”顶掉 |

**用法**：选中 MMD 骨架 → `N` 面板 → **MMD → Mixamo** 子面板 → 点 **转换（保留头发 / 裙子骨）**。
默认勾了 **先复制一份再改**，原模型不动；关掉就直接改当前骨架（可 `Ctrl+Z`）。

| 选项 | 默认 | 说明 |
| --- | --- | --- |
| Mixamo 前缀 | `mixamorig:` | 同场景多角色用 `mixamorig1:`；也可以不要前缀 |
| 删除 MMD 控制骨 | 开 | IK / センター / 全ての親 这些 Mixamo 没有槽位的骨 |
| 头发 / 裙子等骨骼名翻成英文 | 开 | 只是改名，骨骼本身保留 |
| 同步重命名顶点组 | 开 | 关掉 = 蒙皮全丢，面板会警告 |
| 被删骨骼的权重并入父骨 | 开 | 例：足D 的权重 → `LeftUpLeg` |
| 摘掉 MMD Tools 追加变换 | 开 | 骨骼约束 + 驱动器 |
| 先复制一份再改 | 开 | 保留原始 MMD 模型 |

**转换后要上传 Mixamo / 做重定向时注意**：

* 这一步**只改名字和层级**，不动静止姿势（MMD 是 A-pose、Mixamo 是 T-pose）。要上传 Mixamo
  自动绑定时先摆成 T-pose；用 Mixamo 动画驱动 MMD 模型（重定向）时，静止姿势差异由重定向工具处理；
* 骨架 / 物体缩放要是 1（不是 1 就 `Ctrl+A` 应用缩放），面板会提示；
* `腕捩 / 手捩` 这类捩り骨会**保留**成额外骨（`arm_twist.L` …），它们跟着 `Arm` 继承；
  Mixamo 不认这些槽位，但删了会掉体积，所以默认保留；
* 这一步跑完，插件自己的「应用到骨架」仍然认 `mixamorig:LeftArm` 这类名字吗？**不认** ——
  `自动匹配骨骼` 是按 Rigify / 元骨架的名字匹配的。要抓姿态请用 Rigify 骨架，别用转换后的 MMD 骨架。

**真机实测**（仓库里那份「星穹铁道—流萤」PMX，2.2 MB，340 根骨）：

| 项 | 结果 |
| --- | --- |
| 骨架 | `下半身 → mixamorig:Hips`、`上半身 → Spine`、`上半身2 → Spine1`，Spine 已挂到 Hips 下 |
| 映射 | 65 根 → Mixamo（含 15 根手指、`右目/左目 → RightEye/LeftEye`、`右足先EX → ToeBase`）；总改名 292 根，**无重名** |
| **头发** | 认出 **81 根**（`左後髪1-1 → back_hair1_1.L`、`左馬尾1-1 → ponytail1_1.L`、`中劉海1 → mid_bangs1`、`髮帶…`），**一根没删** |
| **裙子** | 认出 **112 根**（`裙_0_0 → skirt_0_0` … 这款模型用的是中文「裙」），**一根没删** |
| 控制骨 | 删 23 根（全ての親 / センター / グルーブ / 足IK親 / 足ＩＫ / つま先ＩＫ / 肩P / 肩C / ダミー / 腰キャンセル） |
| 保留 | 317 根；只剩 19 根名字翻不了（`操作中心`、`腰`、`肩+`、`髪翼飾`…）→ 原样保留 |


---


### 5.3 Mixamo 骨架也能直接套姿态（`骨架标准 = Mixamo 标准`）

抓到的姿态现在可以**直接写到 Mixamo 命名的骨架上**（比如用 5.2 转换出来的模型、或者从 Mixamo
下载的 FBX）：面板 → **2. 目标骨架** → `骨架标准 = Mixamo 标准` → `自动匹配骨骼` → `应用到骨架`。

| 角色 | Mixamo 骨 |
| --- | --- |
| 骨盆 / 根位移 | `mixamorig:Hips`（根位移也写这里） |
| 脊椎 | `Spine` → `Spine1` → `Spine2`（有几根用几根） |
| 颈 / 头 | `Neck` / `Head` |
| 锁骨 / 大臂 / 小臂 / 手 | `{Side}Shoulder` / `{Side}Arm` / `{Side}ForeArm` / `{Side}Hand` |
| 大腿 / 小腿 / 脚 / 前掌 | `{Side}UpLeg` / `{Side}Leg` / `{Side}Foot` / `{Side}ToeBase`（没有就 `Toe_End`） |
| 手指 | `{Side}HandThumb1..3`、`HandIndex1..3`、`HandMiddle1..3`、`HandRing1..3`、`HandPinky1..3` |

* **前缀自动探测**：`mixamorig:` / `mixamorig1:`（同场景多角色）/ 无前缀三种都认；
* 不需要 IK / 快照那套（Mixamo 骨架没有 Rigify 的控制器），写入方式是纯 FK；
* 点「新建元骨架 / 生成控制骨架」时会自动把 `骨架标准` 切回 `Rigify 标准`（那两根按钮造的是 Rigify 骨架）；
* 实测（合成 Mixamo 骨架 + SAM 3D Body 的姿态）：`LeftArm` / `LeftForeArm` / `RightUpLeg` /
  `RightLeg` / `Head` 与解算方向的误差 **0.00~0.02°**。

### 5.4 一键给头发 / 裙子 / 飘带加 Goo Physics

如果你装了 **Goo Physics**（模块名 `goo_physics`），面板里会多一个 **物理（Goo Physics）** 子面板：

```
选中骨架 → 勾「头发 / 裙子 / 飘带」→ 物理类型（自动 / 软体 / 布料 / Geo Nodes / 抖动）→ 一键加物理
```

它会：

1. 按骨名认出三类骨（日文 / 中文 / 英文都认）：`髪 / 髮 / hair / 前髪 / 馬尾 / ponytail / 劉海`、
   `スカート / 裙 / 裾 / skirt`、`リボン / 领结 / 飘带 / 尾巴 / 披风`；
2. 按类型调 Goo Physics 的操作器（**类型/预设走它自己的偏好**，所以和它面板里点是一样的效果）：

   | 我们的选项 | Goo 偏好 `gp_physics_type` | 操作器 |
   | --- | --- | --- |
   | 自动（头发 / 飘带） | `SIMP_SOFTBODY` | `goophys.add_soft_physics_to_selected` |
   | 自动（裙子） | `CLOTH` | `goophys.add_cloth_physics_to_selected` |
   | 软体 / 布料 / Geo Nodes / 抖动 | 对应值 | `add_*_physics_to_selected` |

3. 只选这三类骨（不碰身体骨），并自动打开 Goo 的 **Physics Active**；
4. 可选**给身体网格加碰撞体**（`goophys.add_collider`，用带骨架修改器的网格）；
5. 回来**核对**：逐个骨骼检查 `gp_has_sb_physics / gp_has_cl_physics / gp_has_gn_physics /
   gp_has_jg_physics`，面板会写「头发 3 根（软体链，已挂 3）；裙子 3 根（布料链，已挂 3）；碰撞体已生成」；
6. 还有一个 **移除这些骨上的物理**（`goophys.remove_physics_from_selected`）。

注意事项：
* **Jiggle（抖动）只适合单根骨**，而且要求骨头是「断开父级 / 保持偏移」（Goo 自己的限制），
  头发 / 裙子链请用软体或布料；
* 预设、阻尼、风力、烘焙仍在 **Goo Physics 自己的面板**（`N` 面板 → Goo Animation Tools）里调；
* 骨架必须能进 **Pose 模式**（Goo 的操作器读的是 pose 模式下的选骨），插件会自动切过去；
* 没装 / 没启用 Goo Physics 时面板会直接说清楚（“装了但没启用” / “没有找到”），不会静默失败。

---


## 6. 已知限制（单张图固有问题）

> 下面大多是「关键点 + 单目深度」那条路的限制；`3D 来源 = SAM 3D Body`（见 **5.1**）时
> 深度相关的几条不适用，但「单张图没有时序 / 遮挡时手指抖 / 骨架比例用你自己的」仍然成立。



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

**依赖 / 模型下载不下来**（离线、公司网络、连不上 `huggingface.co` / pypi）：用 **离线包** ——
把 `dist/pose_capture_datafiles.zip` 解压到 `<版本>\datafiles\`（不要解压进插件目录），
重启 Blender 即可，详见第 2 节末尾的 **2.1 依赖 / 模型都下不下来？用离线包**。
只想手动补模型的话，模型直链在第 2 节里；依赖也可以用下面那条 pip 命令自己装。

**装依赖失败**：手动在命令行执行（Windows 示例，路径按自己安装位置改）：

```bat
"D:\SteamLibrary\steamapps\common\Blender\5.2\python\bin\python.exe" -m pip install ^
  --target "%APPDATA%\Blender Foundation\Blender\5.2\datafiles\pose_capture\libs" onnxruntime
```

装完重启 Blender。（`libs` 必须与 Blender 自带 Python 的版本匹配：5.x = Python 3.13、4.2 LTS = 3.11；
用离线包里的 `libs` 换大版本时要重装。）

**没有检测到人体**：把“人体框置信度阈值”降到 0.1~0.2，或换一张主体更大的图。

**姿态很扁平 / 很夸张**：调 `透视强度`（偏平→调大；偏夸张→调小），或调“比例微调”。

**手肘/膝盖的朝向和照片不符**：先点 **深度图预览窗口** 看深度图是否合理（伪彩近=蓝、远=红，人物应该比背景蓝；灰度图越白越远），再核对窗口里那行**深度方向**：如果写着物理与拟合不一致（✗），说明整套姿态前后反了，勾 **反转方向** 重算一次。
深度图正常但个别肢体仍反了，说明那处遮挡严重：可以把 `深度来源` 改成 `骨长解算` 用旧方式，或勾 `反转深度方向` 试另一个方向。

**提示“深度解算骨长误差偏大，已退回骨长解算”**：深度图与骨架比例对不上（人物太小、背景杂乱、或档位太低），
插件已自动改用骨长解算保证不出错；想用深度就把人物拍大一点或换 fp32 档位。

**手臂、头拧成麻花**：先确认 `估算骨骼扭转` 是开着的（关掉会退回“极小旋转”，手臂 roll 完全随机）；仍然拧就降低“扭转强度”。若整条小臂像被翻折 180°，多半是上面那条深度歧义，按前一节翻转对应手臂。

**手掌横着长在腕上 / 手的方向和照片差很多**：调 `手部沿小臂` —— 它是“把手朝小臂方向拉多少”，默认 0.5（偏角减半），`1` 完全顺着小臂伸展，`0` 完全不拉（只把超过 30° 的偏角夹住）。手部关键点只到指根，握拳、侧掌、指尖错到掌根另一侧时偏角能到 40~90°，照单全收就会横在腕上；调大更“顺着手臂”，调小更“照手部关键点”。手长与手的形状（握拳 / 张开）不受影响。

**想让脚踏住 / 手扶住，又想要肘的 roll 准**：`肢体写入方式` 选 **`腿 IK + 臂 FK`**。
腿写 IK：`foot_ik` 在根空间，应用完再调躯干 / 骨盆时脚不会跟着跑（实测转躯干 10°，脚只动 0.2 mm；
`FK 骨骼` 模式会动 174 mm）；臂写 FK：肘的 roll 由弯曲轴直接算出，比 IK 的极点搜索更直接。
手要扶 / 抓东西（末端落点更值钱）时反过来选 `腿 FK + 臂 IK`。两类肢体的模式会一起写进日志
（“已把 … 切到 FK” / “已直写 IK 控制器（…）”）。

**日志说“这些肢体直写后的实际偏差偏大，已改走精确的 IK 快照”**：这条肢体用 IK 直写时，最后被
IK 的 stretch / 极点带偏了几毫米以上（膝盖最明显）—— 插件自动改用 Rigify 自己的 `IK->FK` 快照，
姿态一样对，只是慢一点。不想做快照就把 `直写没成功时` 改成 **停在 FK 骨骼**（那几条肢体停在 FK）。

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
:: 1d) SAM 3D Body 适配层自检（MHR70 -> recon / 相机投票 / 朝向归一 / 网格 / build_targets）—— 不需要 Blender
python dev_tests/test_sam3d.py
:: 1e) SAM 3D Body 路径的 Blender 自检（合成 MHR70 写进 Rigify 骨架 + 网格导入对齐）
blender.exe -b --factory-startup --python "dev_tests/test_sam3d_rig.py"
:: 1f) 真实结果数值检查（纯 numpy；参数是 worker 产出的 npz）
python dev_tests/test_sam3d_inspect.py "D:\\code\\sam3d_out\\sam3d_0.npz"
:: 1g) 真实结果端到端（写 Rigify + 导网格 + 渲染 output/sam3d_real/*.png）
blender.exe -b --factory-startup --python "dev_tests/test_sam3d_real.py" -- "D:\\code\\sam3d_out\\sam3d_0.npz"
:: 1h) MMD → Mixamo 纯逻辑自检（映射 / 层级 / 删除 / 头发裙子保留 / 幂等）—— 不需要 Blender
python dev_tests/test_mmd_mixamo.py
:: 1i) MMD → Mixamo 真骨架自检（合成 MMD 骨架 + 网格顶点组，跑操作器）
blender.exe -b --factory-startup --python "dev_tests/test_mmd_mixamo_rig.py"
:: 1j) MMD → Mixamo 真实 PMX 自检（直接读模型骨骼表；把 <model.pmx> 换成你的模型）
python dev_tests/test_mmd_mixamo_pmx.py "<model.pmx>"
:: 1k) Mixamo 标准骨架自检（映射 / 前缀 / 量尺寸 / 写姿态）
blender.exe -b --factory-startup --python "dev_tests/test_mixamo_target.py"
:: 1l) Goo Physics 一键加物理自检（需要装了 goo_physics 插件；没装会跳过）
blender.exe -b --factory-startup --python "dev_tests/test_goo_physics.py"





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

预期：`test_pipeline` 比例误差 <1%、四肢方向平均 ~3°/最大 <10°；`test_rigify_rig` 映射全 OK、画面内偏差 <25 像素；`test_rigify_limbs` 四条肢体链解析 OK、**直写的肢体**与 FK 的 DEF 关节最大位置差 ≤2 cm（实测手臂 4.4~6.5 mm；腿 11 mm 以上会被最终复核改成快照）、**退回快照 / 停在 FK 的肢体** ≤8 mm（实测腿 5.8 mm / 2.4 mm —— 旧文档里的 `0.000000 m` 是依赖图没推完读到的旧值）、**`腿 IK + 臂 FK` 混合模式**与 FK 结果 0.0000 m、控制器拖动有效、显示同步 OK；`test_limb_logic` 全部通过（不需要 Blender，含按种类拆分的匹配 / 直写计划 / 关键帧）；`test_depth` 三组合成深度图（线性 / 视差 / 前后取反）关节误差平均 ~4 mm、最大 ~10 mm，**大臂朝镜头伸**时手仍在身体前面且大臂长度回到骨长，**兜底**时也用深度图的符号（没深度图才退回老假设），**手部关键点**被用上、手指能生成；`test_depth_view` 写入的图不是全黑、PNG 回读正常、预览窗口能建起来；`test_retarget_roll` 手臂姿态矩阵误差 <15°（roll 对齐）；`test_onnx_end2end` 人体结构检查全 OK、写入一致性 0.00°、停在 IK 模式且控制器可拖；`test_installed` 显示“与仓库源码完全一致 ✓”且版本号与 `bl_info` 一致；“通过 ✓”。`test_sam3d` 六组全 OK（相机投票选中 OpenCV / 加 cam_t、骨盆换算、肩线 +X、头朝上、朝向归一后恒为 -Y、网格与关节同一套变换、`build_targets` 认得出 52 个目标）；`test_sam3d_rig` 显示「通过 (all OK)」（关键角色都写入、5 根骨骼方向差 0.0°、左大臂相对静止转 35.7°、网格最低 0.017 m = 骨架地面 0.017 m、骨盆对齐 0.124 m）。

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
│  ├─ sam3d.py              SAM 3D Body 适配层（MHR70 → 插件 recon；纯 numpy，可单测）
│  ├─ sam3d_worker.py       外部 python 里跑的推理脚本（torch 只在那边加载）
│  ├─ mmd_mixamo.py         MMD 骨骼 → Mixamo 改名 + 层级整理（头发 / 裙子骨保留；纯逻辑可单测）
│  ├─ goo_physics.py        Goo Physics 桥接：一键给头发 / 裙子 / 飘带加物理（分类 + 选骨 + 核对）
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
│  ├─ test_sam3d.py         SAM 3D Body 适配层自检（相机投票 / MHR70→recon / 朝向 / 网格；普通 Python 可跑）
│  ├─ test_sam3d_rig.py     SAM 3D Body → Rigify 自检（合成 MHR70 写姿态 + 网格导入对齐）
│  ├─ test_mmd_mixamo.py    MMD → Mixamo 纯逻辑自检（映射 / 层级 / 保留头发裙子；普通 Python 可跑）
│  ├─ test_mmd_mixamo_rig.py MMD → Mixamo 真骨架自检（合成 MMD 骨架 + 顶点组权重 + 复制模式）
│  ├─ test_mmd_mixamo_pmx.py MMD → Mixamo 真实 PMX 自检（读骨骼表，不需要 Blender / MMD Tools）
│  ├─ test_mixamo_target.py Mixamo 标准骨架自检（映射 / 前缀 / read_metrics / apply_pose）
│  ├─ test_goo_physics.py   Goo Physics 一键加物理自检（分类 / 选骨 / 挂物理 / 碰撞体 / 移除）
│  ├─ test_rigify_limbs.py  Rigify 肢体 IK 控制器自检（直写 + 快照退回 / 显示同步）
│  ├─ _diag_ik_frames.py    Rigify 结构诊断（IK 控制器 / 极向目标 / 约束 vs FK 链；排错用）
│  ├─ test_depth_view.py    深度预览自检（黑图回归 / PNG / 预览页面数据）
│  ├─ test_installed.py     已装副本 vs 仓库源码：逐文件 hash 比对 + 能否注册
│  └─ install_local.py      一条命令替换本机已装版本（打包+校验+备份+替换）
├─ dist/                    打包产物 zip；dist/backup/ 放旧版本备份
│  └─ pose_capture_datafiles.zip  离线包：libs（依赖）+ models（三个 onnx），解压到 <版本>\datafiles\
├─ blender_manifest.toml    扩展清单模板
├─ build_zip.py             打包脚本
└─ README.md
```

---

## 11. 许可

* 插件代码：MIT
* 模型：`dw-ll_ucoco_384.onnx` 来自 [IDEA-Research/DWPose](https://github.com/IDEA-Research/DWPose)（Apache-2.0），`yolox_l.onnx` 来自 [Megvii YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)（Apache-2.0），`depth_anything_v2_small.onnx` 来自 [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2)（Apache-2.0，ONNX 导出：[onnx-community](https://huggingface.co/onnx-community/depth-anything-v2-small)）。插件**只提供下载/导入功能，不随插件分发模型文件**。
