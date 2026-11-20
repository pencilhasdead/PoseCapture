"""SAM 3D Body（MHR70）适配层：外部 python 子进程 + MHR → 插件统一 recon。

为什么走子进程
--------------
SAM 3D Body 要 PyTorch / torchvision（想用自带人体检测器还要 detectron2 / SAM3 / MoGe），
Blender 自带的 python 装不下、也不该装。这里只把「图片 + 人框」交给一个外部解释器
（``sam3d_worker.py``，按官方 INSTALL.md 建的 conda 环境），拿回一个 npz：
70 点 3D 关键点 / 2D 关键点 / 相机 / 网格，然后用纯 numpy 换算成和
``reconstruction.reconstruct()`` **完全一样**的 recon 结构 —— 重定向那一层不用改。

本模块刻意不 import bpy / mathutils（只用 numpy + 标准库），所以能脱离 Blender
用普通 python 跑自检：``dev_tests/test_sam3d.py``。

坐标系
------
SAM 3D Body 的 ``pred_keypoints_3d`` 是相机坐标系（米）。OpenCV 约定是
X 右 / Y 下 / Z 前，插件内部用的是 X 右 / **Y 朝画面里** / Z 上，所以
``(x, y, z)_cam -> (x, z, -y)``。不同版本 / 不同导出脚本的 Y 轴方向不一定一致，
``_choose_camera`` 会用 ``pred_keypoints_2d`` 做一次重投影投票（Y 上/下 × 是否加
``pred_cam_t``），选误差最小的那个，并把结论写进 notes —— 不靠猜。
"""
from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import tempfile

import numpy as np

# --------------------------------------------------------------------------- MHR70
# sam_3d_body/metadata/mhr70.py 的 mhr_names（前 70 个 MHR 关键点，顺序固定）
MHR_NAMES = (
    "nose", "left-eye", "right-eye", "left-ear", "right-ear",
    "left-shoulder", "right-shoulder", "left-elbow", "right-elbow",
    "left-hip", "right-hip", "left-knee", "right-knee",
    "left-ankle", "right-ankle",
    "left-big-toe-tip", "left-small-toe-tip", "left-heel",
    "right-big-toe-tip", "right-small-toe-tip", "right-heel",
    "right-thumb-tip", "right-thumb-first-joint", "right-thumb-second-joint",
    "right-thumb-third-joint",
    "right-index-tip", "right-index-first-joint", "right-index-second-joint",
    "right-index-third-joint",
    "right-middle-tip", "right-middle-first-joint", "right-middle-second-joint",
    "right-middle-third-joint",
    "right-ring-tip", "right-ring-first-joint", "right-ring-second-joint",
    "right-ring-third-joint",
    "right-pinky-tip", "right-pinky-first-joint", "right-pinky-second-joint",
    "right-pinky-third-joint",
    "right-wrist",
    "left-thumb-tip", "left-thumb-first-joint", "left-thumb-second-joint",
    "left-thumb-third-joint",
    "left-index-tip", "left-index-first-joint", "left-index-second-joint",
    "left-index-third-joint",
    "left-middle-tip", "left-middle-first-joint", "left-middle-second-joint",
    "left-middle-third-joint",
    "left-ring-tip", "left-ring-first-joint", "left-ring-second-joint",
    "left-ring-third-joint",
    "left-pinky-tip", "left-pinky-first-joint", "left-pinky-second-joint",
    "left-pinky-third-joint",
    "left-wrist",
    "left-olecranon", "right-olecranon",
    "left-cubital-fossa", "right-cubital-fossa",
    "left-acromion", "right-acromion",
    "neck",
)

JOINT = {name: index for index, name in enumerate(MHR_NAMES)}
JOINT_COUNT = len(MHR_NAMES)

# 解剖常数：「耳道口 -> 鼻尖」相对水平低约 10°（算头向上轴前先抬平，见 _head_up）
FACE_NOSE_TILT_DEG = 10.0

# 每根手指：MHR 的第三关节 -> 第二 -> 第一 -> 指尖（第三关节靠手腕，顺序由近到远）
FINGER_CHAIN = ("thumb", "index", "middle", "ring", "pinky")

WORKER_NAME = "sam3d_worker.py"
DEFAULT_HF = "facebook/sam-3d-body-vith"
DEFAULT_TIMEOUT = 900


class Sam3DError(RuntimeError):
    pass


# --------------------------------------------------------------------------- 配置
def _expand(path: str) -> str:
    if not path:
        return ""
    return os.path.expanduser(os.path.expandvars(str(path).strip().strip('"')))


def worker_path() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), WORKER_NAME)


def config_from_prefs(prefs) -> dict:
    """把偏好设置读成一个纯 dict（本模块不认识 bpy）。"""
    def get(name, default=""):
        return getattr(prefs, name, default)

    return {
        "python": _expand(get("sam3d_python", "")),
        "repo": _expand(get("sam3d_repo", "")),
        "ckpt": _expand(get("sam3d_ckpt", "")),
        "mhr": _expand(get("sam3d_mhr", "")),
        "hf": str(get("sam3d_hf", DEFAULT_HF) or "").strip(),
        "device": str(get("sam3d_device", "auto") or "auto"),
        "timeout": int(get("sam3d_timeout", DEFAULT_TIMEOUT) or DEFAULT_TIMEOUT),
        "body_only": bool(get("sam3d_body_only", False)),
    }


def find_python(cfg: dict) -> str | None:
    """外部解释器：优先偏好里填的，其次 PATH 上的 python / python3。"""
    given = _expand(cfg.get("python") or "")
    if given and os.path.isfile(given):
        return given
    for name in ("python", "python3"):
        found = shutil.which(name)
        if found:
            return found
    return None


def checkpoint_paths(cfg: dict) -> tuple[str, str]:
    """返回 (checkpoint, mhr 资产)。ckpt 可以是官方下载的目录，也可以直接是 .ckpt。"""
    ckpt = _expand(cfg.get("ckpt") or "")
    mhr = _expand(cfg.get("mhr") or "")
    if ckpt and os.path.isdir(ckpt):
        folder = ckpt
    elif ckpt and os.path.isfile(ckpt):
        folder = os.path.dirname(ckpt)
    else:
        folder = ""

    if ckpt and os.path.isfile(ckpt):
        path = ckpt
    elif folder:
        candidate = os.path.join(folder, "model.ckpt")
        path = candidate if os.path.isfile(candidate) else ""
    else:
        path = ""

    if not mhr and folder:
        for candidate in (os.path.join(folder, "assets", "mhr_model.pt"),
                          os.path.join(folder, "mhr_model.pt")):
            if os.path.isfile(candidate):
                mhr = candidate
                break
    return path, mhr


def status(cfg: dict) -> dict:
    """只看路径、不启动子进程（面板每次重绘都会调，必须便宜）。"""
    python = find_python(cfg)
    repo = _expand(cfg.get("repo") or "")
    ckpt, mhr = checkpoint_paths(cfg)
    hf = str(cfg.get("hf") or "").strip()
    problems = []
    if not python:
        problems.append("没找到外部 python（要装 PyTorch + sam-3d-body 的环境）")
    if not repo:
        problems.append("没填 sam-3d-body 代码目录")
    elif not os.path.isdir(repo):
        problems.append("代码目录不存在：%s" % repo)
    if not ckpt and not hf:
        problems.append("没填模型权重（本地 ckpt 目录，或 HuggingFace 仓库名）")
    if ckpt and not os.path.isfile(ckpt):
        problems.append("找不到 model.ckpt：%s" % (ckpt or ""))
    return {
        "available": not problems,
        "python": python or "",
        "repo": repo,
        "ckpt": ckpt,
        "mhr": mhr,
        "hf": hf,
        "problems": problems,
        "message": ("已配置（点「检查环境」确认那个 python 能导入）" if not problems
                    else "；".join(problems)),
    }


def _parse_json_line(text: str, prefix: str) -> dict:
    for line in reversed((text or "").splitlines()):
        if line.startswith(prefix):
            try:
                return json.loads(line[len(prefix):])
            except ValueError as exc:
                raise Sam3DError("解析子进程返回失败：%s" % exc) from exc
    return {}


def _run(args: list[str], timeout: int = 300, cwd: str | None = None) -> dict:
    """跑一次 worker，返回它打出的 JSON（没有就返回 {'ok': False, ...}）。"""
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env.setdefault("PYTHONUNBUFFERED", "1")
    try:
        completed = subprocess.run(args, capture_output=True, text=True, timeout=timeout,
                                   cwd=cwd, env=env, encoding="utf-8", errors="replace")
    except subprocess.TimeoutExpired as exc:
        raise Sam3DError("SAM 3D Body 超时（>%d 秒）" % timeout) from exc
    except OSError as exc:
        raise Sam3DError("无法启动外部 python：%s" % exc) from exc

    payload = _parse_json_line(completed.stdout, "SAM3D_JSON ")
    if completed.returncode != 0 or not payload.get("ok"):
        detail = (payload.get("error") or "").strip()
        tail = "\n".join([line for line in (completed.stderr or "").splitlines() if line.strip()][-8:])
        message = detail or tail or (completed.stdout or "")[-400:] or "未知错误"
        raise Sam3DError("子进程失败（返回码 %d）：%s" % (completed.returncode, message))
    return payload


def probe(cfg: dict) -> dict:
    """跑一次 ``sam3d_worker.py --check``：验证那个解释器真的能 import 模型。"""
    python = find_python(cfg)
    if not python:
        raise Sam3DError("没找到外部 python：请在插件偏好设置里指定装好 SAM 3D Body 的 python.exe")
    repo = _expand(cfg.get("repo") or "")
    payload = _run([python, worker_path(), "--check", "--repo", repo], timeout=300, cwd=repo or None)
    payload["python"] = python
    return payload


# --------------------------------------------------------------------------- 推理
def load_npz(path: str) -> dict:
    """读 worker 写出的 npz（全部是普通数组，不用 pickle）。"""
    data: dict = {}
    with np.load(path, allow_pickle=False) as archive:
        for key in archive.files:
            data[key] = archive[key]
    return data


def estimate(cfg: dict, image, out_dir: str | None = None, bbox=None,
             person_index: int = 0, save_vertices: bool = True) -> dict:
    """对一张图跑 SAM 3D Body，返回 worker 的 npz 内容（已经 load 成 dict）。

    ``image`` 可以是路径，也可以是 (H, W, 3) uint8 的 RGB 数组（Blender 图像数据块
    走这条：存成 .npy 交给子进程，省掉一次 PNG 编解码）。
    """
    python = find_python(cfg)
    if not python:
        raise Sam3DError("没找到外部 python：请在插件偏好设置里指定装好 SAM 3D Body 的 python.exe")
    repo = _expand(cfg.get("repo") or "")
    if repo and not os.path.isdir(repo):
        raise Sam3DError("sam-3d-body 代码目录不存在：%s" % repo)

    out_dir = out_dir or tempfile.mkdtemp(prefix="pose_capture_sam3d_")
    os.makedirs(out_dir, exist_ok=True)
    npz_path = os.path.join(out_dir, "sam3d_%d.npz" % int(person_index))

    args = [python, worker_path(), "--repo", repo, "--out", npz_path,
            "--person", str(int(person_index)), "--device", str(cfg.get("device") or "auto")]
    ckpt, mhr = checkpoint_paths(cfg)
    if ckpt:
        args += ["--ckpt", ckpt]
    if mhr:
        args += ["--mhr", mhr]
    if str(cfg.get("hf") or "").strip():
        args += ["--hf", str(cfg["hf"]).strip()]
    if bbox is not None:
        values = [float(v) for v in np.asarray(bbox, dtype=float).reshape(-1)[:4]]
        if len(values) == 4:
            args += ["--bbox"] + ["%.3f" % value for value in values]
    if not save_vertices:
        args.append("--no-vertices")
    if cfg.get("body_only"):
        args.append("--body-only")

    array = np.asarray(image)
    if array.ndim == 3:
        npy_path = os.path.join(out_dir, "image_%d.npy" % int(person_index))
        np.save(npy_path, np.ascontiguousarray(array[:, :, :3], dtype=np.uint8))
        args += ["--image-npy", npy_path]
    else:
        args += ["--image", os.path.abspath(str(image))]

    payload = _run(args, timeout=int(cfg.get("timeout") or DEFAULT_TIMEOUT), cwd=repo or None)
    data = load_npz(payload.get("out") or npz_path)
    data["meta"] = payload
    return data


# --------------------------------------------------------------------------- 相机 / 坐标
def _safe_unit(vec):
    vec = np.asarray(vec, dtype=np.float64)
    norm = float(np.linalg.norm(vec))
    if not math.isfinite(norm) or norm < 1e-9:
        return None
    return vec / norm


def _to_plugin(points: np.ndarray, y_down: bool) -> np.ndarray:
    """相机系 -> 插件系：X 右 / Y 朝画面里 / Z 上。"""
    points = np.asarray(points, dtype=np.float64)
    out = np.empty_like(points)
    out[..., 0] = points[..., 0]
    out[..., 1] = points[..., 2]
    out[..., 2] = -points[..., 1] if y_down else points[..., 1]
    return out


def _project(points: np.ndarray, focal: float, size, y_down: bool):
    height, width = size
    cx, cy = (width - 1) * 0.5, (height - 1) * 0.5
    z = points[:, 2]
    safe = np.abs(z) > 1e-6
    u = np.full(len(points), np.nan)
    v = np.full(len(points), np.nan)
    y = points[:, 1] if y_down else -points[:, 1]
    u[safe] = points[safe, 0] * focal / z[safe] + cx
    v[safe] = y[safe] * focal / z[safe] + cy
    return u, v


def _reproj_error(points, keypoints_2d, focal: float, size, y_down: bool) -> float:
    if keypoints_2d is None or focal <= 0 or not size or min(size) < 4:
        return float("inf")
    u, v = _project(points, focal, size, y_down)
    valid = (np.isfinite(u) & np.isfinite(v)
             & np.all(np.isfinite(keypoints_2d), axis=1)
             & (np.abs(keypoints_2d).sum(axis=1) > 1e-6))
    if int(valid.sum()) < 4:
        return float("inf")
    error = np.hypot(u[valid] - keypoints_2d[valid, 0], v[valid] - keypoints_2d[valid, 1])
    return float(np.median(error))


def _choose_camera(k3: np.ndarray, keypoints_2d, focal: float, size, cam_t: np.ndarray):
    """用 2D 关键点对「Y 上/下 × 是否加 cam_t」投一次票，返回 (y_down, use_offset, 误差)。"""
    options = []
    for use_offset in (False, True):
        if use_offset and not np.any(cam_t):
            continue
        points = k3 + cam_t if use_offset else k3
        for y_down in (True, False):
            options.append((_reproj_error(points, keypoints_2d, focal, size, y_down),
                            y_down, use_offset))
    if not options:
        return True, False, float("inf")
    options.sort(key=lambda item: item[0])
    error, y_down, use_offset = options[0]
    if not math.isfinite(error):
        return True, False, float("inf")
    return y_down, use_offset, error


def _swap_permutation() -> np.ndarray:
    """左右互换的索引表（swap_lr：模型把人物的左右判反时用）。"""
    order = np.arange(JOINT_COUNT)
    for index, name in enumerate(MHR_NAMES):
        if name.startswith("left-"):
            other = "right-" + name[5:]
        elif name.startswith("right-"):
            other = "left-" + name[6:]
        else:
            continue
        order[index] = JOINT[other]
    return order


def _prepare(data: dict, swap: bool = False):
    """npz -> 插件坐标系的 70 个点 + 坐标变换信息 + notes。"""
    raw = data.get("keypoints_3d")
    if raw is None:
        raise Sam3DError("SAM 3D Body 没有返回 3D 关键点（pred_keypoints_3d 缺失）")
    k3 = np.asarray(raw, dtype=np.float64)
    while k3.ndim > 2:
        k3 = k3[0]
    if k3.shape[0] != JOINT_COUNT and k3.shape[1] == JOINT_COUNT:
        k3 = k3.T
    if k3.ndim != 2 or k3.shape[0] < JOINT_COUNT:
        raise Sam3DError("3D 关键点形状不对：%s（期望 (%d, 3)）" % (k3.shape, JOINT_COUNT))
    k3 = k3[:JOINT_COUNT, :3]

    cam_t = np.asarray(data.get("cam_t", np.zeros(3)), dtype=np.float64).reshape(-1)[:3]
    if cam_t.size < 3:
        cam_t = np.zeros(3)

    k2 = data.get("keypoints_2d")
    if k2 is not None:
        k2 = np.asarray(k2, dtype=np.float64)
        while k2.ndim > 2:
            k2 = k2[0]
        if k2.ndim == 2 and k2.shape[0] != JOINT_COUNT and k2.shape[1] == JOINT_COUNT:
            k2 = k2.T
        if k2.ndim != 2 or k2.shape[0] < JOINT_COUNT:
            k2 = None
        else:
            k2 = k2[:JOINT_COUNT, :2]

    size = data.get("image_size")
    try:
        height, width = int(size[0]), int(size[1])
        size = (height, width)
    except Exception:  # noqa: BLE001
        size = (0, 0)

    focal = float(np.asarray(data.get("focal_length", 0.0)).reshape(-1)[0])
    if k2 is not None and min(size) >= 4 and float(np.nanmax(np.abs(k2))) <= 2.0:
        k2 = k2 * np.array((width, height), dtype=np.float64)      # 归一化坐标 -> 像素

    y_down, use_offset, error = _choose_camera(k3, k2, focal, size, cam_t)
    points = k3 + cam_t if use_offset else k3
    if swap:
        points = points[_swap_permutation()]
    plugin = _to_plugin(points, y_down)

    transform = {"y_down": bool(y_down), "use_offset": bool(use_offset),
                 "offset": _to_plugin(cam_t, y_down) if use_offset else np.zeros(3),
                 "focal": focal, "image_size": list(size), "reproj_error": error}
    notes = []
    if math.isfinite(error):
        notes.append("相机：%s，重投影误差 %.1f px%s"
                     % ("Y 向下(OpenCV)" if y_down else "Y 向上", error,
                        "，已加 pred_cam_t" if use_offset else ""))
        if error > 12.0:
            notes.append("重投影误差偏大（%.1f px）：模型的相机/坐标可能和预期不同，"
                         "姿态方向仍按 3D 点算，但建议核对" % error)
    else:
        notes.append("相机：没有 2D 关键点可投票，按 OpenCV（Y 向下）约定")
    return plugin, transform, notes


# --------------------------------------------------------------------------- 角色 / 参照轴
def _point(points, name: str) -> np.ndarray:
    return np.asarray(points[JOINT[name]], dtype=np.float64)


def _mid(points, *names) -> np.ndarray:
    return np.mean([_point(points, name) for name in names], axis=0)


def _facing_yaw(points) -> tuple[float, str]:
    """人物在插件系里朝哪 -> 「转到角色面朝 -Y」的 yaw（弧度），与 reconstruction 同一套公式。"""
    left_dir = _safe_unit(_point(points, "left-shoulder") - _point(points, "right-shoulder"))
    note = ""
    if left_dir is None:
        left_dir = _safe_unit(_point(points, "left-hip") - _point(points, "right-hip"))
        note = "肩线退化，改用胯线判朝向"
    if left_dir is None:
        return 0.0, "肩线 / 胯线都退化，朝向按 0°"

    up = _safe_unit(_point(points, "neck") - _mid(points, "left-hip", "right-hip"))
    if up is None:
        up = np.array((0.0, 0.0, 1.0), dtype=np.float64)
    front = _safe_unit(np.cross(left_dir, up))
    if front is None:
        return 0.0, (note + "；" if note else "") + "身体轴退化，朝向按 0°"
    return math.atan2(float(front[0]), -float(front[1])), note


def _rotate_points(points, yaw: float, pivot) -> np.ndarray:
    """绕 ``pivot`` 的 Z 轴转 -yaw（与 reconstruction 的朝向归一完全一致）。"""
    cos_y, sin_y = math.cos(-yaw), math.sin(-yaw)
    out = np.array(points, dtype=np.float64, copy=True)
    pivot = np.asarray(pivot, dtype=np.float64).reshape(3)
    delta = out - pivot
    out[..., 0] = delta[..., 0] * cos_y - delta[..., 1] * sin_y + pivot[0]
    out[..., 1] = delta[..., 0] * sin_y + delta[..., 1] * cos_y + pivot[1]
    return out


def _rotate_about(vector, axis, angle: float):
    """Rodrigues：把 vector 绕单位轴 axis 转 angle 弧度。"""
    vector = np.asarray(vector, dtype=np.float64)
    axis = np.asarray(axis, dtype=np.float64)
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    return (vector * cos_a + np.cross(axis, vector) * sin_a
            + axis * float(np.dot(axis, vector)) * (1.0 - cos_a))


def _head_up(ear_mid, nose, eye_side, torso_up):
    """头向上轴 =（耳中点 -> 鼻尖）× 眼线，先按解剖常数把鼻子方向抬平。

    成人「耳道口 -> 鼻尖」相对水平本来就低约 10°，不修正的话平视时 head 骨会
    常年低头 10°（两只眼睛看不出来，但和脖子 / 下巴的关系会明显偏）。
    """
    forward = np.asarray(nose, dtype=np.float64) - np.asarray(ear_mid, dtype=np.float64)
    if torso_up is not None and float(np.linalg.norm(forward)) > 1e-9:
        axis = _safe_unit(np.cross(forward, torso_up))
        if axis is not None:
            forward = _rotate_about(forward, axis, math.radians(FACE_NOSE_TILT_DEG))
    return _safe_unit(np.cross(forward, eye_side))


def _roles(points, notes):
    """MHR70 -> 插件角色点（pelvis/chest/head_base…）+ 手指链。"""
    joints = {}
    pelvis = _mid(points, "left-hip", "right-hip")
    joints["pelvis"] = pelvis

    shoulder_mid = _mid(points, "left-shoulder", "right-shoulder")
    acromion_mid = _mid(points, "left-acromion", "right-acromion")
    neck = _point(points, "neck")
    chest = acromion_mid if np.all(np.isfinite(acromion_mid)) else shoulder_mid
    # 胸点如果低到让「胯 -> 胸 -> 颈」打折，就取中间（脊椎是一条直线，不许回折）
    if float(np.dot(chest - pelvis, neck - chest)) < 0.0:
        chest = 0.5 * (chest + neck)
        notes.append("肩峰 / 颈根顺序异常，胸点取两者中点")
    joints["chest"] = chest
    joints["neck_start"] = neck

    # 头：耳道高度当头骨起点，头向上轴 =（鼻子 - 耳中点）× 眼线
    ear_mid = _mid(points, "left-ear", "right-ear")
    eye_side = _point(points, "left-eye") - _point(points, "right-eye")
    head_base = neck + 0.75 * (ear_mid - neck)
    torso_up = _safe_unit(neck - pelvis)
    up_axis = _head_up(ear_mid, _point(points, "nose"), eye_side, torso_up)
    if up_axis is None:
        up_axis = torso_up
        notes.append("头向上轴退化（五官点共线），改用躯干轴")

    if up_axis is None:
        up_axis = np.array((0.0, 0.0, 1.0), dtype=np.float64)
    if torso_up is not None and float(np.dot(up_axis, torso_up)) < 0.0:
        up_axis = -up_axis
    ear_span = float(np.linalg.norm(_point(points, "left-ear") - _point(points, "right-ear")))
    head_len = max(0.06, 0.45 * ear_span)
    joints["head_base"] = head_base
    joints["head_tip"] = head_base + up_axis * head_len

    fingers = {}
    for side, tag in (("L", "left"), ("R", "right")):
        joints["shoulder_origin.%s" % side] = chest
        joints["shoulder.%s" % side] = _point(points, tag + "-shoulder")
        joints["elbow.%s" % side] = _point(points, tag + "-elbow")
        joints["wrist.%s" % side] = _point(points, tag + "-wrist")
        joints["hip.%s" % side] = _point(points, tag + "-hip")
        joints["knee.%s" % side] = _point(points, tag + "-knee")
        joints["ankle.%s" % side] = _point(points, tag + "-ankle")
        joints["ball.%s" % side] = _point(points, tag + "-big-toe-tip")

        tips = []
        per_hand = {}
        for finger in FINGER_CHAIN:
            base = "%s-%s-" % (tag, finger)
            chain = [_point(points, base + "third-joint"),
                     _point(points, base + "second-joint"),
                     _point(points, base + "first-joint"),
                     _point(points, base + "tip")]
            per_hand[finger] = chain
            tips.append(chain[3])
        fingers[side] = per_hand
        joints["hand_tip.%s" % side] = np.mean(tips, axis=0)
    return joints, fingers



def _refs(joints, eye_line) -> dict:
    """参照轴：和 reconstruction._build_refs 同一套（L-R 方向、固定顺序叉乘）。"""
    shoulder_line = joints["shoulder.L"] - joints["shoulder.R"]
    refs = {"hip_line": joints["hip.L"] - joints["hip.R"],
            "shoulder_line": shoulder_line}
    eye = np.asarray(eye_line, dtype=np.float64)
    if float(np.dot(eye, shoulder_line)) < 0.0:
        eye = -eye
    unit = _safe_unit(eye)
    refs["eye_line"] = unit if unit is not None else shoulder_line

    for side in ("L", "R"):
        upper = joints["elbow.%s" % side] - joints["shoulder.%s" % side]
        fore = joints["wrist.%s" % side] - joints["elbow.%s" % side]
        thigh = joints["knee.%s" % side] - joints["hip.%s" % side]
        shin = joints["ankle.%s" % side] - joints["knee.%s" % side]
        axis = _safe_unit(np.cross(upper, fore))
        if axis is not None:
            refs["armbend.%s" % side] = axis
        axis = _safe_unit(np.cross(thigh, shin))
        if axis is not None:
            refs["kneebend.%s" % side] = axis
    return refs


# --------------------------------------------------------------------------- 主入口
def build_recon(data: dict, options: dict | None = None) -> dict:
    """SAM 3D Body 的 npz -> 与 ``reconstruction.reconstruct()`` 同结构的 recon。"""
    options = dict(options or {})
    points, transform, notes = _prepare(data, bool(options.get("swap_lr")))
    pelvis = _mid(points, "left-hip", "right-hip")

    yaw, yaw_note = _facing_yaw(points)
    if yaw_note:
        notes.append(yaw_note)
    points = _rotate_points(points, yaw, pelvis)

    joints, fingers = _roles(points, notes)
    eye_line = _point(points, "left-eye") - _point(points, "right-eye")
    refs = _refs(joints, eye_line)
    spine = [pelvis + (joints["chest"] - pelvis) * (index / 4.0) for index in range(5)]

    yaw_deg = math.degrees(yaw)
    transform["yaw_deg"] = yaw_deg
    transform["pivot"] = pelvis.copy()

    facing = str(options.get("facing") or "FRONT").upper()
    if facing != "FRONT":
        notes.append("SAM 3D Body 从 3D 点自己判朝向，「人物朝向 = %s」这一项不参与" % facing)
    notes.insert(0, "3D 来源：SAM 3D Body（MHR70 共 %d 点，身体 / 脚 / 手指都由模型预测）"
                 % JOINT_COUNT)
    notes.append("朝向：由 3D 结果自动判定（yaw %.1f°）" % yaw_deg)

    meta = data.get("meta") or {}
    return {
        "ok": True,
        "scale": 1.0,
        "yaw_deg": yaw_deg,
        "mode": "SAM3D",
        "joints": joints,
        "refs": refs,
        "spine_points": spine,
        "fingers": fingers,
        "points2d": {},
        "synth": [],
        "notes": notes,
        "transform": transform,
        "sam3d": {
            "focal_length": transform.get("focal", 0.0),
            "reproj_error": transform.get("reproj_error"),
            "image_size": transform.get("image_size"),
            "people": meta.get("people"),
            "has_mesh": "vertices" in data,
            "mesh_vertices": int(np.asarray(data["vertices"]).shape[0]) if "vertices" in data else 0,
        },
    }


def mesh_arrays(data: dict, recon: dict) -> tuple[np.ndarray, np.ndarray]:
    """结果里的网格（相机系顶点 -> 插件世界系），和 recon 用的是同一套变换。"""
    if data.get("vertices") is None:
        raise Sam3DError("这次结果里没有网格：检测时关掉了顶点导出")
    verts = np.asarray(data["vertices"], dtype=np.float64)
    faces = np.asarray(data.get("faces"), dtype=np.int64)
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise Sam3DError("网格面片形状不对：%s" % (faces.shape,))

    transform = (recon or {}).get("transform") or {}
    verts = _to_plugin(verts, bool(transform.get("y_down", True)))
    if transform.get("use_offset"):
        verts = verts + np.asarray(transform.get("offset"), dtype=np.float64).reshape(3)
    if transform.get("pivot") is not None:
        verts = _rotate_points(verts, math.radians(float(transform.get("yaw_deg", 0.0))),
                               np.asarray(transform["pivot"], dtype=np.float64).reshape(3))
    return verts, faces

