"""在**外部 python** 里跑 SAM 3D Body，把结果写成一个 npz 给 Blender 插件读。

为什么是独立脚本
----------------
SAM 3D Body 要 PyTorch / torchvision（可选 detectron2 / SAM3 / MoGe），Blender 自带的
python 装不下、也不该装。插件只负责「拼参数 + 起子进程 + 读 npz」，这个脚本负责全部
torch 相关的活。按官方 INSTALL.md 建一个 conda 环境，然后让插件指向那个 python.exe。

用法（插件会自动拼好；也可以手动跑）
------------------------------------
    python sam3d_worker.py --check --repo D:\\code\\sam-3d-body
    python sam3d_worker.py --repo D:\\code\\sam-3d-body ^
        --ckpt D:\\ckpt\\sam-3d-body-dinov3 --image photo.jpg --out out.npz
    python sam3d_worker.py --repo ... --hf facebook/sam-3d-body-vith ^
        --image-npy rgb.npy --out out.npz --bbox 10 20 300 500 --person 0

输出
----
``out.npz``：keypoints_3d(70,3) / keypoints_2d(70,2) / cam_t(3) / focal_length /
bbox(4) / image_size(2) / vertices(V,3) / faces(F,3)
最后打一行 ``SAM3D_JSON {...}``（父进程只认这一行，其余 stdout 随便打）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback

import numpy as np

PREFIX = "SAM3D_JSON "


def emit(**payload) -> None:
    print(PREFIX + json.dumps(payload, ensure_ascii=False))


def fail(message: str, code: int = 1) -> int:
    emit(ok=False, error=str(message))
    return code


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="SAM 3D Body worker（给 Pose Capture 插件用）")
    parser.add_argument("--repo", default="", help="facebookresearch/sam-3d-body 代码目录")
    parser.add_argument("--ckpt", default="", help="官方下载的 model.ckpt（或它所在目录）")
    parser.add_argument("--mhr", default="", help="mhr_model.pt（默认在 ckpt 目录的 assets/ 里找）")
    parser.add_argument("--hf", default="facebook/sam-3d-body-vith",
                        help="HuggingFace 仓库名（没有本地 ckpt 时用）")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--image", default="", help="图片路径")
    parser.add_argument("--image-npy", default="", help="(H,W,3) uint8 RGB 的 .npy（Blender 数据块走这条）")
    parser.add_argument("--out", default="", help="输出 npz")
    parser.add_argument("--bbox", nargs=4, type=float, default=None,
                        metavar=("X0", "Y0", "X1", "Y1"), help="人框（像素）；不给就用整图")
    parser.add_argument("--person", type=int, default=0, help="多人时取第几个（按检测顺序）")
    parser.add_argument("--body-only", action="store_true", help="只跑身体解码（不跑手）")
    parser.add_argument("--no-vertices", action="store_true", help="不导出网格顶点（更快更小）")
    parser.add_argument("--check", action="store_true", help="只检查环境能不能用")
    return parser.parse_args(argv)


def _resolve_ckpt(ckpt: str, mhr: str) -> tuple[str, str]:
    if ckpt and os.path.isdir(ckpt):
        folder = ckpt
        candidate = os.path.join(folder, "model.ckpt")
        ckpt = candidate if os.path.isfile(candidate) else ""
    else:
        folder = os.path.dirname(ckpt) if ckpt else ""
    if not mhr and folder:
        for name in (os.path.join(folder, "assets", "mhr_model.pt"),
                     os.path.join(folder, "mhr_model.pt")):
            if os.path.isfile(name):
                mhr = name
                break
    return ckpt, mhr


def check(repo: str) -> int:
    info = {"ok": True, "python": sys.version.split()[0], "executable": sys.executable,
            "repo": repo, "torch": None, "cuda": False, "device": "cpu", "version": None}
    if repo:
        if not os.path.isdir(repo):
            return fail("代码目录不存在：%s" % repo)
        if repo not in sys.path:
            sys.path.insert(0, repo)
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda"] = bool(torch.cuda.is_available())
        info["device"] = "cuda" if info["cuda"] else "cpu"
    except Exception as exc:  # noqa: BLE001
        return fail("这个 python 里没有可用的 torch：%s" % exc)
    try:
        import cv2  # noqa: F401  模型预处理要用
    except Exception as exc:  # noqa: BLE001
        return fail("这个 python 里没有 opencv-python：%s" % exc)
    try:
        import sam_3d_body
        info["version"] = getattr(sam_3d_body, "__version__", "?")
    except Exception as exc:  # noqa: BLE001
        return fail("导入 sam_3d_body 失败（--repo 指到仓库根目录了吗？）：%s" % exc)
    emit(**info)
    return 0


# --------------------------------------------------------------------------- 推理
def load_image(args):
    if args.image_npy:
        array = np.load(args.image_npy)
        if array.ndim == 2:
            array = np.stack([array] * 3, axis=-1)
        return np.ascontiguousarray(array[:, :, :3], dtype=np.uint8)
    if not args.image or not os.path.isfile(args.image):
        raise RuntimeError("找不到图片：%s" % (args.image or "(空)"))
    import cv2
    bgr = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError("读不了图片（格式不支持？）：%s" % args.image)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def build_estimator(args):
    """按官方 demo 的方式装模型：有本地 ckpt 用 load_sam_3d_body，否则走 HF。"""
    import torch
    from sam_3d_body import (SAM3DBodyEstimator, load_sam_3d_body,
                             load_sam_3d_body_hf)

    device = "cpu"
    if args.device in ("auto", "cuda") and torch.cuda.is_available():
        device = "cuda"
    ckpt, mhr = _resolve_ckpt(args.ckpt, args.mhr)
    if ckpt:
        model, cfg = load_sam_3d_body(ckpt, device=torch.device(device), mhr_path=mhr or "")
    else:
        model, cfg = load_sam_3d_body_hf(args.hf, device=device)
    return SAM3DBodyEstimator(sam_3d_body_model=model, model_cfg=cfg), device


def _first(value, default=0.0):
    try:
        return float(np.asarray(value).reshape(-1)[0])
    except Exception:  # noqa: BLE001
        return float(default)


def main(argv=None) -> int:
    args = parse_args(argv)
    repo = os.path.abspath(args.repo) if args.repo else ""
    if args.check:
        return check(repo)

    try:
        if repo:
            if not os.path.isdir(repo):
                return fail("代码目录不存在：%s" % repo)
            if repo not in sys.path:
                sys.path.insert(0, repo)
        if not args.out:
            return fail("缺少 --out")

        rgb = load_image(args)
        height, width = rgb.shape[:2]
        estimator, device = build_estimator(args)

        if args.bbox:
            boxes = np.asarray([args.bbox], dtype=np.float32)
        else:                      # 没给框就把整图当框（插件一般会给人框）
            boxes = np.asarray([[0.0, 0.0, float(width - 1), float(height - 1)]],
                               dtype=np.float32)
        outputs = estimator.process_one_image(
            rgb, bboxes=boxes,
            inference_type=("body" if args.body_only else "full"))
        if not outputs:
            return fail("SAM 3D Body 没有给出结果（人框是不是全在画面外？）")

        index = max(0, min(int(args.person), len(outputs) - 1))
        person = outputs[index]
        payload = {
            "keypoints_3d": np.asarray(person["pred_keypoints_3d"], dtype=np.float32),
            "cam_t": np.asarray(person.get("pred_cam_t", np.zeros(3)), dtype=np.float32).reshape(-1)[:3],
            "focal_length": np.asarray([_first(person.get("focal_length"))], dtype=np.float32),
            "bbox": np.asarray(person.get("bbox", boxes[0]), dtype=np.float32).reshape(-1)[:4],
            "image_size": np.asarray([height, width], dtype=np.int32),
        }
        keypoints_2d = person.get("pred_keypoints_2d")
        if keypoints_2d is not None:
            payload["keypoints_2d"] = np.asarray(keypoints_2d, dtype=np.float32)
        if not args.no_vertices and person.get("pred_vertices") is not None:
            payload["vertices"] = np.asarray(person["pred_vertices"], dtype=np.float32)
            payload["faces"] = np.asarray(estimator.faces, dtype=np.int32)

        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        np.savez_compressed(args.out, **payload)

        emit(ok=True, out=os.path.abspath(args.out), people=len(outputs), person=index,
             device=device, has_vertices="vertices" in payload,
             vertices=int(payload["vertices"].shape[0]) if "vertices" in payload else 0,
             bbox=[round(float(v), 2) for v in payload["bbox"]],
             focal=round(_first(payload["focal_length"]), 2),
             image_size=[height, width])
        return 0
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001 - 子进程里所有异常都要变成一行 JSON
        traceback.print_exc()
        return fail("%s: %s" % (type(exc).__name__, exc))


if __name__ == "__main__":
    sys.exit(main())
