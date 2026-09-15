# -*- coding: utf-8 -*-
"""
Đánh giá công bằng Stage 2 cho 3 detector:
1) YOLO26n-OBB
2) DBNet
3) PP-OCRv6 Small Det

Protocol chung:
- Cùng ảnh test
- Cùng GT polygon
- Cùng rule matching: Polygon IoU >= --iou-th (mặc định 0.5)
- One-to-one greedy matching theo IoU giảm dần
- GT có transcription == "###" được bỏ qua
- Prediction overlap vùng ignore theo IoG >= --ignore-iog cũng được bỏ qua
- Tính chung: TP, FP, FN, Precision, Recall, F1
- Đo latency/model trên cùng máy, không tính thời gian đọc ảnh từ disk

"""

import os
import sys
import cv2
import json
import time
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

import torch
from ultralytics import YOLO

import paddle
from paddle import inference


# ============================================================
# Utils
# ============================================================

def model_files(model_dir):
    model_dir = Path(model_dir)
    params = model_dir / "inference.pdiparams"
    if not params.exists():
        raise FileNotFoundError(f"Không thấy: {params}")

    for name in ("inference.json", "inference.pdmodel"):
        model = model_dir / name
        if model.exists():
            return str(model), str(params)

    raise FileNotFoundError(
        f"Không thấy inference.json hoặc inference.pdmodel trong: {model_dir}"
    )


def create_cpu_predictor(model_dir, threads=4, mkldnn=False, ir_optim=True):
    model, params = model_files(model_dir)

    print(f"Load model : {model}")
    print(f"Load params: {params}")

    cfg = inference.Config(model, params)
    cfg.disable_gpu()
    cfg.set_cpu_math_library_num_threads(int(threads))

    try:
        cfg.switch_ir_optim(bool(ir_optim))
        print(f"Paddle IR optim: {'ON' if ir_optim else 'OFF'}")
    except Exception as e:
        print("Warning switch_ir_optim:", e)

    if mkldnn:
        try:
            cfg.enable_mkldnn()
            print("oneDNN/MKLDNN: ON")
        except Exception as e:
            print("Warning MKLDNN:", e)
    else:
        print("oneDNN/MKLDNN explicit enable: OFF")

    return inference.create_predictor(cfg)


def run_predictor(pred, x):
    names = pred.get_input_names()
    handle = pred.get_input_handle(names[0])
    handle.reshape(x.shape)
    handle.copy_from_cpu(x)
    pred.run()
    return [
        pred.get_output_handle(name).copy_to_cpu()
        for name in pred.get_output_names()
    ]


def as_convex(poly):
    poly = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
    if len(poly) < 3:
        return None
    hull = cv2.convexHull(poly).reshape(-1, 2).astype(np.float32)
    if len(hull) < 3:
        return None
    if abs(cv2.contourArea(hull)) < 1e-6:
        return None
    return hull


def polygon_iou(a, b):
    a = as_convex(a)
    b = as_convex(b)
    if a is None or b is None:
        return 0.0

    area_a = float(abs(cv2.contourArea(a)))
    area_b = float(abs(cv2.contourArea(b)))
    inter, _ = cv2.intersectConvexConvex(a, b)
    inter = max(float(inter), 0.0)
    union = area_a + area_b - inter

    if union <= 0:
        return 0.0
    return inter / union


def polygon_iog(pred, gt):
    """
    IoG ở đây = intersection / area(pred).
    Dùng để bỏ prediction nằm chủ yếu trong vùng GT ignore (###).
    """
    pred = as_convex(pred)
    gt = as_convex(gt)
    if pred is None or gt is None:
        return 0.0

    area_pred = float(abs(cv2.contourArea(pred)))
    if area_pred <= 0:
        return 0.0

    inter, _ = cv2.intersectConvexConvex(pred, gt)
    return max(float(inter), 0.0) / area_pred


def common_match(preds, gts, ignores, iou_th=0.5, ignore_iog=0.5):
    """
    Matching chung cho cả 3 model.

    B1: Loại prediction chủ yếu nằm trong vùng ignore (###).
    B2: Tính tất cả cặp pred-GT có IoU >= threshold.
    B3: Sort IoU giảm dần và greedy one-to-one.
    """
    valid_preds = []
    ignored_pred_count = 0

    for p in preds:
        if any(polygon_iog(p, ign) >= ignore_iog for ign in ignores):
            ignored_pred_count += 1
        else:
            valid_preds.append(p)

    candidates = []
    for pi, pred in enumerate(valid_preds):
        for gi, gt in enumerate(gts):
            iou = polygon_iou(pred, gt)
            if iou >= iou_th:
                candidates.append((iou, pi, gi))

    candidates.sort(key=lambda x: x[0], reverse=True)

    used_p = set()
    used_g = set()
    matched_ious = []

    for iou, pi, gi in candidates:
        if pi in used_p or gi in used_g:
            continue
        used_p.add(pi)
        used_g.add(gi)
        matched_ious.append(iou)

    tp = len(used_p)
    fp = len(valid_preds) - tp
    fn = len(gts) - tp

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "num_pred": len(valid_preds),
        "num_gt": len(gts),
        "ignored_pred": ignored_pred_count,
        "matched_ious": matched_ious,
    }


def calc_metrics(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1


# ============================================================
# Ground truth loader
# PaddleOCR detection label format:
# image_path<TAB>[{"transcription":"...","points":[[x,y],...]}, ...]
# ============================================================

def resolve_image_path(label_path, image_root):
    p = Path(label_path)

    if p.is_absolute() and p.exists():
        return p

    image_root = Path(image_root)

    cands = [
        image_root / label_path,
        image_root / p.name,
    ]

    # Xử lý slash Windows/Linux trong label file
    normalized = label_path.replace("\\", "/")
    cands.append(image_root / Path(normalized).name)

    for c in cands:
        if c.exists():
            return c

    return None


def load_paddle_det_gt(label_file, image_root):
    records = []

    with open(label_file, "r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            if "\t" not in line:
                raise ValueError(
                    f"Line {line_no}: không đúng format image_path<TAB>json"
                )

            img_rel, ann_text = line.split("\t", 1)

            try:
                anns = json.loads(ann_text)
            except Exception as e:
                raise ValueError(
                    f"Line {line_no}: JSON annotation lỗi: {e}"
                )

            img_path = resolve_image_path(img_rel, image_root)
            if img_path is None:
                raise FileNotFoundError(
                    f"Line {line_no}: không tìm thấy ảnh '{img_rel}' "
                    f"trong image_root='{image_root}'"
                )

            gts = []
            ignores = []

            for ann in anns:
                pts = ann.get("points", None)
                if pts is None:
                    continue

                poly = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
                if len(poly) < 3:
                    continue

                text = str(ann.get("transcription", "")).strip()

                if text == "###" or bool(ann.get("ignore", False)):
                    ignores.append(poly)
                else:
                    gts.append(poly)

            records.append({
                "image_path": str(img_path),
                "image_name": img_path.name,
                "gts": gts,
                "ignores": ignores,
            })

    return records


# ============================================================
# YOLO26n-OBB
# ============================================================

class YoloObbDetector:
    def __init__(self, weights, conf=0.25, imgsz=640, device="cpu"):
        self.model = YOLO(weights)
        self.conf = float(conf)
        self.imgsz = int(imgsz)
        self.device = device

    def predict(self, image):
        result = self.model.predict(
            image,
            conf=self.conf,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )[0]

        if result.obb is None or len(result.obb) == 0:
            return []

        polys = result.obb.xyxyxyxy.detach().cpu().numpy()
        return [
            np.asarray(p, dtype=np.float32).reshape(4, 2)
            for p in polys
        ]


# ============================================================
# Paddle DB-style detector
# DBNet và PP-OCRv6 Small Det đều dùng DBPostProcess.
# ============================================================

class PaddleDBDetector:
    def __init__(
        self,
        model_dir,
        paddleocr_dir,
        resize_h,
        resize_w,
        thresh,
        box_thresh,
        unclip_ratio,
        max_candidates,
        cpu_threads=4,
        mkldnn=False,
        ir_optim=True,
    ):
        if paddleocr_dir not in sys.path:
            sys.path.insert(0, paddleocr_dir)

        from ppocr.data import create_operators, transform
        from ppocr.postprocess import build_post_process

        self.transform_fn = transform

        self.ops = create_operators([
            {"DetResizeForTest": {"image_shape": [int(resize_h), int(resize_w)]}},
            {
                "NormalizeImage": {
                    "scale": "1./255.",
                    "mean": [0.485, 0.456, 0.406],
                    "std": [0.229, 0.224, 0.225],
                    "order": "hwc",
                }
            },
            {"ToCHWImage": None},
            {"KeepKeys": {"keep_keys": ["image", "shape"]}},
        ])

        self.post = build_post_process({
            "name": "DBPostProcess",
            "thresh": float(thresh),
            "box_thresh": float(box_thresh),
            "max_candidates": int(max_candidates),
            "unclip_ratio": float(unclip_ratio),
        })

        self.pred = create_cpu_predictor(
            model_dir,
            threads=cpu_threads,
            mkldnn=mkldnn,
            ir_optim=ir_optim,
        )

    def predict(self, image):
        x, shape = self.transform_fn(
            {"image": image.copy()},
            self.ops,
        )

        x = np.expand_dims(x, 0).copy()
        shape = np.expand_dims(shape, 0)

        outs = run_predictor(self.pred, x)

        # DB output thường là NCHW probability map.
        maps = next(
            (o for o in outs if getattr(o, "ndim", 0) == 4),
            outs[0],
        )

        result = self.post({"maps": maps}, shape)

        if not result:
            return []

        points = result[0].get("points", [])
        return [
            np.asarray(p, dtype=np.float32).reshape(-1, 2)
            for p in points
        ]


# ============================================================
# Visualization
# ============================================================

def draw_polys(image, gts, preds):
    vis = image.copy()

    # GT: green
    for p in gts:
        cv2.polylines(
            vis,
            [np.asarray(p, np.int32)],
            True,
            (0, 255, 0),
            2,
        )

    # Prediction: red
    for p in preds:
        cv2.polylines(
            vis,
            [np.asarray(p, np.int32)],
            True,
            (0, 0, 255),
            2,
        )

    return vis


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()

    # Dataset
    ap.add_argument("--image-root", required=True)
    ap.add_argument("--label-file", required=True)
    ap.add_argument("--paddleocr-dir", required=True)

    # Models
    ap.add_argument("--yolo-weights", required=True)
    ap.add_argument("--dbnet-infer", required=True)
    ap.add_argument("--ppocrv6-infer", required=True)

    # Common evaluator
    ap.add_argument("--iou-th", type=float, default=0.5)
    ap.add_argument("--ignore-iog", type=float, default=0.5)

    # YOLO operating point
    ap.add_argument("--yolo-conf", type=float, default=0.25)
    ap.add_argument("--yolo-imgsz", type=int, default=640)

    # DBNet operating point
    ap.add_argument("--db-thresh", type=float, default=0.30)
    ap.add_argument("--db-box-thresh", type=float, default=0.60)
    ap.add_argument("--db-unclip", type=float, default=1.50)
    ap.add_argument("--db-max-candidates", type=int, default=1000)

    # PP-OCRv6 operating point
    ap.add_argument("--pp-thresh", type=float, default=0.20)
    ap.add_argument("--pp-box-thresh", type=float, default=0.45)
    ap.add_argument("--pp-unclip", type=float, default=1.40)
    ap.add_argument("--pp-max-candidates", type=int, default=3000)

    # Paddle det resize: theo config test hiện tại của bạn
    ap.add_argument("--det-h", type=int, default=736)
    ap.add_argument("--det-w", type=int, default=1280)

    # Runtime
    ap.add_argument("--cpu-threads", type=int, default=4)
    ap.add_argument("--mkldnn", action="store_true")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--max-images", type=int, default=0)

    # Output
    ap.add_argument(
        "--output-dir",
        default="output_stage2_fair_eval",
    )
    ap.add_argument("--save-vis", action="store_true")

    args = ap.parse_args()

    # ============================================================
    # STRICT CPU SPEED MODE
    # ============================================================
    # Mục tiêu: so tốc độ 3 model trên cùng điều kiện:
    # - CPU only
    # - batch = 1
    # - cùng số CPU threads
    # - tắt oneDNN/MKLDNN ở cả PyTorch và Paddle
    # - latency bao gồm preprocess + inference + postprocess
    # - không tính thời gian đọc ảnh từ disk
    try:
        torch.set_num_threads(int(args.cpu_threads))
        torch.set_num_interop_threads(1)
    except Exception:
        pass

    try:
        torch.backends.mkldnn.enabled = False
    except Exception:
        pass

    try:
        cv2.setNumThreads(int(args.cpu_threads))
    except Exception:
        pass

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 76)
    print("STAGE 2 - COMMON FAIR EVALUATOR")
    print("=" * 76)
    print(f"Python      : {sys.version.split()[0]}")
    print(f"Paddle      : {paddle.__version__}")
    print(f"Common IoU  : {args.iou_th}")
    print(f"Ignore IoG  : {args.ignore_iog}")
    print(f"CPU threads : {args.cpu_threads}")
    print("Mode        : COMMON ACCURACY PROTOCOL; DBNet legacy IR optim OFF for compatibility")
    print()

    records = load_paddle_det_gt(
        args.label_file,
        args.image_root,
    )

    if args.max_images > 0:
        records = records[:args.max_images]

    if not records:
        raise RuntimeError("Không có ảnh test.")

    print(f"Images      : {len(records)}")
    print()

    print("[1/3] Load YOLO26n-OBB")
    yolo = YoloObbDetector(
        args.yolo_weights,
        conf=args.yolo_conf,
        imgsz=args.yolo_imgsz,
        device="cpu",
    )

    print("\n[2/3] Load DBNet")
    print("DBNet legacy: oneDNN OFF + IR optim OFF (Paddle 3.x compatibility)")
    # DBNet của bạn được export ở định dạng legacy .pdmodel từ Paddle 2.6.x.
    # Trên Paddle 3.2.1 + Windows, oneDNN có thể fuse thành fused_conv2d rồi lỗi:
    # "OneDnnContext does not have the input Filter".
    # Vì vậy tắt oneDNN riêng cho DBNet legacy để đảm bảo inference ổn định.
    dbnet = PaddleDBDetector(
        model_dir=args.dbnet_infer,
        paddleocr_dir=args.paddleocr_dir,
        resize_h=args.det_h,
        resize_w=args.det_w,
        thresh=args.db_thresh,
        box_thresh=args.db_box_thresh,
        unclip_ratio=args.db_unclip,
        max_candidates=args.db_max_candidates,
        cpu_threads=args.cpu_threads,
        mkldnn=False,
        ir_optim=False,
    )

    print("\n[3/3] Load PP-OCRv6 Small Det")
    print("PP-OCRv6: oneDNN/MKLDNN OFF (strict speed mode)")
    # Strict speed mode: tắt oneDNN cho PP-OCRv6 để cùng điều kiện với DBNet.
    ppocr = PaddleDBDetector(
        model_dir=args.ppocrv6_infer,
        paddleocr_dir=args.paddleocr_dir,
        resize_h=args.det_h,
        resize_w=args.det_w,
        thresh=args.pp_thresh,
        box_thresh=args.pp_box_thresh,
        unclip_ratio=args.pp_unclip,
        max_candidates=args.pp_max_candidates,
        cpu_threads=args.cpu_threads,
        mkldnn=False,
        ir_optim=True,
    )

    detectors = {
        "YOLO26n-OBB": yolo,
        "DBNet": dbnet,
        "PP-OCRv6-Small-Det": ppocr,
    }

    # Warmup cùng một ảnh, không tính latency.
    warm_img = cv2.imread(records[0]["image_path"])
    if warm_img is None:
        raise RuntimeError("Không đọc được ảnh warmup.")

    if args.warmup > 0:
        print(f"\nWarmup: {args.warmup} lần/model")
        for name, detector in detectors.items():
            for _ in range(args.warmup):
                detector.predict(warm_img)
            print(f"  {name}: done")

    totals = {
        name: {
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "ignored_pred": 0,
            "latency_ms": [],
            "matched_ious": [],
        }
        for name in detectors
    }

    per_image_rows = []

    vis_dirs = {}
    if args.save_vis:
        for name in detectors:
            d = out_dir / "visualizations" / name.replace("/", "_")
            d.mkdir(parents=True, exist_ok=True)
            vis_dirs[name] = d

    print("\nEvaluating...")
    for rec in tqdm(records, desc="Common eval"):
        image = cv2.imread(rec["image_path"])
        if image is None:
            raise RuntimeError(
                f"Không đọc được ảnh: {rec['image_path']}"
            )

        gts = rec["gts"]
        ignores = rec["ignores"]

        for name, detector in detectors.items():
            t0 = time.perf_counter()
            preds = detector.predict(image)
            latency_ms = (time.perf_counter() - t0) * 1000.0

            m = common_match(
                preds,
                gts,
                ignores,
                iou_th=args.iou_th,
                ignore_iog=args.ignore_iog,
            )

            totals[name]["tp"] += m["tp"]
            totals[name]["fp"] += m["fp"]
            totals[name]["fn"] += m["fn"]
            totals[name]["ignored_pred"] += m["ignored_pred"]
            totals[name]["latency_ms"].append(latency_ms)
            totals[name]["matched_ious"].extend(m["matched_ious"])

            p, r, f1 = calc_metrics(
                m["tp"],
                m["fp"],
                m["fn"],
            )

            per_image_rows.append({
                "image": rec["image_name"],
                "model": name,
                "num_gt": m["num_gt"],
                "num_pred": m["num_pred"],
                "tp": m["tp"],
                "fp": m["fp"],
                "fn": m["fn"],
                "precision": p,
                "recall": r,
                "f1": f1,
                "mean_matched_iou": (
                    float(np.mean(m["matched_ious"]))
                    if m["matched_ious"] else 0.0
                ),
                "ignored_pred": m["ignored_pred"],
                "latency_ms": latency_ms,
            })

            if args.save_vis:
                vis = draw_polys(image, gts, preds)
                cv2.imwrite(
                    str(vis_dirs[name] / rec["image_name"]),
                    vis,
                )

    # Summary
    summary_rows = []

    for name in detectors:
        d = totals[name]
        p, r, f1 = calc_metrics(
            d["tp"],
            d["fp"],
            d["fn"],
        )

        lat = np.asarray(d["latency_ms"], dtype=np.float64)

        summary_rows.append({
            "model": name,
            "images": len(records),
            "tp": d["tp"],
            "fp": d["fp"],
            "fn": d["fn"],
            "precision_pct": p * 100.0,
            "recall_pct": r * 100.0,
            "f1_pct": f1 * 100.0,
            "mean_matched_iou_pct": (
                float(np.mean(d["matched_ious"])) * 100.0
                if d["matched_ious"] else 0.0
            ),
            "ignored_pred": d["ignored_pred"],
            "mean_ms": float(np.mean(lat)),
            "median_ms": float(np.median(lat)),
            "p95_ms": float(np.percentile(lat, 95)),
            "fps_from_mean": (
                1000.0 / float(np.mean(lat))
                if float(np.mean(lat)) > 0 else 0.0
            ),
        })

    summary_df = pd.DataFrame(summary_rows)
    per_image_df = pd.DataFrame(per_image_rows)

    summary_df.to_csv(
        out_dir / "summary_common_protocol.csv",
        index=False,
        encoding="utf-8-sig",
    )

    per_image_df.to_csv(
        out_dir / "per_image_common_protocol.csv",
        index=False,
        encoding="utf-8-sig",
    )

    protocol = {
        "dataset": {
            "image_root": str(Path(args.image_root).resolve()),
            "label_file": str(Path(args.label_file).resolve()),
            "images": len(records),
        },
        "common_evaluation": {
            "polygon_iou_threshold": args.iou_th,
            "one_to_one_matching": True,
            "matching_strategy": "greedy_descending_iou",
            "ignore_transcription": "###",
            "ignore_prediction_iog_threshold": args.ignore_iog,
        },
        "operating_points": {
            "YOLO26n-OBB": {
                "conf": args.yolo_conf,
                "imgsz": args.yolo_imgsz,
            },
            "DBNet": {
                "thresh": args.db_thresh,
                "box_thresh": args.db_box_thresh,
                "unclip_ratio": args.db_unclip,
                "max_candidates": args.db_max_candidates,
                "resize": [args.det_h, args.det_w],
            },
            "PP-OCRv6-Small-Det": {
                "thresh": args.pp_thresh,
                "box_thresh": args.pp_box_thresh,
                "unclip_ratio": args.pp_unclip,
                "max_candidates": args.pp_max_candidates,
                "resize": [args.det_h, args.det_w],
            },
        },
        "runtime": {
            "device": "CPU",
            "cpu_threads": args.cpu_threads,
            "mkldnn": False,
            "torch_mkldnn": False,
            "opencv_threads": args.cpu_threads,
            "speed_mode": "strict_cpu_no_onednn",
            "warmup_per_model": args.warmup,
        },
    }

    with open(
        out_dir / "evaluation_protocol.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(protocol, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 76)
    print("KẾT QUẢ - CÙNG MỘT EVALUATION PROTOCOL")
    print("=" * 76)

    show_cols = [
        "model",
        "tp",
        "fp",
        "fn",
        "precision_pct",
        "recall_pct",
        "f1_pct",
        "mean_matched_iou_pct",
        "mean_ms",
        "fps_from_mean",
    ]

    print(
        summary_df[show_cols].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print("\nSaved:", out_dir.resolve())
    print("=" * 76)


if __name__ == "__main__":
    main()
