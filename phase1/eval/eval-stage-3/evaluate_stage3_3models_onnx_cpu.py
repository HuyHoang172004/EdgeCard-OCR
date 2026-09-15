#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
EdgeCard - Stage 3 Recognition Benchmark on PC CPU
==================================================

So sánh 3 recognizer dưới CÙNG ONNX Runtime CPU:
1) MobileNetV3-CRNN
2) PP-OCRv6 Small Rec
3) RepSVTR

Giữ đúng logic đánh giá từ notebook cũ:
- Cùng test set: test_label.txt
- Unicode NFC
- Batch = 1
- Warm-up = 30 crop
- Load/decode toàn bộ ảnh vào RAM trước -> không tính disk I/O
- Latency = preprocess + ONNX Runtime inference + CTC decode
- Exact Accuracy
- CER
- NED
- Mean / Median / P95 latency
- FPS = 1000 / mean latency

"""

import os
import sys
import json
import time
import math
import unicodedata
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import onnxruntime as ort


# ============================================================
# 1. CHỈ CẦN SỬA CÁC ĐƯỜNG DẪN Ở ĐÂY
# ============================================================

# Dataset Stage 3
DATA_DIR = Path(
    r"D:\AI_Project\ocr_the_sv\src\eval_pipeline\eval-stage-3\dataset_rec"
)

TEST_LABEL = Path(
    r"D:\AI_Project\ocr_the_sv\src\eval_pipeline\eval-stage-3\dataset_rec\test_label.txt"
)

# Ba model ONNX
CRNN_ONNX = Path(
    r"D:\AI_Project\ocr_the_sv\src\eval_pipeline\weights\stage-3\mobilenetv3-crnn\mobilenetv3_crnn.onnx"
)

PPOCR_ONNX = Path(
    r"D:\AI_Project\ocr_the_sv\src\eval_pipeline\weights\stage-3\pp-ocrv6\ppocrv6_small_rec.onnx"
)

REPSVTR_ONNX = Path(
    r"D:\AI_Project\ocr_the_sv\src\eval_pipeline\weights\stage-3\repstvr\repsvtr.onnx"
)

# Dictionary
PPOCR_DICT = Path(
    r"D:\AI_Project\ocr_the_sv\src\eval_pipeline\weights\stage-3\pp-ocrv6\reproducibility\studentcard_vi_dict.txt"
)

REPSVTR_DICT = Path(
    r"D:\AI_Project\ocr_the_sv\src\eval_pipeline\weights\stage-3\pp-ocrv6\reproducibility\studentcard_vi_dict.txt"
)

# Thư mục lưu kết quả
OUTPUT_DIR = Path(
    r"D:\AI_Project\ocr_the_sv\src\eval_pipeline\eval-stage-3\output"
)


# ============================================================
# 2. CẤU HÌNH BENCHMARK
# ============================================================

WARMUP = 30
CPU_THREADS = 4            # i5-10300H: 4 physical cores
EXPECTED_SAMPLES = 1855
LIMIT = 0                  # 0 = chạy toàn bộ; ví dụ 50 để test nhanh
SHOW_WRONG = 10

IMG_W = 320
EVAL_H = 48
IMAGE_SHAPE = [3, 48, 320]

IMAGENET_MEAN = np.asarray(
    [0.485, 0.456, 0.406],
    dtype=np.float32
).reshape(3, 1, 1)

IMAGENET_STD = np.asarray(
    [0.229, 0.224, 0.225],
    dtype=np.float32
).reshape(3, 1, 1)

BLANK_IDX = 0


# ============================================================
# 3. VOCABULARY CRNN 
# ============================================================

vietnamese_chars = (
    "aAàÀảẢãÃáÁạẠăĂằẰẳẲẵẴắẮặẶâÂầẦẩẨẫẪấẤậẬ"
    "bBcCdDđĐeEèÈẻẺẽẼéÉẹẸêÊềỀểỂễỄếẾệỆ"
    "fFgGhHiIìÌỉỈĩĨíÍịỊjJkKlLmMnNoO"
    "òÒỏỎõÕóÓọỌôÔồỒổỔỗỖốỐộỘơƠờỜởỞỡỠớỚợỢ"
    "pPqQrRsStTuUùÙủỦũŨúÚụỤưƯừỪửỬữỮứỨựỰ"
    "vVwWxXyYỳỲỷỶỹỸýÝỵỴzZ"
    "0123456789"
    "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
)

CRNN_CHARACTERS = list(dict.fromkeys(vietnamese_chars)) + [" "]


# ============================================================
# 4. ĐỌC TEST SET
# ============================================================

def load_test_samples(label_file, data_dir):
    samples = []

    with open(label_file, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.rstrip("\r\n")

            if not line:
                continue

            parts = line.split("\t", 1)

            if len(parts) != 2:
                raise ValueError(
                    f"Bad label line {line_no}: {line}"
                )

            rel_path, text = parts

            rel_path = rel_path.replace("\\", "/")
            text = unicodedata.normalize("NFC", text)

            full_path = (
                Path(rel_path)
                if os.path.isabs(rel_path)
                else data_dir.joinpath(*rel_path.split("/"))
            )

            if not full_path.exists():
                raise FileNotFoundError(
                    f"Missing image at line {line_no}: {full_path}"
                )

            samples.append(
                (str(full_path), text)
            )

    return samples


def load_dict(path):
    chars = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            ch = line.rstrip("\r\n")

            if ch != "":
                chars.append(ch)

    return chars


# ============================================================
# 5. EVALUATOR CHUNG
# ============================================================

def edit_distance(a, b):
    a = unicodedata.normalize("NFC", a)
    b = unicodedata.normalize("NFC", b)

    prev = list(range(len(b) + 1))

    for i, ca in enumerate(a, 1):
        curr = [i]

        for j, cb in enumerate(b, 1):
            curr.append(min(
                curr[-1] + 1,               # insertion
                prev[j] + 1,                # deletion
                prev[j - 1] + (ca != cb)    # substitution
            ))

        prev = curr

    return prev[-1]


def compute_common_metrics(preds, gts):
    assert len(preds) == len(gts)

    exact = 0
    edits = 0
    gt_chars = 0
    ned_sum = 0.0

    for pred, gt in zip(preds, gts):
        pred = unicodedata.normalize("NFC", str(pred))
        gt = unicodedata.normalize("NFC", str(gt))

        exact += int(pred == gt)

        d = edit_distance(pred, gt)

        edits += d
        gt_chars += len(gt)

        ned_sum += (
            1.0 -
            d / max(len(pred), len(gt), 1)
        )

    n = len(gts)

    return {
        "samples": n,
        "correct": exact,
        "wrong": n - exact,
        "exact_acc": exact / n,
        "cer": edits / max(gt_chars, 1),
        "ned": ned_sum / n,
    }


def make_result(
    model_name,
    preds,
    gts,
    latencies_ms
):
    m = compute_common_metrics(
        preds,
        gts
    )

    latency_mean = float(
        np.mean(latencies_ms)
    )

    latency_median = float(
        np.median(latencies_ms)
    )

    latency_p95 = float(
        np.percentile(latencies_ms, 95)
    )

    fps = 1000.0 / latency_mean

    result = {
        "Model": model_name,
        "Samples": m["samples"],
        "Correct": m["correct"],
        "Wrong": m["wrong"],
        "Exact Acc (%)": m["exact_acc"] * 100,
        "CER (%)": m["cer"] * 100,
        "NED (%)": m["ned"] * 100,
        "Latency (ms)": latency_mean,
        "Median Latency (ms)": latency_median,
        "P95 Latency (ms)": latency_p95,
        "FPS": fps,
    }

    return result


def print_result(r):
    print("=" * 64)
    print(r["Model"])
    print("=" * 64)

    print(
        f'Samples       : '
        f'{r["Samples"]}'
    )

    print(
        f'Correct/Wrong : '
        f'{r["Correct"]}/{r["Wrong"]}'
    )

    print(
        f'Exact Acc     : '
        f'{r["Exact Acc (%)"]:.4f}%'
    )

    print(
        f'CER           : '
        f'{r["CER (%)"]:.4f}%'
    )

    print(
        f'NED           : '
        f'{r["NED (%)"]:.4f}%'
    )

    print(
        f'Latency mean  : '
        f'{r["Latency (ms)"]:.4f} ms/crop'
    )

    print(
        f'Latency median: '
        f'{r["Median Latency (ms)"]:.4f} ms/crop'
    )

    print(
        f'Latency P95   : '
        f'{r["P95 Latency (ms)"]:.4f} ms/crop'
    )

    print(
        f'FPS           : '
        f'{r["FPS"]:.2f}'
    )


# ============================================================
# 6. ONNX RUNTIME CPU
# ============================================================

def create_cpu_session(model_path):
    if not model_path.exists():
        raise FileNotFoundError(model_path)

    options = ort.SessionOptions()

    options.intra_op_num_threads = CPU_THREADS
    options.inter_op_num_threads = 1

    options.execution_mode = (
        ort.ExecutionMode.ORT_SEQUENTIAL
    )

    options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )

    session = ort.InferenceSession(
        str(model_path),
        sess_options=options,
        providers=[
            "CPUExecutionProvider"
        ]
    )

    return session


def choose_ctc_output(
    session,
    expected_classes
):
    outputs = session.get_outputs()

    for i, output in enumerate(outputs):
        shape = output.shape

        if (
            shape and
            len(shape) >= 3
        ):
            last_dim = shape[-1]

            if (
                isinstance(last_dim, int) and
                last_dim == expected_classes
            ):
                return i

    return 0


# ============================================================
# 7. CTC GREEDY DECODER
# ============================================================

def ctc_greedy_decode_one(
    logits,
    idx2char,
    blank_idx=0
):
    if logits.ndim != 3:
        raise ValueError(
            f"Expected [B,T,C], got {logits.shape}"
        )

    seq = np.argmax(
        logits,
        axis=-1
    )[0]

    text = []
    prev = -1

    for idx in seq.tolist():
        idx = int(idx)

        if (
            idx != blank_idx and
            idx != prev
        ):
            if idx not in idx2char:
                raise KeyError(
                    f"Class {idx} không có "
                    f"trong dictionary."
                )

            text.append(
                idx2char[idx]
            )

        prev = idx

    return "".join(text)


# ============================================================
# 8. PREPROCESSING CRNN
# ============================================================

def crnn_resize_pad(
    image,
    target_h=EVAL_H,
    target_w=IMG_W
):
    w, h = image.size

    new_w = max(
        1,
        min(
            target_w,
            round(
                w * target_h / h
            )
        )
    )

    image = image.resize(
        (new_w, target_h),
        Image.Resampling.BILINEAR
    )

    canvas = Image.new(
        "RGB",
        (target_w, target_h),
        (255, 255, 255)
    )

    canvas.paste(
        image,
        (0, 0)
    )

    arr = np.asarray(
        canvas,
        dtype=np.float32
    )

    arr /= 255.0

    arr = arr.transpose(
        2, 0, 1
    )

    arr = (
        arr - IMAGENET_MEAN
    ) / IMAGENET_STD

    return arr[
        np.newaxis, :
    ].astype(
        np.float32,
        copy=False
    )


# ============================================================
# 9. PREPROCESSING PP-OCRv6 / RepSVTR
#    TƯƠNG ĐƯƠNG resize_norm_img(..., [3,48,320], padding=True)
# ============================================================

def paddle_resize_norm_img(
    img_bgr,
    image_shape=IMAGE_SHAPE,
    padding=True
):
    imgC, imgH, imgW = image_shape

    if img_bgr is None:
        raise ValueError(
            "img_bgr is None"
        )

    h, w = img_bgr.shape[:2]

    ratio = (
        w / float(h)
    )

    if padding:
        if math.ceil(
            imgH * ratio
        ) > imgW:
            resized_w = imgW
        else:
            resized_w = int(
                math.ceil(
                    imgH * ratio
                )
            )
    else:
        resized_w = imgW

    resized_w = max(
        1,
        resized_w
    )

    resized_image = cv2.resize(
        img_bgr,
        (resized_w, imgH),
        interpolation=cv2.INTER_LINEAR
    )

    resized_image = (
        resized_image.astype(
            np.float32
        )
    )

    resized_image = (
        resized_image.transpose(
            (2, 0, 1)
        ) / 255.0
    )

    resized_image -= 0.5
    resized_image /= 0.5

    if padding:
        padding_im = np.zeros(
            (imgC, imgH, imgW),
            dtype=np.float32
        )

        padding_im[
            :, :, 0:resized_w
        ] = resized_image

        norm_img = padding_im

    else:
        norm_img = resized_image

    return norm_img[
        np.newaxis, :
    ].copy()


# ============================================================
# 10. PRELOAD ẢNH RA RAM
#     Disk I/O KHÔNG nằm trong latency
# ============================================================

def preload_crnn_images(
    test_samples
):
    images = []

    for path, _ in tqdm(
        test_samples,
        desc="Preload CRNN images"
    ):
        with Image.open(path) as image:
            images.append(
                image.convert(
                    "RGB"
                ).copy()
            )

    return images


def preload_paddle_images(
    test_samples
):
    images = []

    for path, _ in tqdm(
        test_samples,
        desc="Preload Paddle-format images"
    ):
        img = cv2.imread(path)

        if img is None:
            raise RuntimeError(
                f"Không đọc được ảnh: {path}"
            )

        images.append(img)

    return images


# ============================================================
# 11. BENCHMARK CRNN ONNX
# ============================================================

def benchmark_crnn(
    session,
    images,
    test_samples,
    idx2char
):
    model_name = (
        "MobileNetV3-CRNN"
    )

    input_name = (
        session
        .get_inputs()[0]
        .name
    )

    output_idx = (
        choose_ctc_output(
            session,
            len(idx2char) + 1
        )
    )

    print(
        "\n" +
        "=" * 72
    )

    print(
        model_name,
        "- ONNX Runtime CPU"
    )

    print(
        "=" * 72
    )

    print(
        "Input :",
        session
        .get_inputs()[0]
        .shape
    )

    print(
        "Output:",
        session
        .get_outputs()[output_idx]
        .shape
    )

    # Warm-up = 30
    for i in range(WARMUP):
        img = images[
            i % len(images)
        ]

        x = crnn_resize_pad(
            img
        )

        outs = session.run(
            None,
            {
                input_name: x
            }
        )

        _ = (
            ctc_greedy_decode_one(
                outs[output_idx],
                idx2char,
                BLANK_IDX
            )
        )

    preds = []
    times = []

    gts = [
        gt
        for _, gt
        in test_samples
    ]

    for img in tqdm(
        images,
        desc=model_name
    ):
        t0 = (
            time.perf_counter()
        )

        x = crnn_resize_pad(
            img
        )

        outs = session.run(
            None,
            {
                input_name: x
            }
        )

        pred = (
            ctc_greedy_decode_one(
                outs[output_idx],
                idx2char,
                BLANK_IDX
            )
        )

        pred = (
            unicodedata.normalize(
                "NFC",
                pred
            )
        )

        dt_ms = (
            time.perf_counter() -
            t0
        ) * 1000.0

        preds.append(
            pred
        )

        times.append(
            dt_ms
        )

    result = make_result(
        model_name,
        preds,
        gts,
        times
    )

    return (
        result,
        preds,
        gts,
        times
    )


# ============================================================
# 12. BENCHMARK PP-OCRv6 / RepSVTR ONNX
# ============================================================

def benchmark_paddle_onnx(
    model_name,
    session,
    images,
    test_samples,
    idx2char
):
    input_name = (
        session
        .get_inputs()[0]
        .name
    )

    expected_classes = (
        len(idx2char) + 1
    )

    output_idx = (
        choose_ctc_output(
            session,
            expected_classes
        )
    )

    print(
        "\n" +
        "=" * 72
    )

    print(
        model_name,
        "- ONNX Runtime CPU"
    )

    print(
        "=" * 72
    )

    print(
        "Input :",
        session
        .get_inputs()[0]
        .shape
    )

    for i, output in enumerate(
        session.get_outputs()
    ):
        print(
            f"Output[{i}]:",
            output.shape
        )

    # Warm-up = 30
    for i in range(WARMUP):
        img = images[
            i % len(images)
        ]

        x = paddle_resize_norm_img(
            img,
            IMAGE_SHAPE,
            padding=True
        )

        outs = session.run(
            None,
            {
                input_name: x
            }
        )

        _ = (
            ctc_greedy_decode_one(
                outs[output_idx],
                idx2char,
                BLANK_IDX
            )
        )

    preds = []
    times = []

    gts = [
        gt
        for _, gt
        in test_samples
    ]

    first = True

    for img in tqdm(
        images,
        desc=model_name
    ):
        t0 = (
            time.perf_counter()
        )

        x = paddle_resize_norm_img(
            img,
            IMAGE_SHAPE,
            padding=True
        )

        outs = session.run(
            None,
            {
                input_name: x
            }
        )

        logits = (
            outs[output_idx]
        )

        pred = (
            ctc_greedy_decode_one(
                logits,
                idx2char,
                BLANK_IDX
            )
        )

        pred = (
            unicodedata.normalize(
                "NFC",
                pred
            )
        )

        dt_ms = (
            time.perf_counter() -
            t0
        ) * 1000.0

        if first:
            print(
                "Selected output:",
                output_idx
            )

            print(
                "Inference output shape:",
                logits.shape
            )

            if (
                logits.ndim >= 3 and
                logits.shape[-1] !=
                expected_classes
            ):
                raise RuntimeError(
                    f"{model_name}: "
                    f"output classes "
                    f"{logits.shape[-1]} "
                    f"!= expected "
                    f"{expected_classes}"
                )

            first = False

        preds.append(
            pred
        )

        times.append(
            dt_ms
        )

    result = make_result(
        model_name,
        preds,
        gts,
        times
    )

    return (
        result,
        preds,
        gts,
        times
    )


# ============================================================
# 13. LƯU KẾT QUẢ
# ============================================================

def save_model_result(
    prefix,
    test_samples,
    gts,
    preds,
    times,
    result
):
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    pred_path = (
        OUTPUT_DIR /
        f"{prefix}_predictions.csv"
    )

    result_path = (
        OUTPUT_DIR /
        f"{prefix}_result.json"
    )

    pd.DataFrame({
        "image": [
            p
            for p, _
            in test_samples
        ],
        "ground_truth": gts,
        "prediction": preds,
        "latency_ms": times,
    }).to_csv(
        pred_path,
        index=False,
        encoding="utf-8-sig"
    )

    with open(
        result_path,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            result,
            f,
            ensure_ascii=False,
            indent=2
        )

    print(
        "Saved:",
        pred_path
    )

    print(
        "Saved:",
        result_path
    )


def show_wrong_examples(
    model_name,
    test_samples,
    gts,
    preds,
    times
):
    if SHOW_WRONG <= 0:
        return

    wrong = []

    for i, (
        gt,
        pred,
        latency
    ) in enumerate(
        zip(
            gts,
            preds,
            times
        )
    ):
        if (
            unicodedata.normalize(
                "NFC",
                gt
            ) !=
            unicodedata.normalize(
                "NFC",
                pred
            )
        ):
            wrong.append({
                "index": i,
                "image": (
                    test_samples[i][0]
                ),
                "ground_truth": gt,
                "prediction": pred,
                "latency_ms": latency,
            })

    print(
        f"\n{model_name} - "
        f"Wrong: {len(wrong)} / "
        f"{len(gts)}"
    )

    for row in wrong[
        :SHOW_WRONG
    ]:
        print(
            "-" * 72
        )

        print(
            "Index:",
            row["index"]
        )

        print(
            "GT   :",
            row["ground_truth"]
        )

        print(
            "Pred :",
            row["prediction"]
        )

        print(
            f'Time : '
            f'{row["latency_ms"]:.4f} ms'
        )


def reference_check(
    result,
    expected_pct
):
    diff = abs(
        result["Exact Acc (%)"] -
        expected_pct
    )

    if diff > 0.3:
        print(
            "⚠️ Exact Acc lệch đáng kể "
            f"so với benchmark cũ "
            f"({expected_pct:.4f}%). "
            "Kiểm tra preprocessing, "
            "dictionary hoặc model export."
        )
    else:
        print(
            "Reference check OK: "
            f"gần {expected_pct:.4f}%."
        )


# ============================================================
# 14. MAIN
# ============================================================

def main():
    print(
        "=" * 72
    )

    print(
        "EdgeCard - Stage 3 ONNX "
        "Runtime CPU Benchmark"
    )

    print(
        "=" * 72
    )

    print(
        "Python       :",
        sys.version.split()[0]
    )

    print(
        "ONNX Runtime :",
        ort.__version__
    )

    print(
        "Providers    :",
        ort.get_available_providers()
    )

    print(
        "CPU threads  :",
        CPU_THREADS
    )

    print(
        "Warm-up      :",
        WARMUP
    )

    print(
        "Batch        : 1"
    )

    print(
        "Input eval   : "
        "3x48x320"
    )

    print(
        "Latency scope: "
        "RAM image -> preprocess -> "
        "ORT inference -> CTC decode"
    )

    print(
        "Disk I/O     : excluded"
    )

    # --------------------------------------------------------
    # Kiểm tra đường dẫn
    # --------------------------------------------------------
    required = {
        "DATA_DIR": DATA_DIR,
        "TEST_LABEL": TEST_LABEL,
        "CRNN_ONNX": CRNN_ONNX,
        "PPOCR_ONNX": PPOCR_ONNX,
        "REPSVTR_ONNX": REPSVTR_ONNX,
        "PPOCR_DICT": PPOCR_DICT,
        "REPSVTR_DICT": REPSVTR_DICT,
    }

    print(
        "\nPATH CHECK"
    )

    for name, path in required.items():
        if name == "DATA_DIR":
            ok = path.is_dir()
        else:
            ok = path.is_file()

        print(
            f"{name:14s}: "
            f"{str(ok):5s}  "
            f"{path}"
        )

        if not ok:
            raise FileNotFoundError(
                path
            )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Test set
    # --------------------------------------------------------
    test_samples = (
        load_test_samples(
            TEST_LABEL,
            DATA_DIR
        )
    )

    if (
        LIMIT == 0 and
        EXPECTED_SAMPLES > 0 and
        len(test_samples) !=
        EXPECTED_SAMPLES
    ):
        raise RuntimeError(
            f"Expected "
            f"{EXPECTED_SAMPLES} samples, "
            f"got {len(test_samples)}"
        )

    if LIMIT > 0:
        test_samples = (
            test_samples[:LIMIT]
        )

    print(
        "\nTest samples:",
        len(test_samples)
    )

    # --------------------------------------------------------
    # Dictionary
    # --------------------------------------------------------
    pp_chars = load_dict(
        PPOCR_DICT
    )

    rep_chars = load_dict(
        REPSVTR_DICT
    )

    if pp_chars != rep_chars:
        raise RuntimeError(
            "Dictionary PP-OCRv6 "
            "và RepSVTR không giống nhau."
        )

    if (
        CRNN_CHARACTERS[:-1]
        != pp_chars
    ):
        raise RuntimeError(
            "Dictionary Paddle hiện tại "
            "không khớp vocabulary CRNN "
            "trong notebook benchmark cũ."
        )

    paddle_characters = (
        pp_chars + [" "]
    )

    crnn_idx2char = {
        i + 1: c
        for i, c in enumerate(
            CRNN_CHARACTERS
        )
    }

    paddle_idx2char = {
        i + 1: c
        for i, c in enumerate(
            paddle_characters
        )
    }

    print(
        "Dictionary size:",
        len(pp_chars),
        "(không tính space)"
    )

    print(
        "CTC classes:",
        len(paddle_characters) + 1
    )

    print(
        "Dictionary compatible: OK"
    )

    # --------------------------------------------------------
    # ONNX sessions
    # --------------------------------------------------------
    print(
        "\nCreating ONNX sessions..."
    )

    crnn_session = (
        create_cpu_session(
            CRNN_ONNX
        )
    )

    pp_session = (
        create_cpu_session(
            PPOCR_ONNX
        )
    )

    rep_session = (
        create_cpu_session(
            REPSVTR_ONNX
        )
    )

    # --------------------------------------------------------
    # Preload ảnh trước khi timing
    # --------------------------------------------------------
    print(
        "\nPreloading images "
        "into RAM..."
    )

    crnn_images = (
        preload_crnn_images(
            test_samples
        )
    )

    paddle_images = (
        preload_paddle_images(
            test_samples
        )
    )

    # --------------------------------------------------------
    # CRNN
    # --------------------------------------------------------
    (
        crnn_result,
        crnn_preds,
        crnn_gts,
        crnn_times
    ) = benchmark_crnn(
        crnn_session,
        crnn_images,
        test_samples,
        crnn_idx2char
    )

    print_result(
        crnn_result
    )

    reference_check(
        crnn_result,
        98.2749
    )

    save_model_result(
        "crnn_onnx_cpu",
        test_samples,
        crnn_gts,
        crnn_preds,
        crnn_times,
        crnn_result
    )

    show_wrong_examples(
        crnn_result["Model"],
        test_samples,
        crnn_gts,
        crnn_preds,
        crnn_times
    )

    # --------------------------------------------------------
    # PP-OCRv6 Small Rec
    # --------------------------------------------------------
    (
        pp_result,
        pp_preds,
        pp_gts,
        pp_times
    ) = benchmark_paddle_onnx(
        "PP-OCRv6 Small Rec",
        pp_session,
        paddle_images,
        test_samples,
        paddle_idx2char
    )

    print_result(
        pp_result
    )

    reference_check(
        pp_result,
        99.0296
    )

    save_model_result(
        "ppocrv6_onnx_cpu",
        test_samples,
        pp_gts,
        pp_preds,
        pp_times,
        pp_result
    )

    show_wrong_examples(
        pp_result["Model"],
        test_samples,
        pp_gts,
        pp_preds,
        pp_times
    )

    # --------------------------------------------------------
    # RepSVTR
    # --------------------------------------------------------
    (
        rep_result,
        rep_preds,
        rep_gts,
        rep_times
    ) = benchmark_paddle_onnx(
        "RepSVTR",
        rep_session,
        paddle_images,
        test_samples,
        paddle_idx2char
    )

    print_result(
        rep_result
    )

    reference_check(
        rep_result,
        98.0593
    )

    save_model_result(
        "repsvtr_onnx_cpu",
        test_samples,
        rep_gts,
        rep_preds,
        rep_times,
        rep_result
    )

    show_wrong_examples(
        rep_result["Model"],
        test_samples,
        rep_gts,
        rep_preds,
        rep_times
    )

    # --------------------------------------------------------
    # Bảng chung
    # --------------------------------------------------------
    results = [
        crnn_result,
        pp_result,
        rep_result
    ]

    df = pd.DataFrame(
        results
    )

    main_cols = [
        "Model",
        "Exact Acc (%)",
        "CER (%)",
        "NED (%)",
        "Latency (ms)",
        "FPS",
    ]

    print(
        "\n" +
        "=" * 95
    )

    print(
        "FINAL COMPARISON - "
        "SAME ONNX RUNTIME CPU"
    )

    print(
        "=" * 95
    )

    print(
        df[
            main_cols
        ].round({
            "Exact Acc (%)": 4,
            "CER (%)": 4,
            "NED (%)": 4,
            "Latency (ms)": 4,
            "FPS": 2,
        }).to_string(
            index=False
        )
    )

    comparison_path = (
        OUTPUT_DIR /
        "recognition_comparison_onnx_cpu.csv"
    )

    df.to_csv(
        comparison_path,
        index=False,
        encoding="utf-8-sig"
    )

    summary_path = (
        OUTPUT_DIR /
        "benchmark_summary.json"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            {
                "runtime": (
                    "ONNX Runtime "
                    "CPUExecutionProvider"
                ),
                "onnxruntime_version":
                    ort.__version__,
                "cpu_threads":
                    CPU_THREADS,
                "batch_size": 1,
                "input_shape":
                    [3, 48, 320],
                "warmup":
                    WARMUP,
                "test_samples":
                    len(test_samples),
                "latency_scope": (
                    "decoded image in RAM "
                    "-> preprocess "
                    "-> ONNX inference "
                    "-> CTC decode; "
                    "disk I/O excluded"
                ),
                "models":
                    results,
            },
            f,
            ensure_ascii=False,
            indent=2
        )

    print(
        "\nSaved comparison:",
        comparison_path
    )

    print(
        "Saved summary   :",
        summary_path
    )

    print(
        "\nDONE."
    )


if __name__ == "__main__":
    main()
