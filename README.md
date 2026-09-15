# EdgeCard

**EdgeCard** is a four-stage pipeline for student card detection and information recognition, designed for deployment on resource-constrained edge devices such as the Raspberry Pi 4.

The system combines card pose estimation, perspective rectification, oriented text detection, OCR, and configurable field-level post-processing. The final pipeline is evaluated on both a PC CPU and a Raspberry Pi 4 CPU.

## Overview

EdgeCard consists of two main phases:

- **Phase 1 — Offline Training and Model Selection**
  - Stage 1: card localization and four-corner keypoint estimation
  - Stage 2: text detection benchmark and model selection
  - Stage 3: text recognition benchmark and model selection

- **Phase 2 — End-to-End Inference**
  - Stage 1: card localization and perspective rectification
  - Stage 2: text detection and spatial assignment
  - Stage 3: text recognition
  - Stage 4: configurable post-processing and structured output

The final selected models are:

| Stage | Task | Selected model |
|---|---|---|
| 1 | Card localization / pose estimation | YOLO26n-Pose |
| 2 | Text detection | YOLO26n-OBB |
| 3 | Text recognition | PP-OCRv6 Small Recognition |
| 4 | Post-processing | Rule-based field processing |

## System Architecture

```text
Input student-card image
        |
        v
YOLO26n-Pose
4 semantic keypoints: TL, TR, BR, BL
        |
        v
Perspective rectification
Fixed output size: 856 x 540
        |
        v
YOLO26n-OBB
Oriented text-region detection
        |
        v
Spatial assignment using template ROIs
IoG threshold = 0.4
        |
        v
OBB padding + perspective crop
Input normalization to 3 x 48 x 320
        |
        v
PP-OCRv6 Small Recognition
CTC decoding
        |
        v
Field-specific post-processing
        |
        v
Structured final result
```

The spatial layout is configured through `student_card_template.json`, allowing the field definitions, card size, ROIs, IoG threshold, and post-processing rules to be changed without modifying the deep-learning models.

## Project Structure

```text
.
├── README.md
├── .gitignore
│
├── phase1/
│   ├── finetune/
│   │   ├── stage-1/
│   │   │   └── stage_1_finetune_yolo.ipynb
│   │   ├── stage-2/
│   │   │   ├── yolo26n_obb.ipynb
│   │   │   ├── DBNet_eval.ipynb
│   │   │   └── PP_OCRv6_det.ipynb
│   │   └── stage-3/
│   │       ├── mobilenetv3_crnn3.ipynb
│   │       ├── PP_OCRv6_rec.ipynb
│   │       └── repsvtrv2.ipynb
│   └── eval/
│       ├── eval-stage-2/
│       │   ├── eval_stage2_common_protocol.py
│       │   ├── README_stage2_common_eval.txt
│       │   └── run_stage2_common_eval.bat
│       └── eval-stage-3/
│           ├── evaluate_stage3_3models_onnx_cpu.py
│           └── output/
│
└── phase2/
    └── E2E-inference/
        ├── eval_pipeline_pc.py
        ├── eval_pipeline_pi.py
        ├── requirements_pc_cpu.txt
        ├── run_eval_pc.bat
        ├── metadata/
        │   └── student_card_template.json
        ├── reproducibility/
        │   └── studentcard_vi_dict.txt
        └── result/
            ├── final_field_summary.csv
            ├── latency_summary.csv
            └── summary.json
```

Model weights are distributed separately through Hugging Face. Raw student-card images, annotations, ground-truth files, and sensitive per-sample prediction files are not publicly released.

## Phase 1: Offline Training and Model Selection

### Stage 1 — Card Pose Estimation

YOLO26n-Pose is trained to detect the student card and predict four semantic keypoints in the fixed order:

```text
TL -> TR -> BR -> BL
```

These points are used directly for perspective rectification.

Training notebook:

```text
phase1/finetune/stage-1/stage_1_finetune_yolo.ipynb
```

### Stage 2 — Text Detection

Three text detectors are trained and evaluated:

- YOLO26n-OBB
- DBNet
- PP-OCRv6 Small Det

Training notebooks:

```text
phase1/finetune/stage-2/yolo26n_obb.ipynb
phase1/finetune/stage-2/DBNet_eval.ipynb
phase1/finetune/stage-2/PP_OCRv6_det.ipynb
```

Common evaluation script:

```text
phase1/eval/eval-stage-2/eval_stage2_common_protocol.py
```

The evaluation uses polygon-based one-to-one matching with `IoU >= 0.5`.

### Stage 3 — Text Recognition

Three recognizers are compared:

- MobileNetV3-CRNN
- PP-OCRv6 Small Recognition
- RepSVTR

Training notebooks:

```text
phase1/finetune/stage-3/mobilenetv3_crnn3.ipynb
phase1/finetune/stage-3/PP_OCRv6_rec.ipynb
phase1/finetune/stage-3/repsvtrv2.ipynb
```

For a fair CPU-speed comparison, the three exported recognizers are evaluated using the same ONNX Runtime CPU environment:

```text
phase1/eval/eval-stage-3/evaluate_stage3_3models_onnx_cpu.py
```

Recognition metrics include:

- Exact Accuracy
- Character Error Rate (CER)
- Normalized Edit Similarity (NES)
- Mean latency
- Throughput

NES is computed from normalized edit distance as a similarity score, so values closer to `100%` are better.

## Phase 2: End-to-End Inference

The complete pipeline is implemented for both PC and Raspberry Pi 4.

### PC

```text
phase2/E2E-inference/eval_pipeline_pc.py
```

Runtime:

- YOLO models: PyTorch / Ultralytics
- OCR recognizer: PaddlePaddle
- CPU-only end-to-end evaluation

### Raspberry Pi 4

```text
phase2/E2E-inference/eval_pipeline_pi.py
```

Runtime:

- YOLO26n-Pose: NCNN
- YOLO26n-OBB: NCNN
- PP-OCRv6 Small Recognition: ONNX Runtime
- CPU-only inference

The same overall processing logic and template configuration are retained across PC and Raspberry Pi 4.

## Spatial Assignment

Text detections are represented as oriented polygons.

Each detected text region is assigned to a target field using Intersection over Ground (IoG):

```text
IoG(B, R) = Area(B ∩ R) / Area(B)
```

where:

- `B` is the detected text polygon
- `R` is the field ROI

The current threshold is:

```text
IoG >= 0.4
```

Detections that do not belong to any target ROI are discarded before recognition.

If multiple detections are assigned to the same field, they are sorted in reading order from top to bottom and then left to right.

## Text Recognition Pre-processing

Before OCR:

1. The selected OBB is expanded by approximately 8% on each side.
2. The padded polygon is perspective-rectified into a rectangular crop.
3. The crop is normalized to the recognizer input size `3 x 48 x 320`.
4. The output sequence is decoded using CTC.

The original OBB polygon is used for IoG assignment; padding is applied only to the recognition crop.

## Post-processing

Field-specific post-processing is configured through reusable rules:

- `free_text`
- `identifier`
- `year_range`
- `date`
- `dictionary_text`
- `enum`

The current student-card configuration uses:

| Field | Rule |
|---|---|
| `name` | `free_text` |
| `course` | `year_range` |
| `class` | `identifier` |
| `student_id` | `identifier` |

Depending on the selected rule, post-processing can perform string normalization, letter-digit confusion correction, pattern/regular-expression validation, candidate matching, and Levenshtein-distance-based correction when a valid candidate set is available.

## Datasets

The experimental data were collected from student cards issued by the **Academy of Cryptography Techniques (ACTVN)** and are organized into four subsets.

| Dataset | Purpose | Size |
|---|---|---:|
| D1 | Card pose estimation | 1,237 images / 221 student IDs |
| D2 | Text detection | 1,228 images / 220 student IDs |
| D3 | Text recognition | 18,231 text crops |
| Dataset-E2E | End-to-end evaluation | 126 card images |

The data are split at the student-ID level to ensure that all images belonging to the same student appear in only one split, thereby preventing data leakage between the training, validation, and test sets.

Due to privacy and data-protection considerations, the raw student-card images, annotations, and end-to-end ground-truth files are **not publicly released** in this repository.

For reference, a small set of representative student-card images is provided at the following link to illustrate the visual characteristics and layout of the data used in this study:

**Sample images:** [View representative student-card images](https://drive.google.com/drive/folders/1p0hIUrsuHtTDTiSAAe6VnqwPQyCpQxXC?usp=drive_link)

Researchers who wish to reproduce the experiments or request access to the dataset may contact the corresponding authors for further information and data-access arrangements.

> **Privacy note:** The dataset contains personally identifiable information, including student names, student identifiers, and class information. Public redistribution is therefore restricted. The publicly shared sample images are provided only with appropriate authorization.

## Model Weights

Pretrained and fine-tuned model weights are distributed separately through Hugging Face to avoid storing large binary files in the Git history.

The final EdgeCard pipeline uses:

- **Stage 1:** YOLO26n-Pose
- **Stage 2:** YOLO26n-OBB
- **Stage 3:** PP-OCRv6 Small Recognition

Weights for the alternative models evaluated in the benchmarking experiments are also provided for reproducibility, including DBNet, PP-OCRv6 Small Detection, MobileNetV3-CRNN, and RepSVTR.

Available deployment formats include PyTorch (`.pt`, `.pth`), ONNX (`.onnx`), NCNN (`.param`, `.bin`), and Paddle inference formats where applicable.

**Model weights:** [Hugging Face](https://huggingface.co/Hoang17z/EdgeCard-OCR-Models)

## Installation

### PC end-to-end environment

From the repository root:

```bash
cd phase2/E2E-inference
pip install -r requirements_pc_cpu.txt
```

The end-to-end scripts expect the required model weights and evaluation data to be available locally.

Because dataset and weight files are distributed separately, update the corresponding local paths before running an evaluation.

## Running End-to-End Evaluation

### Windows PC

A helper script is provided:

```text
phase2/E2E-inference/run_eval_pc.bat
```

or run the Python evaluator directly after configuring the required paths:

```bash
python phase2/E2E-inference/eval_pipeline_pc.py
```

### Raspberry Pi 4

After installing the required NCNN / ONNX Runtime dependencies and downloading the exported models:

```bash
python phase2/E2E-inference/eval_pipeline_pi.py
```

## Experimental Results

### Stage 1 — YOLO26n-Pose

| Metric | Result |
|---|---:|
| Precision | 99.60% |
| Recall | 98.50% |
| mAP@0.5 | 99.43% |
| mAP@0.5:0.95 | 99.22% |
| Latency | 7.40 ms/image on Tesla T4 |

### Stage 2 — Text Detection

| Model | Precision | Recall | F1 | Matched IoU | Latency |
|---|---:|---:|---:|---:|---:|
| **YOLO26n-OBB** | **96.31%** | **98.28%** | **97.28%** | **79.04%** | **122.36 ms** |
| DBNet | 88.69% | 95.42% | 91.93% | 76.41% | 1244.37 ms |
| PP-OCRv6 Small Det | 92.18% | 93.91% | 93.04% | 73.07% | 1973.40 ms |

YOLO26n-OBB is selected for the final pipeline.

### Stage 3 — Text Recognition

| Model | Exact Accuracy | CER | NES | Latency | Throughput |
|---|---:|---:|---:|---:|---:|
| MobileNetV3-CRNN | 98.22% | 0.26% | 99.73% | 11.18 ms | 89.42 crop/s |
| **PP-OCRv6 Small Rec** | **99.03%** | **0.11%** | **99.91%** | **10.24 ms** | **97.66 crop/s** |
| RepSVTR | 98.06% | 0.20% | 99.83% | 12.57 ms | 79.55 crop/s |

PP-OCRv6 Small Recognition is selected for the final pipeline.

### End-to-End Results

| Metric | PC | Raspberry Pi 4 |
|---|---:|---:|
| Pipeline Success | 100.00% | 100.00% |
| Complete Output | 92.86% | 92.86% |
| Overall Coverage | 96.63% | 96.23% |
| Overall Exact Accuracy | 94.25% | 94.25% |
| Overall CER | 3.19% | 4.15% |
| Card Exact Accuracy | 86.51% | 88.10% |

Field-level Exact Accuracy:

| Field | PC | Raspberry Pi 4 |
|---|---:|---:|
| `name` | 89.68% | 89.68% |
| `course` | 95.24% | 95.24% |
| `class` | 95.24% | 96.03% |
| `student_id` | 96.83% | 96.03% |

### End-to-End Latency

| Stage | PC | Raspberry Pi 4 |
|---|---:|---:|
| Stage 1 | 102.46 ms | 387.94 ms |
| Stage 2 | 90.86 ms | 358.45 ms |
| Stage 3 | 305.05 ms | 359.32 ms |
| Stage 4 | 0.05 ms | 0.09 ms |
| **Total** | **498.44 ms** | **1105.82 ms** |

The resulting throughput is approximately:

- **PC:** 2.01 cards/s
- **Raspberry Pi 4:** 0.90 cards/s

Peak Resident Set Size on Raspberry Pi 4 is approximately **795 MiB**.

## Reproducibility

The repository includes training notebooks for all benchmarked models, common Stage 2 evaluation code, common Stage 3 ONNX Runtime benchmark code, end-to-end PC evaluation code, Raspberry Pi 4 inference/evaluation code, template configuration, OCR character dictionary, and aggregate result files.

The trained model weights used in the experiments are distributed separately through the Hugging Face Model Hub.

Due to privacy and data-protection considerations, the following resources are not publicly released:

- raw student-card images
- detection and recognition annotations
- end-to-end ground truth
- per-sample prediction files containing potentially sensitive information

For reproducibility purposes, qualified researchers may contact the authors regarding possible access to the restricted data, subject to applicable institutional and privacy requirements.

## Privacy

The student-card dataset contains personally identifiable information, including student names, student identifiers, and class information.

Therefore, raw card images, annotations, end-to-end ground-truth files, and other records containing identifiable student information are not publicly distributed. Access to such data is subject to appropriate authorization, institutional requirements, and privacy-protection procedures.

## Paper

This repository accompanies the EdgeCard research project:

**EdgeCard: Student Card Detection and Recognition on Low-Power Devices**

Publication information will be added after acceptance/publication.

## Citation

Citation information will be updated after publication.

```bibtex
@misc{edgecard,
  title  = {EdgeCard: Student Card Detection and Recognition on Low-Power Devices},
  author = {Le Duc Thuan and Vu Thi Linh and Nguyen Huy Hoang and Duong Thi Thu Trang},
  year   = {2026},
  note   = {Manuscript}
}
```

## License

A project license has not yet been specified.

