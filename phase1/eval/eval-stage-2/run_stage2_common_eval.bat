@echo off
setlocal

REM ============================================================
REM STRICT CPU FAIR SPEED EVAL
REM Same device, batch=1, same CPU threads, oneDNN/MKLDNN OFF
REM ============================================================

set PY=.\pipeline_env\Scripts\python.exe

REM ===== SỬA CÁC PATH NÀY =====
set IMAGE_ROOT=D:\AI_Project\ocr_the_sv\src\eval_pipeline\eval-stage-2\dataset_det\test_images
set LABEL_FILE=D:\AI_Project\ocr_the_sv\src\eval_pipeline\eval-stage-2\dataset_det\test_label.txt
set PADDLEOCR_DIR=D:\AI_Project\ocr_the_sv\src\eval_pipeline\PaddleOCR

set YOLO_WEIGHTS=D:\AI_Project\ocr_the_sv\src\eval_pipeline\weights\stage-2\yolo26n-obb\best-stage2.pt
set DBNET_INFER=D:\AI_Project\ocr_the_sv\src\eval_pipeline\weights\stage-2\dbnet
set PPOCRV6_INFER=D:\AI_Project\ocr_the_sv\src\eval_pipeline\weights\stage-2\pp-ocrv6

REM ===== COMMON SETTINGS =====
set CPU_THREADS=4

%PY% eval-stage-2/eval_stage2_common_protocol.py ^
  --image-root "%IMAGE_ROOT%" ^
  --label-file "%LABEL_FILE%" ^
  --paddleocr-dir "%PADDLEOCR_DIR%" ^
  --yolo-weights "%YOLO_WEIGHTS%" ^
  --dbnet-infer "%DBNET_INFER%" ^
  --ppocrv6-infer "%PPOCRV6_INFER%" ^
  --iou-th 0.5 ^
  --ignore-iog 0.5 ^
  --yolo-conf 0.25 ^
  --yolo-imgsz 640 ^
  --db-thresh 0.30 ^
  --db-box-thresh 0.60 ^
  --db-unclip 1.50 ^
  --pp-thresh 0.20 ^
  --pp-box-thresh 0.45 ^
  --pp-unclip 1.40 ^
  --det-h 736 ^
  --det-w 1280 ^
  --cpu-threads %CPU_THREADS% ^
  --warmup 5 ^
  --output-dir output_stage2_fair_eval_strict

pause
