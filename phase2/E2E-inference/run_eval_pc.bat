@echo off

setlocal

REM ============================================================
REM EdgeCard END-TO-END
REM Stage 1: YOLO26n-Pose
REM Stage 2: YOLO26n-OBB + Spatial ROI/IoG
REM Stage 3: PP-OCRv6 Small Rec
REM Stage 4: Configurable field-aware post-processing
REM OBB crop padding: 8% moi phia
REM ============================================================

set POSE_WEIGHTS=D:\AI_Project\ocr_the_sv\src\eval_pipeline\weights\stage-1\best-stage1.pt
set OBB_WEIGHTS=D:\AI_Project\ocr_the_sv\src\eval_pipeline\weights\stage-2\yolo26n-obb\best-stage2.pt
set RAW_TEST_DIR=D:\AI_Project\ocr_the_sv\src\eval_pipeline\test-card
set TEMPLATE_JSON=D:\AI_Project\ocr_the_sv\src\eval_pipeline\metadata\student_card_template.json
set GT_JSON=D:\AI_Project\ocr_the_sv\src\eval_pipeline\metadata\pipeline_gt_complete.json
set PADDLEOCR_DIR=D:\AI_Project\ocr_the_sv\src\eval_pipeline\PaddleOCR
set REC_INFER=D:\AI_Project\ocr_the_sv\src\eval_pipeline\weights\stage-3\pp-ocrv6
set REC_DICT=D:\AI_Project\ocr_the_sv\src\eval_pipeline\weights\stage-3\pp-ocrv6\reproducibility\studentcard_vi_dict.txt
set OUTPUT_DIR=D:\AI_Project\ocr_the_sv\src\eval_pipeline\output_yolo_obb_padding_6rules

python eval_pipeline_pc.py ^
  --pose-weights "%POSE_WEIGHTS%" ^
  --obb-weights "%OBB_WEIGHTS%" ^
  --obb-conf 0.25 ^
  --obb-imgsz 640 ^
  --obb-padding 0.08 ^
  --raw-test-dir "%RAW_TEST_DIR%" ^
  --template-json "%TEMPLATE_JSON%" ^
  --ground-truth-json "%GT_JSON%" ^
  --paddleocr-dir "%PADDLEOCR_DIR%" ^
  --rec-infer "%REC_INFER%" ^
  --rec-dict "%REC_DICT%" ^
  --output-dir "%OUTPUT_DIR%" ^
  --mkldnn ^
  --save-vis

pause