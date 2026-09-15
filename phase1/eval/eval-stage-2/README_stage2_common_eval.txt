STAGE 2 - FAIR COMMON EVALUATION

Mục tiêu:
So sánh YOLO26n-OBB, DBNet và PP-OCRv6 Small Det bằng CÙNG evaluator.

Protocol:
- Cùng ảnh test
- Cùng GT polygon
- Polygon IoU >= 0.5
- One-to-one matching
- Cùng công thức TP/FP/FN -> Precision/Recall/F1
- GT "###" được ignore

Input label:
PaddleOCR detection format:
image_path<TAB>[{"transcription":"...","points":[[x,y],...]}, ...]

Output:
1. summary_common_protocol.csv
   -> bảng chính để đưa vào báo cáo

2. per_image_common_protocol.csv
   -> TP/FP/FN, P/R/F1, latency từng ảnh từng model

3. evaluation_protocol.json
   -> lưu chính xác protocol/threshold đã dùng


Các default đang đặt theo setup hiện tại:
DBNet:
  thresh=0.30
  box_thresh=0.60
  unclip=1.50

PP-OCRv6 Small Det:
  thresh=0.20
  box_thresh=0.45
  unclip=1.40

YOLO26n-OBB:
  conf=0.25

Det resize Paddle:
  736 x 1280
