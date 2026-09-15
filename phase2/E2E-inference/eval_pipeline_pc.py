#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
EdgeCard - End-to-End Pipeline Evaluation on PC CPU (PIR-compatible)
Python 3.10

Pipeline thật:
Ảnh gốc -> YOLO26n-Pose -> Perspective Warp
-> YOLO26n-OBB -> Spatial ROI/IoG
-> PP-OCRv6 Small Rec -> Post-process -> kết quả cuối

File JSON dùng:
1) template_json:
   - card_size
   - ROI 4 trường name/course/class/student_id
   - IoG threshold

2) ground_truth_json:
   {
     "AT200210_03_jpg.rf.xxxxx.jpg": {
       "name": "NGUYỄN QUANG ĐẠT",
       "course": "2023-2028",
       "class": "AT20B",
       "student_id": "AT200210"
     },
     ...
   }

Chỉ đánh giá END-TO-END:
- Field Exact Accuracy
- CER
- Detection Coverage
- Card Exact Accuracy
- Pipeline Success / Complete Output
- Latency Stage 1/2/3/4 + Total
"""

import os,sys,re,json,time,argparse,unicodedata
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO
import paddle
from paddle import inference

IMG_EXT={".jpg",".jpeg",".png",".bmp",".webp"}
DIGIT_MAP=str.maketrans({"O":"0","Q":"0","I":"1","L":"1","Z":"2","S":"5","B":"8","G":"6"})
LETTER_MAP=str.maketrans({"0":"O","1":"I"})

def parse_args():
    p=argparse.ArgumentParser("EdgeCard end-to-end evaluation - PC CPU")
    p.add_argument("--pose-weights",required=True)
    p.add_argument("--raw-test-dir",required=True)
    p.add_argument("--template-json",required=True)
    p.add_argument("--ground-truth-json",required=True)
    p.add_argument("--paddleocr-dir",required=True)
    p.add_argument("--obb-weights",required=True,help="YOLO26n-OBB Stage 2 weights (.pt)")
    p.add_argument("--obb-conf",type=float,default=0.25)
    p.add_argument("--obb-imgsz",type=int,default=640)
    p.add_argument(
        "--obb-padding",
        type=float,
        default=0.08,
        help="Padding cho OBB trước OCR; 0.08 = thêm 8%% mỗi phía"
    )
    p.add_argument("--rec-infer",required=True)
    p.add_argument("--rec-dict",required=True)
    p.add_argument("--postprocess-dict",default="",help="Optional JSON từ điển hợp lệ; không tạo từ test GT")
    p.add_argument("--output-dir",default="pipeline_eval_pc_yolo_obb")
    p.add_argument("--pose-conf",type=float,default=0.25)
    p.add_argument("--iog-th",type=float,default=-1,help="<0: lấy threshold từ template JSON")
    p.add_argument("--cpu-threads",type=int,default=max(1,min(4,os.cpu_count() or 4)))
    p.add_argument("--mkldnn",action="store_true")
    p.add_argument("--save-vis",action="store_true")
    p.add_argument("--vis-limit",type=int,default=20)
    p.add_argument("--warmup",type=int,default=2)
    p.add_argument("--limit",type=int,default=0)
    return p.parse_args()

def norm(s):return " ".join(unicodedata.normalize("NFC",str(s)).strip().split())

def edit_distance(a,b):
    a,b=norm(a),norm(b)
    prev=list(range(len(b)+1))
    for i,ca in enumerate(a,1):
        cur=[i]
        for j,cb in enumerate(b,1):
            cur.append(min(cur[-1]+1,prev[j]+1,prev[j-1]+(ca!=cb)))
        prev=cur
    return prev[-1]

def order_quad(pts):
    pts=np.asarray(pts,dtype=np.float32).reshape(4,2)
    s=pts.sum(1);d=np.diff(pts,axis=1).ravel()
    return np.array([pts[np.argmin(s)],pts[np.argmin(d)],pts[np.argmax(s)],pts[np.argmax(d)]],np.float32)

def roi_poly(roi):
    x1,y1,x2,y2=map(float,roi)
    return np.array([[x1,y1],[x2,y1],[x2,y2],[x1,y2]],np.float32)

def poly_iog(det,roi):
    a=cv2.convexHull(np.asarray(det,np.float32))
    b=cv2.convexHull(np.asarray(roi,np.float32))
    area=float(abs(cv2.contourArea(a)))
    if area<=0:return 0.0
    inter,_=cv2.intersectConvexConvex(a,b)
    return float(inter)/area

def model_files(model_dir):
    model_dir=Path(model_dir)
    for name in ("inference.json","inference.pdmodel"):
        p=model_dir/name
        if p.exists():
            params=model_dir/"inference.pdiparams"
            if not params.exists():raise FileNotFoundError(params)
            return str(p),str(params)
    raise FileNotFoundError(f"Không thấy inference.json/pdmodel trong {model_dir}")

def create_cpu_predictor(model_dir,threads=4,mkldnn=False):
    mf,pf=model_files(model_dir)

    print(f"Load model : {mf}")
    print(f"Load params: {pf}")

    cfg=inference.Config(mf,pf)
    cfg.disable_gpu()
    cfg.set_cpu_math_library_num_threads(int(threads))

    if mkldnn:
        try:
            cfg.enable_mkldnn()
            print("oneDNN/MKLDNN: ON")
        except Exception as e:
            print("Warning MKLDNN:",e)

    # Không dùng enable_memory_optim() với Paddle 3.x + PIR inference.json.
    # Trên Windows có thể gây:
    # "Not find predictor_id ... pass_name memory_optimize_pass".
    return inference.create_predictor(cfg)

def run_predictor(pred,x):
    names=pred.get_input_names()
    h=pred.get_input_handle(names[0])
    h.reshape(x.shape);h.copy_from_cpu(x);pred.run()
    return [pred.get_output_handle(n).copy_to_cpu() for n in pred.get_output_names()]

def load_gt(path):
    with open(path,"r",encoding="utf-8") as f:data=json.load(f)
    if not isinstance(data,dict):raise ValueError("Ground truth JSON phải là object/dict")
    gt={}
    for k,v in data.items():
        if not isinstance(v,dict):continue
        # Giữ toàn bộ field trong GT; danh sách field cần chấm sẽ lấy từ template JSON.
        gt[os.path.basename(k)]=dict(v)
    return gt

def match_gt(raw_name,gt_keys):
    keys=list(gt_keys)
    if raw_name in keys:return raw_name

    stem,ext=os.path.splitext(raw_name)
    ext_token=ext.lower().lstrip(".")

    # Ví dụ ảnh gốc:
    # AT210539_01.jpg
    # -> GT:
    # AT210539_01_jpg.rf.HASH.jpg
    pref=f"{stem}_{ext_token}.rf.".lower()
    cand=[k for k in keys if k.lower().startswith(pref)]
    if len(cand)==1:return cand[0]

    # Nếu ảnh raw đã chứa token _jpg/_png nhưng bỏ rf.hash
    pref=f"{stem}.rf.".lower()
    cand=[k for k in keys if k.lower().startswith(pref)]
    if len(cand)==1:return cand[0]

    return None

def load_post_dict(path):
    if not path:return {}
    with open(path,"r",encoding="utf-8") as f:return json.load(f)

def nearest(value,cands,max_dist=1):
    if not value or not cands:return value
    best=min((edit_distance(value,x),x) for x in cands)
    return best[1] if best[0]<=max_dist else value

def _dict_candidates(dic,field,cfg,default_key=None):
    key=cfg.get("dict_key") or default_key
    if key and isinstance(dic.get(key),list):
        return dic.get(key,[])
    if isinstance(dic.get(field),list):
        return dic.get(field,[])
    plural=f"{field}s"
    if isinstance(dic.get(plural),list):
        return dic.get(plural,[])
    return []

def _apply_identifier_pattern(value,pattern):
    if not pattern:
        return value
    out=[]
    for i,ch in enumerate(value):
        if i>=len(pattern):
            out.append(ch)
            continue
        token=pattern[i].upper()
        if token=="L":
            out.append(ch.translate(LETTER_MAP))
        elif token=="D":
            out.append(ch.translate(DIGIT_MAP))
        else:
            out.append(ch)
    return "".join(out)

def postprocess(field,text,dic,config=None):
    s=norm(text)
    cfg=config if isinstance(config,dict) else {}
    rule=config if isinstance(config,str) else cfg.get("postprocess","auto")
    rule=str(rule or "auto").strip().lower()

    # JSON cũ không khai báo postprocess vẫn giữ hành vi tương đương cho 4 field hiện tại.
    auto_defaults={
        "name":{"postprocess":"dictionary_text","dict_key":"names"},
        "course":{"postprocess":"year_range","dict_key":"courses"},
        "class":{"postprocess":"identifier","dict_key":"classes"},
        "student_id":{"postprocess":"identifier","dict_key":"student_ids","pattern":"LLDDDDDD"}
    }
    if rule=="auto":
        base=auto_defaults.get(str(field).lower(),{"postprocess":"free_text"})
        cfg={**base,**cfg}
        rule=str(cfg.get("postprocess","free_text")).strip().lower()

    # Tương thích với tên rule cũ; bên trong vẫn quy về 6 rule chung.
    if rule=="name":
        cfg={"dict_key":"names",**cfg}
        rule="dictionary_text"
    elif rule in ("class","alnum_upper"):
        cfg={"dict_key":"classes",**cfg}
        rule="identifier"
    elif rule=="student_id":
        cfg={"dict_key":"student_ids","pattern":"LLDDDDDD",**cfg}
        rule="identifier"
    elif rule=="course":
        cfg={"dict_key":"courses",**cfg}
        rule="year_range"

    # Rule 1: free_text / normalize — fallback mặc định, không ép cấu trúc.
    if rule in ("free_text","normalize","none","pass","passthrough"):
        return s

    # Rule 2: identifier — mã SV, lớp, khóa dạng K20, mã định danh...
    if rule=="identifier":
        extra=str(cfg.get("extra_chars",cfg.get("allowed_extra_chars","")))
        allowed_extra=set(extra)
        x="".join(ch for ch in s.upper() if ch.isalnum() or ch in allowed_extra)
        x=_apply_identifier_pattern(x,str(cfg.get("pattern","")).strip())
        cands=_dict_candidates(dic,field,cfg)
        return nearest(x,cands,int(cfg.get("max_dist",1))) if cands else x

    # Rule 3: year_range — khoảng năm như 2023-2026 hoặc 2023 - 2026.
    if rule=="year_range":
        x=s.upper().replace("/","-").replace("–","-").replace("—","-").translate(DIGIT_MAP)
        m=re.search(r"(\d{4})(\s*-\s*)(\d{4})",x)
        if m:
            style=str(cfg.get("separator_style","preserve")).lower()
            if style=="compact":sep="-"
            elif style=="spaced":sep=" - "
            else:sep=m.group(2)
            x=f"{m.group(1)}{sep}{m.group(3)}"
        default_key="courses" if str(field).lower()=="course" else None
        cands=_dict_candidates(dic,field,cfg,default_key)
        return nearest(x,cands,int(cfg.get("max_dist",1))) if cands else x

    # Rule 4: date — ngày sinh/ngày cấp/ngày hết hạn; chỉ sửa khi ngày hợp lệ.
    if rule=="date":
        x=s.upper().translate(DIGIT_MAP)
        m=re.search(r"(\d{1,2})\s*([./-])\s*(\d{1,2})\s*[./-]\s*(\d{4})",x)
        if not m:
            return s
        day,month,year=int(m.group(1)),int(m.group(3)),int(m.group(4))
        try:
            import datetime
            datetime.date(year,month,day)
        except ValueError:
            return s
        sep=str(cfg.get("separator",m.group(2)))
        zero_pad=bool(cfg.get("zero_pad",False))
        dd=f"{day:02d}" if zero_pad else str(day)
        mm=f"{month:02d}" if zero_pad else str(month)
        return f"{dd}{sep}{mm}{sep}{year:04d}"

    # Rule 5: dictionary_text — văn bản có tập giá trị hợp lệ biết trước.
    if rule=="dictionary_text":
        cands=_dict_candidates(dic,field,cfg)
        return nearest(s,cands,int(cfg.get("max_dist",1))) if cands else s

    # Rule 6: enum — tập giá trị nhỏ, match không phân biệt hoa/thường.
    if rule=="enum":
        vals=cfg.get("values",[])
        if not isinstance(vals,list) or not vals:
            vals=_dict_candidates(dic,field,cfg)
        if not vals:
            return s
        for v in vals:
            if norm(v).casefold()==s.casefold():
                return norm(v)
        max_dist=int(cfg.get("max_dist",1))
        best=min((edit_distance(s.casefold(),norm(v).casefold()),norm(v)) for v in vals)
        return best[1] if best[0]<=max_dist else s

    # Rule không nhận diện: an toàn nhất là chỉ normalize và giữ output OCR.
    return s

class Pipeline:
    def __init__(self,args,tpl):
        self.args=args
        self.card_w=int(tpl["card_size"]["width"])
        self.card_h=int(tpl["card_size"]["height"])
        self.field_rois={k:v["roi"] for k,v in tpl["fields"].items()}
        # Danh sách field lấy trực tiếp từ template JSON, không hardcode 4 field cố định.
        self.fields=tuple(self.field_rois.keys())
        # Giữ toàn bộ cấu hình field để Stage 4 có thể dùng 6 rule hậu xử lý.
        # JSON cũ không có postprocess vẫn chạy nhờ chế độ "auto".
        self.field_configs={k:dict(v) for k,v in tpl["fields"].items()}
        self.iog_th=float(tpl.get("spatial_filter",{}).get("threshold",0.5)) if args.iog_th<0 else args.iog_th

        sys.path.insert(0,args.paddleocr_dir)
        from ppocr.data.imaug.rec_img_aug import resize_norm_img
        from ppocr.postprocess import build_post_process

        self.resize_norm_img=resize_norm_img
        self.rec_decoder=build_post_process({
            "name":"CTCLabelDecode",
            "character_dict_path":args.rec_dict,
            "use_space_char":True
        })

        # Stage 1: YOLO26n-Pose
        self.pose=YOLO(args.pose_weights)

        # Stage 2: YOLO26n-OBB
        self.obb=YOLO(args.obb_weights)

        # Stage 3: PP-OCRv6 Small Rec
        self.rec=create_cpu_predictor(args.rec_infer,args.cpu_threads,args.mkldnn)
        self.rec_shape=[3,48,320]

    def stage1(self,img):
        r=self.pose.predict(
            img,
            conf=self.args.pose_conf,
            device="cpu",
            verbose=False
        )[0]

        if r.keypoints is None or len(r.keypoints.xy)==0:
            return None

        idx=int(r.boxes.conf.argmax().item()) if r.boxes is not None and len(r.boxes)>0 else 0

        # Ultralytics: shape thường là (num_keypoints, 2).
        # Ép rõ về ndarray float32 C-contiguous để OpenCV getPerspectiveTransform
        # không lỗi checkVector trên một số bản OpenCV 5.x / Windows.
        pts=r.keypoints.xy[idx].detach().cpu().numpy()
        pts=np.asarray(pts,dtype=np.float32)

        if pts.ndim!=2 or pts.shape[0]<4 or pts.shape[1]<2:
            print(f"[Stage1] Invalid keypoint shape: {pts.shape}")
            return None

        # QUAN TRỌNG:
        # Giữ NGUYÊN thứ tự semantic đã train:
        # 1=top-left, 2=top-right, 3=bottom-right, 4=bottom-left của THẺ.
        # Không dùng order_quad() ở Stage 1.
        src=np.ascontiguousarray(pts[:4,:2],dtype=np.float32)

        dst=np.ascontiguousarray([
            [0,0],
            [self.card_w-1,0],
            [self.card_w-1,self.card_h-1],
            [0,self.card_h-1]
        ],dtype=np.float32)

        if src.shape!=(4,2):
            print(f"[Stage1] Invalid src shape: {src.shape}")
            return None

        M=cv2.getPerspectiveTransform(src,dst)
        return cv2.warpPerspective(img,M,(self.card_w,self.card_h))

    def stage2(self,card):
        """
        Stage 2:
        YOLO26n-OBB detect text regions trên ảnh card đã warp.
        OBB được lấy dưới dạng 4 điểm polygon (pixel coordinates),
        sau đó dùng cùng ROI template + IoG như pipeline cũ.
        """
        r=self.obb.predict(
            card,
            conf=self.args.obb_conf,
            imgsz=self.args.obb_imgsz,
            device="cpu",
            verbose=False
        )[0]

        boxes=[]
        if r.obb is not None and r.obb.xyxyxyxy is not None and len(r.obb.xyxyxyxy)>0:
            polys=r.obb.xyxyxyxy.detach().cpu().numpy()
            boxes=[
                np.ascontiguousarray(
                    np.asarray(b,dtype=np.float32).reshape(4,2)
                )
                for b in polys
            ]

        assigned={f:[] for f in self.fields}
        for f in self.fields:
            rp=roi_poly(self.field_rois[f])
            for b in boxes:
                score=poly_iog(b,rp)
                if score>=self.iog_th:
                    assigned[f].append((b,score))

            # Giữ thứ tự đọc từ trên xuống, trái sang phải nếu có nhiều box.
            assigned[f].sort(
                key=lambda z:(
                    float(np.mean(z[0][:,1])),
                    float(np.mean(z[0][:,0]))
                )
            )

        return boxes,assigned

    def expand_quad(self,pts,img_shape,padding=None):
        """
        Nới OBB trước khi crop cho OCR recognition.

        padding=0.08 nghĩa là thêm khoảng 8% ở MỖI phía,
        tức tổng width/height tăng khoảng 16%.

        Chỉ dùng polygon đã nới để CROP cho Stage 3.
        Polygon gốc vẫn được dùng cho ROI/IoG ở Stage 2,
        nên không làm thay đổi logic spatial assignment.
        """
        p=np.asarray(pts,dtype=np.float32).reshape(4,2)

        pad=self.args.obb_padding if padding is None else float(padding)
        if pad<=0:
            return np.ascontiguousarray(p,dtype=np.float32)

        center=np.mean(p,axis=0,keepdims=True)

        # +pad mỗi phía => kích thước tổng tăng ~2*pad.
        scale=1.0 + 2.0*pad
        q=center + (p-center)*scale

        h,w=img_shape[:2]
        q[:,0]=np.clip(q[:,0],0,max(w-1,0))
        q[:,1]=np.clip(q[:,1],0,max(h-1,0))

        return np.ascontiguousarray(q,dtype=np.float32)

    def crop_quad(self,img,pts):
        p=order_quad(pts)
        w=max(int(np.linalg.norm(p[0]-p[1])),int(np.linalg.norm(p[2]-p[3])),2)
        h=max(int(np.linalg.norm(p[0]-p[3])),int(np.linalg.norm(p[1]-p[2])),2)
        dst=np.array([[0,0],[w-1,0],[w-1,h-1],[0,h-1]],np.float32)
        crop=cv2.warpPerspective(img,cv2.getPerspectiveTransform(p,dst),(w,h))
        if crop.shape[0]/max(crop.shape[1],1)>=1.5:crop=np.rot90(crop)
        return np.ascontiguousarray(crop)

    def recognize(self,crop):
        x,_=self.resize_norm_img(crop,self.rec_shape,padding=True)
        outs=run_predictor(self.rec,x[np.newaxis,:].copy())
        raw=outs[0] if len(outs)==1 else outs
        res=self.rec_decoder(raw)
        if not res:return "",0.0
        return norm(res[0][0]),float(res[0][1])

    def stage3(self,card,assigned):
        raw={};conf={}
        for f in self.fields:
            texts=[];scores=[]
            for b,_ in assigned[f]:
                # Padding chỉ áp dụng khi crop cho OCR recognition.
                # ROI/IoG vẫn dùng OBB gốc để tránh kéo text lân cận vào field.
                b_crop=self.expand_quad(b,card.shape)
                crop=self.crop_quad(card,b_crop)
                txt,sc=self.recognize(crop)
                if txt:texts.append(txt)
                scores.append(sc)
            raw[f]=norm(" ".join(texts))
            conf[f]=float(np.mean(scores)) if scores else 0.0
        return raw,conf

    def draw(self,card,boxes):
        v=card.copy()
        for f,roi in self.field_rois.items():
            x1,y1,x2,y2=map(int,roi)
            cv2.rectangle(v,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(v,f,(x1,max(15,y1-4)),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,255,0),1,cv2.LINE_AA)
        for b in boxes:
            cv2.polylines(v,[b.astype(np.int32)],True,(255,0,0),1)
        return v

def main():
    args=parse_args()
    out=Path(args.output_dir)
    out.mkdir(parents=True,exist_ok=True)

    required=[
        args.pose_weights,args.obb_weights,args.raw_test_dir,args.template_json,args.ground_truth_json,
        args.paddleocr_dir,args.rec_infer,args.rec_dict
    ]
    for p in required:
        if not Path(p).exists():raise FileNotFoundError(p)

    with open(args.template_json,"r",encoding="utf-8") as f:tpl=json.load(f)
    gt=load_gt(args.ground_truth_json)
    post_dic=load_post_dict(args.postprocess_dict)
    pipe=Pipeline(args,tpl)
    fields=pipe.fields

    images=sorted([
        p for p in Path(args.raw_test_dir).rglob("*")
        if p.suffix.lower() in IMG_EXT
    ])
    if args.limit>0:images=images[:args.limit]
    if not images:raise RuntimeError("Không tìm thấy ảnh test")

    print("="*72)
    print("EdgeCard END-TO-END Evaluation - PC CPU")
    print("Python:",sys.version.split()[0])
    print("Paddle:",paddle.__version__)
    print("Images:",len(images))
    print("Ground truth cards:",len(gt))
    print("Card size:",pipe.card_w,"x",pipe.card_h)
    print("Stage 2 detector: YOLO26n-OBB")
    print("OBB conf:",args.obb_conf)
    print("OBB imgsz:",args.obb_imgsz)
    print("OBB crop padding:",args.obb_padding)
    print("IoG threshold:",pipe.iog_th)
    print("="*72)

    # Warm-up: không đưa vào latency benchmark
    for p in images[:min(args.warmup,len(images))]:
        img=cv2.imread(str(p))
        if img is None:continue
        try:
            card=pipe.stage1(img)
            if card is not None:
                _,assigned=pipe.stage2(card)
                pipe.stage3(card,assigned)
        except Exception:
            pass

    vis_dir=out/"visualizations"
    if args.save_vis:vis_dir.mkdir(exist_ok=True)

    rows=[];field_rows=[]
    unmatched=[]

    for idx,p in enumerate(tqdm(images,desc="End-to-end")):
        img=cv2.imread(str(p))
        if img is None:
            rows.append({"image":p.name,"status":"read_error"})
            continue

        gt_key=match_gt(p.name,gt.keys())
        gt_card=gt.get(gt_key,{}) if gt_key else {}
        if gt_key is None:unmatched.append(p.name)

        total_start=time.perf_counter()

        # Stage 1: Pose + Warp
        t=time.perf_counter()
        card=pipe.stage1(img)
        stage1_ms=(time.perf_counter()-t)*1000

        if card is None:
            total_ms=(time.perf_counter()-total_start)*1000
            rows.append({
                "image":p.name,"gt_key":gt_key,"status":"pose_fail",
                "stage1_ms":stage1_ms,"stage2_ms":0.0,"stage3_ms":0.0,"stage4_ms":0.0,
                "total_ms":total_ms,"pipeline_success":False,"complete_output":False,
                "complete_gt":bool(gt_card) and all(gt_card.get(f) for f in fields),
                "card_correct":False
            })
            continue

        # Stage 2: YOLO26n-OBB text detector + spatial filtering
        t=time.perf_counter()
        boxes,assigned=pipe.stage2(card)
        stage2_ms=(time.perf_counter()-t)*1000

        # Stage 3: recognition
        t=time.perf_counter()
        raw,conf=pipe.stage3(card,assigned)
        stage3_ms=(time.perf_counter()-t)*1000

        # Stage 4: post-processing
        t=time.perf_counter()
        final={f:postprocess(f,raw[f],post_dic,pipe.field_configs.get(f,{})) for f in fields}
        stage4_ms=(time.perf_counter()-t)*1000

        total_ms=(time.perf_counter()-total_start)*1000
        complete_output=all(bool(final[f]) for f in fields)

        row={
            "image":p.name,"gt_key":gt_key,"status":"ok",
            "stage1_ms":stage1_ms,"stage2_ms":stage2_ms,
            "stage3_ms":stage3_ms,"stage4_ms":stage4_ms,"total_ms":total_ms,
            "pipeline_success":True,"complete_output":complete_output,
            "det_count":len(boxes)
        }

        for f in fields:
            g=gt_card.get(f)
            row[f"gt_{f}"]=g
            row[f"raw_{f}"]=raw[f]         # lưu để debug
            row[f"final_{f}"]=final[f]
            row[f"conf_{f}"]=conf[f]
            row[f"covered_{f}"]=bool(assigned[f])

            # Chỉ chấm OUTPUT CUỐI, không chấm stage 3 riêng
            if g:
                field_rows.append({
                    "image":p.name,
                    "field":f,
                    "gt":norm(g),
                    "prediction":norm(final[f]),
                    "correct":norm(final[f])==norm(g),
                    "edit":edit_distance(final[f],g),
                    "gt_chars":len(norm(g)),
                    "covered":bool(assigned[f])
                })

        complete_gt=bool(gt_card) and all(gt_card.get(f) for f in fields)
        row["complete_gt"]=complete_gt
        row["card_correct"]=(
            complete_gt and
            all(norm(final[f])==norm(gt_card[f]) for f in fields)
        )
        rows.append(row)

        if args.save_vis and idx<args.vis_limit:
            cv2.imwrite(str(vis_dir/f"{p.stem}.jpg"),pipe.draw(card,boxes))

    df=pd.DataFrame(rows)
    fdf=pd.DataFrame(field_rows)

    df.to_csv(out/"per_image_results.csv",index=False,encoding="utf-8-sig")
    fdf.to_csv(out/"field_results.csv",index=False,encoding="utf-8-sig")

    # Final field metrics
    field_summary=[]
    if len(fdf):
        for f in list(fields)+["OVERALL"]:
            d=fdf if f=="OVERALL" else fdf[fdf.field==f]
            if not len(d):continue
            field_summary.append({
                "field":f,
                "n":len(d),
                "coverage_pct":100*float(d.covered.mean()),
                "exact_accuracy_pct":100*float(d.correct.mean()),
                "cer_pct":100*float(d.edit.sum())/max(int(d.gt_chars.sum()),1)
            })

    fs=pd.DataFrame(field_summary)
    fs.to_csv(out/"final_field_summary.csv",index=False,encoding="utf-8-sig")

    # Card Exact Accuracy
    complete=df[df.complete_gt==True] if "complete_gt" in df else pd.DataFrame()
    card_exact=100*float(complete.card_correct.mean()) if len(complete) else None

    # Latency
    ok=df[df.status=="ok"] if "status" in df else pd.DataFrame()
    lat=[]
    for c in ("stage1_ms","stage2_ms","stage3_ms","stage4_ms","total_ms"):
        vals=pd.to_numeric(ok[c],errors="coerce").dropna() if len(ok) else pd.Series(dtype=float)
        if len(vals):
            lat.append({
                "stage":c,
                "mean_ms":float(vals.mean()),
                "median_ms":float(vals.median()),
                "p95_ms":float(vals.quantile(.95))
            })

    ldf=pd.DataFrame(lat)
    ldf.to_csv(out/"latency_summary.csv",index=False,encoding="utf-8-sig")

    summary={
        "pipeline":"YOLO26n-Pose -> Warp -> YOLO26n-OBB -> ROI/IoG -> PP-OCRv6 Small Rec -> Post-process",
        "device":"CPU",
        "stage2_detector":"YOLO26n-OBB",
        "obb_conf":args.obb_conf,
        "obb_imgsz":args.obb_imgsz,
        "obb_crop_padding":args.obb_padding,
        "iog_threshold":pipe.iog_th,
        "images_total":len(df),
        "ground_truth_cards":len(gt),
        "unmatched_images":unmatched,
        "status_counts":df.status.value_counts().to_dict() if "status" in df else {},
        "pipeline_success_pct":100*float((df.status=="ok").mean()) if len(df) else None,
        "complete_output_pct":100*float(df.get("complete_output",pd.Series(False,index=df.index)).fillna(False).mean()) if len(df) else None,
        "complete_gt_cards_evaluated":len(complete),
        "card_exact_accuracy_pct":card_exact,
        "final_field_metrics":field_summary,
        "latency":lat,
        "note":"Template JSON dùng cho runtime; Ground-truth JSON chỉ dùng chấm output cuối."
    }

    with open(out/"summary.json","w",encoding="utf-8") as f:
        json.dump(summary,f,ensure_ascii=False,indent=2)

    print("\n"+"="*72)
    print("KẾT QUẢ END-TO-END")
    print("="*72)
    print("Matched GT     :",len(df)-len(unmatched),"/",len(df))
    print("Pipeline success:",f"{summary['pipeline_success_pct']:.2f}%" if summary["pipeline_success_pct"] is not None else "N/A")
    print("Complete output :",f"{summary['complete_output_pct']:.2f}%" if summary["complete_output_pct"] is not None else "N/A")
    print("Card Exact Acc  :",f"{card_exact:.4f}%" if card_exact is not None else "N/A")

    if len(fs):
        print("\nFINAL FIELD METRICS")
        print(fs.round(4).to_string(index=False))

    if len(ldf):
        print("\nLATENCY")
        print(ldf.round(4).to_string(index=False))

    if unmatched:
        print("\nẢnh không map được GT:")
        for x in unmatched[:20]:print(" -",x)

    print("\nSaved:",out.resolve())
    print("="*72)

if __name__=="__main__":
    main()
