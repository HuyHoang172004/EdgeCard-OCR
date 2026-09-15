#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, re, json, time, argparse, unicodedata
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO
import onnxruntime as ort

IMG_EXT={'.jpg','.jpeg','.png','.bmp','.webp'}
FIELDS=('name','course','class','student_id')
DIGIT_MAP=str.maketrans({'O':'0','Q':'0','I':'1','L':'1','Z':'2','S':'5','B':'8','G':'6'})
LETTER_MAP=str.maketrans({'0':'O','1':'I'})

def parse_args():
    p=argparse.ArgumentParser('EdgeCard end-to-end evaluation - Raspberry Pi 4')
    p.add_argument('--pose-weights',default='/home/hoang/edgecard/models/stage1/best-stage1_ncnn_model')
    p.add_argument('--obb-weights',default='/home/hoang/edgecard/models/stage2/best-stage2_ncnn_model')
    p.add_argument('--rec-onnx',default='/home/hoang/edgecard/models/stage3/ppocrv6_small_rec.onnx')
    p.add_argument('--rec-dict',default='/home/hoang/edgecard/models/stage3/studentcard_vi_dict.txt')
    p.add_argument('--raw-test-dir',default='/home/hoang/edgecard/test-card')
    p.add_argument('--template-json',default='/home/hoang/edgecard/metadata/student_card_template.json')
    p.add_argument('--ground-truth-json',default='/home/hoang/edgecard/metadata/pipeline_gt_complete.json')
    p.add_argument('--output-dir',default='/home/hoang/edgecard/pipeline_eval_pi')
    p.add_argument('--postprocess-dict',default='')
    p.add_argument('--pose-conf',type=float,default=0.25)
    p.add_argument('--pose-imgsz',type=int,default=640)
    p.add_argument('--obb-conf',type=float,default=0.25)
    p.add_argument('--obb-imgsz',type=int,default=640)
    p.add_argument('--obb-padding',type=float,default=0.08)
    p.add_argument('--iog-th',type=float,default=-1)
    p.add_argument('--cpu-threads',type=int,default=max(1,min(4,os.cpu_count() or 4)))
    p.add_argument('--warmup',type=int,default=2)
    p.add_argument('--limit',type=int,default=0)
    p.add_argument('--save-vis',action='store_true')
    p.add_argument('--vis-limit',type=int,default=20)
    return p.parse_args()

def norm(s): return ' '.join(unicodedata.normalize('NFC',str(s)).strip().split())

def edit_distance(a,b):
    a,b=norm(a),norm(b); prev=list(range(len(b)+1))
    for i,ca in enumerate(a,1):
        cur=[i]
        for j,cb in enumerate(b,1): cur.append(min(cur[-1]+1,prev[j]+1,prev[j-1]+(ca!=cb)))
        prev=cur
    return prev[-1]

def order_quad(pts):
    pts=np.asarray(pts,np.float32).reshape(4,2); s=pts.sum(1); d=np.diff(pts,axis=1).ravel()
    return np.array([pts[np.argmin(s)],pts[np.argmin(d)],pts[np.argmax(s)],pts[np.argmax(d)]],np.float32)

def roi_poly(roi):
    x1,y1,x2,y2=map(float,roi)
    return np.array([[x1,y1],[x2,y1],[x2,y2],[x1,y2]],np.float32)

def poly_iog(det,roi):
    a=cv2.convexHull(np.asarray(det,np.float32)); b=cv2.convexHull(np.asarray(roi,np.float32))
    area=float(abs(cv2.contourArea(a)))
    if area<=0: return 0.0
    inter,_=cv2.intersectConvexConvex(a,b)
    return float(inter)/area

def load_gt(path):
    with open(path,'r',encoding='utf-8') as f: data=json.load(f)
    if not isinstance(data,dict): raise ValueError('Ground truth JSON phải là object/dict')
    gt={}
    for k,v in data.items():
        if isinstance(v,dict):
            gt[os.path.basename(k)]={f:v.get(f) for f in FIELDS}
    return gt

def match_gt(raw_name,gt_keys):
    keys=list(gt_keys)
    if raw_name in keys: return raw_name
    stem,ext=os.path.splitext(raw_name); ext_token=ext.lower().lstrip('.')
    pref=f'{stem}_{ext_token}.rf.'.lower(); cand=[k for k in keys if k.lower().startswith(pref)]
    if len(cand)==1: return cand[0]
    pref=f'{stem}.rf.'.lower(); cand=[k for k in keys if k.lower().startswith(pref)]
    return cand[0] if len(cand)==1 else None

def load_post_dict(path):
    if not path: return {}
    with open(path,'r',encoding='utf-8') as f: return json.load(f)

def nearest(value,cands,max_dist=1):
    if not value or not cands: return value
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
        self.card_w=int(tpl['card_size']['width']); self.card_h=int(tpl['card_size']['height'])
        self.field_rois={k:v['roi'] for k,v in tpl['fields'].items()}
        # Giữ cấu hình từng field để Stage 4 dùng cùng rule hậu xử lý như PC.
        self.field_configs={k:dict(v) for k,v in tpl['fields'].items()}
        self.iog_th=float(tpl.get('spatial_filter',{}).get('threshold',0.5)) if args.iog_th<0 else args.iog_th
        self.pose=YOLO(args.pose_weights,task='pose')
        self.obb=YOLO(args.obb_weights,task='obb')
        so=ort.SessionOptions(); so.intra_op_num_threads=int(args.cpu_threads); so.inter_op_num_threads=1
        so.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL
        so.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.rec=ort.InferenceSession(args.rec_onnx,sess_options=so,providers=['CPUExecutionProvider'])
        self.rec_input=self.rec.get_inputs()[0].name; self.rec_output=self.rec.get_outputs()[0].name
        with open(args.rec_dict,'r',encoding='utf-8') as f: self.dict_chars=[x.rstrip('\r\n') for x in f]
        self.rec_shape=[3,48,320]

    def stage1(self,img):
        r=self.pose.predict(img,conf=self.args.pose_conf,imgsz=self.args.pose_imgsz,device='cpu',verbose=False)[0]
        if r.keypoints is None or len(r.keypoints.xy)==0: return None
        idx=int(r.boxes.conf.argmax().item()) if r.boxes is not None and len(r.boxes)>0 else 0
        pts=np.asarray(r.keypoints.xy[idx].detach().cpu().numpy(),dtype=np.float32)
        if pts.ndim!=2 or pts.shape[0]<4 or pts.shape[1]<2: return None
        # GIỮ NGUYÊN semantic order: TL, TR, BR, BL. Không order_quad ở Stage 1.
        src=np.ascontiguousarray(pts[:4,:2],dtype=np.float32)
        dst=np.ascontiguousarray([[0,0],[self.card_w-1,0],[self.card_w-1,self.card_h-1],[0,self.card_h-1]],dtype=np.float32)
        M=cv2.getPerspectiveTransform(src,dst)
        return cv2.warpPerspective(img,M,(self.card_w,self.card_h))

    def stage2(self,card):
        r=self.obb.predict(card,conf=self.args.obb_conf,imgsz=self.args.obb_imgsz,device='cpu',verbose=False)[0]
        boxes=[]
        if r.obb is not None and r.obb.xyxyxyxy is not None and len(r.obb.xyxyxyxy)>0:
            polys=r.obb.xyxyxyxy.detach().cpu().numpy()
            boxes=[np.ascontiguousarray(np.asarray(b,np.float32).reshape(4,2)) for b in polys]
        assigned={f:[] for f in FIELDS}
        for f in FIELDS:
            rp=roi_poly(self.field_rois[f])
            for b in boxes:
                score=poly_iog(b,rp)
                if score>=self.iog_th: assigned[f].append((b,score))
            assigned[f].sort(key=lambda z:(float(np.mean(z[0][:,1])),float(np.mean(z[0][:,0]))))
        return boxes,assigned

    def expand_quad(self,pts,img_shape,padding=None):
        p=np.asarray(pts,np.float32).reshape(4,2)
        pad=self.args.obb_padding if padding is None else float(padding)
        if pad<=0: return np.ascontiguousarray(p,np.float32)
        center=np.mean(p,axis=0,keepdims=True); q=center+(p-center)*(1.0+2.0*pad)
        h,w=img_shape[:2]; q[:,0]=np.clip(q[:,0],0,max(w-1,0)); q[:,1]=np.clip(q[:,1],0,max(h-1,0))
        return np.ascontiguousarray(q,np.float32)

    def crop_quad(self,img,pts):
        p=order_quad(pts)
        w=max(int(np.linalg.norm(p[0]-p[1])),int(np.linalg.norm(p[2]-p[3])),2)
        h=max(int(np.linalg.norm(p[0]-p[3])),int(np.linalg.norm(p[1]-p[2])),2)
        dst=np.array([[0,0],[w-1,0],[w-1,h-1],[0,h-1]],np.float32)
        crop=cv2.warpPerspective(img,cv2.getPerspectiveTransform(p,dst),(w,h))
        if crop.shape[0]/max(crop.shape[1],1)>=1.5: crop=np.rot90(crop)
        return np.ascontiguousarray(crop)

    def rec_preprocess(self,img):
        _,img_h,img_w=self.rec_shape; h,w=img.shape[:2]
        resized_w=min(img_w,int(np.ceil(img_h*(w/float(h)))))
        resized=cv2.resize(img,(resized_w,img_h)).astype(np.float32).transpose((2,0,1))
        resized=(resized/255.0-0.5)/0.5
        padded=np.zeros((3,img_h,img_w),np.float32); padded[:,:,:resized_w]=resized
        return padded[np.newaxis,:]

    def rec_decode(self,pred):
        if pred.ndim!=3: raise RuntimeError(f'Unexpected recognition output shape: {pred.shape}')
        nc=pred.shape[-1]
        if nc==len(self.dict_chars)+2: chars=['blank']+self.dict_chars+[' ']
        elif nc==len(self.dict_chars)+1: chars=['blank']+self.dict_chars
        else: raise RuntimeError(f'Dictionary/model mismatch: classes={nc}, dict={len(self.dict_chars)}')
        idxs=pred.argmax(axis=2)[0]; probs=pred.max(axis=2)[0]; text=[]; confs=[]; prev=None
        for idx,prob in zip(idxs,probs):
            idx=int(idx)
            if idx!=0 and idx!=prev and idx<len(chars): text.append(chars[idx]); confs.append(float(prob))
            prev=idx
        return norm(''.join(text)), float(np.mean(confs)) if confs else 0.0

    def recognize(self,crop):
        pred=self.rec.run([self.rec_output],{self.rec_input:self.rec_preprocess(crop)})[0]
        return self.rec_decode(pred)

    def stage3(self,card,assigned):
        raw={}; conf={}
        for f in FIELDS:
            texts=[]; scores=[]
            for b,_ in assigned[f]:
                crop=self.crop_quad(card,self.expand_quad(b,card.shape))
                txt,sc=self.recognize(crop)
                if txt: texts.append(txt)
                scores.append(sc)
            raw[f]=norm(' '.join(texts)); conf[f]=float(np.mean(scores)) if scores else 0.0
        return raw,conf

    def draw(self,card,boxes):
        v=card.copy()
        for f,roi in self.field_rois.items():
            x1,y1,x2,y2=map(int,roi); cv2.rectangle(v,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(v,f,(x1,max(15,y1-4)),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,255,0),1,cv2.LINE_AA)
        for b in boxes: cv2.polylines(v,[b.astype(np.int32)],True,(255,0,0),1)
        return v

def main():
    args=parse_args(); out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    required=[args.pose_weights,args.obb_weights,args.rec_onnx,args.rec_dict,args.raw_test_dir,args.template_json,args.ground_truth_json]
    for p in required:
        if not Path(p).exists(): raise FileNotFoundError(p)
    with open(args.template_json,'r',encoding='utf-8') as f: tpl=json.load(f)
    gt=load_gt(args.ground_truth_json); post_dic=load_post_dict(args.postprocess_dict); pipe=Pipeline(args,tpl)
    images=sorted([p for p in Path(args.raw_test_dir).rglob('*') if p.suffix.lower() in IMG_EXT])
    if args.limit>0: images=images[:args.limit]
    if not images: raise RuntimeError('Không tìm thấy ảnh test')

    print('='*72)
    print('EdgeCard END-TO-END Evaluation - Raspberry Pi 4')
    print('Python:',sys.version.split()[0]); print('ONNX Runtime:',ort.__version__)
    print('Images:',len(images)); print('Ground truth cards:',len(gt)); print('Card size:',pipe.card_w,'x',pipe.card_h)
    print('Stage 1: YOLO26n-Pose NCNN'); print('Stage 2: YOLO26n-OBB NCNN'); print('Stage 3: PP-OCRv6 Small Rec ONNX')
    print('OBB crop padding:',args.obb_padding); print('IoG threshold:',pipe.iog_th); print('CPU threads:',args.cpu_threads)
    print('='*72)

    for p in images[:min(args.warmup,len(images))]:
        img=cv2.imread(str(p))
        if img is None: continue
        try:
            card=pipe.stage1(img)
            if card is not None:
                _,assigned=pipe.stage2(card); pipe.stage3(card,assigned)
        except Exception as e: print('[Warmup warning]',p.name,e)

    vis_dir=out/'visualizations'
    if args.save_vis: vis_dir.mkdir(exist_ok=True)
    rows=[]; field_rows=[]; unmatched=[]

    for idx,p in enumerate(tqdm(images,desc='End-to-end')):
        img=cv2.imread(str(p))
        if img is None:
            rows.append({'image':p.name,'status':'read_error'}); continue
        gt_key=match_gt(p.name,gt.keys()); gt_card=gt.get(gt_key,{}) if gt_key else {}
        if gt_key is None: unmatched.append(p.name)
        total_start=time.perf_counter()

        t=time.perf_counter(); card=pipe.stage1(img); stage1_ms=(time.perf_counter()-t)*1000
        if card is None:
            total_ms=(time.perf_counter()-total_start)*1000
            rows.append({'image':p.name,'gt_key':gt_key,'status':'pose_fail','stage1_ms':stage1_ms,'stage2_ms':0.0,'stage3_ms':0.0,'stage4_ms':0.0,'total_ms':total_ms,'pipeline_success':False,'complete_output':False,'complete_gt':bool(gt_card) and all(gt_card.get(f) for f in FIELDS),'card_correct':False})
            continue

        t=time.perf_counter(); boxes,assigned=pipe.stage2(card); stage2_ms=(time.perf_counter()-t)*1000
        t=time.perf_counter(); raw,conf=pipe.stage3(card,assigned); stage3_ms=(time.perf_counter()-t)*1000
        t=time.perf_counter(); final={f:postprocess(f,raw[f],post_dic,pipe.field_configs.get(f,{})) for f in FIELDS}; stage4_ms=(time.perf_counter()-t)*1000
        total_ms=(time.perf_counter()-total_start)*1000; complete_output=all(bool(final[f]) for f in FIELDS)

        row={'image':p.name,'gt_key':gt_key,'status':'ok','stage1_ms':stage1_ms,'stage2_ms':stage2_ms,'stage3_ms':stage3_ms,'stage4_ms':stage4_ms,'total_ms':total_ms,'pipeline_success':True,'complete_output':complete_output,'det_count':len(boxes)}
        for f in FIELDS:
            g=gt_card.get(f); row[f'gt_{f}']=g; row[f'raw_{f}']=raw[f]; row[f'final_{f}']=final[f]; row[f'conf_{f}']=conf[f]; row[f'covered_{f}']=bool(assigned[f])
            if g:
                field_rows.append({'image':p.name,'field':f,'gt':norm(g),'prediction':norm(final[f]),'correct':norm(final[f])==norm(g),'edit':edit_distance(final[f],g),'gt_chars':len(norm(g)),'covered':bool(assigned[f])})
        complete_gt=bool(gt_card) and all(gt_card.get(f) for f in FIELDS)
        row['complete_gt']=complete_gt; row['card_correct']=complete_gt and all(norm(final[f])==norm(gt_card[f]) for f in FIELDS)
        rows.append(row)
        if args.save_vis and idx<args.vis_limit: cv2.imwrite(str(vis_dir/f'{p.stem}.jpg'),pipe.draw(card,boxes))

    df=pd.DataFrame(rows); fdf=pd.DataFrame(field_rows)
    df.to_csv(out/'per_image_results.csv',index=False,encoding='utf-8-sig'); fdf.to_csv(out/'field_results.csv',index=False,encoding='utf-8-sig')

    field_summary=[]
    if len(fdf):
        for f in list(FIELDS)+['OVERALL']:
            d=fdf if f=='OVERALL' else fdf[fdf.field==f]
            if len(d): field_summary.append({'field':f,'n':len(d),'coverage_pct':100*float(d.covered.mean()),'exact_accuracy_pct':100*float(d.correct.mean()),'cer_pct':100*float(d.edit.sum())/max(int(d.gt_chars.sum()),1)})
    fs=pd.DataFrame(field_summary); fs.to_csv(out/'final_field_summary.csv',index=False,encoding='utf-8-sig')

    complete=df[df.complete_gt==True] if 'complete_gt' in df else pd.DataFrame(); card_exact=100*float(complete.card_correct.mean()) if len(complete) else None
    ok=df[df.status=='ok'] if 'status' in df else pd.DataFrame(); lat=[]
    for c in ('stage1_ms','stage2_ms','stage3_ms','stage4_ms','total_ms'):
        vals=pd.to_numeric(ok[c],errors='coerce').dropna() if len(ok) else pd.Series(dtype=float)
        if len(vals): lat.append({'stage':c,'mean_ms':float(vals.mean()),'median_ms':float(vals.median()),'p95_ms':float(vals.quantile(.95))})
    ldf=pd.DataFrame(lat); ldf.to_csv(out/'latency_summary.csv',index=False,encoding='utf-8-sig')

    summary={'pipeline':'YOLO26n-Pose NCNN -> Warp -> YOLO26n-OBB NCNN -> ROI/IoG -> PP-OCRv6 Small Rec ONNX -> Post-process','device':'Raspberry Pi 4 CPU','stage1_backend':'NCNN','stage2_backend':'NCNN','stage3_backend':'ONNX Runtime','obb_crop_padding':args.obb_padding,'iog_threshold':pipe.iog_th,'images_total':len(df),'ground_truth_cards':len(gt),'unmatched_images':unmatched,'status_counts':df.status.value_counts().to_dict() if 'status' in df else {},'pipeline_success_pct':100*float((df.status=='ok').mean()) if len(df) else None,'complete_output_pct':100*float(df.get('complete_output',pd.Series(False,index=df.index)).fillna(False).mean()) if len(df) else None,'complete_gt_cards_evaluated':len(complete),'card_exact_accuracy_pct':card_exact,'final_field_metrics':field_summary,'latency':lat,'note':'Template JSON dùng cho runtime; Ground-truth JSON chỉ dùng chấm output cuối.'}
    with open(out/'summary.json','w',encoding='utf-8') as f: json.dump(summary,f,ensure_ascii=False,indent=2)

    print('\n'+'='*72); print('KẾT QUẢ END-TO-END - RASPBERRY PI 4'); print('='*72)
    print('Matched GT       :',len(df)-len(unmatched),'/',len(df))
    print('Pipeline success :',f"{summary['pipeline_success_pct']:.2f}%" if summary['pipeline_success_pct'] is not None else 'N/A')
    print('Complete output  :',f"{summary['complete_output_pct']:.2f}%" if summary['complete_output_pct'] is not None else 'N/A')
    print('Card Exact Acc   :',f'{card_exact:.4f}%' if card_exact is not None else 'N/A')
    if len(fs): print('\nFINAL FIELD METRICS\n'+fs.round(4).to_string(index=False))
    if len(ldf): print('\nLATENCY\n'+ldf.round(4).to_string(index=False))
    if unmatched:
        print('\nẢnh không map được GT:')
        for x in unmatched[:20]: print(' -',x)
    print('\nSaved:',out.resolve()); print('='*72)

if __name__=='__main__': main()
