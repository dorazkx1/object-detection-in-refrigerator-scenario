import torch
import cv2
import numpy as np
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import os,json
import glob, yaml
from pathlib import Path
from PIL import Image
import torch, re, spacy
from transformers import BlipProcessor, BlipForConditionalGeneration, CLIPProcessor, CLIPModel
from difflib import get_close_matches
# === 统一别名（只做“词汇到主类名”的映射；主类名要与 data.yaml 里的名字完全一致）===
ALIAS = {
    # 复数/大小写/空格
    "apples":"apple","bananas":"banana","beans":"bean","carrots":"carrot",
    "yoghurt":"yogurt","icecream":"ice cream","lip stick":"lipstick",
    # 近义/口语
    "soft toy":"toy","spectacular":"glasses","olive":"olive","olives":"olive",
    # 容器/材质统一
    "paper box":"box","plastic box":"box","paperbox":"box","plasticbox":"box",
    "glass bottle":"bottle","plastic bottle":"bottle",
    # 葱类统一
    "spring onion":"onion","green onion":"onion","scallion":"onion",
    # 其他
    "make up":"makeup"
}


# 在“允许列表（BLIP先验）阶段”不要把这些词粗暴映射到具体类，避免误检放大
DISABLE_MANUAL_FOR_ALLOW = {"jar","can","bottle of sauce"}

# 高误报类（按你之前的混淆统计微调）
#DISABLE_MANUAL_FOR_ALLOW = {"jar","can","bottle of sauce","tomato","tomatoes","tomato sauce"}
# 高误报类（按你的混淆矩阵统计来调）
HIGH_FP_NAMES = ["butter","milk","cheese","bread","box","bottle","jam","tomato"]

# 在常量区添加
CONFUSE_GROUPS = {
    "Milk": ["Yogurt","Butter","Jam","Tomato Puree"],
    "Yogurt": ["Milk","Cheese","Butter"],
    "Butter": ["Cheese","Tomato Puree","Pickle","Beans","Jam"],
    "Banana": ["Lemon"],
    "Lemon": ["Banana"],
    "Tomato Puree": ["Butter","ketchup","Jam","Sauce"],
}

# ==== Precision-first switches ====

# 评估时，为了避免“过度手工映射”放大误检，这些 key 的 MANUAL 映射在 BLIP allow-list 阶段禁用

MIN_CONF_SOFT       = 0.05   # YOLO 最低软阈值（低于它也可以被“语义”拯救）
BASE_CONF_CLS       = 0.10   # 普通类 YOLO 置信度参考阈值（从 0.25 下调）
HIGH_FP_CONF        = 0.20   # 高误报类参考阈值（从 0.40/0.45 下调）

CLIP_TAU_DEFAULT    = 0.32   # CLIP 通过线（稍微上调，确保真语义）
CLIP_TAU_HIGHFP     = 0.40   # 高误报类更严格

FUSED_TAU           = 0.06   # 融合阈值：fused = yolo_conf * clip_score
FUSED_TAU_HIGHFP    = 0.09   # 高误报类的融合阈值
SALVAGE_CLIP_TAU    = 0.55   # “低置信度但语义很强”时的拯救线
SOFT_ALLOW_COEFF = 0.85               # 不在 BLIP 先验里的类，分数乘这个系数（=1.0 表示关闭软约束
NMS_PER_CLASS_IOU    = 0.65      # 类内 NMS 稍放宽
TOPK_PER_IMAGE_PER_CLASS = 2
N_CROPS_FOR_BLIP = 12   # 每图最多取 12 个裁剪跑 BLIP
BLIP_CROP_TOPK   = 3

def clip_classify_crop_all(pil_crop, all_cids, coco_id_to_name,
                           prompt_templates, device, clip_model, clip_processor):
    """
    用 CLIP 在所有类别上打分，返回(best_cid, best_score_01, margin_01)
    margin_01 = best - second_best，都是 [0,1] 空间。
    """
    best_id, best_s = None, -1.0
    second = -1.0
    # 逐类打分（数据量不大先直算；如想提速可做文本嵌入缓存）
    for cid in all_cids:
        cname = coco_id_to_name[cid]
        s = clip_score_for_class(pil_crop, cname, prompt_templates, device, clip_model, clip_processor)
        if s > best_s:
            second = best_s
            best_s = s
            best_id = cid
        elif s > second:
            second = s
    margin = max(0.0, best_s - max(second, 0.0))
    return int(best_id), float(best_s), float(margin)

def canonical_name(s: str) -> str:
    """统一大小写/空格/下划线，并用 ALIAS 映射到主类名"""
    n = norm_text(s).replace('_',' ').strip()
    return ALIAS.get(n, n)

def norm_text(s:str)->str:
    """normalize text by removing symbols that are not alphabets, numbers or spaces and converting to lowercase and remove extra spaces before and after the text"""
    return re.sub(r'[^a-z0-9 ]', '', s.lower()).strip()
def build_canonical_maps_from_gt(coco_gt):
    cats = coco_gt.loadCats(coco_gt.getCatIds())
    by_canon = {}
    for c in cats:
        cid  = int(c["id"])
        cname= canonical_name(c["name"])
        by_canon.setdefault(cname, []).append(cid)
    # 选每个 canonical 的“主 id”（用最小 id，稳定且简单）
    canon_to_primary = {cname: min(ids) for cname, ids in by_canon.items()}
    # 所有原 cid -> 主 cid 的映射
    cid_remap = {}
    for cname, ids in by_canon.items():
        p = canon_to_primary[cname]
        for cid in ids:
            cid_remap[cid] = p
    return by_canon, canon_to_primary, cid_remap

def merge_coco_gt_inplace(coco_gt, cid_remap, canon_to_primary):
    # 1) 重写标注里的 category_id
    for ann in coco_gt.dataset["annotations"]:
        ann["category_id"] = int(cid_remap[int(ann["category_id"])])
    # 2) 生成去重后的 categories（名字用 canonical）
    new_cats = []
    for cname, primary in canon_to_primary.items():
        new_cats.append({"id": int(primary), "name": cname})
    # 按 id 排序以稳定
    coco_gt.dataset["categories"] = sorted(new_cats, key=lambda x: x["id"])
    # 3) 重新建索引
    coco_gt.createIndex()
def mask_to_bbox(mask):
    """
    convert a mask into a tight bounding box[x_min, y_min, width, height]
    args: mask: a binary array
    """
    #y_coords, x_coords correspond to an ndarray containing the y_coords, x_coords of the mask
    y_coords, x_coords = np.where(mask)
    #if mask is empty, return None
    if not y_coords.size or not x_coords.size:
        return None
    #calculate the bounding box coordinates
    x_min, x_max = np.min(x_coords), np.max(x_coords)
    y_min, y_max = np.min(y_coords), np.max(y_coords)
    
    width = x_max - x_min 
    height = y_max - y_min 
    return [float(x_min), float(y_min), float(width), float(height)]

def extract_prompts_from_yolo_box(box):
    """extract a single point prompt from a yolo boudning box

    Args:
        box (_type_): a yolo bounding box in the format [x_min, y_min, x_max, y_max]
    returns: the center point coordiantes of the bounding box as a numpy array
    """
    x_min, y_min, x_max, y_max = box
    x_center = (x_min + x_max )/ 2
    y_center = (y_min + y_max)/ 2
    
    return np.array([[x_center, y_center]])

def list_images(img_str:str)-> list[str]:
    """
    recursively lists all images in a given directory
    """
    ext = ['*.jpg','*.jpeg', '*.png', '*.bmp']
    files = []
    for e in ext:
        files.extend(glob.glob(os.path.join(img_str,"**",e),recursive=True))
    return sorted(list(set(files)))

def create_coco_gt_from_yolo_labels(data_yaml_path:str,split:str="test" ):
    """create a coco ground truth object from a yolo yaml file

    Args:
        data_yaml_path (str): path to the yolo data yaml file
        split (str, optional): the split to use, either "train" or "test". Defaults to "test".
    returns: COCO object containing the ground truth annotations
    """
    #safely load the yaml file
    with open(data_yaml_path, "r") as f:
        data = yaml.safe_load(f)
    #get the class names from the yaml file
    names = data.get("names", [])
    #number of classes
    nc = len(names)
    
    img_root_path = data.get(split)
    lbl_root_path = str(Path(img_root_path).parent / 'labels')
    print(f"[DEBUG] lbl_root_path={lbl_root_path}, exists={os.path.exists(lbl_root_path)}")

    #list all image paths
    img_paths = list_images(img_root_path)
    print(f"[DEBUG] img_root_path={img_root_path}, #images={len(img_paths)}")
    images, annotations = [], []
    ann_id_counter = 1
    #create a mapping from image paths to image ids
    image_id_map = {path: i for i, path in enumerate(img_paths)}

    for img_path in tqdm(img_paths, desc="Creating COCO GT"):
        img_id = image_id_map[img_path]
        w, h = Image.open(img_path).size
        images.append({"id": img_id, "file_name": os.path.basename(img_path), "width": w, "height": h})
        
        lbl_path = os.path.join(lbl_root_path, Path(img_path).stem + ".txt")
        if os.path.exists(lbl_path):
            with open(lbl_path, "r") as lf:
                for line in lf:
                    line = line.strip()
                    if not line: continue
                    cls, cx, cy, bw, bh = map(float, line.split())
                    cls = int(cls)
                    #convert normalized coordinates to absolute pixel values
                    x, y, w_bbox, h_bbox = (cx - bw / 2) * w, (cy - bh / 2) * h, bw * w, bh * h

                    annotations.append({
                        "id": ann_id_counter,
                        "image_id": img_id,
                        "category_id": cls + 1,
                        "iscrowd": 0,
                        "area": float(w_bbox * h_bbox),
                        "bbox": [float(x), float(y), float(w_bbox), float(h_bbox)],
                        "segmentation": []
                    })
                    ann_id_counter += 1
    
    categories = [{"id": i + 1, "name": names[i]} for i in range(nc)]
    #coco_gt_dict = {"images": images, "annotations": annotations, "categories": categories}
    coco_gt_dict = {
    "info": {
        "description": "GT built from YOLO labels",
        "version": "1.0",
        "year": 2025
    },
    "licenses": [],
    "images": images,
    "annotations": annotations,
    "categories": categories,
}
    coco = COCO()
    coco.dataset = coco_gt_dict
    coco.createIndex()
    return coco

def blip_candidates_for_image(pil_image:Image.Image, topk:int=12, mode='both', diverse=True)->list[str]:
    inputs = blip_processor(images=pil_image, return_tensors='pt').to(device)
    with torch.no_grad():
        if diverse:
            # 采样：比纯 beam 多样，适合挖更多名词
            outs = blip_model.generate(
                **inputs,
                do_sample=True, top_p=0.9, temperature=0.7,
                num_beams=1, num_return_sequences=topk,
                max_length=25, repetition_penalty=1.1,
            )
        else:
            # 保留原来的 beam 版本
            outs = blip_model.generate(
                **inputs,
                num_beams=topk, num_return_sequences=topk,
                max_length=25
            )
        caps = blip_processor.batch_decode(outs, skip_special_tokens=True)

    phrases, unigrams = set(), set()
    for cap in caps:
        doc = nlp(cap)
        for nc in doc.noun_chunks:
            phrase = norm_text(nc.text)
            head   = norm_text(nc.root.lemma_)
            if len(phrase) >= 2: phrases.add(phrase)
            if len(head)   >= 2: unigrams.add(head)
        for tok in doc:
            if tok.pos_ in ("NOUN","PROPN"):
                t = norm_text(tok.lemma_)
                if len(t) >= 2: unigrams.add(t)
    if mode == "phrase":  return sorted(phrases)
    if mode == "unigram": return sorted(unigrams)
    return sorted(phrases | unigrams)

 

def map_candidates_to_coco_ids(candidates, canon_to_coco, canon_keys, fuzzy_cutoff=0.88):
    coco_ids = set()
    for cand in candidates:
        n = canonical_name(cand)
        if n in canon_to_coco:
            coco_ids.add(int(canon_to_coco[n])); continue
        hit = get_close_matches(n, canon_keys, n=1, cutoff=fuzzy_cutoff)
        if hit: coco_ids.add(int(canon_to_coco[hit[0]]))
    return sorted(coco_ids)

def build_candidate_coco_ids(blip_cands, yolo_classes_for_img, yolo_model,
                             canon_to_coco, canon_keys):
    ids = set(map_candidates_to_coco_ids(blip_cands, canon_to_coco, canon_keys))
    if len(yolo_classes_for_img):
        for yid in np.unique(yolo_classes_for_img):
            cn = canonical_name(yolo_model.names[int(yid)])
            if cn in canon_to_coco:
                ids.add(int(canon_to_coco[cn]))
    return sorted(ids)

def clip_rank_crop(pil_crop, candidate_coco_ids, coco_id_to_name, tau=0.25, return_margin=False):
    """Rank a crop using CLIP model against a set of candidate COCO IDs.

    Args:
        pil_crop (_type_): a PIL image crop to be ranked.
        candidate_coco_ids (_type_): a list of candidate COCO category IDs to rank against.
        coco_id_to_name (_type_): a dictionary mapping COCO category IDs to category names.
        tau (float, optional): a threshold for the best score. Defaults to 0.25.
        return_margin (bool, optional): whether to return the margin between the best and second-best scores. Defaults to False.

    Returns:
        tuple[int|None, float, float]: a tuple containing the best COCO ID, its score in [0,1], and optionally the margin between the best and second-best scores.
    """
    if not candidate_coco_ids:
        return (None, 0.0, 0.0) if return_margin else (None, 0.0)

    texts, id_list = [], []
    #format the candidate coco names into prompt templetes
    for cid in candidate_coco_ids:
        cname = coco_id_to_name[cid]
        for tmpl in prompt_templates:
            texts.append(tmpl.format(cname))
            id_list.append(cid)

    #preprocess the image and text inputs for CLIP
    inputs = clip_processor(text=texts, images=pil_crop, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        out = clip_model(**inputs)#get img and text embeddings
        img = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)#normalize the image embeddings
        txt = out.text_embeds   / out.text_embeds.norm(dim=-1, keepdim=True)#normalize the text embeddings
        sims = (img @ txt.t()).squeeze(0).detach().cpu().numpy()  # [-1,1]

    #get average scores for each COCO ID
    scores = {}
    for s, cid in zip(sims, id_list):
        scores.setdefault(cid, []).append(float(s))
    scores = {cid: float(np.mean(v)) for cid, v in scores.items()}

    #sort the scores in descending order
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    #get the best id and its score
    best_cid, best = ranked[0]
    #get the second-best score if available
    second = ranked[1][1] if len(ranked) > 1 else -1.0
    #calculate the margin between the best and second-best scores
    margin = best - second

    best01 = (best + 1.0) / 2.0# [-1,1] → [0,1]
    
    if best01 < tau:  # ❗低于阈值，返回 None，避免“硬猜”
        return ((None, best01, float(margin)) if return_margin else (None, best01))

    return ((int(best_cid), best01, float(margin)) if return_margin
            else (int(best_cid), best01))

def nms_per_class(preds, iou_thr=0.5):
    import numpy as np
    out = []
    from collections import defaultdict
    by_cat = defaultdict(list)
    for p in preds:
        by_cat[p['category_id']].append(p)
    for cid, items in by_cat.items():
        boxes = np.array([p['bbox'] for p in items], dtype=np.float32)  # xywh
        scores= np.array([p['score'] for p in items], dtype=np.float32)
        x1,y1 = boxes[:,0], boxes[:,1]
        x2,y2 = x1+boxes[:,2], y1+boxes[:,3]
        order = scores.argsort()[::-1]
        keep=[]
        while order.size>0:
            i=order[0]; keep.append(i)
            xx1=np.maximum(x1[i], x1[order[1:]])
            yy1=np.maximum(y1[i], y1[order[1:]])
            xx2=np.minimum(x2[i], x2[order[1:]])
            yy2=np.minimum(y2[i], y2[order[1:]])
            w=np.maximum(0, xx2-xx1); h=np.maximum(0, yy2-yy1)
            inter=w*h
            union=(boxes[i,2]*boxes[i,3]) + (boxes[order[1:],2]*boxes[order[1:],3]) - inter + 1e-9
            iou=inter/union
            inds=np.where(iou<=iou_thr)[0]
            order=order[inds+1]
        out.extend([items[k] for k in keep])
    return out

# ========= 类无关 NMS：每张图上合并所有类 =========
def nms_agnostic_by_image(preds, iou_thr=0.55):
    from collections import defaultdict
    import numpy as np
    by_img = defaultdict(list)
    for p in preds:
        by_img[int(p["image_id"])].append(p)
    kept_all = []
    for img_id, items in by_img.items():
        if not items:
            continue
        boxes = np.array([p["bbox"] for p in items], dtype=np.float32)  # xywh
        scores= np.array([p["score"] for p in items], dtype=np.float32)
        x1,y1 = boxes[:,0], boxes[:,1]
        x2,y2 = x1+boxes[:,2], y1+boxes[:,3]
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]; keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2-xx1); h = np.maximum(0.0, yy2-yy1)
            inter = w*h
            union = (boxes[i,2]*boxes[i,3]) + (boxes[order[1:],2]*boxes[order[1:],3]) - inter + 1e-9
            iou = inter / union
            inds = np.where(iou <= iou_thr)[0]
            order = order[inds + 1]
        kept_all.extend(items[k] for k in keep)
    return kept_all

# ========= 简化版 WBF（类无关）：同图里把高IoU框加权平均 =========
def wbf_agnostic_by_image(preds, iou_thr=0.55):
    from collections import defaultdict
    import numpy as np
    by_img = defaultdict(list)
    for p in preds:
        by_img[int(p["image_id"])].append(p)

    merged = []
    for img_id, items in by_img.items():
        boxes = [np.array(p["bbox"], dtype=np.float32) for p in items]  # xywh
        scores = [float(p["score"]) for p in items]
        cats   = [int(p["category_id"]) for p in items]  # 虽然类无关，但保留原cat方便后面按类评估
        used = [False]*len(items)

        for i in np.argsort(scores)[::-1]:
            if used[i]: 
                continue
            x,y,w,h = boxes[i]; s = scores[i]
            cluster_idxs = [i]
            # 聚类：和当前框 IoU 大于阈值的都并入
            for j in range(i+1, len(items)):
                if used[j]: 
                    continue
                xx,yy,ww,hh = boxes[j]
                # IoU (xywh)
                xi1, yi1 = max(x, xx), max(y, yy)
                xi2, yi2 = min(x+w, xx+ww), min(y+h, yy+hh)
                iw, ih = max(0.0, xi2-xi1), max(0.0, yi2-yi1)
                inter = iw*ih
                union = w*h + ww*hh - inter + 1e-9
                iou = inter/union
                if iou >= iou_thr:
                    cluster_idxs.append(j)

            # 权重=score 的加权平均
            ws = np.array([scores[k] for k in cluster_idxs], dtype=np.float32)
            bs = np.array([boxes[k]  for k in cluster_idxs], dtype=np.float32)
            wsum = float(ws.sum()) + 1e-9
            b_avg = (bs * ws[:,None]).sum(axis=0) / wsum
            s_avg = float(ws.max())
            # 类别随最大score的那个
            k_best = cluster_idxs[int(np.argmax([scores[k] for k in cluster_idxs]))]
            merged.append({
                "image_id": int(img_id),
                "category_id": int(cats[k_best]),
                "bbox": [float(v) for v in b_avg.tolist()],
                "score": s_avg
            })
            for k in cluster_idxs:
                used[k] = True
    return merged

def sanitize_xywh(x, y, w, h, W, H):
    # clip 到图内，保证 w,h >= 0
    x1 = max(0.0, min(x, W - 1))
    y1 = max(0.0, min(y, H - 1))
    x2 = max(0.0, min(x + w, W))
    y2 = max(0.0, min(y + h, H))
    if x2 <= x1 or y2 <= y1:
        return None
    return [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
def bbox_iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter + 1e-9
    return inter / union
def refine_with_sam_on_outs(outs, sam_predictor, W, H, iou_gate=0.60, area_min_ratio=0.30, area_max_ratio=1.10):
    """对 outs 里的 xywh 框做 SAM + 门控精修；只在门控通过时替换，保证不伤 mAP50。"""
    refined = []
    for p in outs:
        x, y, w, h = p["bbox"]
        # 跳过极小框（可按需调）
        if w * h < 20 * 20:
            refined.append(p); 
            continue

        # YOLO xywh -> xyxy 给 SAM
        ox1, oy1, ox2, oy2 = x, y, x + w, y + h
        box_xyxy = np.array([ox1, oy1, ox2, oy2], dtype=np.float32)

        masks, mscores, _ = sam_predictor.predict(
            point_coords=None, point_labels=None,
            box=box_xyxy, multimask_output=True
        )
        if masks is None or len(masks) == 0:
            refined.append(p); 
            continue

        # 选分数最高的掩码 → 紧框
        k = int(np.argmax(mscores))
        mb = mask_to_bbox(masks[k])  # xywh
        if mb is None:
            refined.append(p); 
            continue

        rx, ry, rw, rh = mb
        rb = sanitize_xywh(rx, ry, rw, rh, W, H)
        if rb is None:
            refined.append(p); 
            continue

        # 门控：与原框 IoU、面积缩放范围
        iou = bbox_iou_xyxy([ox1, oy1, ox2, oy2], [rb[0], rb[1], rb[0] + rb[2], rb[1] + rb[3]])
        shrink = (rb[2] * rb[3]) / (w * h + 1e-9)

        if iou >= iou_gate and (area_min_ratio <= shrink <= area_max_ratio):
            q = dict(p)
            q["bbox"] = rb  # 通过门控才替换
            refined.append(q)
        else:
            refined.append(p)

    return refined
def yolo_name_to_coco_cid(yname, canon_to_coco, canon_keys, fuzzy_cutoff=0.80):
    cn = canonical_name(str(yname))
    if cn in canon_to_coco:
        return int(canon_to_coco[cn])
    hit = get_close_matches(cn, canon_keys, n=1, cutoff=fuzzy_cutoff)
    if hit:
        return int(canon_to_coco[hit[0]])
    return None
def xywh_to_xyxy(b):
    x, y, w, h = b
    return [x, y, x + w, y + h]

def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    ub = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / (ua + ub - inter + 1e-9)

def compute_confusion_matrix(coco_gt, preds_cls, iou_thr=0.5):
    """
    按图逐一贪心匹配（score 从高到低），得到 KxK 混淆矩阵；
    另外统计每个预测类的 FP、每个 GT 类的 FN。
    """
    cat_ids = sorted(coco_gt.getCatIds())
    K = len(cat_ids)
    cid2idx = {cid: i for i, cid in enumerate(cat_ids)}
    idx2name = {i: coco_gt.cats[cid]['name'] for i, cid in enumerate(cat_ids)}

    # 组织预测按 image_id 分组
    from collections import defaultdict
    by_img = defaultdict(list)
    for p in preds_cls:
        if p.get("category_id") in cid2idx:  # 只收映射到合法类的
            by_img[int(p["image_id"])].append(p)

    CM = np.zeros((K, K), dtype=np.int64)
    FP = np.zeros((K,), dtype=np.int64)
    FN = np.zeros((K,), dtype=np.int64)

    img_ids = coco_gt.getImgIds()
    for img_id in img_ids:
        # GT
        ann_ids = coco_gt.getAnnIds(imgIds=[img_id])
        gts = coco_gt.loadAnns(ann_ids)
        gt_boxes = [xywh_to_xyxy(a["bbox"]) for a in gts]
        gt_cids  = [int(a["category_id"]) for a in gts]
        gt_matched = [False]*len(gts)

        # Pred
        preds = sorted(by_img.get(int(img_id), []), key=lambda x: x["score"], reverse=True)

        # 逐预测匹配
        for p in preds:
            pb = xywh_to_xyxy(p["bbox"])
            pc = int(p["category_id"])
            pidx = cid2idx.get(pc, None)
            if pidx is None:
                continue
            # 找到 IoU>=thr 且尚未匹配的 GT 中 IoU 最大者
            best_j, best_iou = -1, -1.0
            for j, (gb, matched) in enumerate(zip(gt_boxes, gt_matched)):
                if matched: 
                    continue
                iou = iou_xyxy(pb, gb)
                if iou >= iou_thr and iou > best_iou:
                    best_iou, best_j = iou, j
            if best_j >= 0:
                gt_matched[best_j] = True
                gi = cid2idx[gt_cids[best_j]]
                CM[gi, pidx] += 1
            else:
                # 找不到配对 → FP
                FP[pidx] += 1

        # 未匹配的 GT → FN
        for matched, gcid in zip(gt_matched, gt_cids):
            if not matched:
                FN[cid2idx[gcid]] += 1

    return CM, FP, FN, [idx2name[i] for i in range(K)]
def build_classwise_thresholds(name_to_coco):
    CLS_CONF_BY_CID = {}
    CLIP_TAU_BY_CID = {}
    for name, cid in name_to_coco.items():
        if name in HIGH_FP_NAMES:
            CLS_CONF_BY_CID[cid] = HIGH_FP_CONF
            CLIP_TAU_BY_CID[cid] = CLIP_TAU_HIGHFP
        else:
            CLS_CONF_BY_CID[cid] = BASE_CONF_CLS
            CLIP_TAU_BY_CID[cid] = CLIP_TAU_DEFAULT
    return CLS_CONF_BY_CID, CLIP_TAU_BY_CID

def clip_score_for_class(pil_crop, class_name, prompt_templates, device, clip_model, clip_processor):
    # 返回 [0,1] 的相似度均值
    texts = [t.format(class_name) for t in prompt_templates]
    inputs = clip_processor(text=texts, images=pil_crop, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        out = clip_model(**inputs)
        img = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
        txt = out.text_embeds   / out.text_embeds.norm(dim=-1, keepdim=True)
        sims = (img @ txt.t()).squeeze(0).detach().cpu().numpy()  # [-1,1]
    s01 = float(((sims.mean()) + 1.0) / 2.0)
    return s01

RELABEL_ENABLE = True
RELABEL_MARGIN = 0.06          # 标准类：替换需要的最小优势
RELABEL_MARGIN_HIGHFP = 0.03   # 高误报类（如 Butter/Milk/Cheese/bread/Tomato Puree/Jam）更容易被改走

def clip_blip_gate(outs_cls, image_rgb, allow_ids_set, coco_id_to_name,
                   CLS_CONF_BY_CID, CLIP_TAU_BY_CID,
                   prompt_templates, device, clip_model, clip_processor):
    H, W = image_rgb.shape[:2]
    kept = []
    dropped_conf, dropped_clip, relabeled = 0, 0, 0

    for p in outs_cls:
        cid = int(p["category_id"])
        sc  = float(p["score"])  # YOLO 置信度

        # 裁剪
        x, y, w, h = p["bbox"]
        x1 = max(0, int(np.floor(x)));   y1 = max(0, int(np.floor(y)))
        x2 = min(W, int(np.ceil(x + w)));y2 = min(H, int(np.ceil(h + y)))
        if x2 <= x1 or y2 <= y1:
            continue
        pil_crop = Image.fromarray(image_rgb[y1:y2, x1:x2])

        # 先算“当前预测类”的 CLIP 语义分
        cname = coco_id_to_name[cid]
        s_clip = clip_score_for_class(pil_crop, cname, prompt_templates, device, clip_model, clip_processor)

        # 语义不过线 → 直接丢（比原来更靠前）
        tau_clip = CLIP_TAU_BY_CID.get(cid, CLIP_TAU_DEFAULT)
        if s_clip < tau_clip:
            dropped_clip += 1
            continue

        # 融合：乘法（可解释且稳定）
        fused = sc * s_clip

        # 高误报类用更高的融合阈值
        fused_tau = FUSED_TAU_HIGHFP if (cname in HIGH_FP_NAMES) else FUSED_TAU

        # 软拯救策略：
        # - 如果 YOLO conf 很低（< MIN_CONF_SOFT），但 CLIP 很强（>= SALVAGE_CLIP_TAU），仍然可保留
        # - 否则要求： (sc >= 类参考阈值) 或 (fused >= 融合阈值)
        ref_conf = CLS_CONF_BY_CID.get(cid, BASE_CONF_CLS)
        keep_it = False
        if sc < MIN_CONF_SOFT:
            if s_clip >= SALVAGE_CLIP_TAU and fused >= 0.8 * fused_tau:
                keep_it = True
        else:
            if (sc >= ref_conf) or (fused >= fused_tau):
                keep_it = True

        if not keep_it:
            dropped_conf += 1
            continue

        # BLIP allow-list 软降权
        if (allow_ids_set is not None) and (cid not in allow_ids_set):
            fused *= SOFT_ALLOW_COEFF

        q = dict(p)
        q["score"] = float(fused)
        kept.append(q)

    return kept, dropped_conf, dropped_clip, relabeled


def topk_per_image_per_class(preds, k=TOPK_PER_IMAGE_PER_CLASS):
    from collections import defaultdict
    by_key = defaultdict(list)
    for p in preds:
        by_key[(int(p["image_id"]), int(p["category_id"]))].append(p)
    kept = []
    for key, items in by_key.items():
        items = sorted(items, key=lambda x: x["score"], reverse=True)[:k]
        kept.extend(items)
    return kept
def confuse_ids_for(cid, coco_id_to_name, name_to_coco):
    cname = coco_id_to_name[cid]
    alts = CONFUSE_GROUPS.get(cname, [])
    return [name_to_coco[a] for a in alts if a in name_to_coco]


if __name__ == "__main__":
    print("working")
    #---------step1: model initialization--------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    #load yolo model from best.pt weights file
    yolo_model = YOLO("/Users/a1/Downloads/best100.pt")
    #load the SAM model
    sam_checkpoint = '/Users/a1/Downloads/sam_vit_b_01ec64.pth'
    model_type = "vit_b"
    sam = sam_model_registry[model_type](checkpoint = sam_checkpoint)
    sam_predictor = SamPredictor(sam)
    #load BLIP
    blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device).eval()
    #load clip
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    #load spaCy model
    nlp = spacy.load("en_core_web_sm")
    #prompts template
    prompt_templates = [
        "a photo of a {}",
        "a close-up photo of a {}",
        "a product photo of a {}",
        "a cropped photo of a {}",
        "a {} in a fridge",
        "a {} in a bottle",
        "a {} in a refrigerator",
        "a plastic cup of {} with foil seal",
        "a cardboard carton of {}",
        "a glass jar of {}",
        "a block of {} in a wrapper"
    ]
    
    #---------step2: data and ground truth preparation--------
    data_yaml_path ='/Users/a1/Downloads/HLCV project/project/data/testdata/testsample1/data.yaml'
    print("creating COCO ground truth from YOLO labels...")
    #create COCO ground truth from YOLO labels
    coco_gt = create_coco_gt_from_yolo_labels(data_yaml_path, split="test")
    print(f"COCO ground truth created with {len(coco_gt.getImgIds())} images and {len(coco_gt.getAnnIds())} annotations.")
    #get all id list of images in the COCO ground truth
    by_canon, canon_to_primary, cid_remap = build_canonical_maps_from_gt(coco_gt)
    dup_groups = {k:v for k,v in by_canon.items() if len(v)>1}
    if dup_groups:
        print("[INFO] Found duplicated classes (merged in-memory):")
        for k, ids in dup_groups.items():
            print(f"  - {k}: ids {sorted(ids)} -> primary {canon_to_primary[k]}")
        merge_coco_gt_inplace(coco_gt, cid_remap, canon_to_primary)
    else:
        print("[INFO] No duplicate classes in GT.")
    images_ids = coco_gt.getImgIds()
    #reopen the yaml file to get the test image directory
    with open(data_yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    test_img_dir = data.get('test')
    #a list to store final predictions
    final_predictions = []
    
    #from gt categories to COCO categories
    gt_cats = coco_gt.loadCats(coco_gt.getCatIds()) #get the COCO categories from the ground truth
    name_to_coco = {c["name"]: int(c["id"]) for c in gt_cats} #create a dictionary mapping from category names to COCO category ids
    canon_to_coco = dict(name_to_coco) 
    #norm_to_coco = {norm_text(k): v for k, v in name_to_coco.items()} #normalize the category names and create a mapping to COCO category ids
    coco_id_to_name = {v: k for k, v in name_to_coco.items()}#create a mapping from COCO category ids to category names
    canon_keys = list(canon_to_coco.keys())
    CLS_CONF_BY_CID, CLIP_TAU_BY_CID = build_classwise_thresholds(canon_to_coco)
    ALL_CIDS = sorted(int(cid) for cid in canon_to_coco.values())
    #---step3:core hybrid pipline---
    print("[Notice]Starting the hybrid pipeline...")
    print(f"Number of test images: {len(images_ids)}")
    raw_total,after_total = 0, 0
    tot_drop_conf = tot_drop_clip = tot_relabeled = 0
    for img_id in tqdm(images_ids, desc="processing images"):#loop through each image id in the COCO ground truth
        #---step3.0: image loading and preprocessing---
        img_info = coco_gt.loadImgs(img_id)[0]
        img_path = os.path.join(test_img_dir, img_info['file_name'])
        image = cv2.imread(img_path)
        if image is None: continue
        #image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        H, W = image.shape[:2]
        
        #---step3.1: YOLO detection--
        outs = []
        refined_outs = []
        outs_cls = []
        ANY_CID = int(coco_gt.getCatIds()[0])  # 用 GT 里的合法类id
        yolo_results = yolo_model(image, verbose=False, conf=0.05, iou=0.7,imgsz=960,augment=True,          # 简单TTA
                                    agnostic_nms=False)
        yolo_boxes = yolo_results[0].boxes.xyxy.cpu().numpy()#array (N,4) [x_min, y_min, x_max, y_max] for each row
        yolo_classes = yolo_results[0].boxes.cls.cpu().numpy().astype(int) #array (N,1) class ids for each row
        yolo_scores = yolo_results[0].boxes.conf.cpu().numpy() #array (N,1) confidence scores for each row
        
        
        for (x1,y1,x2,y2), sc in zip(yolo_boxes, yolo_scores):
            box_xywh = sanitize_xywh(float(x1), float(y1), float(x2-x1), float(y2-y1), W, H)
            if box_xywh is None:
                continue
            outs.append({"image_id": int(img_id), "category_id": ANY_CID,
                     "bbox": box_xywh,
                     "score": float(sc)}) 
        
        # 原图：类相关 outs_cls（关键：映射到真实类
        for (x1,y1,x2,y2), c, sc in zip(yolo_boxes, yolo_classes, yolo_scores):
            yname = yolo_model.names[int(c)]
            cid   = yolo_name_to_coco_cid(yname, canon_to_coco, canon_keys)
            if cid is None:
                continue
            box_xywh = sanitize_xywh(float(x1), float(y1), float(x2-x1), float(y2-y1), W, H)
            if box_xywh is None: 
                continue
            outs_cls.append({"image_id": int(img_id), "category_id": int(cid), "bbox": box_xywh, "score": float(sc)})  
                
        #flip 
        img_flip=cv2.flip(image, 1)
        r1 = yolo_model(img_flip, verbose=False, conf=0.05, iou=0.70, imgsz=960, augment=True, agnostic_nms=False)[0]
        b1 = r1.boxes.xyxy.cpu().numpy()
        c1  = r1.boxes.cls.cpu().numpy().astype(int)
        s1 = r1.boxes.conf.cpu().numpy()
        for (x1,y1,x2,y2), sc in zip(b1, s1):
            # 反变换：水平翻转还原到原图坐标
            nx1, nx2 = W - x2, W - x1
            box_xywh = sanitize_xywh(float(nx1), float(y1), float(nx2-nx1), float(y2-y1), W, H)
            if box_xywh is None: continue
            outs.append({"image_id": int(img_id), "category_id": ANY_CID, "bbox": box_xywh, "score": float(sc)})
        # 类相关
        for (x1,y1,x2,y2), cc, sc in zip(b1, c1, s1):  
            yname = yolo_model.names[int(cc)]
            cid   = yolo_name_to_coco_cid(yname, canon_to_coco, canon_keys)
            if cid is None:
                continue
            nx1, nx2 = W - x2, W - x1
            box_xywh = sanitize_xywh(float(nx1), float(y1), float(nx2-nx1), float(y2-y1), W, H)
            if box_xywh is None: 
                continue
            outs_cls.append({"image_id": int(img_id), "category_id": int(cid), "bbox": box_xywh, "score": float(sc)})

        
        raw_total += len(outs)
        # 3) WBF 融合（类无关）
        outs = wbf_agnostic_by_image(outs, iou_thr=0.55)
        # 4) 再做一次类无关 NMS
        outs = nms_agnostic_by_image(outs, iou_thr=0.60)
        
        #sam refinement
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        sam_predictor.set_image(image_rgb)
        #sam refine
        outs_for_cls = refine_with_sam_on_outs(outs_cls, sam_predictor, W, H, iou_gate=0.50, area_min_ratio=0.20, area_max_ratio=1.20)
        #print(f"[DBG] before_gate={len(outs_cls)}")
        
        # 1.构建 BLIP allow-list（整图先验）
        pil_full = Image.fromarray(image_rgb)
        blip_cands = set(blip_candidates_for_image(pil_full, topk=12, mode='both'))
        
        # 2) 选若干“高分裁剪”再跑 BLIP 扩展先验
        sel_crops = sorted(outs_for_cls, key=lambda x: x["score"], reverse=True)[:N_CROPS_FOR_BLIP]
        for p in sel_crops:
            x, y, w, h = p["bbox"]
            x1 = max(0, int(np.floor(x)));   y1 = max(0, int(np.floor(y)))
            x2 = min(W, int(np.ceil(x + w)));y2 = min(H, int(np.ceil(h + y)))
            if x2 <= x1 or y2 <= y1: 
                continue
            pil_crop = Image.fromarray(image_rgb[y1:y2, x1:x2])
            crop_cands = blip_candidates_for_image(pil_crop, topk=BLIP_CROP_TOPK, mode='both', diverse=True)
            blip_cands.update(crop_cands)
        
        # 3.用“原图 YOLO 的类”做融合，拓宽一点先验
        cand_coco_ids = build_candidate_coco_ids(list(blip_cands), yolo_classes, yolo_model, canon_to_coco, canon_keys)
        allow_ids_set = set(cand_coco_ids) if len(cand_coco_ids) else None
        #outs = refine_with_sam_on_outs(outs, sam_predictor, W, H)
        #类无关
        # 4) allow-list 太小则回退为全类（避免漏检被过度软降权）
        if len(allow_ids_set) < 8:
            allow_ids_set = set(ALL_CIDS)

        #after_total += len(outs)
        #final_predictions.extend(outs)
        
        outs_cls = []
        dbg_drop_clip = 0
        dbg_drop_fused = 0
        dbg_salvage = 0
        cand_pool = sorted(allow_ids_set)
        
        # 按类阈值 + CLIP + BLIP 软约束
        """outs_cls, drop_by_conf, drop_by_clip, relabeled_cnt = clip_blip_gate(
            outs_cls, image_rgb, allow_ids_set, coco_id_to_name,
            CLS_CONF_BY_CID, CLIP_TAU_BY_CID,
            prompt_templates, device, clip_model, clip_processor
        )
        print(f"[DBG] after_gate={len(outs_cls)}, drop_conf={drop_by_conf}, drop_clip={drop_by_clip}")
        """
        for p in outs_for_cls:
            x, y, w, h = p["bbox"]
            x1 = max(0, int(np.floor(x)));   y1 = max(0, int(np.floor(y)))
            x2 = min(W, int(np.ceil(x + w)));y2 = min(H, int(np.ceil(y + h)))
            if x2 <= x1 or y2 <= y1:
                continue
            pil_crop = Image.fromarray(image_rgb[y1:y2, x1:x2])

            # —— 全类打分 —— #
            best_cid, s_clip, margin = clip_classify_crop_all(
                pil_crop, ALL_CIDS, coco_id_to_name,
                prompt_templates, device, clip_model, clip_processor
            )
            # 语义不过线直接丢
            tau_clip = CLIP_TAU_BY_CID.get(best_cid, CLIP_TAU_DEFAULT)
            if s_clip < tau_clip:
                dbg_drop_clip += 1
                continue
            
            # 融合 YOLO 框分（位置置信度）与 CLIP 语义分
            fused = float(p["score"]) * float(s_clip)
            cname = coco_id_to_name[best_cid]
            fused_tau = FUSED_TAU_HIGHFP if (cname in HIGH_FP_NAMES) else FUSED_TAU
        
            # 软拯救：框分很低但语义极强也可保留
            if p["score"] < MIN_CONF_SOFT and s_clip >= SALVAGE_CLIP_TAU and fused >= 0.8 * fused_tau:
                dbg_salvage += 1
                pass
            else:
                if fused < fused_tau:
                    dbg_drop_fused += 1
                    continue
            # 不在 BLIP 先验里 → 轻微降权（不是硬拒绝）
            if (allow_ids_set is not None) and (best_cid not in allow_ids_set):
                fused *= SOFT_ALLOW_COEFF

            outs_cls.append({
                "image_id": int(p["image_id"]),
                "category_id": int(best_cid),
                "bbox": [float(x), float(y), float(w), float(h)],
                "score": float(fused)
            })
        print(f"[DBG] cls_from_clip: kept={len(outs_cls)}, drop_clip={dbg_drop_clip}, drop_fused={dbg_drop_fused}, salvage={dbg_salvage}")
        # 类内 NMS + 每图每类 Top-K
        outs_cls = nms_per_class(outs_cls, iou_thr=NMS_PER_CLASS_IOU)
        outs_cls = topk_per_image_per_class(outs_cls, k=TOPK_PER_IMAGE_PER_CLASS)
        
        
        #收集类相关用于“有关 mAP + 混淆矩阵”
        if 'final_predictions_cls' not in globals():
            final_predictions_cls = []
        final_predictions_cls.extend(outs_cls)
        #tot_drop_conf += drop_by_conf
        #tot_drop_clip += drop_by_clip
        #tot_relabeled += relabeled_cnt
        #print(f"[Gate Stats] dropped_by_conf={tot_drop_conf}, dropped_by_clip={tot_drop_clip}, relabeled={tot_relabeled}")

        #convert the image to PIL format for BLIP processing
        #pil_full = Image.fromarray(image_rgb)
        #generate top-k candidates using BLIP
        """
        blip_cands = blip_candidates_for_image(pil_full, topk=5, mode='both')
        #map the candidates to COCO category ids
        cand_coco_ids = build_candidate_coco_ids(
                        blip_cands, yolo_classes, yolo_model, name_to_coco, norm_to_coco) 
        if len(cand_coco_ids) == 0:
            print(f"[DBG] empty candidates for img {img_id}, blip_cands={blip_cands[:10]}")
        
        #sam refine the yolo boxes to a tight bounding box
        sam_predictor.set_image(image_rgb)
        for i, yolo_box in enumerate(yolo_boxes): #for each detected bounding box in the YOLO results
            #---step3.2: SAM prompting---
            #point_promt = extract_prompts_from_yolo_box(yolo_box)
            #---step3.3: SAM mask generation---
            #input the point prompt to the SAM predictor and treat it as a foreground point and only generate one mask
            '''masks, scores, logits = sam_predictor.predict(
                point_coords = point_promt,
                point_labels = np.array([1],dtype=np.int32),
                multimask_output = False
            )'''
            #if masks is None or empty, skip this detection
            #if masks is None or (hasattr(masks, "shape") and masks.shape[0] == 0) or (isinstance(masks, list) and len(masks) == 0):
             #   continue  
            x1, y1,x2,y2 = map(float,yolo_box)
            box = np.array([x1,y1,x2,y2], dtype=np.float32)  # [x1,y1,x2,y2]
            masks, scores, _ = sam_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=box,
                multimask_output=True
            )
            if masks is None or len(masks) == 0:
                continue
            mask = masks[int(np.argmax(scores))]
            
            
            #---step3.4: mask to bounding box conversion---
            #convert the mask to a tight bounding box
            refined_mask = mask_to_bbox(mask)
            if refined_mask is None or refined_mask[2] <= 0 or refined_mask[3] <= 0:
                continue
            
            #---step3.5: CLIP ranking---
            x, y, w, h = refined_mask
            # 仅用于裁剪的整数坐标
            x1i, y1i = int(np.floor(x)), int(np.floor(y))
            x2i, y2i = int(np.ceil(x + w)), int(np.ceil(y + h))
            x1i, y1i = max(0, x1i), max(0, y1i)
            x2i, y2i = min(W, x2i), min(H, y2i)
            if x2i <= x1i or y2i <= y1i:
                continue
            pil_crop = Image.fromarray(image_rgb[y1i:y2i, x1i:x2i])
            
            cid, s, margin = clip_rank_crop(pil_crop, cand_coco_ids, coco_id_to_name,
                                tau=0.20, return_margin=True)

            area_ratio = (w*h) / (W*H + 1e-9)
            UNCERTAIN = (cid is None) or (s < 0.3) or (margin < 0.05) or (area_ratio < 0.01)
            if UNCERTAIN:
                crop_cands = blip_candidates_for_image(pil_crop, topk=2, mode='both')
                crop_ids = map_candidates_to_coco_ids(crop_cands, name_to_coco, norm_to_coco)
                merged = sorted(set(cand_coco_ids) | set(crop_ids))
                cid2, s2, m2 = clip_rank_crop(pil_crop, merged, coco_id_to_name, tau=0.20, return_margin=True)
                if cid2 is not None and (s2 > s + 0.02 or m2 > margin + 0.02):
                    cid, s = cid2, s2

            # 融合分数：乘积
            final_score = float(yolo_scores[i]) * float(s)
            if cid is None or final_score < 0.10:
                continue
            
            if best_cid is None:
                yolo_name = yolo_model.names[int(yolo_classes[i])]
            # 手工映射/规范化再查 name_to_coco
                mapped_name = MANUAL.get(yolo_name, yolo_name)
                best_cid = name_to_coco.get(mapped_name)

            if best_cid is None:
            # 实在没有可用类，就跳过，避免制造必然 FP
                continue
            final_predictions.append({
                "image_id": int(img_id),
                "category_id": int(cid),
                "bbox": [float(x), float(y), float(w), float(h)],
                # 分数可以融合 YOLO 与 CLIP；这里给个简单融合
                "score": final_score
            })
            """
    #---step4: evaluation--
    print("Evaluating the model performance...")
    print(f"#predictions: {len(final_predictions)}")
    #final_predictions = nms_by_image_and_class(final_predictions, iou_thr=0.5)
    print(f"#predictions (before WBF/NMS): {raw_total}")
    print(f"#predictions (after  WBF/NMS): {after_total}")
    
    pred_json_cls = 'final_prediction_coco.json'
    with open(pred_json_cls, 'w') as f:
        json.dump(final_predictions_cls if 'final_predictions_cls' in globals() else [], f)
    if 'final_predictions_cls' in globals() and len(final_predictions_cls) > 0:
        coco_dt_cls = coco_gt.loadRes(pred_json_cls)
        e_cls = COCOeval(coco_gt, coco_dt_cls, 'bbox')  # 默认 useCats=1
        e_cls.evaluate(); e_cls.accumulate(); e_cls.summarize()
        print("\n--- Class-aware (有关) ---")
        print(f"mAP@0.5:0.95: {e_cls.stats[0]:.4f}")
        print(f"mAP@0.5    : {e_cls.stats[1]:.4f}")
        print(f"mAP@0.75   : {e_cls.stats[2]:.4f}")
        
        # 混淆矩阵（默认 IoU=0.5，可改）
        CM, FP, FN, names = compute_confusion_matrix(coco_gt, final_predictions_cls, iou_thr=0.5)
        K = len(names)
        TP_diag = np.diag(CM)
        
        # 每类 Precision/Recall
        prec = np.zeros(K, dtype=np.float64)
        recl = np.zeros(K, dtype=np.float64)
        for k in range(K):
            tp = float(TP_diag[k])
            pp = tp + float(FP[k])                 # 该预测类的预测正样本数（TP+FP）
            gp = tp + float(FN[k])                 # 该GT类的真实正样本数（TP+FN）
            prec[k] = tp / (pp + 1e-9)
            recl[k] = tp / (gp + 1e-9)
            
        print("\n[Confusion Matrix] 仅统计匹配成功( IoU≥阈值 )的 GT×Pred：")
        # 打印前几行（避免太长）
        head = "{:>16s} |".format("")
        for j in range(min(K, 8)):
            head += " {:>12s}".format(names[j][:12])
        print(head)
        for i in range(min(K, 12)):
            row = "{:>16s} |".format(names[i][:16])
            for j in range(min(K, 8)):
                row += " {:12d}".format(int(CM[i, j]))
            print(row)
            
        # Top 混淆对（去掉对角）
        conf_pairs = []
        for i in range(K):
            for j in range(K):
                if i != j and CM[i, j] > 0:
                    conf_pairs.append((int(CM[i, j]), names[i], names[j]))
        conf_pairs.sort(reverse=True)
        print("\nTop-10 Confusions (GT → Pred, count):")
        for cnt, gi, pj in conf_pairs[:10]:
            print(f"{gi:>16s} → {pj:<16s} : {cnt}")
        # 每类 P/R 汇总
        print("\nPer-class Precision / Recall (IoU>=0.5 match):")
        for k in range(K):
            print(f"{names[k]:>16s}: P={prec[k]:.3f}  R={recl[k]:.3f}  TP={int(TP_diag[k])}  FP={int(FP[k])}  FN={int(FN[k])}")
        
        try:
            import csv
            with open('confusion_matrix.csv', 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['GT\\Pred'] + names)
                for i in range(K):
                    writer.writerow([names[i]] + [int(v) for v in CM[i].tolist()])
            with open('per_class_pr.csv', 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['class', 'precision', 'recall', 'TP', 'FP', 'FN'])
                for k in range(K):
                    writer.writerow([names[k], float(prec[k]), float(recl[k]), int(TP_diag[k]), int(FP[k]), int(FN[k])])
            print("\nSaved: confusion_matrix.csv, per_class_pr.csv")
        except Exception as ex:
            print("Save CSV failed:", ex)
    else:
        print("\n[WARN] 没有类相关预测（final_predictions_cls 为空），无法计算有关 mAP/混淆矩阵。")


        
    
    
    
    
    """
    if not final_predictions:
        print("[WARN] No predictions produced; skip COCOeval for now.")
        import sys; sys.exit(0)
    #generate COCO evaluation results
    coco_dt = coco_gt.loadRes(pred_json_path)
    #create a COCO evaluator
    coco_evaluator = COCOeval(coco_gt, coco_dt, 'bbox') 
    coco_evaluator.params.useCats = 0
    #coco_evaluator.params.maxDets = [1, 10, 300]
    coco_evaluator.evaluate()
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    #e2 = COCOeval(coco_gt, coco_dt, 'bbox')  # 默认 useCats = 1
    #e2.evaluate(); e2.accumulate(); e2.summarize()
    print("\n ---Final Evaluation Summary---")
    print(f"无关mAP@0.5-0.95:{coco_evaluator.stats[0]   : .4f}")
    print(f"无关mAP@0.5: {coco_evaluator.stats[1]   : .4f}")
    print(f"无关mAP@0.75: {coco_evaluator.stats[2]   : .4f}")
    #print(f"有关mAP@0.5-0.95:{e2.stats[0]   : .4f}")
    #print(f"有关mAP@0.5: {e2.stats[1]   : .4f}")
    #print(f"有关mAP@0.75: {e2.stats[2]   : .4f}")"""
    
    print("so far everything is ok")