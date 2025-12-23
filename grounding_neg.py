import os, json, argparse
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import box_convert
from GroundingDINO.groundingdino.util.inference import load_model, load_image, predict

import clip

# -------------------- Runtime / thresholds --------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# Fewer, higher-quality proposals while debugging; tune as needed
BOX_THRESHOLD = 0.25
TEXT_THRESHOLD = 0.25
BOX_THRESHOLD_CLASS = 0.25
TEXT_THRESHOLD_CLASS = 0.25

# -------------------- Classifier --------------------
class ClipClassifier(nn.Module):
    def __init__(self, clip_model, embed_dim=512):
        super().__init__()
        self.clip_model = clip_model.to(device)
        for p in self.clip_model.parameters():
            p.requires_grad = False
        self.fc = nn.Linear(clip_model.visual.output_dim, embed_dim)
        self.classifier = nn.Linear(embed_dim, 2)

    def forward(self, images):
        with torch.no_grad():
            feats = self.clip_model.encode_image(images).float().to(device)
        x = F.relu(self.fc(feats))
        return self.classifier(x)

clip_model, preprocess = clip.load("ViT-B/32", device=device)

# Trained-or-dummy load (matches your grounding_pos.py pattern)
model_weights_path = "./data/out/classify/best_model.pth"
if os.path.exists(model_weights_path):
    print("--- Using TRAINED Classifier ---")
    print("Found 'best_model.pth', loading binary classifier...")
    binary_classifier = ClipClassifier(clip_model).to(device)
    binary_classifier.load_state_dict(
        torch.load(model_weights_path, map_location=device), strict=False
    )
    binary_classifier.eval()
else:
    print("--- Using DUMMY Classifier ---")
    print("Warning: 'best_model.pth' not found. Using DUMMY Classifier (always keep).")
    binary_classifier = None

# -------------------- Utilities --------------------
def calculate_iou(box1, box2):
    """
    IoU for xyxy (normalized) boxes: [x1,y1,x2,y2] with 0..1 coordinates.
    """
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])
    inter = max(0.0, xB - xA) * max(0.0, yB - yA)
    a1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    a2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    denom = a1 + a2 - inter
    return inter / denom if denom > 0 else 0.0

def is_valid_patch(patch, binary_classifier, preprocess, device, THRESH=0.5):
    if binary_classifier is None:
        return True, 1.0  # dummy: always keep
    if patch.size[0] <= 0 or patch.size[1] <= 0:
        return False, 0.0

    x = preprocess(patch).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = binary_classifier(x)
        # Support BCE-style [1] or softmax-style [1,2]
        if logits.ndim == 1 or logits.shape[-1] == 1:
            p1 = torch.sigmoid(logits.view(-1)[0]).item()
        else:
            probs = torch.softmax(logits, dim=1)
            p1 = probs[0, 1].item()  # assumes class1 == positive
    ok = (p1 >= THRESH)
    #print(f"[DEBUG] p_pos={p1:.3f} thr={THRESH:.2f} -> {'KEEP' if ok else 'SKIP'}")
    return ok, p1

# -------------------- Core --------------------
def process_images(text_file_path, dataset_path, model, preprocess, binary_classifier, output_folder, device, out_jsonl_path=None):
    os.makedirs(output_folder, exist_ok=True)
    boxes_dict = {}  # <-- RE-ADD THIS

    # For incremental resume (optional)
    processed = set()
    jf = None
    if out_jsonl_path is not None:
        if os.path.exists(out_jsonl_path):
            with open(out_jsonl_path, "r") as inf:
                for line in inf:
                    try:
                        rec = json.loads(line)
                        processed.add(rec["image_name"])
                    except Exception:
                        pass
        jf = open(out_jsonl_path, "a", buffering=1)  # line-buffered append

    with open(text_file_path, "r") as f:
        for line in f:
            image_name, class_name = line.strip().split("\t")
            if image_name in processed:
                continue

            print(f"Processing image: {image_name}")
            text_prompt   = class_name + " ."
            object_prompt = "object ."

            image_path = os.path.join(dataset_path, image_name)
            img = Image.open(image_path).convert("RGB")
            image_source, image = load_image(image_path)
            h, w, _ = image_source.shape

            boxes_object, logits_object, _ = predict(model, image, object_prompt, BOX_THRESHOLD, TEXT_THRESHOLD)
            boxes_class,  logits_class,  _ = predict(model, image, text_prompt,  BOX_THRESHOLD_CLASS, TEXT_THRESHOLD_CLASS)

            patches_object = box_convert(boxes_object, in_fmt="cxcywh", out_fmt="xyxy")
            patches_class  = box_convert(boxes_class,  in_fmt="cxcywh", out_fmt="xyxy")

            iou_matrix = np.zeros((len(patches_object), len(patches_class)), dtype=np.float32)
            for j, b_cls in enumerate(patches_class):
                b_cls_np = b_cls.cpu().numpy()
                bx = (b_cls_np * np.array([w, h, w, h], dtype=np.float32)).astype(int)
                pw, ph = bx[2] - bx[0], bx[3] - bx[1]
                if pw < 5 or ph < 5:  # relaxed large-box rules for negatives
                    continue
                x1_, y1_, x2_, y2_ = np.clip(bx, [0,0,0,0], [w,h,w,h])
                patch_ = img.crop((x1_, y1_, x2_, y2_))
                ok_cls, _ = is_valid_patch(patch_, binary_classifier, preprocess, device)
                if not ok_cls:
                    continue
                for i_obj, b_obj in enumerate(patches_object):
                    iou_matrix[i_obj, j] = calculate_iou(b_obj.cpu().numpy(), b_cls_np)

            top_patches = []
            for i_obj, b_obj in enumerate(patches_object):
                max_iou = np.max(iou_matrix[i_obj]) if iou_matrix.shape[1] > 0 else 0.0
                if max_iou < 0.3:  # relaxed IoU for more negatives
                    b_pix = (b_obj.cpu().numpy() * np.array([w, h, w, h], dtype=np.float32)).astype(int)
                    x1, y1, x2, y2 = b_pix
                    x1, y1, x2, y2 = max(x1,0), max(y1,0), min(x2,w), min(y2,h)
                    pw, ph = x2 - x1, y2 - y1
                    patch = img.crop((x1, y1, x2, y2))

                    ok, p_pos = is_valid_patch(patch, binary_classifier, preprocess, device)
                    reasons = []
                    if patch.size == (0, 0): reasons.append("empty_crop")
                    if not ok:               reasons.append(f"classifier_reject p={p_pos:.3f}")
                    if pw < 5:               reasons.append(f"too_narrow pw={pw}<5")
                    if ph < 5:               reasons.append(f"too_short ph={ph}<5")
                    if reasons:
                        continue

                    patch_logit = logits_object[i_obj].item()
                    top_patches.append((i_obj, patch_logit, patch))

            top_patches.sort(key=lambda x: x[1], reverse=True)
            top_3_indices = [idx for (idx, _, _) in top_patches[:3]]

            base_name, _ = os.path.splitext(image_name)
            for rank, (idx_keep, _, patch_to_save) in enumerate(top_patches[:3]):
                out_path = os.path.join(output_folder, f"{base_name}_neg_{rank}.jpg")
                patch_to_save.save(out_path, "JPEG")

            # ensure 3 indices
            if len(top_3_indices) == 0:
                default_box = torch.tensor([0, 0, 20 / w, 20 / h], device=patches_object.device).unsqueeze(0)
                patches_object = torch.cat([patches_object, default_box], dim=0)
                top_3_indices = [len(patches_object) - 1] * 3
            while len(top_3_indices) < 3:
                top_3_indices.append(top_3_indices[-1])

            # pixel-space xyxy boxes
            boxes_px = [
                (patches_object[idx].cpu().numpy() * np.array([w, h, w, h], dtype=np.float32)).tolist()
                for idx in top_3_indices
            ]

            # save in-memory
            boxes_dict[image_name] = boxes_px

            # incremental append (if requested)
            if jf is not None:
                jf.write(json.dumps({"image_name": image_name, "boxes": boxes_px}) + "\n")
                jf.flush()

    if jf is not None:
        jf.close()

    return boxes_dict


def main(args):
    model_config  = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    model_weights = "GroundingDINO/weights/groundingdino_swint_ogc.pth"

    text_file_path   = os.path.join(args.root_path, "ImageClasses_FSC147.txt")
    dataset_path     = os.path.join(args.root_path, "images_384_VarV2")
    input_json_path  = os.path.join(args.root_path, "annotation_FSC147_384.json")
    output_json_path = os.path.join(args.root_path, "annotation_FSC147_neg.json")
    output_folder    = os.path.join(args.root_path, "annotated_images_n")

    os.makedirs(output_folder, exist_ok=True)

    # Load DINO
    model = load_model(model_config, model_weights, device=device)

    # Run
    boxes_dict = process_images(text_file_path, dataset_path, model, preprocess, binary_classifier, 
    output_folder, device=device, out_jsonl_path=os.path.join(args.root_path, "neg_boxes_cache.jsonl"),)

    # Write JSON (xyxy -> 4 corners)
    with open(input_json_path, "r") as f:
        data = json.load(f)

    for image_name, boxes in boxes_dict.items():
        if image_name in data:
            new_boxes = [[[x1, y1], [x1, y2], [x2, y2], [x2, y1]] for x1, y1, x2, y2 in boxes]
            data[image_name]["box_examples_coordinates"] = new_boxes

    with open(output_json_path, "w") as f:
        json.dump(data, f, indent=4)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root_path", type=str, required=True)
    args = ap.parse_args()
    main(args)
