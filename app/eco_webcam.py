import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from ultralytics import YOLO

from transformers import SegformerForSemanticSegmentation

# transformers compatibility
try:
    from transformers import SegformerImageProcessor as SegProcessor
except ImportError:
    from transformers import SegformerFeatureExtractor as SegProcessor

# torchvision for Places365
import torchvision.transforms as T
from torchvision import models


# ===================== PATHS (match your screenshot) =====================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESOURCES = PROJECT_ROOT / "resources"

SEGFORMER_DIR = RESOURCES / "segformer-b0-finetuned-ade-512-512"
PLACES365_WEIGHTS = RESOURCES / "resnet18_places365.pth.tar"

# Optional: once you train trash YOLO, put weights here
TRASH_YOLO_WEIGHTS = RESOURCES / "yolo_trash_best.pt"  # create later from your trash_dataset_yono

# datasets (not used directly for inference, but we keep paths here for clarity)
TRASH_DATASET_ZIP = RESOURCES / "trash_dataset_yono.zip"
PLANTVILLAGE_DATASET_ZIP = RESOURCES / "plantvillage_dataset_yono.zip"
# ========================================================================


# ===================== QUICK "READY" DATASETS (FOR PROTOTYPE) =====================
DATASETS = {
    "trash_taco_yolo_kaggle": {
        "type": "detect",
        "note": "TACO dataset already converted to YOLO format (Kaggle).",
        "url": "https://www.kaggle.com/datasets/vencerlanz09/taco-dataset-yolo-format",
    },
    "trash_recycle_roboflow": {
        "type": "detect",
        "note": "Small recycle/trash YOLO dataset (Roboflow Universe).",
        "url": "https://universe.roboflow.com/tranngocnhan/recycle-trash-yolo-v8",
    },
    "trash_yolov8_roboflow": {
        "type": "detect",
        "note": "YOLOv8 Trash Detection dataset (Roboflow Universe).",
        "url": "https://universe.roboflow.com/yolov8-trash-detection/yolo-v8-trash-detection-ee4016",
    },
    "garbage_yolov5_roboflow": {
        "type": "detect",
        "note": "Large-ish garbage dataset (Roboflow Universe).",
        "url": "https://universe.roboflow.com/garbage-detection-oa9nh/yolov5-garbage-detection/dataset/1",
    },
    "plantvillage_yolo_detect_kaggle": {
        "type": "detect",
        "note": "PlantVillage adapted for YOLO object detection (Kaggle).",
        "url": "https://www.kaggle.com/datasets/sebastianpalaciob/plantvillage-for-object-detection-yolo",
    },
    "rocks_roboflow": {
        "type": "detect",
        "note": "Small rocks detection dataset (Roboflow Universe).",
        "url": "https://universe.roboflow.com/yolo-training-k3lvi/rocks-detection-jsfs1",
    },
    "yolo_rock_model_roboflow": {
        "type": "detect",
        "note": "Another rocks dataset (Roboflow Universe).",
        "url": "https://universe.roboflow.com/masaai/yolo-rock-model",
    },
}
# ================================================================================


# ===================== RUNTIME SETTINGS =====================
CAM_INDEX = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# speed knobs
RUN_SEG_EVERY = 8           # run segformer every N frames
RUN_SCENE_EVERY = 10        # run places scene every N frames
INFER_WIDTH = 320           # segformer resize width (256/320/384)
SMOOTH_ALPHA = 0.25         # smoothing for vegetation fraction

# YOLO
COCO_MODEL = "yolov8n.pt"
CONF_COCO = 0.20            # LOWERED so it detects more things indoors
CONF_TRASH = 0.30

# ADE20K vegetation-ish labels (best-effort heuristic)
VEG_LABELS = {
    "tree", "grass", "plant", "potted plant", "flower",
    "palm", "bush", "shrub", "vegetation"
}

# scoring weights (tweak freely)
W_VEG = 0.70
W_SCENE = 0.25       # scene prior boost/penalty
W_TRASH = 0.35       # actual trash model penalty
W_LITTER_PROXY = 0.20  # fallback penalty using COCO "litter-like" objects (prototype)
# ===========================================================


# COCO classes we treat as "litter-like" for prototype if trash model is missing
LITTER_LIKE = {
    "bottle", "cup", "wine glass", "fork", "knife", "spoon", "bowl",
    "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "pizza", "donut", "cake",
    "cell phone", "remote", "book", "scissors",
    "plastic bag",  # not actually a COCO class but harmless if absent
}


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def sigmoid01(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


def safe_puttext(img, text, org, scale=0.55, thickness=2, color=(255, 255, 255)):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def draw_label_box(img, xyxy, label, color=(255, 255, 255)):
    x1, y1, x2, y2 = map(int, xyxy)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    safe_puttext(img, label, (x1, max(0, y1 - 6)), scale=0.5, thickness=1, color=color)


# -------------------- Places365 helpers --------------------
def _torch_load_compat(path: Path):
    try:
        return torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location="cpu")


def build_places365_resnet18():
    if not PLACES365_WEIGHTS.exists():
        raise FileNotFoundError(f"Missing Places365 weights: {PLACES365_WEIGHTS}")

    model = models.resnet18(num_classes=365)
    checkpoint = _torch_load_compat(PLACES365_WEIGHTS)

    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
            state_dict = checkpoint["state_dict"]
        elif "model" in checkpoint and isinstance(checkpoint["model"], dict):
            state_dict = checkpoint["model"]
        elif "net" in checkpoint and isinstance(checkpoint["net"], dict):
            state_dict = checkpoint["net"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    new_state = {}
    for k, v in state_dict.items():
        nk = k.replace("module.", "")
        new_state[nk] = v

    model.load_state_dict(new_state, strict=False)
    model.eval().to(DEVICE)

    preprocess = T.Compose([
        T.ToPILImage(),
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])
    return model, preprocess


def scene_prior_from_label(scene_label: str) -> float:
    s = (scene_label or "").lower()

    good = ["forest", "park", "garden", "field", "river", "lake", "mountain", "beach", "jungle", "meadow"]
    bad = ["highway", "street", "parking", "industrial", "construction", "downtown", "factory", "office", "kitchen", "bedroom", "corridor"]

    score = 0.0
    for w in good:
        if w in s:
            score += 0.6
            break
    for w in bad:
        if w in s:
            score -= 0.6
            break

    if "indoor" in s:
        score -= 0.3
    if "outdoor" in s:
        score += 0.2

    return float(max(-1.0, min(1.0, score)))


def compute_ecology_score(veg_frac: float, scene_prior: float, litter_proxy: int, trash_count: int) -> float:
    veg = clamp01(veg_frac)
    scene_term = clamp01(0.5 + 0.5 * scene_prior)

    # normalize counts
    trash_norm = clamp01(trash_count / 10.0)
    proxy_norm = clamp01(litter_proxy / 12.0)

    raw = (
        W_VEG * veg
        + W_SCENE * (scene_term - 0.5)
        - W_TRASH * trash_norm
        - W_LITTER_PROXY * proxy_norm
    )

    return sigmoid01(6.0 * (raw - 0.10))


def main():
    print(f"[INFO] DEVICE = {DEVICE}")
    print(f"[INFO] SegFormer dir: {SEGFORMER_DIR}")
    print(f"[INFO] Places365 weights: {PLACES365_WEIGHTS}")
    print(f"[INFO] Trash dataset zip present: {TRASH_DATASET_ZIP.exists()} ({TRASH_DATASET_ZIP})")
    print(f"[INFO] PlantVillage dataset zip present: {PLANTVILLAGE_DATASET_ZIP.exists()} ({PLANTVILLAGE_DATASET_ZIP})")

    if not SEGFORMER_DIR.exists():
        raise FileNotFoundError(f"SegFormer folder missing: {SEGFORMER_DIR}")

    if DEVICE == "cuda":
        torch.backends.cudnn.benchmark = True

    # -------- SegFormer load --------
    processor = SegProcessor.from_pretrained(str(SEGFORMER_DIR))
    seg_model = SegformerForSemanticSegmentation.from_pretrained(str(SEGFORMER_DIR)).to(DEVICE)
    seg_model.eval()
    if DEVICE == "cuda":
        seg_model = seg_model.half()

    id2label = seg_model.config.id2label
    veg_ids = {i for i, name in id2label.items() if str(name).lower() in VEG_LABELS}
    if not veg_ids:
        print("[WARN] No vegetation labels matched VEG_LABELS. Vegetation mask may be weak.")

    # -------- YOLO load --------
    yolo_coco = YOLO(COCO_MODEL)
    coco_names = yolo_coco.model.names

    yolo_trash = None
    trash_names = None
    if TRASH_YOLO_WEIGHTS.exists():
        yolo_trash = YOLO(str(TRASH_YOLO_WEIGHTS))
        trash_names = yolo_trash.model.names
        print(f"[INFO] Loaded trash YOLO: {TRASH_YOLO_WEIGHTS}")
    else:
        print(f"[INFO] Trash YOLO not found at {TRASH_YOLO_WEIGHTS} (trash detection disabled).")

    # -------- Places365 load (safe fallback) --------
    places_model = None
    places_preprocess = None
    try:
        places_model, places_preprocess = build_places365_resnet18()
        print("[INFO] Places365 loaded.")
    except Exception as e:
        print(f"[WARN] Places365 failed to load ({e}). Continuing without scene scoring.")
        places_model, places_preprocess = None, None

    places_categories = None
    places_cat_file = RESOURCES / "categories_places365.txt"
    if places_cat_file.exists():
        cats = []
        with open(places_cat_file, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    cats.append(" ".join(parts[1:]).replace("_", " "))
                else:
                    cats.append(line.replace("_", " "))
        if len(cats) >= 365:
            places_categories = cats[:365]
            print("[INFO] Loaded categories_places365.txt")

    # -------- Webcam --------
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam.")

    frame_idx = 0
    veg_frac_smooth = 0.0
    last_mask = None

    last_scene_label = "Scene: (disabled)"
    last_scene_prior = 0.0

    prev_t = time.time()
    fps = 0.0

    window_name = "EcoWebcam (q quit)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            h, w = frame.shape[:2]
            frame_idx += 1

            # ---------- YOLO COCO: SHOW EVERYTHING ----------
            litter_proxy = 0
            coco_results = yolo_coco.predict(frame, conf=CONF_COCO, verbose=False)[0]

            if coco_results.boxes is not None and len(coco_results.boxes) > 0:
                for b in coco_results.boxes:
                    cls_id = int(b.cls.item())
                    name = str(coco_names.get(cls_id, cls_id))

                    # draw every detection so you SEE things in a room
                    draw_label_box(frame, b.xyxy[0].tolist(), name, (255, 255, 255))

                    # proxy "trash-like" objects if you haven't trained TRASH_YOLO yet
                    if name in LITTER_LIKE:
                        litter_proxy += 1

            # ---------- YOLO Trash (optional) ----------
            trash_count = 0
            if yolo_trash is not None:
                trash_results = yolo_trash.predict(frame, conf=CONF_TRASH, verbose=False)[0]
                if trash_results.boxes is not None and len(trash_results.boxes) > 0:
                    for b in trash_results.boxes:
                        trash_count += 1
                        cls_id = int(b.cls.item())
                        tname = str(trash_names.get(cls_id, "trash"))
                        draw_label_box(frame, b.xyxy[0].tolist(), f"trash:{tname}", (255, 255, 255))

            # ---------- SegFormer vegetation (every N frames) ----------
            if (frame_idx % RUN_SEG_EVERY) == 0:
                new_w = INFER_WIDTH
                new_h = int(h * (new_w / w))
                small = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
                rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

                inputs = processor(images=rgb, return_tensors="pt")
                pixel_values = inputs["pixel_values"].to(DEVICE)
                if DEVICE == "cuda":
                    pixel_values = pixel_values.half()

                with torch.inference_mode():
                    out = seg_model(pixel_values=pixel_values)
                    logits = out.logits

                    logits = F.interpolate(
                        logits,
                        size=(rgb.shape[0], rgb.shape[1]),
                        mode="bilinear",
                        align_corners=False,
                    )
                    pred = logits.argmax(dim=1)[0].detach().cpu().numpy().astype(np.int32)

                if veg_ids:
                    mask = np.isin(pred, list(veg_ids)).astype(np.uint8)
                else:
                    mask = (pred == pred).astype(np.uint8) * 0

                veg_frac = float(mask.mean())
                veg_frac_smooth = (1.0 - SMOOTH_ALPHA) * veg_frac_smooth + SMOOTH_ALPHA * veg_frac
                last_mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

            # overlay vegetation
            if last_mask is not None:
                overlay = frame.copy()
                overlay[last_mask == 1] = (
                    overlay[last_mask == 1] * 0.70 + np.array([255, 255, 255]) * 0.30
                ).astype(np.uint8)
                frame = overlay

            # ---------- Places365 scene (every M frames; if loaded) ----------
            if places_model is not None and places_preprocess is not None and (frame_idx % RUN_SCENE_EVERY) == 0:
                rgb_full = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                x = places_preprocess(rgb_full).unsqueeze(0).to(DEVICE)
                with torch.inference_mode():
                    logits = places_model(x)
                    probs = torch.softmax(logits, dim=1)
                    top_prob, top_idx = probs[0].max(dim=0)

                idx = int(top_idx.item())
                prob = float(top_prob.item())

                if places_categories is not None and 0 <= idx < len(places_categories):
                    label = places_categories[idx]
                else:
                    label = f"Scene #{idx}"

                last_scene_label = f"{label} ({prob:.2f})"
                last_scene_prior = scene_prior_from_label(label)

            # ---------- Score ----------
            eco = compute_ecology_score(
                veg_frac=veg_frac_smooth,
                scene_prior=last_scene_prior,
                litter_proxy=litter_proxy,
                trash_count=trash_count
            )

            # ---------- FPS ----------
            now = time.time()
            dt = now - prev_t
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)
            prev_t = now

            # ---------- HUD ----------
            safe_puttext(frame, f"FPS: {fps:.1f}", (10, 22))
            safe_puttext(frame, f"Vegetation: {veg_frac_smooth*100:.1f}%", (10, 46))
            safe_puttext(frame, f"Trash (model): {trash_count}   Trash (proxy): {litter_proxy}", (10, 70))
            safe_puttext(frame, f"Scene: {last_scene_label}", (10, 94))
            safe_puttext(frame, f"Ecology Score: {eco:.2f}", (10, 118))

            cv2.imshow(window_name, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
