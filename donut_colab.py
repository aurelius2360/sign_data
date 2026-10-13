"""Fine-tune Donut (naver-clova-ix/donut-base) on Google Colab (GPU) for SIGNATURE detection alone.

Hardware:
  Optimized for NVIDIA A100 80GB VRAM (Automatic high-throughput profile: batch 16, TF32, BF16, 8 workers).
  Also auto-scales dynamically for A100 40GB, L4 24GB, or T4 16GB.

Datasets:
  Primary:  Kaggle 'monkey6769007/sign-detection' (via kagglehub)
  Fallback: Hugging Face 'monkey6769/sign_detection' (merged_dataset.zip auto-extracted)

In Google Colab (A100 80GB recommended):
  !pip install -q transformers sentencepiece protobuf kagglehub huggingface_hub tqdm Pillow
  # Auto-profile (detects A100 80GB, uses batch 16, BF16, TF32):
  !python donut_colab.py --epochs 30
  # Maximum VRAM push (~50-65 GB VRAM utilized):
  !python donut_colab.py --epochs 30 --batch-size 20 --eval-batch-size 20 --grad-accum 1
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    DonutProcessor,
    LogitsProcessor,
    LogitsProcessorList,
    VisionEncoderDecoderModel,
    get_cosine_schedule_with_warmup,
)

Image.MAX_IMAGE_PIXELS = None
TASK_TOKEN = "<s_sigdet>"
CONFIG_NAME = "donut_signature_config.json"


@dataclass
class Record:
    path: str
    boxes: list  # [[x1, y1, x2, y2], ...] absolute pixel coordinates
    labels: list  # ['signature', ...]
    width: int
    height: int


# --------------------------------------------------------------------------- Box Tokenizer & Grammar
class BoxTokenizer:
    """Encodes bounding boxes to discrete sequence tokens and decodes back."""

    def __init__(self, tokenizer, bins: int = 1000):
        self.bins = bins
        self.class_name = "signature"
        cls_token = "<c_signature>"
        loc_tokens = [f"<loc_{i}>" for i in range(bins)]

        tokenizer.add_tokens([TASK_TOKEN, cls_token] + loc_tokens, special_tokens=True)
        to_id = tokenizer.convert_tokens_to_ids

        self.task_id = to_id(TASK_TOKEN)
        self.cls_id = to_id(cls_token)
        self.eos_id = tokenizer.eos_token_id
        self.pad_id = tokenizer.pad_token_id
        self.loc_ids = [to_id(t) for t in loc_tokens]
        self.id2bin = {i: b for b, i in enumerate(self.loc_ids)}

    def encode(self, boxes, W: int, H: int, max_len: int) -> list[int]:
        # Order boxes in top-to-bottom reading order
        order = sorted(range(len(boxes)), key=lambda i: (boxes[i][1], boxes[i][0]))
        q = lambda v, s: min(self.bins - 1, max(0, int(v / s * self.bins)))
        ids = []
        for i in order[: (max_len - 1) // 5]:
            x1, y1, x2, y2 = boxes[i]
            ids += [
                self.cls_id,
                self.loc_ids[q(x1, W)],
                self.loc_ids[q(y1, H)],
                self.loc_ids[q(x2, W)],
                self.loc_ids[q(y2, H)],
            ]
        return ids + [self.eos_id]

    def decode(self, ids: list[int], probs: list[float], W: int, H: int):
        boxes, scores = [], []
        for k in range(0, len(ids) - 4, 5):
            chunk = ids[k : k + 5]
            if chunk[0] != self.cls_id or not all(t in self.id2bin for t in chunk[1:]):
                break
            x1, y1, x2, y2 = [(self.id2bin[t] + 0.5) / self.bins * s for t, s in zip(chunk[1:], (W, H, W, H))]
            boxes.append([min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)])
            # Confidence = geometric mean of token probabilities
            chunk_probs = np.clip(probs[k : k + 5], 1e-9, 1.0)
            scores.append(float(np.exp(np.mean(np.log(chunk_probs)))))
        return boxes, scores


class GrammarLogits(LogitsProcessor):
    """Enforces strict grammar: pos 0 mod 5 = <c_signature> or </s>; pos 1-4 = <loc_i>."""

    def __init__(self, bt: BoxTokenizer):
        self.head = torch.tensor([bt.cls_id, bt.eos_id])
        self.loc = torch.tensor(bt.loc_ids)

    def __call__(self, input_ids, scores):
        allowed = self.head if (input_ids.shape[1] - 1) % 5 == 0 else self.loc
        bias = torch.full_like(scores, float("-inf"))
        bias[:, allowed.to(scores.device)] = 0
        return scores + bias


# --------------------------------------------------------------------------- Dataset & Image Transforms
def letterbox(img: Image.Image, W: int, H: int, jitter: bool = False):
    """Resize keeping aspect ratio onto a white canvas."""
    scale = min(W / img.width, H / img.height)
    if jitter:
        scale *= random.uniform(0.8, 1.0)
    nw, nh = max(1, round(img.width * scale)), max(1, round(img.height * scale))
    dx = random.randint(0, W - nw) if jitter else 0
    dy = random.randint(0, H - nh) if jitter else 0
    canvas = Image.new("RGB", (W, H), (255, 255, 255))
    canvas.paste(img.resize((nw, nh), Image.BILINEAR), (dx, dy))
    return canvas, scale, (dx, dy)


def augment_document(img: Image.Image) -> Image.Image:
    """Document scanner augmentations: brightness, contrast, slight blur, grayscale."""
    if random.random() < 0.7:
        img = ImageEnhance.Brightness(img).enhance(random.uniform(0.8, 1.2))
    if random.random() < 0.7:
        img = ImageEnhance.Contrast(img).enhance(random.uniform(0.75, 1.25))
    if random.random() < 0.25:
        img = img.filter(ImageFilter.GaussianBlur(random.uniform(0.4, 1.0)))
    if random.random() < 0.2:
        img = ImageOps.grayscale(img).convert("RGB")
    return img


class DonutSignatureDataset(Dataset):
    def __init__(self, records: list[Record], bt: BoxTokenizer, size_wh: tuple[int, int], mean, std, max_len: int, train: bool):
        self.records = records
        self.bt = bt
        self.W, self.H = size_wh
        self.mean = np.array(mean, np.float32)
        self.std = np.array(std, np.float32)
        self.max_len = max_len
        self.train = train

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int):
        r = self.records[idx]
        img = Image.open(r.path).convert("RGB")
        if self.train:
            img = augment_document(img)
        canvas, scale, (dx, dy) = letterbox(img, self.W, self.H, jitter=self.train)

        # Scale original pixel boxes to canvas
        boxes = [[b[0] * scale + dx, b[1] * scale + dy, b[2] * scale + dx, b[3] * scale + dy] for b in r.boxes]
        token_ids = self.bt.encode(boxes, self.W, self.H, self.max_len)

        # Normalize pixel values
        arr = (np.asarray(canvas, dtype=np.float32) / 255.0 - self.mean) / self.std
        tensor_pixels = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
        return tensor_pixels, torch.tensor(token_ids, dtype=torch.long), idx


def pad_collate(batch):
    max_len = max(len(b[1]) for b in batch)
    padded_labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    for i, (_, ids, _) in enumerate(batch):
        padded_labels[i, : len(ids)] = ids
    return torch.stack([b[0] for b in batch]), padded_labels, [b[2] for b in batch]


# --------------------------------------------------------------------------- Dataset Loading
def _stream_download(url: str, dst_path: Path):
    """Download a large file via HTTP streaming with progress tracking."""
    import urllib.request

    print(f"[Dataset-HF] Downloading {url} ...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dst_path.with_suffix(".tmp")
    with urllib.request.urlopen(req) as resp, open(tmp_path, "wb") as out_f:
        total = int(resp.headers.get("content-length", 0))
        downloaded = 0
        chunk_size = 2 * 1024 * 1024  # 2 MB chunks
        last_pct = -1
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            out_f.write(chunk)
            downloaded += len(chunk)
            if total > 0:
                pct = int(downloaded * 100 / total)
                if pct != last_pct and pct % 10 == 0:
                    mb_down = downloaded / (1024 * 1024)
                    mb_total = total / (1024 * 1024)
                    print(f"  [HF Download] {pct}% ({mb_down:.1f} MB / {mb_total:.1f} MB)")
                    last_pct = pct
    tmp_path.replace(dst_path)
    print(f"[Dataset-HF] Download finished: {dst_path}")


def download_and_extract_hf(
    hf_repo: str = "monkey6769/sign_detection",
    filename: str = "merged_dataset.zip",
    cache_dir: Path | None = None,
) -> Path:
    """Download zipped dataset from Hugging Face and extract it."""
    import zipfile

    if cache_dir is None:
        cache_dir = Path.home() / ".cache" / "sign_detection"
    cache_dir.mkdir(parents=True, exist_ok=True)
    extract_dir = cache_dir / "hf_extracted"
    marker = extract_dir / ".extract_complete"

    # 1. Fast check if already extracted
    if marker.is_file():
        root = find_dataset_root(extract_dir)
        print(f"[Dataset-HF] Found existing extracted dataset at: {root}")
        return root

    # 2. Check if archive exists locally in current folder or cache
    local_zip = Path(filename)
    cached_zip = cache_dir / filename
    zip_path = None

    if local_zip.is_file():
        print(f"[Dataset-HF] Using local archive: {local_zip.resolve()}")
        zip_path = local_zip
    elif cached_zip.is_file() and cached_zip.stat().st_size > 100_000:
        print(f"[Dataset-HF] Using cached archive: {cached_zip}")
        zip_path = cached_zip
    else:
        # Download from Hugging Face
        print(f"[Dataset-HF] Fetching '{filename}' from Hugging Face: '{hf_repo}'...")
        try:
            from huggingface_hub import hf_hub_download

            downloaded = hf_hub_download(
                repo_id=hf_repo,
                filename=filename,
                repo_type="dataset",
                local_dir=str(cache_dir),
            )
            zip_path = Path(downloaded)
            print(f"[Dataset-HF] Downloaded via huggingface_hub: {zip_path}")
        except Exception as hf_err:
            print(f"[Dataset-HF] huggingface_hub error ({hf_err}). Trying direct HTTPS stream...")
            url = f"https://huggingface.co/datasets/{hf_repo}/resolve/main/{filename}"
            zip_path = cached_zip
            _stream_download(url, zip_path)

    # 3. Extract the zip archive
    print(f"[Dataset-HF] Extracting archive: {zip_path} -> {extract_dir} ...")
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)
    marker.write_text("ok", encoding="utf-8")
    print(f"[Dataset-HF] Extraction complete! Target: {extract_dir}")
    return find_dataset_root(extract_dir)


def download_dataset(
    dataset_handle: str = "monkey6769007/sign-detection",
    hf_repo: str = "monkey6769/sign_detection",
    local_dir: str | None = None,
    use_hf: bool = False,
) -> Path:
    """Download or locate dataset with Kaggle primary source and Hugging Face zip fallback."""
    if local_dir and Path(local_dir).exists():
        print(f"[Dataset] Using specified local dataset: {local_dir}")
        return Path(local_dir)

    # 1. Local merged_dataset folder
    local_default = Path("merged_dataset")
    if (local_default / "annotations_coco.json").is_file() or (local_default / "train.json").is_file():
        print("[Dataset] Found local 'merged_dataset' directory.")
        return local_default

    # 2. Local merged_dataset.zip archive
    if Path("merged_dataset.zip").is_file():
        print("[Dataset] Found local 'merged_dataset.zip'. Extracting...")
        return download_and_extract_hf(hf_repo=hf_repo)

    # 3. Explicit request for Hugging Face
    if use_hf:
        print(f"[Dataset] Explicitly requested Hugging Face dataset: '{hf_repo}'")
        return download_and_extract_hf(hf_repo=hf_repo)

    # 4. Primary: Try Kaggle download via kagglehub
    if dataset_handle:
        print(f"[Dataset] Attempting primary download from Kaggle: '{dataset_handle}'...")
        try:
            import kagglehub

            p = kagglehub.dataset_download(dataset_handle)
            print(f"[Dataset] Successfully downloaded from Kaggle: {p}")
            return Path(p)
        except Exception as k_err:
            print(f"[Dataset] Kaggle download unavailable ({k_err}).")
            print(f"[Dataset] Triggering fallback to Hugging Face: '{hf_repo}'...")

    # 5. Fallback: Download and extract from Hugging Face repository
    return download_and_extract_hf(hf_repo=hf_repo)


def find_dataset_root(root: Path) -> Path:
    if (root / "train.json").is_file() or (root / "annotations_coco.json").is_file():
        return root
    if (root / "merged_dataset" / "train.json").is_file() or (root / "merged_dataset" / "annotations_coco.json").is_file():
        return root / "merged_dataset"
    for sub in root.rglob("train.json"):
        return sub.parent
    return root


def load_signature_records(json_path: Path, dataset_root: Path) -> list[Record]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    sig_cat_id = None
    for c in data.get("categories", []):
        if str(c["name"]).lower() == "signature":
            sig_cat_id = c["id"]
            break
    if sig_cat_id is None:
        sig_cat_id = 4  # default in merged_dataset

    anns_by_img = {}
    for a in data.get("annotations", []):
        if a.get("category_id") == sig_cat_id and a.get("bbox"):
            x, y, w, h = a["bbox"]
            anns_by_img.setdefault(a["image_id"], []).append([x, y, x + w, y + h])

    recs = []
    for img in data.get("images", []):
        fn = img["file_name"]
        full_path = dataset_root / fn
        if not full_path.is_file():
            # Check direct images folder
            full_path = dataset_root / "images" / Path(fn).name
        if not full_path.is_file():
            continue

        boxes = anns_by_img.get(img["id"], [])
        recs.append(
            Record(
                path=str(full_path),
                boxes=boxes,
                labels=["signature"] * len(boxes),
                width=int(img.get("width", 0)),
                height=int(img.get("height", 0)),
            )
        )
    return recs


# --------------------------------------------------------------------------- Evaluation Metrics
def box_iou(a, b) -> float:
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def compute_voc_ap(preds, gts, iou_thr=0.5):
    """Computes standard Pascal VOC AP@0.5 for signature detection."""
    dets = []
    for i, (p_boxes, p_scores) in enumerate(preds):
        for b, s in zip(p_boxes, p_scores):
            dets.append((s, i, b))
    dets.sort(key=lambda d: -d[0])

    n_gt = sum(len(g) for g in gts)
    if n_gt == 0:
        return {"AP50": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    used = [set() for _ in gts]
    tps, fps = [], []
    for s, img_idx, p_box in dets:
        gt_boxes = gts[img_idx]
        best_iou, best_j = 0.0, -1
        for j, g_box in enumerate(gt_boxes):
            cur_iou = box_iou(p_box, g_box)
            if cur_iou > best_iou:
                best_iou = cur_iou
                best_j = j
        if best_iou >= iou_thr and best_j not in used[img_idx]:
            used[img_idx].add(best_j)
            tps.append(1)
            fps.append(0)
        else:
            tps.append(0)
            fps.append(1)

    tp_cum = np.cumsum(tps) if tps else np.zeros(0)
    fp_cum = np.cumsum(fps) if fps else np.zeros(0)
    rec = tp_cum / max(n_gt, 1)
    prec = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)

    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]
    ap = float(np.sum((mrec[1:] - mrec[:-1]) * mpre[1:]))

    p = float(tp_cum[-1] / max(tp_cum[-1] + fp_cum[-1], 1)) if len(tp_cum) else 0.0
    r = float(tp_cum[-1] / n_gt) if len(tp_cum) else 0.0
    f1 = 2 * p * r / max(p + r, 1e-9)
    return {"AP50": ap, "precision": p, "recall": r, "f1": f1}


@torch.no_grad()
def evaluate_donut(model, bt: BoxTokenizer, loader, records, args, device, amp_dtype) -> dict:
    model.eval()
    preds, gts = [], []

    for pixels, _, indices in tqdm(loader, desc="evaluating", leave=False):
        pixels = pixels.to(device, non_blocking=True)
        start_tokens = torch.full((pixels.size(0), 1), bt.task_id, device=device)

        with torch.autocast(device.type, dtype=amp_dtype or torch.float32, enabled=amp_dtype is not None):
            out = model.generate(
                pixels,
                decoder_input_ids=start_tokens,
                max_length=args.max_len,
                num_beams=1,
                do_sample=False,
                logits_processor=LogitsProcessorList([GrammarLogits(bt)]),
                output_scores=True,
                return_dict_in_generate=True,
            )

        logp = model.compute_transition_scores(out.sequences, out.scores, normalize_logits=True)
        seqs = out.sequences[:, 1:].tolist()
        probs = logp.float().exp().tolist()

        for k, idx in enumerate(indices):
            r = records[idx]
            scale = min(args.width / r.width, args.height / r.height)
            pred_boxes, pred_scores = bt.decode(seqs[k], probs[k], args.width, args.height)
            orig_boxes = [[coord / scale for coord in b] for b in pred_boxes]
            preds.append((orig_boxes, pred_scores))
            gts.append(r.boxes)

    return compute_voc_ap(preds, gts, iou_thr=0.5)


def save_complete_model(
    model: VisionEncoderDecoderModel,
    processor: DonutProcessor,
    bt: BoxTokenizer,
    out_dir: Path,
    args: argparse.Namespace,
    metrics: dict | None = None,
    archive_name: str | None = "donut_signature_model",
    drive_dir: str | None = None,
) -> Path:
    """Save complete Hugging Face model, processor, metadata config, predict.py, and standalone ZIP."""
    import shutil

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[Save Model] Saving complete fine-tuned model package to: {out_dir} ...")

    # 1. Save Hugging Face weights and processor
    model.save_pretrained(out_dir)
    processor.save_pretrained(out_dir)
    if hasattr(processor, "tokenizer") and processor.tokenizer:
        processor.tokenizer.save_pretrained(out_dir)

    # 2. Save metadata configuration for direct downstream loading
    meta = {
        "model_type": "donut-signature-detector",
        "base_model": args.init,
        "class_name": "signature",
        "task_token": TASK_TOKEN,
        "class_token": "<c_signature>",
        "coord_bins": bt.bins,
        "image_width": args.width,
        "image_height": args.height,
        "max_length": args.max_len,
        "metrics": metrics or {},
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (out_dir / CONFIG_NAME).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    if metrics:
        (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # 3. Write a self-contained inference script inside the model folder
    inference_script = '''"""Standalone inference script for the fine-tuned Donut signature detector."""
import argparse
import json
from pathlib import Path
import torch
import numpy as np
from PIL import Image, ImageDraw
from transformers import DonutProcessor, VisionEncoderDecoderModel

def predict(img_path, model_dir=".", out_path="prediction.png"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    meta_p = Path(model_dir) / "donut_signature_config.json"
    meta = json.loads(meta_p.read_text(encoding="utf-8")) if meta_p.exists() else {
        "image_width": 960, "image_height": 1280, "max_length": 256,
        "task_token": "<s_sigdet>", "class_token": "<c_signature>", "coord_bins": 1000
    }
    W, H, max_len = meta["image_width"], meta["image_height"], meta["max_length"]
    processor = DonutProcessor.from_pretrained(model_dir)
    model = VisionEncoderDecoderModel.from_pretrained(model_dir).to(device).eval()

    orig = Image.open(img_path).convert("RGB")
    scale = min(W / orig.width, H / orig.height)
    canvas = Image.new("RGB", (W, H), (255, 255, 255))
    canvas.paste(orig.resize((round(orig.width * scale), round(orig.height * scale)), Image.BILINEAR), (0, 0))
    ip = processor.image_processor
    arr = (np.asarray(canvas, dtype=np.float32) / 255.0 - ip.image_mean) / ip.image_std
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)

    start_id = processor.tokenizer.convert_tokens_to_ids(meta["task_token"])
    cls_id = processor.tokenizer.convert_tokens_to_ids(meta["class_token"])
    loc_ids = [processor.tokenizer.convert_tokens_to_ids(f"<loc_{i}>") for i in range(meta["coord_bins"])]
    id2bin = {i: b for b, i in enumerate(loc_ids)}

    with torch.no_grad():
        out = model.generate(tensor, decoder_input_ids=torch.full((1, 1), start_id, device=device), max_length=max_len)
    tokens = out.sequences[0, 1:].tolist()
    boxes = []
    for k in range(0, len(tokens) - 4, 5):
        chunk = tokens[k:k+5]
        if chunk[0] != cls_id or not all(t in id2bin for t in chunk[1:]):
            break
        x1, y1, x2, y2 = [(id2bin[t] + 0.5) / meta["coord_bins"] * s for t, s in zip(chunk[1:], (W, H, W, H))]
        boxes.append([x1 / scale, y1 / scale, x2 / scale, y2 / scale])

    draw = ImageDraw.Draw(orig)
    for b in boxes:
        draw.rectangle(b, outline=(255, 0, 0), width=max(3, orig.width // 400))
        draw.text((b[0], max(0, b[1] - 16)), "signature", fill=(255, 0, 0))
    orig.save(out_path)
    print(f"Found {len(boxes)} signature(s). Annotated image saved to: {out_path}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("image", help="Path to input document image")
    p.add_argument("--model-dir", default=".", help="Directory containing fine-tuned Donut weights")
    p.add_argument("--out", default="prediction.png", help="Path for saved annotated output image")
    args, _ = p.parse_known_args()
    predict(args.image, args.model_dir, out_path=args.out)
'''
    (out_dir / "predict.py").write_text(inference_script, encoding="utf-8")

    # 4. Create standalone ZIP archive of the saved model
    zip_path = None
    if archive_name:
        zip_base = out_dir.parent / archive_name
        print(f"[Save Model] Compressing model into standalone ZIP archive: {zip_base}.zip ...")
        shutil.make_archive(str(zip_base), "zip", out_dir)
        zip_path = Path(f"{zip_base}.zip")
        if zip_path.is_file():
            mb_size = zip_path.stat().st_size / (1024**2)
            print(f"[Save Model] Successfully created archive: {zip_path} ({mb_size:.1f} MB)")

    # 5. Optional Google Drive backup
    if drive_dir:
        drive_path = Path(drive_dir)
        try:
            drive_path.mkdir(parents=True, exist_ok=True)
            print(f"[Google Drive] Backing up model to Google Drive: {drive_path} ...")
            for item in out_dir.iterdir():
                dst = drive_path / item.name
                if item.is_file():
                    shutil.copy2(item, dst)
                elif item.is_dir():
                    shutil.copytree(item, dst, dirs_exist_ok=True)
            if zip_path and zip_path.is_file():
                shutil.copy2(zip_path, drive_path / zip_path.name)
            print(f"[Google Drive] Backup complete: {drive_path}")
        except Exception as e:
            print(f"[Google Drive] Warning: Could not backup to Drive ({e})")

    return out_dir


# --------------------------------------------------------------------------- Main Training Pipeline
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--init", default="naver-clova-ix/donut-base", help="Pretrained Donut model identifier")
    p.add_argument("--epochs", type=int, default=30, help="Training epochs")
    p.add_argument("--batch-size", type=int, default=None, help="Batch size (Auto: 16 on A100 80GB, 8 on A100 40GB, 2 on T4)")
    p.add_argument("--eval-batch-size", type=int, default=None, help="Eval batch size (Auto: 16 on A100 80GB, 4 on T4)")
    p.add_argument("--grad-accum", type=int, default=None, help="Grad accumulation (Auto: 1 on A100 80GB, 8 on T4)")
    p.add_argument("--lr", type=float, default=3e-5, help="Learning rate")
    p.add_argument("--width", type=int, default=960, help="Canvas width (multiple of 32)")
    p.add_argument("--height", type=int, default=1280, help="Canvas height (multiple of 32)")
    p.add_argument("--max-len", type=int, default=256, help="Max output tokens (5 tokens per signature + eos)")
    p.add_argument("--dataset", default="monkey6769007/sign-detection", help="Kaggle dataset handle (primary)")
    p.add_argument("--hf-dataset", default="monkey6769/sign_detection", help="Hugging Face dataset repo (fallback)")
    p.add_argument("--use-hf", action="store_true", help="Directly download from Hugging Face instead of Kaggle")
    p.add_argument("--data-dir", default=None, help="Local dataset directory if already downloaded")
    p.add_argument("--output-dir", default="runs/donut_colab", help="Output directory for checkpoints")
    p.add_argument("--workers", type=int, default=None, help="DataLoader workers (Auto: 8 on A100, 2 on T4)")
    p.add_argument("--eval-every", type=int, default=1, help="Evaluate validation set every N epochs")
    p.add_argument("--drive-dir", default=None, help="Google Drive backup folder path")
    p.add_argument("--save-drive", action="store_true", help="Auto-backup to Google Drive if /content/drive/MyDrive exists")
    args, _ = p.parse_known_args()

    # 1. Hardware Detection & A100 80GB Auto-Tuning Profile
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        # Enable TensorFloat-32 (TF32) and cuDNN benchmark for A100
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

        cpu_cnt = os.cpu_count() or 4
        if vram_gb >= 70:  # NVIDIA A100 80GB
            profile = "NVIDIA A100 80GB High-Throughput Profile"
            default_batch = 16
            default_eval_batch = 16
            default_accum = 1
            default_workers = min(cpu_cnt, 8)
        elif vram_gb >= 35:  # NVIDIA A100 40GB
            profile = "NVIDIA A100 40GB Profile"
            default_batch = 8
            default_eval_batch = 8
            default_accum = 2
            default_workers = min(cpu_cnt, 6)
        elif vram_gb >= 20:  # L4 24GB, RTX 3090/4090
            profile = "24GB GPU Profile (L4 / RTX)"
            default_batch = 4
            default_eval_batch = 8
            default_accum = 4
            default_workers = min(cpu_cnt, 4)
        else:  # Entry GPU (T4 16GB)
            profile = "T4 16GB Profile"
            default_batch = 2
            default_eval_batch = 4
            default_accum = 8
            default_workers = 2

        print(f"[Hardware Profile] Detected: {gpu_name} ({vram_gb:.1f} GB VRAM) -> {profile}")
        print(f"  TF32 Tensor Cores:       Enabled (matmul + cuDNN)")
        print(f"  cuDNN Benchmark:         Enabled (autotuned convolution kernels)")
        print(f"  Mixed Precision:         {amp_dtype} (Hardware accelerated)")
    else:
        device = torch.device("cpu")
        gpu_name = "CPU"
        vram_gb = 0.0
        amp_dtype = None
        default_batch = 1
        default_eval_batch = 1
        default_accum = 16
        default_workers = 0
        print("[WARNING] Running on CPU! Donut training on CPU is slow; enable GPU in Colab.")

    batch_size = args.batch_size if args.batch_size is not None else default_batch
    eval_batch_size = args.eval_batch_size if args.eval_batch_size is not None else default_eval_batch
    grad_accum = args.grad_accum if args.grad_accum is not None else default_accum
    workers = args.workers if args.workers is not None else default_workers

    print(f"\n[Training Configuration]")
    print(f"  Batch Size:              {batch_size} (Effective batch: {batch_size * grad_accum})")
    print(f"  Eval Batch Size:         {eval_batch_size}")
    print(f"  Gradient Accum Steps:    {grad_accum}")
    print(f"  DataLoader Workers:      {workers}")
    print(f"  Image Canvas:            {args.width}x{args.height}")
    print(f"  Eval Frequency:          Every {args.eval_every} epoch(s)")

    # 2. Download and Load Dataset
    raw_dir = download_dataset(args.dataset, args.hf_dataset, args.data_dir, args.use_hf)
    root = find_dataset_root(raw_dir)
    print(f"[Dataset] Resolved dataset root: {root}")

    train_recs = load_signature_records(root / "train.json", root)
    val_recs = load_signature_records(root / "val.json", root)
    test_recs = load_signature_records(root / "test.json", root)

    train_boxes = sum(len(r.boxes) for r in train_recs)
    val_boxes = sum(len(r.boxes) for r in val_recs)
    test_boxes = sum(len(r.boxes) for r in test_recs)
    print(f"[Dataset] Loaded splits for 'signature':")
    print(f"  Train: {len(train_recs)} images ({train_boxes} signatures)")
    print(f"  Val:   {len(val_recs)} images ({val_boxes} signatures)")
    print(f"  Test:  {len(test_recs)} images ({test_boxes} signatures)")

    # 3. Model & Processor Setup
    print(f"\n[Model] Loading Donut pretrained weights: {args.init}...")
    processor = DonutProcessor.from_pretrained(args.init)
    model = VisionEncoderDecoderModel.from_pretrained(args.init)

    bt = BoxTokenizer(processor.tokenizer, bins=1000)
    model.decoder.resize_token_embeddings(len(processor.tokenizer))

    # Configure image processor and generation parameters
    processor.image_processor.size = {"height": args.height, "width": args.width}
    model.config.encoder.image_size = [args.height, args.width]
    for cfg in (model.config, model.generation_config):
        cfg.decoder_start_token_id = bt.task_id
        cfg.pad_token_id = bt.pad_id
        cfg.eos_token_id = bt.eos_id
    model.generation_config.max_length = args.max_len

    model.to(device)

    # 4. High-Performance DataLoaders
    ip = processor.image_processor
    train_ds = DonutSignatureDataset(train_recs, bt, (args.width, args.height), ip.image_mean, ip.image_std, args.max_len, train=True)
    val_ds = DonutSignatureDataset(val_recs, bt, (args.width, args.height), ip.image_mean, ip.image_std, args.max_len, train=False)
    test_ds = DonutSignatureDataset(test_recs, bt, (args.width, args.height), ip.image_mean, ip.image_std, args.max_len, train=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        collate_fn=pad_collate,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
        prefetch_factor=2 if workers > 0 else None,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=workers,
        collate_fn=pad_collate,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
        prefetch_factor=2 if workers > 0 else None,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=workers,
        collate_fn=pad_collate,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
        prefetch_factor=2 if workers > 0 else None,
    )

    # 5. Optimizer (Fused AdamW if available) & Scheduler
    fused_available = False
    if device.type == "cuda":
        try:
            test_opt = torch.optim.AdamW([torch.nn.Parameter(torch.empty(1, device=device))], fused=True)
            fused_available = True
        except Exception:
            fused_available = False

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, fused=fused_available)
    print(f"[Optimizer] AdamW (fused={fused_available}, lr={args.lr:.2e}, weight_decay=0.01)")

    total_steps = math.ceil(len(train_loader) / grad_accum) * args.epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(0.05 * total_steps), total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    best_map = -1.0

    print(f"\n[Training] Starting Donut fine-tuning for {args.epochs} epochs ({total_steps} total optimizer steps)...\n")

    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")

        for step, (px, labels, _) in enumerate(pbar):
            px = px.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.autocast(device.type, dtype=amp_dtype or torch.float32, enabled=amp_dtype is not None):
                loss = model(pixel_values=px, labels=labels).loss

            scaled_loss = loss / grad_accum
            if scaler.is_enabled():
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            if (step + 1) % grad_accum == 0 or (step + 1) == len(train_loader):
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            running_loss += loss.item()
            cur_vram = f"{torch.cuda.memory_allocated() / (1024**3):.1f}GB" if device.type == "cuda" else "cpu"
            pbar.set_postfix(
                loss=f"{running_loss / (step + 1):.4f}",
                lr=f"{scheduler.get_last_lr()[0]:.2e}",
                vram=cur_vram,
            )

        epoch_loss = running_loss / max(len(train_loader), 1)
        vram_peak = torch.cuda.max_memory_allocated() / (1024**3) if device.type == "cuda" else 0.0
        vram_pct = (vram_peak / vram_gb * 100) if vram_gb > 0 else 0.0
        print(f"Epoch {epoch + 1}: Loss = {epoch_loss:.4f} ({time.time() - t0:.0f}s) | Peak VRAM: {vram_peak:.1f} GB ({vram_pct:.1f}%)")

        # Validate according to --eval-every schedule (or final epoch)
        should_eval = ((epoch + 1) % args.eval_every == 0) or (epoch + 1 == args.epochs)
        if should_eval:
            val_m = evaluate_donut(model, bt, val_loader, val_recs, args, device, amp_dtype)
            print(f"  Validation: mAP@50 = {val_m['AP50']:.4f} | Prec = {val_m['precision']:.4f} | Rec = {val_m['recall']:.4f} | F1 = {val_m['f1']:.4f}")

            # Save best checkpoint
            if val_m["AP50"] > best_map:
                best_map = val_m["AP50"]
                best_path = out_root / "best"
                best_path.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(best_path)
                processor.save_pretrained(best_path)
                (best_path / "metrics.json").write_text(json.dumps(val_m, indent=2))
                print(f"  [*] New best model saved -> {best_path} (mAP@50: {best_map:.4f})")

        # Always maintain latest checkpoint
        last_path = out_root / "last"
        last_path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(last_path)
        processor.save_pretrained(last_path)

    # 6. Evaluation on Test Split
    print("\n[Evaluation] Evaluating best model on test split...")
    best_ckpt = out_root / "best" if (out_root / "best").exists() else out_root / "last"
    eval_model = VisionEncoderDecoderModel.from_pretrained(best_ckpt).to(device)
    test_m = evaluate_donut(eval_model, bt, test_loader, test_recs, args, device, amp_dtype)
    print(f"  Test Split: mAP@50 = {test_m['AP50']:.4f} | Prec = {test_m['precision']:.4f} | Rec = {test_m['recall']:.4f} | F1 = {test_m['f1']:.4f}")
    (best_ckpt / "test_metrics.json").write_text(json.dumps(test_m, indent=2))

    # 7. Complete Model Packaging & Saving
    drive_dest = args.drive_dir
    if not drive_dest and args.save_drive and Path("/content/drive/MyDrive").exists():
        drive_dest = "/content/drive/MyDrive/donut_signature_model"

    saved_dir = save_complete_model(
        model=eval_model,
        processor=processor,
        bt=bt,
        out_dir=best_ckpt,
        args=args,
        metrics=test_m,
        archive_name="donut_signature_model",
        drive_dir=drive_dest,
    )

    # 8. Sample Inference Demo
    if test_recs:
        sample = test_recs[0]
        print(f"\n[Inference] Running sample inference on: {Path(sample.path).name}")
        img = Image.open(sample.path).convert("RGB")
        canvas, scale, _ = letterbox(img, args.width, args.height)

        arr = (np.asarray(canvas, dtype=np.float32) / 255.0 - ip.image_mean) / ip.image_std
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
        start_tokens = torch.full((1, 1), bt.task_id, device=device)

        with torch.no_grad():
            out = eval_model.generate(
                tensor,
                decoder_input_ids=start_tokens,
                max_length=args.max_len,
                logits_processor=LogitsProcessorList([GrammarLogits(bt)]),
                output_scores=True,
                return_dict_in_generate=True,
            )
        logp = eval_model.compute_transition_scores(out.sequences, out.scores, normalize_logits=True)
        pred_boxes, pred_scores = bt.decode(out.sequences[0, 1:].tolist(), logp[0].float().exp().tolist(), args.width, args.height)

        draw = ImageDraw.Draw(img)
        for b, s in zip(pred_boxes, pred_scores):
            orig_b = [coord / scale for coord in b]
            draw.rectangle(orig_b, outline=(255, 0, 0), width=max(3, img.width // 400))
            draw.text((orig_b[0], max(0, orig_b[1] - 16)), f"signature {s:.2f}", fill=(255, 0, 0))

        pred_img_path = out_root / "sample_prediction.png"
        img.save(pred_img_path)
        print(f"[Inference] Annotated sample prediction saved to: {pred_img_path}")

    # 9. Summary & Download Helper
    zip_archive = out_root / "donut_signature_model.zip"
    print("\n" + "=" * 65)
    print("   DONUT SIGNATURE TRAINING & EXPORT COMPLETE")
    print("=" * 65)
    print(f"  Model directory:    {saved_dir}")
    if zip_archive.is_file():
        print(f"  Downloadable ZIP:   {zip_archive} ({zip_archive.stat().st_size / (1024**2):.1f} MB)")
    if drive_dest:
        print(f"  Google Drive path:  {drive_dest}")
    print("\nTo download the trained model directly to your computer in Google Colab:")
    print("  from google.colab import files")
    print(f"  files.download('{zip_archive}')")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
