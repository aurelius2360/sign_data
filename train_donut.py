"""Fine-tune Donut (naver-clova-ix/donut-base) as an OCR-free signature *detector*.

Donut is a generative image->text model, so detection is cast as sequence generation
(Pix2Seq style). Each box becomes five new tokens

    <c_signature><loc_x1><loc_y1><loc_x2><loc_y2>

with coordinates quantised into --bins location tokens on a letterboxed (W x H) canvas.
Decoding is grammar-constrained (class/eos, then exactly 4 locations), so every output parses,
and box confidence = geometric mean of the 5 token probabilities (used for mAP@0.5).

Examples
  python train_donut.py --data D:/datasets/signverod --data tobacco=D:/datasets/tobacco800 \
      --hf-data tech4humans/signature-detection --epochs 30 --batch-size 2 --grad-accum 8
  python train_donut.py --resume runs/donut/last ...same data args...
  python train_donut.py --predict page.png --checkpoint runs/donut/best
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (DonutProcessor, LogitsProcessor, LogitsProcessorList,
                          VisionEncoderDecoderModel, get_cosine_schedule_with_warmup)

try:
    from signature_data import Record, add_data_args, normalize_label, splits_from_args
except ImportError:
    from dataclasses import dataclass

    @dataclass
    class Record:
        path: str
        boxes: list
        labels: list
        split: str | None = None
        source: str = ""
        width: int = 0
        height: int = 0

    def normalize_label(name) -> str:
        n = "".join(c for c in str(name).lower() if c.isalpha())
        return "signature" if "signature" in n else n

    def add_data_args(p):
        g = p.add_argument_group("data")
        g.add_argument("--data", action="append", default=[], help="Dataset folder or COCO json")
        g.add_argument("--hf-data", action="append", default=[], help="Hugging Face dataset id")
        g.add_argument("--classes", nargs="+", default=["signature"], help="Classes to train on")
        g.add_argument("--dataset", default="monkey6769007/sign-detection", help="Kaggle dataset handle")
        g.add_argument("--hf-dataset", default="monkey6769/sign_detection", help="Hugging Face dataset handle")
        g.add_argument("--use-hf", action="store_true", help="Directly use Hugging Face dataset")
        g.add_argument("--data-dir", default=None, help="Local dataset directory")
        g.add_argument("--val-ratio", type=float, default=0.1)
        g.add_argument("--test-ratio", type=float, default=0.1)
        g.add_argument("--seed", type=int, default=42)

    def _stream_download(url: str, dst_path: Path):
        import urllib.request
        print(f"[Dataset-HF] Downloading {url} ...")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst_path.with_suffix(".tmp")
        with urllib.request.urlopen(req) as resp, open(tmp, "wb") as f:
            total = int(resp.headers.get("content-length", 0))
            down, chunk_size, last_pct = 0, 2 * 1024 * 1024, -1
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                down += len(chunk)
                if total > 0:
                    pct = int(down * 100 / total)
                    if pct != last_pct and pct % 10 == 0:
                        print(f"  [HF Download] {pct}% ({down / (1024**2):.1f} MB / {total / (1024**2):.1f} MB)")
                        last_pct = pct
        tmp.replace(dst_path)

    def _download_and_extract_hf(hf_repo: str = "monkey6769/sign_detection") -> Path:
        import zipfile
        cache_dir = Path.home() / ".cache" / "sign_detection"
        extract_dir = cache_dir / "hf_extracted"
        marker = extract_dir / ".extract_complete"
        if marker.is_file():
            return _find_root(extract_dir)
        local_zip = Path("merged_dataset.zip")
        cached_zip = cache_dir / "merged_dataset.zip"
        zip_path = local_zip if local_zip.is_file() else cached_zip
        if not zip_path.is_file():
            print(f"[Dataset-HF] Fetching from Hugging Face: '{hf_repo}'...")
            try:
                from huggingface_hub import hf_hub_download
                zip_path = Path(hf_hub_download(repo_id=hf_repo, filename="merged_dataset.zip", repo_type="dataset", local_dir=str(cache_dir)))
            except Exception:
                url = f"https://huggingface.co/datasets/{hf_repo}/resolve/main/merged_dataset.zip"
                _stream_download(url, cached_zip)
                zip_path = cached_zip
        print(f"[Dataset-HF] Extracting {zip_path} ...")
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)
        marker.write_text("ok", encoding="utf-8")
        return _find_root(extract_dir)

    def _find_root(p: Path) -> Path:
        if (p / "train.json").is_file() or (p / "annotations_coco.json").is_file():
            return p
        if (p / "merged_dataset" / "train.json").is_file():
            return p / "merged_dataset"
        for sub in p.rglob("train.json"):
            return sub.parent
        return p

    def _load_coco_split(json_path: Path, root: Path) -> list[Record]:
        if not json_path.is_file():
            return []
        data = json.loads(json_path.read_text(encoding="utf-8"))
        sig_id = next((c["id"] for c in data.get("categories", []) if "signature" in str(c["name"]).lower()), 4)
        anns = {}
        for a in data.get("annotations", []):
            if a.get("category_id") == sig_id and a.get("bbox"):
                x, y, w, h = a["bbox"]
                anns.setdefault(a["image_id"], []).append([x, y, x + w, y + h])
        recs = []
        for img in data.get("images", []):
            fn = img["file_name"]
            p = root / fn if (root / fn).is_file() else root / "images" / Path(fn).name
            if not p.is_file():
                continue
            b = anns.get(img["id"], [])
            recs.append(Record(str(p), b, ["signature"] * len(b), width=int(img.get("width", 0)), height=int(img.get("height", 0))))
        return recs

    def splits_from_args(args) -> dict[str, list[Record]]:
        if getattr(args, "data_dir", None) and Path(args.data_dir).exists():
            root = _find_root(Path(args.data_dir))
        elif getattr(args, "data", None) and len(args.data) > 0 and Path(args.data[0]).exists():
            root = _find_root(Path(args.data[0]))
        elif (Path("merged_dataset") / "train.json").is_file():
            root = Path("merged_dataset")
        elif getattr(args, "use_hf", False) or not getattr(args, "dataset", None):
            root = _download_and_extract_hf(getattr(args, "hf_dataset", "monkey6769/sign_detection"))
        else:
            try:
                import kagglehub
                p = kagglehub.dataset_download(args.dataset)
                root = _find_root(Path(p))
            except Exception as e:
                print(f"[Dataset] Kaggle unavailable ({e}). Falling back to Hugging Face...")
                root = _download_and_extract_hf(getattr(args, "hf_dataset", "monkey6769/sign_detection"))

        print(f"[Dataset] Loaded dataset root: {root}")
        return {
            "train": _load_coco_split(root / "train.json", root),
            "val": _load_coco_split(root / "val.json", root),
            "test": _load_coco_split(root / "test.json", root),
        }


TASK_TOKEN = "<s_sigdet>"
CONFIG_NAME = "sigdet_config.json"


# --------------------------------------------------------------------------- box <-> tokens
class BoxTokenizer:
    """Maps boxes to token ids and back. Holds only ints, so it pickles cheaply into workers."""

    def __init__(self, tokenizer, classes: list[str], bins: int):
        cls_tokens = [f"<c_{c}>" for c in classes]
        loc_tokens = [f"<loc_{i}>" for i in range(bins)]
        tokenizer.add_tokens([TASK_TOKEN] + cls_tokens + loc_tokens, special_tokens=True)  # no-op if present
        to_id = tokenizer.convert_tokens_to_ids
        self.classes, self.bins = classes, bins
        self.task_id, self.eos_id, self.pad_id = to_id(TASK_TOKEN), tokenizer.eos_token_id, tokenizer.pad_token_id
        self.cls_ids = [to_id(t) for t in cls_tokens]
        self.loc_ids = [to_id(t) for t in loc_tokens]
        self.id2cls = {i: c for i, c in zip(self.cls_ids, classes)}
        self.id2bin = {i: b for b, i in enumerate(self.loc_ids)}

    def encode(self, boxes, labels, W, H, max_len) -> list[int]:
        order = sorted(range(len(boxes)), key=lambda i: (boxes[i][1], boxes[i][0]))  # reading order
        q = lambda v, s: min(self.bins - 1, max(0, int(v / s * self.bins)))
        ids = []
        for i in order[: (max_len - 1) // 5]:
            x1, y1, x2, y2 = boxes[i]
            ids += [self.cls_ids[self.classes.index(labels[i])],
                    self.loc_ids[q(x1, W)], self.loc_ids[q(y1, H)], self.loc_ids[q(x2, W)], self.loc_ids[q(y2, H)]]
        return ids + [self.eos_id]

    def decode(self, ids: list[int], probs: list[float], W, H):
        boxes, labels, scores = [], [], []
        for k in range(0, len(ids) - 4, 5):
            chunk = ids[k:k + 5]
            if chunk[0] not in self.id2cls or not all(t in self.id2bin for t in chunk[1:]):
                break  # eos / pad / malformed
            x1, y1, x2, y2 = [(self.id2bin[t] + 0.5) / self.bins * s for t, s in zip(chunk[1:], (W, H, W, H))]
            boxes.append([min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)])
            labels.append(self.id2cls[chunk[0]])
            scores.append(float(np.exp(np.mean(np.log(np.clip(probs[k:k + 5], 1e-9, 1))))))
        return boxes, labels, scores


class GrammarLogits(LogitsProcessor):
    """Position 0 mod 5 -> class token or eos; positions 1-4 -> location tokens."""

    def __init__(self, bt: BoxTokenizer):
        self.head, self.loc = torch.tensor(bt.cls_ids + [bt.eos_id]), torch.tensor(bt.loc_ids)

    def __call__(self, input_ids, scores):
        allowed = self.head if (input_ids.shape[1] - 1) % 5 == 0 else self.loc
        bias = torch.full_like(scores, float("-inf"))
        bias[:, allowed.to(scores.device)] = 0
        return scores + bias


# --------------------------------------------------------------------------- images
def letterbox(img: Image.Image, W: int, H: int, jitter: bool = False):
    """Resize keeping aspect ratio onto a white WxH canvas. Returns canvas, scale, (dx, dy)."""
    scale = min(W / img.width, H / img.height)
    if jitter:
        scale *= random.uniform(0.75, 1.0)
    nw, nh = max(1, round(img.width * scale)), max(1, round(img.height * scale))
    dx = random.randint(0, W - nw) if jitter else 0
    dy = random.randint(0, H - nh) if jitter else 0
    canvas = Image.new("RGB", (W, H), (255, 255, 255))
    canvas.paste(img.resize((nw, nh), Image.BICUBIC), (dx, dy))
    return canvas, scale, (dx, dy)


def photometric_aug(img: Image.Image) -> Image.Image:
    """Scanner-like degradations; geometry is handled by letterbox jitter."""
    if random.random() < 0.8:
        img = ImageEnhance.Brightness(img).enhance(random.uniform(0.75, 1.25))
    if random.random() < 0.8:
        img = ImageEnhance.Contrast(img).enhance(random.uniform(0.7, 1.3))
    if random.random() < 0.3:
        img = img.filter(ImageFilter.GaussianBlur(random.uniform(0.3, 1.2)))
    if random.random() < 0.2:
        img = ImageOps.grayscale(img).convert("RGB")
    return img


def to_pixel_values(canvas: Image.Image, mean, std) -> torch.Tensor:
    arr = (np.asarray(canvas, dtype=np.float32) / 255.0 - mean) / std
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


class SignatureDataset(Dataset):
    def __init__(self, records: list[Record], bt: BoxTokenizer, size_wh, mean, std, max_len, train):
        self.records, self.bt, (self.W, self.H) = records, bt, size_wh
        self.mean, self.std = np.array(mean, np.float32), np.array(std, np.float32)
        self.max_len, self.train = max_len, train

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        r = self.records[i]
        img = Image.open(r.path).convert("RGB")
        if self.train:
            img = photometric_aug(img)
        canvas, s, (dx, dy) = letterbox(img, self.W, self.H, jitter=self.train)
        boxes = [[b[0] * s + dx, b[1] * s + dy, b[2] * s + dx, b[3] * s + dy] for b in r.boxes]
        ids = self.bt.encode(boxes, r.labels, self.W, self.H, self.max_len)
        return to_pixel_values(canvas, self.mean, self.std), torch.tensor(ids), i


def collate(batch):
    labels = torch.full((len(batch), max(len(b[1]) for b in batch)), -100, dtype=torch.long)
    for k, (_, ids, _) in enumerate(batch):
        labels[k, :len(ids)] = ids
    return torch.stack([b[0] for b in batch]), labels, [b[2] for b in batch]


# --------------------------------------------------------------------------- model
def load_model(source: str, classes, bins, size_wh, max_len):
    processor = DonutProcessor.from_pretrained(source)
    model = VisionEncoderDecoderModel.from_pretrained(source)
    bt = BoxTokenizer(processor.tokenizer, classes, bins)
    model.decoder.resize_token_embeddings(len(processor.tokenizer))
    W, H = size_wh
    processor.image_processor.size = {"height": H, "width": W}
    model.config.encoder.image_size = [H, W]
    for cfg in (model.config, model.generation_config):
        cfg.decoder_start_token_id, cfg.pad_token_id, cfg.eos_token_id = bt.task_id, bt.pad_id, bt.eos_id
    model.generation_config.max_length = max_len
    return processor, model, bt


def save_checkpoint(path: Path, model, processor, args, extra_state=None):
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)
    processor.save_pretrained(path)
    (path / CONFIG_NAME).write_text(json.dumps(
        {"classes": args.classes, "bins": args.bins, "image_size_wh": [args.width, args.height],
         "max_len": args.max_len}, indent=2))
    if extra_state is not None:
        torch.save(extra_state, path / "trainer_state.pt")


@torch.no_grad()
def generate(model, bt: BoxTokenizer, pixel_values, max_len, amp_dtype):
    start = torch.full((pixel_values.size(0), 1), bt.task_id, device=pixel_values.device)
    with torch.autocast(pixel_values.device.type, dtype=amp_dtype or torch.bfloat16, enabled=amp_dtype is not None):
        out = model.generate(pixel_values, decoder_input_ids=start, max_length=max_len, num_beams=1,
                             do_sample=False, logits_processor=LogitsProcessorList([GrammarLogits(bt)]),
                             output_scores=True, return_dict_in_generate=True)
    logp = model.compute_transition_scores(out.sequences, out.scores, normalize_logits=True)
    return out.sequences[:, 1:].tolist(), logp.float().exp().tolist()


# --------------------------------------------------------------------------- metrics
def iou(a, b) -> float:
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def detection_metrics(preds, gts, classes, thr=0.5) -> dict:
    """preds: [(boxes, labels, scores)], gts: [(boxes, labels)] per image. VOC all-point AP@thr."""
    res, tp_all, fp_all, ngt_all = {}, 0, 0, 0
    for c in classes:
        dets = sorted(((s, i, b) for i, (bs, ls, ss) in enumerate(preds)
                       for b, l, s in zip(bs, ls, ss) if l == c), key=lambda d: -d[0])
        gt_c = [[b for b, l in zip(*g) if l == c] for g in gts]
        n_gt = sum(map(len, gt_c))
        used = [[False] * len(g) for g in gt_c]
        tps = []
        for _, i, b in dets:
            ious = [iou(b, g) for g in gt_c[i]]
            j = int(np.argmax(ious)) if ious else -1
            hit = j >= 0 and ious[j] >= thr and not used[i][j]
            if hit:
                used[i][j] = True
            tps.append(hit)
        tp = np.cumsum(tps) if tps else np.zeros(0)
        fp = np.cumsum([not t for t in tps]) if tps else np.zeros(0)
        rec = tp / max(n_gt, 1)
        prec = tp / np.maximum(tp + fp, 1e-9)
        mrec, mpre = np.concatenate(([0], rec, [1])), np.concatenate(([0], prec, [0]))
        mpre = np.maximum.accumulate(mpre[::-1])[::-1]
        res[f"AP50_{c}"] = float(np.sum((mrec[1:] - mrec[:-1]) * mpre[1:])) if n_gt else float("nan")
        tp_all += int(tp[-1]) if len(tp) else 0
        fp_all += int(fp[-1]) if len(fp) else 0
        ngt_all += n_gt
    aps = [v for v in res.values() if not math.isnan(v)]
    p, r = tp_all / max(tp_all + fp_all, 1), tp_all / max(ngt_all, 1)
    res.update(mAP50=float(np.mean(aps)) if aps else 0.0, precision=p, recall=r,
               f1=2 * p * r / max(p + r, 1e-9))
    return res


def evaluate(model, bt, loader, records, args, device, amp_dtype, desc="eval") -> dict:
    model.eval()
    preds, gts = [], []
    for px, _, idx in tqdm(loader, desc=desc, leave=False):
        seqs, probs = generate(model, bt, px.to(device), args.max_len, amp_dtype)
        for k, i in enumerate(idx):
            r = records[i]
            s = min(args.width / r.width, args.height / r.height)  # undo letterbox (no jitter at eval)
            boxes, labels, scores = bt.decode(seqs[k], probs[k], args.width, args.height)
            preds.append(([[v / s for v in b] for b in boxes], labels, scores))
            gts.append((r.boxes, r.labels))
    return detection_metrics(preds, gts, bt.classes)


# --------------------------------------------------------------------------- train / predict
def train(args) -> None:
    random.seed(args.seed), np.random.seed(args.seed), torch.manual_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    amp_dtype = None
    if device.type == "cuda" and not args.no_amp:
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    splits = splits_from_args(args)
    if "all" in [c.lower() for c in args.classes]:
        args.classes = sorted(list(set(l for s in splits.values() for r in s for l in r.labels if l)))
    processor, model, bt = load_model(args.resume or args.init, args.classes, args.bins,
                                      (args.width, args.height), args.max_len)
    if args.grad_checkpointing:
        model.gradient_checkpointing_enable()
    model.to(device)

    ip = processor.image_processor
    mk = lambda recs, tr: SignatureDataset(recs, bt, (args.width, args.height), ip.image_mean, ip.image_std,
                                           args.max_len, tr)
    loader_kw = dict(num_workers=args.workers, collate_fn=collate, pin_memory=device.type == "cuda",
                     persistent_workers=args.workers > 0)
    train_loader = DataLoader(mk(splits["train"], True), batch_size=args.batch_size, shuffle=True,
                              drop_last=len(splits["train"]) > args.batch_size, **loader_kw)
    val_loader = DataLoader(mk(splits["val"], False), batch_size=args.eval_batch_size, **loader_kw)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = math.ceil(len(train_loader) / args.grad_accum) * args.epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(args.warmup_ratio * total_steps), total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)
    start_epoch, best, bad_epochs = 0, -1.0, 0
    out = Path(args.output_dir)

    if args.resume and (Path(args.resume) / "trainer_state.pt").exists():
        st = torch.load(Path(args.resume) / "trainer_state.pt", map_location="cpu", weights_only=False)
        optimizer.load_state_dict(st["optimizer"]), scheduler.load_state_dict(st["scheduler"])
        scaler.load_state_dict(st["scaler"])
        start_epoch, best, bad_epochs = st["epoch"] + 1, st["best"], st["bad_epochs"]
        print(f"Resumed from {args.resume} at epoch {start_epoch} (best={best:.4f})")

    print(f"\nDevice={device} amp={amp_dtype} train={len(splits['train'])} val={len(splits['val'])} "
          f"steps={total_steps} image={args.width}x{args.height}")
    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0, running = time.time(), 0.0
        pbar = tqdm(train_loader, desc=f"epoch {epoch + 1}/{args.epochs}")
        for step, (px, labels, _) in enumerate(pbar):
            px, labels = px.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=amp_dtype or torch.bfloat16, enabled=amp_dtype is not None):
                loss = model(pixel_values=px, labels=labels).loss
            scaler.scale(loss / args.grad_accum).backward()
            if (step + 1) % args.grad_accum == 0 or step + 1 == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
            running += loss.item()
            pbar.set_postfix(loss=f"{running / (step + 1):.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")
        print(f"epoch {epoch + 1}: train_loss={running / max(len(train_loader), 1):.4f} "
              f"({time.time() - t0:.0f}s)")

        if splits["val"] and ((epoch + 1) % args.eval_every == 0 or epoch + 1 == args.epochs):
            m = evaluate(model, bt, val_loader, splits["val"], args, device, amp_dtype, "val")
            print("  val " + "  ".join(f"{k}={v:.4f}" for k, v in m.items()))
            score = m["mAP50"]
            if score > best:
                best, bad_epochs = score, 0
                save_checkpoint(out / "best", model, processor, args)
                (out / "best" / "val_metrics.json").write_text(json.dumps(m, indent=2))
                print(f"  new best mAP50={best:.4f} -> {out / 'best'}")
            else:
                bad_epochs += 1
        save_checkpoint(out / "last", model, processor, args, {
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(), "epoch": epoch, "best": best, "bad_epochs": bad_epochs})
        if args.patience and bad_epochs >= args.patience:
            print(f"Early stopping: no val improvement for {bad_epochs} evaluations")
            break

    if splits["test"]:
        ckpt = out / "best" if (out / "best").exists() else out / "last"
        processor, model, bt = load_model(str(ckpt), args.classes, args.bins, (args.width, args.height),
                                          args.max_len)
        model.to(device)
        test_loader = DataLoader(mk(splits["test"], False), batch_size=args.eval_batch_size, **loader_kw)
        m = evaluate(model, bt, test_loader, splits["test"], args, device, amp_dtype, "test")
        print("TEST " + "  ".join(f"{k}={v:.4f}" for k, v in m.items()))
        (ckpt / "test_metrics.json").write_text(json.dumps(m, indent=2))


def predict(args) -> None:
    ckpt = Path(args.checkpoint)
    cfg = json.loads((ckpt / CONFIG_NAME).read_text())
    W, H = cfg["image_size_wh"]
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    processor, model, bt = load_model(str(ckpt), cfg["classes"], cfg["bins"], (W, H), cfg["max_len"])
    model.to(device).eval()
    ip = processor.image_processor
    mean, std = np.array(ip.image_mean, np.float32), np.array(ip.image_std, np.float32)
    for path in args.predict:
        img = Image.open(path).convert("RGB")
        canvas, s, _ = letterbox(img, W, H)
        seqs, probs = generate(model, bt, to_pixel_values(canvas, mean, std)[None].to(device), cfg["max_len"], None)
        boxes, labels, scores = bt.decode(seqs[0], probs[0], W, H)
        dets = [{"label": l, "score": round(sc, 4), "box_xyxy": [round(v / s, 1) for v in b]}
                for b, l, sc in zip(boxes, labels, scores) if sc >= args.conf]
        print(json.dumps({"image": str(path), "detections": dets}, indent=2))
        draw = ImageDraw.Draw(img)
        for d in dets:
            draw.rectangle(d["box_xyxy"], outline=(255, 0, 0), width=max(2, img.width // 400))
            draw.text((d["box_xyxy"][0], max(0, d["box_xyxy"][1] - 12)), f"{d['label']} {d['score']:.2f}",
                      fill=(255, 0, 0))
        out_path = Path(path).with_name(Path(path).stem + "_donut_pred.png")
        img.save(out_path)
        print(f"saved {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_data_args(p)
    g = p.add_argument_group("model / training")
    g.add_argument("--init", default="naver-clova-ix/donut-base", help="Pretrained Donut checkpoint")
    g.add_argument("--output-dir", default="runs/donut")
    g.add_argument("--resume", default=None, help="Checkpoint dir to resume from (e.g. runs/donut/last)")
    g.add_argument("--width", type=int, default=960, help="Canvas width (multiple of 32)")
    g.add_argument("--height", type=int, default=1280, help="Canvas height (multiple of 32)")
    g.add_argument("--bins", type=int, default=1000, help="Number of coordinate tokens")
    g.add_argument("--max-len", type=int, default=256, help="Max decoder tokens (5 per box + eos)")
    g.add_argument("--epochs", type=int, default=30)
    g.add_argument("--batch-size", type=int, default=2)
    g.add_argument("--eval-batch-size", type=int, default=4)
    g.add_argument("--grad-accum", type=int, default=8)
    g.add_argument("--lr", type=float, default=3e-5)
    g.add_argument("--weight-decay", type=float, default=0.01)
    g.add_argument("--warmup-ratio", type=float, default=0.05)
    g.add_argument("--max-grad-norm", type=float, default=1.0)
    g.add_argument("--eval-every", type=int, default=1, help="Validate every N epochs")
    g.add_argument("--patience", type=int, default=8, help="Early-stop after N evals w/o gain (0=off)")
    g.add_argument("--workers", type=int, default=4)
    g.add_argument("--device", default=None)
    g.add_argument("--no-amp", action="store_true")
    g.add_argument("--grad-checkpointing", action="store_true", help="Trade compute for GPU memory")
    g = p.add_argument_group("inference")
    g.add_argument("--predict", nargs="+", default=None, help="Image(s) to run inference on")
    g.add_argument("--checkpoint", default="runs/donut/best")
    g.add_argument("--conf", type=float, default=0.3)
    args, _ = p.parse_known_args()
    if args.width % 32 or args.height % 32:
        p.error("--width and --height must be multiples of 32")
    if "all" not in [c.lower() for c in args.classes]:
        args.classes = [normalize_label(c) for c in args.classes]
    predict(args) if args.predict else train(args)


if __name__ == "__main__":  # required on Windows (DataLoader workers use spawn)
    main()
