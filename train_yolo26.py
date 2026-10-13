"""Train Ultralytics YOLO26 for document signature detection.

Any mix of SignverOD / Tobacco-800 / tech4humans HF / COCO / VOC / Supervisely / YOLO sources is
merged into one YOLO-format dataset (cached, hard-linked where possible), then a YOLO26
checkpoint is fine-tuned with document-friendly augmentation.

Examples
  python train_yolo26.py --data D:/datasets/signverod --data D:/datasets/tobacco800 \
      --hf-data tech4humans/signature-detection --model yolo26s.pt --imgsz 1024 --epochs 150
  python train_yolo26.py --data D:/datasets/signverod --classes signature initials date redaction
  python train_yolo26.py --resume runs/yolo26/signature/weights/last.pt
  python train_yolo26.py ... --extra mosaic=0.5 --extra lr0=0.005   # any ultralytics train arg
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
from pathlib import Path

import yaml
from PIL import Image

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
        g.add_argument("--data", action="append", default=[], help="Dataset folder, data.yaml or COCO json")
        g.add_argument("--hf-data", action="append", default=[], help="Hugging Face dataset id")
        g.add_argument("--classes", nargs="+", default=["signature"], help="Classes to train on")
        g.add_argument("--dataset", default="monkey6769007/sign-detection", help="Kaggle dataset handle")
        g.add_argument("--hf-dataset", default="monkey6769/sign_detection", help="Hugging Face dataset handle")
        g.add_argument("--use-hf", action="store_true", help="Directly use Hugging Face dataset")
        g.add_argument("--data-dir", default=None, help="Local dataset directory")
        g.add_argument("--val-ratio", type=float, default=0.1)
        g.add_argument("--test-ratio", type=float, default=0.1)
        g.add_argument("--no-empty", action="store_true")
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
        if (p / "yolo" / "data.yaml").is_file() or (p / "train.json").is_file():
            return p
        if (p / "merged_dataset" / "yolo" / "data.yaml").is_file():
            return p / "merged_dataset"
        for sub in p.rglob("data.yaml"):
            return sub.parent.parent if sub.parent.name == "yolo" else sub.parent
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

LINKABLE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}  # everything else (e.g. .tif) -> png


def export_image(src: str, dst_stem: Path) -> None:
    ext = Path(src).suffix.lower()
    if ext in LINKABLE_EXTS:
        dst = dst_stem.with_suffix(ext)
        if not dst.exists():
            try:
                os.link(src, dst)  # no extra disk space
            except OSError:
                shutil.copy2(src, dst)
    else:  # bilevel / CCITT TIFF scans (Tobacco-800) are decoded once by PIL
        dst = dst_stem.with_suffix(".png")
        if not dst.exists():
            with Image.open(src) as im:
                im.convert("RGB").save(dst)


def yolo_lines(r: Record, classes: list[str]) -> list[str]:
    lines = []
    for (x1, y1, x2, y2), lab in zip(r.boxes, r.labels):
        cx, cy = (x1 + x2) / 2 / r.width, (y1 + y2) / 2 / r.height
        w, h = (x2 - x1) / r.width, (y2 - y1) / r.height
        lines.append(f"{classes.index(lab)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return lines


def build_yolo_dataset(splits: dict[str, list[Record]], classes: list[str], out: Path) -> Path:
    yaml_path = out / "data.yaml"
    if yaml_path.exists():
        print(f"Reusing cached YOLO dataset: {out}")
        return yaml_path
    print(f"Building YOLO dataset in {out}")
    for split, recs in splits.items():
        if not recs:
            continue
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        for i, r in enumerate(recs):
            stem = re.sub(r"[^A-Za-z0-9_-]+", "_", f"{i:06d}_{r.source}_{Path(r.path).stem}")[:150]
            export_image(r.path, out / "images" / split / stem)
            (out / "labels" / split / f"{stem}.txt").write_text("\n".join(yolo_lines(r, classes)))
    cfg = {"path": str(out.resolve()), "train": "images/train", "val": "images/val",
           "names": dict(enumerate(classes))}
    if splits.get("test"):
        cfg["test"] = "images/test"
    yaml_path.write_text(yaml.safe_dump(cfg, sort_keys=False))  # written last = "build complete" marker
    return yaml_path


def parse_extra(items: list[str]) -> dict:
    out = {}
    for item in items:
        k, _, v = item.partition("=")
        out[k] = yaml.safe_load(v)  # "0.5" -> 0.5, "true" -> True, "[0,1]" -> [0, 1]
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_data_args(p)
    g = p.add_argument_group("training")
    g.add_argument("--model", default="yolo26s.pt", help="yolo26{n,s,m,l,x}.pt or a .pt/.yaml path")
    g.add_argument("--epochs", type=int, default=150)
    g.add_argument("--imgsz", type=int, default=1024, help="Signatures are small on full pages; >=1024 helps")
    g.add_argument("--batch", type=float, default=8, help="Batch size, or -1 / 0.0-1.0 for AutoBatch")
    g.add_argument("--device", default=None, help="'0', '0,1', 'cpu', 'mps' (default: auto)")
    g.add_argument("--workers", type=int, default=4)
    g.add_argument("--patience", type=int, default=30)
    g.add_argument("--project", default="runs/yolo26")
    g.add_argument("--name", default="signature")
    g.add_argument("--dataset-root", default="runs/yolo_datasets", help="Where merged YOLO datasets are cached")
    g.add_argument("--export", default=None, help="Export best.pt after training, e.g. onnx, engine, openvino")
    g.add_argument("--resume", default=None, help="Path to an interrupted run's last.pt")
    g.add_argument("--extra", action="append", default=[], help="Extra ultralytics train arg KEY=VALUE")
    args, _ = p.parse_known_args()

    from ultralytics import YOLO

    if args.resume:
        YOLO(args.resume).train(resume=True)
        return

    # Fast-path: if pointing directly to a data.yaml or a folder with yolo/data.yaml
    direct_yaml = None
    if len(args.data) == 1:
        cand = Path(args.data[0])
        if cand.is_file() and cand.suffix in (".yaml", ".yml"):
            direct_yaml = cand
        elif (cand / "yolo" / "data.yaml").is_file():
            direct_yaml = cand / "yolo" / "data.yaml"
        elif (cand / "data.yaml").is_file():
            direct_yaml = cand / "data.yaml"

    if direct_yaml:
        data_yaml = direct_yaml
        print(f"Using existing YOLO dataset configuration: {data_yaml}")
        splits = {}
    else:
        splits = splits_from_args(args)
        if "all" in [c.lower() for c in args.classes]:
            classes = sorted(list(set(l for s in splits.values() for r in s for l in r.labels if l)))
        else:
            classes = [normalize_label(c) for c in args.classes]
        key = json.dumps([sorted(args.data), sorted(args.hf_data), classes, args.val_ratio,
                          args.test_ratio, args.seed, args.no_empty])
        data_yaml = build_yolo_dataset(splits, classes,
                                       Path(args.dataset_root) / hashlib.sha1(key.encode()).hexdigest()[:10])

    batch = int(args.batch) if args.batch >= 1 or args.batch == -1 else args.batch
    train_kwargs = dict(
        data=str(data_yaml), epochs=args.epochs, imgsz=args.imgsz, batch=batch, device=args.device,
        workers=args.workers, patience=args.patience, project=args.project, name=args.name,
        seed=args.seed, cos_lr=True, close_mosaic=10, plots=True,
        # Document-specific augmentation: no flips (mirrored handwriting/text is unrealistic),
        # almost no rotation (scans are near-upright), no hue shifts (mostly greyscale pages).
        fliplr=0.0, flipud=0.0, degrees=2.0, shear=0.0, perspective=0.0,
        translate=0.1, scale=0.5, mosaic=1.0, mixup=0.0, hsv_h=0.0, hsv_s=0.3, hsv_v=0.4,
    )
    train_kwargs.update(parse_extra(args.extra))

    model = YOLO(args.model)
    model.train(**train_kwargs)
    best = Path(model.trainer.best)
    print(f"\nBest weights: {best}")

    best_model = YOLO(best)
    if splits.get("test"):
        m = best_model.val(data=str(data_yaml), split="test", imgsz=args.imgsz, device=args.device,
                           project=args.project, name=f"{args.name}_test")
        print(f"TEST  mAP50={m.box.map50:.4f}  mAP50-95={m.box.map:.4f}  "
              f"P={m.box.mp:.4f}  R={m.box.mr:.4f}")
    if args.export:
        print(f"Exported: {best_model.export(format=args.export, imgsz=args.imgsz)}")


if __name__ == "__main__":  # required on Windows (DataLoader workers use spawn)
    main()
