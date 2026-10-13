"""Train Ultralytics YOLO26 on Google Colab (GPU) for SIGNATURE detection alone.

Datasets:
  Primary:  Kaggle 'monkey6769007/sign-detection' (via kagglehub)
  Fallback: Hugging Face 'monkey6769/sign_detection' (merged_dataset.zip auto-extracted)

In Google Colab:
  !pip install -q ultralytics kagglehub huggingface_hub
  !python yolo_colab.py --model yolo26s.pt --epochs 100 --imgsz 1024 --batch 16
"""
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import yaml
import torch


def detect_gpu():
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"[GPU] CUDA available: {gpu_name} ({vram_gb:.1f} GB VRAM)")
        return 0
    else:
        print("[WARNING] No GPU detected! Running on CPU.")
        print("[TIP] In Google Colab, go to 'Runtime' -> 'Change runtime type' -> Select 'T4 GPU'.")
        return "cpu"


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

            downloaded_path = kagglehub.dataset_download(dataset_handle)
            print(f"[Dataset] Successfully downloaded from Kaggle: {downloaded_path}")
            return Path(downloaded_path)
        except Exception as k_err:
            print(f"[Dataset] Kaggle download unavailable ({k_err}).")
            print(f"[Dataset] Triggering fallback to Hugging Face: '{hf_repo}'...")

    # 5. Fallback: Download and extract from Hugging Face repository
    return download_and_extract_hf(hf_repo=hf_repo)


def find_dataset_root(root: Path) -> Path:
    """Find the directory containing yolo/ or train.json in case of nested archives."""
    if (root / "yolo" / "data.yaml").is_file() or (root / "train.json").is_file():
        return root
    if (root / "merged_dataset" / "yolo" / "data.yaml").is_file() or (root / "merged_dataset" / "train.json").is_file():
        return root / "merged_dataset"
    for sub in root.rglob("data.yaml"):
        if sub.parent.name == "yolo" or (sub.parent / "images").is_dir():
            return sub.parent.parent if sub.parent.name == "yolo" else sub.parent
    for sub in root.rglob("train.json"):
        return sub.parent
    return root


def setup_single_class_yolo(dataset_root: Path, out_dir: Path) -> Path:
    """Extract signature-only annotations and configure a 1-class YOLO dataset."""
    src_yolo = dataset_root / "yolo" if (dataset_root / "yolo").is_dir() else dataset_root
    src_yaml = src_yolo / "data.yaml"

    sig_class_idx = 3  # default in merged_dataset
    if src_yaml.is_file():
        try:
            cfg = yaml.safe_load(src_yaml.read_text(encoding="utf-8")) or {}
            names = cfg.get("names", {})
            if isinstance(names, dict):
                for idx, name in names.items():
                    if str(name).lower() == "signature":
                        sig_class_idx = int(idx)
            elif isinstance(names, list):
                for idx, name in enumerate(names):
                    if str(name).lower() == "signature":
                        sig_class_idx = idx
        except Exception as e:
            print(f"[Warning] Could not parse src data.yaml: {e}")

    # Reuse already generated dataset if available
    target_yaml = out_dir / "data.yaml"
    if target_yaml.is_file() and any((out_dir / "images" / "train").glob("*.*")):
        print(f"[Setup] Reusing existing 1-class dataset at: {out_dir}")
        return target_yaml

    print(f"[Setup] Filtering YOLO dataset for 'signature' (source class idx = {sig_class_idx})...")
    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir = out_dir / "images"
    labels_dir = out_dir / "labels"

    counts = {}
    for split in ("train", "val", "test"):
        split_src_lbl = src_yolo / "labels" / split
        split_src_img = src_yolo / "images" / split

        if not split_src_lbl.is_dir():
            continue

        (images_dir / split).mkdir(parents=True, exist_ok=True)
        (labels_dir / split).mkdir(parents=True, exist_ok=True)

        sig_boxes_count = 0
        img_count = 0

        for lbl_file in split_src_lbl.glob("*.txt"):
            lines = lbl_file.read_text(encoding="utf-8").splitlines()
            sig_lines = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5 and int(parts[0]) == sig_class_idx:
                    # Remap signature to class 0
                    sig_lines.append(f"0 {' '.join(parts[1:])}")
                    sig_boxes_count += 1

            # Write label file (empty if background page without signatures)
            (labels_dir / split / lbl_file.name).write_text("\n".join(sig_lines), encoding="utf-8")

            # Link or copy corresponding image file
            for ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
                src_img = split_src_img / (lbl_file.stem + ext)
                if src_img.is_file():
                    dst_img = images_dir / split / src_img.name
                    if not dst_img.exists():
                        try:
                            os.link(src_img, dst_img)
                        except OSError:
                            shutil.copy2(src_img, dst_img)
                    img_count += 1
                    break

        counts[split] = {"images": img_count, "signatures": sig_boxes_count}
        print(f"  [{split}] {img_count} images | {sig_boxes_count} signatures")

    # Generate 1-class data.yaml
    target_yaml = out_dir / "data.yaml"
    data_cfg = {
        "path": str(out_dir.resolve()).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {0: "signature"},
    }
    target_yaml.write_text(yaml.safe_dump(data_cfg, sort_keys=False), encoding="utf-8")
    print(f"[Setup] Generated 1-class configuration: {target_yaml}")
    return target_yaml


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="yolo26s.pt", help="Ultralytics model (e.g. yolo26n.pt, yolo26s.pt, yolo26m.pt)")
    p.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    p.add_argument("--imgsz", type=int, default=1024, help="Input resolution (>=1024 recommended for document scans)")
    p.add_argument("--batch", type=int, default=16, help="Batch size (decrease if GPU runs out of VRAM)")
    p.add_argument("--dataset", default="monkey6769007/sign-detection", help="Kaggle dataset handle (primary)")
    p.add_argument("--hf-dataset", default="monkey6769/sign_detection", help="Hugging Face dataset repo (fallback)")
    p.add_argument("--use-hf", action="store_true", help="Directly download from Hugging Face instead of Kaggle")
    p.add_argument("--data-dir", default=None, help="Local dataset directory if already downloaded")
    p.add_argument("--project", default="runs/yolo_colab", help="Output directory for training runs")
    p.add_argument("--name", default="signature_model", help="Run name")
    p.add_argument("--workers", type=int, default=4, help="DataLoader workers")
    args, _ = p.parse_known_args()

    # 1. GPU Check
    device = detect_gpu()

    # 2. Dataset Download & Root Discovery
    download_dir = download_dataset(args.dataset, args.hf_dataset, args.data_dir, args.use_hf)
    dataset_root = find_dataset_root(download_dir)
    print(f"[Dataset] Resolved dataset root: {dataset_root}")

    # 3. Setup pure 1-Class Signature Dataset
    sig_dataset_dir = Path("yolo_signature_colab").resolve()
    data_yaml = setup_single_class_yolo(dataset_root, sig_dataset_dir)

    # 4. Initialize and Train YOLO26
    from ultralytics import YOLO

    print(f"\n[Training] Loading model: {args.model}")
    model = YOLO(args.model)

    print(f"[Training] Launching YOLO26 signature training ({args.epochs} epochs @ {args.imgsz}px)...")
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        project=args.project,
        name=args.name,
        workers=args.workers,
        patience=30,
        cos_lr=True,
        close_mosaic=10,
        save=True,
        plots=True,
        # Document-specific augmentations (no flips, minimal rotation)
        fliplr=0.0,
        flipud=0.0,
        degrees=2.0,
        scale=0.5,
        mosaic=1.0,
        hsv_h=0.0,
        hsv_s=0.3,
        hsv_v=0.4,
    )

    best_weights = Path(model.trainer.best)
    print(f"\n[Success] Training complete! Best weights saved at: {best_weights}")

    # 5. Evaluate on Held-out Test Split
    print("\n[Evaluation] Evaluating best model on test split...")
    best_model = YOLO(str(best_weights))
    metrics = best_model.val(data=str(data_yaml), split="test", imgsz=args.imgsz, device=device)
    print(f"  Test mAP@50:     {metrics.box.map50:.4f}")
    print(f"  Test mAP@50-95:  {metrics.box.map:.4f}")
    print(f"  Test Precision:  {metrics.box.mp:.4f}")
    print(f"  Test Recall:     {metrics.box.mr:.4f}")

    # 6. Sample Inference Demo
    test_images = list((sig_dataset_dir / "images" / "test").glob("*.*"))
    if test_images:
        sample_img = test_images[0]
        print(f"\n[Inference] Running sample inference on: {sample_img.name}")
        preds = best_model.predict(str(sample_img), imgsz=args.imgsz, conf=0.3, device=device)
        pred_out_dir = Path(args.project) / args.name / "sample_prediction"
        pred_img_path = pred_out_dir / sample_img.name
        preds[0].save(filename=str(pred_img_path))
        print(f"[Inference] Sample prediction image saved to: {pred_img_path}")

    # 7. Summary & Download Helper
    print("\n" + "=" * 65)
    print("   YOLO26 SIGNATURE TRAINING & EXPORT COMPLETE")
    print("=" * 65)
    print(f"  Best weights file:  {best_weights}")
    print("\nTo download the trained YOLO26 model directly in Google Colab:")
    print("  from google.colab import files")
    print(f"  files.download('{best_weights}')")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
