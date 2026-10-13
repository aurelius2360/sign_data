"""Run signature detection inference using trained YOLO26 weights (best.pt).

Usage:
  # Process all images in test/ folder (saves annotated images to output/):
  python detect_signatures.py

  # Custom paths and confidence threshold:
  python detect_signatures.py --weights best.pt --source test --output output --conf 0.30

  # Also save cropped signature snippets:
  python detect_signatures.py --crops
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from PIL import Image

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def detect_device() -> str:
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"[Device] Using GPU: {gpu_name} (CUDA available)")
        return "0"
    print("[Device] CUDA not detected; running on CPU.")
    return "cpu"


def find_weights(weights_arg: str | None) -> Path:
    if weights_arg and Path(weights_arg).is_file():
        return Path(weights_arg)

    # Check common locations
    candidates = [
        Path("best.pt"),
        Path("runs/detect/runs/yolo_colab/signature_model-3/weights/best.pt"),
        Path("runs/yolo_colab/signature_model/weights/best.pt"),
    ]
    for c in candidates:
        if c.is_file():
            return c

    # Search recursively for any best.pt
    for p in Path(".").glob("**/best.pt"):
        if p.is_file():
            return p

    raise FileNotFoundError(
        "Could not find 'best.pt' model weights. Please specify --weights path/to/best.pt"
    )


def collect_images(source_dir: Path) -> list[Path]:
    if source_dir.is_file():
        return [source_dir]
    if not source_dir.is_dir():
        return []
    files = [f for f in source_dir.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTS]
    return sorted(files)


def run_detection(args: argparse.Namespace) -> None:
    from ultralytics import YOLO

    weights_path = find_weights(args.weights)
    print(f"[Model] Loading YOLO26 weights from: {weights_path.resolve()}")
    model = YOLO(str(weights_path))

    source_path = Path(args.source)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    images = collect_images(source_path)

    if not images:
        print("\n" + "=" * 65)
        print(f" [INFO] No document images found in '{source_path.resolve()}'")
        print("=" * 65)
        print(f"Please place your document images (.png, .jpg, .jpeg, etc.) into:")
        print(f"  📁 {source_path.resolve()}")
        print("\nThen run this script again:")
        print(f"  python detect_signatures.py")
        print("=" * 65 + "\n")
        return

    device = detect_device() if args.device is None else args.device
    crops_dir = output_path / "crops" if args.crops else None
    if crops_dir:
        crops_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[Processing] Found {len(images)} image(s) in '{source_path}'.")
    print(f"  Confidence Threshold: {args.conf}")
    print(f"  Input Resolution:     {args.imgsz}")
    print(f"  Output Directory:     {output_path.resolve()}\n")

    detections_record = []
    total_signatures = 0

    print("-" * 65)
    for idx, img_file in enumerate(images, start=1):
        try:
            # 1. Check for empty / 0-byte files
            if img_file.stat().st_size == 0:
                print(f"[{idx}/{len(images)}] {img_file.name:<35} -> ⚠️ Skipped (0-byte empty file)")
                detections_record.append({
                    "image_name": img_file.name,
                    "image_path": str(img_file.resolve()),
                    "output_annotated_image": None,
                    "num_signatures": 0,
                    "detections": [],
                    "status": "skipped_empty_file",
                })
                continue

            # 2. Verify image integrity with PIL
            try:
                with Image.open(img_file) as im_check:
                    im_check.verify()
            except Exception as img_err:
                print(f"[{idx}/{len(images)}] {img_file.name:<35} -> ⚠️ Skipped (corrupted image: {img_err})")
                detections_record.append({
                    "image_name": img_file.name,
                    "image_path": str(img_file.resolve()),
                    "output_annotated_image": None,
                    "num_signatures": 0,
                    "detections": [],
                    "status": f"skipped_corrupted: {img_err}",
                })
                continue

            # 3. Run YOLO inference
            results = model.predict(
                source=str(img_file),
                conf=args.conf,
                imgsz=args.imgsz,
                device=device,
                verbose=False,
            )

            if not results or len(results) == 0:
                print(f"[{idx}/{len(images)}] {img_file.name:<35} -> ⚠️ No model output returned")
                continue

            res = results[0]
            boxes = res.boxes
            num_found = len(boxes) if boxes is not None else 0
            total_signatures += num_found

            # Save annotated image
            out_img_file = output_path / img_file.name
            res.save(filename=str(out_img_file))

            img_detections = []
            conf_list = []
            if num_found > 0:
                coords = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()

                # Optional signature crops
                pil_img = Image.open(img_file).convert("RGB") if crops_dir else None

                for b_idx, (b, c) in enumerate(zip(coords, confs), start=1):
                    box_xyxy = [round(float(v), 2) for v in b]
                    conf_val = round(float(c), 4)
                    conf_list.append(conf_val)
                    img_detections.append({
                        "signature_id": b_idx,
                        "confidence": conf_val,
                        "box_xyxy": box_xyxy,
                    })

                    if pil_img and crops_dir:
                        x1, y1, x2, y2 = [max(0, int(v)) for v in box_xyxy]
                        cropped = pil_img.crop((x1, y1, x2, y2))
                        crop_fn = f"{img_file.stem}_sig_{b_idx}.png"
                        cropped.save(crops_dir / crop_fn)

            conf_str = ", ".join(f"{c:.2f}" for c in conf_list) if conf_list else "none"
            status_tag = f"✅ {num_found} signature(s) (conf: {conf_str})" if num_found > 0 else "⚪ 0 signatures"
            print(f"[{idx}/{len(images)}] {img_file.name:<35} -> {status_tag}")

            detections_record.append({
                "image_name": img_file.name,
                "image_path": str(img_file.resolve()),
                "output_annotated_image": str(out_img_file.resolve()),
                "num_signatures": num_found,
                "detections": img_detections,
                "status": "success",
            })

        except Exception as exc:
            print(f"[{idx}/{len(images)}] {img_file.name:<35} -> ❌ Error: {exc}")
            detections_record.append({
                "image_name": img_file.name,
                "image_path": str(img_file.resolve()),
                "output_annotated_image": None,
                "num_signatures": 0,
                "detections": [],
                "status": f"error: {exc}",
            })

    print("-" * 65)

    # 1. Export JSON Report
    json_path = output_path / "detections.json"
    json_path.write_text(json.dumps(detections_record, indent=2), encoding="utf-8")

    # 2. Export CSV Summary Table
    csv_path = output_path / "detections.csv"
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_name", "num_signatures", "max_confidence", "boxes_xyxy"])
        for rec in detections_record:
            num_sig = rec["num_signatures"]
            max_c = max([d["confidence"] for d in rec["detections"]]) if num_sig > 0 else 0.0
            boxes_str = "; ".join(f"{d['box_xyxy']}" for d in rec["detections"])
            writer.writerow([rec["image_name"], num_sig, max_c, boxes_str])

    print("\n" + "=" * 65)
    print("   DETECTION RUN COMPLETED SUCCESSFULLY!")
    print("=" * 65)
    print(f"  Processed Images:       {len(images)}")
    print(f"  Total Signatures Found: {total_signatures}")
    print(f"  Annotated Images:       {output_path.resolve()}")
    print(f"  JSON Metadata:          {json_path.resolve()}")
    print(f"  CSV Summary:            {csv_path.resolve()}")
    if crops_dir:
        print(f"  Signature Crops:        {crops_dir.resolve()}")
    print("=" * 65 + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", default="best.pt", help="Path to best.pt weights (default: best.pt)")
    p.add_argument("--source", default="test", help="Path to input folder or image (default: test)")
    p.add_argument("--output", default="output", help="Path to output folder (default: output)")
    p.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (default: 0.25)")
    p.add_argument("--imgsz", type=int, default=1024, help="Inference resolution (default: 1024)")
    p.add_argument("--device", default=None, help="Device to run on: '0', 'cpu' (default: auto)")
    p.add_argument("--crops", action="store_true", help="Also save cropped signature snippets")
    args, _ = p.parse_known_args()

    run_detection(args)


if __name__ == "__main__":
    main()
