"""Merge all document signature datasets into a single, unified dataset folder.

Sources merged (auto-detected if not specified):
  1. SignverOD (Kaggle CSVs + images/)
  2. Tobacco-800 Benchmark (scans & ground truth)
  3. tech4humans/signature-detection (Hugging Face)

Exports into a self-contained directory without modifying ANY existing files:
  merged_dataset/
    ├── images/                   # All deduplicated document scan images
    ├── annotations_coco.json     # Single master COCO JSON file
    ├── annotations.csv           # Single flat CSV table with all annotations & metadata
    ├── train.json                # COCO train split
    ├── val.json                  # COCO validation split
    ├── test.json                 # COCO test split
    ├── yolo/                     # Ultralytics YOLO26 training layout
    │   ├── data.yaml             # Dataset configuration
    │   ├── images/{train,val,test}
    │   └── labels/{train,val,test}
    └── summary.json              # Statistical distribution & audit metadata

Usage:
  python merge_datasets.py
  python merge_datasets.py --output-dir merged_dataset --val-ratio 0.15 --test-ratio 0.15
  python merge_datasets.py --classes signature   # only merge signatures
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

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
        g.add_argument("--data", action="append", default=[], help="Dataset folder")
        g.add_argument("--classes", nargs="+", default=["signature"], help="Classes to train on")
        g.add_argument("--dataset", default="monkey6769007/sign-detection", help="Kaggle dataset handle")
        g.add_argument("--hf-dataset", default="monkey6769/sign_detection", help="Hugging Face dataset handle")
        g.add_argument("--val-ratio", type=float, default=0.1)
        g.add_argument("--test-ratio", type=float, default=0.1)

    def splits_from_args(args):
        raise SystemExit("signature_data.py is required for merging raw local source datasets.")

LINKABLE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def export_single_image(src: str, dst_path: Path, copy_only: bool = False) -> None:
    """Link or copy an image to dst_path without re-encoding unless format requires it."""
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    if dst_path.exists():
        return
    ext = Path(src).suffix.lower()
    if ext in LINKABLE_EXTS:
        if not copy_only:
            try:
                os.link(src, dst_path)
                return
            except OSError:
                pass
        shutil.copy2(src, dst_path)
    else:
        # Convert non-standard image formats (e.g. multi-page TIFFs or CCITT Group 4) to PNG
        from PIL import Image

        with Image.open(src) as im:
            im.convert("RGB").save(dst_path)


def make_safe_filename(idx: int, source: str, orig_path: str) -> str:
    orig = Path(orig_path)
    ext = orig.suffix.lower() if orig.suffix.lower() in LINKABLE_EXTS else ".png"
    stem_clean = re.sub(r"[^A-Za-z0-9_-]+", "_", orig.stem)[:80]
    src_clean = re.sub(r"[^A-Za-z0-9_-]+", "_", source)[:20]
    return f"{idx:06d}_{src_clean}_{stem_clean}{ext}"


def build_coco_dict(recs: list[Record], rel_paths: dict[str, str], class_to_id: dict[str, int]) -> dict:
    categories = [{"id": cid, "name": cname} for cname, cid in sorted(class_to_id.items(), key=lambda x: x[1])]
    images = []
    annotations = []
    ann_id = 1

    for img_id, r in enumerate(recs, start=1):
        rel_fn = rel_paths[r.path]
        images.append({
            "id": img_id,
            "file_name": rel_fn.replace("\\", "/"),
            "width": r.width,
            "height": r.height,
            "source": r.source,
            "split": r.split or "train",
        })
        for (x1, y1, x2, y2), lab in zip(r.boxes, r.labels):
            if lab not in class_to_id:
                continue
            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)
            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": class_to_id[lab],
                "bbox": [round(x1, 2), round(y1, 2), round(w, 2), round(h, 2)],
                "area": round(w * h, 2),
                "iscrowd": 0,
            })
            ann_id += 1

    return {"images": images, "categories": categories, "annotations": annotations}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_data_args(p)
    p.add_argument("--output-dir", default="merged_dataset", help="Output directory for merged dataset")
    p.add_argument("--copy-only", action="store_true", help="Always copy images instead of hardlinking")
    args, _ = p.parse_known_args()

    out_root = Path(args.output_dir).resolve()
    print(f"\n=======================================================")
    print(f"   STARTING DATASET MERGER -> {out_root}")
    print(f"=======================================================\n")

    # 1. Load and deduplicate splits from all sources
    splits = splits_from_args(args)

    # 2. Collect unique classes and all records
    all_recs = [r for s in ("train", "val", "test") for r in splits.get(s, [])]
    unique_classes = sorted(list(set(lab for r in all_recs for lab in r.labels if lab)))
    if not unique_classes:
        unique_classes = ["signature"]
    class_to_id = {c: i for i, c in enumerate(unique_classes, start=1)}
    id_to_class = {i: c for c, i in class_to_id.items()}

    print(f"\nUnified classes ({len(unique_classes)}): {unique_classes}")
    print(f"Total unique documents to merge: {len(all_recs)}")

    # 3. Create directory layout
    img_dir = out_root / "images"
    yolo_dir = out_root / "yolo"
    img_dir.mkdir(parents=True, exist_ok=True)
    for s in ("train", "val", "test"):
        (yolo_dir / "images" / s).mkdir(parents=True, exist_ok=True)
        (yolo_dir / "labels" / s).mkdir(parents=True, exist_ok=True)

    # 4. Map images to safe destination filenames and export
    rel_paths = {}
    print("\nExporting images to unified directory...")
    for i, r in enumerate(all_recs, start=1):
        safe_fn = make_safe_filename(i, r.source, r.path)
        dst_path = img_dir / safe_fn
        export_single_image(r.path, dst_path, copy_only=args.copy_only)
        rel_paths[r.path] = f"images/{safe_fn}"

        # Export into YOLO directory (hardlinked to merged_dataset/images/)
        split_name = r.split or "train"
        yolo_img_dst = yolo_dir / "images" / split_name / safe_fn
        export_single_image(str(dst_path), yolo_img_dst, copy_only=False)

        # Write YOLO label .txt
        lines = []
        for (x1, y1, x2, y2), lab in zip(r.boxes, r.labels):
            if lab in unique_classes and r.width > 0 and r.height > 0:
                cx = ((x1 + x2) / 2.0) / r.width
                cy = ((y1 + y2) / 2.0) / r.height
                bw = (x2 - x1) / r.width
                bh = (y2 - y1) / r.height
                c_idx = unique_classes.index(lab)
                lines.append(f"{c_idx} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        yolo_lbl_dst = yolo_dir / "labels" / split_name / f"{Path(safe_fn).stem}.txt"
        yolo_lbl_dst.write_text("\n".join(lines), encoding="utf-8")

    # 5. Write YOLO data.yaml
    yolo_cfg = {
        "path": str(yolo_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {i: c for i, c in enumerate(unique_classes)},
    }
    import yaml

    (yolo_dir / "data.yaml").write_text(yaml.safe_dump(yolo_cfg, sort_keys=False), encoding="utf-8")

    # 6. Build Master COCO JSON & Split JSONs
    print("\nWriting COCO JSON annotation files...")
    master_coco = build_coco_dict(all_recs, rel_paths, class_to_id)
    (out_root / "annotations_coco.json").write_text(json.dumps(master_coco, indent=2), encoding="utf-8")

    for s in ("train", "val", "test"):
        split_recs = splits.get(s, [])
        split_coco = build_coco_dict(split_recs, rel_paths, class_to_id)
        (out_root / f"{s}.json").write_text(json.dumps(split_coco, indent=2), encoding="utf-8")

    # 7. Write Master flat CSV table
    print("Writing master annotations.csv...")
    csv_path = out_root / "annotations.csv"
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "image_id", "file_name", "width", "height", "split", "source",
            "category", "category_id", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "bbox_width", "bbox_height", "area"
        ])
        for img_id, r in enumerate(all_recs, start=1):
            rel_fn = rel_paths[r.path].replace("\\", "/")
            if not r.boxes:
                # Keep unannotated / negative images recorded with empty boxes
                writer.writerow([img_id, rel_fn, r.width, r.height, r.split or "train", r.source, "", "", "", "", "", "", "", "", ""])
            else:
                for (x1, y1, x2, y2), lab in zip(r.boxes, r.labels):
                    cid = class_to_id.get(lab, "")
                    w = round(x2 - x1, 2)
                    h = round(y2 - y1, 2)
                    writer.writerow([
                        img_id, rel_fn, r.width, r.height, r.split or "train", r.source,
                        lab, cid, round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2), w, h, round(w * h, 2)
                    ])

    # 8. Compute and write statistical summary
    box_counts = Counter(lab for r in all_recs for lab in r.labels)
    split_stats = {}
    for s in ("train", "val", "test"):
        s_recs = splits.get(s, [])
        s_boxes = Counter(lab for r in s_recs for lab in r.labels)
        split_stats[s] = {"images": len(s_recs), "boxes": dict(s_boxes)}

    source_stats = Counter(r.source for r in all_recs)

    summary = {
        "dataset_name": "Merged Signature Detection Dataset",
        "output_directory": str(out_root),
        "total_images": len(all_recs),
        "total_annotations": sum(len(r.boxes) for r in all_recs),
        "classes": unique_classes,
        "class_counts": dict(box_counts),
        "split_counts": split_stats,
        "source_contributions": dict(source_stats),
        "files": {
            "master_coco": "annotations_coco.json",
            "train_coco": "train.json",
            "val_coco": "val.json",
            "test_coco": "test.json",
            "master_csv": "annotations.csv",
            "yolo_yaml": "yolo/data.yaml",
        },
    }
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # 9. Print final report
    print("\n=======================================================")
    print("   DATASET MERGE COMPLETED SUCCESSFULLY!")
    print("=======================================================")
    print(f"Location:           {out_root}")
    print(f"Total Images:       {len(all_recs)}")
    print(f"Total Annotations:  {sum(len(r.boxes) for r in all_recs)}")
    print(f"Classes:            {dict(box_counts)}")
    print(f"Splits:             Train={len(splits.get('train', []))}, Val={len(splits.get('val', []))}, Test={len(splits.get('test', []))}")
    print(f"Sources:            {dict(source_stats)}")
    print("\nFiles generated:")
    print(f"  - {out_root / 'annotations_coco.json'} (Single Master COCO JSON)")
    print(f"  - {out_root / 'annotations.csv'}       (Single Master CSV Table)")
    print(f"  - {out_root / 'train.json'}, val.json, test.json")
    print(f"  - {out_root / 'yolo' / 'data.yaml'}    (YOLO26 Ready)")
    print(f"  - {out_root / 'summary.json'}")
    print("=======================================================\n")


if __name__ == "__main__":
    main()
