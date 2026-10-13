"""Shared dataset loading for signature detection (SignverOD, Tobacco-800, tech4humans HF, ...).

Every source is parsed into a flat list of `Record`s with absolute xyxy pixel boxes, so the
Donut trainer and the YOLO26 trainer consume exactly the same data and splits.

Supported on-disk formats (auto-detected per --data folder, or forced with `fmt=path`):
  coco         *.json with images/annotations/categories   (SignverOD Kaggle, Roboflow exports)
  supervisely  meta.json + <dataset>/ann/*.json + <dataset>/img/*   (Dataset Ninja downloads)
  voc          Pascal VOC *.xml                              (some Kaggle Tobacco-800 mirrors)
  tobacco      Tobacco-800 GEDI ground truth *.xml (<DL_ZONE gedi_type="DLSignature" .../>)
  yolo         labels/*.txt + images/* (+ data.yaml / classes.txt for class names)
Hugging Face datasets (e.g. tech4humans/signature-detection) are cached to disk once.
A single COCO json can also be passed directly: --data path/to/train.json
"""
from __future__ import annotations

import json
import os
import random
import re
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

try:
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None  # scanned pages can be huge
except ImportError:
    Image = None

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SPLITS = ("train", "val", "test")
FORMATS = ("coco", "supervisely", "voc", "tobacco", "yolo", "kaggle", "csv")
_ALIASES = {"initial": "initials", "dates": "date", "dllogo": "logo", "redactions": "redaction"}
IGNORED_DIRS = {".venv", "venv", "env", ".git", "__pycache__", ".gemini", ".idea", ".vscode", "runs", "cache"}


def _is_ignored(path: Path) -> bool:
    return any(p in IGNORED_DIRS or p.startswith(".") for p in path.parts)


@dataclass
class Record:
    path: str
    boxes: list  # [[x1, y1, x2, y2], ...] absolute pixels of the original image
    labels: list  # class names, parallel to boxes
    split: str | None = None
    source: str = ""
    width: int = 0
    height: int = 0


def normalize_label(name) -> str:
    n = re.sub(r"[^a-z]", "", str(name).lower())
    if "signature" in n:  # "Signature", "signatures", "DLSignature", "handwritten-signature"
        return "signature"
    return _ALIASES.get(n, n)


def guess_split(rel_path) -> str | None:
    """Infer split from the nearest path component containing train/val(id)/test."""
    for part in reversed(Path(rel_path).parts):
        m = re.search(r"(?:^|[^a-z])(train|val|test)", part.lower())
        if m:
            return m.group(1)
    return None


class ImageIndex:
    """Finds image files under a root by file name / stem (annotation paths are often stale)."""

    def __init__(self, root: Path):
        self.by_name, self.by_stem = defaultdict(list), defaultdict(list)
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in IGNORED_DIRS and not d.startswith(".")]
            for fn in filenames:
                ext = os.path.splitext(fn)[1].lower()
                if ext in IMG_EXTS:
                    p = Path(dirpath) / fn
                    self.by_name[fn.lower()].append(p)
                    self.by_stem[os.path.splitext(fn)[0].lower()].append(p)

    def find(self, name, near: Path | None = None) -> Path | None:
        name = Path(str(name).replace("\\", "/"))
        cands = self.by_name.get(name.name.lower()) or self.by_stem.get(name.stem.lower()) or []
        if len(cands) > 1 and near is not None:  # prefer the file closest to the annotation
            near_s = str(near.resolve())
            cands = sorted(cands, key=lambda p: -len(os.path.commonpath([near_s, str(p.resolve())])))
        return cands[0] if cands else None


def _tag(el) -> str:
    return el.tag.rsplit("}", 1)[-1]


# --------------------------------------------------------------------------- format parsers
def _is_coco(p: Path) -> bool:
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except (ValueError, UnicodeDecodeError, OSError):
        return False
    return isinstance(d, dict) and "images" in d and "annotations" in d


def parse_coco(json_path: Path, root: Path, index: ImageIndex, source: str) -> list[Record]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    cats = {c["id"]: normalize_label(c["name"]) for c in data.get("categories", [])}
    anns = defaultdict(list)
    for a in data["annotations"]:
        anns[a["image_id"]].append(a)
    split = guess_split(json_path.relative_to(root))
    recs, missing = [], 0
    for img in data["images"]:
        direct = json_path.parent / img["file_name"]
        path = direct if direct.is_file() else index.find(img["file_name"], near=json_path.parent)
        if path is None:
            missing += 1
            continue
        boxes, labels = [], []
        for a in anns[img["id"]]:
            if not a.get("bbox"):
                continue
            x, y, w, h = a["bbox"]
            boxes.append([x, y, x + w, y + h])
            labels.append(cats.get(a["category_id"], str(a["category_id"])))
        recs.append(Record(str(path), boxes, labels, split, source, img.get("width", 0), img.get("height", 0)))
    print(f"  [coco] {json_path.name}: {len(recs)} images (split={split}, missing images={missing})")
    return recs


def parse_supervisely(ann_path: Path, root: Path, index: ImageIndex, source: str) -> Record | None:
    data = json.loads(ann_path.read_text(encoding="utf-8"))
    img_name = ann_path.name[:-5]  # "page.png.json" -> "page.png"
    path = ann_path.parent.parent / "img" / img_name
    path = path if path.is_file() else index.find(img_name, near=ann_path.parent.parent)
    if path is None:
        return None
    boxes, labels = [], []
    for obj in data.get("objects", []):
        pts = obj.get("points", {}).get("exterior", [])
        if len(pts) < 2:
            continue
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        boxes.append([min(xs), min(ys), max(xs), max(ys)])
        labels.append(normalize_label(obj.get("classTitle", "")))
    size = data.get("size", {})
    return Record(str(path), boxes, labels, guess_split(ann_path.relative_to(root)), source,
                  size.get("width", 0), size.get("height", 0))


def parse_xml(xml_path: Path, root: Path, index: ImageIndex, source: str) -> list[Record]:
    """Pascal VOC or Tobacco-800 GEDI xml (dispatched on content)."""
    try:
        tree = ET.parse(xml_path).getroot()
    except ET.ParseError:
        return []
    split = guess_split(xml_path.relative_to(root))
    recs = []
    if _tag(tree) == "annotation":  # Pascal VOC
        path = index.find(tree.findtext("filename") or xml_path.stem, near=xml_path.parent) \
            or index.find(xml_path.stem, near=xml_path.parent)
        if path is None:
            return []
        boxes, labels = [], []
        for obj in tree.iter("object"):
            bb = obj.find("bndbox")
            if bb is None:
                continue
            boxes.append([float(bb.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax")])
            labels.append(normalize_label(obj.findtext("name")))
        size = tree.find("size")
        w = int(float(size.findtext("width") or 0)) if size is not None else 0
        h = int(float(size.findtext("height") or 0)) if size is not None else 0
        return [Record(str(path), boxes, labels, split, source, w, h)]
    for page in (el for el in tree.iter() if _tag(el) == "DL_PAGE"):  # Tobacco-800 GEDI
        path = index.find(page.get("src", xml_path.stem), near=xml_path.parent) \
            or index.find(xml_path.stem, near=xml_path.parent)
        if path is None:
            continue
        boxes, labels = [], []
        for z in (el for el in page.iter() if _tag(el) == "DL_ZONE"):
            x, y = float(z.get("col", 0)), float(z.get("row", 0))
            boxes.append([x, y, x + float(z.get("width", 0)), y + float(z.get("height", 0))])
            labels.append(normalize_label(z.get("gedi_type", "")))
        recs.append(Record(str(path), boxes, labels, split, source,
                           int(page.get("width", 0) or 0), int(page.get("height", 0) or 0)))
    return recs


def _yolo_names(root: Path) -> list[str] | None:
    for y in list(root.rglob("*.yaml")) + list(root.rglob("*.yml")):
        import yaml  # PyYAML ships with ultralytics and transformers

        names = (yaml.safe_load(y.read_text(encoding="utf-8")) or {}).get("names")
        if isinstance(names, dict):
            return [names[k] for k in sorted(names)]
        if isinstance(names, list):
            return names
    for n in ("classes.txt", "obj.names", "_darknet.labels"):
        for p in root.rglob(n):
            return [line.strip() for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    return None


def parse_yolo(root: Path, index: ImageIndex, source: str) -> list[Record]:
    names = _yolo_names(root)
    if names is None:
        print(f"  [yolo] WARNING: no data.yaml/classes.txt in {root}; assuming class 0 = signature")
        names = ["signature"]
    names = [normalize_label(n) for n in names]
    recs = []
    for txt in root.rglob("*.txt"):
        if "labels" not in (p.lower() for p in txt.parts) or txt.name == "classes.txt":
            continue
        img_dir = Path(*["images" if p.lower() == "labels" else p for p in txt.parent.parts])
        path = next((img_dir / (txt.stem + e) for e in IMG_EXTS if (img_dir / (txt.stem + e)).is_file()), None)
        path = path or index.find(txt.stem, near=img_dir)
        if path is None:
            continue
        boxes, labels = [], []
        for line in txt.read_text(encoding="utf-8").splitlines():
            v = line.split()
            if len(v) < 5:
                continue
            c, pts = int(float(v[0])), [float(t) for t in v[1:]]
            if len(pts) == 4:  # cx cy w h (normalised)
                cx, cy, w, h = pts
                box = [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]
            else:  # segmentation polygon -> bounding box
                xs, ys = pts[0::2], pts[1::2]
                box = [min(xs), min(ys), max(xs), max(ys)]
            boxes.append(box)  # normalised; scaled to pixels in _finalize()
            labels.append(names[c] if c < len(names) else str(c))
        recs.append(Record(str(path), boxes, labels, guess_split(txt.relative_to(root)), source))
    print(f"  [yolo] {root.name}: {len(recs)} images, classes={names}")
    return recs


def parse_kaggle_csv(root: Path, index: ImageIndex, source: str) -> list[Record]:
    """Parse Kaggle SignverOD dataset format (image_ids.csv, train.csv, test.csv, categories.csv)."""
    import ast
    import csv

    cat_file = root / "categories.csv" if (root / "categories.csv").is_file() else next(root.rglob("categories.csv"), None)
    cats_map = {1: "signature", 2: "initials", 3: "redaction", 4: "date"}
    if cat_file:
        try:
            with open(cat_file, mode="r", encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    cats_map[int(float(r["id"]))] = normalize_label(r["name"])
        except Exception:
            pass

    img_file = root / "image_ids.csv" if (root / "image_ids.csv").is_file() else next(root.rglob("image_ids.csv"), None)
    if not img_file:
        return []

    with open(img_file, mode="r", encoding="utf-8") as f:
        img_rows = list(csv.DictReader(f))

    # Segment image_ids into chunks when id restarts (Kaggle SignverOD layout: 2133 + 632 = 2765)
    img_chunks = []
    curr = {}
    prev = -1
    for r in img_rows:
        cid = int(float(r["id"]))
        if cid <= prev and curr:
            img_chunks.append(curr)
            curr = {}
        curr[cid] = r
        prev = cid
    if curr:
        img_chunks.append(curr)

    # Search for annotation CSV files (train.csv, test.csv, val.csv)
    ann_files = []
    for split_cand in ("train", "test", "val", "validation"):
        candidates = [root / f"{split_cand}.csv"] + [p for p in root.rglob(f"*{split_cand}*.csv") if not _is_ignored(p)]
        for c in candidates:
            if c.is_file() and c != img_file and c != cat_file and c not in [a[0] for a in ann_files]:
                ann_files.append((c, "val" if split_cand == "validation" else split_cand))

    ann_dict = defaultdict(lambda: {"boxes": [], "labels": [], "split": None})
    for ann_csv, split_name in ann_files:
        with open(ann_csv, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames or "image_id" not in reader.fieldnames or "bbox" not in reader.fieldnames:
                continue
            rows = list(reader)
        chunk_idx = 0
        prev_id = -1
        for r in rows:
            cid = int(float(r["image_id"]))
            if cid < prev_id:
                chunk_idx += 1
            prev_id = cid
            k = (chunk_idx, cid)
            try:
                b = ast.literal_eval(r["bbox"])
                cat = cats_map.get(int(float(r["category_id"])), "signature")
                ann_dict[k]["boxes"].append(b)
                ann_dict[k]["labels"].append(cat)
                ann_dict[k]["split"] = split_name
            except Exception:
                continue

    recs, missing = [], 0
    for chunk_idx, chunk_map in enumerate(img_chunks):
        for cid, info in chunk_map.items():
            fname = info["file_name"]
            w = float(info.get("width", 0))
            h = float(info.get("height", 0))
            direct = root / "images" / fname
            path = direct if direct.is_file() else (root / fname if (root / fname).is_file() else index.find(fname, near=root))
            if path is None:
                missing += 1
                continue
            ann = ann_dict.get((chunk_idx, cid), {"boxes": [], "labels": [], "split": None})
            abs_boxes = []
            for b in ann["boxes"]:
                if w > 0 and h > 0:
                    abs_boxes.append([b[0] * w, b[1] * h, (b[0] + b[2]) * w, (b[1] + b[3]) * h])
                else:
                    abs_boxes.append([b[0], b[1], b[0] + b[2], b[1] + b[3]])
            recs.append(Record(str(path), abs_boxes, ann["labels"], ann.get("split"), source, int(w), int(h)))

    print(f"  [kaggle-csv] {root.name}: {len(recs)} images ({len(img_chunks)} chunks, missing images={missing})")
    return recs


def load_folder(spec: str) -> list[Record]:
    fmt, _, path_s = spec.partition("=") if spec.split("=", 1)[0] in FORMATS else ("", "", spec)
    root = Path(path_s).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"--data path not found: {root}")
    print(f"Loading {root} (format={fmt or 'auto'})")
    if root.is_file():  # a single COCO json
        return parse_coco(root, root.parent, ImageIndex(root.parent), root.parent.name)
    source, index = root.name, ImageIndex(root)
    jsons = [j for j in root.rglob("*.json") if not _is_ignored(j)]
    ann_jsons = [j for j in jsons if j.parent.name == "ann"]
    xmls = [x for x in root.rglob("*.xml") if not _is_ignored(x)]

    if fmt in ("", "kaggle", "csv"):
        if (root / "image_ids.csv").is_file() or any(p for p in root.rglob("image_ids.csv") if not _is_ignored(p)):
            return parse_kaggle_csv(root, index, source)
    if fmt in ("", "coco"):
        cocos = [j for j in jsons if j.parent.name != "ann" and j.name != "meta.json" and _is_coco(j)]
        if cocos:
            split_cocos = [j for j in cocos if j.name in ("train.json", "val.json", "test.json", "valid.json")]
            chosen = split_cocos if split_cocos else cocos
            return [r for j in sorted(chosen) for r in parse_coco(j, root, index, source)]
    if fmt in ("", "supervisely") and ann_jsons:
        recs = [r for j in sorted(ann_jsons) if (r := parse_supervisely(j, root, index, source))]
        print(f"  [supervisely] {len(recs)} images")
        return recs
    if fmt in ("", "voc", "tobacco") and xmls:
        recs = [r for x in sorted(xmls) for r in parse_xml(x, root, index, source)]
        print(f"  [xml voc/tobacco] {len(recs)} images")
        return recs
    if fmt in ("", "yolo"):
        yolo_recs = parse_yolo(root, index, source)
        if yolo_recs:
            return yolo_recs

    # Fallback: unannotated image collection (useful for background/negative document scans)
    img_files = [p for p in root.rglob("*") if not _is_ignored(p) and p.suffix.lower() in IMG_EXTS]
    if img_files:
        print(f"  [unannotated-images] {root.name}: {len(img_files)} background document scans")
        return [Record(str(p), [], [], guess_split(p.relative_to(root)), source) for p in sorted(img_files)]

    raise ValueError(f"Could not find {fmt or 'any supported'} annotations or images under {root}")


def load_hf(name: str, cache_root: Path) -> list[Record]:
    """Cache a HF detection dataset (objects.bbox in COCO xywh, objects.category) to disk."""
    from datasets import Image as HFImage, load_dataset

    print(f"Loading Hugging Face dataset {name}")
    out_root = cache_root / name.replace("/", "__")
    recs = []
    for split_name, ds in load_dataset(name).items():
        names = None
        for get in (lambda o: o.feature["category"].names, lambda o: o["category"].feature.names,
                    lambda o: o["category"].names):
            try:
                names = list(get(ds.features["objects"]))
                break
            except (KeyError, AttributeError, TypeError):
                pass
        ds = ds.cast_column("image", HFImage(decode=False))  # raw bytes: fast and lossless
        out = out_root / split_name
        out.mkdir(parents=True, exist_ok=True)
        for i, ex in enumerate(ds):
            img = ex["image"]
            raw = img.get("bytes") or Path(img["path"]).read_bytes()
            path = out / f"{i:06d}{'.png' if raw[:4] == b'\x89PNG' else '.jpg'}"
            if not path.exists():
                path.write_bytes(raw)
            objs = ex.get("objects") or {}
            bbs = objs.get("bbox", [])
            cats = objs.get("category", [0] * len(bbs))
            boxes = [[x, y, x + w, y + h] for x, y, w, h in bbs]
            labels = [normalize_label(names[c] if names and isinstance(c, int) else c) for c in cats]
            recs.append(Record(str(path), boxes, labels, guess_split(split_name), name,
                               ex.get("width") or 0, ex.get("height") or 0))
        print(f"  [hf] {split_name}: {len(ds)} images")
    return recs


# --------------------------------------------------------------------------- post-processing
def _finalize(r: Record, classes: list[str], min_size: float = 2.0) -> Record | None:
    if r.width <= 0 or r.height <= 0:
        if Image is not None:
            try:
                with Image.open(r.path) as im:  # header only; also catches unreadable files
                    r.width, r.height = im.size
            except OSError:
                return None
        else:
            return None
    w, h = r.width, r.height
    if r.boxes and max(max(b) for b in r.boxes) <= 1.0:  # normalised coords (YOLO, some COCO exports)
        r.boxes = [[b[0] * w, b[1] * h, b[2] * w, b[3] * h] for b in r.boxes]
    boxes, labels = [], []
    for b, lab in zip(r.boxes, r.labels):
        if lab not in classes:
            continue
        x1, x2 = sorted((min(max(b[0], 0), w), min(max(b[2], 0), w)))
        y1, y2 = sorted((min(max(b[1], 0), h), min(max(b[3], 0), h)))
        if x2 - x1 >= min_size and y2 - y1 >= min_size:
            boxes.append([x1, y1, x2, y2])
            labels.append(lab)
    r.boxes, r.labels = boxes, labels
    return r


def _assign_splits(records: list[Record], val_ratio: float, test_ratio: float, seed: int) -> None:
    rng = random.Random(seed)
    by_source = defaultdict(list)
    for r in sorted(records, key=lambda r: r.path):
        by_source[r.source].append(r)
    for recs in by_source.values():
        unsplit = [r for r in recs if r.split is None]
        rng.shuffle(unsplit)
        n_test, n_val = round(len(unsplit) * test_ratio), round(len(unsplit) * val_ratio)
        for i, r in enumerate(unsplit):
            r.split = "test" if i < n_test else "val" if i < n_test + n_val else "train"
        if not any(r.split == "val" for r in recs):  # e.g. SignverOD ships only train/test
            train = [r for r in recs if r.split == "train"]
            rng.shuffle(train)
            for r in train[: round(len(train) * val_ratio)]:
                r.split = "val"


def load_splits(data: list[str], hf_data: list[str], classes: list[str], val_ratio=0.1, test_ratio=0.1,
                seed=0, cache_dir="cache", keep_empty=True) -> dict[str, list[Record]]:
    records = [r for spec in data for r in load_folder(spec)]
    records += [r for name in hf_data for r in load_hf(name, Path(cache_dir))]
    
    # Sort to prioritize records with annotations over unannotated records
    records.sort(key=lambda r: (0 if r.boxes else 1, r.path))
    seen_paths, seen_stems, unique = set(), set(), []
    for r in records:
        key = os.path.normcase(os.path.abspath(r.path))
        stem = Path(r.path).name.lower()
        if key not in seen_paths and (not stem or stem not in seen_stems or r.boxes):
            seen_paths.add(key)
            seen_stems.add(stem)
            unique.append(r)
    if len(unique) < len(records):
        print(f"INFO: filtered {len(records) - len(unique)} redundant/duplicate images across sources")

    # If user passed --classes all, dynamically include all classes present
    all_labels = Counter(lab for r in unique for lab in r.labels)
    if "all" in [c.lower() for c in classes]:
        classes = sorted(list(set(all_labels.keys())))
        if not classes:
            classes = ["signature"]

    final = [f for r in unique if (f := _finalize(r, classes))]
    if not keep_empty:
        final = [r for r in final if r.boxes]
    _assign_splits(final, val_ratio, test_ratio, seed)
    splits = {s: [r for r in final if r.split == s] for s in SPLITS}

    print(f"\nRaw labels found: {dict(all_labels)}  -> training classes: {classes}")
    for s, recs in splits.items():
        per_src = Counter(r.source for r in recs)
        per_cls = Counter(lab for r in recs for lab in r.labels)
        print(f"  {s:5s}: {len(recs):5d} images | boxes {dict(per_cls)} | sources {dict(per_src)}")
    if not splits["train"]:
        raise RuntimeError("No training images found - check your --data / --hf-data paths.")
    return splits


def add_data_args(p) -> None:
    g = p.add_argument_group("data")
    g.add_argument("--data", action="append", default=[],
                   help="Dataset folder or COCO json (repeatable). Optional format prefix: "
                        "coco=, supervisely=, voc=, tobacco=, yolo=, kaggle=, csv=")
    g.add_argument("--hf-data", action="append", default=[],
                   help="Hugging Face dataset id (repeatable), e.g. tech4humans/signature-detection")
    g.add_argument("--classes", nargs="+", default=["signature"],
                   help="Classes to train on (SignverOD also has: initials date redaction; or use 'all')")
    g.add_argument("--val-ratio", type=float, default=0.1)
    g.add_argument("--test-ratio", type=float, default=0.1, help="Only used for sources without splits")
    g.add_argument("--no-empty", action="store_true", help="Drop images without any target box")
    g.add_argument("--cache-dir", default="cache", help="Where HF datasets are materialised")
    g.add_argument("--seed", type=int, default=42)


def splits_from_args(args) -> dict[str, list[Record]]:
    if not args.data and not args.hf_data:
        cwd = Path(".").resolve()
        detected_data = []
        detected_hf = []
        # Check if unified merged_dataset exists in current folder
        if (cwd / "merged_dataset" / "annotations_coco.json").is_file():
            detected_data.append(str(cwd / "merged_dataset"))
        # Check if Kaggle SignverOD dataset exists in current folder
        if (cwd / "image_ids.csv").is_file() and (cwd / "train.csv").is_file():
            detected_data.append(str(cwd))
        # Check if Tobacco 800 Dataset exists in current folder
        if (cwd / "Tobacco 800 Dataset").is_dir():
            detected_data.append(str(cwd / "Tobacco 800 Dataset"))
        # Check if tech4humans/signature-detection is cached or accessible
        try:
            hf_cache_dir = Path.home() / ".cache" / "huggingface" / "datasets" / "tech4humans___signature-detection"
            if hf_cache_dir.exists():
                detected_hf.append("tech4humans/signature-detection")
        except Exception:
            pass

        if detected_data or detected_hf:
            args.data = detected_data
            args.hf_data = detected_hf
            print(f"Auto-detected local datasets:\n  --data: {args.data}\n  --hf-data: {args.hf_data}")
        else:
            raise SystemExit("Give at least one --data or --hf-data source.")
    return load_splits(args.data, args.hf_data, [normalize_label(c) for c in args.classes],
                       args.val_ratio, args.test_ratio, args.seed, args.cache_dir, not args.no_empty)
