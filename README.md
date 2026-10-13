# Signature Detection (`sign_data`)

Automated signature detection and document analysis models trained on composite signature and document datasets (including Tobacco 800 and merged signature benchmarks).

## Repository Contents

- `best.pt` - Fine-tuned YOLO detection model weights for signature localization.
- `detect_signatures.py` - Inference pipeline to detect and crop signatures in document images.
- `train_yolo26.py` - Script for training YOLO signature detector models.
- `train_donut.py` - Script for training Donut document understanding models.
- `merge_datasets.py` & `signature_data.py` - Data merging, conversion, and preprocessing pipeline.
- `requirements.txt` - Python environment dependencies.
- `test/` - Sample document test images.

## Dataset

Due to GitHub's file size restrictions on Git repositories, the full 1.8 GB dataset archive (`merged_dataset.zip`) is hosted on [GitHub Releases](https://github.com/aurelius2360/sign_data/releases).

To download and extract the dataset:
1. Download `merged_dataset.zip` from the latest release: [Releases Page](https://github.com/aurelius2360/sign_data/releases).
2. Place `merged_dataset.zip` in the root of this repository.
3. Extract the archive into `merged_dataset/`.

## Installation

```bash
pip install -r requirements.txt
```

> If you have an NVIDIA GPU, install PyTorch with CUDA support first:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
> ```

## Running Detection Inference

To run signature detection on images in the `test/` folder:

```bash
# Default run (uses best.pt, reads test/, outputs annotated images to output/)
python detect_signatures.py

# Custom threshold and save cropped signature snippets
python detect_signatures.py --weights best.pt --source test --output output --conf 0.30 --crops
```

Results (annotated images, JSON coordinates, and bounding box CSV) will be saved to `output/`.

## Training

### Train YOLO
```bash
python train_yolo26.py --data merged_dataset/data.yaml --epochs 50 --batch 16
```

### Train Donut
```bash
python train_donut.py --dataset-dir merged_dataset
```
