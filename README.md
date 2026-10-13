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

The complete 1.8 GB dataset archive (`merged_dataset.zip`) is hosted on [GitHub Releases](https://github.com/aurelius2360/sign_data/releases/tag/v1.0.0).

- **Direct Download Link:** [merged_dataset.zip (1.8 GB)](https://github.com/aurelius2360/sign_data/releases/download/v1.0.0/merged_dataset.zip)

### Download via Terminal / Colab:
```bash
# Using wget:
wget https://github.com/aurelius2360/sign_data/releases/download/v1.0.0/merged_dataset.zip

# Or using curl:
curl -L -O https://github.com/aurelius2360/sign_data/releases/download/v1.0.0/merged_dataset.zip

# Unzip the dataset
unzip merged_dataset.zip -d merged_dataset
```

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
