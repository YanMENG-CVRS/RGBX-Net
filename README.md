# RGBX-Net

Official implementation of **Waterbody extraction from the perspective of RGB+X semantic segmentation**

This repository provides the training and evaluation code for two datasets:

- **GID-5**
- **S1S2-Water**

## Environment

Install the main dependencies with your preferred CUDA-compatible PyTorch build.

```bash
conda create -n rgbxnet python=3.10 -y
conda activate rgbxnet
pip install torch torchvision torchaudio
pip install numpy scipy scikit-image tifffile tqdm easydict tensorboardX opencv-python tabulate matplotlib
```

## Dataset Preparation

Update the dataset root in:

- `local_configs/_base_/datasets/S1S2Water.py`
- `local_configs/_base_/datasets/GID.py`

Expected S1S2Water structure:

```text
S1S2Water/
  train/
    img/
    msk/
  val/
    img/
    msk/
  test/
    img/
    msk/
```

Expected GID structure:

```text
GID/
  train_filtered.txt
  val_filtered.txt
  test_filtered.txt
  output/
    <sample>.npz
    <sample>_m.png
```

## Training

Train RGBXNet on S1S2Water:

```bash
python train.py --config local_configs.S1S2Water.RGBXNet --gpus 1
```

Train RGBXNet on GID:

```bash
python train.py --config local_configs.GID.RGBXNet --gpus 1
```

Checkpoints and logs are saved under `checkpoints/`.

## Evaluation

Evaluate a trained checkpoint:

```bash
python eval.py \
  --config local_configs.S1S2Water.RGBXNet \
  --gpus 1 \
  --continue_fpath /path/to/checkpoint.pth
```

For GID, replace the config with `local_configs.GID.RGBXNet`.

## Repository Layout

```text
models/
  RGBXNet.py          # main RGBXNet architecture
  FusionModule.py     # Fusion Module
  MLPDecoder.py
  swin.py

utils/
  dataloader/
    S1S2Water.py
    GID.py
  engine/
  val_mm.py
  metrics_new.py

local_configs/
  S1S2Water/RGBXNet.py
  GID/RGBXNet.py
  _base_/datasets/
```
