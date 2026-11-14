# AGENTS.md

## Overview
Multi-model chest X-ray abnormality detection served with NVIDIA Triton Inference Server: one classification model (normal / abnormal) plus three detection models, each followed by a Python post-processing model, chained by an ensemble. Trained on the VinBigData / VinDr-CXR dataset (14 abnormality classes + "No finding"). Educational/research use only, not a medical device.

## Layout
- `triton_pipeline/` — the full Triton model repository: `models/{classification_model, detection_model_1..3, postprocess_1..3, ensemble_model}` (ONNX models + `config.pbtxt`; `postprocess_*/1/model.py` are Python-backend models), `Dockerfile` (Triton 23.02: gRPC 8004, HTTP 8005, metrics 8006), `run_infer.py` (gRPC client; reads a DICOM, applies VOI LUT, draws boxes), `utils.py`.
- `classification/` and `detection/` — standalone single-model Triton repositories with their own `Dockerfile` and `main.py` client (used before the ensemble was built).
- `onnx_runtime/` — `convert_2_onnx.py` (export to ONNX), `run_onnx.py` (local ONNX Runtime inference).
- `yte/` — training workspace and data: `data.py`, `loss.py`, `pipeline/` (classification + detection + predict, `configs/config1.yaml`), `csvs/` (annotations: `train.csv`, YOLO/classification splits), image folders `data/`, `data_train/`, `data_val/`, `data_yolo_c1/` (YOLO format, `data.yaml`), `metrics.ipynb`, `runs/detect/` (YOLO training results).
- `yte2/` — second YOLO dataset/run: `train/` (`data_yolo/{images,labels}`, `csv/`, `runs/`) and `deploy/` (a Triton model repository + `utils.py`).
- Image folders hold ~38,000 JPEGs (~3.8 GB); the repo is ~1.9 GB — clone with `--filter=blob:none` (partial clone) or `--depth 1`.

## Commands
- Build/run Triton: `cd triton_pipeline && docker build -t triton-cxr . && docker run --gpus all -p 8004:8004 -p 8005:8005 -p 8006:8006 -v $(pwd)/models:/models triton-cxr`.
- Client: `python triton_pipeline/run_infer.py --help` (needs `tritonclient[grpc]`, `pydicom`, `opencv-python`).
- Export: `python onnx_runtime/convert_2_onnx.py`.

## Rules
- Never commit patient data or raw DICOM (`*.dcm` is gitignored; one `.dcm` is still tracked in `yte/`). Respect the dataset's license — do not add more raw dataset images.
- Do not commit `__pycache__`, `*.cache`, logs or `.bash_history` (gitignored; previously tracked ones were untracked).
- Scripts contain absolute paths from the original server (`/home/ubuntu/ductq/...`); use relative paths when editing.
- Keep model names in `config.pbtxt`, the ensemble and the client in sync.
- Do not run plain `git commit`/`git status`/`git diff` in a partial (blobless) clone of this repo: they download thousands of image blobs. Use `--filter=blob:none --no-checkout`, `git read-tree HEAD`, then plumbing (`hash-object`, `update-index`, `write-tree`, `commit-tree`).
