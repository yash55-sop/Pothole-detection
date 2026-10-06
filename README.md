# Pothole Detection

A small Streamlit app that detects potholes in one uploaded road image using an existing pothole-trained YOLOv8 Nano model. It shows bounding boxes, confidence scores, a pothole count, and a downloadable annotated PNG. No training or GPU is needed.

This is a **pretrained-model demo related to my Cross-Region Road Damage Detection project**. These weights were trained by the source author on a Roboflow pothole dataset, **not RDD2022**. I did not train these weights or validate their cross-region accuracy. Predictions can miss potholes or mistake other objects for potholes.

## Requirements and Windows setup

Use **64-bit Python 3.10–3.12** (Python 3.12 recommended). The pinned CPU PyTorch wheels support this range. An internet connection is needed for installing dependencies and downloading the weights; inference runs locally afterward.

Open a terminal in this project directory and run:

```bat
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python download_model.py
python -m streamlit run app.py
```

These activation commands work in Command Prompt. In PowerShell use `.\.venv\Scripts\Activate.ps1`. If PowerShell blocks activation, use the virtual environment's interpreter directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe download_model.py
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open the local URL printed by Streamlit (usually `http://localhost:8501`). Upload one JPG, JPEG, or PNG, choose a threshold, and click **Detect Potholes**. Changing the image or threshold clears the previous result; click again to run detection.

## Model download and troubleshooting

`download_model.py` downloads only when `models/best.pt` is missing. It writes a temporary file first and reports network or incomplete-download errors. The app never downloads substitute weights.

For a manual download:

1. Open the [exact best.pt download](https://github.com/aughbhay/pothole-detection-yolov8/raw/refs/heads/main/best.pt) in your browser.
2. Save the actual binary file as `best.pt` inside this project's `models` folder, giving `models/best.pt`. Do not save the GitHub HTML page or a file named `best.pt.html`.
3. Restart the app. If a corrupt file already exists, delete only `models/best.pt`, then rerun `python download_model.py` or replace it manually.

On first detection, the app loads `ultralytics.YOLO` and inspects `model.names` for a pothole class. An incompatible checkpoint, missing file, or missing pothole class produces an actionable error. Invalid images and failed inference also show errors instead of invented predictions.

## Confidence and laptop performance

The default confidence threshold is **0.35**: only predictions scoring at least that threshold are kept. A higher threshold usually removes uncertain detections but may miss potholes; a lower one can produce more false positives. A confidence score is a model score, not a guarantee of correctness.

Inference explicitly uses `device="cpu"` and `imgsz=416`, one image at a time, and only when the detection button is clicked. `st.cache_resource` keeps the loaded model in memory. Ultralytics runtime auto-installation is disabled. Images with a longest side above 1600 pixels are resized without changing their aspect ratio; the downloaded annotation uses this resized resolution. Pillow rejects dangerous oversized images. Large compressed images still need memory to decode. PyTorch is a substantial dependency even with CPU-only wheels.

Only pothole-class boxes are annotated and counted. YOLO's annotated BGR array is converted to RGB before display and PNG encoding. Results persist across download-button reruns.

## Source, attribution, and reuse permissions

- Weights: [aughbhay/pothole-detection-yolov8](https://github.com/aughbhay/pothole-detection-yolov8), by GitHub user **aughbhay**. The supplied `best.pt` is used as provided; no training scripts are run here.
- The source's [training script](https://github.com/aughbhay/pothole-detection-yolov8/blob/main/train_pothole_yolov8.py) references [GeraPotHole / pothole-detection-yolov8, version 1](https://universe.roboflow.com/gerapothole/pothole-detection-yolov8/dataset/1).
- Checked on **6 October 2026**: the source repository has no visible LICENSE file or explicit reuse grant for its code/weights. Its public availability does not establish unrestricted reuse permission. The referenced dataset's license could not be confirmed from the accessible page. Permission to redistribute these third-party weights or dataset remains unclear; confirm terms with the source author and dataset owner before redistributing or commercial use. Weights are excluded from Git by `.gitignore`.
- [Ultralytics licensing](https://www.ultralytics.com/license) offers AGPL-3.0 and an Enterprise option. Those dependency/model obligations apply separately and do not resolve the source author's or dataset owner's unclear permissions. Review the linked terms for your intended use.

## Files

- `app.py`: upload UI, model loading, CPU inference, and annotated downloads.
- `download_model.py`: standalone one-time model downloader (Python standard library).
- `requirements.txt`: app dependencies and CPU-only PyTorch wheels.
- `models/best.pt`: downloaded third-party checkpoint; `.gitkeep` preserves the folder.
- `.gitignore`: excludes the environment, weights, and local verification artifacts.

## Verification

Checked locally on **6 October 2026**, with Python 3.12.10, Streamlit 1.65.0, Ultralytics 8.4.174, Pillow 12.3.0, and PyTorch 2.6.0+cpu:

- Dependencies installed in `.venv`; `pip check` found no broken requirements. Both Python files passed syntax compilation.
- Downloaded the exact requested checkpoint (6,248,746 bytes). Running the downloader again skipped the existing file.
- Loaded the checkpoint successfully: `model.names = {0: 'Potholes'}`. Its training metadata identifies `yolov8n.pt`; the model has 3,011,043 parameters.
- Ran real CPU inference at `imgsz=416`, confidence 0.35, on the source repository's [640 × 640 road sample](https://github.com/aughbhay/pothole-detection-yolov8/blob/main/144_jpg.rf.2915a910e518a4cdf45d85c1f0c165cd.jpg): **2 pothole detections**, scoring approximately **0.9203** and **0.5950**. This photo already includes source annotations; these new detections are only a software smoke check, not an accuracy evaluation or evidence of cross-region performance.
- Streamlit's AppTest runner passed checks for the initial UI, real button-triggered inference, count, RGB PNG download, persisted results, threshold reset, and the no-detection message at 0.95. Invalid-image handling, simulated load/inference failures, and resizing from 4000 × 2000 to 1600 × 800 also passed. The test runner supplied upload bytes programmatically; a manual browser upload/download interaction was not verified.
- No training, RDD2022 evaluation, or cross-region accuracy validation was performed. CPU timing on your laptop may differ.
