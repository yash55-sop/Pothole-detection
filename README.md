# Pothole Detection

A Streamlit demo using a pretrained YOLOv8 pothole detector for an image-level yes/no answer. A custom convolutional model trained from scratch is also retained in the project.

## Run

On a fresh checkout, create a virtual environment, install dependencies, and download the pretrained checkpoint first:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe download_model.py
```

Model weights are not included in Git. Run `train_model.py` to recreate the custom model's weights.

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Upload a JPG or PNG and click **Detect Potholes**. The retained custom model predicts at 128 × 128 resolution on an 8 × 8 grid.

## Train and evaluate

```powershell
.\.venv\Scripts\python.exe train_model.py
```

The script uses the 24 labeled images in `expanded_data`, shuffles them with seed 42 and splits them into 16 training, 4 validation and 4 held-out test images. It trains for 30 epochs with horizontal flip augmentation, selects weights using validation loss, and evaluates the selected model on the held-out test set as a yes/no image-level decision at a fixed confidence threshold of 0.25. An image is predicted positive if any box passes the threshold; its annotation is positive if it contains a labeled pothole. It saves `models/tiny_detector.pt` and `evaluation_results/tiny_test.json`. The app verifies the model hash before displaying results and reloads weights when they change.

The small test set cannot reliably estimate general performance. All four test images contain potholes, so the evaluation cannot measure false alarms on images without potholes; image-level accuracy alone is misleading. This sample is biased: the extra images were the first successfully downloaded files from an interrupted sample. See `expanded_data/source.json` for provenance and hashes. The images and annotations come from [Chitale et al.'s pothole dataset](https://github.com/jaygala24/pothole-detection); check the source's reuse terms before redistribution or commercial use. The grid cannot represent more than one pothole per cell.

Run `python evaluate_yolo.py` to evaluate the pretrained YOLOv8 checkpoint on the same test images. Its original training-data overlap with those images is unknown, so this is not a verified independent evaluation. The checkpoint comes from [aughbhay/pothole-detection-yolov8](https://github.com/aughbhay/pothole-detection-yolov8); check the source's reuse terms before redistribution. Ultralytics licensing applies separately.

For a larger randomly selected sample, run `python prepare_eval_data.py --output another_dataset --count 160` and update the dataset path in `train_model.py`. The downloader needs a working network connection and storage space. The original 10-image sample remains in `evaluation_data`.
