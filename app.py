"""CPU-only pothole detection with a small custom model."""

import hashlib
import json
from io import BytesIO
from pathlib import Path
import warnings

from PIL import Image, ImageOps
import streamlit as st

ROOT = Path(__file__).resolve().parent
MAX_IMAGE_SIDE = 1600

@st.cache_resource
def load_model(model_name, model_hash):
    import torch
    if model_name == "YOLOv8 (pretrained)":
        from yolo_model import load_yolo
        return load_yolo(ROOT / "models" / "best.pt")
    from tiny_detector import TinyDetector
    torch.set_num_threads(1)
    model = TinyDetector()
    model.load_state_dict(torch.load(ROOT / "models" / "tiny_detector.pt", map_location="cpu", weights_only=True))
    model.eval()
    return model


st.set_page_config(page_title="Pothole Detection", layout="centered")
st.title("Pothole Detection")
st.write("Upload a road image to detect potholes.")
model_name = "YOLOv8 (pretrained)"
is_yolo = model_name == "YOLOv8 (pretrained)"
model_path = ROOT / "models" / ("best.pt" if is_yolo else "tiny_detector.pt")
report_path = ROOT / "evaluation_results" / ("yolo_test.json" if is_yolo else "tiny_test.json")

default_confidence = 0.35 if is_yolo else 0.25
model_hash = None
with st.expander("Evaluation"):
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        model_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
        if model_hash != report["model_sha256"]:
            raise ValueError("Evaluation does not match the current model weights.")
        default_confidence = float(report["confidence_threshold"])
        counts = {key: len(report["splits"][key]) for key in ("train", "val", "test")}
        if is_yolo:
            st.write(f"Pretrained YOLOv8 evaluated on {counts['test']} test images. Overlap with its original training data is unknown.")
        else:
            st.write(f"Your model was trained on {counts['train']} images, validated on {counts['val']}, and tested on {counts['test']} held-out images.")
        columns = st.columns(2)
        for column, label, key in zip(columns, ("Precision", "Recall"), ("precision", "recall")):
            value = report[key]
            column.metric(label, f"{value:.1%}" if value is not None else "N/A")
        st.caption(f"Presence or absence per test image, at confidence {default_confidence:.2f}.")
        if report['tn'] + report['fp'] == 0:
            st.warning("All test images contain potholes. This test cannot measure false alarms on images without potholes.")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        st.info(f"Evaluation unavailable. Run python {'evaluate_yolo.py' if is_yolo else 'train_model.py'}. Details: {exc}")

upload = st.file_uploader("Road image", type=["jpg", "jpeg", "png"])
confidence = default_confidence

if upload is None:
    st.session_state.pop("detection", None)
    st.stop()

image_bytes = upload.getvalue()
try:
    # Pillow checks decoded content as well as the filename extension.
    with warnings.catch_warnings():
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        with Image.open(BytesIO(image_bytes)) as source:
            if source.format not in {"JPEG", "PNG"}:
                raise ValueError("Please upload a real JPG, JPEG, or PNG image.")
            source.load()
            image = ImageOps.exif_transpose(source).convert("RGB")
    original_size = image.size
    image.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE), Image.Resampling.LANCZOS)
except (OSError, ValueError, ImportError, Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
    st.session_state.pop("detection", None)
    st.error(f"Cannot read this image. Try another JPG or PNG. Details: {exc}")
    st.stop()

st.image(image, caption="Uploaded image", width="stretch")
if image.size != original_size:
    st.caption(f"Resized to {image.width} × {image.height} pixels to reduce memory use.")

# Clear old results when the upload or threshold changes.
input_key = (hashlib.sha256(image_bytes).hexdigest(), confidence, model_name, model_hash)
if st.session_state.get("detection", {}).get("key") != input_key or "present" not in st.session_state.get("detection", {}):
    st.session_state.pop("detection", None)

if st.button("Detect Potholes", type="primary"):
    st.session_state.pop("detection", None)
    try:
        model = load_model(model_name, model_hash)
    except Exception as exc:
        st.error(
            "Cannot load the selected model. Check its weights and install dependencies with "
            "python -m pip install -r requirements.txt. "
            f"Details: {exc}"
        )
    else:
        try:
            with st.spinner("Detecting potholes on CPU…"):
                if is_yolo:
                    from yolo_model import pothole_present
                    present = pothole_present(model, image, confidence)
                else:
                    from tiny_detector import predict
                    present = bool(predict(model, image, confidence))
                st.session_state["detection"] = {
                    "key": input_key, "present": present,
                }
        except Exception as exc:
            st.error(f"Detection failed. Try a smaller image or restart the app. Details: {exc}")

if "detection" in st.session_state:
    detection = st.session_state["detection"]
    if detection["present"]:
        st.write("Pothole detected: Yes")
    else:
        st.write("Pothole detected: No")
