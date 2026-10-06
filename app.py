"""A small, CPU-only demo using the supplied pothole-trained weights."""

import hashlib
from io import BytesIO
import os
from pathlib import Path
import warnings

from PIL import Image, ImageOps
import streamlit as st

MODEL_PATH = Path(__file__).resolve().parent / "models" / "best.pt"
MAX_IMAGE_SIDE = 1600

# Keep inference local: do not let Ultralytics install optional packages at runtime.
os.environ["YOLO_AUTOINSTALL"] = "false"


@st.cache_resource
def load_model():
    # Check the file first so YOLO cannot silently download another model.
    if not MODEL_PATH.is_file():
        raise FileNotFoundError("Run python download_model.py to create models/best.pt.")
    from ultralytics import YOLO

    model = YOLO(str(MODEL_PATH))
    names = model.names
    items = names.items() if isinstance(names, dict) else enumerate(names)
    pothole_ids = [int(i) for i, name in items if str(name).strip().lower() in {"pothole", "potholes"}]
    if model.task != "detect" or not pothole_ids:
        raise ValueError(f"Expected pothole detection weights; model.names = {names!r}.")
    return model, pothole_ids


st.set_page_config(page_title="Pothole Detection", layout="centered")
st.title("Pothole Detection")
st.write("Upload a road image to detect potholes.")
upload = st.file_uploader("Road image", type=["jpg", "jpeg", "png"])
confidence = st.slider("Confidence threshold", 0.05, 0.95, 0.35, 0.05)

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

# Clear old results when the upload or threshold changes; downloading never reruns YOLO.
input_key = (hashlib.sha256(image_bytes).hexdigest(), confidence)
if st.session_state.get("detection", {}).get("key") != input_key:
    st.session_state.pop("detection", None)

if st.button("Detect Potholes", type="primary"):
    st.session_state.pop("detection", None)
    try:
        model, pothole_ids = load_model()
    except Exception as exc:
        st.error(
            "Cannot load the pothole model. Check dependencies with "
            "python -m pip install -r requirements.txt. If the weights are corrupt or incorrect, "
            "remove models/best.pt and run python download_model.py again. "
            f"Details: {exc}"
        )
    else:
        try:
            with st.spinner("Detecting potholes on CPU…"):
                result = model.predict(
                    source=image, conf=confidence, classes=pothole_ids,
                    device="cpu", imgsz=416, verbose=False, save=False,
                )[0]
                # Filter again so both the count and annotation contain only potholes.
                if result.boxes is not None:
                    keep = [int(c) in pothole_ids for c in result.boxes.cls.tolist()]
                    result = result[[i for i, matches in enumerate(keep) if matches]]
                count = len(result.boxes) if result.boxes is not None else 0
                # YOLO plot() returns BGR; Pillow and Streamlit expect RGB.
                annotated = Image.fromarray(result.plot(conf=True, labels=True)[:, :, ::-1])
                output = BytesIO()
                annotated.save(output, format="PNG")
                st.session_state["detection"] = {
                    "key": input_key, "png": output.getvalue(), "count": count,
                }
        except Exception as exc:
            st.error(f"Detection failed. Try a smaller image or restart the app. Details: {exc}")

if "detection" in st.session_state:
    detection = st.session_state["detection"]
    st.image(detection["png"], caption="Detected potholes", width="stretch")
    st.write(f"Total potholes detected: {detection['count']}")
    if detection["count"] == 0:
        st.info("No potholes detected at this threshold.")
    st.download_button(
        "Download annotated image", data=detection["png"],
        file_name="potholes_detected.png", mime="image/png",
    )
