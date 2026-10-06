"""Download the requested weights once, without needing third-party packages."""

from pathlib import Path
import sys
from urllib.request import Request, urlopen

MODEL_URL = "https://github.com/aughbhay/pothole-detection-yolov8/raw/refs/heads/main/best.pt"
MODEL_PATH = Path(__file__).resolve().parent / "models" / "best.pt"


def main():
    if MODEL_PATH.exists():
        print(f"Already exists; skipping download: {MODEL_PATH}")
        return 0

    temporary_path = MODEL_PATH.with_suffix(".pt.part")
    try:
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        print("Downloading pothole-trained YOLOv8 Nano weights…")
        request = Request(MODEL_URL, headers={"User-Agent": "pothole-demo"})
        with urlopen(request, timeout=60) as response, temporary_path.open("wb") as output:
            if "text/html" in response.headers.get("Content-Type", ""):
                raise ValueError("Received a web page instead of model weights.")
            expected_size = response.headers.get("Content-Length")
            downloaded = 0
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
                downloaded += len(chunk)
            if not downloaded or (expected_size and downloaded != int(expected_size)):
                raise ValueError("The download was empty or incomplete.")
        # Publish only a complete download; never leave a partial best.pt.
        temporary_path.replace(MODEL_PATH)
        print(f"Saved {downloaded:,} bytes to {MODEL_PATH}")
        return 0
    except Exception as exc:
        temporary_path.unlink(missing_ok=True)
        print(f"Model download failed: {exc}", file=sys.stderr)
        print(
            f"Manual download: open {MODEL_URL}\n"
            f"Save the downloaded file as {MODEL_PATH}\n"
            "Make sure it is the actual best.pt file, not a saved HTML page. Then restart the app.",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
