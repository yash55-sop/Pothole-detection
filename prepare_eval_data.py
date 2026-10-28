"""Download a public labeled dataset and retain a reproducible 10-image sample."""

import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import random
from urllib.request import Request, urlopen
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parent
SOURCE = "https://github.com/jaygala24/pothole-detection"
URL = SOURCE + "/releases/download/v1.0.0/Pothole.Dataset.IVCNZ.zip"


class RemoteArchive(io.RawIOBase):
    """Seek through an HTTP ZIP using byte ranges; download only needed members."""

    def __init__(self, url):
        super().__init__()
        with urlopen(Request(url, headers={"User-Agent": "pothole-evaluation", "Range": "bytes=0-0"}), timeout=60) as response:
            if response.status != 206:
                raise ValueError("Server does not support partial downloads; provide --archive instead")
            self.url = response.url
            self.size = int(response.headers["Content-Range"].split("/")[-1])
        self.position = 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        position = offset if whence == 0 else (self.position if whence == 1 else self.size) + offset
        if position < 0:
            raise ValueError("Negative seek")
        self.position = position
        return position

    def read(self, size=-1):
        end = self.size - 1 if size < 0 else min(self.size - 1, self.position + size - 1)
        if end < self.position:
            return b""
        request = Request(self.url, headers={"User-Agent": "pothole-evaluation", "Range": f"bytes={self.position}-{end}"})
        with urlopen(request, timeout=60) as response:
            if response.status != 206:
                raise ValueError("Server did not honor byte range")
            data = response.read()
        if len(data) != end - self.position + 1:
            raise ValueError("Incomplete archive byte range")
        self.position += len(data)
        return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, help="Optional already-downloaded ZIP; otherwise fetch only selected entries online")
    parser.add_argument("--output", type=Path, default=ROOT / "evaluation_data")
    parser.add_argument("--count", type=int, default=10)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError(f"Output is not empty: {args.output}. Use a new output folder to avoid mixing samples.")
    source = args.archive if args.archive else RemoteArchive(URL)
    with ZipFile(source) as archive:
        names = set(archive.namelist())
        candidates = sorted(name for name in names
                            if PurePosixPath(name).suffix.lower() in {".jpg", ".jpeg", ".png"}
                            and not name.startswith("__MACOSX/")
                            and str(PurePosixPath(name).with_suffix(".txt")) in names)
        if args.count < 10 or len(candidates) < args.count:
            raise ValueError("Requested sample must contain at least 10 available image/label pairs")
        chosen = sorted(random.Random(42).sample(candidates, args.count))
        if len({PurePosixPath(name).stem for name in chosen}) != args.count:
            raise ValueError("Duplicate image stems in sample")
        manifest = []
        for image in chosen:
            label = str(PurePosixPath(image).with_suffix(".txt"))
            item = {"archive_image": image, "archive_label": label}
            for kind, member in (("images", image), ("labels", label)):
                data = archive.read(member)
                destination = args.output / kind / PurePosixPath(member).name
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(data)
                item[kind + "_sha256"] = hashlib.sha256(data).hexdigest()
            manifest.append(item)
        # Preserve the author's included documentation with the subset.
        for name in sorted(names):
            if PurePosixPath(name).name.lower() in {"readme.md", "readme.txt", "license", "license.txt"}:
                (args.output / ("SOURCE_" + PurePosixPath(name).name)).write_bytes(archive.read(name))
    metadata = {"source": SOURCE, "download_url": URL, "sample_seed": 42,
                "candidate_image_count": len(candidates), "sample_image_count": args.count,
                "archive_sha256": hashlib.sha256(args.archive.read_bytes()).hexdigest() if args.archive else None,
                "limitation": "Public dataset sample; overlap with pretrained model training data is unknown. Not a verified independent held-out benchmark.",
                "files": manifest}
    (args.output / "source.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {args.count} image/label pairs and source.json in {args.output.resolve()}")


if __name__ == "__main__":
    main()
