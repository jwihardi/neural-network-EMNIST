#!/usr/bin/env python3

from pathlib import Path
from urllib.request import urlopen, Request
import gzip
import io
import shutil
import tempfile
import zipfile


def download(url: str, dest: Path) -> None:
    """Stream a URL to dest. Sends a browser User-Agent because some servers
    (e.g. NIST's EMNIST host) 403 the default 'Python-urllib' agent."""
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req) as resp, open(dest, "wb") as f:
        shutil.copyfileobj(resp, f)


def extract_gz(gz_path: Path, out_path: Path) -> None:
    """Decompress a .gz file to out_path."""
    with gzip.open(gz_path, "rb") as f_in, open(out_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)


# ---------------------------------------------------------------------------
# MNIST  (per-file gzipped idx, one CDN file each)
# ---------------------------------------------------------------------------
mnist_urls = [
    "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz",
    "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz",
    "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz",
    "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz",
]

mnist_dir = Path("mnist")
mnist_dir.mkdir(exist_ok=True)

for url in mnist_urls:
    gz_path = mnist_dir / url.split("/")[-1]
    out_path = gz_path.with_suffix("")

    print(f"Downloading {gz_path.name}")
    download(url, gz_path)

    print(f"Extracting {gz_path.name}")
    extract_gz(gz_path, out_path)
    gz_path.unlink()

print("MNIST downloaded\n")

# ---------------------------------------------------------------------------
# EMNIST  (one big zip containing gzipped idx files for every split)
# ---------------------------------------------------------------------------
# EMNIST is distributed as a single ~500 MB zip from NIST. We download it once
# and pull just the chosen split's four idx files into emnist/.
EMNIST_URL = "https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip"

# Which split to extract. Class counts (set num_labels in the C++ accordingly):
#   "digits"   -> 10 classes (drop-in with the current 10-class loader)
#   "mnist"    -> 10 classes
#   "letters"  -> 26 classes
#   "balanced" -> 47 classes
#   "bymerge"  -> 47 classes
#   "byclass"  -> 62 classes
EMNIST_SPLIT = "byclass"

emnist_dir = Path("emnist")
emnist_dir.mkdir(exist_ok=True)

emnist_files = [
    f"emnist-{EMNIST_SPLIT}-train-images-idx3-ubyte",
    f"emnist-{EMNIST_SPLIT}-train-labels-idx1-ubyte",
    f"emnist-{EMNIST_SPLIT}-test-images-idx3-ubyte",
    f"emnist-{EMNIST_SPLIT}-test-labels-idx1-ubyte",
]

with tempfile.TemporaryDirectory() as tmp:
    zip_path = Path(tmp) / "emnist.zip"

    print(f"Downloading EMNIST zip (~500 MB, one-time) for split '{EMNIST_SPLIT}'")
    download(EMNIST_URL, zip_path)

    print("Extracting EMNIST split")
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        for fname in emnist_files:
            # files live somewhere like "gzip/<fname>.gz" inside the archive
            member = next((n for n in names if n.endswith(fname + ".gz")), None)
            if member is None:
                raise FileNotFoundError(f"{fname}.gz not found in EMNIST zip")

            out_path = emnist_dir / fname
            with gzip.open(io.BytesIO(zf.read(member)), "rb") as f_in, open(out_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
            print(f"  -> {out_path.name}")

print("EMNIST downloaded\n")
print("Datasets downloaded")
