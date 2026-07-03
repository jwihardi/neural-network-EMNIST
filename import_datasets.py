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


# ---------------------------------------------------------------------------
# EMNIST  (one big zip containing gzipped idx files for every split)
# ---------------------------------------------------------------------------
# EMNIST is distributed as a single ~500 MB zip from NIST containing every
# split. We download it once and extract each requested split into emnist/.
EMNIST_URL = "https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip"

# Splits to extract, matching what Shared::dataset() in shared.hpp accepts:
#   "digits"   -> 10 classes
#   "letters"  -> 27
#   "byclass"  -> 62
EMNIST_SPLITS = ["digits", "letters", "byclass"]

emnist_dir = Path("emnist")
emnist_dir.mkdir(exist_ok=True)

kinds = (
    "train-images-idx3-ubyte",
    "train-labels-idx1-ubyte",
    "test-images-idx3-ubyte",
    "test-labels-idx1-ubyte",
)

# only fetch what's missing so a re-run doesn't redownload 500 MB for nothing
missing = [
    f"emnist-{split}-{kind}"
    for split in EMNIST_SPLITS
    for kind in kinds
    if not (emnist_dir / f"emnist-{split}-{kind}").exists()
]

if not missing:
    print("EMNIST already present, nothing to do")
else:
    with tempfile.TemporaryDirectory() as tmp:
        zip_path = Path(tmp) / "emnist.zip"

        print(f"Downloading EMNIST zip (~500 MB, one-time) for splits: {', '.join(EMNIST_SPLITS)}")
        download(EMNIST_URL, zip_path)

        print("Extracting EMNIST splits")
        with zipfile.ZipFile(zip_path) as zf:
            names = zf.namelist()
            for fname in missing:
                # files live somewhere like "gzip/<fname>.gz" inside the archive
                member = next((n for n in names if n.endswith(fname + ".gz")), None)
                if member is None:
                    raise FileNotFoundError(f"{fname}.gz not found in EMNIST zip")

                out_path = emnist_dir / fname
                with gzip.open(io.BytesIO(zf.read(member)), "rb") as f_in, open(out_path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
                print(f"  {fname}")

    print("EMNIST downloaded")
