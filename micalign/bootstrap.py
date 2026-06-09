"""
micalign.bootstrap
Populate vendor/micasense/ from MicaSense's imageprocessing repo if missing.

    python -m micalign.bootstrap
"""

import os, sys, io, zipfile, shutil, urllib.request

ZIP_URL = "https://github.com/micasense/imageprocessing/archive/refs/heads/master.zip"
VENDOR_DIR = os.path.join(os.path.dirname(__file__), "vendor")
TARGET = os.path.join(VENDOR_DIR, "micasense")


def is_vendored():
    return os.path.exists(os.path.join(TARGET, "capture.py"))


def install_vendor(force=False, verbose=True):
    def log(m):
        if verbose:
            print(m)
    if is_vendored() and not force:
        log(f"MicaSense already vendored at {TARGET}")
        return TARGET
    if os.path.isdir(TARGET):
        shutil.rmtree(TARGET)
    os.makedirs(VENDOR_DIR, exist_ok=True)
    log(f"Downloading {ZIP_URL} ...")
    with urllib.request.urlopen(ZIP_URL) as resp:
        data = resp.read()
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        for member in zf.namelist():
            parts = member.split("/", 2)
            if len(parts) < 3 or parts[1] != "micasense" or not parts[2]:
                continue
            dest = os.path.join(TARGET, parts[2])
            if member.endswith("/"):
                os.makedirs(dest, exist_ok=True)
            else:
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                with zf.open(member) as s, open(dest, "wb") as d:
                    shutil.copyfileobj(s, d)
    if not is_vendored():
        raise RuntimeError("Vendor install failed — capture.py missing.")
    log(f"Done: {TARGET}")
    return TARGET


if __name__ == "__main__":
    install_vendor(force=("-f" in sys.argv or "--force" in sys.argv))
