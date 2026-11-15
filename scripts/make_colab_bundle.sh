#!/bin/bash
# Build dist/retnet-300m-colab.zip containing all code, scripts, tests, and the
# multi-account Colab notebook. Upload this zip once into your shared Google
# Drive folder (MyDrive/retnet-300m/) so every Google account can run it.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p dist
python3 - <<'PY'
import pathlib, zipfile

out = pathlib.Path("dist/retnet-300m-colab.zip")
import datetime, re, subprocess

version = re.search(r'__version__ = "([^"]+)"', pathlib.Path("retnet/__init__.py").read_text()).group(1)
try:
    rev = subprocess.run(["git", "describe", "--always", "--dirty"], capture_output=True, text=True).stdout.strip()
except OSError:
    rev = "unknown"
info = f"version {version}, git {rev or 'unknown'}, built {datetime.datetime.now():%Y-%m-%d %H:%M}\n"
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    z.writestr("BUILD_INFO.txt", info)
    for d in ["retnet", "scripts", "tests", "colab"]:
        p_dir = pathlib.Path(d)
        if not p_dir.is_dir():
            continue
        for p in sorted(p_dir.rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts and not p.name.endswith(".pyc"):
                z.write(p)
    if pathlib.Path("README.md").is_file():
        z.write("README.md")
    tok = pathlib.Path("data/tinymix/tokenizer.json")
    if tok.is_file():
        z.write(tok, "tokenizer/tokenizer.json")
print(f"wrote {out} ({out.stat().st_size // 1024} KB, {len(z.namelist())} files): {info.strip()}")
PY
