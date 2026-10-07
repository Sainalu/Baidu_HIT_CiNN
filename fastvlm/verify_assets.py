import hashlib
import json
import sys
from pathlib import Path

root = Path(__file__).parent
weights = Path(sys.argv[1])
manifest = json.loads((root / "asset_manifest.json").read_text())
for entry in manifest:
    path = (weights if entry["kind"] == "weight" else root) / entry["file"]
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != entry["sha256"] or path.stat().st_size != entry["bytes"]:
        raise ValueError(f"Asset mismatch: {path}")
print("All checkpoint/data/config SHA256 checks passed.")
