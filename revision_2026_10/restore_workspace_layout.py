"""Rebuild the original workspace layout from this folder, checking every SHA-256.

usage: python restore_workspace_layout.py <workspace root>

Copies every file listed in SOURCE_MAP.json to <workspace root>/<workspace_path>,
including the modules and inputs that already exist elsewhere in this repository
(for example analysis_code/run_pipeline.py). Existing files are left alone if their
hash matches and reported otherwise.
"""

import hashlib
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    root = Path(sys.argv[1])
    m = json.loads((HERE / "SOURCE_MAP.json").read_text(encoding="utf-8"))
    pairs = [(HERE / e["path"], e["workspace_path"], e["sha256"]) for e in m["files"]]
    pairs += [(REPO / e["public_path"], e["workspace_path"], e["sha256"]) for e in m["already_in_repository"]]
    copied = 0
    for src, rel, want in pairs:
        if sha(src) != want:
            raise SystemExit(f"hash mismatch in this repository: {src}")
        dst = root / rel
        if dst.exists():
            if sha(dst) != want:
                raise SystemExit(f"exists with different content: {dst}")
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied += 1
    print(f"{copied} files copied, {len(pairs) - copied} already present, under {root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
