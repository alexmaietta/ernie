# SPDX-License-Identifier: BSD-3-Clause
import contextlib
import re
import shutil
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
IMAGES = ROOT / "docs/_src/images"

code = re.search(r"```python\n(.*?)```", (ROOT / "README.md").read_text(), re.S).group(1)
IMAGES.mkdir(parents=True, exist_ok=True)
with tempfile.TemporaryDirectory() as directory, contextlib.chdir(directory):
    exec(code, {})
    images = list(Path(directory).glob("*.png"))
    last = max(int(path.stem.rsplit("_", 1)[1]) for path in images)
    for path in images:
        camera, source, step = path.stem.split("_")
        if int(step) == last:
            shutil.move(path, IMAGES / f"readme_{camera}_{source}.png")
