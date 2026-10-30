"""The vendored original Lenia catalogue (see README.md): `ANIMALS[index]` is the entry at
that index of the original `animals.json`, with its cells as RLE."""

import json
from pathlib import Path

ANIMALS = {a["index"]: a for a in json.loads((Path(__file__).parent / "animals.json").read_text())}
