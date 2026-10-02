"""Small atomic checkpoint and manifest helpers for resumable local runs."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


def input_digest(*paths: Path) -> str:
    """Bind a checkpoint to exact input bytes and file ordering."""

    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def atomic_json_write(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class CheckpointStore:
    def __init__(self, path: Path, digest: str):
        self.path = path
        self.digest = digest
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("version") != 1 or data.get("input_digest") != digest:
                raise ValueError("Checkpoint input differs; use a new checkpoint path")
            if not isinstance(data.get("results"), dict):
                raise ValueError("Checkpoint results must be an object")
            self.results: dict[str, dict[str, Any]] = data["results"]
        else:
            self.results = {}

    def contains(self, key: str) -> bool:
        return key in self.results

    def save_result(self, key: str, result: dict[str, Any]) -> None:
        self.results[key] = result
        atomic_json_write(
            self.path,
            {"version": 1, "input_digest": self.digest, "results": self.results},
        )

\n