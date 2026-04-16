from __future__ import annotations

import json
from pathlib import Path


def ensure_dir(path: Path | str) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: dict, path: Path | str) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def load_json(path: Path | str) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_list_argument(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def target_channel_count(targets: list[str]) -> int:
    return len(targets)
