from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple


SELECTION_VALID = "Valid classification"
SELECTION_SIMILAR = "Similar"
SELECTION_INCORRECT = "Incorrect"
ALL_SELECTIONS = [SELECTION_VALID, SELECTION_SIMILAR, SELECTION_INCORRECT]


def is_image_file(path: str) -> bool:
    _, ext = os.path.splitext(path.lower())
    return ext in {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tif", ".tiff"}


def scan_output_dir(output_dir: str) -> List[Tuple[str, str]]:
    """
    Scan the output directory for labeled image snippets.

    Returns a list of tuples (relative_path, label), where relative_path is
    relative to the output_dir, e.g. "label1/UUID.png".

    Only scans one level of subdirectories under output_dir.
    """
    if not os.path.isdir(output_dir):
        raise FileNotFoundError(f"Output directory does not exist: {output_dir}")

    entries: List[Tuple[str, str]] = []
    for name in sorted(os.listdir(output_dir)):
        sub = os.path.join(output_dir, name)
        if not os.path.isdir(sub):
            continue
        label = name
        for fname in sorted(os.listdir(sub)):
            if is_image_file(fname):
                rel = os.path.join(label, fname)
                entries.append((rel.replace("\\", "/"), label))
    return entries


@dataclass(frozen=True)
class Snippet:
    rel_path: str  # relative to output_dir, e.g. "label1/UUID.png"
    label: str


class Progress:
    def __init__(self, output_dir: str, snippets: List[Snippet]):
        self.output_dir = os.path.abspath(output_dir)
        # Decisions keyed by rel_path
        self.decisions: Dict[str, str] = {}
        # Maintain a list of remaining snippet relative paths in deterministic order
        self._remaining: List[str] = [s.rel_path for s in snippets]
        # Lookup for labels by rel_path
        self._labels: Dict[str, str] = {s.rel_path: s.label for s in snippets}

    @property
    def total(self) -> int:
        return len(self._remaining) + len(self.decisions)

    @property
    def remaining(self) -> int:
        return len(self._remaining)

    @property
    def reviewed(self) -> int:
        return len(self.decisions)

    def next_rel_path(self) -> str | None:
        return self._remaining[0] if self._remaining else None

    def get_label(self, rel_path: str) -> str:
        return self._labels.get(rel_path, rel_path.split("/", 1)[0])

    def make_decision(self, rel_path: str, selection: str) -> None:
        if selection not in ALL_SELECTIONS:
            raise ValueError(f"Invalid selection: {selection}")
        try:
            self._remaining.remove(rel_path)
        except ValueError:
            # Already removed (or loaded via progress), that's fine
            pass
        self.decisions[rel_path] = selection

    def counts_by_selection(self) -> Dict[str, int]:
        counts: Dict[str, int] = {s: 0 for s in ALL_SELECTIONS}
        for sel in self.decisions.values():
            if sel in counts:
                counts[sel] += 1
        return counts

    def to_json_dict(self) -> Dict:
        return {
            "app_version": "0.1.0",
            "output_dir": self.output_dir,
            "saved_at": datetime.utcnow().isoformat() + "Z",
            "decisions": self.decisions,
        }

    def save(self, file_path: str) -> None:
        data = self.to_json_dict()
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)


def create_progress_from_output_dir(output_dir: str) -> Progress:
    entries = scan_output_dir(output_dir)
    snippets = [Snippet(rel, label) for rel, label in entries]
    return Progress(output_dir, snippets)


def load_progress(file_path: str) -> Progress:
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    output_dir = data.get("output_dir")
    if not output_dir:
        raise ValueError("Invalid progress file: missing 'output_dir'")

    # Rescan current filesystem state to derive remaining items
    entries = scan_output_dir(output_dir)
    snippets = [Snippet(rel, label) for rel, label in entries]
    prog = Progress(output_dir, snippets)

    decisions: Dict[str, str] = data.get("decisions", {})
    # Apply prior decisions and remove from remaining
    for rel_path, selection in decisions.items():
        if selection in ALL_SELECTIONS:
            prog.make_decision(rel_path, selection)
    return prog


def abs_path(output_dir: str, rel_path: str) -> str:
    return os.path.join(output_dir, rel_path).replace("/", os.sep)