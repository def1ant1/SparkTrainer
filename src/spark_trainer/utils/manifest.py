"""
Manifest schema v1 for video datasets.
JSONL format with rows: {id, frames_dir, audio, meta}
"""
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union


@dataclass
class ManifestV1:
    """
    Manifest entry schema v1.

    Attributes:
        id: Unique identifier for the video (typically hash or filename)
        frames_dir: Path to directory containing extracted frames
        audio: Path to extracted audio file (optional)
        meta: Additional metadata dictionary
    """

    id: str
    frames_dir: str
    audio: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ManifestV1":
        """Create from dictionary."""
        return cls(**data)

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str: str) -> "ManifestV1":
        """Create from JSON string."""
        return cls.from_dict(json.loads(json_str))


def load_manifest(path: Union[str, Path]) -> List[ManifestV1]:
    """
    Load manifest from JSONL file.

    Args:
        path: Path to manifest JSONL file

    Returns:
        List of ManifestV1 entries
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Manifest file not found: {path}")

    entries = []
    with open(path, "r") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                entries.append(ManifestV1.from_dict(data))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at line {line_num}: {e}")
            except TypeError as e:
                raise ValueError(f"Invalid manifest entry at line {line_num}: {e}")

    return entries


def save_manifest(entries: List[ManifestV1], path: Union[str, Path], append: bool = False) -> None:
    """
    Save manifest to JSONL file.

    Args:
        entries: List of ManifestV1 entries
        path: Path to output JSONL file
        append: If True, append to existing file. If False, overwrite.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    mode = "a" if append else "w"
    with open(path, mode) as f:
        for entry in entries:
            f.write(entry.to_json() + "\n")


def iter_manifest(path: Union[str, Path]) -> Iterator[ManifestV1]:
    """
    Iterate over manifest entries without loading entire file into memory.

    Args:
        path: Path to manifest JSONL file

    Yields:
        ManifestV1 entries
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Manifest file not found: {path}")

    with open(path, "r") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                yield ManifestV1.from_dict(data)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at line {line_num}: {e}")
            except TypeError as e:
                raise ValueError(f"Invalid manifest entry at line {line_num}: {e}")


def validate_manifest(path: Union[str, Path]) -> tuple[bool, List[str]]:
    """
    Validate a manifest file and check if referenced files exist.

    Args:
        path: Path to manifest JSONL file

    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    path = Path(path)

    if not path.exists():
        return False, [f"Manifest file not found: {path}"]

    try:
        entries = load_manifest(path)
    except Exception as e:
        return False, [f"Failed to load manifest: {e}"]

    if not entries:
        errors.append("Manifest is empty")

    for i, entry in enumerate(entries, 1):
        # Check frames_dir exists
        frames_dir = Path(entry.frames_dir)
        if not frames_dir.exists():
            errors.append(f"Entry {i} (id={entry.id}): frames_dir not found: {frames_dir}")
        elif not frames_dir.is_dir():
            errors.append(f"Entry {i} (id={entry.id}): frames_dir is not a directory: {frames_dir}")

        # Check audio exists if specified
        if entry.audio:
            audio_path = Path(entry.audio)
            if not audio_path.exists():
                errors.append(f"Entry {i} (id={entry.id}): audio file not found: {audio_path}")

    is_valid = len(errors) == 0
    return is_valid, errors
