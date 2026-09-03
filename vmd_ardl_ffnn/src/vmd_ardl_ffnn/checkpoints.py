from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import pickle
from typing import Any
from uuid import uuid4

import pandas as pd


CHECKPOINT_SCHEMA_VERSION = 1


def _normalized(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_normalized(item) for item in value]
    if isinstance(value, list):
        return [_normalized(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _normalized(item) for key, item in value.items()}
    return value


@dataclass
class CandidateCheckpointStats:
    hits: int = 0
    misses: int = 0
    saves: int = 0


class CandidateCheckpointStore:
    """Atomic run-scoped storage for completed FFNN candidate evaluations."""

    def __init__(self, root: str | Path, run_identity: dict[str, Any]) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.run_identity = _normalized(run_identity)
        self.stats = CandidateCheckpointStats()

    def signature(self, descriptor: dict[str, Any]) -> str:
        payload = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "run_identity": self.run_identity,
            "candidate": _normalized(descriptor),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def load(
        self, descriptor: dict[str, Any]
    ) -> tuple[dict[str, Any], pd.DataFrame] | None:
        entry = self.root / self.signature(descriptor)
        if not (entry / "complete.json").is_file() or not (entry / "payload.pkl").is_file():
            self.stats.misses += 1
            return None
        try:
            with (entry / "payload.pkl").open("rb") as handle:
                payload = pickle.load(handle)
            if payload.get("descriptor") != _normalized(descriptor):
                raise ValueError("descriptor mismatch")
            row = payload["row"]
            prediction = payload["prediction"]
            if not isinstance(row, dict) or not isinstance(prediction, pd.DataFrame):
                raise TypeError("invalid checkpoint payload")
        except Exception:
            self.stats.misses += 1
            return None
        self.stats.hits += 1
        return row, prediction

    def save(
        self,
        descriptor: dict[str, Any],
        row: dict[str, Any],
        prediction: pd.DataFrame,
    ) -> None:
        entry = self.root / self.signature(descriptor)
        entry.mkdir(parents=True, exist_ok=True)
        payload = {
            "descriptor": _normalized(descriptor),
            "row": row,
            "prediction": prediction,
        }
        temporary = entry / f".payload.{uuid4().hex}.tmp"
        with temporary.open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        temporary.replace(entry / "payload.pkl")
        marker = entry / f".complete.{uuid4().hex}.tmp"
        marker.write_text(
            json.dumps({"schema_version": CHECKPOINT_SCHEMA_VERSION}),
            encoding="utf-8",
        )
        marker.replace(entry / "complete.json")
        self.stats.saves += 1
