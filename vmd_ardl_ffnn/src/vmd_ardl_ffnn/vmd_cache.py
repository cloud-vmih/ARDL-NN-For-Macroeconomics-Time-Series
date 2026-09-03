from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import pickle
from typing import Callable
from uuid import uuid4

import pandas as pd

from .config import VMDConfig


CACHE_SCHEMA_VERSION = 1


@dataclass
class VMDCacheStats:
    hits: int = 0
    misses: int = 0
    component_loads: int = 0


@dataclass(frozen=True)
class VMDCacheRef:
    key: str
    components: tuple[str, ...]
    _cache: "ContentAddressedVMDCache"

    def load(self, component: str) -> pd.DataFrame:
        return self._cache.load_component(self.key, component)


class ContentAddressedVMDCache:
    """Persistent VMD cache with a small component-level in-memory LRU."""

    def __init__(
        self,
        root: str | Path,
        config: VMDConfig,
        *,
        max_memory_entries: int = 2,
    ) -> None:
        if max_memory_entries < 1:
            raise ValueError("max_memory_entries must be >= 1")
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.config = config
        self.max_memory_entries = int(max_memory_entries)
        self.stats = VMDCacheStats()
        self._memory: OrderedDict[tuple[str, str], pd.DataFrame] = OrderedDict()

    @property
    def memory_entry_count(self) -> int:
        return len(self._memory)

    def key_for(self, data: pd.DataFrame) -> str:
        digest = hashlib.sha256()
        header = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "config": asdict(self.config),
            "columns": [str(column) for column in data.columns],
            "dtypes": [str(dtype) for dtype in data.dtypes],
            "index_dtype": str(data.index.dtype),
        }
        digest.update(json.dumps(header, sort_keys=True, separators=(",", ":")).encode())
        digest.update(pd.util.hash_pandas_object(data, index=True).values.tobytes())
        return digest.hexdigest()

    def get_or_compute(
        self,
        data: pd.DataFrame,
        compute: Callable[[pd.DataFrame], dict[str, pd.DataFrame]],
    ) -> VMDCacheRef:
        key = self.key_for(data)
        metadata = self._read_complete_metadata(key)
        if metadata is not None:
            self.stats.hits += 1
            return VMDCacheRef(key, tuple(metadata["components"]), self)

        self.stats.misses += 1
        components = compute(data)
        if not components:
            raise ValueError("VMD decomposition returned no components.")
        entry = self.root / key
        entry.mkdir(parents=True, exist_ok=True)
        files: dict[str, str] = {}
        checksums: dict[str, str] = {}
        for index, (component, frame) in enumerate(components.items()):
            filename = f"component-{index}.pkl"
            component_path = entry / filename
            self._atomic_pickle(frame, component_path)
            files[str(component)] = filename
            checksums[str(component)] = self._file_sha256(component_path)
        metadata = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "key": key,
            "components": list(files),
            "files": files,
            "sha256": checksums,
        }
        self._atomic_text(
            json.dumps(metadata, sort_keys=True), entry / "metadata.json"
        )
        self._atomic_text(json.dumps({"key": key}), entry / "complete.json")
        return VMDCacheRef(key, tuple(files), self)

    def load_component(self, key: str, component: str) -> pd.DataFrame:
        memory_key = (key, component)
        if memory_key in self._memory:
            frame = self._memory.pop(memory_key)
            self._memory[memory_key] = frame
            return frame
        metadata = self._read_complete_metadata(key)
        if metadata is None or component not in metadata["files"]:
            raise KeyError(f"Incomplete VMD cache entry: {key}/{component}")
        path = self.root / key / metadata["files"][component]
        try:
            frame = pd.read_pickle(path)
        except Exception as exc:
            raise KeyError(f"Corrupt VMD cache entry: {path}") from exc
        if not isinstance(frame, pd.DataFrame):
            raise KeyError(f"Invalid VMD cache payload: {path}")
        self.stats.component_loads += 1
        self._memory[memory_key] = frame
        while len(self._memory) > self.max_memory_entries:
            self._memory.popitem(last=False)
        return frame

    def _read_complete_metadata(self, key: str) -> dict[str, object] | None:
        entry = self.root / key
        marker = entry / "complete.json"
        metadata_path = entry / "metadata.json"
        if not marker.is_file() or not metadata_path.is_file():
            return None
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("key") != key:
                return None
            files = metadata.get("files")
            checksums = metadata.get("sha256")
            components = metadata.get("components")
            if (
                not isinstance(files, dict)
                or not isinstance(checksums, dict)
                or not isinstance(components, list)
            ):
                return None
            for component, filename in files.items():
                payload = entry / str(filename)
                if not payload.is_file():
                    return None
                if checksums.get(component) != self._file_sha256(payload):
                    return None
            return metadata
        except (OSError, json.JSONDecodeError):
            return None

    @staticmethod
    def _atomic_pickle(value: object, path: Path) -> None:
        temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        with temporary.open("wb") as handle:
            pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
        temporary.replace(path)

    @staticmethod
    def _atomic_text(value: str, path: Path) -> None:
        temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        temporary.write_text(value, encoding="utf-8")
        temporary.replace(path)

    @staticmethod
    def _file_sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
