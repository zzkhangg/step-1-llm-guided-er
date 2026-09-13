"""Durable, pair-indexed results for one matcher draw, independent of response caching."""

import fcntl
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


class MatcherCheckpoint:
    def __init__(self, directory, identity):
        self.directory = Path(directory)
        self.identity = identity
        self.fingerprint = digest(identity)
        self.lock = None

    def __enter__(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        self.lock = (self.directory / ".lock").open("a+")
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            manifest_path = self.directory / "manifest.json"
            if manifest_path.exists():
                manifest = json.loads(manifest_path.read_text())
                if manifest.get("fingerprint") != self.fingerprint:
                    raise ValueError(
                        "Matcher checkpoint does not match the current data, pairs, selections, "
                        "or request settings. Use a separate directory for a different draw/configuration."
                    )
            else:
                if any(self.directory.glob("pair_*.json")):
                    raise ValueError("Matcher checkpoint has results but no manifest")
                self._write_atomic(manifest_path, {
                    "version": 1,
                    "created_at": utc_now(),
                    "fingerprint": self.fingerprint,
                    "identity": self.identity,
                })
        except BaseException:
            self.lock.close()
            self.lock = None
            raise
        return self

    def __exit__(self, *exc):
        self.lock.close()
        self.lock = None

    @staticmethod
    def _write_atomic(path, data):
        temporary = path.with_suffix(".tmp")
        with temporary.open("w") as stream:
            json.dump(data, stream, ensure_ascii=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)

    def load(self, pair_index, i, j):
        path = self.directory / f"pair_{pair_index:08d}.json"
        if not path.exists():
            return None
        saved = json.loads(path.read_text())
        row = saved["result"]
        if saved.get("fingerprint") != self.fingerprint or (row["indexA"], row["indexB"]) != (i, j):
            raise ValueError(f"Mismatched matcher checkpoint result: {path}")
        # Errors remain on disk for diagnosis, but never count as completed pairs.
        if row.get("answer") not in ("Yes", "No") or row.get("llm_error"):
            return None
        return {**row, "checkpoint_hit": True}

    def save(self, pair_index, row):
        path = self.directory / f"pair_{pair_index:08d}.json"
        if path.exists():
            previous = json.loads(path.read_text())["result"]
            if previous.get("answer") in ("Yes", "No") and not previous.get("llm_error"):
                raise ValueError(f"Refusing to overwrite a completed matcher pair: {path}")
            # Keep failed attempts across process restarts, including their reported cost.
            row = dict(row)
            row["attempts"] = previous.get("attempts", []) + row.get("attempts", [])
            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                row[key] += previous.get(key, 0)
        self._write_atomic(path, {"fingerprint": self.fingerprint, "result": row})
        return row
