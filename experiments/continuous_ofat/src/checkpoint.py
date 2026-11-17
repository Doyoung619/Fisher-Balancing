"""Durable same-filesystem replacement and advisory single-writer locks."""
from __future__ import annotations

import contextlib
import fcntl
import json
import os
from pathlib import Path
import pickle
import tempfile


def _replace(path, write):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            write(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
        directory_fd = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(name)


def atomic_pickle(path, value):
    _replace(path, lambda stream: pickle.dump(value, stream, protocol=5))


def load_pickle(path, default=None):
    try:
        with open(path, "rb") as stream:
            return pickle.load(stream)
    except FileNotFoundError:
        return default


def atomic_json(path, value):
    data = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    _replace(path, lambda stream: stream.write(data))


def load_json(path, default=None):
    try:
        with open(path) as stream:
            return json.load(stream)
    except FileNotFoundError:
        return default


@contextlib.contextmanager
def file_lock(path, blocking=False):
    """Lock files stay on disk: deleting an inode would break mutual exclusion."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as stream:
        flag = fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB)
        try:
            fcntl.flock(stream, flag)
        except BlockingIOError as exc:
            raise RuntimeError(f"Already locked: {path}") from exc
        try:
            yield stream
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


# Public short aliases for callers and audit notebooks.
atomic_write_json = atomic_json
atomic_write_pickle = atomic_pickle
load = load_pickle
flock = file_lock
