"""Fail closed on accidental live I/O during an offline benchmark operation.

This is a guard for trusted research code, not a security sandbox for arbitrary
Python. It does not grant permissions beyond the host filesystem sandbox.
"""
from contextlib import contextmanager
import os
from pathlib import Path
import sys
from threading import RLock


_POLICIES = []
_LIFETIME_LOCK = RLock()
_INSTALLED = False
_BLOCKED_IMPORTS = ("chromadb", "rag_cli_v2", "ml_edge", "odds_provider",
                    "data_platform.db", "core.projections")


def _inside(path, directory):
    return path == directory or directory in path.parents


def _audit(event, args):
    policy = _POLICIES[-1] if _POLICIES else None
    if policy is None:
        return
    root, output, reads, runtime = policy
    if event.startswith("socket.") or event in {"sqlite3.connect", "subprocess.Popen", "os.system"}:
        raise RuntimeError(f"Offline benchmark forbids {event}")
    if event == "import":
        name = args[0]
        if any(name == item or name.startswith(item + ".") or name.endswith("." + item)
               for item in _BLOCKED_IMPORTS):
            raise RuntimeError(f"Offline benchmark forbids live module {name}")
    if event != "open" and event not in {"os.mkdir", "os.remove", "os.rmdir", "os.rename"}:
        return
    targets = args[:2] if event == "os.rename" else args[:1]
    for value in targets:
        if isinstance(value, int) or not isinstance(value, (str, bytes, os.PathLike)):
            continue
        path = Path(os.fsdecode(value)).resolve()
        writing = event != "open"
        if event == "open":
            mode, flags = args[1:3]
            writing = (isinstance(mode, str) and any(char in mode for char in "wax+")) or (
                isinstance(flags, int) and bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC)))
        if writing and not _inside(path, output):
            raise RuntimeError(f"Offline benchmark write outside experiment: {path}")
        if _inside(path, output) or path in reads or _inside(path, runtime):
            continue
        if (_inside(path, root / "Index") or _inside(path, root / "Output")
                or path.name == ".env" or path.suffix in {".db", ".sqlite", ".sqlite3"}):
            raise RuntimeError(f"Offline benchmark forbids protected data read: {path}")


@contextmanager
def offline_guard(*, root, output, readable_files=()):
    """Allow only named research inputs and experiment writes; prohibit network/DBs."""
    global _INSTALLED
    if not _INSTALLED:
        sys.addaudithook(_audit)
        _INSTALLED = True
    # Process-visible so Python/native-library worker threads cannot lose the
    # policy through a fresh ContextVar context. Concurrent guard lifetimes are
    # serialized; same-thread nesting is supported.
    with _LIFETIME_LOCK:
        policy = (Path(root).resolve(), Path(output).resolve(),
                  frozenset(Path(p).resolve() for p in readable_files), Path(sys.prefix).resolve())
        previous = sys.dont_write_bytecode
        sys.dont_write_bytecode = True
        _POLICIES.append(policy)
        try:
            yield
        finally:
            _POLICIES.pop()
            sys.dont_write_bytecode = previous
