"""Capture bounded CUDA allocation history inside GPU workers for Memory Viz."""

import functools
import os
import socket
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

_started = False
_saved_stages = set()


def cuda_memory_snapshot(method):
    """Save first successful stage and failures; never mask a training exception."""
    @functools.wraps(method)
    def wrapped(*args, **kwargs):
        global _started
        import torch

        if os.environ.get("CUDA_MEMORY_HISTORY", "1") == "0" or not torch.cuda.is_available():
            return method(*args, **kwargs)
        if not _started:
            try:
                torch.cuda.memory._record_memory_history(max_entries=100_000)
                _started = True
                print("[CUDA memory] Allocation history enabled (max_entries=100000).", flush=True)
            except Exception as error:
                print(f"[CUDA memory] Could not enable history: {error}", flush=True)
        stage = method.__qualname__
        try:
            result = method(*args, **kwargs)
        except Exception:
            if _started:
                _dump_snapshot(stage + "_error", args[0])
            raise
        if _started and stage not in _saved_stages:
            _dump_snapshot(stage, args[0])
            _saved_stages.add(stage)
        return result

    return wrapped


def _dump_snapshot(stage, worker):
    import torch

    try:
        log_dir = Path(os.environ.get("TENSORBOARD_DIR", "tensorboard_log"))
        directory = (log_dir / "memory_snapshots").resolve()
        directory.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S_%f")
        rank = getattr(worker, "rank", os.environ.get("RANK", "unknown"))
        filename = f"my_snapshot_{timestamp}_{socket.gethostname()}_rank{rank}_pid{os.getpid()}_{stage}.pickle"
        path = directory / filename
        torch.cuda.memory._dump_snapshot(str(path))
        print(f"[CUDA memory] Snapshot saved to {path}", flush=True)
    except Exception as error:
        print(f"[CUDA memory] Could not save snapshot for {stage}: {error}", flush=True)
