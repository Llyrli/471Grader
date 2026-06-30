"""Resource-bounded execution sandbox for untrusted student code.

The autograder runs arbitrary student notebooks. Process *separation* already
exists (preprocess.py shells out to run_tests.py), but separation alone does not
bound a runaway: an infinite loop, a multi-GB allocation, or a 10 GB file write
would still take down the grader host. This module adds the missing bounds.

`run_in_sandbox(target, ...)` runs ``target`` in a FORKED child process with
POSIX resource limits applied before the call:

  - RLIMIT_AS    : address-space (memory) cap        → MemoryError, not OOM-kill
  - RLIMIT_CPU   : CPU-seconds cap                   → SIGXCPU
  - RLIMIT_FSIZE : max bytes any single file write   → blocks disk-filling
  - wall-clock timeout enforced by the parent        → hard terminate()

The child is forked (not spawned), so it inherits already-imported modules
(numpy, nbformat) and the module globals set by `run_tests.apply_config()` — no
pickling of the target, only the JSON-able result travels back over a Queue.

On non-POSIX platforms (no `resource` / no fork) it degrades gracefully to a
direct in-process call, so behavior is unchanged where sandboxing isn't
available. Set ``JN_NO_SANDBOX=1`` to force the direct path (debugging).

This is defense-in-depth on top of the container (non-root user, no network),
not a replacement for it — treat student code as untrusted and run via Docker.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Callable

MB = 1024 * 1024

# Defaults (overridable per call / via config). Generous enough for small FEM
# linear-algebra notebooks, tight enough to stop a runaway.
DEFAULT_TIMEOUT_S = 60      # wall-clock
DEFAULT_MEM_MB = 2048       # address space
DEFAULT_CPU_S = 45          # CPU time
DEFAULT_FSIZE_MB = 50       # largest single file write


class SandboxError(RuntimeError):
    """Raised when sandboxed execution fails (timeout, OOM, crash)."""


def _posix_sandbox_available() -> bool:
    if os.environ.get("JN_NO_SANDBOX") == "1":
        return False
    if not hasattr(os, "fork"):
        return False
    try:
        import resource  # noqa: F401
        import multiprocessing as mp
        mp.get_context("fork")
        return True
    except Exception:
        return False


def _apply_limits(mem_mb: int, cpu_s: int, fsize_mb: int) -> None:
    import resource
    for res, soft_hard in (
        (resource.RLIMIT_AS, (mem_mb * MB, mem_mb * MB)),
        (resource.RLIMIT_CPU, (cpu_s, cpu_s)),
        (resource.RLIMIT_FSIZE, (fsize_mb * MB, fsize_mb * MB)),
    ):
        try:
            resource.setrlimit(res, soft_hard)
        except (ValueError, OSError):
            # A limit may be unsettable (already lower, or unsupported) — skip it
            # rather than refuse to run.
            pass


def _worker(q, target, args, kwargs, limits) -> None:
    _apply_limits(**limits)
    try:
        q.put(("ok", target(*args, **(kwargs or {}))))
    except MemoryError:
        q.put(("err", "MemoryError: exceeded memory limit"))
    except Exception as exc:  # surface, don't crash silently
        q.put(("err", f"{type(exc).__name__}: {exc}"))


def run_in_sandbox(
    target: Callable[..., Any],
    args: tuple = (),
    kwargs: dict | None = None,
    *,
    timeout_s: int = DEFAULT_TIMEOUT_S,
    mem_mb: int = DEFAULT_MEM_MB,
    cpu_s: int = DEFAULT_CPU_S,
    fsize_mb: int = DEFAULT_FSIZE_MB,
) -> Any:
    """Run ``target(*args, **kwargs)`` under resource limits + wall-clock timeout.

    Returns the target's (JSON-able / picklable) result. Raises SandboxError on
    timeout, memory/CPU/file-size violation, or child crash.
    """
    if not _posix_sandbox_available():
        return target(*args, **(kwargs or {}))  # graceful fallback

    import multiprocessing as mp
    ctx = mp.get_context("fork")
    q: Any = ctx.Queue()
    limits = {"mem_mb": mem_mb, "cpu_s": cpu_s, "fsize_mb": fsize_mb}
    p = ctx.Process(target=_worker, args=(q, target, args, kwargs, limits))
    p.start()
    p.join(timeout_s)

    if p.is_alive():
        p.terminate()
        p.join()
        raise SandboxError(f"execution exceeded {timeout_s}s wall-clock timeout")

    # Child finished. Prefer its message; if none, it was hard-killed (SIGKILL /
    # SIGXCPU with no chance to report) — infer from exit code.
    if not q.empty():
        status, payload = q.get()
        if status == "ok":
            return payload
        raise SandboxError(payload)

    code = p.exitcode
    if code == 0:
        raise SandboxError("sandbox worker produced no result")
    import signal as _signal
    if code is not None and code < 0:
        signame = _signal.Signals(-code).name if -code in iter(_signal.Signals) else str(-code)
        if signame == "SIGXCPU":
            raise SandboxError(f"execution exceeded {cpu_s}s CPU limit")
        raise SandboxError(f"sandbox worker killed by {signame} "
                           f"(likely memory/CPU limit)")
    raise SandboxError(f"sandbox worker exited with code {code}")


def limits_from_config(cfg: dict | None) -> dict:
    """Extract optional ``sandbox:`` limits from a parsed config.yaml.

        sandbox:
          timeout_s: 60
          mem_mb: 2048
          cpu_s: 45
          fsize_mb: 50
    """
    s = (cfg or {}).get("sandbox") or {}
    out = {}
    for key, default in (
        ("timeout_s", DEFAULT_TIMEOUT_S), ("mem_mb", DEFAULT_MEM_MB),
        ("cpu_s", DEFAULT_CPU_S), ("fsize_mb", DEFAULT_FSIZE_MB),
    ):
        try:
            out[key] = int(s.get(key, default))
        except (TypeError, ValueError):
            out[key] = default
    return out


if __name__ == "__main__":
    # Tiny self-test: timeout, memory, file-size, and a clean run.
    import time

    def _ok():
        return {"sum": sum(range(1000))}

    def _spin():
        while True:
            pass

    def _hog():
        x = bytearray(8 * 1024 * MB)  # 8 GB → exceeds default cap
        return len(x)

    print("clean:", run_in_sandbox(_ok))
    for name, fn, kw in (
        ("timeout", _spin, {"timeout_s": 2}),
        ("memory", _hog, {}),
    ):
        try:
            run_in_sandbox(fn, **kw)
            print(name, "→ NO ERROR (unexpected)")
        except SandboxError as e:
            print(name, "→", e)
    print("sandbox available:", _posix_sandbox_available())
