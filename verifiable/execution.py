"""Bounded candidate execution. This is process isolation, not a security sandbox."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

from .common import ROOT


def worker_environment() -> dict[str, str]:
    # Allowlist, so provider credentials and .env paths are not inherited.
    env = {name: os.environ[name] for name in ("PATH", "LANG", "LC_ALL", "LD_LIBRARY_PATH", "SYSTEMROOT") if name in os.environ}
    env.update({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1", "PYTHONHASHSEED": "0"})
    return env


def run_solution(task: str, problem: dict, code: str | Path | None = None, *,
                 timeout_s: float = 10, memory_mb: int = 2048) -> dict:
    if not 0.05 <= timeout_s <= 600 or not 128 <= memory_mb <= 16384:
        raise ValueError("timeout must be .05..600 seconds and memory 128..16384 MB")
    argv = [sys.executable, "-I", str(ROOT / "worker.py"), "--task", task,
            "--memory-mb", str(memory_mb), "--cpu-seconds", str(max(1, math.ceil(timeout_s)))]
    if code is None:
        argv.append("--baseline")
    else:
        argv.extend(["--code", str(Path(code).resolve())])
    body = json.dumps({"problem": problem}, allow_nan=False).encode()
    with tempfile.TemporaryDirectory(prefix="verifiable-cpu-") as directory:
        work = Path(directory)
        with (work / "input.json").open("wb") as handle:
            handle.write(body)
        with (work / "input.json").open("rb") as stdin, (work / "stdout").open("wb") as stdout, (work / "stderr").open("wb") as stderr:
            started = time.perf_counter()
            proc = subprocess.Popen(argv, stdin=stdin, stdout=stdout, stderr=stderr, cwd=work,
                                    env=worker_environment(), start_new_session=True)
            timed_out = False
            try:
                proc.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                timed_out = True
            finally:
                # Also clean up descendants of a child that exited successfully.
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait()
            wall = time.perf_counter() - started
        diagnostic = (work / "stderr").read_bytes()[-4000:].decode(errors="replace")
        if timed_out:
            return {"status": "timeout", "wall_seconds": wall, "reason": f"Exceeded {timeout_s}s", "stderr_tail": diagnostic}
        if (work / "stdout").stat().st_size > 1024**2:
            return {"status": "error", "wall_seconds": wall, "reason": "Solution envelope exceeds 1 MiB"}
        try:
            output = json.loads((work / "stdout").read_text(), parse_constant=lambda v: (_ for _ in ()).throw(ValueError(v)))
            if not isinstance(output, dict) or output.get("status") not in {"ok", "error"}:
                raise ValueError("Malformed worker envelope")
            if output["status"] == "ok":
                runtime = output.get("solve_seconds")
                if isinstance(runtime, bool) or not isinstance(runtime, (int, float)) or not math.isfinite(runtime) or runtime < 0:
                    raise ValueError("Malformed runtime")
                if "solution" not in output:
                    raise ValueError("Missing solution")
            if proc.returncode != 0:
                raise ValueError(f"Worker exited with code {proc.returncode}")
        except (ValueError, OSError) as exc:
            output = {"status": "error", "reason": f"Invalid worker output: {exc}"}
        output.update({"wall_seconds": wall, "worker_exit_code": proc.returncode, "stderr_tail": diagnostic})
        return output
